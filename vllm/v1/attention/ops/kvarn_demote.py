# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fp8->kvarn in-place demotion kernel for the precision tail.

When the kvarn precision tail advances, a token that exits the exact-recency
window must be re-quantized from its fp8_ds_mla-format record (528 B content:
512 x E4M3 latent + 4 x fp32 group-128 scales) to a kvarn per-token record
(536 B stride) at the SAME pool slot. This kernel does that in place, one
program per slot, capture-safe (no host syncs, no ``int()`` on device
tensors, grid sized from the host-known slot count with ``slot >= 0``
masking).

The kvarn quantization mirrors ``pack_kvarn_mla_records`` exactly (per-token
asymmetric RTN on the dequantized latent, shared min/max for K and V, LSB-
first dense bit packing, fp16 inline scale/zp), so the demoted record is
bit-identical to packing the dequantized fp8 latent directly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.v1.attention.ops.kvarn_mla import KvarnMlaLayout

__all__ = ["demote_fp8_records"]


@triton.jit
def _kvarn_pack_codes(
    codes,  # [latent_dim] int32
    bits: tl.constexpr,
    out_ptr,  # *u8
    latent_dim: tl.constexpr,
):
    """Pack ``codes`` LSB-first dense into ``out_ptr`` (in registers).

    Stream bit ``d*bits + b`` lands in byte ``(d*bits + b) // 8`` at bit
    position ``(d*bits + b) % 8``. Each output byte's 8 bits are gathered
    from their source codes and OR-combined (the bit positions are distinct,
    so max == OR), matching the reference ``pack_bits`` scatter exactly.
    """
    out_bytes: tl.constexpr = (latent_dim * bits + 7) // 8
    O: tl.constexpr = 512
    o = tl.arange(0, O)
    byte = tl.zeros((O,), tl.int32)
    for j in tl.range(0, 8):
        pos = o * 8 + j
        d_idx = tl.minimum(pos // bits, latent_dim - 1)
        b_idx = pos % bits
        gathered = tl.gather(codes, d_idx, 0)
        bit_val = (gathered >> b_idx) & 1
        byte = byte | (bit_val << j)
    tl.store(out_ptr + o, byte.to(tl.uint8), mask=o < out_bytes)

@triton.jit
def _kvarn_demote_kernel(
    kv_cache_ptr,  # *u8 [pages, page_size, record_bytes]
    slots_ptr,  # *i32/i64 [K]
    k_bits: tl.constexpr,
    v_bits: tl.constexpr,
    record_bytes,  # kvarn record width (layout.record_bytes)
    latent_dim: tl.constexpr,  # 512
    group: tl.constexpr,  # 128
    page_size,  # kv_cache.shape[1]
    cache_stride0,  # kv_cache.stride(0) in bytes
    cache_stride1,  # kv_cache.stride(1) in bytes
):
    pid = tl.program_id(0)
    slot = tl.load(slots_ptr + pid).to(tl.int64)
    if slot < 0:
        return

    k_code_bytes: tl.constexpr = (latent_dim * k_bits + 7) // 8
    v_code_bytes: tl.constexpr = (latent_dim * v_bits + 7) // 8
    code_region: tl.constexpr = k_code_bytes + v_code_bytes
    content_bytes: tl.constexpr = code_region + 8
    k_scale_off: tl.constexpr = code_region
    k_zp_off: tl.constexpr = k_scale_off + 2
    v_scale_off: tl.constexpr = k_zp_off + 2
    v_zp_off: tl.constexpr = v_scale_off + 2

    block = slot // page_size
    local = slot % page_size
    base = (
        kv_cache_ptr
        + block * tl.cast(cache_stride0, tl.int64)
        + local * tl.cast(cache_stride1, tl.int64)
    )

    # --- dequant the fp8 record content [0,512) + group scales [512,528) ---
    d = tl.arange(0, latent_dim)
    fp8_bytes = tl.load(base + d)
    vals = tl.cast(fp8_bytes, tl.float8e4nv, bitcast=True).to(tl.float32)
    g = tl.arange(0, 4)
    scales = tl.load(base.to(tl.pointer_type(tl.float32)) + 128 + g)
    group_idx = d // group
    scale_per_dim = tl.gather(scales, group_idx, 0)
    latent = vals * scale_per_dim

    # --- per-token asymmetric RTN (shared min/max for K and V) ---
    mn = tl.min(latent, axis=0)
    mx = tl.max(latent, axis=0)

    # --- zero the record pad (code/scale regions are overwritten below) ---
    pad_idx = tl.arange(0, 16)
    tl.store(
        base + content_bytes + pad_idx,
        tl.zeros((16,), tl.uint8),
        mask=pad_idx < (record_bytes - content_bytes),
    )

    # --- K plane ---
    # NOTE: torch's `tensor / python_int` lowers to a MULTIPLY by the fp32
    # reciprocal (verified: diverges from true division on ~63% of random
    # inputs), and `_rtn_codes` -- the storage contract -- uses that path.
    # div_rn(range, levels) is 1 ULP off for many inputs, which flips codes
    # sitting near a .5 boundary. Use the reciprocal multiply here.
    levels_k: tl.constexpr = (1 << k_bits) - 1
    inv_k: tl.constexpr = 1.0 / ((1 << k_bits) - 1)
    scale_k = tl.maximum(mx - mn, 1e-8) * inv_k
    sub_k = tl.extra.cuda.libdevice.sub_rn(latent, mn)
    codes_k = tl.extra.cuda.libdevice.rint(
        tl.extra.cuda.libdevice.div_rn(sub_k, scale_k)
    ).to(tl.int32)
    codes_k = tl.minimum(tl.maximum(codes_k, 0), levels_k)
    _kvarn_pack_codes(codes_k, k_bits, base, latent_dim)

    # --- V plane ---
    levels_v: tl.constexpr = (1 << v_bits) - 1
    inv_v: tl.constexpr = 1.0 / ((1 << v_bits) - 1)
    scale_v = tl.maximum(mx - mn, 1e-8) * inv_v
    sub_v = tl.extra.cuda.libdevice.sub_rn(latent, mn)
    codes_v = tl.extra.cuda.libdevice.rint(
        tl.extra.cuda.libdevice.div_rn(sub_v, scale_v)
    ).to(tl.int32)
    codes_v = tl.minimum(tl.maximum(codes_v, 0), levels_v)
    _kvarn_pack_codes(codes_v, v_bits, base + k_code_bytes, latent_dim)

    # --- inline fp16 scale / zero-point ---
    base_f16 = base.to(tl.pointer_type(tl.float16))
    tl.store(base_f16 + k_scale_off // 2, scale_k.to(tl.float16))
    tl.store(base_f16 + k_zp_off // 2, mn.to(tl.float16))
    tl.store(base_f16 + v_scale_off // 2, scale_v.to(tl.float16))
    tl.store(base_f16 + v_zp_off // 2, mn.to(tl.float16))


def demote_fp8_records(
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    layout: KvarnMlaLayout,
) -> None:
    """Demote fp8 tail records to kvarn records in place at ``slots``.

    Args:
        kv_cache: ``[pages, page_size, record_bytes]`` uint8 pool whose row
            stride is ``record_bytes`` (derived from the tensor strides).
        slots: ``[K]`` int32/int64 global record slots; ``slot < 0`` is
            skipped (padded CUDA-graph slots).
        layout: kvarn record geometry (k_bits, v_bits, record_bytes).
    """
    if slots.numel() == 0:
        return
    if kv_cache.ndim != 3 or kv_cache.dtype != torch.uint8:
        raise ValueError(
            "kv_cache must be a 3-D uint8 tensor, got "
            f"{tuple(kv_cache.shape)} {kv_cache.dtype}"
        )
    if layout.record_bytes < 528:
        raise ValueError(
            "fp8 tail content (528 B) does not fit a "
            f"{layout.record_bytes}-byte kvarn record; the fp8 precision "
            "tail requires record_bytes >= 528 (k4v4/k5v5)"
        )
    if int(kv_cache.shape[-1]) != layout.record_bytes:
        raise ValueError(
            "kv_cache record width "
            f"{int(kv_cache.shape[-1])} != layout.record_bytes "
            f"{layout.record_bytes}"
        )
    if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "slots must be 1-D int32 or int64, got "
            f"{tuple(slots.shape)} {slots.dtype}"
        )
    if not slots.is_contiguous():
        raise ValueError("slots must be contiguous")

    grid = (slots.shape[0],)
    _kvarn_demote_kernel[grid](
        kv_cache,
        slots,
        layout.k_bits,
        layout.v_bits,
        layout.record_bytes,
        layout.latent_dim,
        128,
        kv_cache.shape[1],
        kv_cache.stride(0),
        kv_cache.stride(1),
        num_warps=4,
    )
