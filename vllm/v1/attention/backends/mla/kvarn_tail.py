"""KVarN precision-tail helpers (design: lil/kvarn-mla-design.md §10).

The tail is the newest ``N`` tokens of each request, stored as fp8_ds_mla
records (528B content) inside the 536B kvarn record slots; positions
``< S - N`` hold kvarn records. A slot's format is determined by the
token's (DCP-local) position vs the per-request tail boundary; on
boundary crossing the slot is re-quantized in place (demotion).

All helpers are capture-safe: device-only, no host syncs, no ``int()`` on
device tensors. Positions and sequence lengths are DCP-local (the same
space as ``cache_seq_lens_per_token``, which stores context length =
position + 1 per token).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.v1.attention.ops.kvarn_demote import _kvarn_pack_codes


def kvarn_tail_classify(
    cache_seq_lens_per_token: torch.Tensor,
    req_id_per_token: torch.Tensor,
    num_reqs: int,
    n_local: int,
    hit_len: torch.Tensor | None,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify batch tokens into fp8-tail vs kvarn-body stores.

    Returns ``(is_tail, boundary, s_per_token)`` where ``is_tail[i]`` is 1
    when token ``i`` lands inside its request's tail window (store fp8) and
    0 when it must be written directly as a kvarn record. The boundary is
    BLOCK-ALIGNED and clamped at the request's hit length H:
    ``boundary = max(align_up(S - N, block_size), H)`` — published (shared)
    blocks always sit below the boundary and therefore always read as
    kvarn records, whatever request reads them.
    """
    num_tokens = cache_seq_lens_per_token.shape[0]
    device = cache_seq_lens_per_token.device
    # S_r per request: segment max of csl by req_id (scatter-reduce amax),
    # then gather back per token. Pure torch ops, capture-safe.
    s_per_req = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    s_per_req.scatter_reduce_(
        0,
        req_id_per_token.to(torch.int64),
        cache_seq_lens_per_token,
        reduce="amax",
    )
    s_per_token = s_per_req[req_id_per_token.to(torch.int64)]
    # Block-aligned raw boundary: ceil((S-N)/bs)*bs, clamped at 0.
    raw = (s_per_token - n_local).clamp_min_(0)
    aligned = ((raw + block_size - 1) // block_size) * block_size
    if hit_len is not None:
        h_per_token = hit_len.to(torch.int32)[req_id_per_token.to(torch.int64)]
        boundary = torch.maximum(aligned, h_per_token)
    else:
        boundary = aligned
    is_tail = (cache_seq_lens_per_token - 1) >= boundary
    return is_tail.to(torch.int32), boundary, s_per_token


def kvarn_tail_positions_to_slots(
    positions: torch.Tensor,
    req_id_per_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map request-relative positions to physical record slots.

    ``slot = block_table[req, pos // block_size] * block_size + pos % block_size``
    (same mapping as ``triton_convert_req_index_to_global_index``).
    """
    bt = block_table.to(torch.int64)[
        req_id_per_token.to(torch.int64), positions.to(torch.int64) // block_size
    ]
    return bt * block_size + (positions.to(torch.int64) % block_size)


@triton.jit
def _kvarn_tail_demote_ranges_kernel(
    kv_cache_ptr,  # *u8 [pages, page_size, record_bytes]
    ranges_ptr,  # int32 [num_reqs, 2]: (lo, hi) demotion positions, (0,0)=none
    block_table_ptr,  # int32 [num_reqs, max_blocks]
    cache_stride0,  # kv_cache.stride(0) bytes
    cache_stride1,  # kv_cache.stride(1) bytes
    page_size,
    max_blocks,
    k_bits: tl.constexpr,
    v_bits: tl.constexpr,
    record_bytes: tl.constexpr,
    latent_dim: tl.constexpr,
    group: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Demote every fp8 record in each request's [lo, hi) position range.

    One program per (request, CHUNK-position slice); programs whose slice
    starts past the range exit immediately, so the common no-crossing step
    is a cheap no-op launch. The per-slot body mirrors
    ``ops/kvarn_demote._kvarn_demote_kernel`` (bit-exact vs the torch
    reference pack: reciprocal-multiply scale, sub_rn, rint).
    """
    req = tl.program_id(0)
    chunk = tl.program_id(1)
    lo = tl.load(ranges_ptr + req * 2)
    hi = tl.load(ranges_ptr + req * 2 + 1)
    start = lo + chunk * CHUNK
    if start >= hi:
        return
    end = tl.minimum(hi, start + CHUNK)

    k_code_bytes: tl.constexpr = (latent_dim * k_bits + 7) // 8
    v_code_bytes: tl.constexpr = (latent_dim * v_bits + 7) // 8
    code_region: tl.constexpr = k_code_bytes + v_code_bytes
    content_bytes: tl.constexpr = code_region + 8
    k_scale_off: tl.constexpr = code_region
    k_zp_off: tl.constexpr = k_scale_off + 2
    v_scale_off: tl.constexpr = k_zp_off + 2
    v_zp_off: tl.constexpr = v_scale_off + 2

    levels_k: tl.constexpr = (1 << k_bits) - 1
    inv_k: tl.constexpr = 1.0 / ((1 << k_bits) - 1)
    levels_v: tl.constexpr = (1 << v_bits) - 1
    inv_v: tl.constexpr = 1.0 / ((1 << v_bits) - 1)

    d = tl.arange(0, latent_dim)
    pad_idx = tl.arange(0, 16)
    g = tl.arange(0, 4)

    for pos in range(start, end):
        blk = tl.load(block_table_ptr + req.to(tl.int64) * max_blocks + pos // BLOCK_SIZE)
        slot = blk * BLOCK_SIZE + pos % BLOCK_SIZE
        base = (
            kv_cache_ptr
            + (slot // page_size) * tl.cast(cache_stride0, tl.int64)
            + (slot % page_size) * tl.cast(cache_stride1, tl.int64)
        )
        fp8_bytes = tl.load(base + d)
        vals = tl.cast(fp8_bytes, tl.float8e4nv, bitcast=True).to(tl.float32)
        scales = tl.load(base.to(tl.pointer_type(tl.float32)) + 128 + g)
        scale_per_dim = tl.gather(scales, d // group, 0)
        latent = vals * scale_per_dim
        mn = tl.min(latent, axis=0)
        mx = tl.max(latent, axis=0)
        pad_mask = pad_idx < (record_bytes - content_bytes)
        tl.store(
            base + content_bytes + pad_idx,
            tl.zeros((16,), tl.uint8),
            mask=pad_mask,
        )
        # torch tensor/python-int division lowers to a reciprocal multiply;
        # the storage contract (pack_kvarn_mla_records) uses that path.
        scale_k = tl.maximum(mx - mn, 1e-8) * inv_k
        sub = tl.extra.cuda.libdevice.sub_rn(latent, mn)
        codes_k = tl.extra.cuda.libdevice.rint(
            tl.extra.cuda.libdevice.div_rn(sub, scale_k)
        ).to(tl.int32)
        codes_k = tl.minimum(tl.maximum(codes_k, 0), levels_k)
        _kvarn_pack_codes(codes_k, k_bits, base, latent_dim)
        scale_v = tl.maximum(mx - mn, 1e-8) * inv_v
        codes_v = tl.extra.cuda.libdevice.rint(
            tl.extra.cuda.libdevice.div_rn(sub, scale_v)
        ).to(tl.int32)
        codes_v = tl.minimum(tl.maximum(codes_v, 0), levels_v)
        _kvarn_pack_codes(codes_v, v_bits, base + k_code_bytes, latent_dim)
        base_f16 = base.to(tl.pointer_type(tl.float16))
        tl.store(base_f16 + k_scale_off // 2, scale_k.to(tl.float16))
        tl.store(base_f16 + k_zp_off // 2, mn.to(tl.float16))
        tl.store(base_f16 + v_scale_off // 2, scale_v.to(tl.float16))
        tl.store(base_f16 + v_zp_off // 2, mn.to(tl.float16))


def kvarn_tail_demote_ranges(
    kv_cache: torch.Tensor,
    demote_ranges: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    layout,
    num_tokens: int,
) -> None:
    """Re-quantize fp8 records to kvarn over per-request position ranges.

    Args:
        kv_cache: ``[pages, page_size, record_bytes]`` uint8 kvarn pool.
        demote_ranges: ``[num_reqs, 2]`` int32 (lo, hi) per request,
            ``(0, 0)`` = nothing to demote (the common case).
        block_table: ``[num_reqs, max_blocks]`` int32 physical pages.
        block_size: scheduler block size (tokens per block-table entry).
        layout: kvarn record geometry (KvarnMlaLayout).
        num_tokens: batch token count, bounding each range's length
            (range <= tokens scheduled this step + one block).
    """
    if kv_cache.ndim != 3 or kv_cache.dtype != torch.uint8:
        raise ValueError(
            "kv_cache must be a 3-D uint8 tensor, got "
            f"{tuple(kv_cache.shape)} {kv_cache.dtype}"
        )
    if layout.record_bytes < 528:
        raise ValueError(
            "fp8 tail content (528 B) does not fit a "
            f"{layout.record_bytes}-byte kvarn record"
        )
    num_reqs = demote_ranges.shape[0]
    if num_reqs == 0:
        return
    # Each request's range is bounded by (tokens scheduled this step +
    # one block); CHUNK slices cover it with a static grid.
    chunks = (num_tokens + block_size - 1) // block_size + 1
    grid = (num_reqs, chunks)
    _kvarn_tail_demote_ranges_kernel[grid](
        kv_cache,
        demote_ranges,
        block_table,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.shape[1],
        block_table.shape[1],
        k_bits=layout.k_bits,
        v_bits=layout.v_bits,
        record_bytes=layout.record_bytes,
        latent_dim=layout.latent_dim,
        group=128,
        BLOCK_SIZE=block_size,
        CHUNK=256,
        num_warps=4,
    )


def merge_tail_lse(
    out_tail: torch.Tensor,
    lse_tail: torch.Tensor,
    out_body: torch.Tensor,
    lse_body: torch.Tensor,
) -> torch.Tensor:
    """Re-merge two partial attentions via natural-log LSE.

    Each launch returns normalized output ``O = sum / denom`` and
    ``lse = log(denom)`` (lse_scale="natural"). The combined attention is
    ``(sum_tail + sum_body) / (denom_tail + denom_body)``. Rows whose LSE
    is -inf (empty segment) contribute nothing.
    """
    a = lse_tail.float()
    b = lse_body.float()
    m = torch.maximum(a, b)
    # exp(lse - m) with -inf handling: where m == -inf both weights are 0.
    wa = torch.exp(a - m)
    wb = torch.exp(b - m)
    denom = wa + wb
    safe = denom > 0
    # An empty segment (all candidates masked) has lse = -inf and a
    # garbage/NaN normalized output; zero it so 0-weight rows stay clean.
    at = torch.where((a > float("-inf"))[..., None], out_tail.float(), 0.0)
    bt = torch.where((b > float("-inf"))[..., None], out_body.float(), 0.0)
    out = (at * wa[..., None] + bt * wb[..., None]) / (denom[..., None])
    return torch.where(safe[..., None], out, at).to(out_tail.dtype)
