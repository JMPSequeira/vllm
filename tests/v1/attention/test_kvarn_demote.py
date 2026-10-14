# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fp8->kvarn in-place demotion kernel (precision tail)."""

import pytest
import torch

from vllm.v1.attention.ops.kvarn_demote import demote_fp8_records
from vllm.v1.attention.ops.kvarn_mla import (
    KvarnMlaLayout,
    pack_kvarn_mla_records,
    unpack_kvarn_mla_records,
)

# The fp8 tail content is 528 B, so only kvarn variants whose record is at
# least 528 B can host it (k4v4=536, k5v5=648); k4v2=400 is rejected.
TAIL_DTYPES = [
    "kvarn_mla_k4v4_g256",
    "kvarn_mla_k5v5_g256",
]


def _layout(dtype: str) -> KvarnMlaLayout:
    from vllm.v1.kv_cache_interface import parse_kvarn_mla_dtype

    k_bits, v_bits, group = parse_kvarn_mla_dtype(dtype)
    return KvarnMlaLayout(k_bits=k_bits, v_bits=v_bits, group=group)


def _make_fp8_record(latent: torch.Tensor) -> torch.Tensor:
    """Torch reference GLM fp8 record content: 512 x E4M3 + 4 x fp32 scales.

    Matches the b12x GLM_NEXT fp8 writer: per-128-dim group scale =
    amax / 448 (unit scale for an all-zero group), saturating E4M3 codes.
    Returns ``[T, 528]`` (content only; the pool pad stays zero).
    """
    T = latent.shape[0]
    x = latent.to(torch.float32).reshape(T, 4, 128)
    amax = x.abs().amax(dim=2, keepdim=True)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / 448.0)
    codes = (x / scale).to(torch.float8_e4m3fn)
    rec = torch.zeros(T, 528, dtype=torch.uint8, device=latent.device)
    rec[:, :512] = codes.reshape(T, 512).view(torch.uint8)
    rec[:, 512:528] = scale.reshape(T, 4).view(torch.uint8)
    return rec


def _dequant_fp8_record(rec: torch.Tensor) -> torch.Tensor:
    """Dequant [T, 528] fp8 record content -> [T, 512] fp32 latent."""
    codes = rec[:, :512].view(torch.float8_e4m3fn).float()
    scales = rec[:, 512:528].view(torch.float32)  # [T, 4]
    return codes * scales.repeat_interleave(128, dim=1)


@pytest.mark.parametrize("dtype", TAIL_DTYPES)
def test_demote_roundtrip_bitexact(dtype: str):
    """demote(fp8_record) == pack(dequant(fp8_record)) bit-exact."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    layout = _layout(dtype)
    torch.manual_seed(0)
    T = 300
    latent = torch.randn(T, 512, device="cuda", dtype=torch.bfloat16)

    fp8_rec = _make_fp8_record(latent)
    latent_fp32 = _dequant_fp8_record(fp8_rec)

    page_size = 256
    pages = (T + page_size - 1) // page_size
    pool = torch.zeros(pages, page_size, layout.record_bytes,
                       dtype=torch.uint8, device="cuda")
    # Place the 528 B fp8 content at slots 0..T-1 (spans page 0 and page 1).
    pool.reshape(-1, layout.record_bytes)[:T, :528] = fp8_rec

    slots = torch.arange(T, device="cuda", dtype=torch.int64)
    # A negative slot must be skipped without affecting the valid ones.
    slots = torch.cat([slots, torch.tensor([-1], device="cuda", dtype=torch.int64)])

    demote_fp8_records(pool, slots, layout)

    demoted = pool.reshape(-1, layout.record_bytes)[:T]
    k_hat, v_hat = unpack_kvarn_mla_records(demoted, layout)
    ref_rec = pack_kvarn_mla_records(latent_fp32, layout)
    k_ref, v_ref = unpack_kvarn_mla_records(ref_rec, layout)

    assert (k_hat == k_ref).all(), "K plane not bit-exact"
    assert (v_hat == v_ref).all(), "V plane not bit-exact"
    # Record bytes (codes + inline scales) match the reference pack exactly.
    assert torch.equal(demoted[:, : layout.content_bytes],
                       ref_rec[:, : layout.content_bytes]), "record bytes differ"
    # Pad region is zero.
    assert int(demoted[:, layout.content_bytes:].sum()) == 0


@pytest.mark.parametrize("dtype", TAIL_DTYPES)
def test_demote_cosine_snr(dtype: str):
    """Kvarn reconstruction of the fp8-dequantized latent keeps high cosine."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    layout = _layout(dtype)
    torch.manual_seed(1)
    T = 256
    latent = torch.randn(T, 512, device="cuda", dtype=torch.bfloat16)

    fp8_rec = _make_fp8_record(latent)
    latent_fp32 = _dequant_fp8_record(fp8_rec)

    pool = torch.zeros(1, 256, layout.record_bytes, dtype=torch.uint8,
                       device="cuda")
    pool.reshape(-1, layout.record_bytes)[:T, :528] = fp8_rec
    slots = torch.arange(T, device="cuda", dtype=torch.int64)
    demote_fp8_records(pool, slots, layout)

    demoted = pool.reshape(-1, layout.record_bytes)[:T]
    k_hat, _ = unpack_kvarn_mla_records(demoted, layout)

    k_hat_n = k_hat / k_hat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    lat_n = latent_fp32 / latent_fp32.norm(dim=-1, keepdim=True)
    cos = (k_hat_n * lat_n).sum(-1)
    # k4-level per-token RTN on 512 dims: mean cosine well above 0.95.
    assert cos.mean() > 0.95, f"score-path cosine too low: {cos.mean():.4f}"


def test_demote_rejects_small_record():
    """k4v2 (400 B record) cannot host 528 B fp8 content."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    layout = _layout("kvarn_mla_k4v2_g256")
    pool = torch.zeros(1, 256, layout.record_bytes, dtype=torch.uint8,
                       device="cuda")
    slots = torch.arange(4, device="cuda", dtype=torch.int64)
    with pytest.raises(ValueError, match="record_bytes >= 528"):
        demote_fp8_records(pool, slots, layout)


def test_demote_skips_negative_slots():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    layout = _layout("kvarn_mla_k4v4_g256")
    torch.manual_seed(2)
    T = 64
    latent = torch.randn(T, 512, device="cuda", dtype=torch.bfloat16)
    fp8_rec = _make_fp8_record(latent)

    pool = torch.zeros(1, 256, layout.record_bytes, dtype=torch.uint8,
                       device="cuda")
    pool.reshape(-1, layout.record_bytes)[:T, :528] = fp8_rec

    slots = torch.full((T + 4,), -1, device="cuda", dtype=torch.int32)
    slots[:T] = torch.arange(T, device="cuda", dtype=torch.int32)
    demote_fp8_records(pool, slots, layout)

    # Valid slots demoted; the trailing -1 slots leave their (zero) rows untouched.
    demoted = pool.reshape(-1, layout.record_bytes)[:T]
    assert int(demoted[:, : layout.content_bytes].sum()) > 0
    tail = pool.reshape(-1, layout.record_bytes)[T:]
    assert int(tail.sum()) == 0
