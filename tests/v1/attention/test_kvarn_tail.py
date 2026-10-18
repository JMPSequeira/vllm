"""KVarN precision-tail helper tests: boundary math + range demotion + merge.

The tail boundary is block-aligned and clamped at the request's hit length
H so published (shared) blocks always read as kvarn records regardless of
which request reads them. Demotion consumes per-request [lo, hi) ranges
emitted by the KV cache manager at block crossings — an off-by-one there
silently quantizes the wrong token and corrupts verbatim recall.
"""

import pytest
import torch

from vllm.v1.attention.backends.mla.kvarn_tail import (
    kvarn_tail_boundary,
    kvarn_tail_demote_ranges,
    kvarn_tail_global_boundary,
    merge_tail_lse,
)
from vllm.v1.attention.ops.kvarn_demote import demote_fp8_records
from vllm.v1.attention.ops.kvarn_mla import (
    KvarnMlaLayout,
)

BLOCK = 16  # small block size for clean boundary math

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _layout() -> KvarnMlaLayout:
    return KvarnMlaLayout(k_bits=4, v_bits=4, group=256)


@requires_cuda
def test_boundary_from_ts():
    # The manager owns ts; the per-token boundary is a plain gather.
    ts = torch.tensor([64], dtype=torch.int32, device="cuda")
    req = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    csl = torch.tensor([101, 64], dtype=torch.int32, device="cuda")
    boundary = kvarn_tail_boundary(ts, req)
    assert boundary.tolist() == [64, 64]
    # pos 100 (csl 101) is tail; pos 63 (csl 64) is body.
    assert ((csl - 1) >= boundary).tolist() == [True, False]


@requires_cuda
def test_global_boundary_dcp2_interleave4():
    # DCP2, interleave 4: local j owns globals [8j, 8j+4) for rank 0 and
    # [8j+4, 8j+8) for rank 1. ts=8 (local) -> rank 0: g=16, rank 1: g=20.
    ts = torch.tensor([8], dtype=torch.int32, device="cuda")
    req = torch.tensor([0], dtype=torch.int32, device="cuda")
    g0 = kvarn_tail_global_boundary(ts, req, 0, 2, 4)
    g1 = kvarn_tail_global_boundary(ts, req, 1, 2, 4)
    assert g0.item() == 16 and g1.item() == 20


@requires_cuda
def test_global_boundary_dcp1():
    ts = torch.tensor([64], dtype=torch.int32, device="cuda")
    req = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    g = kvarn_tail_global_boundary(ts, req, 0, 1, 1)
    assert g.tolist() == [64, 64]


@requires_cuda
def test_demote_ranges_matches_slot_list():
    """Range-driven demotion must equal per-slot demotion bit-for-bit."""
    layout = _layout()
    torch.manual_seed(3)
    T = 512
    latent = torch.randn(T, 512, device="cuda", dtype=torch.bfloat16)
    fp8 = _make_fp8_record(latent)
    page_size = 256
    pages = (T + page_size - 1) // page_size
    bt = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)
    pool = torch.zeros(
        pages, page_size, layout.record_bytes, dtype=torch.uint8, device="cuda"
    )
    pool.reshape(-1, layout.record_bytes)[:T, :528] = fp8
    # contiguous pages => slot == position
    ranges = torch.tensor([[100, 356], [0, 0]], dtype=torch.int32, device="cuda")
    kvarn_tail_demote_ranges(pool, ranges, bt, page_size, layout, T)
    demoted = pool.reshape(-1, layout.record_bytes)
    # reference: explicit slot list over the same positions
    ref_pool = torch.zeros(
        pages, page_size, layout.record_bytes, dtype=torch.uint8, device="cuda"
    )
    ref_pool.reshape(-1, layout.record_bytes)[:T, :528] = fp8
    slots = torch.arange(100, 356, device="cuda", dtype=torch.int64)
    demote_fp8_records(ref_pool, slots, layout)
    ref = ref_pool.reshape(-1, layout.record_bytes)
    assert torch.equal(demoted, ref)
    # untouched slots keep their fp8 content
    assert torch.equal(demoted[:100, :528], fp8[:100])
    assert torch.equal(demoted[356:T, :528], fp8[356:T])


@requires_cuda
def test_demote_ranges_noop_when_empty():
    layout = _layout()
    torch.manual_seed(5)
    latent = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    fp8 = _make_fp8_record(latent)
    pool = torch.zeros(1, 64, layout.record_bytes, dtype=torch.uint8, device="cuda")
    pool[0, :, :528] = fp8
    before = pool.clone()
    ranges = torch.zeros(1, 2, dtype=torch.int32, device="cuda")
    bt = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    kvarn_tail_demote_ranges(pool, ranges, bt, 64, layout, 64)
    assert torch.equal(pool, before)


def _make_fp8_record(latent: torch.Tensor) -> torch.Tensor:
    T = latent.shape[0]
    x = latent.to(torch.float32).reshape(T, 4, 128)
    amax = x.abs().amax(dim=2, keepdim=True)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / 448.0)
    codes = (x / scale).to(torch.float8_e4m3fn)
    rec = torch.zeros(T, 528, dtype=torch.uint8, device=latent.device)
    rec[:, :512] = codes.reshape(T, 512).view(torch.uint8)
    rec[:, 512:528] = scale.reshape(T, 4).view(torch.uint8)
    return rec


@requires_cuda
def test_merge_tail_lse():
    out_t = torch.tensor([[[1.0, 2.0]]], device="cuda")
    out_b = torch.tensor([[[3.0, 4.0]]], device="cuda")
    # Empty body segment (-inf LSE) must not perturb the tail output.
    m = merge_tail_lse(
        out_t,
        torch.tensor([[0.0]], device="cuda"),
        out_b,
        torch.tensor([[float("-inf")]], device="cuda"),
    )
    assert torch.allclose(m, out_t)
    # denom 1 vs denom 2: weighted 1:2.
    m2 = merge_tail_lse(
        out_t,
        torch.tensor([[0.0]], device="cuda"),
        out_b,
        torch.tensor([[0.6931472]], device="cuda"),
    )
    assert torch.allclose(m2, (out_t * 1 + out_b * 2) / 3)


@requires_cuda
def test_merge_tail_lse_nan_empty_segment():
    # A fully-masked segment produces lse=-inf and NaN output; the merge
    # must zero its contribution instead of propagating NaN (NaN * 0).
    out_t = torch.tensor([[[1.0, 2.0]]], device="cuda")
    out_b = torch.tensor([[[float("nan"), float("nan")]]], device="cuda")
    m = merge_tail_lse(
        out_t,
        torch.tensor([[0.0]], device="cuda"),
        out_b,
        torch.tensor([[float("-inf")]], device="cuda"),
    )
    assert torch.allclose(m, out_t), m


@pytest.mark.parametrize("interleave", [1, 2, 4])
@pytest.mark.parametrize("ws", [1, 2, 4])
@pytest.mark.parametrize("ts", [0, 16, 64, 256, 1000])
def test_local_row_start_matches_ownership(interleave, ws, ts):
    """kvarn_tail_local_row_start == brute-force owned-position count."""
    from vllm.v1.attention.backends.mla.kvarn_tail import (
        kvarn_tail_local_row_start,
    )

    S = 2048
    owned = [g for g in range(S) if (g // interleave) % ws == 0]
    expect = sum(1 for g in owned if g < ts)
    got = int(
        kvarn_tail_local_row_start(
            torch.tensor([ts], dtype=torch.int32), 0, ws, interleave
        ).item()
    )
    assert got == expect


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
