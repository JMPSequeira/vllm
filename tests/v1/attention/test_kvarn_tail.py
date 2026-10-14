"""KVarN precision-tail helper tests: boundary math and LSE merge.

The tail window is the newest N tokens of each request (DCP-local
positions); the format boundary and demotion ranges are pure tensor math
with real edge cases (S < N clamping, multi-token batches, empty
demotions). Demotion slots must hit exactly the positions
``[S_new - N - T_r, S_new - N)`` — off-by-one here silently quantizes the
wrong token and corrupts verbatim recall.
"""

import pytest
import torch

from vllm.v1.attention.backends.mla.kvarn_tail import (
    kvarn_tail_classify,
    kvarn_tail_demote_slots,
    merge_tail_lse,
)

BLOCK = 256


def _page(page: int) -> torch.Tensor:
    bt = torch.zeros(4, 8, dtype=torch.int32, device="cuda")
    bt[:, :] = -1
    for r in range(4):
        bt[r] = torch.arange(page + r * 10, page + r * 10 + 8)
    return bt


def test_classify_decode_batch():
    # 3 requests, one token each; pre-update lens 100, 30, 0 -> pos 100, 30, -.
    csl = torch.tensor([101, 31, 1], dtype=torch.int32, device="cuda")
    req = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    is_tail, boundary, s_tok = kvarn_tail_classify(csl, req, 3, 50)
    assert boundary.tolist() == [51, 0, 0]
    assert s_tok.tolist() == [101, 31, 1]
    # pos 100 >= 51: tail; pos 30 and 0 >= 0 (whole short context is tail).
    assert is_tail.tolist() == [1, 1, 1]


def test_classify_body_token():
    # Request 0 has two new tokens: pos 200 (tail) and pos 40 (body).
    csl = torch.tensor([201, 41], dtype=torch.int32, device="cuda")
    req = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    is_tail, boundary, _ = kvarn_tail_classify(csl, req, 1, 50)
    assert boundary.tolist() == [151, 151]
    assert is_tail.tolist() == [1, 0]


def test_demote_slots_decode():
    # S_new=101, N=50, T=1: position 50 flips fp8->kvarn; slot via page 7.
    csl = torch.tensor([101, 31, 1], dtype=torch.int32, device="cuda")
    req = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    bt = _page(7)
    slots = kvarn_tail_demote_slots(csl, req, 50, bt, BLOCK)
    assert slots[0].item() == 7 * BLOCK + 50
    # pos 30 - 50 < 0 and pos 0 - 50 < 0: nothing below the window.
    assert slots[1].item() == -1
    assert slots[2].item() == -1


def test_demote_slots_extend_chunk():
    # Chunk of 4 tokens at positions 100..103, N=50: demotes 50..53.
    csl = torch.tensor([101, 102, 103, 104], dtype=torch.int32, device="cuda")
    req = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device="cuda")
    bt = _page(7)
    slots = kvarn_tail_demote_slots(csl, req, 50, bt, BLOCK)
    assert slots.tolist() == [7 * BLOCK + p for p in (50, 51, 52, 53)]


def test_demote_positions_multiple_requests():
    # Req 0: two tokens (pos 200, 201), N=50 -> demote 150, 151.
    # Req 1: one token (pos 20), N=50 -> none.
    csl = torch.tensor([201, 202, 21], dtype=torch.int32, device="cuda")
    req = torch.tensor([0, 0, 1], dtype=torch.int32, device="cuda")
    bt = _page(3)
    slots = kvarn_tail_demote_slots(csl, req, 50, bt, BLOCK)
    assert slots.tolist() == [
        3 * BLOCK + 150,
        3 * BLOCK + 151,
        -1,
    ]


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


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
