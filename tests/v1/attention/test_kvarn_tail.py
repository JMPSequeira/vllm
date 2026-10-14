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
    kvarn_tail_demote_positions,
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
    _, boundary, _ = kvarn_tail_classify(csl, req, 3, 50)
    bt = _page(7)
    slots = kvarn_tail_demote_slots(csl, req, boundary, bt, BLOCK)
    assert slots[0].item() == 7 * BLOCK + 50
    # boundary 0 -> nothing below the window to demote.
    assert slots[1].item() == -1
    assert slots[2].item() == -1


def test_demote_positions_extend():
    # Request with T=4 new tokens, boundary 51: positions [47, 51) flip.
    bnd = torch.tensor([51, 51, 51, 51], dtype=torch.int32, device="cuda")
    rid = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device="cuda")
    qsl = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    bt = _page(7)
    pos = kvarn_tail_demote_positions(bnd, qsl, rid, bt, BLOCK)
    assert pos.tolist() == [7 * BLOCK + p for p in (47, 48, 49, 50)]


def test_demote_positions_clamped_short_context():
    # S=31, N=50 -> boundary 0; no demotion regardless of T.
    bnd = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device="cuda")
    rid = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device="cuda")
    qsl = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    bt = _page(7)
    pos = kvarn_tail_demote_positions(bnd, qsl, rid, bt, BLOCK)
    assert pos.numel() == 0


def test_demote_positions_multiple_requests():
    # Req 0: T=2, boundary 51 -> [49, 51). Req 1: T=1, boundary 10 -> [9, 10).
    bnd = torch.tensor([51, 51, 10], dtype=torch.int32, device="cuda")
    rid = torch.tensor([0, 0, 1], dtype=torch.int32, device="cuda")
    qsl = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    bt = _page(3)
    pos = kvarn_tail_demote_positions(bnd, qsl, rid, bt, BLOCK)
    assert pos.tolist() == [
        3 * BLOCK + 49,
        3 * BLOCK + 50,
        13 * BLOCK + 9,
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
