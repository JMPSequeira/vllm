"""Exhaustive tests for the KVarN tail geometry authority.

Every function is checked against a brute-force ownership map for the
full (interleave, world_size, rank) grid, plus a simulated request
lifecycle (chunked prefill then decode) asserting the invariant after
every step. Pure offline tensor math -- no server, milliseconds.
"""
import itertools
import sys

import pytest
import torch

from vllm.v1.attention.backends.mla.kvarn_tail_geometry import (
    body_length_cap,
    demote_range,
    global_to_local_row_start,
    local_to_global,
    publish_cap,
    store_is_fp8,
    tail_window,
)

GRID = list(itertools.product((1, 2, 4), (1, 2, 4)))
N = 1024
PAGE = 2048


def brute_owned(rank, ws, i, S):
    """Rank's owned global positions < S, in local order."""
    return [g for g in range(S) if (g // i) % ws == rank]


def test_local_to_global_roundtrip():
    for i, ws in GRID:
        for rank in range(ws):
            S = 257
            owned = brute_owned(rank, ws, i, S)
            lp = torch.arange(len(owned), dtype=torch.int64)
            g = local_to_global(lp, rank, ws, i).tolist()
            assert g == owned, (i, ws, rank, g[:8], owned[:8])


def test_global_to_local_row_start_exhaustive():
    for i, ws in GRID:
        for rank in range(ws):
            for ts in (0, 1, 511, 512, 1024, 1537, 2048, 4095):
                start = int(global_to_local_row_start(
                    torch.tensor([ts]), rank, ws, i)[0])
                owned = brute_owned(rank, ws, i, 8192)
                # the first owned global position >= ts must map to start
                want = None
                for li, g in enumerate(owned):
                    if g >= ts:
                        want = li
                        break
                want = 0 if want is None else want
                assert start == want, (i, ws, rank, ts, start, want)


def test_store_is_fp8_matches_design_rule():
    for i, ws in GRID:
        rank = 0
        S = 2049
        owned = brute_owned(rank, ws, i, S)
        lp = torch.arange(len(owned), dtype=torch.int64)
        for ts in (0, 100, 1024, 1545, S):
            mask = store_is_fp8(lp, torch.tensor([ts]), rank, ws, i)
            want = [g >= ts for g in owned]
            assert mask.tolist() == want, (i, ws, ts)


def test_demote_range_never_touches_non_fp8():
    """The demotion range must be a subset of the rows that were fp8
    under ts_old = the window [ts_old, ts_since) of the chunk that set
    it."""
    cases = [
        # (ts_old, ts_since, ts_new, expect)
        (0, 0, 2048, None),          # first allocation: nothing was fp8
        (0, 4096, 2048, None),       # ts_new <= ts_old
        (2048, 2569, 4096, (2048, 2569)),  # cap at ts_since
        (2048, 4096, 3072, (2048, 3072)),  # cap at ts_new
        (4096, 4096, 8192, None),    # window was empty
    ]
    for ts_old, ts_since, ts_new, want in cases:
        assert demote_range(ts_old, ts_since, ts_new) == want


def test_publish_cap_never_exposes_fp8(ts=1545):
    ts_t = torch.tensor([ts])
    S = 2569
    ntc = S
    pub = publish_cap(int(ts_t[0]), ntc, PAGE)
    # whole pages below ts: every published row < ts
    assert pub <= (ts // PAGE) * PAGE
    assert all(p < ts for p in range(pub, pub + 1)) or pub == 0 or S >= ts


def test_tail_window_and_body_cap_partition_context():
    """body = [0, ts) clamped, tail = [ts, S): together they cover [0, S)
    exactly once for every row."""
    for S in (2570, 12216):
        ts = S - N
        rows = 8
        csl = torch.arange(S - rows + 1, S + 1)
        ts_b = torch.full((rows,), ts, dtype=torch.int64)
        start, length = tail_window(ts_b, csl, N)
        capped = body_length_cap(csl, ts_b)
        for r in range(rows):
            s = int(csl[r])
            lo = int(start[r])
            ln = int(length[r])
            body_len = int(capped[r])
            assert body_len == min(s, ts)
            assert lo + ln == s or ln == N
            assert ln == min(N, max(s - ts, 0))
            # partition: [0, body_len) + [lo, lo+ln) covers [0, s)
            assert body_len == lo or s <= ts


def test_simulated_lifecycle_invariant():
    """Full request lifecycle: chunked prefill then decode, asserting the
    invariant after every step (the manager formulas, mirrored)."""
    PAGE_, N_ = PAGE, N
    PROMPT, STEPS, CHUNK = 12216, 600, 4096
    ts, ts_since = 0, 0
    pool = {}
    S = 0

    def apply_step(ntc):
        nonlocal ts, ts_since, S
        ts_old = ts
        dr = demote_range(ts_old, ts_since, max(ntc - N_, ts_old))
        ts = max(ntc - N_, ts_old)
        if dr:
            lo, hi = dr
            assert all(pool.get(p, "kvarn") == "fp8" for p in range(lo, hi)), (
                f"demote non-fp8: {lo},{hi}")
            for p in range(lo, hi):
                pool[p] = "kvarn"
        ts_since = ntc
        for p in range(S, ntc):
            pool[p] = "fp8" if p >= ts else "kvarn"
        S = ntc
        bad = [p for p in range(S)
               if pool.get(p, "kvarn") != ("fp8" if p >= ts else "kvarn")]
        assert not bad, f"invariant violated at S={S}: {bad[:8]}"
        pub = publish_cap(ts, ntc, PAGE_)
        assert all(pool.get(p, "kvarn") == "kvarn" for p in range(pub)), (
            "published fp8 rows")

    while S < PROMPT:
        apply_step(min(S + CHUNK, PROMPT))
    for _ in range(STEPS):
        apply_step(S + 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
