"""KVarN precision-tail geometry: the SINGLE AUTHORITY for "which rows
are fp8 right now, in whose coordinate system".

Invariant (maintained by the manager, consumed by everyone else):

    pool rows [ts, S) hold fp8 records; rows [0, ts) hold kvarn.

``ts`` is a GLOBAL request-relative position, exact (never page-aligned:
page-aligning it with N < page puts ts at/above S and the demotion
requantizes kvarn rows as fp8 -- destroyed context). Prefix-cache safety
comes from capping PUBLISHED tokens at whole pages fully below ts, not
from aligning the boundary.

Every consumer asks here instead of re-deriving conversions locally.
All functions are pure tensor math so each can be exhaustively tested
against a brute-force ownership map (see
tests/v1/attention/test_kvarn_tail_geometry.py).
"""

from __future__ import annotations

import torch


def local_to_global(
    local_pos: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    interleave: int,
) -> torch.Tensor:
    """DCP-local sequence position -> global request-relative position.

    Inverse of the gather mapper's ownership rule:
        owner     = g // I % ws
        local_idx = (g // (ws*I)) * I + g % I
    =>  g         = (i // I) * (ws*I) + r * I + (i % I)
    """
    i = max(1, int(interleave))
    ws = max(1, int(dcp_world_size))
    p = local_pos.to(torch.int64)
    return (p // i) * (ws * i) + int(dcp_rank) * i + (p % i)


def global_to_local_row_start(
    ts: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    interleave: int,
) -> torch.Tensor:
    """First LOCAL row index of ``dcp_rank`` at/after global ``ts``.

    Rank r owns globals ``g = (g // (i*ws)) * (i*ws) + r*i + (g % i)``;
    the number of its local rows below ts is::

        t    = max(ts - r*i, 0)
        j_lo = (t // (i*ws)) * i + min(t % (i*ws), i)
    """
    i = max(1, int(interleave))
    ws = max(1, int(dcp_world_size))
    t = (ts.to(torch.int64) - int(dcp_rank) * i).clamp_min_(0)
    return (
        (t // (i * ws)) * i + (t % (i * ws)).clamp_max_(i)
    ).to(torch.int32)


def store_is_fp8(
    local_positions: torch.Tensor,
    ts: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    interleave: int,
) -> torch.Tensor:
    """Per-token fp8 store test: the design rule is fp8 iff
    ``global_pos >= ts``. ``local_positions`` are the rank's DCP-local
    rope positions; convert through the interleave first."""
    pos_g = local_to_global(local_positions, dcp_rank, dcp_world_size, interleave)
    return pos_g >= ts.to(torch.int64)


def demote_range(
    ts_old: int,
    ts_since: int,
    ts_new: int,
) -> tuple[int, int] | None:
    """Rows to demote when the boundary advances ts_old -> ts_new.

    Only rows that WERE fp8 -- stored while the boundary was ts_old --
    may demote, i.e. the window [ts_old, ts_since) of the chunk that set
    ts_old (ts_since = S at the time). Capping at ts_since matters: later
    chunks stored kvarn under the new boundary, and demoting them reads
    kvarn records as fp8 (destroyed context). Returns None when the range
    is empty (first allocation: ts_old == 0 -- nothing below ts_new was
    ever fp8; prefix-hit rows likewise).
    """
    if ts_old <= 0:
        return None
    hi = min(ts_new, ts_since)
    if hi <= ts_old:
        return None
    return (ts_old, hi)


def publish_cap(ts: int, num_tokens_to_cache: int, block_size: int) -> int:
    """Largest token count that may be published to the prefix cache:
    whole pages fully below ts. Published blocks are uniformly kvarn, so
    prefix hits read consistent records."""
    return min(num_tokens_to_cache, (ts // block_size) * block_size)


def body_length_cap(
    cache_seq_lens: torch.Tensor,
    ts: torch.Tensor,
) -> torch.Tensor:
    """Dense-reader body lengths: stop at the tail boundary. Rows whose
    own S is below ts are unclamped (their whole context is kvarn)."""
    return torch.minimum(cache_seq_lens, ts.to(cache_seq_lens.dtype))


def tail_window(
    ts: torch.Tensor,
    cache_seq_lens: torch.Tensor,
    tail_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row tail window (start, length) in the units of
    ``cache_seq_lens``: [ts_row, S_row), length <= tail_tokens. Rows with
    S <= ts get length 0 (the fp8 set is empty for them)."""
    start = ts.to(torch.int64)
    length = (cache_seq_lens.to(torch.int64) - start).clamp_min_(0).clamp_max_(tail_tokens)
    return start, length.to(torch.int32)
