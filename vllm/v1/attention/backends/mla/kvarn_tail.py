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


def kvarn_tail_classify(
    cache_seq_lens_per_token: torch.Tensor,
    req_id_per_token: torch.Tensor,
    num_reqs: int,
    n_local: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify batch tokens into fp8-tail vs kvarn-body stores.

    Returns ``(is_tail, boundary, s_per_token)`` where ``is_tail[i]`` is 1
    when token ``i`` lands inside its request's tail window (store fp8) and
    0 when it must be written directly as a kvarn record.
    ``boundary[i] = S_r - N`` (clamped at 0) is the per-token tail start;
    ``s_per_token[i] = S_r`` is the request's post-batch context length.
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
    boundary = (s_per_token - n_local).clamp_min_(0)
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
def _kvarn_tail_demote_slots_kernel(
    boundary_ptr,  # int32 [num_tokens]: S_r - N_local per token
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_reqs, max_blocks]
    slots_ptr,  # int32 [num_tokens] out: demote slot per token (-1 none)
    num_tokens,
    max_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One demotion candidate per token: position ``boundary - T_r``.

    For request ``r`` with ``T_r`` new tokens, the fp8 records leaving the
    window sit at positions ``[S_new - N - T_r, S_new - N)``. Decode batches
    have ``T_r`` tokens of request ``r`` in this batch, so the per-token
    slot of the OLDEST fp8 record that must flip is
    ``boundary_r - (number of tokens of r after i in this batch)``. This
    kernel handles the common T_r == 1 (pure decode) case; multi-token
    requests must call :func:`kvarn_tail_demote_range` instead.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < num_tokens
    boundary = tl.load(boundary_ptr + offs, mask=m, other=0)
    req = tl.load(req_id_ptr + offs, mask=m, other=0).to(tl.int64)
    # This token's own position; the demoted position is boundary - 1 for
    # T_r == 1 requests (the token itself is the only newcomer).
    demote_pos = boundary - 1
    valid = m & (demote_pos >= 0)
    bt_row = req * max_blocks
    blk = tl.load(
        block_table_ptr + bt_row + demote_pos.to(tl.int64) // BLOCK_SIZE,
        mask=valid,
        other=0,
    )
    slot = blk * BLOCK_SIZE + demote_pos % BLOCK_SIZE
    tl.store(slots_ptr + offs, tl.where(valid, slot, -1), mask=m)


def kvarn_tail_demote_slots(
    cache_seq_lens_per_token: torch.Tensor,
    req_id_per_token: torch.Tensor,
    boundary: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Slots of fp8 records that must be re-quantized to kvarn this step.

    Returns an int32 tensor of shape ``[num_tokens]`` with one candidate
    slot per token (``-1`` = nothing to demote). Valid for pure-decode
    batches (one new token per request); extend batches pass the full
    demotion range through :func:`kvarn_tail_demote_positions`.
    """
    num_tokens = cache_seq_lens_per_token.shape[0]
    slots = torch.empty(num_tokens, dtype=torch.int32, device=boundary.device)
    grid = (triton.cdiv(num_tokens, 256),)
    _kvarn_tail_demote_slots_kernel[grid](
        boundary,
        req_id_per_token,
        block_table,
        slots,
        num_tokens,
        block_table.shape[1],
        BLOCK_SIZE=block_size,
        BLOCK=256,
    )
    return slots


def kvarn_tail_demote_positions(
    boundary: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_id_per_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """All demotion positions for a batch (extend-safe), compacted.

    Per request ``r``: positions ``[max(boundary_r - T_r, 0), boundary_r)``
    where ``T_r`` is the request's query length. Returns a compact int32
    tensor of physical slots (length = total demotions across requests).
    """
    device = boundary.device
    num_reqs = query_start_loc.shape[0] - 1
    qsl = query_start_loc.to(torch.int64)
    query_lens = qsl[1:] - qsl[:-1]  # T_r per request
    s_per_req = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    s_per_req.scatter_reduce_(
        0,
        req_id_per_token.to(torch.int64),
        boundary + 0,  # boundary is per token; amax recovers per-request value
        reduce="amax",
    )
    # Per-request demotion ranges [lo_r, hi_r), lo = max(bnd - T, 0).
    hi = s_per_req.clamp_min(0)
    lo = (hi - query_lens).clamp_min(0)
    counts = hi - lo  # >= 0
    total = int(counts.sum().item())  # host sync: extend path only (not captured)
    if total == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    # Compact: positions per request via a small torch expansion.
    max_count = int(counts.max().item())
    req_idx = torch.arange(num_reqs, device=device)
    offsets = torch.arange(max_count, device=device)
    pos = lo[:, None] + offsets[None, :]
    valid = offsets[None, :] < counts[:, None]
    pos = torch.where(valid, pos, torch.zeros_like(pos))
    slots = kvarn_tail_positions_to_slots(
        pos.reshape(-1),
        req_idx.repeat_interleave(max_count),
        block_table,
        block_size,
    )
    slots = slots.reshape(num_reqs, max_count)[valid]
    return slots.to(torch.int32)


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
    out = (out_tail.float() * wa[..., None] + out_body.float() * wb[..., None]) / (
        denom[..., None]
    )
    return torch.where(safe[..., None], out, out_tail.float()).to(out_tail.dtype)
