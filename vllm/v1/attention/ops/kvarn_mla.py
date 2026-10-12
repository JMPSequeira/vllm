# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN MLA (arXiv:2606.03458) packed-record layout and reference codec.

KVarN = Hadamard rotation + dual-axis variance normalization (VarN) +
round-to-nearest quantization of the glm5_next MLA compressed latent. The
latent serves both the score (K) and value-aggregation (V) attention roles
from one cache record, so each record carries TWO code planes at independent
bit widths (``k_bits`` score path, ``v_bits`` value path).

Dtype grammar: ``kvarn_mla_k<k>[v<v>]_g<group>`` where ``group`` equals the
vLLM cache block size (one VarN tile == one page). Launched variants:
``kvarn_mla_k4v4_g256``, ``kvarn_mla_k5v5_g256``, ``kvarn_mla_k4v2_g256``.

Record layout (per token, latent_dim=512, rope_dim=64), little-endian:

======================  ==========================================
field                   size
======================  ==========================================
k_codes                 ceil(latent_dim * k_bits / 8)   (per-channel)
v_codes                 ceil(latent_dim * v_bits / 8)   (per-token)
v_scale   (fp16)        2
v_zp      (fp16)        2
k_row_scale (fp16)      2   (VarN token scale, score path)
rope      (bf16)        rope_dim * 2  (exact, post-RoPE)
======================  ==========================================

Per-page tail (page == one VarN tile of ``group`` tokens, carried via
``page_tail_bytes_per_token`` = tail_bytes / group):

======================  ==========================================
k_col_scale (fp16)      latent_dim * 2   (VarN channel scale, score path)
k_col_rt_scale (fp16)   latent_dim * 2   (per-channel RTN scale)
k_col_rt_zp    (fp16)   latent_dim * 2   (per-channel RTN zero-point)
======================  ==========================================

Dequantization identities:

    k_hat[r, c] = (k_codes[r, c] - k_col_rt_zp[c]) * k_col_rt_scale[c]
                  * k_col_scale[c] * k_row_scale[r]
    v_hat[r, c] = (v_codes[r, c] * v_scale[r] + v_zp[r])

The Hadamard rotation is applied by the caller at store time and absorbed
into the q/v projections at load, so records hold rotated latents and no
inverse transform is needed at read time.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from vllm.v1.kv_cache_interface import parse_kvarn_mla_dtype

__all__ = [
    "KvarnMlaLayout",
    "kvarn_mla_layout_from_dtype",
    "hadamard_matrix",
    "varn_balance",
    "pack_kvarn_mla_tile",
    "unpack_kvarn_mla_tile",
    "pack_bits",
    "unpack_bits",
]


@dataclasses.dataclass(frozen=True)
class KvarnMlaLayout:
    """Byte geometry of the dual-rate KVarN MLA record."""

    k_bits: int
    v_bits: int
    group: int
    latent_dim: int = 512
    rope_dim: int = 64

    @property
    def k_code_bytes(self) -> int:
        return math.ceil(self.latent_dim * self.k_bits / 8)

    @property
    def v_code_bytes(self) -> int:
        return math.ceil(self.latent_dim * self.v_bits / 8)

    @property
    def k_col_scale_bytes(self) -> int:
        # VarN channel scale + RTN scale + RTN zero-point, fp16 each.
        return self.latent_dim * 2 * 3

    @property
    def record_bytes(self) -> int:
        token_bytes = (
            self.k_code_bytes
            + self.v_code_bytes
            + 2  # v_scale
            + 2  # v_zp
            + 2  # k_row_scale
            + self.rope_dim * 2  # exact bf16 rope
        )
        return (token_bytes + 7) // 8 * 8

    @property
    def page_tail_bytes_per_token(self) -> float:
        return self.k_col_scale_bytes / self.group

    @property
    def page_size_bytes(self) -> int:
        return self.record_bytes * self.group + self.k_col_scale_bytes

    @property
    def k_code_offset(self) -> int:
        return 0

    @property
    def v_code_offset(self) -> int:
        return self.k_code_bytes

    @property
    def v_scale_offset(self) -> int:
        return self.v_code_offset + self.v_code_bytes

    @property
    def v_zp_offset(self) -> int:
        return self.v_scale_offset + 2

    @property
    def k_row_scale_offset(self) -> int:
        return self.v_zp_offset + 2

    @property
    def rope_offset(self) -> int:
        return self.k_row_scale_offset + 2


def kvarn_mla_layout_from_dtype(kv_cache_dtype: str) -> KvarnMlaLayout:
    parsed = parse_kvarn_mla_dtype(kv_cache_dtype)
    if parsed is None:
        raise ValueError(f"Not a kvarn MLA dtype: {kv_cache_dtype!r}")
    k_bits, v_bits, group = parsed
    return KvarnMlaLayout(k_bits=k_bits, v_bits=v_bits, group=group)


@torch.no_grad()
def hadamard_matrix(dim: int, device: torch.device) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix (dim must be a power of two)."""
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < dim:
        h = torch.cat(
            [torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0
        )
    assert h.shape[0] == dim
    return h / math.sqrt(dim)


def _imbalance(x: torch.Tensor) -> torch.Tensor:
    """Imb(X): max/min variance over columns plus over rows (Alg. 1, l.3-6)."""
    var_col = x.var(dim=1).clamp_min(1e-8)  # per-column variance [C]
    var_row = x.var(dim=2).clamp_min(1e-8)  # per-row variance [R]
    return var_col.amax(dim=-1) / var_col.amin(dim=-1) + var_row.amax(
        dim=-1
    ) / var_row.amin(dim=-1)


@torch.no_grad()
def varn_balance(
    tile: torch.Tensor,
    iterations: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Variance-normalize tiles by alternating log-domain dual-scale balancing.

    Implements KVarN Alg. 1 (adapted from SINQ). ``tile`` is ``[N, R, C]``.

    Returns ``(col_scale [N,1,C], row_scale [N,R,1], balanced [N,R,C])`` such
    that ``balanced == tile / col_scale / row_scale``.
    """
    n, rows, cols = tile.shape
    l_c = tile.new_zeros(n, 1, cols)
    l_r = tile.new_zeros(n, rows, 1)
    cand = tile / l_c.exp() / l_r.exp()
    best = _imbalance(cand)
    best_l_c = l_c.clone()
    best_l_r = l_r.clone()
    for _ in range(iterations):
        # Alternate: balance column variance, then row variance (log domain).
        v = cand.var(dim=1).clamp(1e-6, 1e6).unsqueeze(1)  # [N,1,C]
        l_c = (l_c + 0.5 * v.log()).clamp(-0.3, 10.0)
        cand = tile / l_c.exp() / l_r.exp()
        v = cand.var(dim=2).clamp(1e-6, 1e6).unsqueeze(2)  # [N,R,1]
        l_r = (l_r + 0.5 * v.log()).clamp(-0.3, 10.0)
        cand = tile / l_c.exp() / l_r.exp()
        curr = _imbalance(cand)
        improved = curr <= best
        best = torch.where(improved, curr, best)
        improved = improved[:, None, None]
        best_l_c = torch.where(improved, l_c, best_l_c)
        best_l_r = torch.where(improved, l_r, best_l_r)
    col_scale = best_l_c.exp()
    row_scale = best_l_r.exp()
    return col_scale, row_scale, tile / col_scale / row_scale


@torch.no_grad()
def pack_bits(
    values: torch.Tensor, bits: int
) -> torch.Tensor:
    """Pack unsigned integer codes ([..., C], 0 <= code < 2**bits) little-endian.

    Returns ``[..., ceil(C * bits / 8)]`` uint8. Bit i of the logical stream
    occupies byte ``i // 8``, bit position ``i % 8`` (LSB-first, dense).
    """
    orig_shape = values.shape
    flat = values.reshape(-1, orig_shape[-1]).to(torch.int32)
    cols = flat.shape[1]
    out_bytes = math.ceil(cols * bits / 8)
    acc = torch.zeros(
        flat.shape[0], out_bytes, dtype=torch.int32, device=values.device
    )
    bit_index = torch.arange(cols, device=values.device) * bits
    rows = flat.shape[0]
    for b in range(bits):
        bit_val = (flat >> b) & 1
        # position of this code-bit in the dense LSB-first stream
        pos = bit_index + b
        dst_byte = (pos // 8).long().unsqueeze(0).expand(rows, cols)
        dst_shift = (pos % 8).to(torch.int32)
        acc.scatter_add_(1, dst_byte, bit_val << dst_shift)
    return acc.to(torch.uint8).reshape(*orig_shape[:-1], out_bytes)


@torch.no_grad()
def unpack_bits(
    packed: torch.Tensor, bits: int, cols: int
) -> torch.Tensor:
    """Inverse of :func:`pack_bits`; returns int32 codes [..., cols]."""
    orig_shape = packed.shape
    flat = packed.reshape(-1, packed.shape[-1]).to(torch.int32)
    rows = flat.shape[0]
    bit_index = torch.arange(cols, device=packed.device) * bits
    codes = torch.zeros(rows, cols, dtype=torch.int32, device=packed.device)
    for b in range(bits):
        pos = bit_index + b
        src_byte = flat[:, (pos // 8).long()]
        codes |= ((src_byte >> (pos % 8).to(torch.int32)) & 1) << b
    return codes.reshape(*orig_shape[:-1], cols)


def _rtn_codes(
    x: torch.Tensor, bits: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-slice RTN along ``dim``.

    Returns ``(codes, scale, zp)`` with ``x_hat = codes * scale + zp``.
    """
    levels = 2**bits - 1
    mn = x.amin(dim=dim, keepdim=True)
    mx = x.amax(dim=dim, keepdim=True)
    scale = (mx - mn).clamp_min(1e-8) / levels
    codes = torch.round((x - mn) / scale).clamp_(0, levels).to(torch.int32)
    # zp folded: x_hat = (codes + mn/scale_rounded...) -> store scale & zp
    zp = mn
    return codes, scale.squeeze(dim), zp.squeeze(dim)


@torch.no_grad()
def pack_kvarn_mla_tile(
    latent: torch.Tensor,
    rope: torch.Tensor,
    layout: KvarnMlaLayout,
    varn_iters: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one tile (group tokens) of rotated latents into records.

    Args:
        latent: ``[group, latent_dim]`` float tensor, already Hadamard-rotated.
        rope: ``[group, rope_dim]`` float tensor, exact post-RoPE values.
        layout: record geometry.

    Returns:
        ``(records [group, record_bytes] uint8, page_tail [tail_bytes] uint8)``
    """
    group, latent_dim = latent.shape
    assert group == layout.group and latent_dim == layout.latent_dim
    x = latent.to(torch.float32)
    # VarN: normalize column (channel) and row (token) variance jointly.
    col_scale, row_scale, balanced = varn_balance(
        x.unsqueeze(0), iterations=varn_iters
    )
    col_scale = col_scale.squeeze(0)  # [C]
    row_scale = row_scale.squeeze(0).squeeze(-1)  # [R]

    # Score path: per-channel asymmetric RTN on the balanced tile
    # (reduce over tokens = dim 1 of the [1, R, C] batch).
    k_codes, k_col_rt_scale, k_col_rt_zp = _rtn_codes(
        balanced, layout.k_bits, dim=1
    )
    # k_hat = (k_codes * k_col_rt_scale + k_col_rt_zp) * col_scale * row_scale
    k_codes = k_codes.squeeze(0).contiguous()  # [R, C] token-major codes
    k_row_scale = row_scale.to(torch.float16)

    # Value path: per-token asymmetric RTN (reduce over channels = dim 2).
    v_codes, v_scale, v_zp = _rtn_codes(balanced, layout.v_bits, dim=2)
    v_codes = v_codes.squeeze(0)
    v_scale = v_scale.reshape(-1)
    v_zp = v_zp.reshape(-1)
    # v_hat = v_codes * v_scale + v_zp  (balanced frame)

    records = torch.zeros(
        group, layout.record_bytes, dtype=torch.uint8, device=latent.device
    )
    rec_f16 = records.view(torch.float16)
    # K codes: [R, C] -> bit stream of k_code_bytes per token.
    records[:, layout.k_code_offset : layout.k_code_offset + layout.k_code_bytes] = (
        pack_bits(k_codes, layout.k_bits).to(torch.uint8)
    )
    # V codes.
    records[:, layout.v_code_offset : layout.v_code_offset + layout.v_code_bytes] = (
        pack_bits(v_codes, layout.v_bits).to(torch.uint8)
    )
    rec_f16[:, layout.v_scale_offset // 2] = v_scale.to(torch.float16)
    rec_f16[:, layout.v_zp_offset // 2] = v_zp.to(torch.float16)
    rec_f16[:, layout.k_row_scale_offset // 2] = k_row_scale
    # Exact rope.
    rope_f16 = rope.to(torch.bfloat16).view(torch.uint8)
    records[:, layout.rope_offset : layout.rope_offset + layout.rope_dim * 2] = (
        rope_f16
    )

    # Page tail: tile-shared channel scales (duplicated nowhere).
    tail = torch.zeros(layout.k_col_scale_bytes, dtype=torch.uint8, device=latent.device)
    tail_f16 = tail.view(torch.float16)
    off = 0
    for vec in (col_scale, k_col_rt_scale, k_col_rt_zp.squeeze(-1)):
        vec = vec.reshape(-1).to(torch.float16)
        tail_f16[off : off + latent_dim] = vec
        off += latent_dim
    assert off * 2 == layout.k_col_scale_bytes
    return records, tail


@torch.no_grad()
def unpack_kvarn_mla_tile(
    records: torch.Tensor,
    page_tail: torch.Tensor,
    layout: KvarnMlaLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize one tile back to ``(latent_hat [group, latent_dim] f32, rope)``.

    Reference (non-fused) inverse of :func:`pack_kvarn_mla_tile`.
    """
    group = records.shape[0]
    latent_dim = layout.latent_dim
    rec_f16 = records.view(torch.float16)

    k_codes = unpack_bits(
        records[:, layout.k_code_offset : layout.k_code_offset + layout.k_code_bytes],
        layout.k_bits,
        latent_dim,
    ).to(torch.float32)
    v_codes = unpack_bits(
        records[:, layout.v_code_offset : layout.v_code_offset + layout.v_code_bytes],
        layout.v_bits,
        latent_dim,
    ).to(torch.float32)
    v_scale = rec_f16[:, layout.v_scale_offset // 2].to(torch.float32)
    v_zp = rec_f16[:, layout.v_zp_offset // 2].to(torch.float32)
    k_row_scale = rec_f16[:, layout.k_row_scale_offset // 2].to(torch.float32)

    tail_f16 = page_tail.view(torch.float16)
    col_scale = tail_f16[0:latent_dim].to(torch.float32)
    k_col_rt_scale = tail_f16[latent_dim : 2 * latent_dim].to(torch.float32)
    k_col_rt_zp = tail_f16[2 * latent_dim : 3 * latent_dim].to(torch.float32)

    # Score path: k_hat = (codes * rt_scale + rt_zp) * col_scale * row_scale.
    k_hat = (k_codes * k_col_rt_scale.unsqueeze(0) + k_col_rt_zp.unsqueeze(0)) * (
        col_scale.unsqueeze(0) * k_row_scale.unsqueeze(1)
    )
    # Value path: v_orig = (codes * v_scale + v_zp) * col_scale * row_scale
    # (balanced frame rescaled to the original latent frame; the Hadamard
    # inverse lives in the folded W_uv, not here).
    v_hat = (v_codes * v_scale.unsqueeze(1) + v_zp.unsqueeze(1)) * (
        col_scale.unsqueeze(0) * k_row_scale.unsqueeze(1)
    )

    rope = (
        records[:, layout.rope_offset : layout.rope_offset + layout.rope_dim * 2]
        .view(torch.bfloat16)
        .to(torch.float32)
    )
    return k_hat, v_hat, rope
