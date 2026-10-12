# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the KVarN MLA record codec (layout, VarN, pack/unpack)."""

import pytest
import torch

from vllm.v1.attention.ops.kvarn_mla import (
    KvarnMlaLayout,
    hadamard_matrix,
    kvarn_mla_layout_from_dtype,
    pack_bits,
    unpack_bits,
    unpack_kvarn_mla_tile,
    varn_balance,
    pack_kvarn_mla_tile,
)
from vllm.v1.kv_cache_interface import get_kv_quant_mode, parse_kvarn_mla_dtype

DTYPES = [
    "kvarn_mla_k4v4_g256",
    "kvarn_mla_k5v5_g256",
    "kvarn_mla_k4v2_g256",
]


@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype_dispatch(dtype: str):
    assert get_kv_quant_mode(dtype) == 11  # KVQuantMode.KVARN_MLA
    k_bits, v_bits, group = parse_kvarn_mla_dtype(dtype)
    layout = kvarn_mla_layout_from_dtype(dtype)
    assert (layout.k_bits, layout.v_bits, layout.group) == (k_bits, v_bits, group)
    assert layout.record_bytes % 8 == 0


def test_layout_geometry():
    k4v4 = KvarnMlaLayout(k_bits=4, v_bits=4, group=256)
    k5v5 = KvarnMlaLayout(k_bits=5, v_bits=5, group=256)
    k4v2 = KvarnMlaLayout(k_bits=4, v_bits=2, group=256)
    # 512*4/8=256 code + 256 code + 2+2+2 scales = 518 -> 520
    assert k4v4.record_bytes == 520
    # 320 + 320 + 6 = 646 -> 648
    assert k5v5.record_bytes == 648
    # 256 + 128 + 6 = 390 -> 392
    assert k4v2.record_bytes == 392
    for layout in (k4v4, k5v5, k4v2):
        # tail = 3 * 512 * 2 = 3072; page = records + tail
        assert layout.k_col_scale_bytes == 3072
        assert layout.page_size_bytes == layout.record_bytes * 256 + 3072
        # offsets must be non-overlapping and inside the record
        offs = [
            layout.k_code_offset,
            layout.v_code_offset,
            layout.v_scale_offset,
            layout.v_zp_offset,
            layout.k_row_scale_offset,
        ]
        assert offs == sorted(offs)
        assert layout.k_row_scale_offset + 2 <= layout.record_bytes


@pytest.mark.parametrize("bits", [2, 4, 5])
@pytest.mark.parametrize("cols", [512, 517])
def test_pack_bits_roundtrip(bits: int, cols: int):
    gen = torch.Generator(device="cpu").manual_seed(0)
    codes = torch.randint(
        0, 2**bits, (7, cols), generator=gen, dtype=torch.int32
    )
    packed = pack_bits(codes, bits)
    assert packed.shape[-1] == (cols * bits + 7) // 8
    assert (unpack_bits(packed, bits, cols) == codes).all()


def test_varn_balance_reduces_imbalance():
    gen = torch.Generator(device="cpu").manual_seed(1)
    # Deliberately imbalanced: channel 0 huge, token 3 huge.
    tile = torch.randn(2, 64, 128, generator=gen)
    tile[:, :, 0] *= 50.0
    tile[:, 3, :] *= 30.0

    def imb(x):
        col = x.var(dim=1).clamp_min(1e-8)
        row = x.var(dim=2).clamp_min(1e-8)
        return (col.amax(1) / col.amin(1) + row.amax(1) / row.amin(1)).sum()

    assert imb(tile) > 100.0
    col_scale, row_scale, balanced = varn_balance(tile, iterations=8)
    assert col_scale.shape == (2, 1, 128)
    assert row_scale.shape == (2, 64, 1)
    assert imb(balanced) < imb(tile) / 10.0
    # exact rescale identity
    torch.testing.assert_close(
        balanced * col_scale * row_scale, tile, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_pack_unpack_parity(dtype: str):
    """Relative-error bound: 4-bit RTN on a balanced tile, loose sanity gate."""
    layout = kvarn_mla_layout_from_dtype(dtype)
    gen = torch.Generator(device="cpu").manual_seed(2)
    latent = torch.randn(layout.group, layout.latent_dim, generator=gen)

    records, page_tail = pack_kvarn_mla_tile(latent, layout)
    assert records.shape == (layout.group, layout.record_bytes)
    assert page_tail.numel() == layout.k_col_scale_bytes

    k_hat, v_hat = unpack_kvarn_mla_tile(records, page_tail, layout)
    assert k_hat.shape == latent.shape
    assert v_hat.shape == latent.shape
    # Dequantized scores preserve cosine structure of the Hadamard-rotated
    # latent (what attention logits see): mean cosine similarity must be high.
    k_hat_n = k_hat / k_hat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    lat_n = latent / latent.norm(dim=-1, keepdim=True)
    cos = (k_hat_n * lat_n).sum(-1)
    assert cos.mean() > 0.97, f"score-path cosine degraded: {cos.mean():.4f}"
    # Value path relative error bound for the variant's bits.
    rel = (v_hat - latent).norm() / latent.norm()
    if layout.v_bits >= 4:
        assert rel < 0.12, f"value-path relative error too high: {rel:.4f}"
    else:  # 2-bit value path
        assert rel < 0.65, f"value-path relative error too high: {rel:.4f}"


def test_pack_unpack_vs_fp16_reference_cosine():
    """k5v5 must beat k4v2 on score-path fidelity (monotone in bits)."""
    gen = torch.Generator(device="cpu").manual_seed(3)
    latent = torch.randn(256, 512, generator=gen)

    def mean_cos(dtype):
        layout = kvarn_mla_layout_from_dtype(dtype)
        recs, tail = pack_kvarn_mla_tile(latent, layout)
        k_hat, _ = unpack_kvarn_mla_tile(recs, tail, layout)
        n = k_hat / k_hat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ref = latent / latent.norm(dim=-1, keepdim=True)
        return (n * ref).sum(-1).mean().item()

    assert mean_cos("kvarn_mla_k5v5_g256") >= mean_cos("kvarn_mla_k4v2_g256")


def test_hadamard_orthonormal():
    h = hadamard_matrix(512, torch.device("cpu"))
    torch.testing.assert_close(h @ h.T, torch.eye(512), rtol=1e-5, atol=1e-6)
