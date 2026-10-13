"""End-to-end torch A/B: kvarn pipeline vs exact MLA attention.

Verifies the DESIGN math assembly (rotation + folds + codec) on GPU:
  reference:  score = (q_nope @ W_UK_T) . kv_c ;  o = P @ (kv_c @ W_UV)
  kvarn:      l = kv_c @ H^T (store frame)
              q_abs' = q_nope @ (W_UK_T @ H^T)
              score' = q_abs' . dequant(pack(l))
              o'     = P @ dequant(l) @ (H @ W_UV)
  H orthonormal => o' ~= o up to quantization noise.

Run: CUDA_VISIBLE_DEVICES=2 ~/lil/vllm/.venv/bin/python tests/kvarn_ab_math.py
"""

import math
import sys

import torch

sys.path.insert(0, "/home/js/lil/vllm")
from vllm.v1.attention.ops.kvarn_mla import (  # noqa: E402
    KvarnMlaLayout,
    hadamard_matrix,
    kvarn_mla_layout_from_dtype,
    pack_kvarn_mla_records,
    unpack_kvarn_mla_records,
)

torch.manual_seed(7)
DEV = torch.device("cuda:0")
GROUP, L, N, V = 256, 512, 4, 512
layout = kvarn_mla_layout_from_dtype("kvarn_mla_k4v4_g256")

kv_c = torch.randn(GROUP, L, device=DEV, dtype=torch.float32) * 0.5
q_nope = torch.randn(GROUP, N, 128, device=DEV, dtype=torch.float32) * 0.5
W_UK = torch.randn(L, N, 128, device=DEV, dtype=torch.float32) * 0.05
W_UV = torch.randn(L, N, V, device=DEV, dtype=torch.float32) * 0.05

# ---- reference (exact) ----
q_c = torch.einsum("snp,lnp->snl", q_nope, W_UK)  # [S,N,L]
score = torch.einsum("snl,sl->sn", q_c, kv_c) * (1.0 / math.sqrt(L))
P = torch.softmax(score, dim=-1)
# o per token: o_s,n = sum_t P[s,t] * v_t,n ;  v_t,n = kv_c[t] @ W_UV[:, n, :]
W_UV_flat = W_UV.reshape(L, N * V)  # [L, N*V]
v = (kv_c @ W_UV_flat).view(GROUP, N, V)
o_ref = torch.einsum("sn,snv->snv", P, v)

# ---- kvarn pipeline ----
h = hadamard_matrix(L, DEV).to(torch.float32)
h_t = h.transpose(-2, -1)
l_rot = kv_c @ h_t  # stored frame [S, L]

records = pack_kvarn_mla_records(l_rot, layout)
l_hat, _v2_hat = unpack_kvarn_mla_records(records, layout)

# folded weights
W_UK_T = W_UK.permute(1, 2, 0)  # [N, P, L]
W_UK_T_folded = W_UK_T @ h_t  # q_abs' = q_nope @ W_UK_T' (rotated frame)
W_UV_n = W_UV.permute(1, 0, 2)  # [N, L, V]
W_UV_folded = h @ W_UV_n  # o = P l @ (H W_UV)

q_abs = torch.einsum("snp,npl->snl", q_nope, W_UK_T_folded)
score2 = torch.einsum("snl,sl->sn", q_abs, l_hat) * (1.0 / math.sqrt(L))
P2 = torch.softmax(score2, dim=-1)
v2 = torch.einsum("sl,nlv->snv", l_hat, W_UV_folded)
o2 = torch.einsum("sn,snv->snv", P2, v2)

lerr = (l_hat - l_rot).abs()
lsnr = 10 * torch.log10((l_rot**2).mean() / ((l_hat - l_rot) ** 2).mean())
print(f"codec: mean|l_rot|={l_rot.abs().mean():.4f} mean|l_err|={lerr.mean():.6f} SNR={lsnr:.1f} dB")
err = (o2 - o_ref).abs()
ref = o_ref.abs().mean()
snr = 10 * torch.log10((o_ref**2).mean() / ((o2 - o_ref) ** 2).mean())
print(f"mean|o_ref|={ref:.4f} mean|err|={err.mean():.6f} max|err|={err.max():.4f} SNR={snr:.1f} dB")
# also compare score frames directly
serr = (score2 - score).abs().mean()
print(f"mean|score_err|={serr:.6f} mean|score|={score.abs().mean():.4f}")
