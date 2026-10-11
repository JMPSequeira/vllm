# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-only slicing and preparation of NVFP4 compressed scale planes."""

from collections.abc import Iterable

import torch

from vllm.model_executor.model_loader.csf_utils import CsfMatrix, tp_extent

_PADDED_SCALE = 0x38


class TensorView:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def get_shape(self):
        return list(self.tensor.shape)

    def get_dtype(self):
        return "U8" if self.tensor.dtype == torch.uint8 else str(self.tensor.dtype)

    def __getitem__(self, extent):
        return self.tensor[extent]


def _slice_scale_plane(
    scales,
    rows,
    columns,
    row_slice,
    column_slice,
    out_rows=None,
    out_columns=None,
):
    """Regroup a logical TP rectangle and cover any zero-filled weight tails."""
    from b12x.loader._csf_blocks import pad_csf_blocks, slice_csf_blocks

    r0, r1 = row_slice
    c0, c1 = column_slice
    if (scales.rows, scales.columns, scales.quantization) != (rows, columns, "nvfp4"):
        raise ValueError("NVFP4-CSF block geometry differs from its packed weights")
    out_rows = r1 - r0 if out_rows is None else out_rows
    out_columns = c1 - c0 if out_columns is None else out_columns
    return pad_csf_blocks(
        slice_csf_blocks(scales, row_range=row_slice, column_range=column_slice),
        rows=out_rows,
        columns=out_columns,
        row_base=_PADDED_SCALE,
    )


def prepare_nvfp4_csf_weights(
    experts: Iterable[tuple[CsfMatrix, CsfMatrix, CsfMatrix]],
    *,
    num_experts,
    hidden_size,
    intermediate_size,
    tp_rank,
    tp_size,
    device,
    w13_scale_scratch,
    w2_scale_scratch,
    local_size=None,
    packed=None,
):
    """Slice and upload up/gate/down-ordered expert projections.

    The iterable must yield exactly ``num_experts`` projection triples. Tensor
    stores, manifests and model-specific tensor names belong to the caller.
    Expanded scale buffers remain caller-owned for serialized layer execution.
    ``local_size`` is the per-rank width of a TP-padded expert (GLM-5.3 at
    TP6: 2048 channels padded to 6 x 352): rank r holds checkpoint channels
    [r * local_size, (r + 1) * local_size) and zeros past the checkpoint.
    ``packed`` is an optional (w13, w2) pair that already holds every sliced
    projection in the packed layout; it is used as the weight storage, and
    the projections then supply only scales and calibration.
    """
    from b12x.loader._csf_blocks import concatenate_csf_blocks
    from b12x.moe.fused_moe import CsfBlockPlanes, Nvfp4CsfWeights, PackedWeights

    if num_experts <= 0 or hidden_size <= 0 or hidden_size % 128:
        raise ValueError("NVFP4-CSF requires experts and 128-aligned hidden channels")
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("NVFP4-CSF requires a valid TP rank and size")
    if local_size is None:
        first, last = tp_extent(intermediate_size, tp_rank, tp_size, 32)
        local = last - first
    else:
        local = int(local_size)
        if local <= 0 or local % 32 or local * tp_size < intermediate_size:
            raise ValueError(
                "NVFP4-CSF padded shards need 32-aligned widths covering the expert"
            )
        first = min(tp_rank * local, intermediate_size)
        last = min(first + local, intermediate_size)
    real = last - first
    if real <= 0:
        raise ValueError("NVFP4-CSF requires checkpoint channels on every TP rank")
    shapes = (
        (num_experts, 2 * local, hidden_size // 2),
        (num_experts, hidden_size, local // 2),
    )
    if packed is None:
        w13, w2 = (torch.zeros(s, dtype=torch.uint8, device=device) for s in shapes)
    else:
        w13, w2 = packed
        if any(
            tuple(t.shape) != s or t.dtype != torch.uint8
            for t, s in zip(packed, shapes, strict=True)
        ):
            raise ValueError("NVFP4-CSF packed weights do not match the geometry")
    # Model loaders may set BF16 as the default dtype. Calibration belongs to
    # the source FP32 contract and must not be rounded with the model weights.
    g13, g2 = (
        torch.empty(num_experts, dtype=torch.float32, device="cpu") for _ in range(2)
    )
    a13, a2 = (
        torch.empty(num_experts, dtype=torch.float32, device="cpu") for _ in range(2)
    )
    scales13, scales2 = [], []

    def scalar(value, role):
        if value is None or value.numel() != 1 or value.dtype != torch.float32:
            raise ValueError(f"NVFP4 {role} must be one FP32 value")
        if not bool(torch.isfinite(value).all() and (value > 0).all()):
            raise ValueError(f"NVFP4 {role} must be positive and finite")
        return value.reshape(())

    for expert, (first_projection, second_projection, down) in zip(
        range(num_experts), experts, strict=True
    ):
        s13, global13, input13 = [], [], []
        for matrix, projection in enumerate(
            (first_projection, second_projection, down)
        ):
            view = projection.weight
            expected = (
                [intermediate_size, hidden_size // 2]
                if matrix < 2
                else [hidden_size, intermediate_size // 2]
            )
            if view.get_shape() != expected or view.get_dtype() != "U8":
                raise ValueError(
                    f"NVFP4 nibble geometry/dtype mismatch: "
                    f"expert={expert}, projection={matrix}"
                )
            global_scale, input_scale = (
                scalar(projection.global_scale, "global_scale"),
                scalar(projection.input_scale, "input_scale"),
            )
            if matrix < 2:
                if real and packed is None:
                    w13[expert, matrix * local : matrix * local + real].copy_(
                        view[first:last, :]
                    )
                rows, columns = intermediate_size, hidden_size // 16
                row_slice, column_slice = (first, last), (0, columns)
                out_rows, out_columns = local, columns
                global13.append(global_scale)
                input13.append(input_scale)
            else:
                if real and packed is None:
                    w2[expert, :, : real // 2].copy_(view[:, first // 2 : last // 2])
                rows, columns = hidden_size, intermediate_size // 16
                row_slice, column_slice = (0, rows), (first // 16, last // 16)
                out_rows, out_columns = rows, local // 16
                g2[expert], a2[expert] = global_scale, input_scale
            scales = _slice_scale_plane(
                projection.scales,
                rows,
                columns,
                row_slice,
                column_slice,
                out_rows,
                out_columns,
            )
            if matrix < 2:
                s13.append(scales)
            else:
                scales2.append(scales)
        if not torch.equal(global13[0], global13[1]):
            raise ValueError("NVFP4-CSF requires equal gate/up global weight scales")
        g13[expert] = global13[0]
        a13[expert] = torch.stack(input13).amax()
        scales13.append(concatenate_csf_blocks(s13))
    return Nvfp4CsfWeights(
        packed=PackedWeights(
            w13=w13,
            w2=w2,
            w13_block_scales=w13_scale_scratch,
            w2_block_scales=w2_scale_scratch,
            w13_global_scales=g13.to(device),
            w2_global_scales=g2.to(device),
            input_scale=a13.to(device).reciprocal(),
            intermediate_scale=a2.to(device).reciprocal(),
            immutable_input_scales=True,
        ),
        w13_scales=CsfBlockPlanes(tuple(scales13)),
        w2_scales=CsfBlockPlanes(tuple(scales2)),
    )
