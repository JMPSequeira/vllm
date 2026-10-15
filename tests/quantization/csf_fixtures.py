# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small compressed matrices with known bytes across TP partition boundaries."""

import numpy as np
import torch

from vllm.model_executor.model_loader.csf_utils import CsfMatrix


class TensorView:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_shape(self):
        return list(self.tensor.shape)

    def get_dtype(self):
        return "I8" if self.tensor.dtype == torch.int8 else "U8"

    def __getitem__(self, extent):
        return self.tensor[extent]


def matrix(rows, channels, *, group_size, seed, finite_scales=False):
    """Return a source projection and its uncompressed reference scale bytes."""
    from b12x.loader._csf_blocks import export_csf_blocks

    columns = channels // group_size
    bases = (
        (np.arange(rows) + seed) % (16 if finite_scales else 190)
        + (43 if finite_scales else 20)
    ).astype(np.uint8)
    offsets = (np.arange(rows * columns) + seed) % (2 if group_size == 32 else 16)
    offsets = offsets.reshape(rows, columns).astype(np.uint8)
    logical = bases[:, None] + offsets
    positions = np.unique(
        [
            0,
            columns - 1,
            rows // 2 * columns - 1,
            rows // 2 * columns,
            rows * columns - 1,
        ]
    ).astype(np.uint32)
    values = (np.arange(len(positions), dtype=np.uint32) * 7 + seed) % 256
    if finite_scales and group_size == 16:
        values %= 127
    logical.flat[positions] = values.astype(np.uint8)
    packed = (
        np.packbits(offsets, axis=1, bitorder="little")
        if group_size == 32
        else offsets[:, ::2] | (offsets[:, 1::2] << 4)
    )
    fixed = np.concatenate(
        (bases.reshape(rows // 16, 16), packed.reshape(rows // 16, -1)), axis=1
    )
    weight = (torch.arange(rows * channels // 2) + seed).to(torch.uint8)
    source = CsfMatrix(
        weight=TensorView(weight.reshape(rows, channels // 2)),
        scales=export_csf_blocks(
            fixed,
            positions | (values << 24),
            rows=rows,
            columns=columns,
            quantization="mxfp4" if group_size == 32 else "nvfp4",
        ),
    )
    return source, torch.from_numpy(logical.copy())


def decode_planes(planes, rows, columns, group_size):
    """Reconstruct CPU scale bytes independently of B12X preparation."""
    from types import SimpleNamespace

    from b12x.loader._csf_blocks import CsfBlockPlane
    from b12x.loader._csf_source import CsfSource

    if isinstance(planes, CsfSource):
        source = planes
        result = []
        starts = source.starts.cpu().tolist()
        r0, r1 = source.row_range or (0, source.source_rows)
        c0, c1 = source.column_range or (0, source.source_columns)
        for index in range(source.segments):
            components = [t[index].cpu().numpy() for t in source.components[:-1]]
            exceptions = (
                source.components[-1][starts[index] : starts[index + 1]].cpu().numpy()
            )
            if source.encoding == "csf":
                fixed = components[0]
                sr, sc = source.source_rows, source.source_columns
                bits = 1 if group_size == 32 else 4
                packed = fixed[:, 16:].reshape(sr, -1)
                shifts = np.arange(sc) * bits
                logical = fixed[:, :16].reshape(sr, 1) + (
                    (packed[:, shifts // 8] >> (shifts % 8)) & ((1 << bits) - 1)
                )
                logical = logical.astype(np.uint8)
                logical.flat[exceptions & 0xFFFFFF] = (exceptions >> 24).astype(
                    np.uint8
                )
                logical = torch.from_numpy(logical)
            else:
                plane = CsfBlockPlane(
                    source.source_rows,
                    source.source_columns,
                    source.quantization,
                    *components,
                    exceptions,
                )
                logical = decode_planes(
                    SimpleNamespace(planes=(plane,)),
                    plane.rows,
                    plane.columns,
                    group_size,
                )[0]
            selected = torch.full(
                (source.rows // source.join, source.columns),
                source.padding_base,
                dtype=torch.uint8,
            )
            selected[: r1 - r0, : c1 - c0] = logical[r0:r1, c0:c1]
            result.append(selected)
        return torch.stack(
            [
                torch.cat(result[i : i + source.join])
                for i in range(0, len(result), source.join)
            ]
        )
    if isinstance(planes.planes[0], CsfSource):
        return torch.cat(
            [decode_planes(p, rows, columns, group_size) for p in planes.planes]
        )
    result = []
    bits = 1 if group_size == 32 else 4
    for plane in planes.planes:
        logical = np.empty((rows, columns), dtype=np.uint8)
        block = group = cursor = 0
        for tr in range(0, rows, 128):
            for tc in range(0, columns, 4):
                for c in range(tc, min(tc + 4, columns), 2):
                    for r in range(tr, min(tr + 128, rows), 32):
                        width = min(2, columns - c)
                        size = 32 * width * bits // 8
                        fields = np.unpackbits(
                            plane.codes[cursor : cursor + size], bitorder="little"
                        ).reshape(-1, bits)
                        offsets = (fields * (1 << np.arange(bits))).sum(axis=1)
                        values = (
                            plane.bases[r : r + 32, None] + offsets.reshape(32, width)
                        ).astype(np.uint8)
                        page = group // 32
                        start, stop = map(
                            int, plane.offsets[block + page : block + page + 2]
                        )
                        origin = int(plane.page_bases[page]) // 2
                        for record in plane.exceptions[origin + start : origin + stop]:
                            row, column = divmod(int(record) & 255, 2)
                            values[row, column] = int(record) >> 8
                        logical[r : r + 32, c : c + width] = values
                        block += 1
                        cursor += size
                group += 1
        result.append(torch.from_numpy(logical))
    return torch.stack(result)
