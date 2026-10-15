# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 mixed-format KDA input projection and block-FP8 weight dequantization."""

from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMixedPrecisionConfig,
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
)


def _mixed_precision_config(quantized_layers: dict) -> ModelOptMixedPrecisionConfig:
    return ModelOptMixedPrecisionConfig(
        kv_cache_quant_method=None,
        exclude_modules=[],
        quantized_layers=quantized_layers,
        fp8_config=ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        ),
        nvfp4_config=ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
        w4a16_nvfp4_config=ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
        mxfp8_config=ModelOptMxFp8Config(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
    )


@pytest.mark.parametrize(
    ("qkv_algo", "small_algo", "split"),
    [
        ("MXFP8", None, True),
        ("FP8_PB_WO", None, True),
        ("MXFP8", "MXFP8", False),
        (None, None, False),
    ],
)
def test_glm_kda_input_projection_splits_mixed_formats(qkv_algo, small_algo, split):
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
        split_kda_input_projection,
    )
    from vllm.models.glm5next.nvidia.model import GLM5NEXT_PACKED_MODULES_MAPPING

    prefix = "model.layers.3.self_attn"
    layers = {}
    for name, algo in (
        ("q_proj", qkv_algo),
        ("k_proj", qkv_algo),
        ("v_proj", qkv_algo),
        ("b_proj", small_algo),
        ("f_a_proj", small_algo),
    ):
        if algo is not None:
            layers[f"{prefix}.{name}"] = {"quant_algo": algo}
    config = _mixed_precision_config(layers)
    config.packed_modules_mapping = GLM5NEXT_PACKED_MODULES_MAPPING

    assert split_kda_input_projection(config, prefix) is split
    if split:
        # The split GEMMs resolve to their own formats instead of the fused
        # lookup applying the Q/K/V format to the BF16 rows.
        assert config._resolve_quant_algo(f"{prefix}.in_proj_qkv") == qkv_algo
        assert config._resolve_quant_algo(f"{prefix}.in_proj_bfa") is None
    assert split_kda_input_projection(None, prefix) is False


def test_glm_kda_input_projection_rejects_mixed_qkv_formats():
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
        split_kda_input_projection,
    )

    prefix = "model.layers.3.self_attn"
    config = _mixed_precision_config(
        {
            f"{prefix}.q_proj": {"quant_algo": "FP8_PB_WO"},
            f"{prefix}.k_proj": {"quant_algo": "MXFP8"},
            f"{prefix}.v_proj": {"quant_algo": "MXFP8"},
        }
    )
    with pytest.raises(ValueError, match="one format for q/k/v"):
        split_kda_input_projection(config, prefix)


def test_block_fp8_weights_dequantize_without_running_the_linear():
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_and_maybe_dequant_weights,
    )

    out_rows, in_cols = 130, 256
    torch.manual_seed(0)
    reference = torch.randn(out_rows, in_cols) * 0.5
    # Kernels pad the rows to whole 128-row blocks; scales are [ceil(out/128), in/128].
    weight = torch.zeros(256, in_cols, dtype=torch.float8_e4m3fn)
    weight[:out_rows] = reference.to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 2.0], [4.0, 0.25]], dtype=torch.float32)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    layer.weight_block_size = [128, 128]
    layer.output_size_per_partition = out_rows
    layer.input_size_per_partition = in_cols
    layer.quant_method = Mock()
    layer.quant_method.apply.side_effect = AssertionError("must not run the GEMM")

    result = get_and_maybe_dequant_weights(layer, out_dtype=torch.float32)

    expected = weight[:out_rows].to(torch.float32) * scale.repeat_interleave(128, 0)[
        :out_rows
    ].repeat_interleave(128, 1)
    assert result.shape == (out_rows, in_cols)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_kda_activation_option_marks_projections(monkeypatch):
    """a16 tags every call for W8A16; prefill tags the layer for a second plan."""
    import vllm.envs as envs
    from vllm.models.glm5next.nvidia.kda import mark_b12x_activation_mode

    monkeypatch.setenv("VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE", "prefill")
    assert envs.VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE == "prefill"
    monkeypatch.delenv("VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE")
    assert envs.VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE == "auto"

    a16, prefill, auto = (torch.nn.Linear(8, 8) for _ in range(3))
    mark_b12x_activation_mode((a16,), "a16")
    mark_b12x_activation_mode((prefill,), "prefill")
    mark_b12x_activation_mode((auto,), "auto")
    assert a16.b12x_activation_mode == "a16"
    assert prefill.b12x_prefill_a16 and not hasattr(prefill, "b12x_activation_mode")
    assert not hasattr(auto, "b12x_activation_mode")
    assert not hasattr(auto, "b12x_prefill_a16")


def test_prefill_holder_takes_prefill_sized_calls(monkeypatch):
    """Calls of at least 128 rows run the A16 plan; shorter ones keep the default."""
    from vllm.utils import b12x as b12x_utils

    calls = []

    class Holder:
        def __init__(self, tag, workspace):
            self.tag, self.workspace = tag, workspace

        def run(self, source, bias):
            calls.append(self.tag)
            return source

        def get_workspace_size(self, rows):
            return self.workspace

    layer = torch.nn.Module()
    layer.b12x_linear = Holder("auto", 100)
    layer.b12x_linear_prefill = Holder("a16", 4096)
    monkeypatch.setattr(b12x_utils, "b12x_layer", lambda name: layer)
    monkeypatch.setattr(b12x_utils, "_resolve_layer_name", lambda name: name)
    rows = b12x_utils.B12X_MXFP8_PREFILL_A16_MIN_ROWS
    for m in (1, 8, rows - 1, rows, 4096):
        b12x_utils._b12x_blockscaled_linear(torch.zeros(m, 16), None, 16, "layer")
    assert calls == ["auto", "auto", "auto", "a16", "a16"]
    assert b12x_utils.get_b12x_projection_workspace_sizes(1, layer) == (4096,)
    layer.b12x_linear_prefill = None
    assert b12x_utils.get_b12x_projection_workspace_sizes(1, layer) == (100,)
