# MXFP4-CSF and NVFP4-CSF

The `csf_32x2_v1` encoding compresses FP4 block-scale tensors losslessly. Weight nibbles,
original scale bytes, calibration values and other tensors are preserved.
The B12X backend keeps scales compressed between layers and reconstructs
the needed scales into shared GPU scratch before expert computation.

Use `--quantization mxfp4_csf --load-format mxfp4_csf` for supported Kimi-K3,
DeepSeek-V4.1-Flash, DeepSeek-V4-Flash or DeepSeek-V4-Flash-Vision-Exp MXFP4
containers. NVFP4 compressed scales use ordinary ModelOpt mixed-precision
recipes and the standard safetensors loader.

MXFP4 uses a one-bit unsigned offset from a row's base byte. NVFP4 uses a
four-bit offset. Both preserve out-of-interval scale bytes as exceptions.
Expert activations follow each model's native B12X policy. DeepSeek-V4.1-Flash
and DeepSeek-V4-Flash MXFP4-CSF quantize expert activations to MXFP8 (W4A8)
unless `VLLM_B12X_MOE_FP4_FORCE_A16=1` requests BF16 activations. Kimi-K3
MXFP4-CSF uses BF16 expert activations. NVFP4-CSF uses native calibrated FP4
expert activations.

## GLM-5.3-Flash on two 96 GiB Blackwell GPUs

The checkpoint's `config.json` identifies compressed scales in each affected
expert recipe. Other recipes, including MTP and vision, retain their own
quantization settings:

```json
{
  "quantization_config": {
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "quantized_layers": {
      "model.language_model.layers.3.mlp.experts": {
        "quant_algo": "NVFP4",
        "group_size": 16,
        "weight_scale_encoding": "csf_32x2_v1"
      }
    }
  }
}
```

Each compressed `weight_scale` has five components in ordinary safetensors
shards: `.csf32_bases` and `.csf32_codes` (uint8), `.csf32_page_bases`
(uint32), and `.csf32_offsets` and `.csf32_exceptions` (uint16). The standard
shard index locates every component, including components in separate files.
Blocks cover 32 rows and two scale columns, grouped into logical 128-row by
four-column tiles. Boundary tiles store only logical fields. Index pages
contain 32 whole tile groups. No CSF manifest, checkpoint-root setting, or
custom load format is required for NVFP4.

Run in an environment with matching vLLM/B12X CSF support:

```bash
vllm serve /models/GLM-5.3-Flash-NVFP4-CSF \
  --quantization modelopt_mixed --load-format safetensors \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --moe-backend b12x --linear-backend b12x --attention-backend B12X \
  --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}' \
  --language-model-only --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 1073741824 \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 1024 \
  --cudagraph-capture-sizes 1 2 4 --max-cudagraph-capture-size 4 \
  --host 0.0.0.0 --port 8000
```

This example reserves 1 GiB/GPU for KV cache. Increase
`--kv-cache-memory-bytes` only when the model, execution buffers and CUDA
graphs leave enough memory. The quantization method supports tensor parallel execution
with pipeline parallel size 1, without expert/data parallelism or ubatching.
Separate concurrent execution streams need separate decoded-scale scratch.
Leave `VLLM_B12X_MOE_FP4_FORCE_A16=0` (the default) to retain the source
NVFP4 activation arithmetic.

## Qwen3.8-Flash-Next on two 96 GiB Blackwell GPUs

Use the same per-recipe scale encoding as GLM. Main routed
experts retain calibrated four-bit activations; MTP, vision, shared experts
and the PLE embedding table retain their source precision.

```bash
VLLM_PLE_TABLE_MEMORY=ram vllm serve /models/Qwen3.8-Flash-Next-NVFP4-CSF \
  --quantization modelopt_mixed --load-format safetensors \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --dtype bfloat16 \
  --moe-backend b12x --linear-backend b12x \
  --mm-encoder-tp-mode data --mamba-cache-mode align \
  --no-enable-prefix-caching --enable-chunked-prefill \
  --kv-cache-dtype bfloat16 --kv-cache-memory-bytes 2147483648 --block-size 16 \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 1024 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4]}' \
  --reasoning-parser qwen3 --host 0.0.0.0 --port 8000
```

`VLLM_PLE_TABLE_MEMORY` is unset by default and leaves placement to the
model configuration. The explicit `ram` setting above requires about 26.8
GiB of host RAM for both TP ranks' table payloads, plus loading and process
memory. `VLLM_PLE_TABLE_MEMORY=disk` selects file-backed table access; its
performance depends on storage and is not represented by the RAM example.
The example reserves 2 GiB/GPU for KV cache. Qwen's intermediate dimension
supports TP1, TP2, and TP4 scale preparation. Its TP4 intermediate partitions
have 160 channels. The checkpoint remains unpadded; expansion writes the native
kernel's boundary tiles into shared scratch.

## DeepSeek-V4-Flash on two 96 GiB Blackwell GPUs

This also applies to DeepSeek-V4-Flash-Vision-Exp. The serving directory
contains the source tokenizer metadata and a `config.json` whose
`quantization_config` keeps the source block-FP8 fields, including
`weight_block_size` equal to `[128, 128]`, and sets `quant_method` to
`mxfp4_csf`, `format_version` to 1 and an absolute `checkpoint_root`. The
checkpoint at that root declares `weight_scale_encoding: "csf_32x2_v1"`
and uses a standard `model.safetensors.index.json` with the same five scale
components as NVFP4. Its model configuration determines expert tensor names.

The `deepseek_v4_flash` container compresses the routed-expert scales of the
43 target layers. Attention, dense and shared-expert weights retain block
FP8. MTP/DSpark draft layers (`mtp.*`), the vision tower, the aligner and the
image embeddings retain their source tensors and precision. A `config.json`
with a vision tower selects the multimodal model.

```bash
vllm serve /models/DeepSeek-V4-Flash-MXFP4-CSF/serve \
  --quantization mxfp4_csf --load-format mxfp4_csf \
  --tensor-parallel-size 2 --moe-backend b12x --attention-backend B12X \
  --block-size 256 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 --host 0.0.0.0 --port 8000
```

The scale reader accepts equal TP partitions containing complete 32-channel
groups. Model and native-kernel geometry requirements also apply. Execution
requires pipeline parallel size 1, without expert/data parallelism or ubatching.

## Compatibility

Install matching vLLM and B12X CSF support. For NVFP4, the standard model
loader delivers scale components through the model's normal weight-loading
hooks. ModelOpt selects the CSF expert method from `weight_scale_encoding`;
that method slices packed weights and stages compressed scale components without
CPU decoding or regrouping. B12X validates and gathers TP-local scale fields on
CUDA, scans exception counts, and constructs compact native-layout payloads.

Both `csf_32x2_v1` and the slab encoding `csf` are accepted. Slab checkpoints
store 16-row base/code streams and 32-bit position/value exceptions under
`nvfp4_csf_fixed` / `nvfp4_csf_exceptions` or their `mxfp4_csf_` equivalents.
Both encodings produce the same compact runtime representation. MXFP4's
separate loader preserves native dense, draft, and file-backed embedding tensors.

Preparation permutes compressed fields and exception positions for the selected
scale destination: F8_128x4, packed W4A16, or MXFP4 A8 N256/N64 tiles. Expansion
writes that destination directly. Small calls expand routed experts; large
calls expand every expert. vLLM declares expansion sweeps as dependencies of
MoE tuning, and prepared bindings retain their selected programs. CSF requests
time candidates through CUDA graph replay with live activation and route
producers, using separate selection keys from direct-launch timing. Trial
graphs and buffers are released after tuning. Ordinary
expert kernels consume native scales with no inline CSF decoding.

Runtimes without the corresponding scale decoder require uncompressed scale
tensors. Including both representations in one download removes the storage
benefit.
