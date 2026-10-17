# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x sparse MLA attention backend."""

from dataclasses import dataclass, replace
from math import gcd
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import os

import numpy as np
import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group
from vllm.distributed.device_communicators.all_reduce_utils import (
    should_nccl_symm_mem_ag_rs,
)
import os
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_sparse_mla,
    set_b12x_preparation_provider,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    _remap_tiling,
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    refresh_dcp_local_seq_lens_,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    MLAAttentionSpec,
    get_kv_quant_mode,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from b12x.attention.sparse_mla import Caps as B12xMLACaps
    from b12x.preparation import Plan as B12xPlan

    from vllm.model_executor.models.deepseek_v2 import Indexer
    from vllm.v1.attention.backend import CommonAttentionMetadata


_GLM_NEXT_MODEL_TYPES = frozenset(("glm5_next", "glm5_next_text"))
_GLM_DSA_MODEL_TYPES = frozenset(("glm_moe_dsa",))
_GLM_NEXT_CACHE_RECORD_BYTES = 528
_GLM_NEXT_NVFP4_CACHE_RECORD_BYTES = 304
_GLM_NEXT_INDEX_TAIL_BYTES_PER_TOKEN = 132 // 4
_GLM_NEXT_INDEX_PAGE_BYTES = 64 * 132
_GLM_DSA_NVFP4_CACHE_RECORD_BYTES = 368

logger = init_logger(__name__)

# KVARN_TAIL_HASH diagnostic: running checksums of the decode-tail bindings,
# written by the captured tail branch each replay and read from the per-step
# metadata build (which runs outside any captured region).
_KVTAIL_HASH_BUF: torch.Tensor | None = None


def _kvarn_dbg_save(obj, path):
    # torch.save straight to a path races when both DCP ranks save
    # simultaneously (zip-writer "unexpected pos" corruption); serialize
    # in memory and rename atomically instead.
    import io
    _buf = io.BytesIO()
    torch.save(obj, _buf)
    _tmp = f"{path}.tmp{os.getpid()}"
    with open(_tmp, "wb") as _f:
        _f.write(_buf.getvalue())
    os.replace(_tmp, path)


def _kvarn_force_sync():
    # KVARN_TAIL_FORCE_SYNC=1: hard barrier at every stage boundary to
    # test whether the tail poison is an async-ordering race (passes
    # canary => race; still fails => structural). Skipped under CUDA
    # graph capture (synchronize is illegal there; the captured decode
    # path is same-stream ordered anyway, and the failing extend path
    # is eager).
    if (
        os.environ.get("KVARN_TAIL_FORCE_SYNC") == "1"
        and not torch.cuda.is_current_stream_capturing()
    ):
        torch.cuda.synchronize()


@runtime_checkable
class B12xPhysicalSelectionProvider(Protocol):
    def get_b12x_physical_selection(
        self,
        *,
        num_tokens: int,
        num_prefills: int,
        num_decode_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None: ...


def _round_up_heads(num_heads: int) -> int:
    """B12X sparse MLA query heads: whole groups of eight."""
    return (num_heads + 7) // 8 * 8


def _packed_heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    """The first ``num_heads`` heads of ``tensor`` as a new packed tensor."""
    real = tensor.new_empty((tensor.shape[0], num_heads, *tensor.shape[2:]))
    real.copy_(tensor[:, :num_heads])
    return real


def _is_glm_next_config(hf_config: object | None) -> bool:
    return getattr(hf_config, "model_type", None) in _GLM_NEXT_MODEL_TYPES


def _is_glm_dsa_config(hf_config: object | None) -> bool:
    return getattr(hf_config, "model_type", None) in _GLM_DSA_MODEL_TYPES


def _current_hf_text_config() -> object | None:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or vllm_config.model_config is None:
        return None
    return vllm_config.model_config.hf_text_config


def _is_glm_next_spec(spec: AttentionSpec) -> bool:
    if isinstance(spec, MLAAttentionSpec) and spec.model_version == "glm5_next":
        return True
    hf_config = _current_hf_text_config()
    return hf_config is not None and _is_glm_next_config(hf_config)


def _is_glm_dsa_spec(spec: AttentionSpec) -> bool:
    if isinstance(spec, MLAAttentionSpec) and spec.model_version == "glm_moe_dsa":
        return True
    hf_config = _current_hf_text_config()
    return hf_config is not None and _is_glm_dsa_config(hf_config)


def _nvfp4_run_options(*, is_glm_next: bool) -> dict[str, bool | int]:
    if is_glm_next:
        # GLM5Next has no RoPE payload and stores one latent scale per token.
        return {
            "scale_format": 2,
            "fp8_rope": False,
            "latent_scale_per_token": True,
        }
    # GLM-5.2/5.3 DSA stores its 64-dimensional RoPE tail as E4M3.
    return {"scale_format": 2, "fp8_rope": True}


def _glm_next_recipe_error(hf_config: object) -> str | None:
    expected = {
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "index_kpool": 4,
    }
    mismatches = [
        f"{name}={getattr(hf_config, name, None)!r} (expected {value})"
        for name, value in expected.items()
        if getattr(hf_config, name, None) != value
    ]
    if mismatches:
        return "B12X GLM5Next sparse MLA requires " + ", ".join(mismatches)
    return None


def _glm_next_dcp_error(vllm_config: VllmConfig) -> str | None:
    parallel_config = vllm_config.parallel_config
    dcp_size = int(parallel_config.decode_context_parallel_size)
    if dcp_size <= 1:
        return None
    interleave = int(parallel_config.cp_kv_cache_interleave_size)
    if interleave % 4:
        return (
            "B12X GLM5Next C4 DCP requires cp_kv_cache_interleave_size divisible by 4"
        )
    return None


def _selected_index_block_stride_rows(
    kv_cache: torch.Tensor,
    *,
    block_size: int,
) -> int:
    # B12X selected indices are physical token slots. The kernel applies the
    # cache's byte page stride when it addresses the selected page.
    if int(kv_cache.shape[1]) != block_size:
        raise ValueError(
            "B12X sparse MLA cache page size does not match attention metadata: "
            f"cache={int(kv_cache.shape[1])}, metadata={block_size}"
        )
    return block_size


def _max_speculative_decode_query_len(vllm_config: VllmConfig) -> int:
    spec_config = vllm_config.speculative_config
    if spec_config is None:
        return 1
    num_speculative_tokens = int(spec_config.num_speculative_tokens or 0)
    multiplier = 2 if spec_config.parallel_drafting else 1
    return 1 + multiplier * num_speculative_tokens


def _is_speculative_decode_batch(
    common_attn_metadata: "CommonAttentionMetadata",
    max_speculative_decode_query_len: int,
) -> bool:
    is_prefilling = getattr(common_attn_metadata, "is_prefilling", None)
    return (
        max_speculative_decode_query_len > 1
        and 1 < common_attn_metadata.max_query_len <= max_speculative_decode_query_len
        and is_prefilling is not None
        and not bool(torch.any(is_prefilling[: common_attn_metadata.num_reqs]))
    )


def _is_native_ckv_source_layout(
    kv_cache: torch.Tensor,
    *,
    page_size: int,
    record_bytes: int,
) -> bool:
    return (
        kv_cache.dtype == torch.uint8
        and kv_cache.ndim == 3
        and tuple(kv_cache.shape[1:]) == (page_size, record_bytes)
        and kv_cache.stride(1) == record_bytes
        and kv_cache.stride(2) == 1
    )


def _use_b12x_full_ckv_gather(
    *,
    enabled: bool,
    is_glm_next: bool,
    dcp_world_size: int,
    max_query_len: int,
    num_tokens: int,
    num_decode_tokens: int,
    min_tokens: int,
    max_tokens: int,
    is_glm_dsa: bool = False,
    is_spec_decode: bool = False,
) -> bool:
    return (
        enabled
        and (is_glm_next or (is_glm_dsa and not is_spec_decode))
        and dcp_world_size > 1
        and max_query_len > 1
        and num_decode_tokens == 0
        and num_tokens > min_tokens
        and num_tokens <= max_tokens
    )


def _ckv_rank_token_alignment(page_size: int, dcp_world_size: int) -> int:
    """Per-rank padding that makes the gathered CKV page-addressable.

    All-gather concatenates one equally sized token span from every DCP rank.
    Individual rank boundaries do not need to coincide with KV-page boundaries;
    only the concatenated span must contain an integral number of pages.
    """
    if page_size <= 0 or dcp_world_size <= 0:
        raise ValueError("CKV page size and DCP world size must be positive")
    return page_size // gcd(page_size, dcp_world_size)


def _round_up_ckv_rank_tokens(
    token_count: int,
    *,
    page_size: int,
    dcp_world_size: int,
) -> int:
    alignment = _ckv_rank_token_alignment(page_size, dcp_world_size)
    return (token_count + alignment - 1) // alignment * alignment


def _dcp_all_gather_current_stream(
    group,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
        raise ValueError("CKV all-gather tensors must be contiguous")
    if output_tensor.numel() != input_tensor.numel() * group.world_size:
        raise ValueError("CKV all-gather tensors have incompatible sizes")

    communicator = getattr(group, "device_communicator", None)
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is not None and not getattr(pynccl_comm, "disabled", False):
        pynccl_comm.all_gather(output_tensor, input_tensor)
        return

    device_group = getattr(group, "device_group", None)
    if device_group is None:
        device_group = getattr(communicator, "device_group", None)
    if device_group is not None:
        dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=device_group,
            async_op=False,
        )
        return

    output_tensor.copy_(group.all_gather(input_tensor, dim=0))


@triton.jit
def _mask_page_table_after_nsa_len_kernel(
    page_table_ptr,
    nsa_len_ptr,
    page_stride0,
    page_stride1,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs < width
    nsa_len = tl.load(nsa_len_ptr + row)
    tl.store(
        page_table_ptr + row * page_stride0 + offs * page_stride1,
        -1,
        mask=valid & (offs >= nsa_len),
    )


def _mask_page_table_after_nsa_len(
    page_table: torch.Tensor,
    nsa_cache_seqlens: torch.Tensor,
) -> None:
    width = page_table.shape[1]
    if width == 0 or page_table.shape[0] == 0:
        return
    block_n = 128
    _mask_page_table_after_nsa_len_kernel[
        (page_table.shape[0], triton.cdiv(width, block_n))
    ](
        page_table,
        nsa_cache_seqlens,
        page_table.stride(0),
        page_table.stride(1),
        width,
        BLOCK_N=block_n,
    )


def _global_causal_lens_for_ckv_gather(
    global_seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_id_per_token: torch.Tensor,
    num_actual_tokens: int,
) -> torch.Tensor:
    """Compute each query token's causal length in the gathered global cache.

    Args:
        global_seq_lens: Global cache sequence length for each request.
        query_start_loc: Start offset of each request's query chunk.
        req_id_per_token: Request identifier for each query token.
        num_actual_tokens: Number of active query tokens.

    Returns:
        Causal cache length for each active query token.
    """
    num_reqs = global_seq_lens.shape[0]
    qsl = query_start_loc[: num_reqs + 1].to(torch.int32)
    req_ids = req_id_per_token[:num_actual_tokens].to(torch.int64)
    chunk_start = qsl[:-1][req_ids]
    chunk_len = (qsl[1:] - qsl[:-1])[req_ids]
    full_seq = global_seq_lens[req_ids].to(torch.int32)
    token_idx = torch.arange(
        num_actual_tokens,
        device=global_seq_lens.device,
        dtype=torch.int32,
    )
    return full_seq - chunk_len + (token_idx - chunk_start) + 1


@triton.jit
def _map_global_topk_to_gathered_ckv_kernel(
    req_id_ptr,
    token_indices_ptr,
    rank_req_starts_ptr,
    rank_req_lens_ptr,
    out_ptr,
    valid_count_ptr,
    starts_stride0,
    starts_stride1,
    lens_stride0,
    lens_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
    padded_rank_tokens,
    DCP_SIZE: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < NUM_TOPK_TOKENS
    req = tl.load(req_id_ptr + row)
    tok = tl.load(
        token_indices_ptr + row * ti_stride0 + cols * ti_stride1,
        mask=col_mask,
        other=-1,
    )
    owner = (tok // DCP_INTERLEAVE) % DCP_SIZE
    local_idx = (
        tok // (DCP_SIZE * DCP_INTERLEAVE)
    ) * DCP_INTERLEAVE + tok % DCP_INTERLEAVE
    valid_tok = col_mask & (tok >= 0)
    req_start = tl.load(
        rank_req_starts_ptr + owner * starts_stride0 + req * starts_stride1,
        mask=valid_tok,
        other=0,
    )
    req_len = tl.load(
        rank_req_lens_ptr + owner * lens_stride0 + req * lens_stride1,
        mask=valid_tok,
        other=0,
    )
    valid = valid_tok & (local_idx >= 0) & (local_idx < req_len)
    gathered_slot = owner * padded_rank_tokens + req_start + local_idx
    valid_i32 = valid.to(tl.int32)
    local_offset = tl.cumsum(valid_i32) - valid_i32
    tile_valid_count = tl.sum(valid_i32)
    if tl.constexpr(BLOCK_N >= NUM_TOPK_TOKENS):
        output_base = 0
        tl.store(valid_count_ptr + row, tile_valid_count)
        tl.store(
            out_ptr + row * out_stride0 + cols * out_stride1,
            -1,
            mask=col_mask & (cols >= tile_valid_count),
        )
    else:
        output_base = tl.atomic_add(valid_count_ptr + row, tile_valid_count)
    tl.store(
        out_ptr + row * out_stride0 + (output_base + local_offset) * out_stride1,
        gathered_slot,
        mask=valid,
    )


def _t_stack(rows):
    import torch as _t
    return _t.stack(rows)


def _dbg_dump_gather_selection(
    num_tokens, selected_indices, active_counts, cache_seq_lens, attn_metadata,
    raw_topk=None,
) -> None:
    """Env-gated one-shot dump of the gather-route extend selection."""
    import torch as _t

    last = num_tokens - 1
    a = active_counts[:num_tokens]
    sel_last = selected_indices[last, : int(a[last])].to(_t.int64)
    g = attn_metadata.global_cache_seq_lens_per_req
    cu = (
        attn_metadata.dcp_local_cu_seq_lens.tolist()
        if attn_metadata.dcp_local_cu_seq_lens is not None else None
    )
    print(
        "[kvarn-gather-dbg] tokens=%d act[min/med/max]=%d/%d/%d "
        "causal[last]=%d glob_seq=%s sel[last] n=%d min=%d max=%d "
        "padded=%s local_cu=%s"
        % (
            num_tokens, int(a.min()), int(a.float().median()), int(a.max()),
            int(cache_seq_lens[last]),
            g[:1].tolist() if g is not None else None,
            int(a[last]), int(sel_last.min()) if sel_last.numel() else -1,
            int(sel_last.max()) if sel_last.numel() else -1,
            getattr(attn_metadata, "dcp_padded_total_tokens", None),
            cu,
        ),
        flush=True,
    )
    print(
        "[kvarn-gather-dbg] head act[:24]=%s causal[:24]=%s causal[last:]=%s"
        % (
            a[:24].tolist(),
            cache_seq_lens[:24].tolist(),
            cache_seq_lens[max(0, num_tokens - 4): num_tokens].tolist(),
        ),
        flush=True,
    )
    if raw_topk is not None:
        raw = raw_topk[last].to(_t.int64)
        raw_valid = raw[raw >= 0]
        print(
            "[kvarn-gather-dbg] raw[last] n=%d min=%d max=%d top=%s"
            % (
                int(raw_valid.numel()),
                int(raw_valid.min()) if raw_valid.numel() else -1,
                int(raw_valid.max()) if raw_valid.numel() else -1,
                sorted(raw_valid.tolist())[-8:],
            ),
            flush=True,
        )
    big = sel_last[sel_last > 1023]
    if big.numel():
        print(
            f"[kvarn-gather-dbg] OOR slots n={int(big.numel())} "
            f"vals={sorted(big.tolist())[:12]}",
            flush=True,
        )


def _map_global_topk_to_gathered_ckv(
    req_ids: torch.Tensor,
    token_indices: torch.Tensor,
    rank_req_starts: torch.Tensor,
    rank_req_lens: torch.Tensor,
    out: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    dcp_size: int,
    cp_kv_cache_interleave_size: int,
    padded_rank_tokens: int,
) -> None:
    if token_indices.shape != out.shape:
        raise ValueError("CKV gather index output shape does not match top-k input")
    if rank_req_starts.shape != rank_req_lens.shape:
        raise ValueError("CKV gather request starts/lens shapes do not match")
    if rank_req_starts.shape[0] != dcp_size:
        raise ValueError("CKV gather request metadata does not match DCP size")
    if any(
        tensor.dtype != torch.int32
        for tensor in (
            req_ids,
            token_indices,
            rank_req_starts,
            rank_req_lens,
            out,
            valid_counts,
        )
    ):
        raise TypeError("CKV gather index metadata must be int32")

    single_tile, block_n, tiles_per_row, num_warps = _remap_tiling(
        token_indices.shape[1], 128, True
    )
    if not single_tile:
        out.fill_(-1)
        valid_counts.zero_()
    _map_global_topk_to_gathered_ckv_kernel[(token_indices.shape[0], tiles_per_row)](
        req_ids,
        token_indices,
        rank_req_starts,
        rank_req_lens,
        out,
        valid_counts,
        rank_req_starts.stride(0),
        rank_req_starts.stride(1),
        rank_req_lens.stride(0),
        rank_req_lens.stride(1),
        token_indices.stride(0),
        token_indices.stride(1),
        out.stride(0),
        out.stride(1),
        padded_rank_tokens,
        DCP_SIZE=dcp_size,
        DCP_INTERLEAVE=cp_kv_cache_interleave_size,
        NUM_TOPK_TOKENS=token_indices.shape[1],
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


class B12xMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
        "kvarn_mla_k4v4_g256",
        "kvarn_mla_k5v5_g256",
        "kvarn_mla_k5v4_g256",
        "kvarn_mla_k4v2_g256",
    ]

    @staticmethod
    def get_name() -> str:
        return "B12X"

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return B12xMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["B12xMLASparseMetadataBuilder"]:
        return B12xMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 576]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        is_glm_next = _is_glm_next_spec(spec)
        is_glm_dsa_nvfp4 = isinstance(spec, MLAAttentionSpec) and (
            spec.cache_dtype_str == "nvfp4_ds_mla" and _is_glm_dsa_spec(spec)
        )
        if not is_glm_next and not is_glm_dsa_nvfp4:
            return spec
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        if is_glm_dsa_nvfp4:
            if spec.head_size != 576:
                raise ValueError(
                    "B12X GLM DSA NVFP4 sparse MLA requires head_size=576, got "
                    f"{spec.head_size}."
                )
            return replace(
                spec,
                state_content_bytes=_GLM_DSA_NVFP4_CACHE_RECORD_BYTES,
                model_version="glm_moe_dsa",
            )
        if spec.head_size != 512:
            raise ValueError(
                "B12X GLM5Next sparse MLA requires head_size=512, got "
                f"{spec.head_size}."
            )
        record_bytes = (
            _GLM_NEXT_NVFP4_CACHE_RECORD_BYTES
            if spec.cache_dtype_str == "nvfp4_ds_mla"
            else _GLM_NEXT_CACHE_RECORD_BYTES
        )
        page_tail = _GLM_NEXT_INDEX_TAIL_BYTES_PER_TOKEN
        if get_kv_quant_mode(spec.cache_dtype_str) == KVQuantMode.KVARN_MLA:
            from vllm.v1.attention.ops.kvarn_mla import kvarn_mla_layout_from_dtype

            layout = kvarn_mla_layout_from_dtype(spec.cache_dtype_str)
            record_bytes = layout.record_bytes
            page_tail = layout.page_tail_bytes_per_token
        return replace(
            spec,
            state_content_bytes=record_bytes,
            alignment=_GLM_NEXT_INDEX_PAGE_BYTES,
            page_tail_bytes_per_token=page_tail,
            model_version="glm5_next",
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The DSA kernels consume one layer-compact, token-major cache view.
        return (KVCacheLayout.LBNHC,)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return (capability.major, capability.minor) in ((12, 0), (12, 1))

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config

        module = get_b12x_sparse_mla()
        if module is None:
            return "B12X sparse MLA requires the optional b12x package"
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_config = vllm_config.model_config.hf_text_config
            if getattr(hf_config, "index_topk", None) is None:
                return "B12X sparse MLA requires a model with index_topk"
            if _is_glm_next_config(hf_config):
                if recipe_error := _glm_next_recipe_error(hf_config):
                    return recipe_error
                if head_size != 512:
                    return "B12X GLM5Next sparse MLA requires head_size=512"
                if dcp_error := _glm_next_dcp_error(vllm_config):
                    return dcp_error
                return None
            if kv_cache_dtype == "nvfp4_ds_mla" and not _is_glm_dsa_config(hf_config):
                return (
                    "B12X nvfp4_ds_mla requires GLM5Next or the "
                    "GLM-5.2/5.3 DSA architecture"
                )
            if head_size != 576:
                return "B12X sparse MLA requires head_size=576"
            if int(getattr(hf_config, "kv_lora_rank", 0)) != 512:
                return "B12X sparse MLA requires kv_lora_rank=512"
            if int(getattr(hf_config, "qk_rope_head_dim", 0)) != 64:
                return "B12X sparse MLA requires qk_rope_head_dim=64"
        return None


class B12xGLMDSAMLASparseBackend(B12xMLASparseBackend):
    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM DSA sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        return super().customize_spec(replace(spec, model_version="glm_moe_dsa"))


class B12xGLM5NextMLASparseBackend(B12xMLASparseBackend):
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # GLM-Next's pooled index tail shares manager blocks with its MLA
        # latent page. Keep the layer dimension inside the manager block so
        # copies and swaps carry both cache regions together.
        return (KVCacheLayout.BLHNC,)

    @staticmethod
    def get_builder_cls() -> type["B12xGLM5NextMLASparseMetadataBuilder"]:
        return B12xGLM5NextMLASparseMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM5Next sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        return super().customize_spec(replace(spec, model_version="glm5_next"))

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Keep the hybrid manager page intact so its FP8 pooled-index tail is
        # copied and recycled with the corresponding MLA page.
        return [MultipleOf(64)]


@dataclass
class B12xMLASparseMetadata(SparseMLACommonMetadata):
    dcp_global_seq_lens: torch.Tensor | None = None
    kvarn_tail_start: torch.Tensor | None = None
    kvarn_tail_demote_ranges: torch.Tensor | None = None
    global_positions: torch.Tensor | None = None
    prefill_query_lens_cpu: torch.Tensor | None = None
    prefill_seq_lens_cpu: torch.Tensor | None = None
    cache_seq_lens_per_token: torch.Tensor | None = None
    selector_state_slot_ids: torch.Tensor | None = None
    selector_state_is_fresh: torch.Tensor | None = None
    selector_num_accepted_tokens: torch.Tensor | None = None
    selector_is_prefilling: torch.Tensor | None = None
    is_spec_decode: bool = False
    ckv_selected_indices: torch.Tensor | None = None
    ckv_active_counts: torch.Tensor | None = None
    dcp_rank_req_starts: torch.Tensor | None = None
    dcp_rank_req_lens: torch.Tensor | None = None
    dcp_local_cu_seq_lens: torch.Tensor | None = None
    global_cache_seq_lens_per_req: torch.Tensor | None = None
    dcp_local_total_tokens: int = 0
    dcp_padded_total_tokens: int = 0
    dcp_ckv_gather_eligible: bool = False


class B12xMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[B12xMLASparseMetadata]
):
    metadata_cls = B12xMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    requires_glm_next_selector_metadata: bool
    _is_glm_dsa: bool = False

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        self.requires_glm_next_selector_metadata = _is_glm_next_config(hf_config)
        self._is_glm_dsa = _is_glm_dsa_config(hf_config)
        if self.requires_glm_next_selector_metadata and (
            dcp_error := _glm_next_dcp_error(vllm_config)
        ):
            raise ValueError(dcp_error)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # All step-dependent state is persistent. Generic DSA DCP additionally
        # refreshes its rank-local sequence lengths in place between steps.
        self.supports_draft_decode_metadata_update = True
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        scheduler_config = vllm_config.scheduler_config
        max_tokens = scheduler_config.max_num_batched_tokens
        max_reqs = int(scheduler_config.max_num_seqs)
        # Async scheduling keeps the host ahead of the device; the tail's
        # captured decode graphs consume per-step staging that the next
        # build() rewrite would race. A host-side drain at build() entry
        # restores correctness (bisect: syncs inside build() fix async+tail,
        # a post-forward sync does not). Only tail + async configs pay it.
        self._kvarn_async_build_fence = bool(
            getattr(vllm_config.cache_config, "kvarn_tail_tokens", 0) > 0
            and scheduler_config.async_scheduling
        )
        self._ckv_max_reqs = max_reqs
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        if self.requires_glm_next_selector_metadata:
            self._capture_default_state_slot_ids = torch.arange(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_state_slot_ids = torch.empty(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_state_is_fresh = torch.ones(
                max_reqs, dtype=torch.bool, device=device
            )
            self._capture_num_accepted_tokens = torch.ones(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_is_prefilling = torch.zeros(
                max_reqs, dtype=torch.bool, device=device
            )
        self._ckv_gather_requested = (
            (self.requires_glm_next_selector_metadata or self._is_glm_dsa)
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
        )
        if self._ckv_gather_requested:
            # GLM DSA selects index_topk tokens; GLM5Next appends its
            # incomplete-pool tail to the selection.
            ckv_topk_tokens = int(hf_config.index_topk)
            if self.requires_glm_next_selector_metadata:
                ckv_topk_tokens += int(hf_config.index_kpool) - 1
            self.ckv_selected_indices_buffer = torch.empty(
                (max_tokens, ckv_topk_tokens), dtype=torch.int32, device=device
            )
            self.ckv_active_counts_buffer = torch.empty(
                (max_tokens,), dtype=torch.int32, device=device
            )
            self.dcp_rank_req_lens_buffer = torch.empty(
                (self.dcp_world_size, max_reqs), dtype=torch.int32, device=device
            )
            self.dcp_rank_req_starts_buffer = torch.empty(
                (self.dcp_world_size, max_reqs), dtype=torch.int32, device=device
            )
            self.dcp_local_cu_seq_lens_buffer = torch.empty(
                (max_reqs + 1,), dtype=torch.int32, device=device
            )
        else:
            self.ckv_selected_indices_buffer = None
            self.ckv_active_counts_buffer = None
            self.dcp_rank_req_lens_buffer = None
            self.dcp_rank_req_starts_buffer = None
            self.dcp_local_cu_seq_lens_buffer = None
        num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
            num_q_heads, 1024
        )
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )
        self._max_speculative_decode_query_len = _max_speculative_decode_query_len(
            vllm_config
        )

    def _stage_glm_next_selector_metadata(
        self,
        *,
        num_reqs: int,
        for_cudagraph_capture: bool,
        selector_state_slot_ids: torch.Tensor | None,
        selector_state_is_fresh: torch.Tensor | None,
        selector_num_accepted_tokens: torch.Tensor | None,
        selector_is_prefilling: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        values = (
            selector_state_slot_ids,
            selector_state_is_fresh,
            selector_num_accepted_tokens,
            selector_is_prefilling,
        )
        if not self.requires_glm_next_selector_metadata:
            if any(value is not None for value in values):
                raise TypeError(
                    "GLM5Next selector metadata was provided to a non-GLM "
                    "B12X sparse MLA builder"
                )
            return (None, None, None, None)

        capacity = int(self._capture_state_slot_ids.numel())
        if not 0 <= num_reqs <= capacity:
            raise ValueError(
                "GLM5Next selector request count exceeds the metadata buffer "
                f"capacity: num_reqs={num_reqs}, capacity={capacity}"
            )
        if not for_cudagraph_capture and any(value is None for value in values):
            raise RuntimeError(
                "B12X GLM5Next sparse MLA requires selector state slots, fresh "
                "flags, accepted-token counts, and prefill flags"
            )

        if for_cudagraph_capture:
            self._capture_state_slot_ids[:num_reqs].copy_(
                self._capture_default_state_slot_ids[:num_reqs]
            )
            self._capture_state_is_fresh[:num_reqs].fill_(True)
            self._capture_num_accepted_tokens[:num_reqs].fill_(1)
            self._capture_is_prefilling[:num_reqs].fill_(False)
        else:
            typed_values = tuple(value for value in values if value is not None)
            if any(
                value.ndim != 1 or value.numel() < num_reqs for value in typed_values
            ):
                raise ValueError(
                    "GLM5Next selector metadata must be one-dimensional and "
                    "cover every padded request row"
                )
            self._capture_state_slot_ids[:num_reqs].fill_(-1)
            self._capture_state_is_fresh[:num_reqs].fill_(True)
            self._capture_num_accepted_tokens[:num_reqs].fill_(1)
            self._capture_is_prefilling[:num_reqs].fill_(False)
            assert selector_state_slot_ids is not None
            assert selector_state_is_fresh is not None
            assert selector_num_accepted_tokens is not None
            assert selector_is_prefilling is not None
            self._capture_state_slot_ids[:num_reqs].copy_(
                selector_state_slot_ids[:num_reqs]
            )
            self._capture_state_is_fresh[:num_reqs].copy_(
                selector_state_is_fresh[:num_reqs]
            )
            self._capture_num_accepted_tokens[:num_reqs].copy_(
                selector_num_accepted_tokens[:num_reqs]
            )
            self._capture_is_prefilling[:num_reqs].copy_(
                selector_is_prefilling[:num_reqs]
            )

        return (
            self._capture_state_slot_ids[:num_reqs],
            self._capture_state_is_fresh[:num_reqs],
            self._capture_num_accepted_tokens[:num_reqs],
            self._capture_is_prefilling[:num_reqs],
        )

    def _build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
        *,
        for_cudagraph_capture: bool,
        selector_state_slot_ids: torch.Tensor | None = None,
        selector_state_is_fresh: torch.Tensor | None = None,
        selector_num_accepted_tokens: torch.Tensor | None = None,
        selector_is_prefilling: torch.Tensor | None = None,
    ) -> B12xMLASparseMetadata:
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        common = common_attn_metadata
        num_tokens = common.num_actual_tokens
        use_dcp = self.dcp_world_size > 1
        seq_lens = (
            common.dcp_local_seq_lens
            if use_dcp and common.dcp_local_seq_lens is not None
            else common.seq_lens
        )
        metadata.seq_lens = seq_lens
        metadata.dcp_global_seq_lens = (
            common.seq_lens[: common.num_reqs] if use_dcp else None
        )

        if self.supports_varlen_decode_cudagraph:
            per_token_lens = self.cache_seq_lens_per_token_buffer[:num_tokens]
        elif common.max_query_len <= 1 and num_tokens == common.num_reqs:
            per_token_lens = seq_lens[:num_tokens]
        elif common.positions is not None and not common.is_prefilling:
            # Decode rows (incl. DCP): derive the per-token cache length as
            # local position + 1 entirely on-device. Stream-ordered, so it is
            # safe under async scheduling, and this buffer's address is what
            # a captured FULL graph replays against. The CPU-staged fallback
            # below rewrites its pinned source on every build, which races a
            # still-pending replay under async scheduling -- fatal for the
            # kvarn tail, whose dense window consumes every length.
            per_token_lens = self.cache_seq_lens_per_token_buffer[:num_tokens]
            per_token_lens.copy_(common.positions[:num_tokens])
            per_token_lens += 1
        else:
            starts = np.asarray(common.query_start_loc_cpu, dtype=np.int32)
            query_lens = np.diff(starts)
            seq_lens_cpu_source = common.seq_lens_cpu_upper_bound
            if seq_lens_cpu_source is None:
                raise ValueError("B12X sparse MLA requires host sequence length bounds")
            seq_lens_cpu = seq_lens_cpu_source.numpy().astype(np.int32, copy=False)
            host_lens = np.zeros((num_tokens,), dtype=np.int32)
            for req_id, query_len in enumerate(query_lens):
                if query_len <= 0:
                    continue
                start = int(starts[req_id])
                end = int(starts[req_id + 1])
                context_len = int(seq_lens_cpu[req_id]) - int(query_len)
                request_lens = torch.arange(
                    context_len + 1,
                    context_len + int(query_len) + 1,
                    dtype=torch.int32,
                )
                if use_dcp:
                    request_lens = get_dcp_local_seq_lens(
                        request_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    )
                host_lens[start:end] = request_lens.numpy()
            host_tensor = torch.from_numpy(host_lens).pin_memory()
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                host_tensor, non_blocking=True
            )
            per_token_lens = self.cache_seq_lens_per_token_buffer[:num_tokens]

        metadata.cache_seq_lens_per_token = per_token_lens
        metadata.kvarn_tail_start = getattr(common, "kvarn_tail_start", None)
        metadata.global_positions = (
            common.positions[:num_tokens]
            if common.positions is not None
            else None
        )
        if os.environ.get("KVARN_TAIL_DEBUG"):
            logger.warning(
                "KVARN_TAIL_DEBUG bld: common_type=%s tail=%s",
                type(common).__name__,
                None
                if metadata.kvarn_tail_start is None
                else metadata.kvarn_tail_start.shape,
            )
        metadata.kvarn_tail_demote_ranges = getattr(
            common, "kvarn_tail_demote_ranges", None
        )
        metadata.is_spec_decode = _is_speculative_decode_batch(
            common,
            self._max_speculative_decode_query_len,
        )
        if metadata.num_prefills:
            prefill_start = metadata.num_decodes
            prefill_end = prefill_start + metadata.num_prefills + 1
            metadata.prefill_query_lens_cpu = torch.diff(
                common.query_start_loc_cpu[prefill_start:prefill_end]
            )
            seq_lens_cpu_source = common.seq_lens_cpu_upper_bound
            if seq_lens_cpu_source is None:
                raise ValueError("B12X sparse MLA requires host sequence length bounds")
            metadata.prefill_seq_lens_cpu = seq_lens_cpu_source[
                prefill_start : prefill_start + metadata.num_prefills
            ].clone()
        _gather_gate = (
            get_kv_quant_mode(self.kv_cache_spec.cache_dtype_str)
            == KVQuantMode.KVARN_MLA
        ) and _use_b12x_full_ckv_gather(
            enabled=self._ckv_gather_requested,
            is_glm_next=self.requires_glm_next_selector_metadata,
            # GLM DSA gathers only in eager prefill: captured graphs and
            # speculative verification keep the query exchange.
            is_glm_dsa=self._is_glm_dsa and not for_cudagraph_capture,
            is_spec_decode=metadata.is_spec_decode,
            dcp_world_size=self.dcp_world_size,
            max_query_len=common.max_query_len,
            num_tokens=num_tokens,
            num_decode_tokens=metadata.num_decode_tokens,
            min_tokens=envs.VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS,
            max_tokens=envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS,
        )
        if os.environ.get("KVARN_GATHER_DEBUG") == "1":
            print(
                "[kvarn-gather-gate] gate=%s tokens=%d decode=%d spec=%s "
                "mql=%d enabled=%s glmnext=%s spec_dtype=%s" % (
                    _gather_gate, num_tokens, metadata.num_decode_tokens,
                    metadata.is_spec_decode, common.max_query_len,
                    self._ckv_gather_requested,
                    self.requires_glm_next_selector_metadata,
                    getattr(self.kv_cache_spec, "cache_dtype_str", None),
                ),
                flush=True,
            )
        if _gather_gate:
            assert self.ckv_selected_indices_buffer is not None
            assert self.ckv_active_counts_buffer is not None
            assert self.dcp_rank_req_lens_buffer is not None
            assert self.dcp_rank_req_starts_buffer is not None
            assert self.dcp_local_cu_seq_lens_buffer is not None
            global_seq_lens = common.seq_lens[: common.num_reqs]
            all_rank_lens = get_dcp_local_seq_lens(
                global_seq_lens,
                self.dcp_world_size,
                dcp_rank=None,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            ).transpose(0, 1)
            rank_req_lens = self.dcp_rank_req_lens_buffer[
                : self.dcp_world_size, : common.num_reqs
            ]
            rank_req_lens.copy_(all_rank_lens)
            rank_req_starts = self.dcp_rank_req_starts_buffer[
                : self.dcp_world_size, : common.num_reqs
            ]
            rank_req_starts[:, 0].zero_()
            if common.num_reqs > 1:
                torch.cumsum(rank_req_lens[:, :-1], dim=1, out=rank_req_starts[:, 1:])
            local_cu_seq_lens = self.dcp_local_cu_seq_lens_buffer[: common.num_reqs + 1]
            local_cu_seq_lens[0].zero_()
            torch.cumsum(
                rank_req_lens[self.dcp_rank],
                dim=0,
                out=local_cu_seq_lens[1:],
            )
            rank_totals = rank_req_lens.sum(dim=1).tolist()
            local_total_tokens = int(rank_totals[self.dcp_rank])
            page_size = int(self.kv_cache_spec.block_size)
            padded_total_tokens = _round_up_ckv_rank_tokens(
                max(int(total) for total in rank_totals),
                page_size=page_size,
                dcp_world_size=self.dcp_world_size,
            )
            max_local_capacity = _round_up_ckv_rank_tokens(
                (envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS + self.dcp_world_size - 1)
                // self.dcp_world_size
                + self._ckv_max_reqs * self.cp_kv_cache_interleave_size,
                page_size=page_size,
                dcp_world_size=self.dcp_world_size,
            )
            if 0 < padded_total_tokens <= max_local_capacity:
                metadata.ckv_selected_indices = self.ckv_selected_indices_buffer[
                    :num_tokens
                ]
                metadata.ckv_active_counts = self.ckv_active_counts_buffer[:num_tokens]
                metadata.dcp_rank_req_lens = rank_req_lens
                metadata.dcp_rank_req_starts = rank_req_starts
                metadata.dcp_local_cu_seq_lens = local_cu_seq_lens
                metadata.global_cache_seq_lens_per_req = global_seq_lens
                metadata.dcp_local_total_tokens = local_total_tokens
                metadata.dcp_padded_total_tokens = padded_total_tokens
                metadata.dcp_ckv_gather_eligible = True
        (
            metadata.selector_state_slot_ids,
            metadata.selector_state_is_fresh,
            metadata.selector_num_accepted_tokens,
            metadata.selector_is_prefilling,
        ) = self._stage_glm_next_selector_metadata(
            num_reqs=common.num_reqs,
            for_cudagraph_capture=for_cudagraph_capture,
            selector_state_slot_ids=selector_state_slot_ids,
            selector_state_is_fresh=selector_state_is_fresh,
            selector_num_accepted_tokens=selector_num_accepted_tokens,
            selector_is_prefilling=selector_is_prefilling,
        )
        return metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
        selector_state_slot_ids: torch.Tensor | None = None,
        selector_state_is_fresh: torch.Tensor | None = None,
        selector_num_accepted_tokens: torch.Tensor | None = None,
        selector_is_prefilling: torch.Tensor | None = None,
    ) -> B12xMLASparseMetadata:
        if (
            os.environ.get("KVARN_TAIL_HASH") == "1"
            and common_attn_metadata.num_actual_tokens <= 32
        ):
            _hb = globals().get("_KVTAIL_HASH_BUF")
            if _hb is not None:
                try:
                    with open(
                        "/home/js/kvarn-probes/kvtailbuild-r0.log", "a"
                    ) as _f:
                        _f.write(
                            "replay=%d idx_sum=%d len_sum=%d csl_sum=%d "
                            "build_csl_sum=%d nt=%d\n"
                            % (
                                int(_hb[3].item()),
                                int(_hb[0].item()),
                                int(_hb[1].item()),
                                int(_hb[2].item()),
                                int(common_attn_metadata.query_start_loc[-1].item()),
                                int(common_attn_metadata.query_start_loc.numel() - 1),
                            )
                        )
                except OSError:
                    pass
        if (
            getattr(self, "_kvarn_async_build_fence", False)
            and not torch.cuda.is_current_stream_capturing()
        ):
            # Tail + async: the host is ahead of the device here. Drain the
            # stream BEFORE staging so this build's staging writes cannot
            # race the previous step's still-pending captured work.
            _n = getattr(self, "_kvarn_fence_fires", 0) + 1
            self._kvarn_fence_fires = _n
            if _n <= 3:
                logger.warning("KVARN build fence fire #%d", _n)
            torch.cuda.synchronize()
        result = self._build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
            for_cudagraph_capture=False,
            selector_state_slot_ids=selector_state_slot_ids,
            selector_state_is_fresh=selector_state_is_fresh,
            selector_num_accepted_tokens=selector_num_accepted_tokens,
            selector_is_prefilling=selector_is_prefilling,
        )
        return result

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> B12xMLASparseMetadata:
        return self._build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            for_cudagraph_capture=True,
        )

    def update_draft_decode_metadata(
        self,
        metadata: B12xMLASparseMetadata,
    ) -> None:
        if self.dcp_world_size > 1:
            global_seq_lens = metadata.dcp_global_seq_lens
            if global_seq_lens is None:
                raise RuntimeError(
                    "B12X fused DCP draft decode requires global sequence lengths"
                )
            refresh_dcp_local_seq_lens_(
                metadata.seq_lens,
                global_seq_lens,
                metadata.num_reqs,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )

        if self.requires_glm_next_selector_metadata:
            accepted = metadata.selector_num_accepted_tokens
            if accepted is None:
                raise RuntimeError(
                    "GLM5Next draft decode metadata requires accepted-token counts"
                )
            accepted.fill_(1)


@triton.jit
def _glm_device_token_metadata_kernel(
    query_start_loc,
    seq_lens,
    request_ids,
    causal_lens,
    NUM_REQS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), NUM_REQS, tl.int32)
    for _ in range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + tl.minimum(mid + 1, NUM_REQS))
        advance = (lo < hi) & (end <= token)
        hi = tl.where((lo < hi) & ~advance, mid, hi)
        lo = tl.where(advance, mid + 1, lo)
    valid = (token < NUM_TOKENS) & (lo < NUM_REQS)
    start = tl.load(query_start_loc + lo, mask=valid, other=0)
    end = tl.load(query_start_loc + lo + 1, mask=valid, other=0)
    length = tl.load(seq_lens + lo, mask=valid, other=0)
    length = tl.maximum(length - (end - start) + token - start + 1, 0)
    if DCP_SIZE > 1:
        base = length // (INTERLEAVE * DCP_SIZE) * INTERLEAVE
        length = base + tl.minimum(
            tl.maximum(length - base * DCP_SIZE - DCP_RANK * INTERLEAVE, 0),
            INTERLEAVE,
        )
    tl.store(request_ids + token, tl.where(valid, lo, 0), token < NUM_TOKENS)
    tl.store(causal_lens + token, tl.where(valid, length, 0), token < NUM_TOKENS)


class B12xGLM5NextMLASparseMetadataBuilder(B12xMLASparseMetadataBuilder):
    # The pooled selector must commit every fresh or extended prompt row
    # through run_prefill; decode commits only accepted prior rows.
    treat_short_extends_as_decodes: ClassVar[bool] = False
    supports_varlen_decode_cudagraph: ClassVar[bool] = True

    def _build_req_id_per_token(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> torch.Tensor:
        common = common_attn_metadata
        num_tokens = common.num_actual_tokens
        # Adaptive verification redistributes only the decode prefix. Its total
        # length and the CPU prefill boundaries remain exact.
        if num_tokens:
            _glm_device_token_metadata_kernel[(triton.cdiv(num_tokens, 128),)](
                common.query_start_loc,
                common.seq_lens,
                self.req_id_per_token_buffer,
                self.cache_seq_lens_per_token_buffer,
                NUM_REQS=common.num_reqs,
                NUM_TOKENS=num_tokens,
                SEARCH_STEPS=common.num_reqs.bit_length(),
                DCP_SIZE=self.dcp_world_size,
                DCP_RANK=self.dcp_rank,
                INTERLEAVE=self.cp_kv_cache_interleave_size,
                BLOCK=128,
            )
        return self.req_id_per_token_buffer[:num_tokens]


class B12xMLASparseImpl(SparseMLACommonImpl[B12xMLASparseMetadata]):
    can_return_lse_for_decode = True
    lse_base_on_e = True
    supports_dense_mha_prefill = False
    supports_pcp = False
    # B12X maps selections into caller-owned scratch or uses the indexer's
    # prepared physical selection. Generic group buffers are never consumed.
    uses_index_group = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "B12X sparse MLA does not support ALiBi, sliding window, or "
                "logit soft caps."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X sparse MLA supports decoder self-attention only."
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        hf_config = vllm_config.model_config.hf_text_config
        self._is_glm_next = _is_glm_next_config(hf_config)
        self._is_glm_dsa = _is_glm_dsa_config(hf_config)
        self._physical_selection_provider = (
            indexer if isinstance(indexer, B12xPhysicalSelectionProvider) else None
        )
        self.supports_mtp_with_cp_non_trivial_interleave_size = self._is_glm_next
        if self._is_glm_next:
            if recipe_error := _glm_next_recipe_error(hf_config):
                raise ValueError(recipe_error)
            if dcp_error := _glm_next_dcp_error(vllm_config):
                raise ValueError(dcp_error)
            if head_size != 512:
                raise ValueError("B12X GLM5Next sparse MLA requires head_size=512.")
        else:
            if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
                raise ValueError(
                    "B12X sparse MLA requires kv_lora_rank=512 and qk_rope_head_dim=64."
                )
            if head_size != 576:
                raise ValueError("B12X sparse MLA requires head_size=576.")
        if self.topk_indices_buffer is None:
            raise ValueError("B12X sparse MLA requires a top-k index buffer.")
        uses_nvfp4_cache = kv_cache_dtype == "nvfp4_ds_mla"
        self._uses_kvarn_cache = (
            get_kv_quant_mode(kv_cache_dtype) == KVQuantMode.KVARN_MLA
        )
        self._kvarn_layout = None
        if self._uses_kvarn_cache:
            from vllm.v1.attention.ops.kvarn_mla import kvarn_mla_layout_from_dtype

            self._kvarn_layout = kvarn_mla_layout_from_dtype(kv_cache_dtype)
        self._uses_glm_dsa_nvfp4_cache = self._is_glm_dsa and uses_nvfp4_cache
        if uses_nvfp4_cache and not (
            self._is_glm_next or self._uses_glm_dsa_nvfp4_cache
        ):
            raise ValueError(
                "B12X nvfp4_ds_mla requires GLM5Next or the "
                "GLM-5.2/5.3 DSA architecture."
            )
        if not self._uses_kvarn_cache and kv_cache_dtype not in (
            "fp8_ds_mla",
            "nvfp4_ds_mla",
        ):
            raise ValueError(
                "B12X sparse MLA requires a packed fp8_ds_mla, nvfp4_ds_mla "
                f"or kvarn_mla_* KV cache; got kv_cache_dtype={kv_cache_dtype!r}."
            )
        self._uses_nvfp4_cache = uses_nvfp4_cache

        module = get_b12x_sparse_mla()
        if module is None:
            raise RuntimeError("B12X sparse MLA requires `pip install vllm[b12x]`.")
        if not module.is_supported():
            raise RuntimeError("B12X sparse MLA is not supported on this device.")
        required_symbols = ["Caps", "bind", "plan", "run"]
        if self._uses_kvarn_cache:
            required_symbols.append("concat_and_cache_glm_next_mla_kvarn")
        elif self._is_glm_next:
            required_symbols.append(
                "concat_and_cache_glm_next_mla_nvfp4"
                if uses_nvfp4_cache
                else "concat_and_cache_glm_next_mla_fp8"
            )
        if self._uses_glm_dsa_nvfp4_cache:
            required_symbols.append("concat_and_cache_nvfp4_mla_fp8_rope")
        for name in required_symbols:
            getattr(module, name)
        self._bind = module.bind
        self._run = module.run
        self._model_type: int | None = None
        self._concat_and_cache_nvfp4_mla_fp8_rope = None
        self._concat_and_cache_glm_next_mla = None
        if self._uses_kvarn_cache:
            self._model_type = int(module.ModelType.KVARN_MLA)
            self._concat_and_cache_glm_next_mla_kvarn = (
                module.concat_and_cache_glm_next_mla_kvarn
            )
            # Precision tail (design doc §10): newest N tokens hold fp8
            # records inside the kvarn pool; 0 disables the feature and
            # reproduces the stock per-token kvarn store byte-for-byte.
            self._kvarn_tail_local_tokens = 0
            self._kvarn_tail_tokens = 0
            self._cache_writer_tail_plan = None
            self._concat_and_cache_glm_next_mla_fp8_tail = None
            tail_tokens = int(
                getattr(vllm_config.cache_config, "kvarn_tail_tokens", 0) or 0
            )
            if tail_tokens:
                dcp_size = max(
                    1, int(vllm_config.parallel_config.decode_context_parallel_size)
                )
                if tail_tokens % 256:
                    raise ValueError(
                        "--kvarn-tail-tokens must be a multiple of the 256-token "
                        f"tile group; got {tail_tokens}"
                    )
                if tail_tokens % dcp_size:
                    raise ValueError(
                        f"--kvarn-tail-tokens ({tail_tokens}) must be divisible "
                        f"by the DCP world size ({dcp_size})"
                    )
                if dcp_size == 1:
                    # The b12x fp8 arm is decode-only (commit 63a86aed); the
                    # MG prefill kernel has no KVARN_TAIL_FP8 shape. DCP2
                    # extends read the tail through the ckv-gather +
                    # gathered-demote + MG-kvarn route; DCP1 extends have no
                    # sound read path, so refuse instead of serving garbage.
                    raise ValueError(
                        "--kvarn-tail-tokens currently requires "
                        "--decode-context-parallel-size 2 (the fp8 tail arm "
                        "is decode-only; DCP1 extends have no fp8 read path)"
                    )
                self._kvarn_tail_local_tokens = tail_tokens // dcp_size
                self._kvarn_tail_tokens = tail_tokens
                logger.info(
                    "kvarn precision tail: %d tokens (%d per DCP rank), "
                    "record_bytes=%d",
                    tail_tokens,
                    self._kvarn_tail_local_tokens,
                    self._kvarn_layout.record_bytes,
                )
                self._concat_and_cache_glm_next_mla_fp8_tail = (
                    module.concat_and_cache_glm_next_mla_fp8_tail
                )
        elif self._is_glm_next:
            self._model_type = int(module.ModelType.GLM_NEXT)
            self._concat_and_cache_glm_next_mla = (
                module.concat_and_cache_glm_next_mla_nvfp4
                if self._uses_nvfp4_cache
                else module.concat_and_cache_glm_next_mla_fp8
            )
        elif self._uses_glm_dsa_nvfp4_cache:
            self._model_type = int(module.ModelType.GLM_NSA)
            self._concat_and_cache_nvfp4_mla_fp8_rope = (
                module.concat_and_cache_nvfp4_mla_fp8_rope
            )

        scheduler_config = vllm_config.scheduler_config
        max_tokens = int(scheduler_config.max_num_batched_tokens)
        max_seqs = int(scheduler_config.max_num_seqs)
        self._input_num_heads = self.num_heads * self.dcp_world_size
        # B12X sparse MLA runs whole groups of eight heads. TP-padded models
        # (GLM-5.3 at TP6: 11 heads per rank) get zero heads up to the next
        # group, and their outputs are dropped again.
        self._kernel_num_heads = _round_up_heads(self._input_num_heads)
        self._q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self._topk_tokens = int(self.topk_indices_buffer.shape[-1])
        if self._is_glm_next:
            expected_width = int(hf_config.index_topk) + int(hf_config.index_kpool) - 1
            if self._topk_tokens != expected_width:
                raise ValueError(
                    "B12X GLM5Next sparse MLA requires a selector output width "
                    f"of {expected_width}, got {self._topk_tokens}."
                )
        self._max_tokens = max_tokens
        self._max_seqs = max_seqs
        self._max_speculative_decode_query_len = _max_speculative_decode_query_len(
            vllm_config
        )
        self._decode_max_rows = min(
            max_tokens,
            max_seqs * self._max_speculative_decode_query_len,
        )
        if self._uses_kvarn_cache and self._kvarn_tail_local_tokens > 0:
            # The decode-tail launch consumes these under captured decode
            # graphs, so they must be persistent buffers (same contract as
            # ckv_selected_indices_buffer): the per-step computed temporaries
            # are copied in before every launch, and the bind always sees a
            # capture-stable address.
            _tail_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            self._kvarn_tail_indices_buffer = torch.empty(
                (self._decode_max_rows, self._kvarn_tail_local_tokens),
                dtype=torch.int32,
                device=_tail_dev,
            )
            self._kvarn_tail_lengths_buffer = torch.empty(
                (self._decode_max_rows,),
                dtype=torch.int32,
                device=_tail_dev,
            )
        self._kv_dtype = torch.uint8
        kernel_page_size = (
            int(vllm_config.cache_config.block_size) if self._is_glm_next else 64
        )
        self._ckv_gather_enabled = (
            (self._is_glm_next or self._is_glm_dsa)
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
        )
        max_ckv_tokens = envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS
        cp_kv_cache_interleave_size = int(
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        self._cp_kv_cache_interleave_size = cp_kv_cache_interleave_size
        self._ckv_capacity_tokens = (
            max_ckv_tokens + self.dcp_world_size - 1
        ) // self.dcp_world_size + max_seqs * cp_kv_cache_interleave_size
        self._ckv_local_capacity = 0
        self._module = module
        self._kernel_page_size = 0
        self._kernel_page_size_finalized = not self._is_glm_next
        self._plans: dict[tuple[str, int], B12xPlan] = {}
        self._plan_caps: dict[tuple[str, int], B12xMLACaps] = {}
        self._cache_writer_plan: object | None = None
        self._bound_kv_cache: torch.Tensor | None = None
        self._set_kernel_page_size(kernel_page_size)
        self.supports_quant_query_input = False
        if not getattr(self, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(self, self)

    def _set_kernel_page_size(self, kernel_page_size: int) -> None:
        if kernel_page_size <= 0 or kernel_page_size % 64:
            raise ValueError(
                "B12X sparse MLA kernel page size must be a positive multiple "
                f"of 64, got {kernel_page_size}."
            )
        if kernel_page_size == self._kernel_page_size:
            return
        # Plan declarations read the page size.
        self._kernel_page_size = kernel_page_size
        keys = [("decode", self._decode_max_rows), ("extend", self._max_tokens)]
        if self._uses_kvarn_cache and self._kvarn_tail_local_tokens > 0:
            keys.append(("extend_tail", self._max_tokens))
            keys.append(("extend_tail_gather", self._max_tokens))
        if self._ckv_gather_enabled:
            keys.append(("ckv_extend", self._max_tokens))
        self._plans = {}
        self._plan_caps = {}
        for key in keys:
            self._plan_caps[key], self._plans[key] = self._declare_plan(*key)
        self._cache_record_bytes = int(self._plan_caps[keys[0]].cache_record_bytes)
        # This family's scratch layout is a deterministic function of Caps
        # alone (see plan_sparse_mla_scratch) and grows with the row capacity,
        # so the largest plan declared here also covers every decode plan
        # declared later for fewer rows.
        self._scratch_nbytes = max(
            int(plan.scratch_specs()[0].nbytes) for plan in self._plans.values()
        )
        if self._is_glm_next and self.dcp_world_size > 1:
            # Query gathering finishes before attention consumes scratch. Its
            # rank-major receive storage can therefore reuse the kernel arena.
            self._scratch_nbytes = max(
                self._scratch_nbytes,
                self._max_tokens
                * self._input_num_heads
                * self._q_head_dim
                * torch.bfloat16.itemsize,
            )
        self._ckv_local_capacity = _round_up_ckv_rank_tokens(
            self._ckv_capacity_tokens,
            page_size=kernel_page_size,
            dcp_world_size=self.dcp_world_size,
        )

    def _declare_plan(self, mode: str, rows: int):
        """Declare the plan for mode at rows query rows; returns (caps, plan).

        The decode launch is exact in its row count, so decode plans are
        declared per row count. The extend modes accept any row count up to
        their declared capacity. ``decode_tail`` declares the precision-tail
        segment: the KVARN_TAIL_FP8 arm (GLM fp8 math over 536B pool rows)
        attending the dense tail window.
        """
        tail_mode = mode in ("decode_tail", "extend_tail", "extend_tail_gather")
        caps_kwargs = dict(
            device=torch.device("cuda", torch.accelerator.current_device_index()),
            num_q_heads=(
                _round_up_heads(self.num_heads)
                if mode == "ckv_extend"
                else self._kernel_num_heads
            ),
            max_q_rows=rows,
            max_width=self._topk_tokens,
            softmax_scale=self.scale,
            dtype=torch.bfloat16,
            kv_dtype=self._kv_dtype,
            head_dim=self._q_head_dim,
            v_head_dim=self.kv_lora_rank,
            mode="extend" if mode == "ckv_extend" else (
                # decode_tail launches the plain decode kernel; the tail is
                # identified by model_type (KVARN_TAIL_FP8), not by mode.
                "decode" if mode == "decode_tail" else mode
            ),
            max_batch=rows,
            max_chunks_per_row=max(1, (self._topk_tokens + 63) // 64),
            page_size=self._kernel_page_size,
            return_lse=self.need_to_return_lse_for_decode,
            lse_scale="natural",
        )
        if self._model_type is not None:
            caps_kwargs["model_type"] = self._model_type
        if self._uses_kvarn_cache:
            layout = self._kvarn_layout
            if tail_mode:
                # Precision-tail segment: dense always-attended window read
                # by the fp8 arm; LSE is always required (two-launch merge).
                # The fp8 arm lives on the decode kernel only, so the tail
                # segment always launches decode mode; max_chunks_per_row=2
                # is the body extend's split cap (keeps the extend_tail
                # split-K scratch at ~268 MB, not rows x 64 chunks).
                caps_kwargs.update(
                    model_type=int(self._module.ModelType.KVARN_TAIL_FP8),
                    scale_format=1,  # ScaleFormat.ARBITRARY_FP32 (GLM fp8)
                    kvarn_k_bits=layout.k_bits,
                    kvarn_v_bits=layout.v_bits,
                    kvarn_group=layout.group,
                    max_width=(
                        self._kvarn_tail_tokens
                        if mode == "extend_tail_gather"
                        else self._kvarn_tail_local_tokens
                    ),
                    cache_record_bytes=layout.record_bytes,
                    return_lse=True,
                    mode="decode",
                    num_q_heads=self._kernel_num_heads,
                    max_chunks_per_row=2,
                )
                if mode == "extend_tail_gather":
                    # Binds the gathered arena: one KV page equals one tile
                    # (group records), mirroring the body ckv_extend plan.
                    caps_kwargs["page_size"] = int(layout.group)
            else:
                caps_kwargs.update(
                    scale_format=3,  # ScaleFormat.KVARN_DUAL
                    kvarn_k_bits=layout.k_bits,
                    kvarn_v_bits=layout.v_bits,
                    kvarn_group=layout.group,
                    cache_record_bytes=layout.record_bytes,
                    # The body launch feeds the tail LSE merge.
                    return_lse=(
                        self.need_to_return_lse_for_decode
                        or self._kvarn_tail_local_tokens > 0
                    ),
                )
            if mode == "ckv_extend":
                # The full-CKV gather route binds the tile-stride gathered
                # arena (records + per-tile tail, mirroring the source pool),
                # so one KV page equals one kernel tile: the decode kernel
                # addresses records as block*tile_stride + row*record and
                # every selected row lands at its true record offset. A flat
                # (page, record) view shifts every tile past the first by
                # its accumulated per-tile tails.
                caps_kwargs["page_size"] = layout.group
            if not tail_mode and mode in ("extend", "ckv_extend"):
                # MG prefill runs the KVARN_DUAL arm (scalar record IO --
                # the 536B-class row is 8B- not 16B-aligned -- with inline
                # dequant): keep the extend mode and the DCP-global padded
                # head width the forward pass binds (64 = four 16-head MG
                # groups replicated across CTAs; mg_n_hg==1 keeps one group
                # per CTA in smem), and keep the full chunk count
                # (ceil(topk / 64); MG carries no split-K scratch).
                caps_kwargs["num_q_heads"] = self._kernel_num_heads
                caps_kwargs.pop("decode_max_rows", None)
        if self._uses_nvfp4_cache:
            caps_kwargs.update(_nvfp4_run_options(is_glm_next=self._is_glm_next))
        caps = self._module.Caps(**caps_kwargs)
        return caps, self._module.plan(caps)

    def _preparation_prefix(self) -> str:
        return f"attention.sparse_mla.{id(self):x}"

    def _request_name(self, key: tuple[str, int]) -> str:
        mode, rows = key
        return f"{self._preparation_prefix()}.{mode}.m{rows}"

    def _plan(self, key: tuple[str, int]):
        """The plan for key, declared on first use with its default configuration."""
        plan = self._plans.get(key)
        if plan is None:
            caps, plan = self._declare_plan(*key)
            self._plans[key] = plan
            self._plan_caps[key] = caps
        return plan

    def _make_prepare_call(self, state, caps):
        from b12x.preparation import PreparedCall

        if self._bound_kv_cache is None or self._bound_kv_cache.numel() == 0:
            raise PreparationResourceUnavailableError(
                "sparse MLA benchmark requires the published KV cache"
            )
        rows = int(caps.max_q_rows)
        q = torch.empty(
            (rows, int(caps.num_q_heads), int(caps.head_dim)),
            dtype=caps.dtype,
            device=caps.device,
        )
        # Selection reads the real cache.  Page zero is a legal physical page
        # and avoids a synthetic cache or a pool-sized checkpoint. The
        # ckv_extend plan probes the full-CKV gather workspace as the runtime
        # binds it -- a tile-stride arena view (page == tile, per-tile tail)
        # -- so the prepared launcher compiles for the arena's address
        # contract, not the source pool's.
        kv_cache = self._bound_kv_cache
        if self._uses_kvarn_cache and int(caps.page_size) == self._kvarn_layout.group:
            kv_cache = self._ckv_arena_probe()
        selected = torch.zeros(
            (rows, int(caps.max_width)), dtype=torch.int32, device=caps.device
        )
        cache_lengths = torch.full(
            (int(caps.max_batch),),
            int(caps.page_size),
            dtype=torch.int32,
            device=caps.device,
        )
        selected_lengths = torch.ones(rows, dtype=torch.int32, device=caps.device)
        (scratch_spec,) = state.scratch_specs()
        # Preparation calls execute serially. Borrow caller-owned scratch so
        # exact-row plans do not each allocate a workspace.
        workspace = current_workspace_manager()
        workspace.reserve_all(((self._scratch_nbytes,), torch.uint8))
        (scratch,) = workspace.get_simultaneous(
            (scratch_spec.shape, scratch_spec.dtype)
        )
        binding = state.bind(
            scratch=scratch,
            q=q,
            selected_indices=selected,
            cache_seqlens_int32=cache_lengths,
            nsa_cache_seqlens_int32=selected_lengths,
            kv_cache=kv_cache,
        )
        # Launcher construction belongs to the declaration's preparation
        # callback: runtime only consumes these retained launchers.
        state.prime(binding, kv_cache=kv_cache)

        def produce():
            q.fill_(1)

        return PreparedCall(
            run=lambda: state.run(binding, kv_cache=kv_cache),
            produce=produce,
            # Closures retain probes until preparation has synchronized. The
            # published launcher binds live inputs and must not pin these probes.
        )

    def _ckv_arena_probe(self) -> torch.Tensor:
        """Tile-stride arena view over the CKV gather workspace.

        Mirrors :meth:`_gather_full_ckv`'s returned geometry: staging span
        first, then one ``tile_stride_bytes`` tile per ``group`` gathered
        tokens. The workspace persists (reserved at page-geometry
        finalization), so prepared launchers retain a live binding.
        """
        specs = self._workspace_specs(
            input_num_heads=self._kernel_num_heads, include_ckv=True
        )
        gathered = next(
            workspace
            for workspace in current_workspace_manager().get_simultaneous(*specs)
            if workspace.shape[0] == self._ckv_gather_bytes()
        )
        layout = self._kvarn_layout
        record = self._cache_record_bytes
        capacity_rows = self.dcp_world_size * self._ckv_local_capacity
        staging_bytes = capacity_rows * record
        return gathered[staging_bytes:].as_strided(
            (capacity_rows // layout.group, layout.group, record),
            (layout.tile_stride_bytes, record, 1),
        )

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        if layer is not self:
            raise ValueError("sparse MLA preparation owner mismatch")
        if self._bound_kv_cache is None or self._bound_kv_cache.numel() == 0:
            return ()
        requests = []
        # Decode launches are exact in their row count, so every planned token
        # count that can route to decode gets its own plan.
        for rows in workload.token_counts:
            key = ("decode", rows)
            if rows <= self._decode_max_rows and key not in self._plans:
                self._plan_caps[key], self._plans[key] = self._declare_plan(*key)
            if (
                self._uses_kvarn_cache
                and self._kvarn_tail_local_tokens > 0
                and rows <= self._decode_max_rows
            ):
                tail_key = ("decode_tail", rows)
                if tail_key not in self._plans:
                    self._plan_caps[tail_key], self._plans[tail_key] = (
                        self._declare_plan(*tail_key)
                    )
                # The tail segment's split-K scratch can exceed the body's
                # (N_local up to 4096 > topk 2048 chunks); the shared
                # workspace reservation must cover the max.
                self._scratch_nbytes = max(
                    self._scratch_nbytes,
                    int(self._plans[tail_key].scratch_specs()[0].nbytes),
                )
        for key, declaration in self._plans.items():
            caps = self._plan_caps[key]
            requests.append(
                declaration.request(
                    name=self._request_name(key),
                    prepare_call=lambda state, caps=caps: self._make_prepare_call(
                        state, caps
                    ),
                    benchmark_call=lambda state, caps=caps: self._make_prepare_call(
                        state, caps
                    ),
                )
            )
        if self._is_glm_next:
            from b12x.preparation import PreparedCall
            from torch._subclasses.fake_tensor import FakeTensorMode

            # The declaration retains tensor formats for compilation, not
            # live inputs. Allocate execution probes only inside each call.
            with FakeTensorMode():
                kv_probe = torch.empty(
                    (self._max_tokens, self.kv_lora_rank),
                    dtype=torch.bfloat16,
                    device=self._bound_kv_cache.device,
                )
                ignored_slots = torch.empty(
                    (self._max_tokens,),
                    dtype=torch.int64,
                    device=self._bound_kv_cache.device,
                )
            writer_plan = self._module.plan_cache_writer(
                kv_probe, self._bound_kv_cache, ignored_slots
            )
            self._cache_writer_plan = writer_plan
            def prepare_writer(state):
                probe = torch.empty(
                    (self._max_tokens, self.kv_lora_rank),
                    dtype=torch.bfloat16,
                    device=state.kv_cache.device,
                )
                # The writer touches only slot zero; retain that single page
                # for restoration instead of cloning the serving KV pool.
                slots = torch.zeros(
                    (self._max_tokens,), dtype=torch.int64, device=state.kv_cache.device
                )
                saved_page = state.kv_cache[0].clone()

                def produce():
                    probe.fill_(1)

                def restore():
                    state.kv_cache[0].copy_(saved_page)

                return PreparedCall(
                    run=lambda: state.run(probe, state.kv_cache, slots),
                    produce=produce,
                    restore=restore,
                    # Closures own these tensors until preparation completes.
                    # Published launchers rebind live inputs and must not keep
                    # the probes or the temporary cache-page backup alive.
                )

            requests.append(
                writer_plan.request(
                    name=f"{self._preparation_prefix()}.cache_writer",
                    prepare_call=prepare_writer,
                    benchmark_call=prepare_writer,
                    dependencies=tuple(request.name for request in requests),
                )
            )
            if self._kvarn_tail_local_tokens:
                tail_writer_plan = self._module.plan_cache_writer_fp8_tail(
                    kv_probe, self._bound_kv_cache, ignored_slots
                )
                self._cache_writer_tail_plan = tail_writer_plan
                requests.append(
                    tail_writer_plan.request(
                        name=f"{self._preparation_prefix()}.cache_writer_fp8_tail",
                        prepare_call=prepare_writer,
                        benchmark_call=prepare_writer,
                        dependencies=tuple(request.name for request in requests),
                    )
                )
        provider = getattr(self, "_physical_selection_provider", None)
        selection_request = getattr(
            provider, "get_b12x_physical_selection_preparation_request", None
        )
        if selection_request is not None and self.dcp_world_size == 1:
            requests.append(selection_request())
        if not requests:
            return ()
        return (
            B12xPreparationUnit(
                name="sparse MLA",
                key=self._preparation_prefix(),
                requests=tuple(requests),
                stage="state",
            ),
        )

    def _use_decode_execution(
        self,
        attn_metadata: B12xMLASparseMetadata,
        num_tokens: int,
    ) -> bool:
        if attn_metadata.max_query_len <= 1:
            return True
        return (
            attn_metadata.is_spec_decode
            and attn_metadata.max_query_len <= self._max_speculative_decode_query_len
            and num_tokens
            <= attn_metadata.num_reqs * self._max_speculative_decode_query_len
            and num_tokens <= self._decode_max_rows
        )

    def _workspace_specs(
        self,
        *,
        input_num_heads: int,
        include_ckv: bool,
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        q_spec = (
            (self._max_tokens, input_num_heads, self._q_head_dim),
            torch.bfloat16,
        )
        # Full-CKV attention has only local query heads and does not gather
        # queries. Its cache receive buffer need not coexist with global-head
        # query-gather scratch or the decode plan's split-K intermediates.
        scratch_nbytes = (
            int(self._plan(("ckv_extend", self._max_tokens)).scratch_specs()[0].nbytes)
            if include_ckv
            else self._scratch_nbytes
        )
        if include_ckv and self._uses_kvarn_cache and self._kvarn_tail_local_tokens > 0:
            # The extend tail launch runs the fp8 DECODE arm over the full
            # extend row count: its split-K scratch is ~2x the MG body's.
            # Size the shared scratch for the larger of the two, or the
            # first tail-on prefill fails the bind-time scratch check.
            tail_nbytes = int(
                self._plan(("extend_tail_gather", self._max_tokens))
                .scratch_specs()[0].nbytes
            )
            scratch_nbytes = max(scratch_nbytes, tail_nbytes)
        scratch_spec = ((scratch_nbytes,), torch.uint8)
        if self._uses_kvarn_cache:
            ckv_specs = (((self._ckv_gather_bytes(),), torch.uint8),)
        else:
            ckv_specs = (
                (
                    (
                        self.dcp_world_size * self._ckv_local_capacity,
                        self._cache_record_bytes,
                    ),
                    torch.uint8,
                ),
            )
        return (
            q_spec,
            scratch_spec,
            *(ckv_specs if include_ckv else ()),
        )

    def _ckv_gather_bytes(self) -> int:
        """Bytes of the KVarN full-CKV gather workspace.

        The KVarN decode kernel addresses the gathered cache in the pool's
        tile geometry (``group`` records + per-tile tail), so the arena
        mirrors it: a contiguous staging span for the NCCL all-gather plus
        one tile-stride tile per ``group`` gathered tokens. Staging rows are
        always a multiple of ``group``: per-rank capacity rounds to
        ``page / gcd(page, dcp)`` tokens and the kernel page is a multiple
        of the tile group.
        """
        rows = self.dcp_world_size * self._ckv_local_capacity
        layout = self._kvarn_layout
        assert rows % layout.group == 0
        return (
            rows * self._cache_record_bytes
            + rows // layout.group * layout.tile_stride_bytes
        )

    def reserve_full_ckv_workspace(self) -> None:
        """Reserve the full-CKV prefill workspace before KV-cache sizing.

        The memory profile skips attention. Without this reservation, the
        first eligible prefill would grow the locked workspace by the gathered
        cache receive buffer. The selection remap kernels are compiled here
        too, so that prefill does not compile them either.
        """
        if not (self._ckv_gather_enabled and self._kernel_page_size_finalized):
            return
        current_workspace_manager().get_simultaneous(
            *self._workspace_specs(
                input_num_heads=_round_up_heads(self.num_heads),
                include_ckv=True,
            )
        )
        assert self.topk_indices_buffer is not None
        # Serving binds row slices of persistent buffers with these widths and
        # strides; matching them selects the same compiled specializations.
        device = self.topk_indices_buffer.device
        rank_req = torch.zeros(
            (self.dcp_world_size, self._max_seqs), dtype=torch.int32, device=device
        )[:, :1]
        selected = torch.empty((1, self._topk_tokens), dtype=torch.int32, device=device)
        counts = torch.empty(1, dtype=torch.int32, device=device)
        alignment = _ckv_rank_token_alignment(
            self._kernel_page_size, self.dcp_world_size
        )
        # Rank spans are multiples of the alignment, with or without a factor
        # of 16; the integer specialization differs between the two.
        for padded_rank_tokens in (alignment, 2 * alignment):
            _map_global_topk_to_gathered_ckv(
                torch.zeros(1, dtype=torch.int32, device=device),
                self.topk_indices_buffer[:1],
                rank_req,
                rank_req,
                selected,
                counts,
                dcp_size=self.dcp_world_size,
                cp_kv_cache_interleave_size=self._cp_kv_cache_interleave_size,
                padded_rank_tokens=padded_rank_tokens,
            )
        _mask_page_table_after_nsa_len(selected, counts)

    def _borrow_workspaces(
        self,
        *,
        input_num_heads: int | None = None,
        include_ckv: bool = False,
    ) -> list[torch.Tensor]:
        input_num_heads = (
            self._input_num_heads if input_num_heads is None else input_num_heads
        )
        specs = self._workspace_specs(
            input_num_heads=input_num_heads,
            include_ckv=include_ckv,
        )
        return current_workspace_manager().get_simultaneous(*specs)

    def supports_fused_mla_query_output(
        self,
        num_heads: int,
        output_dtype: torch.dtype,
    ) -> bool:
        return bool(
            self.dcp_world_size == 1
            and output_dtype == torch.bfloat16
            and num_heads == self._input_num_heads == self._kernel_num_heads
            and self._q_head_dim == 576
        )

    def reduce_scatter_dcp_output(self, output: torch.Tensor) -> torch.Tensor | None:
        """Reuse consumed query storage for the prefill output collective.

        The corrected attention output remains in disjoint kernel scratch.
        Pack it into the query buffer in head-major order, then let NCCL write
        each rank's reduction at its in-place receive offset. The resulting
        head-major matrices stay live until the following value projection.
        Decode and symmetric-memory transports retain their existing path.
        """
        if not self._is_glm_next or output.shape[0] <= self._decode_max_rows:
            return None
        group = get_dcp_group()
        communicator = getattr(group, "device_communicator", None)
        comm = getattr(communicator, "pynccl_comm", None)
        if comm is None or comm.disabled or should_nccl_symm_mem_ag_rs():
            return None

        rows, heads, head_dim = output.shape
        q_buffer = self._borrow_workspaces()[0]
        assert output.dtype == q_buffer.dtype
        packed = q_buffer.view(-1)[: output.numel()].view(heads, rows, head_dim)
        packed.copy_(output.transpose(0, 1))
        local_heads = heads // self.dcp_world_size
        local = packed.narrow(0, group.rank_in_group * local_heads, local_heads)
        comm.reduce_scatter(local, packed)
        return local.transpose(0, 1)

    def gather_dcp_query(self, query: torch.Tensor) -> torch.Tensor:
        """Gather GLM prefill heads into caller-owned attention query storage.

        The rank-major receive view borrows kernel scratch only until the
        layout copy completes on the execution stream. The attention call
        rebinds that scratch while retaining the final query at the same address.
        Decode and transport-specific all-gathers keep their established path.
        """
        group = get_dcp_group()
        if not self._is_glm_next or query.shape[0] <= self._decode_max_rows:
            return group.all_gather(query, dim=1)
        communicator = getattr(group, "device_communicator", None)
        b12x_comm = getattr(communicator, "b12x_ar_comm", None)
        if (
            b12x_comm is not None
            and not b12x_comm.disabled
            and getattr(b12x_comm, "should_all_gather", None) is not None
            and b12x_comm.should_all_gather(query, 1)
        ):
            return group.all_gather(query, dim=1)

        rows, local_heads, head_dim = query.shape
        q_buffer, scratch = self._borrow_workspaces()[:2]
        output = q_buffer[:rows]
        receive = scratch[: output.numel() * output.element_size()].view(query.dtype)
        receive = receive.view(self.dcp_world_size, rows, local_heads, head_dim)
        local = receive[group.rank_in_group]
        local.copy_(query)
        _dcp_all_gather_current_stream(group, local.view(-1), receive.view(-1))
        output.view(rows, self.dcp_world_size, local_heads, head_dim).copy_(
            receive.movedim(0, 1)
        )
        return output

    def get_fused_mla_query_output(
        self,
        num_tokens: int,
        num_heads: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if (
            not self.supports_fused_mla_query_output(num_heads, output_dtype)
            or num_tokens <= 0
            or num_tokens > self._max_tokens
        ):
            return None
        q_buffer = self._borrow_workspaces()[0]
        output = q_buffer[:num_tokens, :num_heads]
        if not output.is_contiguous():
            raise RuntimeError("B12X fused MLA query output must be contiguous.")
        return output

    def _plan_key(
        self, attn_metadata: B12xMLASparseMetadata, num_tokens: int
    ) -> tuple[str, int]:
        """The route and declared row count serving one batch."""
        if self.uses_full_ckv_dcp(attn_metadata, num_tokens):
            return ("ckv_extend", self._max_tokens)
        if self._use_decode_execution(attn_metadata, num_tokens):
            return ("decode", num_tokens)
        return ("extend", self._max_tokens)

    def finalize_kv_cache_geometry(self, kernel_page_size: int) -> None:
        """Finalize kernel plans and workspace memory before KV profiling.

        GLM5Next hybrid cache alignment resolves the physical page size after
        model construction. Full-CKV gather workspace depends on that value,
        so every execution slot must be sized while the memory profiler can
        still subtract the allocation from the KV-cache budget.

        Args:
            kernel_page_size: Resolved physical KV-cache page size in tokens.

        Raises:
            RuntimeError: If an established page size is changed.
        """
        if not self._is_glm_next:
            return
        if self._kernel_page_size_finalized:
            if kernel_page_size != self._kernel_page_size:
                raise RuntimeError(
                    "B12X GLM5Next KV-cache page size is immutable after "
                    f"finalization: {self._kernel_page_size} != {kernel_page_size}."
                )
            return
        self._set_kernel_page_size(kernel_page_size)
        self._kernel_page_size_finalized = True
        # Pre-reserve the full-CKV gather workspace NOW: the workspace
        # locks after warmup, and the warmup/profile batches do not
        # reliably engage the ckv_extend route -- a first real gather
        # after the lock would then fail on arena growth. Reserving here
        # (pre-lock, geometry final) sizes the arena for the largest
        # simultaneous set; smaller decode sets reuse it.
        if self._ckv_gather_enabled:
            from vllm.v1.worker.workspace import current_workspace_manager

            current_workspace_manager().get_simultaneous(
                *self._workspace_specs(
                    input_num_heads=self._kernel_num_heads,
                    include_ckv=True,
                )
            )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if self._uses_kvarn_cache:
            import os as _os

            if _os.environ.get("B12X_KVARN_DEBUG"):
                logger.info(
                    "kvarn bind: kv_cache %s strides %s dtype %s; "
                    "layout record %d page_tail %d page %d",
                    tuple(kv_cache.shape),
                    tuple(kv_cache.stride()),
                    kv_cache.dtype,
                    self._kvarn_layout.record_bytes,
                    self._kvarn_layout.page_tail_bytes_per_token,
                    self._kvarn_layout.page_size_bytes,
                )
        if self._uses_glm_dsa_nvfp4_cache and (
            kv_cache.ndim != 3
            or int(kv_cache.shape[-1]) != _GLM_DSA_NVFP4_CACHE_RECORD_BYTES
            or kv_cache.dtype != torch.uint8
        ):
            raise ValueError(
                "B12X GLM DSA NVFP4 cache must have shape "
                f"[pages, page_size, {_GLM_DSA_NVFP4_CACHE_RECORD_BYTES}] uint8"
            )
        if self._is_glm_next:
            if (
                kv_cache.ndim != 3
                or int(kv_cache.shape[-1]) != self._cache_record_bytes
            ):
                raise ValueError(
                    "B12X GLM5Next cache must have shape "
                    f"[pages, page_size, {self._cache_record_bytes}], got "
                    f"shape={tuple(kv_cache.shape)}, stride={kv_cache.stride()}, "
                    f"dtype={kv_cache.dtype}"
                )
            cache_page_size = int(kv_cache.shape[1])
            if self._kernel_page_size_finalized:
                if cache_page_size != self._kernel_page_size:
                    raise RuntimeError(
                        "B12X GLM5Next bound cache does not match the finalized "
                        f"page size: {cache_page_size} != {self._kernel_page_size}."
                    )
            elif self._ckv_gather_enabled:
                raise RuntimeError(
                    "B12X GLM5Next full-CKV gather requires page geometry "
                    "finalization before KV-cache memory profiling."
                )
            else:
                self._set_kernel_page_size(cache_page_size)
        # The cache writer plan is bound to the outgoing generation's tensor;
        # get_b12x_preparation_units declares a fresh one for the new pool.
        self._cache_writer_plan = None
        self._bound_kv_cache = kv_cache

    def unbind_kv_cache(self) -> None:
        self._cache_writer_plan = None
        self._bound_kv_cache = None

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        attn_metadata=None,
    ) -> None:
        if self._uses_kvarn_cache:
            if kv_cache.numel() == 0:
                return
            plan = self._cache_writer_plan
            if plan is None:
                raise PreparationResourceUnavailableError(
                    f"{self._preparation_prefix()} has no declared cache_writer plan"
                )
            slots = slot_mapping.flatten()
            tail_plan = self._cache_writer_tail_plan
            if (
                tail_plan is None
                or attn_metadata is None
                or attn_metadata.cache_seq_lens_per_token is None
            ):
                # kv_c_normed is already Hadamard-rotated (MLA wrapper). The
                # per-token record scheme stores each token directly at its
                # slot (inline scale/zp) -- no shadow pool, no whole-page
                # re-quant, no host syncs; capture-safe at any batch size.
                self._concat_and_cache_glm_next_mla_kvarn(
                    kv_c_normed, kv_cache, slots, plan=plan
                )
                return
            # Precision tail: uniform sliding rule — stores split fp8
            # (pos >= ts) vs kvarn (pos < ts) for extends and decodes
            # alike. Dispatch is shape-static: both writers see the full
            # batch with non-mine slots masked to -1 (writers skip
            # slot < 0).
            from vllm.v1.attention.backends.mla.kvarn_tail import (
                kvarn_tail_boundary,
                kvarn_tail_demote_ranges,
            )

            num_tokens = slots.shape[0]
            req_ids = attn_metadata.req_id_per_token[:num_tokens]
            demote_ranges = getattr(attn_metadata, "kvarn_tail_demote_ranges", None)
            # The fp8 test is in GLOBAL request-relative units: ts is the
            # manager's global boundary and the rope positions are the
            # rank's DCP-LOCAL sequence positions (0..local_len). Convert
            # local -> global through the DCP interleave -- the exact
            # inverse of the gather mapper's ownership rule
            # (owner = g//I % ws, local_idx = (g//(ws*I))*I + g%I).
            pos_l = attn_metadata.global_positions
            if pos_l is None:
                pos_l = attn_metadata.cache_seq_lens_per_token[:num_tokens] - 1
            else:
                pos_l = pos_l[:num_tokens]
            ws = self.dcp_world_size
            itl = int(attn_metadata.cp_kv_cache_interleave_size)
            pos_g = (
                (pos_l // itl) * (ws * itl)
                + self.dcp_rank * itl
                + (pos_l % itl)
            )
            ts = getattr(attn_metadata, "kvarn_tail_start", None)
            if ts is None:
                boundary = None
            else:
                boundary = kvarn_tail_boundary(
                    ts[: attn_metadata.num_reqs], req_ids
                )
            if boundary is None:
                # warmup/profile forward: no manager state yet; store
                # everything as kvarn (the dummy cache is discarded).
                tail_slots = slots.new_full(slots.shape, -1)
                body_slots = slots
            else:
                is_tail = pos_g >= boundary
                tail_slots = torch.where(
                    is_tail.bool(), slots, slots.new_full((), -1)
                )
                body_slots = torch.where(
                    is_tail.bool(), slots.new_full((), -1), slots
                )
            self._module.concat_and_cache_glm_next_mla_fp8_tail(
                kv_c_normed, kv_cache, tail_slots, plan=tail_plan
            )
            self._concat_and_cache_glm_next_mla_kvarn(
                kv_c_normed, kv_cache, body_slots, plan=plan
            )
            _kvarn_force_sync()
            if (
                os.environ.get("KVARN_TAIL_DEBUG")
                and slots.shape[0] > 1000
            ):
                try:
                    _sl_seq = globals().get("_KVAR_DBG_SEQ", 0)
                    if _sl_seq <= 12:
                        torch.cuda.synchronize()
                        _pr = list(range(760, 790)) + list(range(1520, 1570))
                        _bs = body_slots[_pr].to(torch.int64)
                        _tsl = tail_slots[_pr].to(torch.int64)
                        _kc = (
                            kv_cache.view(torch.uint8)
                            if kv_cache.dtype == torch.float8_e4m3fn
                            else kv_cache
                        )
                        _prec = {}
                        for _i, _s in enumerate(_bs.tolist()):
                            if _s >= 0:
                                _prec[_pr[_i]] = _kc[
                                    _s // _kc.shape[1], _s % _kc.shape[1]
                                ].cpu().clone()
                        _kvarn_dbg_save(
                            {
                                "rank": self.dcp_rank,
                                "seq": _sl_seq,
                                "slots": _bs.cpu(),
                                "tail_slots": _tsl.cpu(),
                                "pool_recs": _prec,
                                "latents": kv_c_normed[_pr].detach().clone().cpu(),
                            },
                            f"/home/js/kvarn-probes/kvarn-stage{_sl_seq}-poststore-r{self.dcp_rank}.pt",
                        )
                except Exception as _e:
                    logger.warning("stage store probe failed %s", _e)
            if demote_ranges is not None:
                _dbg_demote = None
                if (
                    os.environ.get("KVARN_TAIL_DEBUG")
                    and not torch.cuda.is_current_stream_capturing()
                    and not os.path.exists(
                        f"/home/js/kvarn-probes/kvarn-demote-capture-r{self.dcp_rank}.pt"
                    )
                ):
                    _dr = demote_ranges[: attn_metadata.num_reqs]
                    if bool((_dr[:, 1] > _dr[:, 0]).any()):
                        # Ranges are GLOBAL request-relative positions; the
                        # kernel demotes LOCAL rows. Mirror its conversion
                        # (kvarn_tail_geometry.global_to_local_row_start).
                        from vllm.v1.attention.backends.mla.kvarn_tail_geometry import (
                            global_to_local_row_start,
                        )

                        _rec = self._cache_record_bytes
                        _itl = int(attn_metadata.cp_kv_cache_interleave_size)
                        _bt_all = attn_metadata.block_table
                        _rows = []
                        for _r in range(_dr.shape[0]):
                            _lo, _hi = int(_dr[_r, 0]), int(_dr[_r, 1])
                            if _hi <= _lo:
                                _rows.append((_r, None, None))
                                continue
                            _jlo = int(
                                global_to_local_row_start(
                                    torch.tensor([_lo]),
                                    self.dcp_rank,
                                    self.dcp_world_size,
                                    _itl,
                                )[0]
                            )
                            _jhi = int(
                                global_to_local_row_start(
                                    torch.tensor([_hi]),
                                    self.dcp_rank,
                                    self.dcp_world_size,
                                    _itl,
                                )[0]
                            )
                            _bt = _bt_all[_r]
                            _jj = torch.arange(
                                _jlo, _jhi, device=_bt.device, dtype=torch.int64
                            )
                            _slots = (
                                _bt[_jj // attn_metadata.block_size]
                                * attn_metadata.block_size
                                + _jj % attn_metadata.block_size
                            )
                            _rows.append(
                                (
                                    _r,
                                    _slots.to(torch.int64),
                                    kv_cache.reshape(-1, _rec)[
                                        _slots.to(torch.int64)
                                    ].clone(),
                                )
                            )
                        _dbg_demote = (
                            _dr.clone(),
                            _rows,
                            _rec,
                            _jlo if _rows else 0,
                        )
                kvarn_tail_demote_ranges(
                    kv_cache,
                    demote_ranges[: attn_metadata.num_reqs],
                    attn_metadata.block_table,
                    attn_metadata.block_size,
                    self._kvarn_layout,
                    num_tokens,
                    dcp_rank=self.dcp_rank,
                    dcp_size=self.dcp_world_size,
                    interleave=int(attn_metadata.cp_kv_cache_interleave_size),
                )
                if _dbg_demote is not None:
                    try:
                        _dr, _rows, _rec, _jlo = _dbg_demote
                        _after = []
                        for _r, _slots, _ in _rows:
                            _after.append(
                                None
                                if _slots is None
                                else kv_cache.reshape(-1, _rec)[_slots].clone()
                            )
                        _kvarn_dbg_save(
                            {
                                "ranges": _dr,
                                "jlo": _jlo,
                                "slots": [r[1] for r in _rows],
                                "before": [r[2] for r in _rows],
                                "after": _after,
                                "block_size": attn_metadata.block_size,
                                "dcp_rank": self.dcp_rank,
                                "dcp_size": self.dcp_world_size,
                                "interleave": int(
                                    attn_metadata.cp_kv_cache_interleave_size
                                ),
                            },
                            f"/home/js/kvarn-probes/kvarn-demote-capture-r{self.dcp_rank}.pt",
                        )
                        logger.warning("KVARN_TAIL_DEBUG: demote capture saved")
                    except Exception as _e:
                        logger.warning("KVARN_TAIL_DEBUG: demote capture failed %s", _e)
            if os.environ.get("KVARN_TAIL_DEBUG") and (
                not torch.cuda.is_current_stream_capturing()
            ):
                valid = slots[slots >= 0][:8192]
                if valid.numel():
                    rows = kv_cache.reshape(-1, self._cache_record_bytes)[
                        valid.to(torch.int64)
                    ]
                    # kvarn pad bytes [520,528) are zero; fp8 group scales
                    # make them nonzero (unit-latent groups excluded).
                    n_fp8 = int((rows[:, 520:528] != 0).any(-1).sum().item())
                    owned = slots >= 0
                    csl_v = attn_metadata.cache_seq_lens_per_token[:num_tokens][owned]
                    ts_v = (
                        boundary[owned]
                        if boundary is not None
                        else None
                    )
                    logger.warning(
                        "KVARN_TAIL_DEBUG store: tokens=%d prefills=%d "
                        "slots=%d fp8=%d kvarn=%d ts=%s csl_min=%d "
                        "kvarn_csl_max=%s",
                        num_tokens,
                        attn_metadata.num_prefills,
                        int(valid.numel()),
                        n_fp8,
                        int(valid.numel()) - n_fp8,
                        (
                            ts[: attn_metadata.num_reqs].tolist()
                            if ts is not None
                            else None
                        ),
                        int(csl_v.min().item()),
                        (
                            int(csl_v[~(csl_v - 1 >= ts_v)].max().item())
                            if ts_v is not None and not bool((csl_v - 1 >= ts_v).all())
                            else None
                        ),
                    )
                if (
                    boundary is not None
                    and int(boundary.max()) > 0
                    and not os.path.exists(
                        f"/home/js/kvarn-probes/kvarn-pool-dump-r{self.dcp_rank}.pt"
                    )
                    and attn_metadata.num_prefills > 0
                ):
                    try:
                        _bs = int(attn_metadata.block_size)
                        _bt0 = attn_metadata.block_table[0]
                        _jj = torch.arange(
                            760, 780, device=_bt0.device, dtype=torch.int64
                        )
                        _pslots = (
                            _bt0[_jj // _bs] * _bs + _jj % _bs
                        ).to(torch.int64)
                        _itl = int(attn_metadata.cp_kv_cache_interleave_size)
                        _pg = (pos_l[:num_tokens].to(torch.int64) // _itl) * (
                            self.dcp_world_size * _itl
                        ) + self.dcp_rank * _itl + (pos_l[:num_tokens].to(torch.int64) % _itl)
                        _req0 = req_ids == 0
                        _prow = {}
                        for _t in range(num_tokens):
                            if not bool(_req0[_t]):
                                continue
                            _g = int(_pg[_t])
                            if 1525 <= _g <= 1560:
                                _prow[_g] = kv_cache.reshape(-1, self._cache_record_bytes)[
                                    int(slots[_t])
                                ].clone()
                        _kvarn_dbg_save(
                            {
                                "rank": self.dcp_rank,
                                "rows_by_global": _prow,
                                "local_760_780": kv_cache.reshape(
                                    -1, self._cache_record_bytes
                                )[_pslots].cpu(),
                                "ts": int(boundary[0]),
                                "req0_slots": slots[_req0].cpu(),
                                "req0_body_slots": body_slots[_req0].cpu(),
                                "req0_tail_slots": tail_slots[_req0].cpu(),
                                "req0_pos_l": pos_l[_req0].to(torch.int64).cpu(),
                                "req0_pos_g": _pg[_req0].cpu(),
                                "req0_is_tail": (
                                    is_tail[_req0].cpu()
                                    if boundary is not None
                                    else None
                                ),
                            },
                            f"/home/js/kvarn-probes/kvarn-pool-dump-r{self.dcp_rank}.pt",
                        )
                        logger.warning("KVARN_TAIL_DEBUG: pool dump saved")
                    except Exception as _e:
                        logger.warning("KVARN_TAIL_DEBUG: pool dump failed %s", _e)
            if (
                os.environ.get("KVARN_TAIL_HASH") == "1"
                and getattr(self, "_kvarn_tail_indices_buffer", None) is not None
                and num_tokens <= self._decode_max_rows
                and self.dcp_rank == 0
                and attn_metadata.num_prefills == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Per-step checksum of the decode-tail bindings. The writer
                # dispatch runs eagerly per step, so after each graph replay
                # this shows whether the captured copy kernels recomputed the
                # persistent tail buffers from live metadata (hash varies
                # with csl/ts) or the replay froze them.
                _tib = self._kvarn_tail_indices_buffer[:num_tokens]
                _tlb = self._kvarn_tail_lengths_buffer[:num_tokens]
                _csl_t = attn_metadata.cache_seq_lens_per_token[:num_tokens]
                try:
                    with open(
                        f"/home/js/kvarn-probes/kvtailhash-r{self.dcp_rank}.log",
                        "a",
                    ) as _f:
                        _f.write(
                            "nt=%d idx_sum=%d len_sum=%d csl_sum=%d ts_sum=%s\n"
                            % (
                                num_tokens,
                                int(_tib.sum().item()),
                                int(_tlb.sum().item()),
                                int(_csl_t.sum().item()),
                                (
                                    int(ts.sum().item())
                                    if ts is not None
                                    else -1
                                ),
                            )
                        )
                except OSError:
                    pass
            return
        if self._uses_glm_dsa_nvfp4_cache:
            if kv_cache.numel() == 0:
                return
            assert self._concat_and_cache_nvfp4_mla_fp8_rope is not None
            self._concat_and_cache_nvfp4_mla_fp8_rope(
                kv_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                slot_mapping.flatten(),
                k_scale,
            )
            return
        if not self._is_glm_next:
            return super().do_kv_cache_update(
                kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
            )
        if kv_cache.numel() == 0:
            return
        plan = self._cache_writer_plan
        if plan is None:
            raise PreparationResourceUnavailableError(
                f"{self._preparation_prefix()} has no declared cache_writer plan"
            )
        if int(k_pe.shape[-1]) != 0:
            raise ValueError(
                "B12X GLM5Next cache updates require a zero-width RoPE tensor, "
                f"got shape={tuple(k_pe.shape)}."
            )
        assert self._concat_and_cache_glm_next_mla is not None
        self._concat_and_cache_glm_next_mla(
            kv_c_normed, kv_cache, slot_mapping.flatten(), plan=plan
        )

    def uses_full_ckv_dcp(
        self, attn_metadata: B12xMLASparseMetadata, num_tokens: int
    ) -> bool:
        # Decode layers ask once per step; reject them before the CUDA query.
        if not (self._ckv_gather_enabled and attn_metadata.dcp_ckv_gather_eligible):
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        return (
            self._kernel_page_size_finalized
            and attn_metadata.num_decode_tokens == 0
            and num_tokens == attn_metadata.num_actual_tokens
            and 0 < attn_metadata.dcp_padded_total_tokens <= self._ckv_local_capacity
            and attn_metadata.dcp_local_total_tokens
            <= attn_metadata.dcp_padded_total_tokens
            and all(
                value is not None
                for value in (
                    attn_metadata.ckv_selected_indices,
                    attn_metadata.ckv_active_counts,
                    attn_metadata.dcp_rank_req_starts,
                    attn_metadata.dcp_rank_req_lens,
                    attn_metadata.dcp_local_cu_seq_lens,
                    attn_metadata.global_cache_seq_lens_per_req,
                )
            )
        )

    def _gather_full_ckv(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        gathered_buffer: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_full_ckv_dcp(attn_metadata, attn_metadata.num_actual_tokens):
            raise RuntimeError("full CKV gather called for an ineligible batch")
        # A packed FP8 cache may arrive through an E4M3 view. The gather copies
        # whole native records, including scales and the RoPE tail.
        if kv_cache.dtype == torch.float8_e4m3fn:
            kv_cache = kv_cache.view(torch.uint8)
        if not _is_native_ckv_source_layout(
            kv_cache,
            page_size=self._kernel_page_size,
            record_bytes=self._cache_record_bytes,
        ):
            raise ValueError(
                "CKV gather requires native "
                f"{self._cache_record_bytes}-byte records; "
                f"got shape={tuple(kv_cache.shape)}, stride={kv_cache.stride()}"
            )
        capacity_rows = self.dcp_world_size * self._ckv_local_capacity
        if self._uses_kvarn_cache:
            # Tile-stride arena: staging span, then one tile per group tokens.
            expected_gathered_shape = (self._ckv_gather_bytes(),)
        else:
            expected_gathered_shape = (capacity_rows, self._cache_record_bytes)
        if tuple(gathered_buffer.shape) != expected_gathered_shape:
            raise RuntimeError("CKV gathered workspace has an invalid shape")

        assert attn_metadata.dcp_local_cu_seq_lens is not None
        local_tokens = attn_metadata.dcp_local_total_tokens
        padded_tokens = attn_metadata.dcp_padded_total_tokens
        group = get_dcp_group()
        if self._uses_kvarn_cache:
            staging = gathered_buffer[
                : capacity_rows * self._cache_record_bytes
            ].view(capacity_rows, self._cache_record_bytes)
        else:
            staging = gathered_buffer
        # NCCL in-place all-gather reads each rank's contribution from its
        # receive slice. The stride is this call's padded count, not capacity.
        local_buffer = staging.narrow(
            0, group.rank_in_group * padded_tokens, padded_tokens
        )
        if local_tokens:
            ops.cp_gather_cache(
                src_cache=kv_cache,
                dst=local_buffer[:local_tokens],
                block_table=attn_metadata.block_table,
                cu_seq_lens=attn_metadata.dcp_local_cu_seq_lens,
                batch_size=attn_metadata.num_reqs,
            )
        if local_tokens < padded_tokens:
            local_buffer[local_tokens:padded_tokens].zero_()
        if (
            os.environ.get("KVARN_TAIL_QPROBE") == "1"
            and attn_metadata.num_actual_tokens > 1000
        ):
            try:
                _gl_seq = globals().get("_KVAR_DBG_SEQ", 0)
                if _gl_seq <= 12:
                    torch.cuda.synchronize()
                    _gl_rows = list(range(760, 790))
                    _bt = attn_metadata.block_table[: attn_metadata.num_reqs, :8]
                    _cu = attn_metadata.dcp_local_cu_seq_lens[
                        : attn_metadata.num_reqs + 1
                    ]
                    _kcache_u8 = (
                        kv_cache.view(torch.uint8)
                        if kv_cache.dtype == torch.float8_e4m3fn
                        else kv_cache
                    )
                    _psz = _kcache_u8.shape[1]
                    _ref = {}
                    for _r in _gl_rows:
                        if _r >= int(_cu[1]):
                            continue
                        _pg = int(_bt[0, _r // _psz])
                        _ref[_r] = (
                            _kcache_u8[_pg, _r % _psz].cpu().clone()
                            if _pg >= 0
                            else None
                        )
                    _kvarn_dbg_save(
                        {
                            "rank": group.rank_in_group,
                            "seq": _gl_seq,
                            "rows_idx": _gl_rows,
                            "rows": local_buffer[_gl_rows].cpu().contiguous(),
                            "local_tokens": int(local_tokens),
                            "padded": int(padded_tokens),
                            "block_table_head": _bt.cpu(),
                            "cu_seq_lens": _cu.cpu(),
                            "ref_rows": _ref,
                            "pool_shape": tuple(_kcache_u8.shape),
                        },
                        f"/home/js/kvarn-probes/kvarn-stage{_gl_seq}-postgatherlocal-r{group.rank_in_group}.pt",
                    )
            except Exception as _e:
                logger.warning("stage postgatherlocal probe failed %s", _e)
        if os.environ.get("KVARN_TAIL_DEBUG") and (
            50_000_000 < gathered_buffer.numel() < 100_000_000
            and not os.path.exists(f"/home/js/kvarn-probes/kvarn-prexchg-{os.getpid()}.pt")
        ):
            # Split gather-vs-exchange: the dense gather's output before the
            # NCCL all-gather. LOGICAL rows of the tile-strided arena.
            try:
                _pr = [760 + i for i in range(20)]
                _lb = local_buffer.reshape(-1)
                _mine = torch.stack(
                    [_lb[r * 536: r * 536 + 536] for r in _pr]
                ).contiguous()
                _kvarn_dbg_save(
                    {
                        "rank": group.rank_in_group,
                        "pid": os.getpid(),
                        "rows": _mine.cpu(),
                        "local_tokens": int(local_tokens),
                        "padded": int(padded_tokens),
                    },
                    f"/home/js/kvarn-probes/kvarn-prexchg-{os.getpid()}.pt",
                )
            except Exception as _e:
                logger.warning("prexchg probe failed %s", _e)
        if os.environ.get("KVARN_GATHER_DEBUG") == "1" and local_tokens:
            # Torch-side reference gather: pool rows via block_table, in
            # local order. Diffs against what cp_gather_cache produced.
            bt = attn_metadata.block_table[: attn_metadata.num_reqs]
            cu = attn_metadata.dcp_local_cu_seq_lens[: attn_metadata.num_reqs + 1]
            ref_rows = []
            for req in range(attn_metadata.num_reqs):
                beg, end = int(cu[req]), int(cu[req + 1])
                for j in range(beg, end):
                    pos = j - beg
                    lb = pos // int(self._kernel_page_size)
                    off = pos - lb * int(self._kernel_page_size)
                    ref_rows.append(kv_cache[int(bt[req, lb]), off, :])
            ref = _t_stack(ref_rows)
            got = local_buffer[:local_tokens]
            if ref.shape[0] == got.shape[0]:
                diff = (ref.to(torch.uint8) != got).any(dim=1)
                # Discriminating check: pool rows at the rank's TRUE owned
                # global positions (interleave-aware) vs local positions.
                itl = int(attn_metadata.cp_kv_cache_interleave_size)
                ws = self.dcp_world_size
                rank = group.rank_in_group
                total = (
                    int(attn_metadata.global_cache_seq_lens_per_req[0])
                    if attn_metadata.global_cache_seq_lens_per_req is not None
                    else 523
                )
                own = [t for t in range(total) if (t // itl) % ws == rank]
                own = own[:local_tokens]
                own_rows = []
                for pos in own:
                    lb = pos // int(self._kernel_page_size)
                    own_rows.append(
                        kv_cache[int(bt[0, lb]), pos - lb * int(self._kernel_page_size), :]
                    )
                ref_own = _t_stack(own_rows)
                d_own = (ref_own.to(torch.uint8) != got).any(dim=1)
                pz = [
                    int((kv_cache[int(bt[0, p // int(self._kernel_page_size)]),
                                  p % int(self._kernel_page_size)] == 0).all())
                    for p in range(min(24, local_tokens))
                ]
                print(
                    "[kvarn-gather-dbg] cp_gather: rows=%d mismatched=%d "
                    "maxbyte=%d own_global_mismatched=%d "
                    "pool_local_pos_zero[:24]=%s" % (
                        local_tokens, int(diff.sum()),
                        int((ref.to(torch.int16) - got.to(torch.int16)).abs().max()),
                        int(d_own.sum()), pz,
                    ),
                    flush=True,
                )
        _dbg_xchg = None
        if os.environ.get("KVARN_GATHER_DEBUG") == "1":
            lb = local_buffer[:padded_tokens].view(-1)
            _dbg_xchg = torch.stack([
                lb.to(torch.int64).sum(),
                (lb.to(torch.int64) * torch.arange(1, lb.numel() + 1, device=lb.device)).sum(),
            ])
            _dbg_all = torch.zeros(
                (self.dcp_world_size, 2), dtype=torch.int64, device=lb.device
            )
            _dcp_all_gather_current_stream(
                group,
                _dbg_xchg.view(-1),
                _dbg_all.view(-1),
            )
        _dcp_all_gather_current_stream(
            group,
            local_buffer[:padded_tokens].view(-1),
            staging[: self.dcp_world_size * padded_tokens].view(-1),
        )
        _kvarn_force_sync()
        if self._kvarn_tail_local_tokens > 0:
            # KVarN precision tail: rows stored as fp8 (positions >= ts)
            # STAY fp8 in the gathered copy — the extend tail launch reads
            # them through the fp8 arm and merges with the body launch
            # (design doc §10.4). The old throwaway-copy demote made the
            # extend attention all-kvarn, which is why prefill-scored KLD
            # never observed the tail. Debug: count surviving fp8 rows.
            ts = getattr(attn_metadata, "kvarn_tail_start", None)
            if ts is not None and os.environ.get("KVARN_TAIL_DEBUG") and (
                not torch.cuda.is_current_stream_capturing()
            ):
                rows_all = gathered_buffer[
                    : self.dcp_world_size * padded_tokens
                ].reshape(-1, self._cache_record_bytes)
                n_fp8_dense = int(
                    (rows_all[:, 520:528] != 0).any(-1).sum().item()
                )
                logger.warning(
                    "KVARN_TAIL_DEBUG gather: rows=%d fp8_rows=%d "
                    "ts=%s local_total=%d",
                    int(rows_all.shape[0]),
                    n_fp8_dense,
                    ts[: attn_metadata.num_reqs].tolist(),
                    int(attn_metadata.dcp_local_total_tokens),
                )
        if _dbg_xchg is not None:
            full = staging[: self.dcp_world_size * padded_tokens].view(-1)
            row = full.to(torch.int64)
            chk = torch.stack([
                row.sum(), (row * torch.arange(1, row.numel() + 1, device=row.device)).sum(),
            ])
            ok0 = int(chk[0] == _dbg_all[group.rank_in_group][0])
            ok1 = int(chk[1] == _dbg_all[group.rank_in_group][1])
            other = 1 - group.rank_in_group
            whole = staging[: self.dcp_world_size * padded_tokens].reshape(
                self.dcp_world_size, padded_tokens, -1
            ).to(torch.int64)
            own_ok = int((whole[group.rank_in_group].sum() == _dbg_xchg[0]))
            print(
                "[kvarn-gather-dbg] xchg rank=%d own_sum_ok=%d own_wsum_ok=%d "
                "local_sum=%d post_sum=%d remote_sum=%d expect_remote=%d" % (
                    group.rank_in_group, ok0, ok1, int(_dbg_xchg[0]),
                    int(whole[group.rank_in_group].sum()),
                    int(whole[other].sum()), int(_dbg_all[other][0]),
                ),
                flush=True,
            )
        if self._uses_kvarn_cache:
            # Mirror the pool's tile geometry into the arena: the decode
            # kernel addresses block*stride + tile*tile_stride + row*record
            # from plan constants; with page == tile for the gathered cache
            # every selected row lands at its true record offset.
            layout = self._kvarn_layout
            tile_tokens = int(layout.group)
            record = self._cache_record_bytes
            tile_stride = int(layout.tile_stride_bytes)
            used_rows = self.dcp_world_size * padded_tokens
            tiles = used_rows // tile_tokens
            staging_bytes = capacity_rows * record
            arena = gathered_buffer[staging_bytes:].as_strided(
                (tiles, tile_tokens, record), (tile_stride, record, 1)
            )
            if os.environ.get("KVARN_GATHER_DEBUG") == "1":
                _sh = staging[:12]
                print(
                    "[kvarn-gather-dbg] staging-head zeros=%d/%d "
                    "staging_ptr=%#x arena_ptr=%#x gbuf_ptr=%#x "
                    "cap_rows=%d padded=%d local=%d"
                    % (
                        int((_sh == 0).all(dim=1).sum()), _sh.shape[0],
                        staging.data_ptr(), arena.data_ptr(),
                        gathered_buffer.data_ptr(), capacity_rows,
                        padded_tokens, local_tokens,
                    ),
                    flush=True,
                )
                _stg_snap = staging[:2048].clone()
                self._dbg_stg_snap = _stg_snap
            arena.view(tiles, tile_tokens * record).copy_(
                staging[:used_rows].view(tiles, tile_tokens * record)
            )
            _kvarn_force_sync()
            if (
                os.environ.get("KVARN_TAIL_DEBUG")
                and not os.path.exists(
                    f"/home/js/kvarn-probes/kvarn-stg-dump-r{group.rank_in_group}.pt"
                )
            ):
                _ts_g = getattr(attn_metadata, "kvarn_tail_start", None)
                if _ts_g is not None and int(_ts_g.max()) > 0:
                    try:
                        _probe = list(range(760, 780))
                        _kvarn_dbg_save(
                            {
                                "rank": group.rank_in_group,
                                "pid": os.getpid(),
                                "arena_ptr": int(arena.data_ptr()),
                                "gbuf_ptr": int(gathered_buffer.data_ptr()),
                                "gbuf_numel": int(gathered_buffer.numel()),
                                "padded": int(padded_tokens),
                                "local": int(local_tokens),
                                "ts": _ts_g[: attn_metadata.num_reqs].cpu(),
                                "staging_rows": staging[_probe]
                                .cpu(),
                                "remote_staging_rows": staging[
                                    padded_tokens + _probe[0] : padded_tokens
                                    + _probe[-1]
                                    + 1
                                ]
                                .cpu(),
                                "arena_rows": arena.reshape(-1, record)[_probe].cpu(),
                                "tile_stride": tile_stride,
                                "record": record,
                            },
                            f"/home/js/kvarn-probes/kvarn-stg-dump-r{group.rank_in_group}.pt",
                        )
                        logger.warning("KVARN_TAIL_DEBUG: staging dump saved")
                    except Exception as _e:
                        logger.warning("KVARN_TAIL_DEBUG: staging dump failed %s", _e)
            if os.environ.get("KVARN_GATHER_DEBUG") == "1":
                self._dbg_postcopy = arena.reshape(-1, record)[:24].clone()
            # The per-tile tail sits AFTER the tile's records (tile_base +
            # tile_tokens*record .. +tile_stride); zero it every gather so
            # stale workspace bytes never dequantize into garbage scales.
            gathered_buffer[
                staging_bytes + tile_tokens * record:
            ].as_strided(
                (tiles, tile_stride - tile_tokens * record),
                (tile_stride, 1),
            ).zero_()
            if (
                os.environ.get("KVARN_TAIL_DEBUG")
                and attn_metadata.num_actual_tokens > 1000
                and arena.shape[0] * 256 >= 2048
            ):
                try:
                    _pz_seq = globals().get("_KVAR_DBG_SEQ", 0)
                    if _pz_seq <= 12:
                        torch.cuda.synchronize()
                        _ar2 = arena.reshape(-1, record)
                        _rows = list(range(760, 790)) + list(range(2810, 2830))
                        _kvarn_dbg_save(
                            {
                                "rank": group.rank_in_group,
                                "seq": _pz_seq,
                                "rows_idx": _rows,
                                "rows": _ar2[_rows].cpu().contiguous(),
                            },
                            f"/home/js/kvarn-probes/kvarn-stage{_pz_seq}-postgather-r{group.rank_in_group}.pt",
                        )
                except Exception as _e:
                    logger.warning("stage postgather probe failed %s", _e)
            return arena
        return gathered_buffer.view(
            -1, self._kernel_page_size, self._cache_record_bytes
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del layer
        cache_page_size = int(kv_c_and_k_pe_cache.shape[1])
        metadata_page_size = int(attn_metadata.block_size)
        if self._is_glm_next and (
            cache_page_size != self._kernel_page_size
            or metadata_page_size != self._kernel_page_size
        ):
            raise RuntimeError(
                "B12X GLM5Next page geometry does not match the bound plan: "
                f"cache={cache_page_size}, metadata={metadata_page_size}, "
                f"plan={self._kernel_page_size}"
            )
        num_tokens = int(q[0].shape[0] if isinstance(q, tuple) else q.shape[0])
        plan_key = self._plan_key(attn_metadata, num_tokens)
        plan = self._plan(plan_key)
        use_ckv_gather = plan_key[0] == "ckv_extend"
        if use_ckv_gather:
            logger.info_once("Using full-CKV gather for B12X DCP prefill")
        input_num_heads = self.num_heads if use_ckv_gather else self._input_num_heads
        # Plans and scratch are declared at the DCP-global padded head width;
        # the gather route binds the same 64-head scratch with its 32 real
        # heads zero-padded (the MG kernel drops the pad, like TP6 padding).
        kernel_num_heads = self._kernel_num_heads
        workspace_specs = self._workspace_specs(
            input_num_heads=kernel_num_heads,
            include_ckv=use_ckv_gather,
        )
        workspaces = current_workspace_manager().get_simultaneous(*workspace_specs)
        q_buffer = workspaces[0]
        scratch = workspaces[1]

        q_all = q_buffer[:num_tokens]
        q_real = q_all[:, :input_num_heads]
        if os.environ.get("KVARN_TAIL_QPROBE") == "1" and plan_key[0] == "ckv_extend" and num_tokens > 1000:
            # One-shot per-rank capture: the MODEL's q before staging and the
            # workspace q after it, plus segment pointers. Distinguishes
            # model-side NaN from workspace-side corruption.
            _qp = f"/home/js/kvarn-probes/kvarn-qprobe-r{self.dcp_rank}.pt"
            if not os.path.exists(_qp):
                try:
                    _src = q[0] if isinstance(q, tuple) else q
                    torch.cuda.synchronize()
                    _model_q = _src[-8:].detach().clone().cpu()
                    _qa_ptr = int(q_all.data_ptr())
                    _qs_ptr = int(q_buffer.data_ptr())
                    _sc_ptr = int(scratch.data_ptr()) if scratch is not None else -1
                    _ck_ptr = int(workspaces[2].data_ptr()) if len(workspaces) > 2 else -1
                except Exception as _e:
                    logger.warning("KVARN_TAIL_QPROBE pre-stage failed %s", _e)
        if isinstance(q, tuple):
            q_nope, q_pe = q
            if int(q_pe.shape[-1]) == 0:
                q_real.copy_(q_nope)
            else:
                ops.concat_mla_q(q_nope, q_pe, q_real)
        else:
            exact_workspace_alias = (
                tuple(q.shape) == tuple(q_real.shape)
                and tuple(q.stride()) == tuple(q_real.stride())
                and q.dtype == q_real.dtype
                and q.device == q_real.device
                and q.untyped_storage().data_ptr()
                == q_real.untyped_storage().data_ptr()
                and q.storage_offset() == q_real.storage_offset()
            )
            if not exact_workspace_alias:
                q_real.copy_(q)
        if kernel_num_heads != input_num_heads:
            q_all[:, input_num_heads:].zero_()

        if os.environ.get("KVARN_TAIL_QPROBE") == "1" and plan_key[0] == "ckv_extend" and num_tokens > 1000:
            _qp = f"/home/js/kvarn-probes/kvarn-qprobe-r{self.dcp_rank}.pt"
            if not os.path.exists(_qp):
                try:
                    torch.cuda.synchronize()
                    _staged = q_all[-8:].detach().clone().cpu()
                    with open(_qp, "wb") as _f:
                        _mq = _model_q.contiguous()
                        _st = _staged.contiguous()
                        import struct as _struct
                        _f.write(_struct.pack("iiiiii",
                                              _mq.shape[0], _mq.shape[1], _mq.shape[2],
                                              _st.shape[0], _st.shape[1], _st.shape[2]))
                        _f.write(_mq.view(torch.uint16).numpy().tobytes())
                        _f.write(_st.view(torch.uint16).numpy().tobytes())
                    logger.warning("KVARN_TAIL_QPROBE saved r%d ptrs q_all=%s qbuf=%s scratch=%s ckv=%s",
                                   self.dcp_rank, _qa_ptr, _qs_ptr, _sc_ptr, _ck_ptr)
                except Exception as _e:
                    logger.warning("KVARN_TAIL_QPROBE save failed %s", _e)

        if int(q.shape[1] if not isinstance(q, tuple) else q[0].shape[1]) != (
            input_num_heads
        ):
            raise ValueError(
                "B12X sparse MLA query heads do not match the planned head "
                f"count: {q_real.shape[1]} != {input_num_heads}."
            )

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]
        ts = getattr(attn_metadata, "kvarn_tail_start", None)
        tail_active = (
            self._uses_kvarn_cache
            and self._kvarn_tail_local_tokens > 0
            and ts is not None
        )
        if tail_active:
            # Precision tail: drop body selections inside the tail window
            # (they are attended densely by the tail segment instead).
            # topk_indices are request-relative GLOBAL positions; the
            # manager-owned local ts converts to a global boundary (exact
            # when ts is a multiple of the interleave size, which block
            # alignment guarantees).
            from vllm.v1.attention.backends.mla.kvarn_tail import (
                kvarn_tail_boundary,
                merge_tail_lse,
            )

            num_reqs = int(attn_metadata.num_reqs)
            req_ids = attn_metadata.req_id_per_token[:num_tokens]
            # ts is already a GLOBAL request-relative position; topk
            # candidates are global positions too.
            boundary_global = kvarn_tail_boundary(ts[:num_reqs], req_ids)
            torch.where(
                topk_indices >= boundary_global[:, None],
                torch.full((), -1, dtype=topk_indices.dtype, device=topk_indices.device),
                topk_indices,
                out=topk_indices,
            )
        kv_cache_for_run = kv_c_and_k_pe_cache
        if use_ckv_gather:
            gathered_buffer = workspaces[2]
            kv_cache_for_run = self._gather_full_ckv(
                kv_c_and_k_pe_cache,
                attn_metadata,
                gathered_buffer,
            )
            if os.environ.get("KVARN_GATHER_DEBUG") == "1":
                print(
                    "[kvarn-gather-dbg] pool shape=%s strides=%s dtype=%s"
                    % (
                        tuple(kv_c_and_k_pe_cache.shape),
                        tuple(kv_c_and_k_pe_cache.stride()),
                        kv_c_and_k_pe_cache.dtype,
                    ),
                    flush=True,
                )
            assert attn_metadata.ckv_selected_indices is not None
            assert attn_metadata.ckv_active_counts is not None
            assert attn_metadata.dcp_rank_req_starts is not None
            assert attn_metadata.dcp_rank_req_lens is not None
            selected_indices = attn_metadata.ckv_selected_indices[
                :num_tokens, : topk_indices.shape[1]
            ]
            active_counts = attn_metadata.ckv_active_counts[:num_tokens]
            _map_global_topk_to_gathered_ckv(
                attn_metadata.req_id_per_token[:num_tokens],
                topk_indices,
                attn_metadata.dcp_rank_req_starts,
                attn_metadata.dcp_rank_req_lens,
                selected_indices,
                active_counts,
                dcp_size=self.dcp_world_size,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                padded_rank_tokens=attn_metadata.dcp_padded_total_tokens,
            )
            assert attn_metadata.global_cache_seq_lens_per_req is not None
            cache_seq_lens = _global_causal_lens_for_ckv_gather(
                attn_metadata.global_cache_seq_lens_per_req,
                attn_metadata.query_start_loc,
                attn_metadata.req_id_per_token,
                num_tokens,
            ).contiguous()
            torch.minimum(active_counts, cache_seq_lens, out=active_counts)
            _mask_page_table_after_nsa_len(selected_indices, active_counts)
            _dbg_postcopy = getattr(self, "_dbg_postcopy", None)
            if _dbg_postcopy is not None:
                _ar0 = kv_cache_for_run.reshape(-1, 536)[:24]
                _z0 = (_ar0 == 0).all(dim=1)
                _d0 = (_ar0 != _dbg_postcopy).any(dim=1)
                print(
                    "[kvarn-gather-dbg] phase=post-map zero_rows=%d "
                    "changed_vs_postcopy=%d first_changed=%s"
                    % (int(_z0.sum()), int(_d0.sum()),
                       _d0.nonzero().flatten()[:6].tolist()),
                    flush=True,
                )
            if os.environ.get("KVARN_GATHER_DEBUG") == "1":
                _dbg_dump_gather_selection(
                    num_tokens, selected_indices, active_counts,
                    cache_seq_lens, attn_metadata, raw_topk=topk_indices,
                )
        else:
            physical_selection = (
                self._physical_selection_provider.get_b12x_physical_selection(
                    num_tokens=num_tokens,
                    num_prefills=int(attn_metadata.num_prefills),
                    num_decode_tokens=int(attn_metadata.num_decode_tokens),
                )
                if self._physical_selection_provider is not None
                else None
            )
            if physical_selection is not None:
                selected_indices, active_counts = physical_selection
            elif self.dcp_world_size > 1:
                block_stride_rows = _selected_index_block_stride_rows(
                    kv_c_and_k_pe_cache,
                    block_size=attn_metadata.block_size,
                )
                selected_indices, active_counts = triton_filter_and_convert_dcp_index(
                    attn_metadata.req_id_per_token[:num_tokens],
                    attn_metadata.block_table,
                    topk_indices,
                    dcp_size=self.dcp_world_size,
                    dcp_rank=self.dcp_rank,
                    cp_kv_cache_interleave_size=(
                        attn_metadata.cp_kv_cache_interleave_size
                    ),
                    BLOCK_SIZE=attn_metadata.block_size,
                    BLOCK_STRIDE_ROWS=block_stride_rows,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                    return_valid_counts=True,
                )
            elif not self._is_glm_next:
                selected_indices = topk_indices
                cache_seq_lens_per_token = attn_metadata.cache_seq_lens_per_token
                assert cache_seq_lens_per_token is not None
                active_counts = cache_seq_lens_per_token[:num_tokens]
            else:
                block_stride_rows = _selected_index_block_stride_rows(
                    kv_c_and_k_pe_cache,
                    block_size=attn_metadata.block_size,
                )
                selected_indices, active_counts = (
                    triton_convert_req_index_to_global_index(
                        attn_metadata.req_id_per_token[:num_tokens],
                        attn_metadata.block_table,
                        topk_indices,
                        BLOCK_SIZE=attn_metadata.block_size,
                        BLOCK_STRIDE_ROWS=block_stride_rows,
                        NUM_TOPK_TOKENS=topk_indices.shape[1],
                        return_valid_counts=True,
                        compact_valid_to_front=False,
                    )
                )

        if not use_ckv_gather:
            if self._is_glm_next:
                cache_seq_lens = attn_metadata.cache_seq_lens_per_token
                assert cache_seq_lens is not None
                cache_seq_lens = cache_seq_lens[:num_tokens].contiguous()
            else:
                assert attn_metadata.seq_lens is not None
                cache_seq_lens = attn_metadata.seq_lens[
                    : attn_metadata.num_reqs
                ].contiguous()
        if (
            os.environ.get("KVARN_GATHER_DEBUG") == "1"
            and num_tokens > 0
            and not torch.cuda.is_current_stream_capturing()
        ):
            _qh = q_all.float().sum(dim=(0, 2))
            _r0 = q_all[0].float().sum(dim=1)
            _qr = (
                q[0].float().sum(dim=1)
                if not isinstance(q, tuple) else q[0][0].float().sum(dim=1)
            )
            print(
                "[kvarn-gather-dbg] qsum route=%s rows=%d heads=%d "
                "first8=%s last8=%s total=%.6e row0_8=%s row0_all=%.6e "
                "raw0_8=%s raw0_all=%.6e"
                % (
                    plan_key[0], num_tokens, int(q_all.shape[1]),
                    [round(float(x), 3) for x in _qh[:8].tolist()],
                    [round(float(x), 3) for x in _qh[-8:].tolist()],
                    float(_qh.sum()),
                    [round(float(x), 3) for x in _r0[:8].tolist()],
                    float(_r0.sum()),
                    [round(float(x), 3) for x in _qr[:8].tolist()],
                    float(_qr.sum()),
                ),
                flush=True,
            )
        full_cache_lens = cache_seq_lens
        if tail_active and plan_key[0] != "decode":
            # The MG extend body is DENSE over cache_lengths: with fp8 tail
            # rows in the pool/arena it must stop at the tail boundary or it
            # dequantizes fp8 records as kvarn (the lock-loop signature).
            # The tail launch keeps the unclamped length as its window end.
            cache_seq_lens = torch.minimum(cache_seq_lens, boundary_global)
        binding = self._bind(
            plan,
            scratch=scratch,
            q=q_all,
            kv_cache=kv_cache_for_run,
            selected_indices=selected_indices,
            cache_lengths=cache_seq_lens,
            selected_lengths=active_counts,
        )
        if os.environ.get("KVARN_TAIL_WRITEMAP") == "1":
            # Write-map diagnostic: fill the arena with a per-row known
            # pattern before the body launch. The post-body dump then
            # identifies exactly which rows the kernel writes and with what.
            _ar2d = kv_cache_for_run.reshape(-1, self._cache_record_bytes)
            _rid = (
                torch.arange(_ar2d.shape[0], device=_ar2d.device, dtype=torch.uint8)
                .unsqueeze(1)
            )
            _ar2d.copy_(_rid)
        if os.environ.get("KVARN_TAIL_QPROBE") == "1" and plan_key[0] == "ckv_extend" and num_tokens > 1000:
            global _KVAR_DBG_SEQ
            try:
                _KVAR_DBG_SEQ
            except NameError:
                _KVAR_DBG_SEQ = 0
            _ql = _KVAR_DBG_SEQ
            if _ql <= 12:
                try:
                    torch.cuda.synchronize()
                    _ar_l = kv_cache_for_run.reshape(-1, self._cache_record_bytes)
                    _rl = list(range(760, 790)) + list(range(2810, 2830))
                    _kvarn_dbg_save({"rows": _ar_l[_rl].cpu().contiguous(),
                                "rows_idx": _rl},
                               f"/home/js/kvarn-probes/kvarn-layer{_ql}-prebody-r{self.dcp_rank}.pt")
                except Exception as _e:
                    logger.warning("layer prebody probe failed %s", _e)
        result = self._run(binding)
        # The extend kernel returns LSE even when decode does not request it.
        output, lse = result if isinstance(result, tuple) else (result, None)
        _kvarn_force_sync()
        if os.environ.get("KVARN_TAIL_QPROBE") == "1" and plan_key[0] == "ckv_extend" and num_tokens > 1000:
            try:
                torch.cuda.synchronize()
                _ar_l2 = kv_cache_for_run.reshape(-1, self._cache_record_bytes)
                _rl2 = list(range(760, 790)) + list(range(2810, 2830))
                _kvarn_dbg_save({"rows": _ar_l2[_rl2].cpu().contiguous(),
                            "rows_idx": _rl2},
                           f"/home/js/kvarn-probes/kvarn-layer{_ql}-postbody-r{self.dcp_rank}.pt")
                _ob = output.detach().clone().cpu()
                _kvarn_dbg_save({
                            "out_rows": _ob[1520:1570],
                            "out_nan_rows": [
                                1520 + i for i in range(_ob[1520:1570].shape[0])
                                if bool(torch.isnan(_ob[1520 + i].float()).any())
                            ],
                        },
                       f"/home/js/kvarn-probes/kvarn-layer{_ql}-bodyout-r{self.dcp_rank}.pt")
                _KVAR_DBG_SEQ = _ql + 1
            except Exception as _e:
                logger.warning("layer postbody probe failed %s", _e)
        if os.environ.get("KVARN_TAIL_QPROBE") == "1" and plan_key[0] == "ckv_extend" and num_tokens > 1000:
            _qp2 = f"/home/js/kvarn-probes/kvarn-qprobe-postbody-r{self.dcp_rank}.pt"
            if not os.path.exists(_qp2):
                try:
                    torch.cuda.synchronize()
                    _qpost = q_all[-8:].detach().clone().cpu().contiguous()
                    _ar_post = kv_cache_for_run.reshape(
                        -1, self._cache_record_bytes)
                    _rows = (list(range(0, 780)) + list(range(2048, 2830)))
                    _kvarn_dbg_save(
                        {
                            "pid": os.getpid(),
                            "rows_idx": _rows,
                            "rows": _ar_post[_rows].cpu(),
                        },
                        f"/home/js/kvarn-probes/kvarn-arena-postbody-{os.getpid()}.pt",
                    )
                except Exception as _e:
                    logger.warning("KVARN_TAIL_QPROBE arena-postbody failed %s", _e)
            if not os.path.exists(_qp2):
                try:
                    torch.cuda.synchronize()
                    _qpost = q_all[-8:].detach().clone().cpu().contiguous()
                    _rt = getattr(binding, "runtime", binding)
                    _views = {}
                    try:
                        for _k, _v in vars(_rt).items():
                            if torch.is_tensor(_v):
                                _views[_k] = (int(_v.data_ptr()), tuple(_v.shape), str(_v.dtype))
                            elif hasattr(_v, "data_ptr"):
                                _views[_k] = (int(_v.data_ptr()), None, str(type(_v)))
                    except Exception as _e2:
                        _views["err"] = str(_e2)
                    with open(_qp2, "wb") as _f:
                        import struct as _struct
                        _f.write(_struct.pack("iii", _qpost.shape[0], _qpost.shape[1],
                                              _qpost.shape[2]))
                        _f.write(_qpost.view(torch.uint16).numpy().tobytes())
                        _f.write(repr(_views).encode())
                    logger.warning("KVARN_TAIL_QPROBE post-body r%d views=%s", self.dcp_rank, _views)
                except Exception as _e:
                    logger.warning("KVARN_TAIL_QPROBE post-body failed %s", _e)
        if tail_active and not tail_skip_launch:
            # Second segment: the dense tail window through the fp8 arm,
            # then an exact base-e LSE re-merge of the two normalized
            # partial attentions (design doc §10.4).
            from vllm.v1.attention.backends.mla.kvarn_tail import (
                kvarn_tail_positions_to_slots,
            )

            n_local = self._kvarn_tail_local_tokens
            csl = attn_metadata.cache_seq_lens_per_token[:num_tokens]
            ts = getattr(attn_metadata, "kvarn_tail_start", None)
            if use_ckv_gather:
                # Gather route: global units throughout. The tail window is
                # [boundary_global, csl) in GLOBAL positions; the same
                # mapper the body uses converts them to gathered-arena rows.
                # The arena keeps these rows fp8 (no gathered demote), and
                # the body launch never sees them: the tail mask zeroed
                # their topk selections above.
                n_tail = self._kvarn_tail_tokens
                cols = torch.arange(n_tail, device=csl.device)
                pos = boundary_global[:, None] + cols[None, :]
                # Window end = the UNCLAMPED per-token S: the body stops at
                # the boundary, so the tail owns [boundary, S) -- including
                # the current chunk's own just-stored fp8 rows. Validating
                # against the clamped lens made the tail empty on chunk 2+
                # and dropped the context's last N tokens entirely.
                valid = pos < full_cache_lens[:, None]
                pos = torch.where(
                    valid, pos, torch.full((), -1, dtype=pos.dtype)
                ).to(torch.int32)
                tail_indices = torch.empty(
                    (num_tokens, n_tail),
                    dtype=torch.int32,
                    device=csl.device,
                )
                tail_lengths = torch.empty(
                    (num_tokens,), dtype=torch.int32, device=csl.device
                )
                _map_global_topk_to_gathered_ckv(
                    attn_metadata.req_id_per_token[:num_tokens],
                    pos,
                    attn_metadata.dcp_rank_req_starts,
                    attn_metadata.dcp_rank_req_lens,
                    tail_indices,
                    tail_lengths,
                    dcp_size=self.dcp_world_size,
                    cp_kv_cache_interleave_size=(
                        attn_metadata.cp_kv_cache_interleave_size
                    ),
                    padded_rank_tokens=attn_metadata.dcp_padded_total_tokens,
                )
                tail_plan = self._plan(
                    ("extend_tail_gather", self._max_tokens)
                )
            else:
                from vllm.v1.attention.backends.mla.kvarn_tail import (
                    kvarn_tail_local_row_start,
                )

                boundary_local = kvarn_tail_local_row_start(
                    ts[:num_reqs],
                    self.dcp_rank,
                    self.dcp_world_size,
                    int(attn_metadata.cp_kv_cache_interleave_size),
                )[req_ids.to(torch.int64)]
                # Per-token local sequence length: rows are causal tokens, so
                # each row's tail window ends at its OWN cache length (extend
                # rows within a chunk have distinct, increasing S -- a
                # request-max would let early rows attend future tokens).
                s_local = csl
                cols = torch.arange(n_local, device=csl.device)
                pos = boundary_local[:, None] + cols[None, :]
                valid = pos < s_local[:, None]
                pos = torch.where(valid, pos, torch.zeros_like(pos))
                tail_indices = kvarn_tail_positions_to_slots(
                    pos,
                    req_ids.unsqueeze(1).expand_as(pos),
                    attn_metadata.block_table,
                    attn_metadata.block_size,
                ).to(torch.int32)
                tail_indices = torch.where(
                    valid, tail_indices, torch.full((), -1, dtype=tail_indices.dtype, device=tail_indices.device)
                )
                tail_lengths = (s_local - boundary_local).clamp_(min=0).clamp_max_(n_local)
                tail_plan = self._plan(
                    ("decode_tail", num_tokens)
                    if plan_key[0] == "decode"
                    else ("extend_tail", self._max_tokens)
                )
            # A page-granular ts can sit at or above a row's S (the manager
            # publishes whole pages below S-N, so the fp8 window is empty
            # whenever the last partial page is kvarn; with N < page that is
            # most decode steps). The launch must stay shape-static and
            # capture-safe, but the kernel only skips candidates BEYOND
            # selected_lengths -- a -1 within the count is scored from stale
            # smem. So: launch a 1-candidate -1 dummy, then REPLACE the
            # merged partials with the body's for empty rows (branchless,
            # tensor-select). The body = the exact identity for those rows.
            tail_lens_real = tail_lengths.clone()
            tail_indices = torch.where(
                (tail_lengths > 0)[:, None],
                tail_indices,
                torch.full((), -1, dtype=tail_indices.dtype,
                           device=tail_indices.device),
            )
            tail_lengths = tail_lengths.clamp_min(1)
            if (
                num_tokens <= self._decode_max_rows
                and tail_indices.shape[-1]
                == self._kvarn_tail_indices_buffer.shape[-1]
            ):
                # Captured decode graphs replay the tail launch without
                # re-running this Python: bind the launch to persistent
                # buffers instead of this step's fresh temporaries (the
                # body's selection binds ckv_selected_indices_buffer for the
                # same reason). The copy kernels are part of the capture, so
                # replays recompute the temporaries from the live metadata
                # and land them at a capture-stable address.
                self._kvarn_tail_indices_buffer[:num_tokens].copy_(tail_indices)
                self._kvarn_tail_lengths_buffer[:num_tokens].copy_(tail_lengths)
                tail_indices = self._kvarn_tail_indices_buffer[:num_tokens]
                tail_lengths = self._kvarn_tail_lengths_buffer[:num_tokens]
                if os.environ.get("KVARN_TAIL_HASH") == "1":
                    global _KVTAIL_HASH_BUF
                    if _KVTAIL_HASH_BUF is None:
                        if torch.cuda.is_current_stream_capturing():
                            raise RuntimeError(
                                "KVTAILHASH buffer must be allocated eagerly"
                            )
                        _KVTAIL_HASH_BUF = torch.zeros(
                            4, dtype=torch.int64, device=tail_indices.device
                        )
                    _KVTAIL_HASH_BUF[0].copy_(
                        tail_indices.to(torch.int64).sum()
                    )
                    _KVTAIL_HASH_BUF[1].copy_(
                        tail_lengths.to(torch.int64).sum()
                    )
                    _KVTAIL_HASH_BUF[2].copy_(csl.to(torch.int64).sum())
                    _KVTAIL_HASH_BUF[3] += 1
            if os.environ.get("KVARN_TAIL_PRIVATE_KV") == "1":
                # Decisive fork test: bind the tail launch to a private copy
                # of the arena (same shape/stride). If the canary passes, the
                # arena was being corrupted at its address by a stray write.
                _tkv = getattr(self, "_tail_private_kv", None)
                if _tkv is None or _tkv.shape != kv_cache_for_run.shape:
                    _tkv = torch.empty_strided(
                        kv_cache_for_run.shape,
                        kv_cache_for_run.stride(),
                        dtype=kv_cache_for_run.dtype,
                        device=kv_cache_for_run.device,
                    )
                    self._tail_private_kv = _tkv
                _tkv.copy_(kv_cache_for_run)
                _tail_kv = _tkv
            else:
                _tail_kv = kv_cache_for_run
            tail_binding = self._bind(
                tail_plan,
                scratch=scratch,
                q=q_all,
                kv_cache=_tail_kv,
                selected_indices=tail_indices,
                cache_lengths=full_cache_lens,
                selected_lengths=tail_lengths,
            )
            if os.environ.get("KVARN_TAIL_QPROBE") == "1" and plan_key[0] == "ckv_extend" and num_tokens > 1000:
                _tp = f"/home/js/kvarn-probes/kvarn-tailviews-r{self.dcp_rank}.pt"
                if not os.path.exists(_tp):
                    try:
                        _rt = getattr(tail_binding, "runtime", tail_binding)
                        _views = {}
                        for _k, _v in vars(_rt).items():
                            if torch.is_tensor(_v):
                                _views[_k] = (int(_v.data_ptr()), tuple(_v.shape),
                                              tuple(_v.stride()), str(_v.dtype))
                        _views["_arena_ptr"] = int(_tail_kv.data_ptr())
                        with open(_tp, "w") as _f:
                            _f.write(repr(_views))
                        logger.warning("KVARN_TAIL_QPROBE tailviews r%d: %s", self.dcp_rank, _views)
                    except Exception as _e:
                        logger.warning("KVARN_TAIL_QPROBE tailviews failed %s", _e)
            # The body launch's output may alias the shared kernel scratch;
            # keep a private copy before the tail launch reuses it.
            body_output = output.clone()
            body_lse = lse.clone() if lse is not None else None
            if (
                os.environ.get("KVARN_TAIL_DEBUG")
                and not torch.cuda.is_current_stream_capturing()
                and plan_key[0] == "ckv_extend"
                and tail_lens_real.max().item() > 0
                and not os.path.exists("/home/js/kvarn-probes/kvarn-mid-probe.pt")
            ):
                try:
                    _ar = kv_cache_for_run.reshape(-1, self._cache_record_bytes)
                    _kvarn_dbg_save(
                        {
                            "rank": self.dcp_rank,
                            "arena_full": _ar.cpu(),
                            "pid": os.getpid(),
                            "arena_ptr": int(kv_cache_for_run.data_ptr()),
                        },
                        f"/home/js/kvarn-probes/kvarn-mid-{os.getpid()}.pt",
                    )
                    logger.warning("KVARN_TAIL_DEBUG: mid probe saved")
                except Exception as _e:
                    logger.warning("KVARN_TAIL_DEBUG: mid probe failed %s", _e)
            _pre_tail = None
            if (
                os.environ.get("KVARN_TAIL_DEBUG")
                and num_tokens > 1000
                and globals().get("_KVAR_DBG_SEQ", 0) <= 12
            ):
                try:
                    torch.cuda.synchronize()
                    _pt_seq = globals().get("_KVAR_DBG_SEQ", 0)
                    _ar_t = kv_cache_for_run.reshape(-1, self._cache_record_bytes)
                    _rows_t = list(range(1530, 1570)) + list(range(2810, 2830))
                    _pre_tail = _ar_t[_rows_t].cpu().contiguous()
                    _kvarn_dbg_save(
                        {"pid": os.getpid(), "rows_idx": _rows_t,
                         "rows": _pre_tail},
                        f"/home/js/kvarn-probes/kvarn-stage{_pt_seq}-pretail-r{self.dcp_rank}.pt",
                    )
                except Exception as _e:
                    logger.warning("pretail probe failed %s", _e)
            tail_result = self._run(tail_binding)
            tail_output, tail_lse = (
                tail_result if isinstance(tail_result, tuple) else (tail_result, None)
            )
            _kvarn_force_sync()
            if (
                os.environ.get("KVARN_TAIL_DEBUG")
                and not torch.cuda.is_current_stream_capturing()
                and plan_key[0] == "decode"
                and tail_lens_real.max().item() > 100
                and not os.path.exists(
                    f"/home/js/kvarn-probes/kvarn-decodecap-r{self.dcp_rank}.pt"
                )
            ):
                # One-shot decode-tail capture (MTP bug hunt): the decode
                # path runs in FULL CUDA graphs, so this only fires under
                # --enforce-eager. Full launch state for the offline replay.
                try:
                    _ar_d = kv_cache_for_run.reshape(-1, self._cache_record_bytes)
                    _cap = {
                        "rank": self.dcp_rank,
                        "pool_rows": _ar_d.shape[0],
                        "q": q_all[:num_tokens].cpu(),
                        "tail_indices": tail_indices.cpu(),
                        "tail_lengths": tail_lengths.cpu(),
                        "tail_lens_real": tail_lens_real.cpu(),
                        "cache_lengths": full_cache_lens[:num_tokens].cpu(),
                        "csl": csl[:num_tokens].cpu()
                        if csl.shape[0] >= num_tokens else csl.cpu(),
                        "ts": ts[:num_reqs].cpu(),
                        "req_ids": req_ids.cpu(),
                        "block_table": attn_metadata.block_table.cpu(),
                        "block_size": int(attn_metadata.block_size),
                        "tail_out": tail_output.cpu(),
                        "tail_lse": tail_lse.cpu()
                        if tail_lse is not None else None,
                        "body_out": body_output.cpu(),
                        "body_lse": body_lse.cpu()
                        if body_lse is not None else None,
                    }
                    # Save the metadata FIRST (safe ops only): if the record
                    # gather below trips a device assert on a garbage slot
                    # id, the launch state is already on disk.
                    _kvarn_dbg_save(
                        _cap,
                        f"/home/js/kvarn-probes/kvarn-decodecap-r{self.dcp_rank}.pt",
                    )
                    _cap["tail_rows"] = _ar_d[
                        tail_indices.clamp(0, _ar_d.shape[0] - 1)
                    ].cpu()
                    _kvarn_dbg_save(
                        _cap,
                        f"/home/js/kvarn-probes/kvarn-decodecap-r{self.dcp_rank}.pt",
                    )
                    logger.warning("KVARN_TAIL_DEBUG: decodecap saved r%d", self.dcp_rank)
                except Exception as _e:
                    logger.warning("decodecap failed %s", _e)
            if (
                os.environ.get("KVARN_TAIL_DEBUG")
                and _pre_tail is not None
            ):
                try:
                    torch.cuda.synchronize()
                    _pt2_seq = globals().get("_KVAR_DBG_SEQ", 0)
                    _ar_t2 = kv_cache_for_run.reshape(-1, self._cache_record_bytes)
                    _rows_t2 = list(range(1530, 1570)) + list(range(2810, 2830))
                    _kvarn_dbg_save(
                        {"pid": os.getpid(), "rows_idx": _rows_t2,
                         "rows": _ar_t2[_rows_t2].cpu().contiguous()},
                        f"/home/js/kvarn-probes/kvarn-stage{_pt2_seq}-posttail-r{self.dcp_rank}.pt",
                    )
                except Exception as _e:
                    logger.warning("posttail probe failed %s", _e)
            if (
                os.environ.get("KVARN_TAIL_DEBUG")
                and not torch.cuda.is_current_stream_capturing()
                and plan_key[0] == "ckv_extend"
                and tail_lens_real.max().item() > 0
            ):
                self._dbg_ext_count = getattr(self, "_dbg_ext_count", 0) + 1
            if (
                os.environ.get("KVARN_TAIL_DEBUG")
                and not torch.cuda.is_current_stream_capturing()
                and plan_key[0] == "ckv_extend"
                and tail_lens_real.max().item() > 0
                and globals().get("_KVAR_DBG_SEQ", 0) == 7
            ):
                try:
                    _rec = self._cache_record_bytes
                    _arena = kv_cache_for_run.reshape(-1, _rec)
                    _pr = slice(1520, 1570)
                    _si = selected_indices[_pr]
                    _ti = tail_indices[_pr]
                    _kvarn_dbg_save(
                        {
                            "q": q_all[_pr].cpu(),
                            "tail_indices": _ti.cpu(),
                            "tail_lengths": tail_lengths[_pr].cpu(),
                            "tail_lens_real": tail_lens_real[_pr].cpu(),
                            "cache_lengths": full_cache_lens[_pr].cpu(),
                            "boundary": boundary_global[_pr].cpu(),
                            "sel_indices": _si.cpu(),
                            "active_counts": active_counts[_pr].cpu(),
                            "tail_rows": _arena[
                                _ti.clamp(min=0).reshape(-1).to(torch.int64)
                            ].cpu(),
                            "body_rows": _arena[
                                _si.clamp(min=0).reshape(-1).to(torch.int64)
                            ].cpu(),
                            "plan_key": str(plan_key[0]),
                            "tail_out": tail_output[_pr].cpu(),
                            "tail_lse": tail_lse[_pr].cpu(),
                            "body_out": body_output[_pr].cpu(),
                            "body_lse": body_lse[_pr].cpu(),
                            "premerge_out": output[_pr].cpu(),
                            "dbg_ext_count": self._dbg_ext_count,
                        },
                        "/home/js/kvarn-probes/kvarn-extcap-L6tail-r0.pt",
                    )
                    logger.warning(
                        "KVARN_TAIL_DEBUG: ext capture %d saved",
                        self._dbg_ext_count,
                    )
                except Exception as _e:
                    logger.warning("KVARN_TAIL_DEBUG: capture failed %s", _e)
            assert body_lse is not None and tail_lse is not None
            merged = merge_tail_lse(tail_output, tail_lse, body_output, body_lse)
            empty_row = (tail_lens_real == 0)[:, None, None]
            output = torch.where(empty_row, body_output, merged)
            _kvarn_force_sync()
            if (
                os.environ.get("KVARN_TAIL_DEBUG")
                and globals().get("_KVAR_DBG_SEQ", 0) == 7
            ):
                try:
                    torch.cuda.synchronize()
                    _mg = output.detach().clone().float().cpu()
                    _to = tail_output.detach().float().cpu()
                    _tl = tail_lse.detach().float().cpu()
                    _mr = slice(1520, 1570)
                    _kvarn_dbg_save(
                        {
                            "merged_rows": _mg[_mr],
                            "tail_lens_real": tail_lens_real[_mr].cpu(),
                            "tail_out_bad_rows": [
                                1520 + i for i in range(50)
                                if bool(torch.isnan(_to[1520 + i]).any()
                                        or torch.isinf(_to[1520 + i]).any())
                            ],
                            "tail_lse_bad_rows": [
                                1520 + i for i in range(50)
                                if bool(torch.isnan(_tl[1520 + i]).any()
                                        or torch.isinf(_tl[1520 + i]).any())
                            ],
                            "merged_bad_rows": [
                                1520 + i for i in range(50)
                                if bool(torch.isnan(_mg[1520 + i]).any()
                                        or torch.isinf(_mg[1520 + i]).any())
                            ],
                        },
                        f"/home/js/kvarn-probes/kvarn-mergecap-L6tail-r{self.dcp_rank}.pt",
                    )
                except Exception as _e:
                    logger.warning("merge cap failed %s", _e)
            lse = (
                torch.where(
                    (tail_lens_real == 0)[:, None],
                    body_lse.float(),
                    torch.logaddexp(tail_lse.float(), body_lse.float()),
                ).to(tail_lse.dtype)
                if self.need_to_return_lse_for_decode
                else None
            )
        if kernel_num_heads != input_num_heads:
            # Fresh packed tensors: for one row, contiguous() would keep the
            # padded row stride, which the DCP collectives reject.
            output = _packed_heads(output, input_num_heads)
            if lse is not None:
                lse = _packed_heads(lse, input_num_heads)
        if (
            os.environ.get("KVARN_GATHER_DEBUG") == "1"
            and num_tokens > 0
            and not torch.cuda.is_current_stream_capturing()
        ):
            _o0 = output[0].float().sum(dim=1)
            _oa = output.float().sum(dim=(0, 2))
            _rowsum = output.float().sum(dim=(1, 2)).abs()
            _nz = (_rowsum > 0).nonzero().flatten()
            _stg_snap = getattr(self, "_dbg_stg_snap", None)
            _postcopy = getattr(self, "_dbg_postcopy", None)
            if _stg_snap is not None and _postcopy is not None:
                def _ph(tag, ref):
                    _ar = kv_cache_for_run.reshape(-1, 536)[:24]
                    _z = (_ar == 0).all(dim=1)
                    _d = (_ar != ref).any(dim=1)
                    print(
                        "[kvarn-gather-dbg] phase=%s zero_rows=%d "
                        "changed_vs_postcopy=%d first_changed=%s"
                        % (tag, int(_z.sum()), int(_d.sum()),
                           _d.nonzero().flatten()[:6].tolist()),
                        flush=True,
                    )
                _ar = kv_cache_for_run.reshape(-1, 536)[:24]
                _ph("pre-map", _postcopy)
                torch.cuda.synchronize()
            if num_tokens > 100 and not os.path.exists(
                "/home/js/kvarn-probes/kvarn-launch-capture.pt"
            ):
                try:
                    _kvarn_dbg_save(
                        {
                            "q": q_all[:24].cpu(),
                            "sel": selected_indices[:24].cpu(),
                            "act": active_counts[:24].cpu(),
                            "lens": cache_seq_lens[:24].cpu(),
                            "arena_head": (
                                kv_cache_for_run.reshape(-1)[: 2 * 143616].cpu()
                                if kv_cache_for_run.dtype == torch.uint8
                                else None
                            ),
                            "arena_shape": tuple(kv_cache_for_run.shape),
                            "arena_strides": tuple(kv_cache_for_run.stride()),
                            "arena_dtype": str(kv_cache_for_run.dtype),
                            "out_head": output[:24].cpu(),
                        },
                        "/home/js/kvarn-probes/kvarn-launch-capture.pt",
                    )
                except Exception as _e:
                    print("[kvarn-gather-dbg] capture save failed:", _e,
                          flush=True)
            print(
                "[kvarn-gather-dbg] osum route=%s rows=%d heads=%d "
                "o_row0_8=%s o_row0_all=%.6e o_head_total8=%s "
                "zero_rows=%d first_nz=%s last_nz=%d"
                % (
                    plan_key[0], num_tokens, int(output.shape[1]),
                    [round(float(x), 3) for x in _o0[:8].tolist()],
                    float(_o0.sum()),
                    [round(float(x), 3) for x in _oa[:8].tolist()],
                    int((_rowsum == 0).sum()),
                    _nz[:5].tolist(), int(_nz[-1]) if _nz.numel() else -1,
                ),
                flush=True,
            )
        if self.need_to_return_lse_for_decode:
            assert lse is not None
            return output, lse
        return output, None
