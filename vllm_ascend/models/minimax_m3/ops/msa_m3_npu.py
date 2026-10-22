# SPDX-License-Identifier: Apache-2.0
"""NPU sparse attention ops for MiniMax-M3 on Ascend."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch_npu

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op

# Existing mode for A5 FP8 KV-gather-Q prefill.
_SPARSE_ATTN_INNER_PRECISE = 4

# Existing mode for A5 KV-gather-Q prefill with BF16 inputs.
_PREFILL_KV_GATHER_Q_INNER_PRECISE = 1

# A3 KV-gather-Q prefill with BF16 inputs: use FP32 scores and partial
# outputs to preserve accuracy.
_A3_PREFILL_KV_GATHER_Q_INNER_PRECISE = 0
_MSA_INDEX_BLOCK_SIZE = 128
_GBSA_MASK_MODE_CAUSAL = 1
_GBSA_NO_QUANT = 0
_GBSA_FP8_STATIC_CAST_P = 5
_GBSA_SOFTMAX_PRECISION_DEFAULT = 0
# Atlas A5 GBSA supports softmax_precision=1 for both BF16 and FP8 inputs.
_GBSA_SOFTMAX_PRECISION_A5 = 1
_GBSA_BF16_MAX_QUERY_BATCH = 128
_GBSA_BF16_MAX_FUSED_BUFFER_BYTES = 65536
_GBSA_BF16_COPY_ALIGNMENT = 16
_MSA_SCORE_BLOCK_ALIGNMENT = 16
_MSA_DECODE_TOPK_MAX_COLUMNS = 2048
_MSA_DECODE_TOPK_MAX_K = 128
_MSA_DECODE_TOPK_ALIGNMENT = 8
_FP8_E4M3_MAX = 448.0

if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
    from vllm_ascend.models.minimax_m3.ops.msa_m3_triton_a5 import (
        minimax_m3_index_topk as _minimax_m3_index_prefill_topk,
    )
elif get_current_hardware_profile().supports(HardwareCapability.RUNTIME_CUSTOM_OPS):
    from vllm_ascend.models.minimax_m3.ops.msa_m3_triton import (
        minimax_m3_index_topk as _minimax_m3_index_prefill_topk,
    )


def _k2q_csr_block_stats(cu_block_lens: torch.Tensor) -> tuple[int, int]:
    """Derive ``(total_rows, max_kv)`` from ``cu_block_lens`` on host."""
    cu = cu_block_lens.reshape(-1)
    if cu.numel() <= 1:
        return 0, 0
    block_lens = cu[1:] - cu[:-1]
    total_rows = int(cu[-1].item())
    max_kv = int(block_lens.max().item()) if block_lens.numel() else 0
    return total_rows, max_kv


@torch.no_grad()
def _npu_k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_lens: torch.Tensor,
    order_method: int = 0,
    total_rows: int = -1,
    max_kv: int = -1,
    use_simt: int | bool = 0,
    q_global_offset: int | bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert MiniMax-M3 q2k indices to k2q CSR on NPU."""
    # A5 disables the generic custom-op loader, so register the in-tree
    # MiniMax M3 operators lazily after the NPU runtime has been initialized.
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, PLC0415

    enable_custom_op()
    if total_rows < 0 or max_kv < 0:
        derived_total_rows, derived_max_kv = _k2q_csr_block_stats(cu_block_lens)
        if total_rows < 0:
            total_rows = derived_total_rows
        if max_kv < 0:
            max_kv = derived_max_kv
    return torch.ops._C_ascend.npu_k2q_csr(
        q2k,
        cu_seqlens,
        cu_block_lens,
        int(order_method),
        int(total_rows),
        int(max_kv),
        int(use_simt),
        int(q_global_offset),
    )


@dataclass
class MiniMaxM3TPDecodeScoreMetadata:
    """Graph-stable inputs for packed TP decode scoring."""

    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    context_lens: torch.Tensor
    max_block_count: int
    block_size: int
    block_offset: int
    block_count: int
    decode_query_len: int


def _as_ascendc_index_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
) -> torch.Tensor:
    """Convert the runtime index K cache to the AscendC BBND layout.

    The model runner binds this K-only cache as a one-element tuple whose
    tensor is shaped ``[num_blocks, 128, head_dim]``. Direct calls may pass
    that tensor without the tuple. MsaIndexScore expects the BBND shape
    ``[num_blocks, 128, 1, head_dim]`` instead.
    """
    if isinstance(index_kv_cache, tuple):
        index_kv_cache = index_kv_cache[0]
    return index_kv_cache.unsqueeze(2)


def _split_main_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) < 2:
            raise ValueError("Main kv cache tuple must contain K and V tensors")
        k_cache, v_cache = kv_cache[0], kv_cache[1]
    else:
        if kv_cache.ndim != 5:
            raise ValueError(f"Unexpected main kv cache ndim: {kv_cache.ndim}")
        if kv_cache.shape[0] == 2:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        elif kv_cache.shape[1] == 2:
            k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
        else:
            raise ValueError(f"Unexpected main kv cache shape: {tuple(kv_cache.shape)}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(f"Unexpected split main kv cache shapes: {tuple(k_cache.shape)}, {tuple(v_cache.shape)}")
    return k_cache, v_cache


def _select_num_idx_from_topk(topk_idx: torch.Tensor) -> torch.Tensor:
    return (topk_idx >= 0).sum(dim=-1).to(dtype=torch.int32)


def _to_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clamp(min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX).to(torch.float8_e4m3fn)


def _build_cu_seqlens_q(query_lens: torch.Tensor) -> torch.Tensor:
    cu_seqlens_q = torch.empty(
        query_lens.numel() + 1,
        dtype=torch.int64,
        device=query_lens.device,
    )
    cu_seqlens_q[0] = 0
    torch.cumsum(
        query_lens.to(dtype=torch.int64),
        dim=0,
        out=cu_seqlens_q[1:],
    )
    return cu_seqlens_q


@lru_cache
def _is_bf16_sparse_attention_fia_fused_available() -> bool:
    # A5 bypasses the generic custom-op loader. Import only after NPU init.
    try:
        import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, PLC0415
    except ImportError:
        return False
    return hasattr(torch.ops._C_ascend, "npu_msa_sparse_attention_fia_metadata") and hasattr(
        torch.ops._C_ascend, "npu_msa_sparse_attention_fia_finish"
    )


@torch.no_grad()
def _run_bf16_sparse_attention_high_precision(
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    select_num_idx: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_kv_heads: int,
    sm_scale: float,
    block_size: int,
    use_fused: bool = True,
) -> torch.Tensor:
    """Preserve BF16 inputs while avoiding A5 GBSA's BF16 score rounding.

    Treat each query and KV head as an independent paged FIA request. The
    per-query block table retains only the selected logical blocks; a device
    mask handles causality, selection counts and padding. FIA uses FP32
    intermediate scores with inner_precise=0, without materializing dense KV.
    A5 AscendC kernels fuse metadata preparation and output gathering. Older
    extensions retain the equivalent PyTorch path. Query chunks bound the
    temporary query and mask storage during prefill.
    """
    num_tokens, num_heads, head_dim = q.shape
    if num_kv_heads <= 0 or num_heads % num_kv_heads or key.shape[2] != num_kv_heads:
        raise ValueError("Sparse attention Q and KV heads must form complete GQA groups")
    if num_tokens == 0 or topk_idx.shape[-1] == 0 or key.shape[0] == 0:
        return torch.zeros_like(q)
    group_size = num_heads // num_kv_heads
    topk = topk_idx.shape[-1]
    # The last two dimensions of the paged cache are contiguous in the model
    # layout. These views preserve its storage and do not gather or cast KV.
    paged_key = key.flatten(2)
    paged_value = value.flatten(2)
    use_fused = (
        use_fused
        and q.device.type == "npu"
        and q.dtype == torch.bfloat16
        and block_size == _MSA_INDEX_BLOCK_SIZE
        and topk * block_size <= _GBSA_BF16_MAX_FUSED_BUFFER_BYTES
        and group_size * head_dim % _GBSA_BF16_COPY_ALIGNMENT == 0
        and group_size * head_dim * q.element_size() <= _GBSA_BF16_MAX_FUSED_BUFFER_BYTES
        and all(t.is_contiguous() for t in (topk_idx, block_table, select_num_idx, cu_seqlens_q, seq_lens))
        and _is_bf16_sparse_attention_fia_fused_available()
    )
    if not use_fused:
        row_ids = torch.arange(block_size, device=q.device)
        slot_ids = torch.arange(topk, device=q.device)
    chunk_size = max(1, _GBSA_BF16_MAX_QUERY_BATCH // num_kv_heads)
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    for begin in range(0, num_tokens, chunk_size):
        end = min(begin + chunk_size, num_tokens)
        batch = num_kv_heads * (end - begin)
        if use_fused:
            pages, mask, valid = torch.ops._C_ascend.npu_msa_sparse_attention_fia_metadata(
                topk_idx,
                block_table,
                select_num_idx,
                cu_seqlens_q,
                seq_lens,
                begin,
                end - begin,
                key.shape[0],
                block_size,
            )
        else:
            tokens = torch.arange(begin, end, device=q.device)
            requests = torch.searchsorted(cu_seqlens_q[1:].contiguous(), tokens, right=True)
            query_pos = seq_lens[requests] - (cu_seqlens_q[requests + 1] - cu_seqlens_q[requests])
            query_pos = query_pos + tokens - cu_seqlens_q[requests]
            indices = topk_idx[:, begin:end]
            safe_indices = indices.long().clamp(min=0, max=block_table.shape[1] - 1)
            pages = block_table[requests[None, :, None], safe_indices]
            pages = pages.clamp(min=0, max=key.shape[0] - 1).reshape(num_kv_heads * (end - begin), topk)
            positions = indices[..., None] * block_size + row_ids
            valid = (
                (indices[..., None] >= 0)
                & (indices[..., None] < block_table.shape[1])
                & (positions <= query_pos[None, :, None, None])
                & (positions < seq_lens[requests][None, :, None, None])
                & (slot_ids[None, None, :, None] < select_num_idx[:, begin:end, None, None])
            )
            mask = ~valid.reshape(batch, 1, 1, topk * block_size)
        # Each KV head can select different pages. Repeating the small query
        # lets FIA retain the original cache's complete head layout; only the
        # corresponding GQA output group is kept after the call.
        query = q[None, begin:end].expand(num_kv_heads, -1, -1, -1)
        query = query.reshape(batch, 1, num_heads * head_dim).contiguous()
        out, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            paged_key,
            paged_value,
            num_query_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            input_layout="BSH",
            atten_mask=mask,
            block_table=pages,
            block_size=block_size,
            actual_seq_kvlen=[topk * block_size] * batch,
            softmax_scale=sm_scale,
            sparse_mode=0,
            inner_precise=0,
        )
        if use_fused:
            torch.ops._C_ascend.npu_msa_sparse_attention_fia_finish(out, valid, output[begin:end], num_kv_heads)
        else:
            out = torch.where(valid.reshape(batch, -1).any(-1)[:, None, None], out, 0)
            out = out.reshape(num_kv_heads, end - begin, num_kv_heads, group_size, head_dim)
            groups = [out[head, :, head] for head in range(num_kv_heads)]
            output[begin:end].copy_(torch.stack(groups, dim=1).reshape(end - begin, num_heads, head_dim))
    return output


@torch.no_grad()
def _run_generic_block_sparse_attention(
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    select_num_idx: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_kv_heads: int,
    sm_scale: float,
    block_size: int,
    max_query_len: int = -1,
    dequant_scale: torch.Tensor | None = None,
    gbsa_metadata_cache: dict[tuple[Any, ...], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Use CANN GBSA, with high-precision paged FIA for A5 BF16 inputs."""
    del dequant_scale
    topk_idx = topk_idx.to(dtype=torch.int32).contiguous()
    select_num_idx = select_num_idx.to(dtype=torch.int32).contiguous()
    block_table = block_table.to(dtype=torch.int32).contiguous()
    cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int64).contiguous()
    seq_lens = seq_lens.to(dtype=torch.int32).contiguous()

    quant_mode = _GBSA_NO_QUANT
    quant_kwargs: dict[str, Any] = {}
    if key.dtype == torch.float8_e4m3fn:
        if value.dtype != torch.float8_e4m3fn:
            raise TypeError("MiniMax-M3 FP8 sparse attention requires both K and V caches in E4M3")
        q = _to_fp8_e4m3(q)
        quant_mode = _GBSA_FP8_STATIC_CAST_P
        quant_kwargs = {"attention_out_dtype": torch.bfloat16}
    elif q.dtype != key.dtype or value.dtype != key.dtype:
        raise TypeError("MiniMax-M3 sparse attention requires Q, K, and V to have the same dtype")

    supports_fp8 = get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)
    if supports_fp8 and key.dtype == torch.bfloat16:
        return _run_bf16_sparse_attention_high_precision(
            q,
            key,
            value,
            topk_idx,
            block_table,
            select_num_idx,
            cu_seqlens_q,
            seq_lens,
            num_kv_heads=num_kv_heads,
            sm_scale=sm_scale,
            block_size=block_size,
        )
    softmax_precision = _GBSA_SOFTMAX_PRECISION_A5 if supports_fp8 else _GBSA_SOFTMAX_PRECISION_DEFAULT
    block_shape = [1, block_size]
    from cann_ops_transformer.ops import (  # type: ignore[import-not-found]  # noqa: PLC0415
        generic_block_sparse_attention,
        generic_block_sparse_attention_metadata,
    )

    # Opt-in only from the A3 model's uniform-decode metadata builder. Direct
    # callers and prefill must not assume that layer-dependent counts match.
    # The cache belongs to one decode metadata object, so its sequence/prefix
    # tensors are shared by all entries. Include all other scheduling inputs.
    reuse_metadata = (
        gbsa_metadata_cache is not None
        and q.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and block_size == _MSA_INDEX_BLOCK_SIZE
        and max_query_len in (1, 4)
    )
    cache = gbsa_metadata_cache if reuse_metadata else None
    cache_key = (tuple(q.shape), tuple(topk_idx.shape), tuple(select_num_idx.shape), num_kv_heads, max_query_len)
    gbsa_metadata = cache.get(cache_key) if cache is not None else None
    if gbsa_metadata is None:
        gbsa_metadata = generic_block_sparse_attention_metadata(
            topk_idx,
            select_num_idx,
            q.shape[1],
            num_kv_heads,
            q.shape[2],
            block_shape,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seq_lens,
            max_seqlen_q=max_query_len,
            layout_q="TND",
            layout_kv="PA_BBND",
            mask_mode=_GBSA_MASK_MODE_CAUSAL,
            quant_mode=quant_mode,
            softmax_precision=softmax_precision,
        )
        if cache is not None:
            cache[cache_key] = gbsa_metadata
    out, _ = generic_block_sparse_attention(
        q,
        key,
        value,
        topk_idx,
        select_num_idx,
        block_shape,
        metadata=gbsa_metadata,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seq_lens,
        block_table=block_table,
        layout_q="TND",
        layout_kv="PA_BBND",
        softmax_scale=sm_scale,
        mask_mode=_GBSA_MASK_MODE_CAUSAL,
        quant_mode=quant_mode,
        softmax_precision=softmax_precision,
        return_softmax_lse=False,
        **quant_kwargs,
    )
    return out


def _build_cu_block_lens(
    seq_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Build cumulative logical KV-block counts for each prefill request."""
    block_lens = torch.div(
        seq_lens.to(torch.int32) + block_size - 1,
        block_size,
        rounding_mode="floor",
    )
    cu_block_lens = torch.empty(
        block_lens.numel() + 1,
        dtype=torch.int32,
        device=seq_lens.device,
    )
    cu_block_lens[0] = 0
    torch.cumsum(block_lens, dim=0, out=cu_block_lens[1:])
    return cu_block_lens


@torch.no_grad()
def _minimax_m3_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor | None,
    *,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> torch.Tensor:
    """Compute MSA index scores with the bundled AscendC operator.

    ``start_loc`` is the current query block in the *passed block table*.  For
    a TP-sharded table this is a per-request local block index, not the scalar
    global ``block_offset`` used by the Triton decode kernel.

    A causal mask selects sparse mode 3. Passing no mask selects dense mode 0
    for a TP chunk that is entirely before the current query positions.
    ``init_blocks`` and ``local_blocks`` are kept for parity with the index
    scoring interface; candidate forcing is applied by the TopK stage.
    """
    # A5 disables the generic custom-op loader. Register this bundled operator
    # explicitly, as for k2q CSR, after the NPU runtime has been initialized.
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, PLC0415

    index_kv_cache = _as_ascendc_index_kv_cache(index_kv_cache)
    if index_kv_cache.dtype == torch.float8_e4m3fn and idx_q.dtype != index_kv_cache.dtype:
        idx_q = _to_fp8_e4m3(idx_q)
    return torch.ops._C_ascend.npu_msa_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        start_loc,
        atten_mask=causal_mask,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_klen=seq_lens,
        layout_key="BBND",
        sparse_mode=3 if causal_mask is not None else 0,
    )


@torch.no_grad()
def minimax_m3_index_prefill(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor,
    *,
    max_query_len: int,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Compute AscendC prefill scores and finalize their block TopK."""
    score = _minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        start_loc,
        causal_mask,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    logical_block_count = (max_seq_len + _MSA_INDEX_BLOCK_SIZE - 1) // _MSA_INDEX_BLOCK_SIZE
    logical_score_width = (
        (logical_block_count + _MSA_SCORE_BLOCK_ALIGNMENT - 1)
        // _MSA_SCORE_BLOCK_ALIGNMENT
        * _MSA_SCORE_BLOCK_ALIGNMENT
    )
    score = score[..., :logical_score_width]
    return _minimax_m3_index_prefill_topk(
        score,
        cu_seqlens_q,
        context_lens,
        max_query_len,
        topk,
        init_blocks,
        local_blocks,
    )


def _index_score_topk_candidates(
    score: torch.Tensor,
    context_lens: torch.Tensor,
    decode_query_len: int,
    topk: int,
    block_offset: int = 0,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global block IDs and scores after exactly one local TopK."""
    block_count = score.shape[-1]
    query_offsets = torch.arange(
        decode_query_len,
        dtype=context_lens.dtype,
        device=context_lens.device,
    ).repeat(context_lens.shape[0])
    visible_tokens = context_lens.repeat_interleave(decode_query_len) + query_offsets + 1
    global_valid_block_count = torch.div(
        visible_tokens + block_offset * _MSA_INDEX_BLOCK_SIZE + _MSA_INDEX_BLOCK_SIZE - 1,
        _MSA_INDEX_BLOCK_SIZE,
        rounding_mode="floor",
    ).clamp(min=0)
    valid_block_count = (global_valid_block_count - block_offset).clamp(min=0, max=block_count)
    local_block_ids = torch.arange(block_count, device=score.device)
    global_block_ids = local_block_ids + block_offset
    valid_blocks = global_block_ids[None, :] < global_valid_block_count[:, None]
    # A dense raw-score TP chunk can contain whole future speculative blocks
    # for a shorter request. Remove them before TopK so they cannot displace a
    # valid candidate and only then be discarded by output validation.
    score = torch.where(valid_blocks[None, :, :], score, float("-inf"))

    # Apply forced-block scores per token before the one and only TopK.
    if init_blocks > 0:
        init_mask = valid_blocks & (global_block_ids[None, :] < init_blocks)
        score = torch.where(init_mask[None, :, :], 1.0e30, score)
    if local_blocks > 0:
        local_start = (global_valid_block_count - local_blocks).clamp(min=0)
        local_mask = valid_blocks & (global_block_ids[None, :] >= local_start[:, None])
        score = torch.where(local_mask[None, :, :], 1.0e29, score)

    actual_topk_count = min(topk, block_count)
    raw_scores, raw_topk = torch.topk(
        score,
        k=actual_topk_count,
        dim=-1,
    )
    if actual_topk_count == topk:
        topk_indices = raw_topk.to(dtype=torch.int32)
        topk_scores = raw_scores
    else:
        topk_indices = torch.full(
            (*raw_topk.shape[:-1], topk),
            -1,
            dtype=torch.int32,
            device=raw_topk.device,
        )
        topk_scores = torch.full(
            (*raw_scores.shape[:-1], topk),
            float("-inf"),
            dtype=raw_scores.dtype,
            device=raw_scores.device,
        )
        topk_indices[..., :actual_topk_count].copy_(raw_topk)
        topk_scores[..., :actual_topk_count].copy_(raw_scores)

    valid_candidate = (topk_indices >= 0) & (topk_indices < valid_block_count[None, :, None])
    topk_indices = torch.where(
        valid_candidate,
        topk_indices + block_offset,
        -1,
    )
    topk_scores = torch.where(
        valid_candidate,
        topk_scores,
        float("-inf"),
    )
    return topk_indices, topk_scores


def _index_decode_topk_with_counts(
    score: torch.Tensor,
    context_lens: torch.Tensor,
    decode_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse A5 decode masking, TopK and counts; keep the portable fallback."""
    if (
        get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)
        and score.device.type == "npu"
        and score.dtype == torch.float32
        and score.stride(-1) == 1
        and context_lens.dtype == torch.int32
        and context_lens.is_contiguous()
        and 0 < score.shape[-1] <= _MSA_DECODE_TOPK_MAX_COLUMNS
        and 0 < topk <= _MSA_DECODE_TOPK_MAX_K
        and topk % _MSA_DECODE_TOPK_ALIGNMENT == 0
        and hasattr(torch.ops._C_ascend, "npu_msa_index_decode_topk")
    ):
        return torch.ops._C_ascend.npu_msa_index_decode_topk(
            score, context_lens, decode_query_len, topk, init_blocks, local_blocks
        )
    indices, values = _index_score_topk_candidates(
        score, context_lens, decode_query_len, topk, init_blocks=init_blocks, local_blocks=local_blocks
    )
    return indices, values, _select_num_idx_from_topk(indices)


@torch.no_grad()
def _minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor | None,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    decode_query_len: int,
    block_offset: int = 0,
    block_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run AscendC decode score followed by one non-fused local TopK."""
    score = _minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        start_loc,
        causal_mask,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    if block_count is None:
        block_count = block_table.shape[-1]
    return _index_score_topk_candidates(
        score[..., :block_count],
        context_lens,
        decode_query_len,
        topk,
        block_offset,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    decode_query_len: int,
    return_select_num_idx: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return block TopK, optionally with fused A5 valid-selection counts.

    The default retains the A3/TP interface. A5 callers pass counts directly
    to sparse attention to avoid a second reduction over TopK indices.
    """
    if return_select_num_idx:
        score = _minimax_m3_index_score(
            idx_q,
            index_kv_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            start_loc,
            causal_mask,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )
        indices, _, counts = _index_decode_topk_with_counts(
            score[..., : block_table.shape[-1]],
            context_lens,
            decode_query_len,
            topk,
            init_blocks,
            local_blocks,
        )
        return indices, counts
    topk_indices, _ = _minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        context_lens,
        start_loc,
        causal_mask,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        decode_query_len=decode_query_len,
    )
    return topk_indices


@torch.no_grad()
def minimax_m3_index_tp_block_parallel_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    metadata: MiniMaxM3TPDecodeScoreMetadata,
    causal_mask: torch.Tensor,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    tp_group: Any,
) -> torch.Tensor:
    """Run packed-query scoring and TopK over TP-sharded KV blocks."""
    full_idx_q = tp_group.all_gather(idx_q.contiguous(), dim=1).contiguous()
    tp_rank = tp_group.rank_in_group
    block_offset = metadata.block_offset
    block_count = metadata.block_count
    if block_count == 0:
        # MsaIndexScore requires a non-empty block-table width. Ranks without
        # logical blocks contribute neutral candidates to the collectives.
        candidate_shape = (
            full_idx_q.shape[1],
            full_idx_q.shape[0],
            topk,
        )
        local_topk = torch.full(
            candidate_shape,
            -1,
            dtype=torch.int32,
            device=full_idx_q.device,
        )
        local_scores = torch.full(
            candidate_shape,
            float("-inf"),
            dtype=torch.float32,
            device=full_idx_q.device,
        )
    else:
        # Keep all derived tensors in the model forward. FULL_DECODE_ONLY
        # captures and replays this work, whereas tensors allocated by the
        # metadata builder would leave the graph holding stale device pointers.
        halo_blocks = (metadata.decode_query_len - 1 + metadata.block_size - 1) // metadata.block_size
        score_block_end = min(
            block_offset + block_count + halo_blocks,
            metadata.max_block_count,
        )
        score_block_table = metadata.block_table[:, block_offset:score_block_end].contiguous()
        local_context_lens = metadata.context_lens - block_offset * metadata.block_size
        chunk_capacity = block_count * metadata.block_size
        score_capacity = score_block_table.shape[-1] * metadata.block_size
        max_score_k_len = min(
            chunk_capacity + metadata.decode_query_len - 1,
            score_capacity,
        )
        score_k_lens = torch.clamp(
            metadata.context_lens + metadata.decode_query_len - block_offset * metadata.block_size,
            min=0,
            max=max_score_k_len,
        )
        score_start_loc = (
            torch.div(
                metadata.context_lens,
                metadata.block_size,
                rounding_mode="floor",
            )
            .sub(block_offset)
            .clamp(
                min=0,
                max=score_block_table.shape[-1] - 1,
            )
        )
        local_topk, local_scores = _minimax_m3_index_decode(
            full_idx_q,
            index_kv_cache,
            score_block_table,
            metadata.cu_seqlens_q,
            score_k_lens,
            local_context_lens,
            score_start_loc,
            None if metadata.decode_query_len == 1 else causal_mask,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            decode_query_len=metadata.decode_query_len,
            block_offset=block_offset,
            block_count=block_count,
        )

    gathered_scores = tp_group.all_gather(local_scores.contiguous(), dim=-1)
    gathered_topk = tp_group.all_gather(local_topk.contiguous(), dim=-1)

    local_head_count = idx_q.shape[1]
    local_head_start = tp_rank * local_head_count
    local_gathered_scores = gathered_scores.narrow(0, local_head_start, local_head_count)
    _, merged_pos = torch.topk(
        local_gathered_scores,
        k=topk,
        dim=-1,
    )
    local_gathered_topk = gathered_topk.narrow(0, local_head_start, local_head_count)
    return torch.gather(local_gathered_topk, dim=-1, index=merged_pos)


@torch.no_grad()
def _minimax_m3_sparse_attn_a3(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    block_size: int,
    *,
    max_query_len: int = -1,
) -> None:
    key, value = _split_main_kv_cache(kv_cache)
    out = _run_generic_block_sparse_attention(
        q,
        key,
        value,
        topk_idx,
        block_table,
        _select_num_idx_from_topk(topk_idx),
        cu_seqlens_q,
        seq_lens,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
        block_size=block_size,
        max_query_len=max_query_len,
    )
    output.copy_(out)


def _minimax_m3_sparse_attn_kv_gather_q(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    block_size: int,
    total_kv_blocks: int,
    max_kv_blocks: int,
    *,
    supports_fp8: bool,
) -> None:
    key, value = _split_main_kv_cache(kv_cache)

    # A3 uses FP32 scores and partial outputs. Keep A5's existing BF16
    # precision mode; its FP8 cache path overrides this below as before.
    inner_precise = _PREFILL_KV_GATHER_Q_INNER_PRECISE if supports_fp8 else _A3_PREFILL_KV_GATHER_Q_INNER_PRECISE
    if key.dtype == torch.float8_e4m3fn:
        if not supports_fp8:
            raise TypeError("MiniMax-M3 FP8 sparse attention is not supported on this device")
        if value.dtype != torch.float8_e4m3fn:
            raise TypeError("MiniMax-M3 FP8 sparse attention requires both K and V caches in E4M3")
        q = _to_fp8_e4m3(q)
        inner_precise = _SPARSE_ATTN_INNER_PRECISE

    cu_block_lens = _build_cu_block_lens(seq_lens, block_size)
    # Keep the -1 sentinel: K2Q drops invalid entries, while replacing them
    # with zero would add duplicate attention edges for logical KV block 0.
    k2q_row_ptr, k2q_q_indices, k2q_slot_indices = _npu_k2q_csr(
        topk_idx,
        cu_seqlens_q,
        cu_block_lens,
        order_method=1,
        total_rows=total_kv_blocks,
        max_kv=max_kv_blocks,
        use_simt=0,
        q_global_offset=True,
    )

    k2q_row_ptr = k2q_row_ptr.to(dtype=torch.int32).contiguous()
    k2q_q_indices = k2q_q_indices.to(dtype=torch.int32).contiguous()
    k2q_slot_indices = k2q_slot_indices.to(dtype=torch.int32).contiguous()
    q_lens_t = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32).contiguous()
    kv_lens_t = seq_lens.to(torch.int32).contiguous()
    out = torch.ops._C_ascend.npu_sparse_attention_score_prefill(
        q,
        key,
        value,
        block_table,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_slot_indices,
        num_kv_heads,
        sm_scale,
        block_size,
        topk_idx.shape[-1],
        inner_precise,
        actual_seq_lengths=q_lens_t,
        actual_seq_lengths_kv=kv_lens_t,
    )
    output.copy_(out)


@lru_cache
def _is_minimax_sparse_attention_split_kv_available() -> bool:
    """Check compatible vendor entry points once per worker process.

    Restart workers after installing the experimental package and sourcing its
    environment; the ACLNN loader also caches its library search paths.
    """
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, PLC0415

    enable_custom_op()
    return torch.ops._C_ascend.is_minimax_sparse_attention_split_kv_available()


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    block_size: int = 128,
    total_kv_blocks: int = -1,
    max_kv_blocks: int = -1,
) -> None:
    del prefix_lens
    common_args = (
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        block_size,
    )
    hardware_profile = get_current_hardware_profile()
    supports_fp8 = hardware_profile.supports(HardwareCapability.FP8_ATTENTION)
    # The optional A5 split-KV BF16 path also uses low-precision scores. Keep
    # BF16 prefill on the same high-precision route as decode regardless of
    # whether that experimental package happens to be installed.
    if supports_fp8 and q.dtype == torch.bfloat16 and _split_main_kv_cache(kv_cache)[0].dtype == torch.bfloat16:
        _minimax_m3_sparse_attn_a3(*common_args, max_query_len=max_query_len)
        return
    # Select the optional optimization by ACLNN availability. The installed
    # experimental package must provide kernels for the current device.
    if not _is_minimax_sparse_attention_split_kv_available():
        _minimax_m3_sparse_attn_a3(*common_args, max_query_len=max_query_len)
        return

    _minimax_m3_sparse_attn_kv_gather_q(
        *common_args,
        total_kv_blocks,
        max_kv_blocks,
        supports_fp8=supports_fp8,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    block_size: int = 128,
    *,
    select_num_idx: torch.Tensor | None = None,
    dequant_scale: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    gbsa_metadata_cache: dict[tuple[Any, ...], torch.Tensor] | None = None,
) -> None:
    """Run sparse decode through CANN attention with device-specific precision."""
    if q.shape[0] != seq_lens.shape[0] * decode_query_len:
        raise ValueError("Decode query tokens must equal request count times decode_query_len")

    key, value = _split_main_kv_cache(kv_cache)
    # The model metadata builder prepares this immutable uniform prefix once
    # per query length, outside the per-layer attention/graph execution.
    # Keep the fallback for direct callers that do not supply model metadata.
    if cu_seqlens_q is None:
        query_lens = torch.full_like(seq_lens, decode_query_len, dtype=torch.int32)
        cu_seqlens_q = _build_cu_seqlens_q(query_lens)
    elif cu_seqlens_q.shape != (seq_lens.shape[0] + 1,) or cu_seqlens_q.dtype != torch.int64:
        raise ValueError("GBSA cu_seqlens_q must be int64 with request_count + 1 entries")
    if select_num_idx is None:
        select_num_idx = _select_num_idx_from_topk(topk_idx)

    out = _run_generic_block_sparse_attention(
        q,
        key,
        value,
        topk_idx,
        block_table,
        select_num_idx,
        cu_seqlens_q,
        seq_lens,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
        block_size=block_size,
        max_query_len=decode_query_len,
        dequant_scale=dequant_scale,
        gbsa_metadata_cache=gbsa_metadata_cache,
    )
    output.copy_(out)
