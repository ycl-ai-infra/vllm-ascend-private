# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model-free A5 FP8 GBSA decode against the existing CPU torch reference.

Uses the normal E2E conftest initialization for vLLM-Ascend model imports.
"""

from importlib import import_module

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.pull_request.one_card.test_minimax_m3_sparse_attn import (
    _reference_sparse_attn,
    minimax_m3_sparse_attn_decode_npu,
)

BLOCK_SIZE = 128
HEAD_DIM = 128
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
TOPK = 16
KV_LENGTHS = (129, 257)
# quant_mode=5 also casts softmax probabilities to FP8. These FP8 tolerances
# include that extra rounding and do not replace the existing BF16 thresholds.
FP8_MEAN_ATOL = 2.0e-3
FP8_MAX_ATOL = 2.0e-2


@pytest.fixture(scope="module", autouse=True)
def require_a5_gbsa() -> None:
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(0):
        pytest.skip("FP8 GBSA decode requires an Ascend 950 device")
    try:
        ops = import_module("cann_ops_transformer.ops")
    except ImportError as exc:
        pytest.skip(f"CANN Ops GBSA bindings are unavailable: {exc}")
    if not all(
        callable(getattr(ops, name, None))
        for name in ("generic_block_sparse_attention", "generic_block_sparse_attention_metadata")
    ):
        pytest.skip("CANN Ops GBSA bindings are not callable")


@pytest.mark.parametrize("query_len", [1, 2])
@torch.inference_mode()
def test_fp8_gbsa_decode_matches_quantized_input_cpu_reference(query_len: int) -> None:
    generator = torch.Generator().manual_seed(2026 + query_len)
    batch_size = len(KV_LENGTHS)
    pages_per_request = [(length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in KV_LENGTHS]
    num_pages = sum(pages_per_request)
    physical_pages = torch.randperm(num_pages, generator=generator, dtype=torch.int32)
    block_table = torch.zeros(batch_size, max(pages_per_request), dtype=torch.int32)
    page_offset = 0
    for request, num_request_pages in enumerate(pages_per_request):
        block_table[request, :num_request_pages] = physical_pages[page_offset : page_offset + num_request_pages]
        page_offset += num_request_pages

    def discrete(shape: tuple[int, ...]) -> torch.Tensor:
        values = torch.randint(-8, 9, shape, generator=generator, dtype=torch.int32).float() / 8
        return values.to(torch.bfloat16)

    query_cpu = discrete((batch_size * query_len, NUM_Q_HEADS, HEAD_DIM))
    cache_cpu = discrete((num_pages, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)).to(torch.float8_e4m3fn)
    # Reference the exact Q/K/V values consumed after the production FP8 cast.
    query_reference = query_cpu.clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.bfloat16)
    cache_reference = cache_cpu.to(torch.bfloat16)
    cache_npu = cache_cpu.permute(1, 0, 2, 3, 4).contiguous().npu()
    query_npu = query_cpu.npu()
    assert query_npu.shape == (batch_size * query_len, NUM_Q_HEADS, HEAD_DIM)
    assert cache_npu.shape == (2, num_pages, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    assert cache_npu.dtype == torch.float8_e4m3fn
    assert cache_npu.is_contiguous()

    topk_idx = torch.full((NUM_KV_HEADS, batch_size * query_len, TOPK), -1, dtype=torch.int32)
    for request, seq_len in enumerate(KV_LENGTHS):
        for token_offset in range(query_len):
            token = request * query_len + token_offset
            current_block = (seq_len - query_len + token_offset) // BLOCK_SIZE
            topk_idx[:, token, : current_block + 1] = torch.arange(current_block + 1, dtype=torch.int32)
    query_lens = torch.full((batch_size,), query_len, dtype=torch.int32)
    seq_lens = torch.tensor(KV_LENGTHS, dtype=torch.int32)
    expected = _reference_sparse_attn(
        query_reference,
        cache_reference,
        topk_idx,
        block_table,
        query_lens,
        seq_lens,
        seq_lens - query_lens,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        sm_scale=HEAD_DIM**-0.5,
    )

    output = torch.full_like(query_npu, float("nan"))
    cu_seqlens_q = (torch.arange(batch_size + 1, dtype=torch.int64) * query_len).npu()
    minimax_m3_sparse_attn_decode_npu(
        query_npu,
        cache_npu,
        topk_idx.npu(),
        block_table.npu(),
        seq_lens.npu(),
        num_kv_heads=NUM_KV_HEADS,
        sm_scale=HEAD_DIM**-0.5,
        output=output,
        decode_query_len=query_len,
        block_size=BLOCK_SIZE,
        cu_seqlens_q=cu_seqlens_q,
    )
    torch.npu.synchronize()
    assert output.dtype == torch.bfloat16
    actual = output.cpu()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).abs()
    mean_error, max_error = error.mean().item(), error.max().item()
    assert mean_error < FP8_MEAN_ATOL, f"mean FP8 error {mean_error} >= {FP8_MEAN_ATOL}"
    assert max_error < FP8_MAX_ATOL, f"max FP8 error {max_error} >= {FP8_MAX_ATOL}"
