# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""A5 BF16 sparse-attention precision, range and graph regressions."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.pull_request.one_card import test_minimax_m3_sparse_attn as reference
from vllm_ascend.models.minimax_m3.ops import msa_m3_npu as ops
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import minimax_m3_sparse_attn_decode

NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
BLOCK_SIZE = 128
MEAN_ATOL = 2.5e-4
MAX_ATOL = 1.7e-2


@pytest.fixture(scope="module", autouse=True)
def require_a5():
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(0):
        pytest.skip("A5 BF16 high-precision sparse attention requires Ascend 950")


def _inputs(query_len, layout="tensor"):
    torch.manual_seed(2026)
    q, table, lengths, indices, pages = reference._build_decode_inputs(
        (129, 257),
        query_len,
        1,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
    )
    fused = reference._allocate_main_kv_cache_fused(pages, num_kv_heads=NUM_KV_HEADS)
    # Exercise different selections for the two KV heads, not only GQA.
    indices[1, : 2 * query_len, 0] = 0
    indices[1, : 2 * query_len, 1:] = -1
    if layout == "tensor":
        cache = reference._main_kv_cache_for_kernel(fused)
    elif layout == "tuple":
        cache = fused.unbind(1)
    else:
        storage = torch.full((pages * 2, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), -1000, device="npu", dtype=q.dtype)
        storage[::2].copy_(fused)
        cache = storage[::2].unbind(1)
    counts = (indices >= 0).sum(-1).int()
    prefix = torch.arange(lengths.numel() + 1, dtype=torch.int64, device="npu") * query_len
    return q, cache, fused, indices, table, lengths, counts, prefix


def _run(data, query_len, output):
    q, cache, _, indices, table, lengths, counts, prefix = data
    minimax_m3_sparse_attn_decode(
        q,
        cache,
        indices,
        table,
        lengths,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        output,
        query_len,
        select_num_idx=counts,
        cu_seqlens_q=prefix,
    )


def _expected(data, query_len):
    q, _, fused, indices, table, lengths, counts, _ = data
    q, fused, indices, table, lengths, counts = [t.cpu() for t in (q, fused, indices, table, lengths, counts)]
    slots = torch.arange(indices.shape[-1])
    selected = torch.where(slots < counts[..., None], indices, -1)
    query_lens = torch.full_like(lengths, query_len)
    return reference._reference_sparse_attn(
        q,
        fused,
        selected,
        table,
        query_lens,
        lengths,
        lengths - query_lens,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        sm_scale=HEAD_DIM**-0.5,
    )


def _check(actual, expected):
    assert actual.dtype == torch.bfloat16
    actual = actual.cpu()
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).abs()
    assert error.mean().item() < MEAN_ATOL
    assert error.max().item() < MAX_ATOL


@pytest.mark.parametrize("query_len", [1, 2, 4])
@pytest.mark.parametrize("layout", ["tensor", "tuple", "strided_pages"])
@torch.inference_mode()
def test_bf16_sparse_decode_matches_cpu_reference(query_len, layout):
    data = _inputs(query_len, layout)
    output = torch.empty_like(data[0])
    _run(data, query_len, output)
    torch.npu.synchronize()
    _check(output, _expected(data, query_len))
    assert torch.count_nonzero(output[-query_len:]).item() == 0


@pytest.mark.parametrize("query_len", [1, 2])
@torch.inference_mode()
def test_bf16_sparse_decode_graph_replays_changed_lengths_and_counts(query_len):
    data = _inputs(query_len)
    output = torch.empty_like(data[0])
    for _ in range(3):
        _run(data, query_len, output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _run(data, query_len, output)
    # Reuse input storage while changing the device metadata used by replay.
    data[5][0] = 0
    data[6][0, query_len : 2 * query_len] = 1
    output.fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()
    _check(output, _expected(data, query_len))
    assert torch.count_nonzero(output[:query_len]).item() == 0


@torch.inference_mode()
def test_bf16_sparse_decode_keeps_values_outside_fp16_range():
    data = _inputs(1)
    # The output channels are constant under any legal attention weights.
    data[1][1, :, :, :, 0] = 1e5
    data[1][1, :, :, :, 1] = 1e-9
    output = torch.empty_like(data[0])
    _run(data, 1, output)
    torch.npu.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:2, :, 0], torch.full_like(output[:2, :, 0], 1e5), rtol=0.01, atol=0)
    torch.testing.assert_close(output[:2, :, 1], torch.full_like(output[:2, :, 1], 1e-9), rtol=0.02, atol=0)


@pytest.mark.parametrize("topk,tokens", [(1, 6), (3, 137), (16, 6), (512, 6)])
@torch.inference_mode()
def test_fused_metadata_matches_cpu_pages_causality_and_selection(topk, tokens):
    assert ops._is_bf16_sparse_attention_fia_fused_available(), "Rebuild A5 fused sparse-attention kernels"
    generator = torch.Generator().manual_seed(2026)
    heads, width, page_count = 2, 4, 12
    prefix = torch.tensor([0, 2, 2, tokens - 1, tokens], dtype=torch.int64)
    lengths = torch.tensor([1, 0, 257, 0], dtype=torch.int32)
    indices = torch.randint(-2, width + 2, (heads, tokens, topk), generator=generator, dtype=torch.int32)
    counts = torch.randint(-1, topk + 1, (heads, tokens), generator=generator, dtype=torch.int32)
    table = torch.randint(-1, page_count + 1, (4, width), generator=generator, dtype=torch.int32)
    actual = torch.ops._C_ascend.npu_msa_sparse_attention_fia_metadata(
        *[t.npu() for t in (indices, table, counts, prefix, lengths)], 0, tokens, page_count, BLOCK_SIZE
    )
    expected_pages = torch.empty(heads * tokens, topk, dtype=torch.int32)
    expected_mask = torch.ones(heads * tokens, 1, 1, topk * BLOCK_SIZE, dtype=torch.bool)
    expected_valid = torch.zeros(heads * tokens, 8, dtype=torch.int32)
    for head in range(heads):
        for token in range(tokens):
            request = next(i for i in range(4) if prefix[i + 1] > token)
            position = lengths[request] - (prefix[request + 1] - prefix[request]) + token - prefix[request]
            row = head * tokens + token
            for slot in range(topk):
                logical = int(indices[head, token, slot])
                safe = max(0, min(logical, width - 1))
                expected_pages[row, slot] = max(0, min(int(table[request, safe]), page_count - 1))
                if 0 <= logical < width and slot < counts[head, token]:
                    visible = max(0, min(BLOCK_SIZE, int(min(position + 1, lengths[request])) - logical * BLOCK_SIZE))
                    expected_mask[row, 0, 0, slot * BLOCK_SIZE : slot * BLOCK_SIZE + visible] = False
            expected_valid[row].fill_(int((~expected_mask[row]).any()))
    for got, expected in zip(actual, (expected_pages, expected_mask, expected_valid)):
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("kv_heads", [1, 2, 8])
@torch.inference_mode()
def test_fused_finish_keeps_gqa_diagonal_and_zeroes_padding(kv_heads):
    assert ops._is_bf16_sparse_attention_fia_fused_available(), "Rebuild A5 fused sparse-attention kernels"
    tokens, heads, dim = 137, 8, 128
    source = torch.randn(kv_heads, tokens, heads, dim, dtype=torch.bfloat16)
    valid = torch.ones(kv_heads * tokens, 8, dtype=torch.int32)
    valid[::3].zero_()
    source.reshape(kv_heads * tokens, heads, dim)[::3].fill_(float("nan"))
    group = heads // kv_heads
    expected = torch.empty(tokens, heads, dim, dtype=source.dtype)
    for head in range(kv_heads):
        expected[:, head * group : (head + 1) * group] = source[head, :, head * group : (head + 1) * group]
        expected[:, head * group : (head + 1) * group][valid[head * tokens : (head + 1) * tokens, 0] == 0] = 0
    output = torch.empty_like(expected, device="npu")
    torch.ops._C_ascend.npu_msa_sparse_attention_fia_finish(
        source.reshape(kv_heads * tokens, 1, heads * dim).npu(), valid.npu(), output, kv_heads
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("query_len", [1, 4])
@torch.inference_mode()
def test_bf16_fused_sparse_decode_is_bitwise_equal_to_unfused(query_len):
    assert ops._is_bf16_sparse_attention_fia_fused_available(), "Rebuild A5 fused sparse-attention kernels"
    data = _inputs(query_len, "strided_pages")
    q, cache, _, indices, table, lengths, counts, prefix = data
    output = torch.empty_like(q)
    _run(data, query_len, output)
    expected = ops._run_bf16_sparse_attention_high_precision(
        q,
        cache[0],
        cache[1],
        indices,
        table,
        counts,
        prefix,
        lengths,
        num_kv_heads=NUM_KV_HEADS,
        sm_scale=HEAD_DIM**-0.5,
        block_size=BLOCK_SIZE,
        use_fused=False,
    )
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)


@torch.inference_mode()
def test_fused_sparse_attention_meta_shapes():
    assert ops._is_bf16_sparse_attention_fia_fused_available(), "Rebuild A5 fused sparse-attention kernels"
    indices = torch.empty(2, 9, 3, dtype=torch.int32, device="meta")
    table = torch.empty(4, 5, dtype=torch.int32, device="meta")
    counts = torch.empty(2, 9, dtype=torch.int32, device="meta")
    prefix = torch.empty(5, dtype=torch.int64, device="meta")
    lengths = torch.empty(4, dtype=torch.int32, device="meta")
    pages, mask, valid = torch.ops._C_ascend.npu_msa_sparse_attention_fia_metadata(
        indices, table, counts, prefix, lengths, 2, 4, 20, BLOCK_SIZE
    )
    assert pages.shape == (8, 3) and pages.dtype == torch.int32
    assert mask.shape == (8, 1, 1, 384) and mask.dtype == torch.bool
    assert valid.shape == (8, 8) and valid.dtype == torch.int32
    output = torch.empty(4, 8, HEAD_DIM, dtype=torch.bfloat16, device="meta")
    torch.ops._C_ascend.npu_msa_sparse_attention_fia_finish(
        torch.empty(8, 1, 8 * HEAD_DIM, dtype=output.dtype, device="meta"), valid, output, 2
    )
