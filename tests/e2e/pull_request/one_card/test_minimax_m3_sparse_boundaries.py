# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Final sparse selection contracts from vLLM PR63232 / MSA 25c1447.

The independent Torch oracle reserves forced pages before ranking. No NVIDIA
kernel is executed. These checks establish boundary semantics, not model parity.
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode
from vllm_ascend.models.minimax_m3.ops.msa_m3_triton_a5 import minimax_m3_index_decode, minimax_m3_index_topk

BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 16


@pytest.fixture(autouse=True)
def require_fp8_attention():
    if not get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
        pytest.skip("These FP8 selector checks require Ascend 950")


def _canonical(indices):
    sentinel = torch.iinfo(torch.int32).max
    ordered = indices.cpu().to(torch.int32).masked_fill(indices.cpu() < 0, sentinel).sort(dim=-1).values
    return ordered.masked_fill(ordered == sentinel, -1)


def _reserve_then_rank(scores, valid_pages, begin=0, end=1):
    heads = scores.shape[0]
    expected = torch.full((*scores.shape[:-1], TOPK), -1, dtype=torch.int32)
    for token, count in enumerate(valid_pages):
        forced = sorted(set(range(min(begin, count))) | set(range(max(0, count - end), count)))
        available = [page for page in range(count) if page not in forced]
        take = min(TOPK - len(forced), len(available))
        for head in range(heads):
            selected = forced.copy()
            if take:
                order = scores[head, token, available].topk(take).indices.tolist()
                selected.extend(available[index] for index in order)
            expected[head, token, : len(selected)] = torch.tensor(sorted(selected), dtype=torch.int32)
    return expected


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("begin,end", [(0, 1), (2, 3)])
@pytest.mark.parametrize("future_score", [1e35, float("nan")], ids=["high-future", "nan-future"])
@torch.inference_mode()
def test_prefill_forces_current_page_and_masks_future_pages(heads, begin, end, future_score):
    query_lengths = [0, 1, 2, 127, 128, 129, 2, 1, 5, 2, 2]
    prefixes = [0, 1, 127, 128, 2047, 2048, 2049, 4095, 4096, 81919, 131999]
    offsets = [0]
    for length in query_lengths:
        offsets.append(offsets[-1] + length)
    valid_pages = [
        (prefix + offset) // BLOCK_SIZE + 1
        for prefix, length in zip(prefixes, query_lengths)
        for offset in range(length)
    ]
    width = math.ceil(max(valid_pages) / 16) * 16
    scores = torch.arange(width, dtype=torch.float32).expand(heads, offsets[-1], width).clone()
    for token, count in enumerate(valid_pages):
        scores[:, token, count - 1] = -1e20
        scores[:, token, count:] = future_score
    expected = _reserve_then_rank(scores, valid_pages, begin, end)
    actual = minimax_m3_index_topk(
        scores.npu(),
        torch.tensor(offsets, dtype=torch.int32, device="npu"),
        torch.tensor(prefixes, dtype=torch.int32, device="npu"),
        max(query_lengths),
        TOPK,
        begin,
        end,
    )
    torch.testing.assert_close(_canonical(actual), expected, atol=0, rtol=0)


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_decode_forced_partial_page_uses_current_graph_lengths(heads, use_graph):
    lengths = [0, 1, 127, 128, 129, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193]
    width = math.ceil(max(lengths) / BLOCK_SIZE)
    table = torch.arange(len(lengths) * width, dtype=torch.int32).reshape(len(lengths), width)
    cache = torch.full((len(lengths) * width, BLOCK_SIZE, HEAD_DIM), 100.0)
    for request, length in enumerate(lengths):
        for page in range(math.ceil(length / BLOCK_SIZE)):
            cache[request * width + page].zero_()
            cache[request * width + page, :, : page + 1] = 1
        if length:
            page, used = divmod(length - 1, BLOCK_SIZE)
            cache[request * width + page, : used + 1] = -1
            cache[request * width + page, used + 1 :] = 100
    query = torch.ones(len(lengths), heads, HEAD_DIM)
    query *= torch.tensor([1, 2, 4, 8][:heads]).reshape(1, heads, 1)
    query, cache = query.to(torch.float8_e4m3fn), cache.to(torch.float8_e4m3fn)
    query_npu, cache_npu, table_npu = query.npu(), cache.npu(), table.npu()
    sequence_lengths = torch.tensor(lengths, dtype=torch.int32, device="npu")

    def decode():
        return minimax_m3_index_decode(
            query_npu, cache_npu, table_npu, sequence_lengths, max(lengths), TOPK, 0, 1, heads, 1
        )

    if use_graph:
        for _ in range(2):
            decode()
        torch.npu.synchronize()
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            indices, counts = decode()
        torch.npu.current_stream().wait_stream(stream)
    for step in range(3 if use_graph else 1):
        current = [max(0, length - step) for length in lengths]
        sequence_lengths.copy_(torch.tensor(current, dtype=torch.int32, device="npu"))
        if use_graph:
            graph.replay()
        else:
            indices, counts = decode()
        torch.npu.synchronize()
        scores = torch.full((heads, len(lengths), width), float("-inf"))
        valid_pages = []
        for request, length in enumerate(current):
            pages = math.ceil(length / BLOCK_SIZE)
            valid_pages.append(pages)
            for page in range(pages):
                used = min(BLOCK_SIZE, length - page * BLOCK_SIZE)
                keys = cache[request * width + page, :used].float()
                scores[:, request, page] = (query[request].float() @ keys.T).max(-1).values
        torch.testing.assert_close(_canonical(indices), _reserve_then_rank(scores, valid_pages), atol=0, rtol=0)
        expected_counts = torch.tensor([min(TOPK, count) for count in valid_pages], dtype=torch.int32).expand(heads, -1)
        torch.testing.assert_close(counts.cpu(), expected_counts, atol=0, rtol=0)


@pytest.mark.parametrize(
    "mode,query_lengths,prefixes",
    [
        ("decode", [1] * 9, [0, 126, 127, 128, 2046, 2047, 2048, 4095, 81919]),
        ("decode", [4] * 5, [0, 124, 2045, 4094, 81917]),
        ("prefill", [129, 2, 1, 5], [0, 4095, 81919, 2045]),
    ],
)
@torch.inference_mode()
def test_attention_excludes_future_slots_in_forced_partial_page(mode, query_lengths, prefixes):
    lengths = [length + prefix for length, prefix in zip(query_lengths, prefixes)]
    width = math.ceil(max(lengths) / BLOCK_SIZE)
    table = torch.arange(len(lengths) * width, dtype=torch.int32).reshape(len(lengths), width).flip(1).contiguous()
    key = torch.zeros(len(lengths) * width, BLOCK_SIZE, 1, HEAD_DIM)
    value = torch.full_like(key, 448.0)
    for request, length in enumerate(lengths):
        for page in range(math.ceil(length / BLOCK_SIZE)):
            value[table[request, page], : min(BLOCK_SIZE, length - page * BLOCK_SIZE)] = 1.0
    total = sum(query_lengths)
    query = torch.zeros(total, 16, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    indices = torch.full((1, total, TOPK), -1, dtype=torch.int32)
    token = 0
    for length, prefix in zip(query_lengths, prefixes):
        for offset in range(length):
            count = (prefix + offset) // BLOCK_SIZE + 1
            selected = list(range(min(count - 1, TOPK - 1))) + [count - 1]
            indices[0, token, : len(selected)] = torch.tensor(selected[::-1], dtype=torch.int32)
            token += 1
    indices, table = indices.npu(), table.npu()
    sequence_lengths = torch.tensor(lengths, dtype=torch.int32, device="npu")
    key, value = key.to(torch.float8_e4m3fn).npu(), value.to(torch.float8_e4m3fn).npu()
    output = torch.empty_like(query)
    if mode == "decode":
        minimax_m3_sparse_attn_decode(
            query, (key, value), indices, table, sequence_lengths, 1, HEAD_DIM**-0.5, output, query_lengths[0]
        )
    else:
        offsets = torch.tensor([0] + torch.tensor(query_lengths).cumsum(0).tolist(), dtype=torch.int32, device="npu")
        minimax_m3_sparse_attn(
            query,
            (key, value),
            indices,
            table,
            offsets,
            sequence_lengths,
            torch.tensor(prefixes, dtype=torch.int32, device="npu"),
            max(query_lengths),
            1,
            HEAD_DIM**-0.5,
            output,
            total_kv_blocks=sum(math.ceil(length / BLOCK_SIZE) for length in lengths),
            max_kv_blocks=width,
        )
    # Every visible V is 1. Future/stale V is 448, so a partial-page leak is observable.
    torch.testing.assert_close(
        output.float().cpu(), torch.ones_like(output, device="cpu", dtype=torch.float32), atol=0.025, rtol=0
    )
