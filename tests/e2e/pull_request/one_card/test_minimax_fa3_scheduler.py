# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional FA3 numerical coverage, including graph-stable metadata and KV append."""

import math
from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("flash_attn_npu_3")

from vllm_ascend.attention.flash_attention_v3 import AscendFlashAttentionImpl, AscendFlashAttentionMetadataBuilder

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires an NPU and the optional FA3 package")


@pytest.mark.parametrize("graph_stable", [False, True], ids=["eager", "graph_stable"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [None, 16])
def test_paged_attention_with_refreshed_scheduler_metadata(dtype, causal, window, graph_stable):
    torch.manual_seed(20167)
    page_size, heads, dim = 128, 4, 128
    q_lengths, kv_lengths = [7, 8], [130, 9]
    table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
    query = torch.randn(sum(q_lengths), heads, dim, dtype=dtype)
    cache = torch.randn(2, 4, page_size, 1, dim, dtype=dtype)
    new_key = torch.randn(sum(q_lengths), 1, dim, dtype=dtype)
    new_value = torch.randn_like(new_key)
    slots = []
    for request, (qlen, klen) in enumerate(zip(q_lengths, kv_lengths)):
        for position in range(klen - qlen, klen):
            slots.append(int(table[request, position // page_size]) * page_size + position % page_size)
    expected_cache = cache.clone()
    for index, slot in enumerate(slots):
        expected_cache[0, slot // page_size, slot % page_size] = new_key[index]
        expected_cache[1, slot // page_size, slot % page_size] = new_value[index]
    expected, offset = [], 0
    for request, (qlen, klen) in enumerate(zip(q_lengths, kv_lengths)):
        positions = torch.arange(klen)
        keys = expected_cache[0, table[request, positions // page_size].long(), positions % page_size, 0].float()
        values = expected_cache[1, table[request, positions // page_size].long(), positions % page_size, 0].float()
        logits = torch.einsum("qhd,kd->hqk", query[offset : offset + qlen].float(), keys) / math.sqrt(dim)
        relative = positions[None, :] - torch.arange(klen - qlen, klen)[:, None]
        mask = relative <= 0 if causal else torch.ones_like(relative, dtype=torch.bool)
        if window is not None:
            mask &= (relative >= -(window - 1)) & (relative <= (0 if causal else window - 1))
        logits.masked_fill_(~mask[None], float("-inf"))
        expected.append(torch.einsum("hqk,kd->qhd", logits.softmax(-1), values))
        offset += qlen
    builder = object.__new__(AscendFlashAttentionMetadataBuilder)
    builder.capture_sizes = {15} if graph_stable else set()
    builder.capture_query_len = 8 if graph_stable else None
    builder.max_num_reqs, builder.block_size = 2, page_size
    spec = (heads, 1, dim, dtype, 1 / math.sqrt(dim), 0.0, window)
    builder.scheduler_specs, builder.scheduler_buffers = {spec}, {}
    boundaries = torch.tensor([0, 7, 15], dtype=torch.int32)
    common = NS(
        num_reqs=2,
        num_actual_tokens=15,
        num_input_tokens=15,
        max_query_len=8,
        query_start_loc=boundaries.npu(),
        query_start_loc_cpu=boundaries,
        seq_lens=torch.tensor(kv_lengths, dtype=torch.int32, device="npu"),
        block_table_tensor=table.npu(),
        slot_mapping=torch.tensor(slots, device="npu"),
        causal=causal,
    )
    metadata = builder.build(0, common)
    if graph_stable:
        address = metadata.scheduler_metadata[spec].data_ptr()
        # Exercise reuse and real fingerprint validation, without claiming an
        # ACL graph replay: active requests shrink and grow in the same bucket.
        for active_tokens in (7, 15):
            boundaries[-1] = active_tokens
            common.query_start_loc.copy_(boundaries.npu())
            common.num_actual_tokens = active_tokens
            metadata = builder.build(0, common)
            assert metadata.scheduler_metadata[spec].data_ptr() == address
    impl = AscendFlashAttentionImpl(heads, dim, 1 / math.sqrt(dim), num_kv_heads=1, sliding_window=window)
    actual_cache = cache.npu()
    output = torch.empty_like(query, device="npu")
    impl.forward(
        NS(layer_name="test", _k_scale_float=1.0, _v_scale_float=1.0),
        query.npu(),
        new_key.npu(),
        new_value.npu(),
        actual_cache,
        metadata,
        output,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), torch.cat(expected), atol=0.025, rtol=0.025)
    torch.testing.assert_close(actual_cache.cpu(), expected_cache, atol=0, rtol=0)
