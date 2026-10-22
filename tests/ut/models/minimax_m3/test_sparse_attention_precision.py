# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression coverage for paged BF16 high-precision sparse attention."""

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.minimax_m3.ops import msa_m3_npu as ops
from vllm_ascend.utils import AscendDeviceType


@pytest.mark.parametrize("max_batch", [2, 128])
@pytest.mark.parametrize("strided_pages", [False, True])
def test_high_precision_paged_attention_preserves_head_selection_and_causality(max_batch, strided_pages):
    q = torch.zeros(4, 4, 4, dtype=torch.bfloat16)
    cache = torch.empty(3, 2, 2, 4, dtype=torch.bfloat16)
    for page in range(3):
        for row in range(2):
            for head in range(2):
                cache[page, row, head].fill_(page * 10 + row + head * 100)
    if strided_pages:
        storage = torch.full((6, 2, 2, 4), -1000, dtype=torch.bfloat16)
        storage[::2].copy_(cache)
        cache = storage[::2]
    table = torch.tensor([[1, 0], [2, 1], [0, 0]], dtype=torch.int32)
    indices = torch.tensor(
        [[[1, 0], [1, 0], [1, 0], [-1, -1]], [[0, -1], [0, -1], [0, 1], [-1, -1]]], dtype=torch.int32
    )
    counts = torch.tensor([[2, 2, 1, 0], [1, 1, 1, 0]], dtype=torch.int32)
    calls = []

    def paged_fia(query, key, value, **kw):
        assert query.dtype == key.dtype == value.dtype == torch.bfloat16
        assert key.data_ptr() == cache.data_ptr() == value.data_ptr()
        assert kw["inner_precise"] == 0 and kw["input_layout"] == "BSH"
        assert kw["actual_seq_kvlen"] == [4] * query.shape[0]
        calls.append(query.shape[0])
        heads, dim = kw["num_query_heads"], query.shape[-1] // kw["num_query_heads"]
        outputs = []
        for batch in range(query.shape[0]):
            physical = kw["block_table"][batch].long()
            keys = key[physical].reshape(-1, 2, dim).float()
            values = value[physical].reshape(-1, 2, dim).float()
            mask = kw["atten_mask"][batch, 0, 0]
            groups = []
            for head in range(heads):
                logits = keys[:, head // 2] @ query[batch, 0].view(heads, dim)[head].float()
                weights = (logits * kw["softmax_scale"]).masked_fill(mask, -float("inf")).softmax(0)
                groups.append(weights @ values[:, head // 2])
            outputs.append(torch.stack(groups).reshape(1, -1))
        return torch.stack(outputs).to(query.dtype), torch.empty(0)

    with (
        patch.object(ops, "_GBSA_BF16_MAX_QUERY_BATCH", max_batch),
        patch.object(ops.torch_npu, "npu_fused_infer_attention_score_v2", side_effect=paged_fia, create=True),
    ):
        actual = ops._run_bf16_sparse_attention_high_precision(
            q,
            cache,
            cache,
            indices,
            table,
            counts,
            torch.tensor([0, 2, 3, 4]),
            torch.tensor([3, 4, 0]),
            num_kv_heads=2,
            sm_scale=0.5,
            block_size=2,
        )
    expected = torch.tensor(
        [[10.5, 10.5, 110.5, 110.5], [7, 7, 110.5, 110.5], [10.5, 10.5, 120.5, 120.5], [0, 0, 0, 0]],
        dtype=torch.bfloat16,
    )[..., None].expand_as(q)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert max(calls) <= max_batch


@pytest.mark.parametrize("empty", ["query", "selection", "cache"])
def test_high_precision_empty_attention_is_zero(empty):
    q = torch.zeros(0 if empty == "query" else 1, 2, 4, dtype=torch.bfloat16)
    cache = torch.empty(0 if empty == "cache" else 1, 2, 1, 4, dtype=q.dtype)
    indices = torch.empty(1, q.shape[0], 0 if empty == "selection" else 1, dtype=torch.int32)
    with patch.object(ops.torch_npu, "npu_fused_infer_attention_score_v2", create=True) as fia:
        actual = ops._run_bf16_sparse_attention_high_precision(
            q,
            cache,
            cache,
            indices,
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, q.shape[0], dtype=torch.int32),
            torch.tensor([0, q.shape[0]]),
            torch.tensor([0]),
            num_kv_heads=1,
            sm_scale=0.5,
            block_size=2,
        )
    fia.assert_not_called()
    torch.testing.assert_close(actual, torch.zeros_like(q))


@pytest.mark.parametrize("split_kv_available", [False, True])
def test_a5_bf16_prefill_keeps_high_precision_with_optional_split_kv(split_kv_available):
    q = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    cache = torch.zeros(2, 1, 128, 1, 4, dtype=q.dtype)
    with (
        patch.object(ops, "get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A5)),
        patch.object(ops, "_is_minimax_sparse_attention_split_kv_available", return_value=split_kv_available),
        patch.object(ops, "_run_bf16_sparse_attention_high_precision", return_value=torch.ones_like(q)) as high,
        patch.object(ops, "_minimax_m3_sparse_attn_kv_gather_q") as split_kv,
    ):
        output = torch.empty_like(q)
        ops.minimax_m3_sparse_attn(
            q,
            cache,
            torch.zeros(1, 1, 1, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.tensor([0, 1]),
            torch.tensor([1]),
            torch.tensor([0]),
            1,
            1,
            0.5,
            output,
        )
    high.assert_called_once()
    split_kv.assert_not_called()
    torch.testing.assert_close(output, torch.ones_like(q))


def test_high_precision_rejects_incomplete_gqa_groups():
    with pytest.raises(ValueError, match="GQA groups"):
        ops._run_bf16_sparse_attention_high_precision(
            torch.zeros(1, 3, 4),
            torch.zeros(1, 2, 2, 4),
            torch.zeros(1, 2, 2, 4),
            torch.zeros(2, 1, 1),
            torch.zeros(1, 1),
            torch.ones(2, 1),
            torch.tensor([0, 1]),
            torch.tensor([1]),
            num_kv_heads=2,
            sm_scale=0.5,
            block_size=2,
        )
