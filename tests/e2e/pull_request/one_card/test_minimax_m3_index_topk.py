# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""A5 fused decode TopK contracts, including graph-replayed device metadata."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    "950" not in torch.npu.get_device_name(0), reason="A5 fused index TopK requires Ascend 950"
)


def _check(score, context, qlen, topk, init_blocks, local_blocks, actual):
    # Build the reference on CPU without reusing the production masking helper.
    source, lengths = score.cpu(), context.cpu()
    heads, tokens, columns = source.shape
    reference = source.clone()
    visible = []
    for token in range(tokens):
        blocks = max(0, (int(lengths[token // qlen]) + token % qlen + 128) // 128)
        valid = min(columns, blocks)
        visible.append(valid)
        reference[:, token, valid:] = -torch.inf
        reference[:, token, : min(valid, init_blocks)] = 1.0e30
        start = max(0, blocks - local_blocks)
        if local_blocks and start < valid:
            reference[:, token, start:valid] = 1.0e29
    expected = torch.full((heads, tokens, topk), -torch.inf)
    expected[..., : min(topk, columns)] = reference.topk(min(topk, columns), dim=-1).values
    indices, values, counts = (tensor.cpu() for tensor in actual)
    assert indices.dtype == counts.dtype == torch.int32 and values.dtype == torch.float32
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    expected_counts = torch.tensor([min(topk, valid) for valid in visible], dtype=torch.int32)
    torch.testing.assert_close(counts, expected_counts.expand(heads, -1), rtol=0, atol=0)
    torch.testing.assert_close((indices >= 0).sum(-1).int(), counts, rtol=0, atol=0)
    for head in range(heads):
        for token in range(tokens):
            valid_ids = indices[head, token, : counts[head, token]].long()
            assert len(valid_ids.unique()) == len(valid_ids)
            assert ((valid_ids >= 0) & (valid_ids < visible[token])).all()
            torch.testing.assert_close(
                reference[head, token, valid_ids], values[head, token, : len(valid_ids)], rtol=0, atol=0
            )
            assert (indices[head, token, len(valid_ids) :] == -1).all()


@pytest.mark.parametrize(
    "columns,topk,qlen,heads,batch,init_blocks,local_blocks",
    [
        (3, 8, 2, 2, 3, 0, 1),
        (80, 16, 1, 1, 8, 0, 1),
        (129, 16, 4, 2, 35, 3, 2),
        (512, 16, 2, 1, 8, 4, 6),
        (2048, 128, 1, 1, 8, 0, 1),
        (16, 16, 4, 2, 3, 8, 16),
        (80, 16, 1, 1, 9, 0, 0),
        (80, 16, 2, 2, 9, 32, 1),
    ],
)
def test_index_topk_matches_reference_and_replays(columns, topk, qlen, heads, batch, init_blocks, local_blocks):
    torch.manual_seed(2026)
    tokens = batch * qlen
    padded = (columns + 15) // 16 * 16
    # Both token and head strides may exceed the logical score dimensions.
    storage = torch.randn(heads, tokens * 2, padded, device="npu")
    score = storage[:, ::2, :columns]
    lengths = torch.tensor(
        [(-qlen, 0, 127, 128, columns * 128 - qlen, (columns + 7) * 128)[i % 6] for i in range(batch)],
        dtype=torch.int32,
        device="npu",
    )

    def run():
        return torch.ops._C_ascend.npu_msa_index_decode_topk(score, lengths, qlen, topk, init_blocks, local_blocks)

    for _ in range(3):
        _check(score, lengths, qlen, topk, init_blocks, local_blocks, run())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for shift in (129, -256, 8192):
        lengths.add_(shift)
        graph.replay()
        _check(score, lengths, qlen, topk, init_blocks, local_blocks, captured)


def test_index_topk_meta_and_invalid_contracts():
    op = torch.ops._C_ascend.npu_msa_index_decode_topk
    indices, values, counts = op(
        torch.empty(2, 6, 80, device="meta"), torch.empty(3, device="meta", dtype=torch.int32), 2, 16
    )
    assert indices.shape == values.shape == (2, 6, 16)
    assert counts.shape == (2, 6)
    assert indices.dtype == counts.dtype == torch.int32 and values.dtype == torch.float32
    context = torch.zeros(1, device="npu", dtype=torch.int32)
    score = torch.zeros(1, 1, 80, device="npu")
    for bad_score, bad_context, qlen, topk in (
        (score.bfloat16(), context, 1, 16),
        (score, context.long(), 1, 16),
        (score, context, 0, 16),
        (score, context, 1, 7),
        (torch.zeros(1, 1, 2049, device="npu"), context, 1, 16),
    ):
        with pytest.raises(RuntimeError):
            op(bad_score, bad_context, qlen, topk)
