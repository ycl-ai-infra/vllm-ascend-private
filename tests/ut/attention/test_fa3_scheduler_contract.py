# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the real FA3 builder/forward on CPU with only device boundaries mocked."""

import __future__

import ast
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def fa3():
    class Base:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def __class_getitem__(cls, _):
            return cls

    namespace = {
        "__name__": __name__,
        "torch": torch,
        "bisect_left": bisect_left,
        "dataclass": dataclass,
        "field": field,
        "AttentionMetadata": Base,
        "AttentionMetadataBuilder": Base,
        "AttentionImpl": Base,
        "AttentionType": NS(DECODER="decoder"),
        "CUDAGraphMode": NS(FULL_DECODE_ONLY="full_decode"),
        "logger": Mock(),
        "DeviceOperator": NS(reshape_and_cache=Mock()),
        "notify_kv_cache_written": Mock(),
        "record_attention_compute_start": Mock(),
        "get_scheduler_metadata": Mock(),
        "flash_attn_with_kvcache": Mock(),
    }
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/flash_attention_v3.py"
    names = {
        "_get_window_size",
        "AscendFlashAttentionMetadata",
        "AscendFlashAttentionMetadataBuilder",
        "AscendFlashAttentionImpl",
    }
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in names]
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace


def make_builder(fa3, *, graph=False, softcap=0.0):
    impl = fa3["AscendFlashAttentionImpl"](16, 128, 0.123, num_kv_heads=1, logits_soft_cap=softcap, sliding_window=4096)
    config = NS(
        scheduler_config=NS(max_num_seqs=3),
        speculative_config=NS(num_speculative_tokens=7),
        compilation_config=NS(
            cudagraph_capture_sizes=[16, 24] if graph else [],
            cudagraph_mode="full_decode",
            static_forward_context={"layer": NS(impl=impl)},
        ),
    )
    return fa3["AscendFlashAttentionMetadataBuilder"](
        NS(block_size=128, dtype=torch.bfloat16), ["layer"], config, torch.device("cpu")
    )


def make_common():
    offsets = torch.tensor([0, 8, 16, 16], dtype=torch.int32)
    return NS(
        num_reqs=3,
        num_actual_tokens=16,
        num_input_tokens=24,
        max_query_len=8,
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=torch.tensor([137, 73, 0], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
        slot_mapping=torch.arange(24),
        causal=True,
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("softcap", [0.0, 2.0])
def test_scheduler_signature_active_prefix_and_execution_parameters(fa3, causal, softcap):
    builder = make_builder(fa3, softcap=softcap)
    common = make_common()
    common.causal = causal

    # The scheduler accepts a boolean; numerical scale/softcap belong to forward.
    def scheduler(
        *,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens,
        qkv_dtype,
        cu_seqlens_q,
        page_size,
        causal,
        window_size,
        has_softcap,
        num_splits,
    ):
        assert batch_size == 2
        assert max_seqlen_q == 8 and max_seqlen_k == 256
        assert (num_heads_q, num_heads_kv, headdim, page_size, num_splits) == (16, 1, 128, 128, 0)
        assert qkv_dtype == torch.bfloat16
        assert has_softcap == (softcap > 0)
        assert window_size == (4095, 0 if causal else 4095)
        assert cache_seqlens.shape == (2,)
        assert cu_seqlens_q.tolist() == [0, 8, 16]
        return torch.tensor([1], dtype=torch.uint8)

    fa3["get_scheduler_metadata"] = scheduler
    metadata = builder.build(0, common)
    assert metadata.block_tables.shape == (2, 2)
    query = torch.zeros(16, 16, 128, dtype=torch.bfloat16)
    fa3["flash_attn_with_kvcache"].return_value = query
    impl = fa3["AscendFlashAttentionImpl"](16, 128, 0.123, num_kv_heads=1, logits_soft_cap=softcap, sliding_window=4096)
    impl.forward(NS(_k_scale_float=1.0, _v_scale_float=1.0), query, None, None, (None, None), metadata, query.clone())
    arguments = fa3["flash_attn_with_kvcache"].call_args.kwargs
    assert arguments["softmax_scale"] == 0.123
    assert arguments["softcap"] == softcap
    assert arguments["causal"] == causal
    assert arguments["window_size"] == (4095, 0 if causal else 4095)
    assert arguments["scheduler_metadata"] is next(iter(metadata.scheduler_metadata.values()))


@pytest.mark.parametrize("fingerprint", [False, True])
def test_graph_metadata_preserves_address_and_refreshes_contents(fa3, fingerprint):
    builder = make_builder(fa3, graph=True)
    common = make_common()
    created = []

    def scheduler(**kwargs):
        batch = kwargs["batch_size"]
        # FA3's byte metadata contains a fixed header and an int32 KV tail.
        tiling = torch.full((3584 + 4 * (batch + 1),), batch, dtype=torch.uint8)
        if fingerprint:
            tiling._fa_scheduler_params = {"batch_size": batch}
        created.append(tiling)
        return tiling

    fa3["get_scheduler_metadata"] = scheduler
    addresses = []
    for batch in (2, 1, 3, 2):
        common.query_start_loc.copy_(torch.tensor([8 * min(i, batch) for i in range(4)], dtype=torch.int32))
        common.num_actual_tokens = batch * 8
        metadata = builder.build(0, common)
        tiling = next(iter(metadata.scheduler_metadata.values()))
        addresses.append(tiling.data_ptr())
        assert tiling.numel() == 3584 + 4 * (batch + 1)
        assert torch.all(tiling == batch)
        if fingerprint:
            assert tiling._fa_scheduler_params == {"batch_size": batch}
            assert tiling._fa_scheduler_params is not created[-1]._fa_scheduler_params
        else:
            assert not hasattr(tiling, "_fa_scheduler_params")
    assert len(set(addresses)) == 1

    common.num_input_tokens = 16
    other_bucket = next(iter(builder.build(0, common).scheduler_metadata.values()))
    assert other_bucket.data_ptr() != addresses[0]
    common.causal = False
    other_causality = next(iter(builder.build(0, common).scheduler_metadata.values()))
    assert other_causality.data_ptr() not in (addresses[0], other_bucket.data_ptr())
    common.query_start_loc = common.query_start_loc.clone()
    other_input = next(iter(builder.build(0, common).scheduler_metadata.values()))
    assert other_input.data_ptr() != other_causality.data_ptr()


def test_graph_metadata_rejects_growth_beyond_reserved_storage(fa3):
    builder = make_builder(fa3, graph=True)
    common = make_common()
    fa3["get_scheduler_metadata"].return_value = torch.zeros(8, dtype=torch.uint8)
    metadata = next(iter(builder.build(0, common).scheduler_metadata.values()))
    address = metadata.data_ptr()
    fa3["get_scheduler_metadata"].return_value = torch.zeros(1024, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="graph-stable buffer capacity"):
        builder.build(0, common)

    assert metadata.data_ptr() == address


def test_dummy_forward_keeps_one_request_and_eager_tensor_identity(fa3):
    builder = make_builder(fa3)
    common = make_common()
    common.num_actual_tokens = 0
    common.query_start_loc.zero_()
    original = torch.zeros(8, dtype=torch.uint8)
    original._fa_scheduler_params = {"batch_size": 1}
    fa3["get_scheduler_metadata"].return_value = original

    metadata = builder.build(0, common)

    assert fa3["get_scheduler_metadata"].call_args.kwargs["batch_size"] == 1
    assert metadata.query_start_loc.numel() == 2
    assert metadata.seq_lens.numel() == metadata.block_tables.shape[0] == 1
    assert next(iter(metadata.scheduler_metadata.values())) is original
