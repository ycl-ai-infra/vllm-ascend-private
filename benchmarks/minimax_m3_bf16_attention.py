# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Measure the A5 BF16 precision fallback against low-precision CANN GBSA.

Run after sourcing the CANN environment::

    python benchmarks/minimax_m3_bf16_attention.py --device 0

Eager measurements include Python dispatch gaps on both paths. Graph timing
is reported separately for the high-precision path, including selection and
mask construction. Neither measurement includes the model or index scoring.
This benchmark measures latency, not numerical accuracy; see the BF16 E2E
tests for the unchanged mean/max error thresholds and CPU reference checks.
"""

import argparse
import itertools
import json
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from minimax_m3_index_decode import latency_us


def benchmark(args, batch, query_len, kv_len):
    from cann_ops_transformer.ops import generic_block_sparse_attention, generic_block_sparse_attention_metadata

    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _run_bf16_sparse_attention_high_precision

    block_size, heads, dim = 128, 8, 128
    blocks = (kv_len + block_size - 1) // block_size
    topk = min(16, blocks)
    device = f"npu:{args.device}"
    torch.manual_seed(2026)
    q = torch.randn(batch * query_len, heads, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch * blocks, block_size, 1, dim, device=device, dtype=q.dtype)
    value = torch.randn_like(key)
    table = torch.randperm(batch * blocks, device=device).int().view(batch, blocks)
    prefix = torch.arange(batch + 1, device=device, dtype=torch.int64) * query_len
    lengths = torch.full((batch,), kv_len, device=device, dtype=torch.int32)
    candidates = torch.cat((torch.tensor([blocks - 1]), torch.randperm(blocks - 1)[: topk - 1])).int().to(device)
    indices = candidates[None, None].expand(1, batch * query_len, -1).contiguous()
    counts = torch.full((1, batch * query_len), topk, device=device, dtype=torch.int32)
    common = dict(
        cu_seqlens_q=prefix,
        seqused_kv=lengths,
        layout_q="TND",
        layout_kv="PA_BBND",
        mask_mode=1,
        quant_mode=0,
        softmax_precision=1,
    )

    def low_precision_gbsa():
        metadata = generic_block_sparse_attention_metadata(
            indices,
            counts,
            heads,
            1,
            dim,
            [1, block_size],
            max_seqlen_q=query_len,
            **common,
        )
        return generic_block_sparse_attention(
            q,
            key,
            value,
            indices,
            counts,
            [1, block_size],
            metadata=metadata,
            block_table=table,
            softmax_scale=dim**-0.5,
            **common,
        )[0]

    def high_precision(use_fused=True):
        return _run_bf16_sparse_attention_high_precision(
            q,
            key,
            value,
            indices,
            table,
            counts,
            prefix,
            lengths,
            num_kv_heads=1,
            sm_scale=dim**-0.5,
            block_size=block_size,
            use_fused=use_fused,
        )

    settings = SimpleNamespace(
        warmup=args.warmup,
        iterations=args.iterations,
        samples=args.samples,
        graph_batch=args.graph_batch,
        mode="eager",
    )
    eager = {
        name: latency_us(torch, fn, settings)
        for name, fn in (
            ("low_precision_gbsa", low_precision_gbsa),
            ("unfused_high_precision", lambda: high_precision(False)),
            ("high_precision", high_precision),
        )
    }
    settings.mode = "graph"
    graph = latency_us(torch, high_precision, settings)
    unfused_graph = latency_us(torch, lambda: high_precision(False), settings)
    return {
        "batch": batch,
        "query_len": query_len,
        "kv_len": kv_len,
        "query_heads": heads,
        "kv_heads": 1,
        "head_dim": dim,
        "topk": topk,
        "dtype": "bfloat16",
        "eager": eager,
        "high_precision_graph": graph,
        "unfused_high_precision_graph": unfused_graph,
        "fused_eager_speedup": eager["unfused_high_precision"]["median_us"] / eager["high_precision"]["median_us"],
        "fused_graph_speedup": unfused_graph["median_us"] / graph["median_us"],
        "eager_high_over_low": eager["high_precision"]["median_us"] / eager["low_precision_gbsa"]["median_us"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--query-len", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--kv-len", type=int, nargs="+", default=[10240])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--graph-batch", type=int, default=8)
    args = parser.parse_args()
    if (
        min(*args.batch, *args.query_len, *args.kv_len, args.warmup, args.iterations, args.samples, args.graph_batch)
        <= 0
    ):
        raise ValueError("Shapes and timing counts must be positive")
    if min(args.kv_len) < max(args.query_len):
        raise ValueError("KV lengths must be at least the query length")
    torch.npu.set_device(args.device)
    if "950" not in torch.npu.get_device_name(args.device):
        raise RuntimeError("This benchmark requires an A5 / Ascend 950 device")
    from vllm_ascend.utils import adapt_patch

    adapt_patch(True)
    adapt_patch()
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _is_bf16_sparse_attention_fia_fused_available

    print(json.dumps({"device": torch.npu.get_device_name(args.device), "settings": vars(args)}), flush=True)
    if not _is_bf16_sparse_attention_fia_fused_available():
        raise RuntimeError("Rebuild the A5 extension with fused BF16 sparse-attention kernels")
    with torch.inference_mode():
        for batch, query_len, kv_len in itertools.product(args.batch, args.query_len, args.kv_len):
            print(json.dumps(benchmark(args, batch, query_len, kv_len)), flush=True)


if __name__ == "__main__":
    main()
