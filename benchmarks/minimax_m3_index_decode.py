# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare full-table A5 FP8 MiniMax index decode without model weights.

Run in the prepared A5 vLLM-Ascend environment after sourcing CANN::

    python benchmarks/minimax_m3_index_decode.py --device 0

Defaults cover batch={1,8}, query_len={1,2}, KV={10240,65536}, with one
per-rank index head, dimension 128, and a complete randomly mapped block
table. No TP process group is needed. All paths include score, TopK, and
valid-count generation. The unfused AscendC baseline retains separate counts.
GBSA itself is excluded. This benchmark checks output contracts, not numerical
precision; use test_msa_index_score.py for precision coverage.

Graph timing batches calls to reduce host launch gaps. Eager Event timing
includes host dispatch gaps. Compilation, warmup, and capture are excluded.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections.abc import Callable
from typing import Any


def latency_us(torch: Any, fn: Callable[[], Any], args: argparse.Namespace) -> dict[str, float]:
    for _ in range(args.warmup):
        fn()
    torch.npu.synchronize()
    graph, retained = None, None
    divisor = args.iterations
    if args.mode == "graph":
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            retained = [fn() for _ in range(args.graph_batch)]
        for _ in range(args.warmup):
            graph.replay()
        torch.npu.synchronize()
        divisor *= args.graph_batch
    run = graph.replay if graph is not None else fn
    samples = []
    for _ in range(args.samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            result = run()
            if graph is None:
                retained = result
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / divisor)
    # Keep captured output storage alive until timing and synchronization finish.
    del retained, graph
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples)}


def benchmark(
    torch: Any,
    npu_ops: Any,
    triton_decode: Callable[..., Any],
    args: argparse.Namespace,
    batch: int,
    query_len: int,
    kv_len: int,
) -> dict[str, Any]:
    if min(batch, query_len, kv_len) <= 0 or kv_len < query_len:
        raise ValueError("Require batch > 0 and kv_len >= query_len > 0")
    block_size, heads, head_dim = 128, 1, 128
    blocks = (kv_len + block_size - 1) // block_size
    device = f"npu:{args.device}"
    generator = torch.Generator().manual_seed(args.seed + batch + query_len + kv_len)
    query = torch.randn(batch * query_len, heads, head_dim, generator=generator).to(torch.bfloat16).to(device)
    cache = torch.randn(batch * blocks, block_size, head_dim, generator=generator).to(torch.bfloat16).to(device)
    query = query.clamp(-448, 448).to(torch.float8_e4m3fn)
    cache = cache.clamp(-448, 448).to(torch.float8_e4m3fn)
    table = torch.randperm(batch * blocks, generator=generator).to(torch.int32).view(batch, blocks).to(device)
    seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
    context = torch.full((batch,), kv_len - query_len, dtype=torch.int32, device=device)
    prefix = torch.arange(batch + 1, dtype=torch.int32, device=device) * query_len
    start_loc = torch.div(context, block_size, rounding_mode="floor")
    mask = torch.zeros(2048, 2048, dtype=torch.int8, device=device)

    def ascendc():
        return npu_ops.minimax_m3_index_decode(
            query,
            cache,
            table,
            prefix,
            seq_lens,
            context,
            start_loc,
            mask,
            topk=args.topk,
            init_blocks=0,
            local_blocks=1,
            decode_query_len=query_len,
            return_select_num_idx=True,
        )

    def ascendc_unfused():
        indices, _ = npu_ops._minimax_m3_index_decode(
            query,
            cache,
            table,
            prefix,
            seq_lens,
            context,
            start_loc,
            mask,
            topk=args.topk,
            init_blocks=0,
            local_blocks=1,
            decode_query_len=query_len,
        )
        return indices, npu_ops._select_num_idx_from_topk(indices)

    def triton():
        return triton_decode(query, cache, table, seq_lens, kv_len, args.topk, 0, 1, heads, query_len)

    outputs = {}
    expected_counts = (kv_len - query_len + torch.arange(query_len) + block_size) // block_size
    expected_counts = expected_counts.clamp(max=args.topk).repeat(batch).unsqueeze(0).to(torch.int32)
    for name, fn in (("ascendc", ascendc), ("ascendc_unfused", ascendc_unfused), ("triton", triton)):
        for _ in range(args.warmup):
            indices, counts = fn()
        indices, counts = indices.cpu(), counts.cpu()
        assert indices.shape == (heads, batch * query_len, args.topk), name
        assert indices.dtype == torch.int32 and counts.dtype == torch.int32, name
        torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
        torch.testing.assert_close((indices >= 0).sum(-1).to(torch.int32), counts, rtol=0, atol=0)
        assert ((indices >= -1) & (indices < blocks)).all(), name
        outputs[name] = indices
    # The fusion must preserve the original score/TopK result (ordering of
    # tied blocks is not specified by torch.topk).
    torch.testing.assert_close(
        outputs["ascendc"].sort(-1).values, outputs["ascendc_unfused"].sort(-1).values, rtol=0, atol=0
    )
    overlap = []
    for row in range(batch * query_len):
        left = set(outputs["ascendc"][0, row].tolist()) - {-1}
        right = set(outputs["triton"][0, row].tolist()) - {-1}
        overlap.append(len(left & right) / max(1, len(left)))
    functions = [("ascendc", ascendc), ("ascendc_unfused", ascendc_unfused), ("triton", triton)]
    if (batch + query_len) % 2:
        functions.reverse()
    times = {name: latency_us(torch, fn, args) for name, fn in functions}
    return {
        "batch": batch,
        "query_len": query_len,
        "kv_len": kv_len,
        "index_heads": heads,
        "head_dim": head_dim,
        "block_table_width": blocks,
        "topk": args.topk,
        "dtype": "float8_e4m3fn",
        "mode": args.mode,
        "contract_check": "passed",
        "topk_set_overlap_mean": statistics.mean(overlap),
        "latencies": times,
        "triton_over_ascendc": times["triton"]["median_us"] / times["ascendc"]["median_us"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--query-len", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--kv-len", type=int, nargs="+", default=[10240, 65536])
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--mode", choices=["graph", "eager"], default="graph")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--graph-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-adapt-patch", action="store_true")
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.samples, args.graph_batch, args.topk) <= 0:
        raise ValueError("Timing counts and topk must be positive")
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.utils import adapt_patch, bootstrap_custom_op_env

    torch.npu.set_device(args.device)
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    if not args.skip_adapt_patch:
        adapt_patch(is_global_patch=True)
        adapt_patch()
    from vllm_ascend.models.minimax_m3.ops import msa_m3_npu
    from vllm_ascend.models.minimax_m3.ops.msa_m3_triton_a5 import minimax_m3_index_decode

    name = torch.npu.get_device_name(args.device)
    if "950" not in name:
        raise RuntimeError(f"A5/Ascend 950 required, found {name!r}")
    if not hasattr(torch.ops._C_ascend, "npu_msa_index_score"):
        raise RuntimeError("Rebuild the A5 extension/operator: npu_msa_index_score is not registered")
    print(
        json.dumps(
            {"device": name, "torch": torch.__version__, "settings": vars(args), "npu_ops_source": msa_m3_npu.__file__}
        ),
        flush=True,
    )
    with torch.inference_mode():
        for batch, query_len, kv_len in itertools.product(args.batch, args.query_len, args.kv_len):
            result = benchmark(torch, msa_m3_npu, minimax_m3_index_decode, args, batch, query_len, kv_len)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
