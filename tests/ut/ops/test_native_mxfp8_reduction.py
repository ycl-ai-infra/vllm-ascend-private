# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend import ascend_forward_context as context
from vllm_ascend.device.hardware_profile import MoECommPolicy
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicFusedMoEMethod
from vllm_ascend.quantization.utils import uses_minimax_mxfp8_fp32_reduction


def test_native_fp32_reduction_uses_graph_compatible_allgather():
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True, use_sequence_parallel_moe=True), lora_config=None
    )
    policy = MoECommPolicy.CAPACITY_AND_WORLD_SIZE
    with (
        patch.object(context, "is_moe_model", return_value=True),
        patch.object(context, "get_mc2_tokens_capacity", return_value=1024),
        patch.object(context, "get_ep_group", return_value=SimpleNamespace(world_size=8)),
        patch.object(context, "get_current_hardware_profile", return_value=SimpleNamespace(moe_comm_policy=policy)),
        patch.dict(context._MOE_COMM_SELECTORS, {policy: lambda *args: context.MoECommType.MC2}),
        patch("vllm_ascend.quantization.utils.uses_minimax_mxfp8_fp32_reduction", return_value=True, create=True),
    ):
        assert context.select_moe_comm_method(32, config) == context.MoECommType.ALLGATHER


@pytest.mark.parametrize("native", [False, True])
def test_gmm2_preserves_native_fp32_before_router_weighting(native):
    method = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    method._use_fp32_moe_output = native
    method.use_expert_weight_list = False
    inputs = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
    mlp = SimpleNamespace(
        hidden_states=torch.empty(2, 64, dtype=torch.bfloat16),
        layer=SimpleNamespace(w2_weight=torch.empty(1), w2_weight_scale=torch.empty(1)),
        group_list=torch.tensor([2]),
        group_list_type=1,
    )
    expected_dtype = torch.float32 if native else torch.bfloat16

    def gmm(**kwargs):
        assert kwargs["output_dtype"] == expected_dtype
        return [torch.empty(2, 64, dtype=expected_dtype)]

    with patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.torch_npu.npu_grouped_matmul", side_effect=gmm):
        actual = method.apply_gmm2(mlp, inputs, torch.empty(1))
    assert actual.dtype == expected_dtype


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("ep_sequence_parallel", [False, True])
def test_native_result_rounds_only_after_fp32_collective(native, ep_sequence_parallel):
    finalizer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    finalizer.moe_config = SimpleNamespace(in_dtype=torch.bfloat16)
    finalizer._use_fp32_moe_output = native
    values = torch.tensor([[1.003, 0.007]], dtype=torch.float32)
    reduced = values * 3.0

    def reduce_fp32(inputs, *args):
        assert inputs.dtype == torch.float32
        return reduced

    with (
        patch.object(finalizer, "_use_ep_sequence_parallel", return_value=ep_sequence_parallel),
        patch.object(finalizer, "_finalize_with_ep_group", side_effect=reduce_fp32),
        patch.object(finalizer, "_finalize_with_dp_group", side_effect=reduce_fp32),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.uses_minimax_mxfp8_fp32_reduction",
            return_value=native,
            create=True,
        ),
    ):
        actual = finalizer.finalize(values, reduce_results=True)
    expected_dtype = torch.bfloat16 if native else torch.float32
    assert actual.dtype == expected_dtype
    torch.testing.assert_close(actual, reduced.to(expected_dtype), atol=0, rtol=0)


@pytest.mark.parametrize(
    "variation, expected",
    [
        ("native", True),
        ("modelopt", True),
        ("other_quant", False),
        ("fp16", False),
        ("unsupported_hardware", False),
        ("modelslim", False),
        ("other_model", False),
        ("no_ep", False),
        ("no_sequence_parallel", False),
        ("no_quant_config", False),
        ("lora", False),
    ],
)
def test_fp32_reduction_policy_is_scoped_to_native_deployment(variation, expected):
    config = SimpleNamespace(
        quant_config=SimpleNamespace(get_name=lambda: "mxfp8"),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            architectures=["MiniMaxM3SparseForConditionalGeneration"],
            hf_config=SimpleNamespace(quantization_config={"moe_quant_algo": "NVFP4"}),
        ),
        parallel_config=SimpleNamespace(enable_expert_parallel=True, use_sequence_parallel_moe=True),
        lora_config=None,
    )
    if variation == "no_quant_config":
        config.quant_config = None
    if variation == "modelopt":
        config.quant_config.get_name = lambda: "modelopt_mxfp8"
    if variation == "other_quant":
        config.quant_config.get_name = lambda: "ascend"
    if variation == "fp16":
        config.model_config.dtype = torch.float16
    if variation == "modelslim":
        config.model_config.hf_config.quantization_config = {}
    if variation == "other_model":
        config.model_config.architectures = ["Qwen3ForCausalLM"]
    if variation == "no_sequence_parallel":
        config.parallel_config.use_sequence_parallel_moe = False
    if variation == "no_ep":
        config.parallel_config.enable_expert_parallel = False
    if variation == "lora":
        config.lora_config = object()
    with (
        patch(
            "vllm_ascend.quantization.utils.get_current_hardware_profile",
            return_value=SimpleNamespace(supports=lambda cap: variation != "unsupported_hardware"),
        ),
    ):
        assert uses_minimax_mxfp8_fp32_reduction(config) is expected


def test_fp32_policy_is_inactive_without_model_context():
    with patch("vllm.config.get_current_vllm_config", side_effect=AssertionError("no model context")):
        assert not uses_minimax_mxfp8_fp32_reduction()


def test_fp32_finalizer_caches_model_policy_before_graph_execution():
    with (
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_dynamic_mx_quant_scale_alg", return_value=1),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.uses_minimax_mxfp8_fp32_reduction", return_value=True),
    ):
        finalizer = PrepareAndFinalizeWithAllGather(SimpleNamespace(in_dtype=torch.bfloat16))
    values = torch.tensor([[1.003, 0.007]], dtype=torch.float32)
    with (
        patch.object(finalizer, "_use_ep_sequence_parallel", return_value=True),
        patch.object(finalizer, "_finalize_with_ep_group", return_value=values * 3.0),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.uses_minimax_mxfp8_fp32_reduction", return_value=False),
    ):
        actual = finalizer.finalize(values, reduce_results=True)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, (values * 3).bfloat16(), atol=0, rtol=0)
