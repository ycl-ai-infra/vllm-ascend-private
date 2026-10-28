# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.device.device_op import A5DeviceAdaptor, clipped_swiglu_torch
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicFusedMoEMethod
from vllm_ascend.quantization.nvfp4_utils import quantize_mxfp8, requantize_nvfp4_to_mxfp8


@pytest.mark.parametrize("shape", [(35, 6144), (32, 3072), (1, 64)])
@torch.inference_mode()
def test_activation_scale_alg_one_matches_torch_reference(shape):
    if not get_current_hardware_profile().supports(HardwareCapability.DYNAMIC_MX_QUANT_SCALE_ALG_ONE):
        pytest.skip("scale_alg=1 requires DynamicMxQuantV3 support")
    inputs = torch.randn(*shape, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(17))
    inputs[0].zero_()
    reference, reference_scale = quantize_mxfp8(inputs)
    actual, actual_scale = torch_npu.npu_dynamic_mx_quant(inputs.npu(), dst_type=torch.float8_e4m3fn, scale_alg=1)
    assert torch.equal(actual.cpu().view(torch.uint8), reference.view(torch.uint8))
    assert torch.equal(actual_scale.cpu().view(torch.uint8).reshape(reference_scale.shape), reference_scale)


@pytest.mark.parametrize("invert_global_scale", [False, True])
@torch.inference_mode()
def test_torch_nvfp4_conversion_matches_between_cpu_and_npu(invert_global_scale):
    packed = torch.arange(256, dtype=torch.uint8).reshape(4, 64)
    scales = torch.linspace(0.25, 8, 32).reshape(4, 8).to(torch.float8_e4m3fn)
    global_scale = torch.tensor(15123.6923828125)
    reference, reference_scale = requantize_nvfp4_to_mxfp8(
        packed, scales, global_scale, invert_global_scale=invert_global_scale
    )
    actual, actual_scale = requantize_nvfp4_to_mxfp8(
        packed.npu(), scales.npu(), global_scale.npu(), invert_global_scale=invert_global_scale
    )
    assert torch.equal(actual.cpu().view(torch.uint8), reference.view(torch.uint8))
    assert torch.equal(actual_scale.cpu(), reference_scale)


@pytest.mark.parametrize("shape", [(32, 6144), (1024, 6144)])
@torch.inference_mode()
def test_clipped_swiglu_rounding_before_mxfp8_matches_vllm(shape):
    if not get_current_hardware_profile().supports(HardwareCapability.DYNAMIC_MX_QUANT_SCALE_ALG_ONE):
        pytest.skip("MXFP8 activation parity requires DynamicMxQuantV3 support")
    inputs = (torch.randn(*shape, generator=torch.Generator().manual_seed(17)) * 2).to(torch.bfloat16)
    inputs[0].zero_()
    # Native vLLM MXFP8 keeps SwiGLU FP32 through the GEMM2 quantization input.
    gate, up = inputs.float().chunk(2, dim=-1)
    reference = gate.clamp(max=7.0) * torch.sigmoid(gate.clamp(max=7.0) * 1.702)
    reference = reference * (up.clamp(min=-7.0, max=7.0) + 1.0)
    actual = clipped_swiglu_torch(inputs.npu(), 7.0, 1.702, 1.0, output_dtype=torch.float32)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), reference, atol=2e-6, rtol=2e-6)
    dense = A5DeviceAdaptor.clipped_swiglu(inputs.npu(), 7.0, 1.702, 1.0)
    assert torch.equal(dense.cpu(), actual.cpu().to(inputs.dtype))
    quantized, scales = torch_npu.npu_dynamic_mx_quant(actual, dst_type=torch.float8_e4m3fn, scale_alg=1)
    expected, expected_scale = quantize_mxfp8(actual.cpu())
    assert torch.equal(quantized.cpu().view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scales.cpu().reshape(expected_scale.shape), expected_scale)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_gemma_norm_preserves_fp32_weight_offset_on_npu(dtype, with_residual):
    if not get_current_hardware_profile().supports(HardwareCapability.DYNAMIC_MX_QUANT_SCALE_ALG_ONE):
        pytest.skip("A5 Gemma precision validation requires 950DT")
    for hidden_size in (128, 6144):
        generator = torch.Generator().manual_seed(39)
        inputs = torch.randn(32, hidden_size, generator=generator).to(dtype)
        residual = torch.randn(32, hidden_size, generator=generator).to(dtype)
        weight = (torch.randn(hidden_size, generator=generator) * 0.2 - 0.2).to(dtype)
        values = inputs.float() + residual.float() if with_residual else inputs.float()
        reference = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        reference = (reference * (weight.float() + 1.0)).to(dtype)
        if with_residual:
            layer = SimpleNamespace(weight=weight.npu(), variance_epsilon=1e-6)
            with patch("vllm_ascend.ops.layernorm.enable_custom_op", return_value=False):
                actual, actual_residual = AscendGemmaRMSNorm.forward_oot(layer, inputs.npu(), residual.npu())
            assert torch.equal(actual_residual.cpu(), values.to(dtype))
        else:
            actual = A5DeviceAdaptor.npu_gemma_rms_norm(inputs.npu(), weight.npu(), 1e-6)
        actual = actual.cpu()
        assert actual.dtype == dtype
        # The fused reduction may differ at a few dtype rounding boundaries.
        # Prematurely rounding 1 + weight changes roughly a quarter of outputs.
        assert (actual != reference).float().mean() < 0.001
        torch.testing.assert_close(actual, reference, atol=1e-4, rtol=0.01)


@torch.inference_mode()
def test_native_mxfp8_moe_hook_keeps_fp32_through_actual_npu_quantizer():
    if not get_current_hardware_profile().supports(HardwareCapability.DYNAMIC_MX_QUANT_SCALE_ALG_ONE):
        pytest.skip("Native MXFP8 activation parity requires DynamicMxQuantV3 support")
    inputs = (torch.randn(32, 6144, generator=torch.Generator().manual_seed(17)) * 2).bfloat16().npu()
    inputs[0].zero_()
    expected_activation = clipped_swiglu_torch(inputs, 7.0, 1.702, 1.0, output_dtype=torch.float32)
    expected, expected_scale = quantize_mxfp8(expected_activation.cpu())
    method = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    mlp = SimpleNamespace(
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        hidden_states=inputs,
        quant=SimpleNamespace(is_quant=False),
    )
    observed = []

    def gmm2(unused_mlp, data, scales):
        observed.append(scales.cpu())
        return data

    with (
        patch.object(method, "apply_gmm1", return_value=inputs),
        patch.object(method, "apply_gmm2", side_effect=gmm2),
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg", return_value=1),
    ):
        actual, _ = apply_moe_mlp(mlp, method)
    assert torch.equal(actual.cpu().view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(observed[0].reshape(expected_scale.shape), expected_scale)


@torch.inference_mode()
def test_fp32_gmm2_local_expert_mask_and_weighted_sum():
    if not get_current_hardware_profile().supports(HardwareCapability.DYNAMIC_MX_QUANT_SCALE_ALG_ONE):
        pytest.skip("Native MXFP8 FP32 reduction validation requires A5")
    generator = torch.Generator().manual_seed(41)
    inputs = torch.randn(4, 128, generator=generator).bfloat16()
    weights = torch.randn(2, 64, 128, generator=generator).bfloat16()
    expert_ids = torch.tensor([[0, 2], [1, 3], [2, 3], [0, 1]], dtype=torch.int32)
    probs = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]])
    local_probs = probs * (expert_ids < 2)
    sorted_inputs, restore_indices, counts, _ = torch_npu.npu_moe_init_routing_v2(
        inputs.npu(),
        expert_ids.npu(),
        active_num=8,
        expert_num=4,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, 2],
        quant_mode=-1,
    )
    quantized, act_scales = torch_npu.npu_dynamic_mx_quant(sorted_inputs, dst_type=torch.float8_e4m3fn, scale_alg=1)
    weight_data, weight_scales = quantize_mxfp8(weights)
    layer = SimpleNamespace(
        w2_weight=weight_data.transpose(1, 2).contiguous().npu(),
        w2_weight_scale=weight_scales.reshape(2, 64, 2, 2).transpose(1, 2).contiguous().npu(),
    )
    method = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    method._use_fp32_moe_output = True
    method.use_expert_weight_list = False
    mlp = SimpleNamespace(
        hidden_states=sorted_inputs, layer=layer, group_list=counts.to(torch.int64), group_list_type=1
    )
    routed = method.apply_gmm2(mlp, quantized, act_scales)
    assert routed.dtype == torch.float32
    combined = torch_npu.npu_moe_token_unpermute(routed, restore_indices, probs=local_probs.npu())
    assert combined.dtype == torch.float32
    input_data, input_scales = quantize_mxfp8(inputs)
    decoded_inputs = input_data.float() * torch.exp2(input_scales.float() - 127).repeat_interleave(32, -1)
    decoded_weights = weight_data.float() * torch.exp2(weight_scales.float() - 127).repeat_interleave(32, -1)
    reference = torch.zeros(4, 64)
    for row in range(4):
        for slot in range(2):
            expert = int(expert_ids[row, slot])
            if expert < 2:
                reference[row] += (decoded_inputs[row] @ decoded_weights[expert].T) * probs[row, slot]
    torch.testing.assert_close(combined.cpu(), reference, atol=3e-5, rtol=2e-6)
    assert torch.count_nonzero(combined[2]).cpu() == 0
