# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.device.device_op import A5DeviceAdaptor, DeviceOperator
from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3SwiGLUOAI
from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicFusedMoEMethod
from vllm_ascend.quantization.nvfp4_utils import quantize_mxfp8


def _vllm_native_mxfp8_swiglu_reference(inputs, limit, alpha, beta, output_dtype=None):
    # vLLM main b1c66e55, models/minimax_m3/amd/ops/swiglu_oai.py.
    # Native MXFP8 keeps all activation intermediates in FP32 until quantization.
    gate, up = inputs.float().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    activated = gate * torch.sigmoid(alpha * gate) * (up + beta)
    return activated.to(inputs.dtype if output_dtype is None else output_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(("limit", "alpha", "beta"), [(7.0, 1.702, 1.0), (5.0, 1.1, 0.25)])
def test_a5_clipped_swiglu_matches_native_fp32_intermediates(dtype, limit, alpha, beta):
    inputs = (torch.randn(7, 256, generator=torch.Generator().manual_seed(17)) * 2).to(dtype)
    original = inputs.clone()
    expected = _vllm_native_mxfp8_swiglu_reference(inputs, limit, alpha, beta)
    actual = A5DeviceAdaptor.clipped_swiglu(inputs, limit, alpha, beta)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(inputs, original, atol=0, rtol=0)
    assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_minimax_activation_reaches_mxfp8_quantizer_with_vllm_rounding(dtype):
    inputs = (torch.randn(7, 256, generator=torch.Generator().manual_seed(17)) * 2).to(dtype)
    expected = _vllm_native_mxfp8_swiglu_reference(inputs, 7.0, 1.702, 1.0)
    reference, reference_scale = quantize_mxfp8(expected)
    seen = []

    def quantize(activated, *, act_quant_type, use_mxfp_quant):
        assert act_quant_type == torch.float8_e4m3fn and use_mxfp_quant
        seen.append(activated)
        return quantize_mxfp8(activated)

    with patch.object(DeviceOperator, "npu_dynamic_quant", side_effect=quantize):
        actual, scales = MiniMaxM3SwiGLUOAI(alpha=1.702, beta=1.0, limit=7.0, use_mx_quant=True)(inputs)
    torch.testing.assert_close(seen[0], expected, atol=0, rtol=0)
    assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
    assert torch.equal(scales, reference_scale)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_mxfp8_moe_quantizes_fp32_activation_without_bf16_roundtrip(dtype):
    inputs = (torch.randn(7, 256, generator=torch.Generator().manual_seed(17)) * 2).to(dtype)
    expected = _vllm_native_mxfp8_swiglu_reference(inputs, 7.0, 1.702, 1.0, torch.float32)
    expected_data, expected_scale = quantize_mxfp8(expected)
    method = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    mlp = SimpleNamespace(
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        hidden_states=inputs,
        quant=SimpleNamespace(is_quant=False),
    )
    observed_scales = []

    def quantize(activated, *, dst_type, scale_alg):
        assert activated.dtype == torch.float32
        assert dst_type == torch.float8_e4m3fn and scale_alg == 1
        torch.testing.assert_close(activated, expected, atol=0, rtol=0)
        return quantize_mxfp8(activated)

    def gmm2(unused_mlp, data, scales):
        observed_scales.append(scales)
        return data

    with (
        patch.object(method, "apply_gmm1", return_value=inputs),
        patch.object(method, "apply_gmm2", side_effect=gmm2),
        patch.object(DeviceOperator, "clipped_swiglu", side_effect=A5DeviceAdaptor.clipped_swiglu),
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg", return_value=1),
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.torch_npu.npu_dynamic_mx_quant", side_effect=quantize),
    ):
        actual, _ = apply_moe_mlp(mlp, method)
    assert method.supports_fused_activation(MoEActivation.SWIGLUOAI_UNINTERLEAVE)
    assert torch.equal(actual.view(torch.uint8), expected_data.view(torch.uint8))
    assert torch.equal(observed_scales[0].reshape(expected_scale.shape), expected_scale)
