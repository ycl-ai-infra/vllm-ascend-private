# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor
from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm


def _reference(x, weight, epsilon, residual=None):
    values = x.float()
    if residual is not None:
        values = values + residual.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
    return (normalized * (weight.float() + 1.0)).to(x.dtype), values.to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_a5_gemma_norm_offsets_weight_in_fp32(dtype):
    generator = torch.Generator().manual_seed(39)
    x = torch.randn(7, 256, generator=generator).to(dtype)
    weight = (torch.randn(256, generator=generator) * 0.2 - 0.2).to(dtype)
    expected, _ = _reference(x, weight, 1e-6)

    def rms_norm(values, gamma, epsilon):
        assert gamma.dtype == torch.float32
        torch.testing.assert_close(gamma, weight.float() + 1.0, atol=0, rtol=0)
        values = values.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
        return (normalized * gamma).to(x.dtype), None

    with patch("vllm_ascend.device.device_op.torch_npu.npu_rms_norm", side_effect=rms_norm):
        actual = A5DeviceAdaptor.npu_gemma_rms_norm(x, weight, 1e-6)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gemma_residual_fallback_preserves_fp32_sum_and_weight(dtype):
    generator = torch.Generator().manual_seed(39)
    x = torch.randn(7, 256, generator=generator).to(dtype)
    residual = torch.randn(7, 256, generator=generator).to(dtype)
    weight = (torch.randn(256, generator=generator) * 0.2 - 0.2).to(dtype)
    original_x, original_residual = x.clone(), residual.clone()
    expected, expected_residual = _reference(x, weight, 1e-6, residual)
    layer = SimpleNamespace(weight=weight, variance_epsilon=1e-6)

    def add_rms_norm(values, skip, gamma, epsilon):
        assert values.dtype == skip.dtype == gamma.dtype == torch.float32
        torch.testing.assert_close(gamma, weight.float() + 1.0, atol=0, rtol=0)
        summed = values + skip
        normalized = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + epsilon)
        return normalized * gamma, None, summed

    with (
        patch("vllm_ascend.ops.layernorm.enable_custom_op", return_value=False),
        patch("torch_npu.npu_add_rms_norm", side_effect=add_rms_norm, create=True),
    ):
        actual, actual_residual = AscendGemmaRMSNorm.forward_oot(layer, x, residual)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(x, original_x, atol=0, rtol=0)
    torch.testing.assert_close(residual, original_residual, atol=0, rtol=0)
    assert actual.dtype == actual_residual.dtype == dtype
