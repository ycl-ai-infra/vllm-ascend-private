# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from itertools import permutations

import pytest
import torch

from vllm_ascend.quantization.nvfp4_utils import (
    dequantize_nvfp4,
    quantize_mxfp8,
    requantize_minimax_nvfp4_experts,
    requantize_nvfp4_to_mxfp8,
)


def test_all_nibbles_and_block_scale_bytes():
    packed = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    scales = torch.full((16, 2), 2.0).to(torch.float8_e4m3fn)
    expected = []
    magnitudes = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
    for byte in range(256):
        for nibble in (byte & 15, byte >> 4):
            expected.append(float(magnitudes[nibble & 7]) * (-1.0 if nibble & 8 else 1.0) * 6.0)
    expected = torch.tensor(expected).reshape(16, 32).bfloat16()
    for scale in (scales, scales.view(torch.uint8)):
        actual = dequantize_nvfp4(packed, scale, torch.tensor(3.0))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.parametrize("invert", [False, True])
def test_expert_scales_are_independent(invert):
    packed = torch.full((3, 2, 16), 0x22, dtype=torch.uint8)
    scales = torch.ones(3, 2, 2).to(torch.float8_e4m3fn)
    global_scale = torch.tensor([0.5, 2.0, 8.0])
    actual = dequantize_nvfp4(packed, scales, global_scale, invert_global_scale=invert)
    expected_scale = global_scale.reciprocal() if invert else global_scale
    expected = expected_scale[:, None, None].expand(3, 2, 32).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_conversion_rounds_to_model_dtype_before_requantization(dtype):
    packed = torch.randint(0, 256, (7, 64), generator=torch.Generator().manual_seed(17), dtype=torch.uint8)
    scales = torch.linspace(0.05, 7, 56).reshape(7, 8).to(torch.float8_e4m3fn)
    global_scale = torch.tensor(0.1234567)
    decoded = dequantize_nvfp4(packed, scales, global_scale, dtype)
    actual, actual_scale = requantize_nvfp4_to_mxfp8(packed, scales, global_scale, dtype)
    # A scalar reference checks the rounding and scale exponent independently.
    expected = torch.empty(decoded.shape, dtype=torch.float8_e4m3fn)
    expected_scale = torch.empty(7, 4, dtype=torch.uint8)
    for row in range(7):
        for group in range(4):
            block = decoded[row, group * 32 : (group + 1) * 32].float()
            exponent = max(-127, min(127, math.ceil(math.log2(max(block.abs().max().item(), 2**-126) / 448))))
            expected[row, group * 32 : (group + 1) * 32] = block / 2**exponent
            expected_scale[row, group] = exponent + 127
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(actual_scale, expected_scale)


def test_mxfp8_zero_blocks_and_ceil_boundaries():
    # Values immediately above 448*2**n require the next exponent, rather
    # than the floor(log2(amax))-8 rule used by scale_alg=0.
    maxima = torch.tensor([0, 448, 449, 224, 225, 1, 1e-30], dtype=torch.float32)
    weight = maxima[:, None].expand(-1, 32).contiguous()
    actual, scales = quantize_mxfp8(weight)
    assert scales.flatten().tolist() == [0, 127, 128, 126, 127, 119, 19]
    assert torch.isfinite(actual.float()).all()
    assert torch.count_nonzero(actual[0].float()) == 0


@pytest.mark.parametrize("shape", [(2, 32), (3, 2, 64)])
def test_noncontiguous_input(shape):
    source = torch.randn(*shape, 2, generator=torch.Generator().manual_seed(6))[..., 0]
    actual, scale = quantize_mxfp8(source)
    expected, expected_scale = quantize_mxfp8(source.contiguous())
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scale, expected_scale)


@pytest.mark.parametrize(
    "packed,scales,global_scale",
    [
        (torch.zeros(2, 16), torch.ones(2, 2, dtype=torch.uint8), torch.tensor(1.0)),
        (torch.zeros(2, 7, dtype=torch.uint8), torch.ones(2, 1, dtype=torch.uint8), torch.tensor(1.0)),
        (torch.zeros(2, 16, dtype=torch.uint8), torch.ones(2, 1, dtype=torch.uint8), torch.tensor(1.0)),
        (torch.zeros(2, 16, dtype=torch.uint8), torch.ones(2, 2), torch.tensor(1.0)),
        (torch.zeros(3, 2, 16, dtype=torch.uint8), torch.ones(3, 2, 2, dtype=torch.uint8), torch.ones(2)),
    ],
)
def test_invalid_nvfp4_schema(packed, scales, global_scale):
    with pytest.raises(ValueError):
        dequantize_nvfp4(packed, scales, global_scale)


@pytest.mark.parametrize("weight", [torch.ones(32), torch.ones(2, 31), torch.ones(2, 32, dtype=torch.int32)])
def test_invalid_mxfp8_schema(weight):
    with pytest.raises(ValueError):
        quantize_mxfp8(weight)


def test_incompatible_conversion_width():
    with pytest.raises(ValueError, match="divisible by 32"):
        requantize_nvfp4_to_mxfp8(
            torch.zeros(2, 8, dtype=torch.uint8), torch.ones(2, 1, dtype=torch.uint8), torch.tensor(1.0)
        )


def _expert_components(prefix, global_scale=1.0):
    return [
        (f"{prefix}.weight_packed", torch.full((2, 16), 0x53, dtype=torch.uint8)),
        (f"{prefix}.weight_scale", torch.ones(2, 2).to(torch.float8_e4m3fn)),
        (f"{prefix}.weight_global_scale", torch.tensor(global_scale)),
        (f"{prefix}.input_global_scale", torch.tensor(11.0)),
    ]


@pytest.mark.parametrize("order", list(permutations(range(4))))
@pytest.mark.parametrize("root_prefix", ["", "model.", "language_model.model."])
def test_stream_accepts_component_order_and_replaces_activation_calibration(order, root_prefix):
    prefix = "layers.3.block_sparse_moe.experts.0.w1"
    components = _expert_components(prefix, global_scale=3.0)
    output = dict(
        requantize_minimax_nvfp4_experts(
            ((f"{root_prefix}{components[i][0]}", components[i][1]) for i in order), torch.bfloat16, {prefix}
        )
    )
    expected, scale = requantize_nvfp4_to_mxfp8(
        components[0][1], components[1][1], components[2][1], invert_global_scale=True
    )
    assert set(output) == {f"{root_prefix}{prefix}.weight", f"{root_prefix}{prefix}.weight_scale"}
    assert torch.equal(output[f"{root_prefix}{prefix}.weight"].view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(output[f"{root_prefix}{prefix}.weight_scale"], scale)


def test_stream_keeps_gate_up_global_scales_and_interleaved_projections():
    prefixes = [f"layers.3.block_sparse_moe.experts.0.{projection}" for projection in ("w1", "w2", "w3")]
    components = [_expert_components(prefix, global_scale) for prefix, global_scale in zip(prefixes, (1.0, 2.0, 8.0))]
    interleaved = [part for field in range(4) for part in (component[field] for component in components)]
    output = dict(requantize_minimax_nvfp4_experts(interleaved, torch.bfloat16, set(prefixes)))
    for prefix, component in zip(prefixes, components):
        values, scales = requantize_nvfp4_to_mxfp8(
            component[0][1], component[1][1], component[2][1], invert_global_scale=True
        )
        assert torch.equal(output[f"{prefix}.weight"].view(torch.uint8), values.view(torch.uint8))
        assert torch.equal(output[f"{prefix}.weight_scale"], scales)
    assert not torch.equal(output[f"{prefixes[0]}.weight_scale"], output[f"{prefixes[2]}.weight_scale"])


def test_stream_skips_nonlocal_metadata_after_ep_io_filter_and_preserves_dense_weights():
    local_prefix = "layers.3.block_sparse_moe.experts.0.w1"
    nonlocal_prefix = "layers.3.block_sparse_moe.experts.1.w1"
    dense = torch.randn(2, 32)
    source = _expert_components(local_prefix) + _expert_components(nonlocal_prefix)[1:] + [("model.norm.weight", dense)]
    output = dict(requantize_minimax_nvfp4_experts(source, torch.bfloat16, {local_prefix}))
    assert len(output) == 3
    assert output["model.norm.weight"] is dense
    assert not any(nonlocal_prefix in name for name in output)


@pytest.mark.parametrize("missing_index", [0, 1, 2, None])
def test_stream_rejects_missing_local_components_or_entire_projection(missing_index):
    prefix = "layers.3.block_sparse_moe.experts.0.w1"
    components = _expert_components(prefix)
    source = [] if missing_index is None else [part for i, part in enumerate(components) if i != missing_index]
    with pytest.raises(ValueError, match="local projections missing"):
        list(requantize_minimax_nvfp4_experts(source, torch.bfloat16, {prefix}))


@pytest.mark.parametrize("after_conversion", [False, True])
def test_stream_rejects_duplicate_components(after_conversion):
    prefix = "layers.3.block_sparse_moe.experts.0.w1"
    components = _expert_components(prefix)
    source = components + [components[0]] if after_conversion else [components[0], *components]
    with pytest.raises(ValueError, match="Duplicate"):
        list(requantize_minimax_nvfp4_experts(source, torch.bfloat16, {prefix}))
