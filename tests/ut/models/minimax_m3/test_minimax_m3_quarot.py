# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numerical tests of the real loader without importing the NPU model."""

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn import functional as F


@pytest.fixture
def loader():
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/models/minimax_m3/minimax_m3_vl.py"
    model = next(
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) == "MiniMaxM3SparseForConditionalGeneration"
    )
    method = next(node for node in model.body if getattr(node, "name", None) == "load_weights")
    received = {}
    loaded_names = {"loaded_parameter"}

    def load_weights(weights, *, mapper):
        received.update(weights)
        assert mapper is sentinel_mapper
        return loaded_names

    sentinel_mapper = object()
    rotation_loader = Mock()
    namespace = {
        "AutoWeightsLoader": lambda _: SimpleNamespace(load_weights=load_weights),
        "get_rotation_matrix": rotation_loader,
        "logger": Mock(),
    }
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    target = SimpleNamespace(_vision_rotation_path="rotation.safetensors", hf_to_vllm_mapper=sentinel_mapper)
    return SimpleNamespace(
        load=lambda weights: namespace["load_weights"](target, iter(weights)),
        target=target,
        rotation_loader=rotation_loader,
        received=received,
        loaded_names=loaded_names,
    )


@pytest.mark.parametrize("prefix", ["", "vision_tower.", "model.vision_tower."])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_visual_projection_matches_rotated_text_basis(loader, dtype, prefix):
    generator = torch.Generator().manual_seed(17)
    # A non-symmetric orthogonal matrix detects an accidental transpose.
    rotation = torch.linalg.qr(torch.randn(8, 8, generator=generator)).Q
    loader.rotation_loader.return_value = rotation
    weight = torch.randn(8, 5, generator=generator).to(dtype)
    bias = torch.randn(8, generator=generator).to(dtype)
    original_weight, original_bias = weight.clone(), bias.clone()
    inputs = torch.randn(3, 5, generator=generator)
    unrelated = torch.randn(8, 5, generator=generator)
    weight_name = f"{prefix}patch_merge_mlp.linear_2.weight"
    bias_name = f"{prefix}patch_merge_mlp.linear_2.bias"

    result = loader.load([(weight_name, weight), (bias_name, bias), ("language_model.other.weight", unrelated)])

    transformed_weight, transformed_bias = loader.received[weight_name], loader.received[bias_name]
    expected = F.linear(inputs, weight.float(), bias.float()) @ rotation
    actual = F.linear(inputs, transformed_weight.float(), transformed_bias.float())
    tolerance = {torch.float32: 2e-6, torch.float16: 0.005, torch.bfloat16: 0.04}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert transformed_weight.dtype == transformed_bias.dtype == dtype
    assert loader.received["language_model.other.weight"] is unrelated
    torch.testing.assert_close(weight, original_weight, atol=0, rtol=0)
    torch.testing.assert_close(bias, original_bias, atol=0, rtol=0)
    assert result is loader.loaded_names
    loader.rotation_loader.assert_called_once_with("rotation.safetensors")


def test_unrotated_model_keeps_checkpoint_tensors(loader):
    loader.target._vision_rotation_path = None
    weights = [
        ("patch_merge_mlp.linear_2.weight", torch.randn(8, 5)),
        ("patch_merge_mlp.linear_2.bias", torch.randn(8)),
    ]

    loader.load(weights)

    loader.rotation_loader.assert_not_called()
    for name, weight in weights:
        assert loader.received[name] is weight
