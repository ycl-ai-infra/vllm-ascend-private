# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU checkpoint diagnostic for the specific A3 BF16/W8A8 QuaRot pair.

For row-vector activations x_quarot=x_bf@Q, compare the quantized-side
effective router to (W_bf * (1+n_bf))@Q. Directly applying the non-rotated
comparison would mistake an ordinary basis change for a mismatch.

The artifact regression deliberately reproduces the measured 26 differing
layers; passing does not mean these checkpoints are equivalent or accurate.
It skips without the named checkpoints. Synthetic tests need no NPU or weights.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from test_checkpoint_router_norm_equivalence import (
    COLUMN_FIT_TOLERANCE,
    DIFFERENCE_THRESHOLD,
    EXPECTED_MOE_LAYER_COUNT,
    LAYER_PREFIX,
    NORM_SUFFIX,
    ROUTER_SUFFIX,
    _index_norm_router_tensors,
    _read_tensor,
    _relative_l2,
)

A3_BF16_CHECKPOINT = Path("/mnt/share_space/MiniMax-M3.1-Flash-bf16")
A3_QUAROT_CHECKPOINT = Path("/mnt/share_space/MiniMax-M3.1-Flash-w8a8-quarot")
ROTATION_RELATIVE_PATH = Path("optional/quarot.safetensors")
ROTATION_KEY = "global_rotation"
ROTATION_BLOCK_SIZE = 32
EMBEDDING_KEY = "language_model.model.embed_tokens.weight"
FORWARD_SAMPLES = 64
FORWARD_SEED = 20261024
RMS_EPS = 1e-6
EXPECTED_A3_DIFFERING_LAYERS = (
    3,
    4,
    5,
    6,
    8,
    11,
    12,
    13,
    14,
    22,
    25,
    27,
    31,
    32,
    33,
    36,
    37,
    40,
    41,
    45,
    48,
    50,
    55,
    56,
    57,
    59,
)


def _rms_normalize(inputs: torch.Tensor) -> torch.Tensor:
    return inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + RMS_EPS)


def _compare_rotated_norm_router(bf_router, quant_router, bf_norm, quant_norm, rotation, inputs):
    bf_effective = bf_router.float() * (1.0 + bf_norm.float())
    rotated_reference = bf_effective @ rotation
    quant_router = quant_router.float()
    quant_gamma = 1.0 + quant_norm.float()
    scale = (rotated_reference * quant_router).sum(dim=0) / rotated_reference.square().sum(dim=0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    # Independent norm + linear forwards, including RMS in each input basis.
    bf_logits = (_rms_normalize(inputs) * (1.0 + bf_norm.float())) @ bf_router.float().T
    quant_logits = (_rms_normalize(inputs @ rotation) * quant_gamma) @ quant_router.T
    return {
        "norm_folded_rotated_basis_relative_l2": _relative_l2(quant_router * quant_gamma, rotated_reference),
        "rotated_basis_router_column_fit_relative_l2": _relative_l2(quant_router, rotated_reference * scale),
        "random_input_fp32_forward_relative_l2": _relative_l2(quant_logits, bf_logits),
        "scale_times_gamma_minus_one_max_abs": (scale * quant_gamma - 1.0).abs().max().item(),
    }


@pytest.mark.parametrize("relative_change", [0.0, 0.20])
def test_rotation_and_inverse_scaling_require_matching_norm(relative_change):
    # A non-symmetric orthogonal matrix catches a transposed-basis mistake.
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], device="cpu")
    router = torch.tensor([[1.0, 2.0], [3.0, 5.0]], device="cpu")
    norm = torch.tensor([-0.5, 0.5], device="cpu")
    scale = torch.tensor([2.0, 0.5], device="cpu")
    reference = (router * (1.0 + norm)) @ rotation
    quant_router = reference * scale
    quant_norm = (1.0 + relative_change) / scale - 1.0
    result = _compare_rotated_norm_router(router, quant_router, norm, quant_norm, rotation, torch.eye(2, device="cpu"))
    assert result["rotated_basis_router_column_fit_relative_l2"] < COLUMN_FIT_TOLERANCE
    assert result["norm_folded_rotated_basis_relative_l2"] == pytest.approx(relative_change, abs=1e-6)
    assert result["random_input_fp32_forward_relative_l2"] == pytest.approx(relative_change, abs=1e-6)
    assert _relative_l2(quant_router * (1.0 + quant_norm), (router * (1.0 + norm)) @ rotation.T) > 1.0


def _read_embedding_sample(checkpoint: Path) -> torch.Tensor:
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if EMBEDDING_KEY in handle.keys():  # noqa: SIM118 - safe_open is not iterable.
                return handle.get_slice(EMBEDDING_KEY)[:FORWARD_SAMPLES].float()
    raise KeyError(f"Missing {EMBEDDING_KEY} in {checkpoint}")


def _check_rotation(rotation: torch.Tensor) -> dict:
    assert rotation.dtype == torch.float32 and rotation.shape == (6144, 6144)
    blocks = torch.stack(
        [
            rotation[start : start + ROTATION_BLOCK_SIZE, start : start + ROTATION_BLOCK_SIZE]
            for start in range(0, len(rotation), ROTATION_BLOCK_SIZE)
        ]
    )
    outside_nonzeros = torch.count_nonzero(rotation).item() - torch.count_nonzero(blocks).item()
    orthogonality_error = (blocks @ blocks.transpose(-1, -2) - torch.eye(ROTATION_BLOCK_SIZE)).abs().max().item()
    assert outside_nonzeros == 0
    assert orthogonality_error < COLUMN_FIT_TOLERANCE
    return {
        "block_count": len(blocks),
        "outside_nonzeros": outside_nonzeros,
        "orthogonality_max_abs": orthogonality_error,
    }


@pytest.mark.skipif(
    not A3_BF16_CHECKPOINT.is_dir() or not A3_QUAROT_CHECKPOINT.is_dir(),
    reason="Requires the specific A3 BF16 and W8A8 QuaRot checkpoint artifacts",
)
def test_a3_checkpoints_reproduce_26_differing_layers(record_property):
    # Checkpoint-specific artifact regression, CPU only; no full model loading.
    with safe_open(str(A3_QUAROT_CHECKPOINT / ROTATION_RELATIVE_PATH), framework="pt", device="cpu") as handle:
        rotation = handle.get_tensor(ROTATION_KEY)
    rotation_check = _check_rotation(rotation)
    bf_embedding = _read_embedding_sample(A3_BF16_CHECKPOINT)
    quant_embedding = _read_embedding_sample(A3_QUAROT_CHECKPOINT)
    embedding_error = _relative_l2(quant_embedding, bf_embedding @ rotation)
    # BF16 embedding storage rounding; the incorrect Q.T direction is ~1.39.
    assert embedding_error < 0.002
    assert _relative_l2(quant_embedding, bf_embedding @ rotation.T) > 1.0
    bf_index = _index_norm_router_tensors(A3_BF16_CHECKPOINT)
    quant_index = _index_norm_router_tensors(A3_QUAROT_CHECKPOINT)
    names = sorted(name for name in bf_index if name.endswith(ROUTER_SUFFIX))
    assert len(names) == EXPECTED_MOE_LAYER_COUNT
    assert set(names) == {name for name in quant_index if name.endswith(ROUTER_SUFFIX)}
    generator = torch.Generator(device="cpu").manual_seed(FORWARD_SEED)
    inputs = torch.randn(FORWARD_SAMPLES, len(rotation), generator=generator, device="cpu")
    rows = []
    for name in names:
        norm_name = name.removesuffix(ROUTER_SUFFIX) + NORM_SUFFIX
        bf_router, quant_router = _read_tensor(bf_index, name), _read_tensor(quant_index, name)
        assert bf_router.dtype == quant_router.dtype == torch.float32
        assert bf_router.shape == quant_router.shape == (128, 6144)
        bf_norm, quant_norm = _read_tensor(bf_index, norm_name), _read_tensor(quant_index, norm_name)
        assert bf_norm.dtype == torch.bfloat16 and quant_norm.dtype == torch.float32
        assert bf_norm.shape == quant_norm.shape == (6144,)
        row = _compare_rotated_norm_router(bf_router, quant_router, bf_norm, quant_norm, rotation, inputs)
        row["layer"] = int(name.removeprefix(LAYER_PREFIX).split(".")[0])
        rows.append(row)
    rows.sort(key=lambda row: row["layer"])
    differing = [r["layer"] for r in rows if r["norm_folded_rotated_basis_relative_l2"] > DIFFERENCE_THRESHOLD]
    stable = [r for r in rows if r["norm_folded_rotated_basis_relative_l2"] < COLUMN_FIT_TOLERANCE]
    result = {"rotation": rotation_check, "embedding_alignment_relative_l2": embedding_error, "layers": rows}
    print(json.dumps(result, indent=2))
    record_property("a3_quarot_norm_router_comparison", json.dumps(result))
    assert differing == list(EXPECTED_A3_DIFFERING_LAYERS)
    assert len(stable) + len(differing) == EXPECTED_MOE_LAYER_COUNT
    assert all(r["rotated_basis_router_column_fit_relative_l2"] < COLUMN_FIT_TOLERANCE for r in rows)
    assert all(
        r["random_input_fp32_forward_relative_l2"] > DIFFERENCE_THRESHOLD for r in rows if r["layer"] in differing
    )
    assert all(r["random_input_fp32_forward_relative_l2"] < 5e-6 for r in stable)
    by_layer = {r["layer"]: r for r in rows}
    assert by_layer[3]["norm_folded_rotated_basis_relative_l2"] == pytest.approx(0.19099922, abs=1e-6)
    assert by_layer[25]["norm_folded_rotated_basis_relative_l2"] == pytest.approx(0.41889411, abs=1e-6)
    assert by_layer[56]["norm_folded_rotated_basis_relative_l2"] == pytest.approx(0.24536476, abs=1e-6)
