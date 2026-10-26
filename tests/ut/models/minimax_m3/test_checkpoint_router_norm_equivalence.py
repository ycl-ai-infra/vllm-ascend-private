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
"""Reproduce the M3.1 checkpoint norm/router comparison on CPU.

Gemma norm stores ``n`` but multiplies normalized activations by ``1 + n``.
For a router matrix W shaped [experts, hidden], its effective matrix is
``W_eff = W * (1 + n)[None, :]``. The reported difference is
``||W_quant_eff - W_bf16_eff||_F / ||W_bf16_eff||_F``.

Separately, fit each router column with least squares:
``s[j] = dot(W_bf16[:, j], W_quant[:, j]) / ||W_bf16[:, j]||_2**2``.
A small column-fit residual does NOT imply a small effective-matrix error:
the norm must also satisfy ``s * (1 + n_quant) == 1 + n_bf16``.

The small synthetic tests require no checkpoints or NPU. The reference test
reads only FP32 routers and norm vectors from the two supplied checkpoints;
it skips when they are absent. Its expected layer list describes these
specific checkpoint artifacts, not a general quantization accuracy rule.

Run on 27.1 with ``pytest -sv <this-file>`` to print every layer's metrics.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

BF16_CHECKPOINT = Path("/mnt/m31-bf16-gpqa-20261024")
QUANT_CHECKPOINT = Path("/mnt/weight/minimax_m3dot1_w8a8_2")
LAYER_PREFIX = "language_model.model.layers."
ROUTER_SUFFIX = ".block_sparse_moe.gate.weight"
NORM_SUFFIX = ".post_attention_layernorm.weight"
DIFFERENCE_THRESHOLD = 0.10
COLUMN_FIT_TOLERANCE = 1e-6
EXPECTED_MOE_LAYER_COUNT = 57
EXPECTED_DIFFERING_LAYERS = (
    3,
    4,
    5,
    6,
    15,
    16,
    17,
    19,
    20,
    21,
    22,
    24,
    25,
    29,
    30,
    31,
    32,
    37,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    48,
    49,
    50,
    51,
    52,
    54,
    55,
    56,
    57,
    58,
)


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = reference.square().sum()
    if denominator == 0:
        raise ValueError("Relative L2 requires a nonzero reference")
    return ((actual - reference).square().sum() / denominator).sqrt().item()


def _compare_norm_router(
    bf16_router: torch.Tensor,
    quant_router: torch.Tensor,
    bf16_norm: torch.Tensor,
    quant_norm: torch.Tensor,
) -> dict:
    # FP32 checkpoint algebra on CPU, without activation quantization or NPU ops.
    bf16_router = bf16_router.float()
    quant_router = quant_router.float()
    bf16_gamma = 1.0 + bf16_norm.float()
    quant_gamma = 1.0 + quant_norm.float()
    column_energy = bf16_router.square().sum(dim=0)
    scale = (bf16_router * quant_router).sum(dim=0) / column_energy.clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "column_scale": scale,
        "router_weight_exact": torch.equal(bf16_router, quant_router),
        "norm_weight_exact": torch.equal(bf16_norm.float(), quant_norm.float()),
        "router_weight_relative_l2": _relative_l2(quant_router, bf16_router),
        "norm_weight_relative_l2": _relative_l2(quant_norm.float(), bf16_norm.float()),
        "router_weight_max_abs_difference": (quant_router - bf16_router).abs().max().item(),
        "norm_weight_max_abs_difference": (quant_norm.float() - bf16_norm.float()).abs().max().item(),
        "column_fit_relative_l2": _relative_l2(quant_router, bf16_router * scale),
        "norm_folded_router_relative_l2": _relative_l2(quant_router * quant_gamma, bf16_router * bf16_gamma),
    }


def test_inverse_norm_scaling_preserves_router():
    router = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, device="cpu")
    norm = torch.tensor([-0.5, 0.5, 1.0], dtype=torch.float32, device="cpu")
    scale = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32, device="cpu")
    quant_norm = (1.0 + norm) / scale - 1.0

    result = _compare_norm_router(router, router * scale, norm, quant_norm)

    torch.testing.assert_close(result["column_scale"], scale)
    assert result["column_fit_relative_l2"] == 0
    assert result["norm_folded_router_relative_l2"] == 0


@pytest.mark.parametrize("relative_change", [0.16, 0.20])
def test_exact_column_fit_can_hide_norm_mismatch(relative_change):
    router = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, device="cpu")
    norm = torch.tensor([-0.5, 0.5, 1.0], dtype=torch.float32, device="cpu")
    scale = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32, device="cpu")
    quant_router = router * scale
    quant_norm = (1.0 + relative_change) * (1.0 + norm) / scale - 1.0

    result = _compare_norm_router(router, quant_router, norm, quant_norm)

    assert result["column_fit_relative_l2"] == 0
    assert result["norm_folded_router_relative_l2"] == pytest.approx(relative_change, abs=1e-6)

    # Independently verify through norm + linear on a complete input basis.
    inputs = torch.eye(router.shape[1], dtype=torch.float32, device="cpu")
    normalized = inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + 1e-6)
    bf16_logits = (normalized * (1.0 + norm)) @ router.T
    quant_logits = (normalized * (1.0 + quant_norm)) @ quant_router.T
    assert _relative_l2(quant_logits, bf16_logits) == pytest.approx(result["norm_folded_router_relative_l2"], abs=1e-6)


def _index_norm_router_tensors(checkpoint: Path) -> dict[str, Path]:
    # safe_open reads tensor headers here; get_tensor below reads only selected
    # routers/norms, not the hundreds of GiB of expert weights.
    index = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 - safe_open is not iterable.
                if name.startswith(LAYER_PREFIX) and name.endswith((ROUTER_SUFFIX, NORM_SUFFIX)):
                    if name in index:
                        raise ValueError(f"Duplicate checkpoint tensor: {name}")
                    index[name] = shard
    return index


def _read_tensor(index: dict[str, Path], name: str) -> torch.Tensor:
    with safe_open(str(index[name]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


@pytest.mark.skipif(
    not BF16_CHECKPOINT.is_dir() or not QUANT_CHECKPOINT.is_dir(),
    reason="Requires the two M3.1 reference checkpoints on 27.1",
)
def test_reference_checkpoints_reproduce_35_differing_layers(record_property):
    bf16_index = _index_norm_router_tensors(BF16_CHECKPOINT)
    quant_index = _index_norm_router_tensors(QUANT_CHECKPOINT)
    router_names = sorted(name for name in bf16_index if name.endswith(ROUTER_SUFFIX))
    assert len(router_names) == EXPECTED_MOE_LAYER_COUNT
    assert {name for name in quant_index if name.endswith(ROUTER_SUFFIX)} == set(router_names)

    rows = []
    for name in router_names:
        norm_name = name.removesuffix(ROUTER_SUFFIX) + NORM_SUFFIX
        bf16_router = _read_tensor(bf16_index, name)
        quant_router = _read_tensor(quant_index, name)
        # These routers themselves are FP32 in BOTH checkpoints. This audit
        # does not dequantize FP8 expert tensors or approximate their scales.
        assert bf16_router.dtype == quant_router.dtype == torch.float32
        result = _compare_norm_router(
            bf16_router,
            quant_router,
            _read_tensor(bf16_index, norm_name),
            _read_tensor(quant_index, norm_name),
        )
        row = {key: value for key, value in result.items() if key != "column_scale"}
        row["layer"] = int(name.removeprefix(LAYER_PREFIX).split(".")[0])
        rows.append(row)

    rows.sort(key=lambda row: row["layer"])
    differing_layers = [row["layer"] for row in rows if row["norm_folded_router_relative_l2"] > DIFFERENCE_THRESHOLD]
    print(
        json.dumps({"threshold": DIFFERENCE_THRESHOLD, "layers": rows, "differing_layers": differing_layers}, indent=2)
    )
    record_property("router_norm_comparison", json.dumps(rows))
    assert all(row["column_fit_relative_l2"] < COLUMN_FIT_TOLERANCE for row in rows)
    assert differing_layers == list(EXPECTED_DIFFERING_LAYERS)
    by_layer = {row["layer"]: row for row in rows}
    assert by_layer[3]["norm_folded_router_relative_l2"] == pytest.approx(0.15876633, abs=1e-6)
    assert by_layer[30]["norm_folded_router_relative_l2"] == pytest.approx(0.20512274, abs=1e-6)
    assert by_layer[59]["norm_folded_router_relative_l2"] < COLUMN_FIT_TOLERANCE
