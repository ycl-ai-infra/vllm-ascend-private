#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
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
# This file is a part of the vllm-ascend project.
#

import importlib
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile
from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope_vnorm import qkv_rmsnorm_rope_vnorm_fits_ub

# Per-rank (head_dim, num_heads, num_kv_heads) of gemma-4-31B's two attention
# types. At tensor parallel size 4 both fit one vector core's unified buffer; at
# size 2 the full attention layers no longer do.
GEMMA4_TP4_SLIDING = (256, 8, 4)
GEMMA4_TP4_FULL = (512, 8, 1)
GEMMA4_TP2_SLIDING = (256, 16, 8)
GEMMA4_TP2_FULL = (512, 16, 2)
# A head_dim 128 shape that the q/k-only patterns register for, and that also
# fits the unified buffer.
QWEN3_8B_TP1 = (128, 32, 8)

EPSILONS = (1e-6, 1e-5)


@pytest.fixture(scope="module")
def pass_module():
    # base_pattern imports npugraph_ex (or torchair) at module level. Building
    # the pass does not need it, so stand in for it when neither is installed.
    stubbed = importlib.util.find_spec("npugraph_ex") is None and importlib.util.find_spec("torchair") is None
    if stubbed:
        sys.modules["npugraph_ex"] = MagicMock()
    try:
        return importlib.import_module("vllm_ascend.compilation.passes.qknorm_rope_fusion_pass")
    finally:
        if stubbed:
            del sys.modules["npugraph_ex"]


def make_vllm_config():
    vllm_config = MagicMock()
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.compilation_config.splitting_ops = []
    vllm_config.compilation_config.use_inductor_graph_partition = False
    return vllm_config


def register_patterns(pass_module, layer_shapes, device_type=AscendDeviceType.A2):
    """Build the pass over attention layers of the given shapes on the given hardware.

    Return registered shapes and ordering without initializing NPU kernels.
    """
    layers = {
        f"model.layers.{i}.self_attn.attn": SimpleNamespace(head_size=h, num_heads=n, num_kv_heads=kv)
        for i, (h, n, kv) in enumerate(layer_shapes)
    }
    registrations = MagicMock()
    with (
        patch.object(pass_module, "get_layers_from_vllm_config", return_value=layers),
        patch.object(pass_module, "get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        patch.object(pass_module, "HAS_TRITON", True),
        patch.object(pass_module, "qkv_rmsnorm_rope_vnorm_fits_ub", qkv_rmsnorm_rope_vnorm_fits_ub, create=True),
        patch.object(pass_module.QKVNormRopeFusionPattern, "register", autospec=True) as qkv_register,
        patch.object(pass_module.GemmaQKNormRopeFusionPattern, "register", autospec=True) as gemma_register,
        patch.object(pass_module.QKNormRopeFusionPattern, "register", autospec=True) as qk_register,
        patch.object(pass_module.QKNormRopeFusionPatternWithBias, "register", autospec=True) as qk_bias_register,
    ):
        registrations.attach_mock(gemma_register, "gemma")
        registrations.attach_mock(qk_register, "qk")
        registrations.attach_mock(qk_bias_register, "qk_bias")
        pass_module.QKNormRopeFusionPass(make_vllm_config())

    def shapes(register):
        return sorted(
            (p.head_dim, p.num_heads, p.num_kv_heads, p.eps) for p in (c.args[0] for c in register.call_args_list)
        )

    return SimpleNamespace(
        qkv=shapes(qkv_register),
        qk=sorted(shapes(qk_register) + shapes(qk_bias_register)),
        gemma=shapes(gemma_register),
        order=[(call[0], call.args[0].eps) for call in registrations.mock_calls],
    )


def test_pattern_key_is_distinct_per_shape(pass_module):
    """Both Gemma4 shapes must register; a shared key would drop the second."""
    vllm_config = make_vllm_config()
    keys = {
        pass_module.QKVNormRopeFusionPattern(
            vllm_config=vllm_config,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            eps=1e-6,
        ).pattern_key()
        for head_dim, num_heads, num_kv_heads in (GEMMA4_TP4_SLIDING, GEMMA4_TP4_FULL)
    }
    assert len(keys) == 2, f"expected one key per shape, got {keys}"


def test_registers_one_pattern_per_attention_shape(pass_module):
    """Layers sharing a shape register it once, per epsilon."""
    layer_shapes = [GEMMA4_TP4_SLIDING, GEMMA4_TP4_SLIDING, GEMMA4_TP4_FULL, GEMMA4_TP4_SLIDING]

    patterns = register_patterns(pass_module, layer_shapes)

    assert patterns.qkv == sorted((*shape, eps) for shape in (GEMMA4_TP4_SLIDING, GEMMA4_TP4_FULL) for eps in EPSILONS)
    # The q/k-only patterns stay behind their head_dim == 128 guard, so they
    # cannot claim the q/k half of a Gemma4 layer and leave its v norm unfused.
    assert patterns.qk == []
    assert patterns.gemma == []


def test_skips_attention_shape_over_unified_buffer(pass_module):
    """A shape whose single token overflows the unified buffer is not registered."""
    head_dim, num_heads, num_kv_heads = GEMMA4_TP2_FULL
    assert not qkv_rmsnorm_rope_vnorm_fits_ub(
        q_hidden_size=num_heads * head_dim,
        kv_hidden_size=num_kv_heads * head_dim,
        head_dim=head_dim,
        rope_dim=head_dim,
    )

    patterns = register_patterns(pass_module, [GEMMA4_TP2_SLIDING, GEMMA4_TP2_FULL])

    assert patterns.qkv == sorted((*GEMMA4_TP2_SLIDING, eps) for eps in EPSILONS)


def test_hardware_without_the_kernel_keeps_only_qk_patterns(pass_module):
    """Ascend 950 has no split_qkv_rmsnorm_rope_vnorm kernel, so only the q/k/v pattern is skipped.

    The q/k-only patterns still register there, because DeviceOperator routes
    them to the SIMT kernel on that hardware.
    """
    assert not get_hardware_profile(AscendDeviceType.A5).supports(HardwareCapability.GRAPH_QKV_NORM_ROPE_FUSION)

    patterns = register_patterns(pass_module, [QWEN3_8B_TP1], AscendDeviceType.A5)

    assert patterns.qkv == []
    assert patterns.qk == sorted(2 * [(*QWEN3_8B_TP1, eps) for eps in EPSILONS])
    assert patterns.gemma == sorted((*QWEN3_8B_TP1, eps) for eps in EPSILONS)
    for eps in EPSILONS:
        assert patterns.order.index(("gemma", eps)) < patterns.order.index(("qk", eps))


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3])
def test_gemma_weight_offsets_do_not_change_other_hardware_patterns(pass_module, device_type):
    patterns = register_patterns(pass_module, [QWEN3_8B_TP1], device_type)

    assert patterns.gemma == []
    assert patterns.qk == sorted(2 * [(*QWEN3_8B_TP1, eps) for eps in EPSILONS])
    assert patterns.order == [(name, eps) for eps in EPSILONS for name in ("qk", "qk_bias")]


def test_gemma_weight_offsets_keep_head_dimension_guard(pass_module):
    patterns = register_patterns(pass_module, [GEMMA4_TP4_SLIDING], AscendDeviceType.A5)

    assert patterns.qkv == []
    assert patterns.qk == []
    assert patterns.gemma == []


@pytest.fixture(scope="module")
def cpu_fusion_ops():
    """Keep the real FX matcher while replacing only the NPU operator boundary."""
    library = torch.library.Library("vllm_ascend_qknorm_regression", "DEF")
    library.define("rms_norm(Tensor x, Tensor weight, float eps) -> (Tensor, Tensor)")
    library.define(
        "rope(Tensor positions, Tensor q, Tensor k, Tensor cache, int head_dim, int rope_dim, bool neox) "
        "-> (Tensor, Tensor)"
    )
    library.define(
        "fused(Tensor qkv, Tensor q_weight, Tensor k_weight, Tensor cache, Tensor positions, "
        "int q_size, int kv_size, int head_dim, float eps) -> (Tensor, Tensor, Tensor)"
    )
    ops = torch.ops.vllm_ascend_qknorm_regression

    def rms_norm(x, weight, eps):
        inverse_rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
        return (x.float() * inverse_rms * weight.float()).to(x.dtype), inverse_rms

    def rope(positions, q, k, cache, head_dim, rope_dim, neox):
        assert rope_dim == head_dim and neox
        cos, sin = cache[positions].float().chunk(2, dim=-1)

        def rotate(x):
            heads = x.float().view(x.shape[0], -1, head_dim)
            first, second = heads.chunk(2, dim=-1)
            output = torch.cat(
                (first * cos[:, None] - second * sin[:, None], second * cos[:, None] + first * sin[:, None]),
                dim=-1,
            )
            return output.reshape(x.shape).to(x.dtype)

        return rotate(q), rotate(k)

    def fused(qkv, q_weight, k_weight, cache, positions, q_size, kv_size, head_dim, eps):
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = rms_norm(q.view(q.shape[0], -1, head_dim), q_weight, eps)[0].reshape(q.shape)
        k = rms_norm(k.view(k.shape[0], -1, head_dim), k_weight, eps)[0].reshape(k.shape)
        q, k = rope(positions, q, k, cache, head_dim, head_dim, True)
        return q, k, v.clone()

    library.impl("rms_norm", rms_norm, "CPU")
    library.impl("rope", rope, "CPU")
    library.impl("fused", fused, "CPU")
    library.impl(
        "rms_norm",
        lambda x, weight, eps: (torch.empty_like(x), x.new_empty((*x.shape[:-1], 1), dtype=torch.float32)),
        "Meta",
    )
    library.impl(
        "rope",
        lambda positions, q, k, cache, head_dim, rope_dim, neox: (torch.empty_like(q), torch.empty_like(k)),
        "Meta",
    )
    library.impl(
        "fused",
        lambda qkv, q_weight, k_weight, cache, positions, q_size, kv_size, head_dim, eps: (
            qkv.new_empty((qkv.shape[0], q_size)),
            qkv.new_empty((qkv.shape[0], kv_size)),
            qkv.new_empty((qkv.shape[0], kv_size)),
        ),
        "Meta",
    )
    # Keep the library alive until all parameterized cases finish.
    yield ops
    del library


@pytest.mark.parametrize("gemma_weights, shared_offset", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("eps", EPSILONS)
def test_qk_fusion_keeps_weight_producers_before_fused_op(
    pass_module, cpu_fusion_ops, gemma_weights, shared_offset, eps
):
    """A draft-style V split precedes Gemma's weight offsets in the input graph."""
    from torch._inductor import pattern_matcher as pm
    from torch.fx.experimental.proxy_tensor import make_fx

    head_dim, num_heads, num_kv_heads = 128, 2, 1
    q_size, kv_size = head_dim * num_heads, head_dim * num_kv_heads
    generator = torch.Generator().manual_seed(0)
    inputs = [
        torch.randn(5, q_size + 2 * kv_size, generator=generator, dtype=torch.bfloat16),
        torch.randn(head_dim, generator=generator, dtype=torch.bfloat16) * 0.25,
        torch.randn(head_dim, generator=generator, dtype=torch.bfloat16) * 0.25,
        torch.randn(32, head_dim, generator=generator, dtype=torch.bfloat16),
        torch.arange(5, dtype=torch.int64),
    ]

    def draft_head(qkv, q_weight, k_weight, cache, positions):
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        # This early V consumer exposes the invalid insertion point in the
        # original generic replacement when offsets remain external inputs.
        v = v.contiguous()
        q_by_head = q.view(q.shape[0], num_heads, head_dim)
        q_effective_weight = 1.0 + q_weight if gemma_weights else q_weight
        q_norm = cpu_fusion_ops.rms_norm(q_by_head, q_effective_weight, eps)[0]
        k_by_head = k.view(k.shape[0], num_kv_heads, head_dim)
        k_effective_weight = 1.0 + k_weight if gemma_weights else k_weight
        k_norm = cpu_fusion_ops.rms_norm(k_by_head, k_effective_weight, eps)[0]
        q, k = cpu_fusion_ops.rope(
            positions, q_norm.view(q.shape), k_norm.view(k.shape), cache, head_dim, head_dim, True
        )
        if shared_offset:
            return q, k, v, q_effective_weight
        return q, k, v

    graph_module = make_fx(draft_head, tracing_mode="fake")(*inputs)
    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()
    expected = graph_module(*inputs)
    if gemma_weights:
        nodes = list(graph_module.graph.nodes)
        value_copy = next(node for node in nodes if node.target == torch.ops.aten.clone.default)
        weight_offsets = [node for node in nodes if node.target == torch.ops.aten.add.Tensor]
        assert len(weight_offsets) == 2
        assert all(nodes.index(value_copy) < nodes.index(node) for node in weight_offsets)

    def fused_operator(**kwargs):
        assert kwargs["q_bias"] is None and kwargs["k_bias"] is None
        return cpu_fusion_ops.fused(
            kwargs["input"],
            kwargs["q_weight"],
            kwargs["k_weight"],
            kwargs["cos_sin_cache"],
            kwargs["positions"],
            kwargs["q_hidden_size"],
            kwargs["kv_hidden_size"],
            kwargs["head_dim"],
            kwargs["eps"],
        )

    matcher = pm.PatternMatcherPass()
    with (
        patch.object(pass_module, "get_rope_dim", return_value=head_dim),
        patch.object(
            pass_module, "get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A5)
        ),
        patch.object(torch.ops.npu, "npu_rms_norm", cpu_fusion_ops.rms_norm, create=True),
        patch.object(torch.ops.vllm, "npu_rotary_embedding", cpu_fusion_ops.rope, create=True),
        patch.object(pass_module.DeviceOperator, "split_qkv_rmsnorm_rope", side_effect=fused_operator),
    ):
        for pattern_cls in (pass_module.GemmaQKNormRopeFusionPattern, pass_module.QKNormRopeFusionPattern):
            pattern = pattern_cls(make_vllm_config(), head_dim, num_heads, num_kv_heads, eps)
            pm.register_replacement(
                pattern.get_pattern(),
                pattern.get_replacement(),
                inputs,
                pm.fwd_only,
                matcher,
                extra_check=pattern.get_extra_check(),
            )
        assert matcher.apply(graph_module.graph) == (0 if shared_offset else 1)

    graph_module.graph.lint()
    graph_module.recompile()
    assert sum(node.target == cpu_fusion_ops.fused.default for node in graph_module.graph.nodes) == (
        0 if shared_offset else 1
    )
    assert sum(node.target == cpu_fusion_ops.rms_norm.default for node in graph_module.graph.nodes) == (
        2 if shared_offset else 0
    )
    for actual, reference in zip(graph_module(*inputs), expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("pattern_name", ["QKNormRopeFusionPattern", "QKNormRopeFusionPatternWithBias"])
def test_dependency_check_preserves_backend_registration_on_other_hardware(pass_module, device_type, pattern_name):
    base_module = importlib.import_module("vllm_ascend.compilation.passes.base_pattern")
    stream_check = MagicMock(return_value=True)
    matcher = MagicMock()
    with (
        patch.object(pass_module, "get_rope_dim", return_value=128),
        patch.object(pass_module, "get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        patch.object(pass_module, "_inputs_precede_outputs", return_value=True) as dependency_check,
        patch.object(base_module, "_registered_patterns", set()),
        patch.object(base_module.pm, "register_replacement") as pm_register,
        patch.object(base_module.nge, "register_replacement") as nge_register,
    ):
        pattern = getattr(pass_module, pattern_name)(make_vllm_config(), 128, 2, 1)
        with (
            patch.object(pattern, "get_inputs", return_value=[]),
            patch.object(pattern, "get_extra_stream_scope_check", return_value=stream_check),
        ):
            pattern.register(matcher)

        pm_register.assert_called_once()
        nge_register.assert_called_once()
        backend_check = nge_register.call_args.kwargs["extra_check"]
        if device_type != AscendDeviceType.A5:
            assert "extra_check" not in pm_register.call_args.kwargs
            assert backend_check is stream_check
            dependency_check.assert_not_called()
        else:
            assert pm_register.call_args.kwargs["extra_check"] is dependency_check
            match = MagicMock()
            assert backend_check(match)
            stream_check.assert_called_once_with(match)
            dependency_check.assert_called_once_with(match)

            dependency_check.return_value = False
            assert not backend_check(match)
            dependency_check.reset_mock()
            stream_check.return_value = False
            assert not backend_check(match)
            dependency_check.assert_not_called()
