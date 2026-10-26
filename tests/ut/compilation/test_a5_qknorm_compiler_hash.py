# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from hashlib import sha256
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.compilation import compiler_interface
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


@pytest.fixture
def hash_environment(monkeypatch):
    config = SimpleNamespace(
        enable_npugraph_ex=True,
        enable_static_kernel=False,
        enable_super_kernel=False,
        fuse_qknorm_rope=True,
    )
    monkeypatch.setattr(
        compiler_interface, "get_ascend_config", lambda: SimpleNamespace(ascend_compilation_config=config)
    )
    torch_npu = ModuleType("torch_npu")
    torch_npu.__version__ = "2.10.0.post7"
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    source_hash = MagicMock(return_value="qk-source-v1")

    class FusionPass:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Computing the cache key must not initialize NPU fusion patterns")

    FusionPass.source_hash = source_hash
    pass_module = ModuleType("vllm_ascend.compilation.passes.qknorm_rope_fusion_pass")
    pass_module.QKNormRopeFusionPass = FusionPass
    monkeypatch.setitem(sys.modules, pass_module.__name__, pass_module)
    return config, source_hash


@pytest.mark.parametrize(
    ("device_type", "fusion_enabled"),
    [
        (AscendDeviceType.A2, False),
        (AscendDeviceType.A2, True),
        (AscendDeviceType.A3, False),
        (AscendDeviceType.A3, True),
        (AscendDeviceType.A5, False),
    ],
)
def test_unaffected_configurations_keep_original_compiler_hash(
    monkeypatch, hash_environment, device_type, fusion_enabled
):
    config, source_hash = hash_environment
    config.fuse_qknorm_rope = fusion_enabled
    monkeypatch.setattr(compiler_interface, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    # Preserve the original ordered factors, including their exact hash.
    original_factors = {
        "torch_npu_version": "2.10.0.post7",
        "enable_npugraph_ex": True,
        "enable_static_kernel": False,
        "enable_super_kernel": False,
    }
    expected = sha256(str(original_factors).encode(), usedforsecurity=False).hexdigest()[:10]

    assert compiler_interface.AscendCompiler().compute_hash(SimpleNamespace()) == expected
    source_hash.assert_not_called()


def test_a5_qk_fusion_cache_changes_with_rewrite_source(monkeypatch, hash_environment):
    _, source_hash = hash_environment
    monkeypatch.setattr(
        compiler_interface, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A5)
    )
    compiler = compiler_interface.AscendCompiler()
    vllm_config = SimpleNamespace()
    first_hash = compiler.compute_hash(vllm_config)
    assert compiler.compute_hash(vllm_config) == first_hash
    assert compiler.vllm_config is vllm_config

    source_hash.return_value = "qk-source-v2"
    assert compiler.compute_hash(vllm_config) != first_hash
    assert source_hash.call_count == 3
