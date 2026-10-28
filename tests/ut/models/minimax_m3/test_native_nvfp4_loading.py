# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.utils import WeightsMapper

from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3Model, MiniMaxM3SparseForCausalLM
from vllm_ascend.models.minimax_m3.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from vllm_ascend.quantization.nvfp4_utils import requantize_nvfp4_to_mxfp8


class _CheckpointConsumer(torch.nn.Module):
    prepare_weights_for_loading = MiniMaxM3Model.prepare_weights_for_loading

    def __init__(self):
        super().__init__()
        self._native_nvfp4_moe = True
        self._native_weight_dtype = torch.bfloat16
        self.config = SimpleNamespace(num_local_experts=2)
        self.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        for layer in self.layers:
            layer.is_layer_sparse = True
            layer.block_sparse_moe = SimpleNamespace(
                experts=SimpleNamespace(routed_experts=SimpleNamespace(ascend_expert_map=torch.tensor([0, -1])))
            )
        self.received = {}
        self.load_calls = 0

    def load_weights(self, weights):
        self.load_calls += 1
        loaded = dict(weights)
        self.received.update(loaded)
        return set(loaded)


def test_native_conversion_wraps_complete_stream_before_multimodal_grouping():
    # AutoWeightsLoader revisits language_model.model after vision_tower's
    # namespace interrupts the stream. Both target layers must be validated
    # together, and the original module prefixes must survive conversion.
    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.hf_to_vllm_mapper = WeightsMapper()
    model.language_model.load_weights = MethodType(MiniMaxM3SparseForCausalLM.load_weights, model.language_model)
    consumer = _CheckpointConsumer()
    model.language_model.model = consumer
    model.vision_tower = torch.nn.Linear(2, 2, bias=False)
    model.hf_to_vllm_mapper = WeightsMapper()
    packed = torch.full((2, 16), 0x53, dtype=torch.uint8)
    scales = torch.ones(2, 2).to(torch.float8_e4m3fn)
    global_scale = torch.tensor(17.0)
    source = []
    for layer_idx in range(2):
        for projection in ("w1", "w2", "w3"):
            prefix = f"language_model.model.layers.{layer_idx}.block_sparse_moe.experts.0.{projection}"
            source.extend(
                [
                    (f"{prefix}.weight_packed", packed),
                    (f"{prefix}.weight_global_scale", global_scale),
                    (f"{prefix}.weight_scale", scales),
                ]
            )
        if layer_idx == 0:
            source.append(("vision_tower.weight", torch.eye(2)))

    loaded = MiniMaxM3SparseForConditionalGeneration.load_weights(model, iter(source))
    reference, reference_scale = requantize_nvfp4_to_mxfp8(packed, scales, global_scale, invert_global_scale=True)
    assert consumer.load_calls == 2
    assert len(consumer.received) == 12
    assert len(loaded) == 13
    for name, value in consumer.received.items():
        if name.endswith(".weight"):
            assert torch.equal(value.view(torch.uint8), reference.view(torch.uint8))
        else:
            assert torch.equal(value, reference_scale)
    torch.testing.assert_close(model.vision_tower.weight, torch.eye(2))
    assert not consumer._native_conversion_active


def test_native_stream_failure_resets_parent_child_loading_state():
    consumer = _CheckpointConsumer()
    with pytest.raises(ValueError, match="local projections missing"):
        list(consumer.prepare_weights_for_loading(iter(())))
    assert not consumer._native_conversion_active


def test_mxfp8_checkpoints_retain_original_weight_iterator():
    consumer = _CheckpointConsumer()
    consumer._native_nvfp4_moe = False
    weights = iter([("model.norm.weight", torch.ones(2))])
    assert consumer.prepare_weights_for_loading(weights) is weights


@pytest.mark.parametrize(
    ("quant_format", "group_size", "runtime", "error"),
    [
        ("unsupported", 16, "mxfp8", "nvfp4-pack-quantized format and group_size=16"),
        ("nvfp4-pack-quantized", 32, "mxfp8", "nvfp4-pack-quantized format and group_size=16"),
        ("nvfp4-pack-quantized", 16, None, "require Ascend MXFP8 runtime quantization"),
        ("nvfp4-pack-quantized", 16, "ascend", "require Ascend MXFP8 runtime quantization"),
    ],
)
def test_native_checkpoint_rejects_incompatible_format_and_runtime(quant_format, group_size, runtime, error):
    checkpoint = dict(moe_quant_algo="NVFP4", moe_quant_format=quant_format, group_size=group_size)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(quantization_config=checkpoint),
            hf_text_config=SimpleNamespace(),
            dtype=torch.bfloat16,
        ),
        cache_config=None,
        quant_config=None if runtime is None else SimpleNamespace(get_name=lambda: runtime),
    )
    with pytest.raises(ValueError, match=error):
        MiniMaxM3Model(vllm_config=config)
