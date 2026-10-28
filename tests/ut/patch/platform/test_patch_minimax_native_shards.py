# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm_ascend.patch.platform.patch_minimax_native_shards import (
    _ORIGINAL_FILTER,
    filter_minimax_native_safetensors_files,
)


def _checkpoint(tmp_path, *, native=True, nested_exists=True):
    config = {
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "quantization_config": {"moe_quant_algo": "NVFP4"} if native else {},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"target.weight": "model.safetensors", "draft.weight": "dspark/draft.safetensors"}})
    )
    target = tmp_path / "model.safetensors"
    target.touch()
    if nested_exists:
        (tmp_path / "dspark").mkdir()
        (tmp_path / "dspark/draft.safetensors").touch()
    return [str(target)]


def test_discovers_existing_nested_native_shard(tmp_path):
    candidates = _checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError):
        _ORIGINAL_FILTER(candidates, str(tmp_path), "model.safetensors.index.json")
    result = filter_minimax_native_safetensors_files(candidates, str(tmp_path), "model.safetensors.index.json")
    assert set(result) == {*candidates, str(tmp_path / "dspark/draft.safetensors")}


def test_missing_nested_shard_still_fails(tmp_path):
    candidates = _checkpoint(tmp_path, nested_exists=False)
    with pytest.raises(FileNotFoundError):
        filter_minimax_native_safetensors_files(candidates, str(tmp_path), "model.safetensors.index.json")


def test_other_checkpoint_formats_preserve_upstream_behavior(tmp_path):
    candidates = _checkpoint(tmp_path, native=False)
    with pytest.raises(FileNotFoundError):
        filter_minimax_native_safetensors_files(candidates, str(tmp_path), "model.safetensors.index.json")


def test_existing_candidates_are_not_duplicated(tmp_path):
    candidates = _checkpoint(tmp_path)
    candidates.append(str(tmp_path / "dspark/draft.safetensors"))
    assert (
        filter_minimax_native_safetensors_files(candidates, str(tmp_path), "model.safetensors.index.json") == candidates
    )


def test_no_index_preserves_candidates(tmp_path):
    candidates = [str(tmp_path / "model.safetensors")]
    assert (
        filter_minimax_native_safetensors_files(candidates, str(tmp_path), "model.safetensors.index.json") == candidates
    )
