# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Discover indexed nested shards in native MiniMax checkpoints.

The upstream default loader only globs the checkpoint's top directory, then
rejects an existing indexed dspark/*.safetensors file as missing. Extend its
candidate list for native MiniMax NVFP4 checkpoints; retain upstream duplicate
filtering and missing-file validation. Remove this patch when upstream loads
all indexed shards directly, including subdirectories.
"""

import json
from pathlib import Path

from vllm.model_executor.model_loader import default_loader, weight_utils

_ORIGINAL_FILTER = weight_utils.filter_duplicate_safetensors_files


def filter_minimax_native_safetensors_files(hf_weights_files: list[str], hf_folder: str, index_file: str) -> list[str]:
    root = Path(hf_folder)
    config_path = root / "config.json"
    index_path = root / index_file
    if config_path.is_file() and index_path.is_file():
        config = json.loads(config_path.read_text())
        quantization = config.get("quantization_config") or {}
        architectures = config.get("architectures") or []
        if quantization.get("moe_quant_algo") == "NVFP4" and any(
            architecture.startswith("MiniMaxM3") for architecture in architectures
        ):
            index = json.loads(index_path.read_text())
            nested_shards = []
            for relative_name in sorted(set(index["weight_map"].values())):
                relative_path = Path(relative_name)
                if relative_path.parent == Path("."):
                    continue
                shard_path = root / relative_path
                if (
                    shard_path.resolve().is_relative_to(root.resolve())
                    and shard_path.suffix == ".safetensors"
                    and shard_path.is_file()
                ):
                    nested_shards.append(str(shard_path))
            hf_weights_files = list(dict.fromkeys([*hf_weights_files, *nested_shards]))
    return _ORIGINAL_FILTER(hf_weights_files, hf_folder, index_file)


weight_utils.filter_duplicate_safetensors_files = filter_minimax_native_safetensors_files
default_loader.filter_duplicate_safetensors_files = filter_minimax_native_safetensors_files
