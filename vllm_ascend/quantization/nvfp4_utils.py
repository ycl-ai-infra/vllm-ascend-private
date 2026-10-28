# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-only conversion of serialized NVFP4 weights to MXFP8.

These functions operate on unswizzled checkpoint tensors, before the Ascend
runtime's weight transposition and scale layout conversion. They intentionally
do not call upstream helpers that can select Triton or CUDA kernels.
"""

from collections.abc import Iterable, Iterator

import regex as re
import torch

NVFP4_BLOCK_SIZE = 16
MXFP8_BLOCK_SIZE = 32
E8M0_EXPONENT_BIAS = 127
E8M0_MAX_FINITE_EXPONENT = 254
FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MAX_MISSING_PROJECTION_EXAMPLES = 5


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    *,
    invert_global_scale: bool = False,
) -> torch.Tensor:
    """Decode an unswizzled 2D weight, or a batch of expert weights.

    NVFP4 stores the even K element in the low nibble and the odd element
    in the high nibble. Each group of 16 elements has an E4M3 block scale.
    ModelOpt stores a global decode multiplier; compressed-tensors stores
    its reciprocal and requires ``invert_global_scale=True``. For a 3D
    input, pass one global scale per expert. Gate and up projections with
    different global scales must be decoded separately before concatenation.
    """
    if packed.dtype != torch.uint8 or packed.ndim not in (2, 3):
        raise ValueError("NVFP4 weights must be 2D or 3D packed uint8 tensors")
    width = packed.shape[-1] * 2
    if width % NVFP4_BLOCK_SIZE:
        raise ValueError("NVFP4 weight width must be divisible by 16")
    expected_scale_shape = (*packed.shape[:-1], width // NVFP4_BLOCK_SIZE)
    if tuple(block_scale.shape) != expected_scale_shape:
        raise ValueError(f"Expected NVFP4 block scales with shape {expected_scale_shape}")
    if block_scale.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("NVFP4 block scales must contain E4M3 values or their uint8 bytes")
    expected_global_scales = packed.shape[0] if packed.ndim == 3 else 1
    if global_scale.numel() != expected_global_scales:
        raise ValueError("Expected one NVFP4 global scale per matrix")
    if packed.device != block_scale.device or packed.device != global_scale.device:
        raise ValueError("NVFP4 weights and scales must be on the same device")

    nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1)
    magnitudes = torch.tensor(FP4_MAGNITUDES, dtype=torch.float32, device=packed.device)
    decoded = magnitudes[(nibbles & 0x07).long()]
    decoded = decoded * torch.where((nibbles & 0x08).bool(), -1.0, 1.0)
    decoded = decoded.reshape(*packed.shape[:-1], width // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)

    global_scale = global_scale.to(torch.float32)
    if invert_global_scale:
        global_scale = global_scale.reciprocal()
    global_scale = global_scale.reshape((-1, 1, 1) if packed.ndim == 3 else ())
    scale = block_scale.view(torch.float8_e4m3fn).to(torch.float32) * global_scale
    # Match vLLM's FP32 decode followed by the model dtype rounding. The
    # subsequent MXFP8 quantizer must observe BF16 weights for BF16 models.
    return (decoded * scale.unsqueeze(-1)).reshape(*packed.shape[:-1], width).to(dtype)


def quantize_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match vLLM's Torch MXFP8 reference, including zero-block scale bytes.

    Returns E4M3 values and row-major uint8 E8M0 scale bytes. The scale
    algorithm is ceil(log2(amax / 448)), corresponding to scale_alg=1 in
    Ascend DynamicMxQuantV3. This is a load-time conversion helper.
    """
    if weight.ndim not in (2, 3) or weight.shape[-1] % MXFP8_BLOCK_SIZE:
        raise ValueError("MXFP8 requires a 2D or 3D tensor with width divisible by 32")
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("MXFP8 source weights must be BF16, FP16, or FP32")
    blocked = weight.to(torch.float32).reshape(*weight.shape[:-1], -1, MXFP8_BLOCK_SIZE)
    amax = blocked.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    biased_exponent = torch.ceil(torch.log2(amax / torch.finfo(torch.float8_e4m3fn).max)) + E8M0_EXPONENT_BIAS
    biased_exponent = biased_exponent.clamp(0, E8M0_MAX_FINITE_EXPONENT)
    descale = torch.exp2(biased_exponent - E8M0_EXPONENT_BIAS)
    values = (blocked / descale.unsqueeze(-1)).reshape(weight.shape).to(torch.float8_e4m3fn)
    return values, biased_exponent.to(torch.uint8)


def requantize_nvfp4_to_mxfp8(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    *,
    invert_global_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a checkpoint matrix; call per expert to bound temporary memory."""
    if packed.ndim not in (2, 3) or packed.shape[-1] * 2 % MXFP8_BLOCK_SIZE:
        raise ValueError("NVFP4 to MXFP8 conversion requires weight width divisible by 32")
    decoded = dequantize_nvfp4(packed, block_scale, global_scale, dtype, invert_global_scale=invert_global_scale)
    return quantize_mxfp8(decoded)


def requantize_minimax_nvfp4_experts(
    weights: Iterable[tuple[str, torch.Tensor]],
    dtype: torch.dtype,
    expected_prefixes: set[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Convert native MiniMax NVFP4 experts before the existing MXFP8 loader.

    Checkpoint iteration may interleave projection components or cross shard
    boundaries. Retain each packed matrix until both scales arrive, then
    release its source tensors. Nonlocal expert metadata must be skipped too:
    vLLM's EP I/O filter skips heavy weights but retains all scale tensors.
    ``expected_prefixes`` describes the actual local layer/expert placement,
    so missing local matrices or scales fail instead of leaving empty weights.
    """
    required_fields = frozenset({"weight_packed", "weight_scale", "weight_global_scale"})
    pending: dict[str, dict[str, torch.Tensor]] = {}
    converted: set[str] = set()
    for name, tensor in weights:
        match = re.fullmatch(
            r"(?:language_model\.)?(?:model\.)?(layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123])\."
            r"(weight_packed|weight_scale|weight_global_scale|input_global_scale)",
            name,
        )
        if match is None:
            yield name, tensor
            continue
        prefix, field = match.groups()
        if prefix not in expected_prefixes or field == "input_global_scale":
            # NVFP4 activation calibration is replaced by dynamic MXFP8.
            continue
        if prefix in converted or field in pending.get(prefix, {}):
            raise ValueError(f"Duplicate native NVFP4 checkpoint tensor: {name}")
        components = pending.setdefault(prefix, {})
        components[field] = tensor
        if components.keys() != required_fields:
            continue
        quantized, scales = requantize_nvfp4_to_mxfp8(
            components["weight_packed"],
            components["weight_scale"],
            components["weight_global_scale"],
            dtype,
            invert_global_scale=True,
        )
        del pending[prefix]
        del components
        converted.add(prefix)
        checkpoint_prefix = name.rsplit(".", 1)[0]
        yield f"{checkpoint_prefix}.weight", quantized
        yield f"{checkpoint_prefix}.weight_scale", scales
    missing = expected_prefixes - converted
    if missing:
        examples = ", ".join(sorted(missing)[:MAX_MISSING_PROJECTION_EXAMPLES])
        raise ValueError(f"Incomplete native NVFP4 checkpoint: {len(missing)} local projections missing ({examples})")
