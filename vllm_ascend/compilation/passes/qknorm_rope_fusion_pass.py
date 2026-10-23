#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
#
import torch
from torch._inductor.pattern_matcher import Match, PatternMatcherPass, PatternPrettyPrinter
from torch.utils._pytree import tree_leaves
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import Range
from vllm.logger import logger
from vllm.model_executor.layers.attention import Attention
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import get_rope_dim

if HAS_TRITON:
    from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope_vnorm import qkv_rmsnorm_rope_vnorm_fits_ub


def _inputs_precede_outputs(match: Match) -> bool:
    """Reject replacement inputs that would become forward references.

    Multi-output replacements are inserted before their earliest output.
    Gemma weight offsets shared with other consumers can stay outside the
    complete Gemma pattern, so the generic RMSNorm patterns need this guard
    as well. Reject only that match; leave the original operations intact.
    """
    outputs = [node for node in match.output_nodes() if isinstance(node, torch.fx.Node)]
    if not outputs:
        return False
    positions = {node: index for index, node in enumerate(outputs[0].graph.nodes)}
    first_output = min(positions[node] for node in outputs)
    return all(
        positions.get(node, first_output) < first_output
        for node in tree_leaves((match.args, match.kwargs))
        if isinstance(node, torch.fx.Node)
    )


class QKNormRopeFusionPattern(BasePattern):
    def __init__(self, vllm_config, head_dim, num_heads, num_kv_heads, eps=1e-6):
        super().__init__(vllm_config, eps)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.device = vllm_config.device_config.device if vllm_config.device_config else None
        self.rope_dim = get_rope_dim(vllm_config)

    def get_extra_check(self):
        if get_current_hardware_profile().device_adaptor_family == DeviceAdaptorFamily.FP8_OPTIMIZED:
            return _inputs_precede_outputs
        return None

    def get_inputs(self):
        T = 5
        max_position_embeddings = 16384
        qkv = torch.empty(T, self.q_size + 2 * self.kv_size, dtype=torch.bfloat16, device="npu")
        q_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        cos_sin_cache = torch.empty(max_position_embeddings, self.head_dim, dtype=torch.bfloat16, device="npu")
        positions = torch.ones(T, dtype=torch.int64, device="npu")
        return [qkv, q_weight, k_weight, cos_sin_cache, positions]

    def get_pattern(self):
        def pattern(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight, self.eps)

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight, self.eps)

            q_flat = q_norm_out.view(q.shape)
            k_flat = k_norm_out.view(k.shape)
            q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
                positions, q_flat, k_flat, cos_sin_cache, self.head_dim, self.rope_dim, True
            )

            return q_rope, k_rope, v

        return pattern

    def get_replacement(self):
        def replacement(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            results = DeviceOperator.split_qkv_rmsnorm_rope(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.eps,
                q_bias=None,
                k_bias=None,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
            )

            return results

        return replacement


class GemmaQKNormRopeFusionPattern(QKNormRopeFusionPattern):
    """Keep Gemma's weight offsets inside the multi-output replacement.

    The matcher inserts the replacement before its earliest output, which is
    normally the value split. Treating ``1 + weight`` as an external input can
    therefore place the fused op before that input's producer. Rebuilding both
    offsets in the replacement preserves their dependencies without reordering
    unrelated nodes in the graph.
    """

    def get_pattern(self):
        rms_pattern = super().get_pattern()

        def pattern(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            return rms_pattern(qkv, 1.0 + q_weight, 1.0 + k_weight, cos_sin_cache, positions)

        return pattern

    def get_replacement(self):
        rms_replacement = super().get_replacement()

        def replacement(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            return rms_replacement(qkv, 1.0 + q_weight, 1.0 + k_weight, cos_sin_cache, positions)

        return replacement


class QKNormRopeFusionPatternWithBias(BasePattern):
    def __init__(self, vllm_config, head_dim, num_heads, num_kv_heads, eps=1e-6):
        super().__init__(vllm_config, eps)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.device = vllm_config.device_config.device if vllm_config.device_config else None
        self.rope_dim = get_rope_dim(vllm_config)

    def get_extra_check(self):
        if get_current_hardware_profile().device_adaptor_family == DeviceAdaptorFamily.FP8_OPTIMIZED:
            return _inputs_precede_outputs
        return None

    def get_inputs(self):
        T = 5
        max_position_embeddings = 16384
        qkv = torch.empty(T, self.q_size + 2 * self.kv_size, dtype=torch.bfloat16, device="npu")
        q_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        q_bias = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_bias = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        cos_sin_cache = torch.empty(max_position_embeddings, self.head_dim, dtype=torch.bfloat16, device="npu")
        positions = torch.ones(T, dtype=torch.int64, device="npu")

        return [qkv, q_weight, k_weight, q_bias, k_bias, cos_sin_cache, positions]

    def get_pattern(self):
        def pattern(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            q_bias: torch.Tensor,
            k_bias: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight, self.eps)
            q_normed = q_norm_out + q_bias

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight, self.eps)
            k_normed = k_norm_out + k_bias

            q_flat = q_normed.view(q.shape)
            k_flat = k_normed.view(k.shape)
            q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
                positions, q_flat, k_flat, cos_sin_cache, self.head_dim, self.rope_dim, True
            )

            return q_rope, k_rope, v

        return pattern

    def get_replacement(self):
        def replacement(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            q_bias: torch.Tensor,
            k_bias: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            results = DeviceOperator.split_qkv_rmsnorm_rope(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.eps,
                q_bias=q_bias,
                k_bias=k_bias,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
            )
            return results

        return replacement


class QKVNormRopeFusionPattern(BasePattern):
    def __init__(self, vllm_config, head_dim, num_heads, num_kv_heads, eps=1e-6):
        super().__init__(vllm_config, eps)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.device = vllm_config.device_config.device if vllm_config.device_config else None
        # Both Gemma4 attention types rotate the full head: Gemma4RotaryEmbedding
        # passes `rotary_dim=head_size` to the base class, so the cos/sin cache
        # is full width and the model-level `get_rope_dim` is not needed here.
        self.rope_dim = head_dim

    def pattern_key(self) -> str:
        # Registered once per attention shape, so the shape has to be part of
        # the identity or the second registration is silently skipped.
        return f"{super().pattern_key()}_hd{self.head_dim}_nh{self.num_heads}_nkv{self.num_kv_heads}_rd{self.rope_dim}"

    def get_inputs(self):
        T = 5
        max_position_embeddings = 16384
        qkv = torch.empty(T, self.q_size + 2 * self.kv_size, dtype=torch.bfloat16, device="npu")
        q_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        v_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        cos_sin_cache = torch.empty(max_position_embeddings, self.head_dim, dtype=torch.bfloat16, device="npu")
        positions = torch.ones(T, dtype=torch.int64, device="npu")
        return [qkv, q_weight, k_weight, v_weight, cos_sin_cache, positions]

    def get_pattern(self):
        def pattern(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            v_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            # The structural match treats the split sizes as a wildcard, so a
            # pattern registered for one attention shape is offered graphs of
            # every other shape too. The matcher rejects those by re-tracing
            # this function against the matched tensors, but only a
            # RuntimeError counts as a mismatch there, while split raises
            # ValueError when the sizes do not add up. vLLM's own
            # qk_norm_rope_fusion pass converts it the same way.
            try:
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            except ValueError as e:
                raise RuntimeError from e

            # Mirrors the unflatten/flatten idiom in gemma4.py and gemma3n.py,
            # not the view idiom qwen3.py uses for the base patterns.
            q_by_head = q.unflatten(-1, (self.num_heads, self.head_dim))
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight, self.eps)

            k_by_head = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight, self.eps)

            q_flat = q_norm_out.flatten(-2, -1)
            k_flat = k_norm_out.flatten(-2, -1)
            q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
                positions, q_flat, k_flat, cos_sin_cache, self.head_dim, self.rope_dim, True
            )

            v_by_head = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v_norm_out, _ = torch.ops.npu.npu_rms_norm(v_by_head, v_weight, self.eps)
            v_flat = v_norm_out.flatten(-2, -1)

            return q_rope, k_rope, v_flat

        return pattern

    def get_replacement(self):
        def replacement(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            v_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            results = DeviceOperator.split_qkv_rmsnorm_rope_vnorm(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.eps,
                q_bias=None,
                k_bias=None,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
            )

            return results

        return replacement


class QKNormRopeFusionPass(VllmInductorPass):
    """
    A pass for fusing QKV split and RMSNorm operations into a single qk_rmsnorm operator.
    """

    def uuid(self) -> str:
        if get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED:
            return super().uuid()
        return self.source_hash()

    @classmethod
    def source_hash(cls) -> str:
        # AscendCompiler also needs this hash: npugraph_ex's serialized cache
        # does not consume individual pass UUIDs. Avoid constructing a pass,
        # which would register patterns and allocate NPU example tensors.
        return cls.hash_source(
            cls,
            BasePattern,
            QKNormRopeFusionPattern,
            GemmaQKNormRopeFusionPattern,
            QKNormRopeFusionPatternWithBias,
            _inputs_precede_outputs,
        )

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="qknorm_rope_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16,):
            logger.debug("QKNorm and Rope fusion not enabled: unsupported dtype %s", dtype)
            return

        # use one attn layer to get meta (such as head_dim) for QKNormRopeFusionPattern
        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(vllm_config, Attention)
        if len(attn_layers) == 0:
            logger.debug("QKNorm and Rope fusion enabled, but no Attention layers were discovered.")
            return
        layer = next(iter(attn_layers.values()))
        # This adaptor expands Gemma RMSNorm into a weight offset followed by
        # npu_rms_norm. Other adaptors retain their existing fusion patterns.
        uses_gemma_weight_offsets = (
            get_current_hardware_profile().device_adaptor_family == DeviceAdaptorFamily.FP8_OPTIMIZED
        )
        for epsilon in [1e-6, 1e-5]:
            if layer.head_size != 128:
                logger.debug("QKNorm and Rope fusion not enabled: head_dim %d is not equal of 128", layer.head_size)
                continue
            if uses_gemma_weight_offsets:
                # Prefer the complete Gemma pattern over a generic RMSNorm
                # match that would leave its weight-offset producers outside.
                GemmaQKNormRopeFusionPattern(
                    vllm_config=vllm_config,
                    head_dim=layer.head_size,
                    num_heads=layer.num_heads,
                    num_kv_heads=layer.num_kv_heads,
                    eps=epsilon,
                ).register(self.pattern_match_passes)
            QKNormRopeFusionPattern(
                vllm_config=vllm_config,
                head_dim=layer.head_size,
                num_heads=layer.num_heads,
                num_kv_heads=layer.num_kv_heads,
                eps=epsilon,
            ).register(self.pattern_match_passes)

            QKNormRopeFusionPatternWithBias(
                vllm_config=vllm_config,
                head_dim=layer.head_size,
                num_heads=layer.num_heads,
                num_kv_heads=layer.num_kv_heads,
                eps=epsilon,
            ).register(self.pattern_match_passes)

        if not HAS_TRITON:
            logger.debug("QKVNorm and Rope fusion not enabled: triton is unavailable")
            return
        # The q/k-only patterns above stay available here: DeviceOperator routes
        # them to a per-hardware kernel, while split_qkv_rmsnorm_rope_vnorm has
        # only the SIMD Triton kernel.
        if not get_current_hardware_profile().supports(HardwareCapability.GRAPH_QKV_NORM_ROPE_FUSION):
            logger.info_once(
                "QKVNorm and Rope fusion not enabled: split_qkv_rmsnorm_rope_vnorm has no kernel for this hardware",
                scope="global",
            )
            return
        for head_dim, num_heads, num_kv_heads in sorted(
            {(a.head_size, a.num_heads, a.num_kv_heads) for a in attn_layers.values()}
        ):
            if not qkv_rmsnorm_rope_vnorm_fits_ub(
                q_hidden_size=num_heads * head_dim,
                kv_hidden_size=num_kv_heads * head_dim,
                head_dim=head_dim,
                rope_dim=head_dim,
            ):
                logger.info_once(
                    "QKVNorm and Rope fusion not enabled for head_dim %d (num_heads %d, num_kv_heads %d): one "
                    "token's tiles exceed the vector core unified buffer, which a larger tensor parallel size "
                    "shrinks",
                    head_dim,
                    num_heads,
                    num_kv_heads,
                    scope="global",
                )
                continue
            logger.debug(
                "QKVNorm and Rope fusion registered for head_dim %d (num_heads %d, num_kv_heads %d)",
                head_dim,
                num_heads,
                num_kv_heads,
            )
            for epsilon in [1e-6, 1e-5]:
                QKVNormRopeFusionPattern(
                    vllm_config=vllm_config,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    eps=epsilon,
                ).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s QKNorm and Rope patterns", self.matched_count)
        logger.debug("Patterns registered for replacement:")
        pattern_idx = 0
        for pattern_entry in self.pattern_match_passes.patterns.values():
            for p in pattern_entry:
                p_str = PatternPrettyPrinter.run(p.pattern)
                logger.debug("Pattern %d: %s", pattern_idx, p_str)
                pattern_idx += 1
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
