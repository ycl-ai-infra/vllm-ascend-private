// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#pragma once

namespace vllm_ascend {
void msa_index_decode_topk_impl(int coreCount, void* stream, void* score, void* context, void* indices,
    void* values, void* counts, int total, int heads, int columns, int strideH, int strideT,
    int qlen, int topk, int initBlocks, int localBlocks);

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_msa_index_decode_topk(
    const at::Tensor& score, const at::Tensor& context, int64_t qlen, int64_t topk,
    int64_t initBlocks, int64_t localBlocks)
{
    using namespace msa_sparse_attention_fia_detail;
    CheckMetadataTensor(context, at::kInt, score.device());
    TORCH_CHECK(score.scalar_type() == at::kFloat && score.dim() == 3 && score.stride(2) == 1);
    TORCH_CHECK(context.dim() == 1 && qlen > 0 && context.numel() * qlen == score.size(1));
    // Sort workspace is ten FP32 arrays (<=80 KiB at 2048 columns).
    // Aligned TopK rows keep concurrent output writes on separate cache lines.
    TORCH_CHECK(score.size(0) > 0 && score.size(2) > 0 && score.size(2) <= 2048);
    TORCH_CHECK(topk > 0 && topk <= 128 && topk % 8 == 0 && initBlocks >= 0 && localBlocks >= 0);
    TORCH_CHECK(score.numel() <= INT32_MAX && score.stride(0) <= INT32_MAX && score.stride(1) <= INT32_MAX);
    const auto heads = score.size(0), total = score.size(1);
    auto indices = at::empty({heads, total, topk}, score.options().dtype(at::kInt));
    auto values = at::empty_like(indices, score.options());
    auto counts = at::empty({heads, total}, indices.options());
    if (total == 0) return {indices, values, counts};
    auto stream = c10_npu::getCurrentNPUStream().stream();
    auto cores = VectorCoreCount();
    at_npu::native::OpCommand cmd;
    cmd.Name("msa_index_decode_topk");
    cmd.SetCustomHandler([=]() -> int {
        msa_index_decode_topk_impl(cores, stream, score.data_ptr(), context.data_ptr(),
            indices.data_ptr(), values.data_ptr(), counts.data_ptr(), total, heads, score.size(2),
            score.stride(0), score.stride(1), qlen, topk, initBlocks, localBlocks);
        return 0;
    });
    cmd.Run();
    return {indices, values, counts};
}
} // namespace vllm_ascend
