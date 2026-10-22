// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#pragma once

namespace vllm_ascend {
namespace msa_sparse_attention_fia_detail {
constexpr int64_t MAX_BUFFER_BYTES = 65536;
constexpr int64_t VALID_ROW_STRIDE = 8;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t COPY_ALIGNMENT = 16;

inline int VectorCoreCount()
{
    int device = 0;
    int64_t cores = 0;
    TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS);
    TORCH_CHECK(aclGetDeviceCapability(device, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &cores) == ACL_SUCCESS);
    TORCH_CHECK(cores > 0);
    return static_cast<int>(cores);
}

inline void CheckMetadataTensor(const at::Tensor& tensor, at::ScalarType dtype, const c10::Device& device)
{
    TORCH_CHECK(tensor.device() == device && tensor.is_contiguous() && tensor.scalar_type() == dtype,
                "FIA metadata must be contiguous, on the same NPU, and use its declared integer dtype");
}
} // namespace msa_sparse_attention_fia_detail

void msa_sparse_attention_fia_metadata_impl(
    int coreCount, void* stream, void* indices, void* table, void* counts, void* prefix,
    void* lengths, void* pages, void* mask, void* valid, int totalTokens, int heads,
    int topk, int tableWidth, int requests, int begin, int chunk, int pageCount, int blockSize);

void msa_sparse_attention_fia_finish_impl(
    int coreCount, void* stream, void* input, void* valid, void* output,
    int tokens, int heads, int kvHeads, int dim);

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_msa_sparse_attention_fia_metadata(
    const at::Tensor& indices, const at::Tensor& table, const at::Tensor& counts,
    const at::Tensor& prefix, const at::Tensor& lengths,
    int64_t begin, int64_t chunk, int64_t pageCount, int64_t blockSize)
{
    using namespace msa_sparse_attention_fia_detail;
    for (const auto& tensor : {indices, table, counts, lengths}) {
        CheckMetadataTensor(tensor, at::kInt, indices.device());
    }
    CheckMetadataTensor(prefix, at::kLong, indices.device());
    TORCH_CHECK(indices.dim() == 3 && table.dim() == 2 && counts.dim() == 2 &&
                prefix.dim() == 1 && lengths.dim() == 1);
    TORCH_CHECK(counts.size(0) == indices.size(0) && counts.size(1) == indices.size(1));
    TORCH_CHECK(lengths.numel() > 0 && table.size(0) == lengths.numel() &&
                prefix.numel() == lengths.numel() + 1 && table.size(1) > 0);
    TORCH_CHECK(begin >= 0 && chunk > 0 && begin + chunk <= indices.size(1) &&
                pageCount > 0 && blockSize == BLOCK_SIZE);
    const int64_t heads = indices.size(0), topk = indices.size(2);
    TORCH_CHECK(heads > 0 && topk > 0 && topk * blockSize <= MAX_BUFFER_BYTES);
    const int64_t batch = heads * chunk;
    auto pages = at::empty({batch, topk}, indices.options());
    auto mask = at::empty({batch, 1, 1, topk * blockSize}, indices.options().dtype(at::kBool));
    // Give each core an aligned, private validity row to avoid partial cache-line writes.
    auto valid = at::empty({batch, VALID_ROW_STRIDE}, indices.options());
    auto stream = c10_npu::getCurrentNPUStream().stream();
    auto cores = VectorCoreCount();
    at_npu::native::OpCommand cmd;
    cmd.Name("msa_sparse_attention_fia_metadata");
    cmd.SetCustomHandler([=]() -> int {
        msa_sparse_attention_fia_metadata_impl(
            cores, stream, indices.data_ptr(), table.data_ptr(), counts.data_ptr(), prefix.data_ptr(),
            lengths.data_ptr(), pages.data_ptr(), mask.data_ptr(), valid.data_ptr(),
            indices.size(1), heads, topk, table.size(1), lengths.numel(), begin, chunk, pageCount, blockSize);
        return 0;
    });
    cmd.Run();
    return {pages, mask, valid};
}

void npu_msa_sparse_attention_fia_finish(
    const at::Tensor& input, const at::Tensor& valid, at::Tensor& output, int64_t kvHeads)
{
    using namespace msa_sparse_attention_fia_detail;
    CheckMetadataTensor(valid, at::kInt, input.device());
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16 &&
                input.device() == output.device() && input.is_contiguous() && output.is_contiguous());
    TORCH_CHECK(output.dim() == 3 && input.dim() == 3 && kvHeads > 0 && output.size(1) % kvHeads == 0);
    const int64_t tokens = output.size(0), heads = output.size(1), dim = output.size(2);
    const int64_t group = heads / kvHeads * dim;
    TORCH_CHECK(group % COPY_ALIGNMENT == 0 && group * input.element_size() <= MAX_BUFFER_BYTES);
    TORCH_CHECK(input.size(0) == kvHeads * tokens && input.size(1) == 1 && input.size(2) == heads * dim);
    TORCH_CHECK(valid.dim() == 2 && valid.size(0) == kvHeads * tokens && valid.size(1) == VALID_ROW_STRIDE);
    if (tokens == 0) {
        return;
    }
    auto stream = c10_npu::getCurrentNPUStream().stream();
    auto cores = VectorCoreCount();
    at_npu::native::OpCommand cmd;
    cmd.Name("msa_sparse_attention_fia_finish");
    cmd.SetCustomHandler([=]() -> int {
        msa_sparse_attention_fia_finish_impl(cores, stream, input.data_ptr(), valid.data_ptr(), output.data_ptr(),
                                            tokens, heads, kvHeads, dim);
        return 0;
    });
    cmd.Run();
}
} // namespace vllm_ascend
