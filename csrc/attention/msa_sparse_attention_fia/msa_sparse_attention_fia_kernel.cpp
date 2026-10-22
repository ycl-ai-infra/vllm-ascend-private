// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void msa_sparse_attention_fia_metadata(
    GM_ADDR indices, GM_ADDR table, GM_ADDR counts, GM_ADDR prefix, GM_ADDR lengths,
    GM_ADDR pages, GM_ADDR mask, GM_ADDR valid, int totalTokens, int heads,
    int topk, int tableWidth, int requests, int begin, int chunk, int pageCount, int blockSize)
{
    GlobalTensor<int32_t> idx, bt, cnt, lens, pg, ok;
    GlobalTensor<int64_t> cu;
    GlobalTensor<uint8_t> masks;
    idx.SetGlobalBuffer((__gm__ int32_t*)indices);
    bt.SetGlobalBuffer((__gm__ int32_t*)table);
    cnt.SetGlobalBuffer((__gm__ int32_t*)counts);
    cu.SetGlobalBuffer((__gm__ int64_t*)prefix);
    lens.SetGlobalBuffer((__gm__ int32_t*)lengths);
    pg.SetGlobalBuffer((__gm__ int32_t*)pages);
    masks.SetGlobalBuffer((__gm__ uint8_t*)mask);
    ok.SetGlobalBuffer((__gm__ int32_t*)valid);
    TPipe pipe;
    TBuf<TPosition::VECCALC> pageBuf, maskBuf, validBuf;
    int paddedTopk = (topk + 7) / 8 * 8;
    pipe.InitBuffer(pageBuf, paddedTopk * 4);
    pipe.InitBuffer(maskBuf, topk * blockSize);
    pipe.InitBuffer(validBuf, 32);
    auto localPages = pageBuf.Get<int32_t>();
    auto localMask = maskBuf.Get<uint16_t>();
    auto localValid = validBuf.Get<int32_t>();
    for (int row = GetBlockIdx(); row < heads * chunk; row += GetBlockNum()) {
        int head = row / chunk;
        int token = begin + row % chunk;
        int lo = 0, hi = requests;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cu.GetValue(mid + 1) <= token) lo = mid + 1;
            else hi = mid;
        }
        // A malformed prefix must not turn into an out-of-bounds GM read.
        bool hasRequest = lo < requests;
        int req = hasRequest ? lo : 0;
        int length = lens.GetValue(req);
        int queryPos = length - (cu.GetValue(req + 1) - cu.GetValue(req)) + token - cu.GetValue(req);
        int selected = hasRequest ? cnt.GetValue(head * totalTokens + token) : 0;
        int anyValid = 0;
        Duplicate(localMask, uint16_t(0x0101), topk * blockSize / 2);
        PipeBarrier<PIPE_V>();
        for (int slot = 0; slot < topk; ++slot) {
            int logical = idx.GetValue((head * totalTokens + token) * topk + slot);
            int safe = logical < 0 ? 0 : (logical >= tableWidth ? tableWidth - 1 : logical);
            int physical = bt.GetValue(req * tableWidth + safe);
            physical = physical < 0 ? 0 : (physical >= pageCount ? pageCount - 1 : physical);
            localPages.SetValue(slot, physical);
            int keep = 0;
            if (logical >= 0 && logical < tableWidth && slot < selected) {
                keep = min(blockSize, min(queryPos + 1, length) - logical * blockSize);
                keep = max(0, keep);
            }
            if (keep > 0) {
                anyValid = 1;
                if (keep / 2 > 0) Duplicate(localMask[slot * blockSize / 2], uint16_t(0), keep / 2);
                PipeBarrier<PIPE_V>();
                if (keep % 2) {
                    SetFlag<HardEvent::V_S>(EVENT_ID0);
                    WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    localMask.SetValue((slot * blockSize + keep - 1) / 2, uint16_t(0x0100));
                    SetFlag<HardEvent::S_V>(EVENT_ID0);
                    WaitFlag<HardEvent::S_V>(EVENT_ID0);
                }
            }
        }
        Duplicate(localValid, anyValid, 8);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams pageCopy{1, uint32_t(topk * 4), 0, 0, 0};
        DataCopyPad(pg[row * topk], localPages, pageCopy);
        DataCopy(masks[row * topk * blockSize], localMask.ReinterpretCast<uint8_t>(), topk * blockSize);
        DataCopy(ok[row * 8], localValid, 8);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
}

extern "C" __global__ __aicore__ void msa_sparse_attention_fia_finish(
    GM_ADDR input, GM_ADDR valid, GM_ADDR output, int tokens, int heads, int kvHeads, int dim)
{
    GlobalTensor<uint16_t> src, dst;
    GlobalTensor<int32_t> ok;
    src.SetGlobalBuffer((__gm__ uint16_t*)input);
    dst.SetGlobalBuffer((__gm__ uint16_t*)output);
    ok.SetGlobalBuffer((__gm__ int32_t*)valid);
    int group = heads / kvHeads * dim;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, group * 2);
    auto local = buffer.Get<uint16_t>();
    for (int row = GetBlockIdx(); row < tokens * kvHeads; row += GetBlockNum()) {
        int kvHead = row / tokens, token = row % tokens;
        if (ok.GetValue(row * 8)) {
            DataCopy(local, src[row * heads * dim + kvHead * group], group);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        } else {
            Duplicate(local, uint16_t(0), group);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
        DataCopy(dst[token * heads * dim + kvHead * group], local, group);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}

namespace vllm_ascend {
void msa_sparse_attention_fia_metadata_impl(int coreCount, void* stream, void* indices, void* table, void* counts, void* prefix,
    void* lengths, void* pages, void* mask, void* valid, int totalTokens, int heads, int topk,
    int tableWidth, int requests, int begin, int chunk, int pageCount, int blockSize)
{
    int blocks = heads * chunk < coreCount ? heads * chunk : coreCount;
    msa_sparse_attention_fia_metadata<<<blocks, nullptr, stream>>>(indices, table, counts, prefix, lengths,
        pages, mask, valid, totalTokens, heads, topk, tableWidth, requests, begin, chunk, pageCount, blockSize);
}

void msa_sparse_attention_fia_finish_impl(int coreCount, void* stream, void* input, void* valid, void* output,
    int tokens, int heads, int kvHeads, int dim)
{
    int blocks = tokens * kvHeads < coreCount ? tokens * kvHeads : coreCount;
    msa_sparse_attention_fia_finish<<<blocks, nullptr, stream>>>(input, valid, output, tokens, heads, kvHeads, dim);
}

} // namespace vllm_ascend
