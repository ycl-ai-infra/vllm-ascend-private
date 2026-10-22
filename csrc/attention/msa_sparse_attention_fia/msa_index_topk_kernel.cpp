// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "kernel_operator.h"
using namespace AscendC;

__aicore__ inline void FillRange(LocalTensor<float> x, int begin, int end, float value)
{
    // Scalar edge writes and vector fills must not race, including when the
    // local forced range overlaps a preceding init range.
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    int alignedBegin = min(end, (begin + 7) / 8 * 8);
    int alignedEnd = max(alignedBegin, end / 8 * 8);
    for (int i = begin; i < alignedBegin; ++i) x.SetValue(i, value);
    for (int i = alignedEnd; i < end; ++i) x.SetValue(i, value);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    if (alignedBegin < alignedEnd) Duplicate(x[alignedBegin], value, alignedEnd - alignedBegin);
    PipeBarrier<PIPE_V>();
}

extern "C" __global__ __aicore__ void msa_index_decode_topk(
    GM_ADDR score, GM_ADDR context, GM_ADDR indices, GM_ADDR values, GM_ADDR counts,
    int total, int heads, int columns, int strideH, int strideT, int qlen,
    int topk, int initBlocks, int localBlocks)
{
    GlobalTensor<float> src, dstVal;
    GlobalTensor<int32_t> ctx, dstIdx, dstCount;
    src.SetGlobalBuffer((__gm__ float*)score);
    ctx.SetGlobalBuffer((__gm__ int32_t*)context);
    dstIdx.SetGlobalBuffer((__gm__ int32_t*)indices);
    dstVal.SetGlobalBuffer((__gm__ float*)values);
    dstCount.SetGlobalBuffer((__gm__ int32_t*)counts);
    int padded = (columns + 31) / 32 * 32;
    int paddedK = (topk + 7) / 8 * 8;
    TPipe pipe;
    TBuf<TPosition::VECCALC> scoreBuf, indexBuf, tmpBuf, outIdxBuf, outValBuf, countBuf;
    pipe.InitBuffer(scoreBuf, padded * 4);
    pipe.InitBuffer(indexBuf, padded * 4);
    pipe.InitBuffer(tmpBuf, padded * 4 * 8);
    pipe.InitBuffer(outIdxBuf, paddedK * 4);
    pipe.InitBuffer(outValBuf, paddedK * 4);
    pipe.InitBuffer(countBuf, 32);
    auto x = scoreBuf.Get<float>();
    auto ids = indexBuf.Get<uint32_t>();
    auto sorted = tmpBuf.Get<float>();
    auto scratch = tmpBuf.Get<float>()[padded * 2];
    auto outIdx = outIdxBuf.Get<int32_t>();
    auto outVal = outValBuf.Get<float>();
    auto outCount = countBuf.Get<int32_t>();
    ArithProgression(ids.ReinterpretCast<int32_t>(), int32_t(0), int32_t(1), padded);
    const float negInf = -__builtin_inff();
    for (int row = GetBlockIdx(); row < total * heads; row += GetBlockNum()) {
        int head = row / total, token = row % total;
        int req = token / qlen;
        int visible = ctx.GetValue(req) + token % qlen + 1;
        int globalValid = max(0, (visible + 127) / 128);
        int valid = min(columns, globalValid);
        DataCopyExtParams inputCopy{1, uint32_t(columns * 4), 0, 0, 0};
        DataCopyPadExtParams<float> padding{false, 0, 0, 0};
        DataCopyPad(x, src[head * strideH + token * strideT], inputCopy, padding);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (valid < padded) FillRange(x, valid, padded, negInf);
        if (initBlocks > 0 && valid > 0) FillRange(x, 0, min(valid, initBlocks), float(1.0e30));
        PipeBarrier<PIPE_V>();
        int localStart = max(0, globalValid - localBlocks);
        if (localBlocks > 0 && localStart < valid) FillRange(x, localStart, valid, float(1.0e29));
        PipeBarrier<PIPE_V>();
        Sort<float, true>(sorted, x, ids, scratch, padded / 32);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int i = 0; i < topk; ++i) {
            int index = i < columns ? int(sorted.ReinterpretCast<uint32_t>().GetValue(i * 2 + 1)) : -1;
            bool good = index >= 0 && index < valid;
            outIdx.SetValue(i, good ? index : -1);
            outVal.SetValue(i, good ? sorted.GetValue(i * 2) : negInf);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams outputCopy{1, uint32_t(topk * 4), 0, 0, 0};
        DataCopyPad(dstIdx[row * topk], outIdx, outputCopy);
        DataCopyPad(dstVal[row * topk], outVal, outputCopy);
        // One row owns each aligned group of eight counts. All rows use the
        // same visibility formula, so no inter-core partial cache-line writes
        // or second count-reduction kernel are needed.
        if (row % 8 == 0) {
            int countRows = min(8, total * heads - row);
            for (int i = 0; i < countRows; ++i) {
                int countToken = (row + i) % total;
                int length = ctx.GetValue(countToken / qlen) + countToken % qlen + 1;
                int blocks = max(0, (length + 127) / 128);
                outCount.SetValue(i, min(topk, min(columns, blocks)));
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
            DataCopyExtParams countCopy{1, uint32_t(countRows * 4), 0, 0, 0};
            DataCopyPad(dstCount[row], outCount, countCopy);
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}

namespace vllm_ascend {
void msa_index_decode_topk_impl(int coreCount, void* stream, void* score, void* context, void* indices,
    void* values, void* counts, int total, int heads, int columns, int strideH, int strideT,
    int qlen, int topk, int initBlocks, int localBlocks)
{
    int grid = total * heads < coreCount ? total * heads : coreCount;
    msa_index_decode_topk<<<grid, nullptr, stream>>>(score, context, indices, values, counts,
        total, heads, columns, strideH, strideT, qlen, topk, initBlocks, localBlocks);
}
} // namespace vllm_ascend
