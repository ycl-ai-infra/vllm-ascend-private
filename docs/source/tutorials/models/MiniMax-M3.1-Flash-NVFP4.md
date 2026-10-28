# MiniMax-M3.1-Flash 原生 NVFP4 权重

## Introduction

原生 MiniMax-M3.1-Flash checkpoint 的普通线性层使用 MXFP8，routed experts
使用 NVFP4。Ascend 在加载时使用 Torch 将 NVFP4 解码到模型 dtype，再转换成
MXFP8，运行时复用现有 W8A8 MXFP8 算子；不生成另一份离线转换权重。
模型 text_config 为 60 层、hidden size 6,144、128 个 routed experts、每 token top-4。

适配要求原生配置包含 `moe_quant_algo: NVFP4`、
`moe_quant_format: nvfp4-pack-quantized` 和 `group_size: 16`。
缺失本 rank 应加载的专家投影或配对 scale 会直接报错。

## Supported Features

本教程的部署配置为 8 张 950DT、TP4/DP2/EP、V2 model runner、
FP8 KV cache 和 `FULL_DECODE_ONLY`，不启用 speculative decoding。
GPQA 使用文本输入。

### 权重与 activation 的数值约定

- NVFP4 每 16 个元素一个 E4M3 block scale；同一字节先解码低 nibble，再解码高 nibble。
- 原生 `weight_global_scale` 是编码 scale，解码公式为
  `E2M1_value * block_scale / weight_global_scale`。
  gate 和 up 的全局 scale 分别使用，不能合并后取最大值。
- 解码的 FP32 结果先舍入为模型 dtype，本教程使用 BF16。
- MXFP8 每 32 个元素共享一个 E8M0 scale：
  `scale = 2 ** ceil(log2(amax / 448))`。
  FP8 数据和 scale 保持普通行优先布局，之后交给现有 Ascend 布局转换。
- activation 使用同一 ceil 规则，对应 `npu_dynamic_mx_quant` 的 `scale_alg=1`。
  原生 NVFP4 的 `input_global_scale` 不用于 MXFP8 activation。
- A5 Gemma RMSNorm 的 `1 + weight` 在 FP32 中计算；residual fallback
  用 FP32 求和及归一化，最后将输出和 residual 转回输入 dtype，避免 offset 提前舍入。
- attention 保持 Ascend 实现的 FP8 KV cache。
  本部署的原生 BF16、EP 序列并行路径使用 AllGather；dispatch 传输已有 FP8/E8M0，
  routed GEMM2、路由权重乘法、求和和 EP reduce-scatter 保持 FP32，最后转回 BF16。
  MC2 combine 不接收 FP32，因此此路径使用已有 AllGather 通信，不额外量化 combine。

### 上游参考路径

走读固定到 vLLM main 的 `224b2f54f19d44bfb96edfe16c6343c7495853d3`：
`dequantize_nvfp4_moe_weights` 根据 checkpoint 约定选择是否反转 global scale，
分别解码 gate/up，再调用 `dequantize_to_dtype` 舍入到模型 dtype；
`_mxfp8_e4m3_quantize_torch` 随后生成 FP8 与 E8M0。
这些函数位于上游 `quantization/utils/nvfp4_emulation_utils.py` 和
`quantization/utils/mxfp8_utils.py`。
本适配直接实现这条 Torch 数值路径，避免上游包装函数自动选择 Triton、CUDA 或 FlashInfer 后端。

### 原生 MXFP8 activation 调用链

vLLM main `b1c66e55450d4869a7c79bd23df2092306f04adf` 的实际路径为
`fused_moe/experts/mxfp8_native_moe.py` →
`models/minimax_m3/amd/ops/swiglu_oai.py::swiglu_oai_quantize_mxfp8`。
本适配只读取这些 kernel 的数值语义，使用 Torch 重写，不执行 Triton/CUDA kernel。
通用 CUDA packed SwiGLU 和这条 native MXFP8 路径的舍入位置不同，不能互相替代作为参考。

| 阶段 | routed MoE dtype |
| --- | --- |
| GEMM1 输出 | BF16 |
| clamp、sigmoid、up bias、乘法 | FP32 |
| GEMM2 activation 量化输入 | FP32，保持到 `scale_alg=1` |
| GEMM2 输入 / scale | E4M3 / E8M0，每组 32 个值 |
| GEMM2 输出、路由权重乘法 / 求和 | FP32 |
| EP reduce-scatter | FP32，完成后转回 BF16 |

在实际 MoE 调用入口验证量化器接收到 FP32；NPU 量化结果与同一 FP32 输入的 Torch 参考逐字节一致。
FP32 GEMM2/通信策略仅用于原生 NVFP4 MiniMax、BF16、A5、EP 序列并行且未启用 LoRA 的配置。
其余配置保留原通信选择与输出 dtype。AllGather 会增加通信量，本次优先验证精度。

## Environment Preparation

=== "950DT / A5"

    使用支持 DynamicMxQuantV3 `scale_alg=1` 的 CANN、torch-npu 和 vLLM Ascend 环境。
    将原生模型目录挂载到 `/models/MiniMax-M3.1-Flash`，安装包含本适配及 PR16 的插件代码。
    checkpoint 需要完整的 tokenizer、processor、generation config 和索引引用的全部分片。
    索引中的 `dspark/` 子目录分片也必须存在，即使本次没有启用 DSpark。

=== "A2 / A3"

    本适配的 MXFP8 精度验证针对 950DT，不据此声明 A2/A3 已通过验证。
    其他硬件需要另外确认 MXFP8 算子和 ceil scale 算法支持。

## Deployment

在容器中加载 CANN 环境后，使用以下命令。模型配置的理论上下文长度为
1,048,576 tokens，本例请求的服务上限为 133,000 tokens。

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /models/MiniMax-M3.1-Flash \
  --host 0.0.0.0 --port 11223 --served-model-name minimax-m3 \
  --trust-remote-code --distributed-executor-backend mp \
  --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel \
  --dtype bfloat16 --quantization mxfp8 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --max-model-len 133000 --max-num-batched-tokens 16384 --max-num-seqs 500 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill --enable-prefix-caching --async-scheduling \
  --reasoning-parser minimax_m3 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":false,"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_flashcomm1":true}' \
  --safetensors-load-strategy prefetch
```

`--quantization mxfp8` 选择运行时 MXFP8 路径，不表示磁盘上的 NVFP4 专家权重已被修改。
部署原生 checkpoint 时不应强制选择 ModelSlim 的 `--quantization ascend`。

## Functional Verification

确认 `/v1/models` 返回成功后，发送真实推理请求；仅进程启动或权重加载成功不足以证明服务可用。

```bash
curl -sS http://127.0.0.1:11223/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3","messages":[{"role":"user","content":"Compute 12 * 13. Give the result."}],"max_tokens":2048,"temperature":0,"reasoning_effort":"max"}'
```

## Accuracy Evaluation

GPQA Diamond 使用完整 198 题，AISBench 配置为：

| 参数 | 值 |
| --- | --- |
| batch_size | 32 |
| max_out_len | 36000 |
| temperature | 0.6 |
| top_p | 0.95 |
| reasoning_effort | max |

保留逐条请求、答案、评分和服务返回的 `finish_reason`。
只将 `finish_reason=length` 的请求记为截断；回答错误、解析失败和 HTTP 错误不能作为截断排除。
同时报告全量分数、非截断分数、截断数量和错误数量；本次验收要求非截断准确率严格大于 91%。

已完成的完整实测结果如下，准确率单位为百分比：

| 轮次 | 正确 / 全量 | 全量准确率 | 截断 | 正确 / 非截断 | 非截断准确率 | HTTP 错误 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 165 / 198 | 83.3333 | 16 | 165 / 182 | 90.6593 | 0 |
| 2 | 163 / 198 | 82.3232 | 17 | 161 / 181 | 88.9503 | 0 |
| 3 | 165 / 198 | 83.3333 | 17 | 164 / 181 | 90.6077 | 0 |
| 4 | 165 / 198 | 83.3333 | 16 | 164 / 182 | 90.1099 | 0 |

这四轮均未达到非截断准确率 >91% 的验收条件。每轮 198 条请求的参数、UUID、
最终答案和 finish reason 逐条交叉核对通过，监控的在途请求上限均为 32。
第二轮有 2 条截断请求恰好得到正确答案，也按统一规则从非截断分子、分母中同时排除。
各轮分别记录，不以单题重试结果替换原答案。

第 3 轮尝试对齐通用 CUDA packed SwiGLU 的中间舍入；完整评分和独立审计均已完成，仍未达标。
继续走读实际 native MXFP8 调用链后发现，它采用 FP32 SwiGLU 直接生成 MXFP8，
不能用通用 packed CUDA 的 BF16 中间舍入作为这条路径的参考。
最终实现按 native MXFP8 保留 routed activation 的 FP32 值到量化入口；
dense/shared activation 仅在输出时转回模型 dtype。
同时修正 Gemma RMSNorm 的 FP32 offset/residual。第 4 轮完整复测仍未达到目标。
继续对齐 native MXFP8 的 GEMM2 和路由求和 FP32 精度后，26.1 的第 5 次启动完成权重加载，
但在 vision profiling 的 HCCL 初始化失败，未发送 GPQA 请求，因此没有第 5 轮分数。
用户要求释放卡后，26.1 服务与评测容器已停止；后续迁移到 27.1 继续验证。
27.1 候选已完成 49/49 原生分片加载、49/49 FULL_DECODE_ONLY 图捕获；
真实文本返回 156、图片返回 Red。FP32 AllGather/ReduceScatter 图重放及实际 AIV/HCCL 预检通过。
使用用户指定的 26.1 chat_template.jinja，替换前已备份 27.1 旧文件。
新 GPQA 分数尚未完成，不能将上述功能验证当作准确率通过。

### 真实权重验证

| 项目 | 本次验证 |
| --- | --- |
| 权重加载 | 49/49 分片，约 56.58 GiB/rank；本地 2,736 个专家投影/rank 完整性检查通过 |
| FULL_DECODE_ONLY | 49/49 图捕获；真实文本请求正确返回 156 |
| TP4/DP2/EP、flashcomm1、共享专家 DP/重叠 | 同时启用并完成真实请求 |
| Torch 参考对齐 | 36 组、127,488 个 FP8 值；解码结果、FP8、E8M0 逐字节相同 |
| 真 NPU 转换/activation | 真实专家投影与 6,144/3,072 维 activation 数据及 scale 字节差异为 0 |
| 私有 PR16 基线 / 当前工作区单测 | 355 passed（4 skipped）/ 363 passed（4 skipped） |
| V2 专项 / 真 NPU e2e | 148 passed / 13 passed |
| FP32 通信图 | 8 rank AllGather + ReduceScatter 捕获及两次重放通过 |
| 完整格式检查 | `bash format.sh ci` 通过 |
| MTP/DSpark | 本次精度配置未启用 |

未使用 dummy 权重代替真实验证。133,000 是服务配置上限，未据此声明完成满上下文压力测试。
首次加载后触发 API 默认 600 秒初始化超时；将上游已有
`VLLM_ENGINE_READY_TIMEOUT_S` 设置为 3,600 后成功，无需 eager 回退。
若 checkpoint 索引引用 `dspark/` 内分片，适配会补充实际存在的嵌套分片，
并保留上游缺失文件检查；不会忽略索引中真正缺失的文件。

### PR63232 的最终 sparse TopK 语义核查

核查固定使用 [vLLM PR63232](https://github.com/vllm-project/vllm/pull/63232)
的提交 8d0f05a0c33c91cc4d3f0a6f189474420af47174，及其固定的
[MSA 25c1447 实现](https://github.com/vllm-project/MSA/blob/25c144700f727b638067e98066a60507081b63ad/python/fmha_sm100/csrc/include/sparse_topk_select.cuh)。
NVIDIA 的最终 sparse_topk_select 显式接收首尾强制块数和每个 token 的有效页数。
尾块并非仅靠原始 index score 保留；遗漏约束会改变 attention 可见的 token 集合，可能影响精度。

本次原生配置为 block size 128、TopK 16、init 0、local 1。
当前 A5 prefill 在最终 TopK 前将无效页设为负无穷，并提升有效首尾页分数；
decode 按 seq_len - query_len + offset + 1 计算当前 token 的可见长度，
对完整尾页和残尾页均强制保留。1e29/1e30 哨兵足以覆盖本路径有限 FP8 QK 的分值范围。
下游 GBSA wrapper 将逻辑页 ID 升序排列，并保持无效槽为 -1。

27.1 真 A5 验证：

- 独立 Torch 参考先预留强制页，再对其余有效页排序；1/4 index head 共 8,260 行选择结果及有效计数一致。
- 覆盖整页/残页、混合长度、零长度 padding、未来高分/NaN、132,001 token 边界及三次图重放。
- 可见 V 设为 1，未来/陈旧槽设为 448，让残页位于输入索引首位；
  混合 decode、4-token decode、跨页 prefill 共 166 query 输出均为 1，最大绝对误差为 0。
- 回归入口 tests/e2e/pull_request/one_card/test_minimax_m3_sparse_boundaries.py：15 passed。

这些结果支持当前强制保留及因果边界语义已对齐，未据此认定它是 GPQA 掉分原因。
未执行 NVIDIA kernel，也没有 GPU 全模型 logits 对照；raw score 数值、临界 TopK 并列、
attention 算术及 GPU Q8KV4 与本次指定 FP8 KV 的差异，仍不能由这些边界测试排除。

### AISBench 渲染及截断重试协议

逐题对照原始 CSV 后，198 题的选项旋转与 gold 映射一致；四轮实际 prompt 也保持一致。
但旧 AISBench safe_format 逐字段替换，误将题号 77、107、146（从 0 起）的 LaTeX
{A}/{B} 当作选项占位符。这三题在四轮均正确且未截断，不能解释已有非截断错题。
保留历史结果；新评测一次渲染，只将完整 prompt 暴露给模板，避免二次替换。
这属于评测协议修正，不能将新分数变化全部归因于 GEMM2 精度修改。

新入口 tests/e2e/models/run_minimax_m31_gpqa.py 在 AISBench Python 环境运行，
同目录加入 PYTHONPATH。参数包括 --dataset-dir、--work-dir（新目录）、
--aisbench-entry（该环境的 ais_bench 脚本）、--host 和 --port。
首轮全量 198 题使用 36,000 输出上限；仅将 finish_reason=length 的原始题号
再批量运行一次，上限 81,920，batch size 32 和采样参数不变。
加载完整数据后才筛选重试，保持原始选项旋转。所有重试均替换对应结果，包含正确变错误，
不选取较好的答案。分别保留首轮和一次重试后的统计、剩余截断数及原始请求响应。
客户端拒绝超过 81,920 的输出上限。该入口不会自动启动或停止推理服务。
纯协议回归 14 passed；实际 AISBench 的 198 题渲染与 gold、3 个请求体、
原题号筛选和超限拒绝均已离线验证。

27.1 实权重第 30 层 expert 0 单专家实验中，固定合成输入下，GEMM2/路由 BF16 输出
对 Torch FP32 参考的平均绝对误差从约 3.21e-3 降为 1.75e-6。
同一 activation 入口的 MXFP8 数据及 E8M0 scale 字节差异为 0。
输入不是实际请求捕获值，此结果不能代替完整 GPQA 准确率。

## Performance

加载时的转换会增加 CPU 计算和临时内存开销。实现逐投影释放配对的 packed 权重与 scale，
并根据本 rank 的实际 expert placement 跳过非本地专家。
本教程不据此给出吞吐或延迟基准。
