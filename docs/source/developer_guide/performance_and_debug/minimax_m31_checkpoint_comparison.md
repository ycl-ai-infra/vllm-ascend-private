# M3.1 Flash checkpoint 的 norm/router 对比复现

另见 [A3 BF16 / W8A8 QuaRot 检查](minimax_m31_a3_quarot_checkpoint_comparison.md)：
对另一组实际部署权重，在对齐 QuaRot 旋转后复现了 26 层组合不等价，并附专项 UT 与逐层数据。
本页以下的 35 层结果仍仅对应本页指定的 BF16 / MXFP8 checkpoint。

## 结论及范围

本诊断确认两份指定 checkpoint 在部分层的 **norm 与 router 组合上不等价**。
57 个 MoE 层中，35 层的有效矩阵相对 L2 差异超过 10%。
这不是“35 层原始 tensor 有变化”的计数：正常的 SmoothQuant 也会改变原始 tensor；
本诊断检查这些变化在组合后是否抵消。

**尚未确认这些差异就是 GPQA 精度下降的原因，也没有确认具体转换步骤存在错误。**
没有量化转换命令或日志，因此不将“漏处理”“重复平滑”“输入版本不同”等假设写成结论。
此 PR 增加诊断 UT 和证据说明，不修改推理算子、运行时权重或转换算法。

## 对比对象

| 项目 | BF16 checkpoint | MXFP8 checkpoint |
| --- | --- | --- |
| 路径 | `/mnt/m31-bf16-gpqa-20261024` | `/mnt/weight/minimax_m3dot1_w8a8_2` |
| router 保存 dtype | FP32 | FP32 |
| post-attention norm 保存 dtype | BF16 | FP32 |
| 专家权重处理 | 从 NVFP4 展开为 BF16 | 经 SmoothQuant / MXFP8 处理 |

以下两组 tensor 是本 UT 的全部对比范围，`L` 是 **从 0 开始**的模型层号：

```text
language_model.model.layers.L.post_attention_layernorm.weight
language_model.model.layers.L.block_sparse_moe.gate.weight
```

前者形状为 `[6144]`，后者形状为 `[128, 6144]`。
只读取这两类 tensor；不会将 788 GiB / 418 GiB 的完整专家权重载入内存。
router 在两份文件中均为 FP32，所以这里不存在 FP8 解量化尺度计算误差。
本 UT 不审计所有专家矩阵，也不将 raw norm 的 BF16/FP32 dtype 差异本身判作错误。

辅助来源核查发现，26.1 历史浮点目录、27.1 浮点目录中的 57 个 router 和 60 个
post-attention norm（共 117 个 tensor）逐项 SHA256 一致。
27.1 浮点目录的这 117 个 tensor 与参考目录
`/home/x00939195/weight/minimax_m3dot1_flash` 也一致。
后者仅用作原始 tensor 参考，没有替换本次部署使用的 MXFP8 权重。

## 为什么不能只比较原始权重

直白地说，SmoothQuant 可以“前面乘 2、后面除以 2”，让结果保持不变。
因此，原始矩阵变了不等于模型计算变了；需要把前后的缩放合在一起比较。

MiniMax 使用 Gemma RMSNorm，文件保存的是 `n`，实际乘数是 `1 + n`。
对同一份归一化前输入 `x`：

```text
u = x / sqrt(mean(x**2) + eps)
router_logits = (u * (1 + n)) @ W.T
W_eff = W * (1 + n)[None, :]
```

`W` 的行是专家，列是输入通道。相同输入下的等价性要求：

```text
W_quant * (1 + n_quant) == W_bf16 * (1 + n_bf16)
```

UT 先将这四个 tensor 转为 CPU FP32，再计算：

```text
relative_L2 = ||W_quant_eff - W_bf16_eff||_F / ||W_bf16_eff||_F
```

这里的百分比是有效权重的相对 Frobenius/L2 差异，**不是准确率差值**。
分母始终是 BF16 checkpoint 的有效矩阵。

另外，对 router 的每一列拟合一个最小二乘系数：

```text
s[j] = sum(W_bf16[:, j] * W_quant[:, j]) / sum(W_bf16[:, j]**2)
column_fit_relative_L2 = ||W_quant - W_bf16 * s||_F / ||W_bf16 * s||_F
```

57 层的列拟合残差均小于 `1e-6`，说明 router 的变化可由逐列缩放解释。
但只有同时满足 `s * (1 + n_quant) == 1 + n_bf16`，组合才保持等价。
因此“router 列缩放拟合很好”不能替代 norm/router 组合检查。

## 哪些层、哪些权重不同

下表列出有效矩阵差异超过 10% 的 **35 层**。
`router 最大绝对差` 和 `norm 最大绝对差` 是对应两个原始 tensor 转为 FP32 后的
`max(abs(quant - bf16))`；最后一列才是上述折入 norm 的组合指标。
34 层的这两个原始 tensor 都有数值变化；第 56 层 router 逐元素相同，仅 norm 改变。

| 层号 L | router 最大绝对差 | norm 最大绝对差 | 有效矩阵相对 L2 |
| --- | --- | --- | --- |
| 3 | 0.47793654 | 2.351054 | 15.876633% |
| 4 | 0.87731451 | 1.0428982 | 26.027855% |
| 5 | 0.74294233 | 1.5130651 | 20.450512% |
| 6 | 1.0281948 | 0.71567428 | 22.064830% |
| 15 | 0.23755185 | 0.64939988 | 34.680608% |
| 16 | 0.30220303 | 0.52070367 | 30.831930% |
| 17 | 0.38446173 | 0.48155582 | 19.241329% |
| 19 | 0.23778492 | 1.420979 | 63.619906% |
| 20 | 0.12418525 | 0.49058616 | 66.138005% |
| 21 | 0.76424003 | 0.47349155 | 32.642028% |
| 22 | 1.0557144 | 0.49438417 | 28.494632% |
| 24 | 0.25278878 | 0.41799617 | 19.524363% |
| 25 | 0.35070273 | 0.49262249 | 19.031610% |
| 29 | 0.72457379 | 0.5963186 | 12.638803% |
| 30 | 0.81861001 | 0.6428318 | 20.512274% |
| 31 | 0.23617803 | 0.66040534 | 15.676162% |
| 32 | 0.57015204 | 0.66389269 | 38.270250% |
| 37 | 0.43317831 | 0.76804417 | 23.428474% |
| 40 | 0.2549938 | 0.71792656 | 26.702932% |
| 41 | 0.29452223 | 0.8363446 | 15.388599% |
| 42 | 0.51494944 | 0.81610548 | 31.005061% |
| 43 | 0.31084126 | 0.88235837 | 26.865706% |
| 44 | 0.4881801 | 0.89406061 | 12.359379% |
| 45 | 0.43392196 | 0.84766787 | 23.666632% |
| 46 | 0.20441154 | 0.86388439 | 26.082888% |
| 48 | 0.34478411 | 0.89352417 | 55.324137% |
| 49 | 0.80046088 | 0.91753018 | 17.195979% |
| 50 | 0.15594132 | 0.92624438 | 26.436511% |
| 51 | 1.7919075 | 0.93611848 | 25.691846% |
| 52 | 1.0889347 | 0.91853476 | 25.346851% |
| 54 | 0.66115367 | 0.92251849 | 52.052969% |
| 55 | 0.5498997 | 0.99106586 | 17.664517% |
| 56 | 0 | 4.029304 | 41.771179% |
| 57 | 0.32743359 | 1.0652287 | 30.471915% |
| 58 | 0.41645849 | 1.7648461 | 18.126808% |

例如第 3 层的具体 key 是：

```text
language_model.model.layers.3.post_attention_layernorm.weight
language_model.model.layers.3.block_sparse_moe.gate.weight
```

第 3、30 层的组合差异分别为 `0.15876633`、`0.20512274`。
第 56 层两份 router 完全相同，但 norm 改变后组合差异达到 `0.41771179`；
这个例子不依赖 FP8 解量化或 NPU 算子即可复现。

另外 22 层的组合接近等价（误差小于 `1e-6`）：

```text
7, 8, 9, 10, 11, 12, 13, 14, 18, 23, 26, 27, 28, 33, 34, 35, 36, 38, 39, 47, 53, 59
```

“接近等价”指组合，不表示各自的原始 norm/router tensor 完全一样。
例如第 59 层原始 tensor 经缩放后，组合相对 L2 约为 `6.17e-8`。

全部 57 层的原始数值指标见
[机器可读结果](data/minimax_m31_norm_router_comparison_20261024.json)。

## 如何运行 UT

文件：`tests/ut/models/minimax_m3/test_checkpoint_router_norm_equivalence.py`。
依赖 `pytest`、`torch` 和 `safetensors`，计算全部在 CPU 上进行。

```bash
pytest -sv tests/ut/models/minimax_m3/test_checkpoint_router_norm_equivalence.py
```

测试包含：

1. 小矩阵上验证反向调整 norm 后组合保持等价，检查 Gemma 的 `+1` 语义。
2. 两个小矩阵反例，分别复现 16% / 20% 的组合误差，同时保持 router 列拟合残差为零；
   另通过完整输入基上的实际 norm + linear 计算交叉验证指标。
3. 读取指定 checkpoint 的 57 层，打印每层 raw tensor 差异、列拟合残差和组合相对 L2；
   断言同样的 35 层名单，并检查第 3、30、59 层的数值。

在 27.1 容器内将同一测试文件单独放入诊断目录运行，以隔离仓库级 conftest 的
NPU/mock 初始化，结果为 **4 passed（约 3 秒）**：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /work/.venv/bin/python -m pytest -sv \
  /work/m31-bf16-quant-20261024/unit-tests/test_checkpoint_router_norm_equivalence.py \
  --confcutdir=/work/m31-bf16-quant-20261024/unit-tests \
  -o junit_family=xunit1 \
  --junitxml=/work/m31-bf16-quant-20261024/router-norm-ut.xml
```

未挂载两份指定 checkpoint 时，实际文件测试会 skip，三个小矩阵用例仍可运行。
35 层名单是这两份特定文件的复现基线，不是对任意模型量化结果的通用要求。

## 与 10 道差异题的关系

分析的 GPQA 零基索引为 `7, 48, 49, 68, 75, 88, 92, 93, 98, 123`。
历史浮点输出中 6 道因重复循环取消，4 道正常结束但选错；对应历史 PR16 量化记录均答对。
两组历史记录的选项排列全部不同，temperature、top_p、reasoning_effort 和部署方式也不同。
这些记录用于选择诊断题，不构成同条件精度对照。

在 27.1 上使用相同 PR16 代码、TP4/DP2/EP8、V2 runner、FULL_DECODE_ONLY 和 FP8 KV cache：

- 为两套权重固定相同题面、服务渲染的 prompt token IDs，以及历史浮点的前 512 个推理 token。
- 通过 completion API 获取逐位置 top-20 **logprobs（归一化 logits）**，不是完整原始 logits。
- 10 题合计 5120 个位置中，有 727 个位置的首选 token 不同，约 `14.20%`。
- 每套模型分别重复采集，所比较的 logprobs 完全一致。
- 将量化侧 `weight_nz_mode` 配置改为 0，结果也未改变；此项只统一配置，不能证明内部布局相同。
- 10 个相同前缀的 decode 探针中，每套模型自身的 prefill/decode 首选 token 都保持一致，
  但概率分布并非逐位相同；没有做 eager-decode 对照，不能由此全面排除图模式问题。

浮点部署因内存容量需要 CPU prefetch offload，量化部署不需要，因此运行路径并非完全相同。
相关推理实验使用 Ascend PR16 commit
`271e80bc28a96997906046d90af3467d3cc7e772`，vLLM core commit
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
CPU checkpoint UT 不依赖这些推理服务或它们的模型实现。

还观察到，仅将量化重放的 temperature 从 0 改为 0.6，第 7 题从循环变为答对。
这说明生成条件会影响表现；单次对照不能证明所有差异题具有同一个原因。

## 仍需完成的因果验证

当前已证明“文件中的组合系数不同”和“相同前缀的输出分布不同”，
没有证明“这 35 层的缩放差异导致 GPQA 准确率下降”。
确认因果需要保持模型其它权重、题面、采样参数、部署和运行条件一致，
只替换经过验证的缩放系数，检查输出分布与差异题是否恢复，并做反向注入对照。
同一个 norm 还供给专家和 shared-expert 分支，不能只凭 router 拟合系数直接修改 norm。
在转换来源和各分支缩放对应关系明确之前，本诊断不提供未经验证的权重修复。
