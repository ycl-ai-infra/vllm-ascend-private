# M3.1 Flash A3 BF16 / W8A8 QuaRot checkpoint 对比

## 结论与范围

2026-10-24 在 A3 当前部署使用的两套 checkpoint 上复现了
[PR #18 原诊断](minimax_m31_checkpoint_comparison.md)所检查的 norm/router 组合不等价现象。
考虑 QuaRot 旋转后，57 个 MoE 层中 **26 层**的有效矩阵相对 L2 差异超过 10%；
另外 31 层接近等价，最大误差为 `1.5461638724900695e-7`。

**不能单独据此判定 BF16 权重有错，也不能认定差异导致长思考、循环或准确率下降。**
这里的百分比是有效矩阵相对差异，不是准确率差值。
本次在 A3 服务器的 CPU 上只读检查实际 checkpoint，并做 FP32 norm + linear 前向交叉验证；
没有验证 A3 NPU 算子正确性，也没有进行完整模型推理或准确率测评。

原 PR 对比的是另一组 BF16 / MXFP8 checkpoint，得到 35 层；本次是 BF16 / W8A8 QuaRot，
得到 26 层。两者的量化方式、checkpoint 及预期层名单不同。

## 对象与版本

| 项目 | BF16 | W8A8 QuaRot |
| --- | --- | --- |
| 权重路径 | `/mnt/share_space/MiniMax-M3.1-Flash-bf16` | `/mnt/share_space/MiniMax-M3.1-Flash-w8a8-quarot` |
| router 保存 dtype | FP32 | FP32 |
| post-attention norm 保存 dtype | BF16 | FP32 |

两侧 router 形状均为 `[128, 6144]`，norm 为 `[6144]`。
每侧读取 57 个 router、60 个 post-attention norm，共 117 个 tensor；
原始字节与 FP32 SHA256、dtype、shape 均保存在结果文件中。

- 服务源码与 PR 基线：`40ebe9f672babc7b367df1310132ef41489a8784`。
- 诊断起始 PR head：`efeaf7791dc2213afb30a30a62f6f2f8b443ccf1`。
- 执行环境：A3 BF16 部署容器，Python 3.11、`torch==2.10.0+cpu`、`pytest==8.4.2`，CPU 线程数 4。
- 两套权重均可由同一只读挂载读取，没有搬运或改写模型权重。

## 为什么需要对齐 QuaRot 坐标

Gemma RMSNorm 保存的是 `n`，实际乘数为 `gamma = 1 + n`。
没有旋转时可比较 `W * gamma`；QuaRot 改变了输入基，必须先对齐。
实际旋转文件为：

```text
/mnt/share_space/MiniMax-M3.1-Flash-w8a8-quarot/optional/quarot.safetensors
key: global_rotation
shape: [6144, 6144], dtype: FP32
SHA256(FP32): f6e5d5eb170974c148a5b8d64a085f3ad4c41cf1df25ff5001151ff6366ebeed
```

`Q` 包含 192 个 32×32 对角块，块外非零元素数为 0。
检查全部块的最大正交误差为 `5.960464477539063e-8`。
前 64 行 embedding 对齐表明 `x_quarot = x_bf @ Q`：

- `E_quarot` 对 `E_bf @ Q` 的相对 L2 为 `0.0016561821103096008`，与 BF16 存储舍入量级相符。
- 对错误方向 `E_bf @ Q.T` 的差异为 `1.3911653757095337`。

因此本次比较：

```text
B = (W_bf * (1 + n_bf)) @ Q
A = W_quarot * (1 + n_quarot)
relative_L2 = ||A - B||_F / ||B||_F
```

另外以 `A @ Q.T` 对 BF16 有效矩阵交叉计算，57 层指标均在 `1e-6` 内一致。
直接将未经对齐的 raw tensor 或组合差异作为异常，会混入正常的 QuaRot 旋转影响。

对 `W_quarot` 与 `B` 做逐列最小二乘缩放拟合，全部层最大残差只有
`1.4463685715782049e-7`，但 26 层的缩放与实际 norm 没有抵消。
诊断中的逆缩放仅用于系数检查，未修改任何落盘或运行时 tensor。

## 实测结果

差异超过 10% 的 26 层，层号从 0 开始：

```text
3, 4, 5, 6, 8, 11, 12, 13, 14, 22, 25, 27, 31, 32, 33, 36, 37, 40,
41, 45, 48, 50, 55, 56, 57, 59
```

| 层号 | 对齐后的有效矩阵相对 L2 | 64 组输入的 CPU 前向相对 L2 |
| --- | --- | --- |
| 3 | 19.099922% | 19.033943% |
| 4 | 31.959203% | 32.095045% |
| 25（最大） | 41.889411% | 42.147321% |
| 30（接近等价样例） | `1.510069012e-7` | `1.964992407e-6` |
| 56 | 24.536476% | 24.690248% |
| 59 | 19.110875% | 18.999599% |

前向探针使用 `seed=20261024`、64 组随机 FP32 输入、`eps=1e-6`。
先前独立脚本以 `normalize(x) @ Q` 计算旋转后的归一化输入；
新增 UT 还分别计算 `normalize(x)` 与 `normalize(x @ Q)`，确认相同结论。
随机探针验证同一输入下组合行为，不模拟完整模型的隐藏状态分布。

## UT 与复现方式

新增 UT：`tests/ut/models/minimax_m3/test_checkpoint_quarot_router_norm_equivalence.py`。
它复用原文件的 safetensors 索引和相对 L2 工具，不导入或调用推理框架/NPU 算子。

用例包括：

1. 无权重的非对称正交旋转、router 缩放和 norm 逆缩放等价性检查。
2. 列拟合保持精确但 norm 不抵消的 20% 反例，及完整输入基上的独立前向核查。
3. 实际 A3 权重的旋转方向、块正交性、dtype、57 层组合及 64 组 CPU 前向检查。

第 3 项断言本次特定文件的 26 层名单，用于复现已知不等价现象；
**测试通过不表示权重等价或模型准确率合格**。
缺少指定 checkpoint 时该项 skip，两个小矩阵用例仍可运行。

在挂载两套权重的容器内，将两个测试文件原样复制到独立诊断目录，以隔离仓库级
`conftest.py` 对 NPU/mock 的初始化，然后执行：

```bash
cd /mnt/share/y00976694/M3.1-Flash/standalone-20261024-1608/pr18-audit
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -sv \
  test_checkpoint_router_norm_equivalence.py \
  test_checkpoint_quarot_router_norm_equivalence.py \
  --confcutdir=. -o junit_family=xunit1 --junitxml=a3-pr18-ut.xml
```

结果为 **6 passed, 1 skipped**。
跳过的是原 PR 需要另一组 BF16/MXFP8 文件的 reference 用例；
本次 A3 真实 checkpoint 用例实际执行并通过。
没有运行大规模单元测试；没有改动部署服务、模型权重或转换算法。

新增文件及本次修改的检查通过：Ruff check/format、codespell、typos、markdownlint、
Python package-init、boolean context manager 和 long-functions 检查。
最后一项 Windows 钩子的 `python3` 启动器不可用，改用显式 Python 运行同一检查脚本后通过。
已尝试 `bash format.sh ci`，但在安装 actionlint 钩子环境时发生
`RemoteDisconnected: Remote end closed connection without response`；不将完整仓库 lint 标记为通过。

## 导出证据

所有相对路径以本页所在目录为基准：

- [原始 tensor 清单、SHA256 和未对齐/反旋转指标](data/minimax_m31_a3_quarot_20261024/checkpoint-comparison.json)。
- [对齐后的 57 层指标及独立前向结果](data/minimax_m31_a3_quarot_20261024/rotation-scaling-validation.json)。
- [57 层 CSV](data/minimax_m31_a3_quarot_20261024/layers.csv)。
- [机器可读汇总](data/minimax_m31_a3_quarot_20261024/summary.json)。
- [新增 UT 的实际运行日志](data/minimax_m31_a3_quarot_20261024/a3-pr18-ut.log)。
- [JUnit XML](data/minimax_m31_a3_quarot_20261024/a3-pr18-ut.xml)。
- [修改文件检查日志](data/minimax_m31_a3_quarot_20261024/changed-files-lint.log)。
- [完整 lint 的环境安装失败日志](data/minimax_m31_a3_quarot_20261024/format-ci.log)。

服务器上原始审计脚本、日志和全部结果保留在：

```text
/mnt/share/y00976694/M3.1-Flash/standalone-20261024-1608/pr18-audit
```

这些证据确认两套 checkpoint 的组合不等价，不定位具体转换步骤，也未与原始权威
checkpoint 逐项校验 BF16 来源，因此不将错误归因于其中任何一侧。
