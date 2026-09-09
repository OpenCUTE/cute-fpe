# FP32 tiny-subnormal 精度 mismatch 技术报告

本文整理截至目前围绕 `mymac`、tested Blackwell-class GPU `nvmac`、MMA-Sim/论文模型在 FP32 tiny-subnormal 边界上的 mismatch 定位工作。主要问题仍集中在 MXF8F6F4，但 2026-04-30 的 128b 10M 级回归又发现 BF16 也能在同类 tiny-subnormal 边界触发 1 ulp mismatch。重点区分三类结论：

- 已由全量回归或 directed 实验观察到的事实。
- 已尝试但被反例否定的匹配模型。
- 当前仍然合理、但尚未形成可合入主线规则的硬件解释。

当前默认 `mymac` 仍保持数学 exact RZ 模型，不合入任何为匹配 tested Blackwell-class GPU 而写的复杂拟合逻辑。

## 1. 当前结论

1. 当前默认 `mymac` 对 MXF8F6F4 使用 `decodeadd(..., decimal_num=25)`，FP32 输出按 RZ 打包；`C=0` 不强制以 FP32 zero exponent `-126` 参与 `exp_max`。
2. 2026-04-27 的 128b/256b 各 100 万向量全量回归中，BF16、TF32、非 MX F8F6F4、MXFP4/NVFP4、integer 格式均未发现 mismatch；剩余 mismatch 集中在 MXF8F6F4。这个结论只覆盖 1M 随机预算。
3. 2026-04-30 的 128b 10M 级测试中，BF16 在 vec 2,694,985 触发 mismatch，日志汇总为 `mismatch_count=3`；已知 first case 为 `my=0x80000fb9, nv=0x80000fba`，且可压缩成 2 个非零 BF16 product 的稳定复现。因此 BF16 不是绝对免疫，只是随机命中概率明显低于 MXF8F6F4。
4. 同一轮 128b 10M 回归中，TF32、FP16、非 MX F8F6F4、MXFP4/NVFP4、integer dtype 均 pass；当前仍未构造出 TF32 mismatch。10,000 个不同 TF32 subnormal 输出候选和手工仿 BF16 signed-sum 形态构造均 match nvmac，但这只能说明当前构造族未命中，不能证明 TF32 永远不会失败。
5. 128b 10M 回归把随机可见的 MXF8F6F4 失败 dtype 从 11 个扩大到 16 个；新增随机暴露的 dtype 为 `57, 59, 60, 67, 72`，仍然 10M pass 的 MX dtype 为 `69, 70, 71, 74, 75, 76, 79, 80, 81`。
6. 全量随机回归中的已知 MXF8F6F4 first mismatch 全部位于 FP32 zero/subnormal 区间，1M 日志中 first mismatch 都满足 `abs(nv) = abs(my) + 1` 个 FP32 subnormal ulp；10M 当前日志只保留 first vec/count，未保留每个 first mismatch 的 my/nv bit pattern。
7. MXF8F6F4 后续 directed 实验证明，mismatch 方向并不总是 `+1`；也存在 `abs(nv) = abs(my) - 1`，以及 `my=±1 subnormal, nv=0` 的情况。
8. 对 MXF8F6F4，单 product 被 scale 直接搬到 FP32 subnormal 区间时没有触发 mismatch；简单的 `+P - P + residual` 三 product 构造也没有触发 mismatch。BF16 反例显示：非 MX 路径也能触发，但 BF16 的稳定复现条件是 dominant product 加一个极小 aligned residual。
9. MMA-Sim 可以帮助确认 FDA/F=25/RZ/scale 的主结构，但不能作为当前 tested Blackwell-class GPU 的最终 oracle；它对部分 MXF8F6F4 组合支持不完整，zero-C exponent 规则也有已知反例。
10. 曾经构造过能拟合部分样本的硬件边界模型，但条件复杂、false positive 多、RTL 动机不足，已经不作为主线方向。

## 2. 全量回归结果

环境：

- GPU 架构: NVIDIA Blackwell（具体服务器型号省略）
- Driver/CUDA: 使用兼容版本（具体服务器软件版本省略）
- 工作目录: `${C_MODEL_REPO}`
- 回归日期: 已省略

命令：

```bash
make test_nv_fp32_128b VEC_COUNT=1000000 TEST_NV_DUMP=0 TEST_NV_FP32_128B_JOBS=27
make test_nv_fp32_256b VEC_COUNT=1000000 TEST_NV_DUMP=0 TEST_NV_FP32_256B_JOBS=27
```

总体结果：

| 模式 | dtype 数 | 总向量数 | 失败 dtype | 总 mismatch | mismatch 所在类别 |
|---|---:|---:|---:|---:|---|
| 128b | 63 | 63,000,000 | 11 | 36 | MXF8F6F4 |
| 256b | 63 | 63,000,000 | 9 | 32 | MXF8F6F4 |

MXF8F6F4 子集：

| 模式 | MX dtype 数 | MX 向量数 | 失败 MX dtype | MX mismatch | MX mismatch 率 |
|---|---:|---:|---:|---:|---:|
| 128b | 25 | 25,000,000 | 11 | 36 | 1.44e-6 |
| 256b | 25 | 25,000,000 | 9 | 32 | 1.28e-6 |

2026-04-27 1M 回归中未发现 mismatch 的类别：

- BF16: 128b/256b 均 1M pass
- TF32: 128b/256b 均 1M pass
- 非 MX F8F6F4 dtype 32-56: 128b/256b 均 pass
- MXFP4/NVFP4 dtype 9/10: 128b/256b 均 pass
- 当前默认 integer dtype: 128b/256b 均 pass

2026-04-30 128b 10M 回归更新：

日志目录：`fmac_test_nv_fp32_128b_logs`

| 模式 | dtype 数 | 总向量数 | 失败 dtype | 总 mismatch | mismatch 所在类别 |
|---|---:|---:|---:|---:|---|
| 128b | 63 | 630,000,000 | 17 | 452 | BF16 + MXF8F6F4 |

分类统计：

| 类别 | 覆盖 dtype | 失败 dtype | mismatch | 结论 |
|---|---:|---:|---:|---|
| BF16 | 1 | 1 | 3 | 10M 随机可触发，但概率低 |
| TF32 | 1 | 0 | 0 | 10M pass |
| FP16 | 1 | 0 | 0 | 10M pass |
| 非 MX F8F6F4 dtype 32-56 | 25 | 0 | 0 | 10M pass |
| MXF8F6F4 dtype 57-81 | 25 | 16 | 449 | 主要 mismatch 来源 |
| MXFP4/NVFP4 dtype 9/10 | 2 | 0 | 0 | 10M pass |
| integer dtype | 8 | 0 | 0 | 10M pass |

MXF8F6F4 10M 子集：

| MX dtype 数 | MX 向量数 | 失败 MX dtype | MX mismatch | MX mismatch 率 |
|---:|---:|---:|---:|---:|
| 25 | 250,000,000 | 16 | 449 | 1.796e-6 |

10M 回归中仍未发现 mismatch 的 MX dtype：

`69, 70, 71, 74, 75, 76, 79, 80, 81`

## 3. 全量回归 mismatch 明细

`delta magnitude = abs(nv_bits) - abs(my_bits)`，按 FP32 bit pattern 的 magnitude 计算。

### 3.1 128b 1M

| dtype | 格式 | first vec | mismatch_count | my | nv | delta magnitude | 结果类别 |
|---:|---|---:|---:|---:|---:|---:|---|
| 58 | MXF8F6F4_e4m3_X_e5m2 | 394630 | 3 | 0x80000005 | 0x80000006 | +1 | negative subnormal |
| 61 | MXF8F6F4_e4m3_X_e2m1 | 570998 | 1 | 0x80000004 | 0x80000005 | +1 | negative subnormal |
| 62 | MXF8F6F4_e5m2_X_e4m3 | 868445 | 2 | 0x00000001 | 0x00000002 | +1 | positive subnormal |
| 63 | MXF8F6F4_e5m2_X_e5m2 | 13555 | 5 | 0x8000007f | 0x80000080 | +1 | negative subnormal |
| 64 | MXF8F6F4_e5m2_X_e2m3 | 225897 | 5 | 0x00000037 | 0x00000038 | +1 | positive subnormal |
| 65 | MXF8F6F4_e5m2_X_e3m2 | 225897 | 6 | 0x0000000d | 0x0000000e | +1 | positive subnormal |
| 66 | MXF8F6F4_e5m2_X_e2m1 | 225897 | 6 | 0x0000006f | 0x00000070 | +1 | positive subnormal |
| 68 | MXF8F6F4_e2m3_X_e5m2 | 394630 | 1 | 0x8000017f | 0x80000180 | +1 | negative subnormal |
| 73 | MXF8F6F4_e3m2_X_e5m2 | 313657 | 4 | 0x8000003f | 0x80000040 | +1 | negative subnormal |
| 77 | MXF8F6F4_e2m1_X_e4m3 | 927702 | 1 | 0x80000008 | 0x80000009 | +1 | negative subnormal |
| 78 | MXF8F6F4_e2m1_X_e5m2 | 372190 | 2 | 0x00000000 | 0x80000001 | +1 | zero to negative subnormal |

128b 1M 全量随机中未失败的 MX dtype：

`57, 59, 60, 67, 69, 70, 71, 72, 74, 75, 76, 79, 80, 81`

### 3.2 256b 1M

| dtype | 格式 | first vec | mismatch_count | my | nv | delta magnitude | 结果类别 |
|---:|---|---:|---:|---:|---:|---:|---|
| 58 | MXF8F6F4_e4m3_X_e5m2 | 852109 | 1 | 0x00000300 | 0x00000301 | +1 | positive subnormal |
| 62 | MXF8F6F4_e5m2_X_e4m3 | 568544 | 1 | 0x0000000d | 0x0000000e | +1 | positive subnormal |
| 63 | MXF8F6F4_e5m2_X_e5m2 | 610926 | 1 | 0x0000001b | 0x0000001c | +1 | positive subnormal |
| 64 | MXF8F6F4_e5m2_X_e2m3 | 390865 | 5 | 0x8000001d | 0x8000001e | +1 | negative subnormal |
| 65 | MXF8F6F4_e5m2_X_e3m2 | 342934 | 6 | 0x00000017 | 0x00000018 | +1 | positive subnormal |
| 66 | MXF8F6F4_e5m2_X_e2m1 | 340718 | 6 | 0x00000014 | 0x00000015 | +1 | positive subnormal |
| 68 | MXF8F6F4_e2m3_X_e5m2 | 83341 | 5 | 0x00000007 | 0x00000008 | +1 | positive subnormal |
| 73 | MXF8F6F4_e3m2_X_e5m2 | 83341 | 2 | 0x00000001 | 0x00000002 | +1 | positive subnormal |
| 78 | MXF8F6F4_e2m1_X_e5m2 | 83341 | 5 | 0x0000000a | 0x0000000b | +1 | positive subnormal |

256b 全量随机中未失败的 MX dtype：

`57, 59, 60, 61, 67, 69, 70, 71, 72, 74, 75, 76, 77, 79, 80, 81`

### 3.3 128b 10M

这轮日志只输出每个 dtype 的 first mismatch vec 和 mismatch_count，没有逐项打印 `my/nv` bit pattern。因此下表只记录覆盖范围和数量；BF16 first case 的 `my/nv` 来自同一 vec 的后续复现与最小化实验。

| dtype | 格式 | first vec | mismatch_count | 备注 |
|---:|---|---:|---:|---|
| 2 | BF16 | 2694985 | 3 | first case 已复现为 `0x80000fb9 -> 0x80000fba` |
| 57 | MXF8F6F4_e4m3_X_e4m3 | 1451552 | 6 | 1M 未暴露，10M 新增 |
| 58 | MXF8F6F4_e4m3_X_e5m2 | 394630 | 34 | 1M 已暴露 |
| 59 | MXF8F6F4_e4m3_X_e2m3 | 5953803 | 1 | 1M 未暴露，10M 新增 |
| 60 | MXF8F6F4_e4m3_X_e3m2 | 3219883 | 7 | 1M 未暴露，10M 新增 |
| 61 | MXF8F6F4_e4m3_X_e2m1 | 570998 | 6 | 1M 已暴露 |
| 62 | MXF8F6F4_e5m2_X_e4m3 | 868445 | 33 | 1M 已暴露 |
| 63 | MXF8F6F4_e5m2_X_e5m2 | 13555 | 55 | 1M 已暴露 |
| 64 | MXF8F6F4_e5m2_X_e2m3 | 225897 | 45 | 1M 已暴露 |
| 65 | MXF8F6F4_e5m2_X_e3m2 | 225897 | 50 | 1M 已暴露 |
| 66 | MXF8F6F4_e5m2_X_e2m1 | 225897 | 57 | 1M 已暴露 |
| 67 | MXF8F6F4_e2m3_X_e4m3 | 5746871 | 1 | 1M 未暴露，10M 新增 |
| 68 | MXF8F6F4_e2m3_X_e5m2 | 394630 | 44 | 1M 已暴露 |
| 72 | MXF8F6F4_e3m2_X_e4m3 | 1842055 | 3 | 1M 未暴露，10M 新增 |
| 73 | MXF8F6F4_e3m2_X_e5m2 | 313657 | 46 | 1M 已暴露 |
| 77 | MXF8F6F4_e2m1_X_e4m3 | 927702 | 4 | 1M 已暴露 |
| 78 | MXF8F6F4_e2m1_X_e5m2 | 372190 | 57 | 1M 已暴露 |

128b 10M 回归中 pass 的 dtype：

- 浮点/scale 非 MX：`1, 3, 9, 10, 32-56`
- MXF8F6F4：`69, 70, 71, 74, 75, 76, 79, 80, 81`
- integer：`97, 99, 101, 103, 129, 131, 133, 135`

这轮结果把一个重要边界讲清楚：非 MX 路径并非天然不会出现 tiny-subnormal 1 ulp 差异，BF16 已经出现 3 个随机 mismatch；但 mismatch 的主要密度仍在 MXF8F6F4，尤其是包含 e5m2 的组合。`e2m3/e2m3`、`e2m3/e3m2`、`e2m3/e2m1`、`e3m2/e2m3`、`e3m2/e3m2`、`e3m2/e2m1`、`e2m1/e2m3`、`e2m1/e3m2`、`e2m1/e2m1` 在 10M 随机下仍未暴露，说明低指数/低尾数组合不是充分条件；是否触发还依赖 product 分布、signed reduction 后的 residual 形态和 final subnormal pack 边界。

## 4. mymac 当前模型

MXF8F6F4 当前主线路径：

1. 根据 dtype descriptor 解码 A/B 源格式。
2. scale 为 UE8M0；scale 非 NaN 时只修改 operand exponent：`exp += scale - 127`。
3. `decodemul()` 计算每个 product 的 sign、exponent、mantissa。
4. product mantissa 根据源格式有效尾数位宽左移到统一内部精度。
5. `C` 以 FP32 decode，mantissa 左移 2 bit，使其与 `decimal_num=25` 的 FDA 路径对齐。
6. `decodeadd()` 选择非零项的最大 exponent 作为 `exp_max`；当前 `C=0` 不参与 `exp_max`。
7. 每个 product 按 `rsft = exp_max - exp_i` 右移对阶，右移丢弃位按 RZ 截断。
8. signed aligned product 相加，得到有符号尾数和。
9. 对 signed sum 取符号和绝对值，规格化，并按 FP32 RZ 输出。

因此当前 `mymac` 是一个清晰的 exact RZ/FDA 模型，不包含 tested Blackwell-class GPU tiny-subnormal compatibility patch。

## 5. MMA-Sim 与论文模型

论文 `MMA-Sim: Bit-Accurate Reference Model of Tensor Cores and Matrix Cores` 的主要价值在于黑盒推断了 Tensor Core/MMA 的 accumulation precision、rounding、special value、summation order 等主结构。和当前问题相关的结论包括：

- Blackwell 上多类 HMMA/QMMA 路径可用 fixed-point dot-add 模型描述。
- MXF8F6F4 类路径使用约 25 个 fractional bits。
- FP32 输出主要是 RZ。
- MX scale 通过修改 product exponent 参与计算。

但本地实验不能把 MMA-Sim 当 tested Blackwell-class GPU 的最终 oracle：

- 论文验证硬件包含 `NVIDIA RTX PRO 6000 Blackwell`，不是当前 `tested Blackwell-class GPU`。
- 本地 MMA-Sim 对 e3m2/e2m3/e2m1 组合支持不完整。
- MMA-Sim 中 `C=0` 会以 FP32 zero exponent `-126` 参与某些 `max_exponent` 选择；这个规则能解释部分 case，但会引入已知反例。
- dtype58/62/63 部分 case 中 MMA-Sim match `nvmac`；但 zero-C 实验中也存在 MMA-Sim match `mymac` 而不 match `nvmac` 的反例。

结论：MMA-Sim 可用于确认 FDA/F=25/RZ/scale 主结构，但不能单独决定 tested Blackwell-class GPU MXF8F6F4 的 tiny-subnormal 边界行为。

## 6. 已尝试并否定的匹配模型

### 6.1 zero-C exponent 规则

假设：当 `C=0` 时，仍让 C 以 FP32 zero exponent `-126` 参与 `exp_max`。

结果：

- 能修复部分 case。
- 也会引入反例，例如已知 zero-C 实验里 MMA-Sim/mymac 与 nvmac 分裂。
- 不能作为默认规则。

### 6.2 subnormal RNE 或 final pack RNE

假设：FP32 subnormal 输出阶段不是 RZ，而是 RNE 或类似 RNE。

结果：

- 能解释部分 `+1ULP`。
- 但大量 match RZ 的 directed case 会被错误改动。
- BF16/TF32 tiny-subnormal 对照也不支持全局 FP32 pack 规则改变。

### 6.3 单 product tiny-subnormal

构造：只有一个非零 `a_i*b_i`，其它 lane 全 0，用 scale 把单 product 搬到 FP32 zero/subnormal 区间。

结果：

| 实验 | 覆盖 dtype | CPU 候选 | 硬件候选 | mismatch |
|---|---:|---:|---:|---:|
| 单 product，含 zero/subnormal 候选 | 57-81 | 13,285,800 | 50,000 | 0 |
| 单 product，仅非零 subnormal，按 magnitude bucket 分层 | 57-81 | 6,680,232 | 25,000 | 0 |

结论：单 product 直接进入 subnormal 不足以触发 mismatch。

### 6.4 三 product cancellation

构造：只保留 3 个非零 product，其中两个形成 `+P - P`，第三个作为 residual，其余 lane 全 0。

结果：

| 实验 | 覆盖 dtype | CPU 候选 | 硬件候选 | mismatch |
|---|---:|---:|---:|---:|
| 三 product，全局预算 | 57-81 前序候选 | 15,865 | 3,000 | 0 |
| 三 product，每 dtype 500 个硬件候选 | 57-81 | 100,449 | 12,500 | 0 |

结论：简单 `+P - P + residual` 也不足以触发当前已知 mismatch。真实触发条件需要更复杂的多项 signed reduction 结构。

### 6.5 BF16/TF32 tiny-product 对照

早期构造：对 BF16/TF32 生成 `a*b ~= 2^-160` 附近和 K 项 tiny accumulation case，覆盖 128b/256b，`C=±0`。

结果：

| topology | BF16 候选 | BF16 mismatch | TF32 候选 | TF32 mismatch |
|---|---:|---:|---:|---:|
| 256b | 5814 | 0 | 3078 | 0 |
| 128b | 3078 | 0 | 1710 | 0 |

这轮早期实验说明：最终数值落入 FP32 tiny-subnormal 本身不是充分条件。

2026-04-30 又根据 10M 日志中的 BF16 mismatch 做了 targeted 复现和 directed search：

| 实验 | 覆盖 | 硬件候选 | mismatch | 结论 |
|---|---|---:|---:|---|
| BF16 128b 10M random | `vec[0, 10000000)` | 10,000,000 | 3 | 随机可触发，但概率约 `3e-7` |
| BF16 legacy vec replay | `base_seed=113, vec=2694985` | 1 | 1 | 复现日志中的 `0x80000fb9 -> 0x80000fba` |
| BF16 最小化 case | 2 个非零 product | 1 | 1 | 可稳定触发 |
| BF16 random unique subnormal | 10,000 个不同 `mymac` subnormal 输出 | 10,000 | 0 | 随机 subnormal 不足以稳定触发 |
| BF16 paired cancel, `C=0` | 5,000 个候选 | 5,000 | 0 | 简单成对抵消不充分 |
| TF32 128b 10M random | `vec[0, 10000000)` | 10,000,000 | 0 | 10M 随机仍未触发 |
| TF32 random unique subnormal | 10,000 个不同 `mymac` subnormal 输出 | 10,000 | 0 | 当前未触发 |
| TF32 hand BF16-like shape | 手工仿 BF16 signed-sum 形态 | 1 | 0 | 当前未触发 |

修正后的结论：

- BF16 可以触发同类 1 ulp tiny-subnormal mismatch，但触发条件比“任意 tiny output”更窄。
- 已知 BF16 稳定复现只需要 2 个非零 product：一个 dominant negative product，以及一个对阶后只剩 `+8` 内部单位的极小 positive residual。
- TF32 目前没有复现；原因可能是 TF32 的 mantissa/product 精度更高，已有构造没有把 residue 放到硬件敏感边界上。不能据此证明 TF32 永远不会 mismatch。

### 6.6 near-carry/sticky 窗口

假设：subnormal pack 时，右移被丢弃的某些低位 OR 成 sticky，再作为 final mantissa `+1` 的 carry-in。

观察：

- 对已有 JSON mismatch corpus，`mantissa[14:10]` 或更宽窗口可解释大部分 `+1ULP` case。
- 但对每个 dtype 生成 directed 候选并送硬件验证时，false positive 极多。

硬件验证：

| sticky 子窗口 | 覆盖 dtype | 每 dtype 候选 | 总硬件候选 | sticky match | false positive |
|---|---:|---:|---:|---:|---:|
| `mantissa[14:10]` | 57-81 | 40 | 1000 | 2 | 998 |
| `mantissa[14]` | 57-81 | 40 | 1000 | 1 | 999 |

结论：固定 sticky bit mask 不能作为规则。低位形态重要，但硬件是否启用这种边界行为还依赖更早的 signed reduction/partial sum 条件。

### 6.7 复杂硬件边界拟合模型

曾经实现过默认关闭的 `FMAC_MX_SUBNORMAL_HW_MODEL` 一类实验模型，加入同阶抵消、反向 subunit 窗口、dtype73 lane-pair 等条件。它可以让已知原始 mismatch dtype 的 128b 1M 全部通过。

最终没有保留该模型，原因是：

- 判断条件过多，像为样本打补丁。
- RTL 动机不清晰。
- 扩展到更大 directed 空间后没有足够证据支持。
- 与“保持主线模型简洁、可解释、接近论文主算法”的目标冲突。

## 7. directed 实验对 mismatch 范围的修正

全量随机只说明普通随机输入下哪些 dtype 容易暴露；不能证明其它 MX dtype 没有硬件边界差异。

后续 directed search 对此前 128b/256b 全量随机都通过的 MX dtype 做 product cancellation + tiny-subnormal 构造，发现除 dtype81 外，多数 dtype 都可触发 mismatch：

| dtype | 格式 | 硬件候选数 | my | nv | delta magnitude | 结论 |
|---:|---|---:|---:|---:|---:|---|
| 57 | MXF8F6F4_e4m3_X_e4m3 | 926 | 0x000000d2 | 0x000000d3 | +1 | found |
| 59 | MXF8F6F4_e4m3_X_e2m3 | 561 | 0x00000007 | 0x00000006 | -1 | found |
| 60 | MXF8F6F4_e4m3_X_e3m2 | 594 | 0x00000046 | 0x00000047 | +1 | found |
| 67 | MXF8F6F4_e2m3_X_e4m3 | 550 | 0x80000074 | 0x80000075 | +1 | found |
| 69 | MXF8F6F4_e2m3_X_e2m3 | 2628 | 0x00000001 | 0x00000000 | -1 | found |
| 70 | MXF8F6F4_e2m3_X_e3m2 | 1413 | 0x80000001 | 0x00000000 | -1 | found |
| 71 | MXF8F6F4_e2m3_X_e2m1 | 16822 | 0x00000000 | 0x00000001 | +1 | found |
| 72 | MXF8F6F4_e3m2_X_e4m3 | 213 | 0x80000001 | 0x00000000 | -1 | found |
| 74 | MXF8F6F4_e3m2_X_e2m3 | 408 | 0x00000000 | 0x80000001 | +1 | found |
| 75 | MXF8F6F4_e3m2_X_e3m2 | 784 | 0x00000002 | 0x00000001 | -1 | found |
| 76 | MXF8F6F4_e3m2_X_e2m1 | 1788 | 0x00000000 | 0x80000001 | +1 | found |
| 79 | MXF8F6F4_e2m1_X_e2m3 | 19599 | 0x00000000 | 0x00000001 | +1 | found |
| 80 | MXF8F6F4_e2m1_X_e3m2 | 6454 | 0x00000001 | 0x00000000 | -1 | found |
| 81 | MXF8F6F4_e2m1_X_e2m1 | 200000 | - | - | - | not found |

修正后的判断：

- MX scale 路径整体存在 tiny-subnormal 边界差异。
- e5m2 相关组合在随机输入中更容易被命中。
- 非 e5m2 组合也可能通过 directed cancellation 触发。
- dtype81 未触发只能说明当前构造族和预算下没有找到反例，不能证明永远 match。

## 8. subnormal 最大边界实验

针对“是否存在低于某个 subnormal magnitude 才会 mismatch，高于该边界一定 match”的问题，做过两类实验：

| 实验 | 覆盖 dtype | 硬件候选 | exact max-subnormal 候选 | mismatch |
|---|---:|---:|---:|---:|
| exact `0x007fffff` | 57-81 | 13 | 13 | 0 |
| `0x007fffff ± 0x1000` | 57-81 | 2539 | 15 | 0 |

同时，boundary bucket 实验已经把观测到的 mismatch magnitude 推到至少 `my_mag=5641`：

| topology | dtype | 格式 | max mismatch my_mag | delta 方向 |
|---|---:|---|---:|---|
| 128b | 58 | MXF8F6F4_e4m3_X_e5m2 | 5641 | +1 |
| 128b | 66 | MXF8F6F4_e5m2_X_e2m1 | 283 | +1 |
| 256b | 58 | MXF8F6F4_e4m3_X_e5m2 | 177 | +1 |
| 256b | 59 | MXF8F6F4_e4m3_X_e2m3 | 9 | -1 |

结论：

- 已不能把问题限定在极低的几十个 subnormal ulp 内。
- 最大 subnormal 附近暂未发现 mismatch。
- 目前还不能证明存在绝对安全边界。

## 9. 代表性 mismatch 样本

下面样本均按 signed aligned product 展开。分析时应先看 `signed aligned product sum`，再看取绝对值后的 `abs_sum_bin`。硬件输出的 sign 来自 signed sum 的符号，FP32 subnormal mantissa 来自 `abs(signed_sum)` 后的规格化/pack。

### 9.1 128b dtype58, `abs(nv)=abs(my)+1`

```text
dtype=58  MXF8F6F4_e4m3_X_e5m2
scale_total=-155  a_scale=0x63  b_scale=0x00  C=0x00000000
mymac=0x80001609  nvmac=0x8000160a
a=9caef575fc7c2aaa00000000000000004fcf1b9b54d4b8090000000000000000
b=b1b1675338380c0c000000000000000012124949eeee07070000000000000000
product_shift=20  exp_max=-138
```

| lane | sign | scaled_exp | rsft | mant_a * mant_b | Product hex/bin | Product << 20 bin | aligned product bin |
|---:|:---:|---:|---:|---:|---|---|---|
| 0 | + | -162 | 24 | 12 * 5 | 0x3c / 00111100 | 0011110000000000000000000000 | 0000000000000000000000000011 |
| 1 | + | -160 | 22 | 14 * 5 | 0x46 / 01000110 | 0100011000000000000000000000 | 0000000000000000000000010001 |
| 2 | - | -138 | 0 | 13 * 7 | 0x5b / 01011011 | 0101101100000000000000000000 | 0101101100000000000000000000 |
| 3 | + | -143 | 5 | 13 * 7 | 0x5b / 01011011 | 0101101100000000000000000000 | 0000001011011000000000000000 |
| 4 | - | -148 | 10 | 12 * 4 | 0x30 / 00110000 | 0011000000000000000000000000 | 0000000000001100000000000000 |
| 5 | + | -148 | 10 | 12 * 4 | 0x30 / 00110000 | 0011000000000000000000000000 | 0000000000001100000000000000 |
| 16 | + | -164 | 26 | 15 * 6 | 0x5a / 01011010 | 0101101000000000000000000000 | 0000000000000000000000000001 |
| 17 | - | -164 | 26 | 15 * 6 | 0x5a / 01011010 | 0101101000000000000000000000 | 0000000000000000000000000001 |
| 18 | + | -156 | 18 | 11 * 5 | 0x37 / 00110111 | 0011011100000000000000000000 | 0000000000000000000011011100 |
| 19 | - | -156 | 18 | 11 * 5 | 0x37 / 00110111 | 0011011100000000000000000000 | 0000000000000000000011011100 |
| 20 | - | -140 | 2 | 12 * 6 | 0x48 / 01001000 | 0100100000000000000000000000 | 0001001000000000000000000000 |
| 21 | + | -140 | 2 | 12 * 6 | 0x48 / 01001000 | 0100100000000000000000000000 | 0001001000000000000000000000 |

```text
16 active aligned product signed sum = -0x5827fec (-92438508)
signed 28b two's complement        = 1010011111011000000000010100
abs_sum                            = 0x5827fec
abs_sum_bin                        = 0101100000100111111111101100
positive sum                       = 0x14e40f1
negative sum                       = 0x6d0c0dd
```

### 9.2 128b dtype78, `abs(nv)=abs(my)+1`

```text
dtype=78  MXF8F6F4_e2m1_X_e5m2
scale_total=-163  a_scale=0x5b  b_scale=0x00  C=0x00000000
mymac=0x80000008  nvmac=0x80000009
a=2808f068f0d0bc840000000000000000fcdc7c5c6c4ce8c80000000000000000
b=9c9c0707111176950000000000000000c0409898020291910000000000000000
product_shift=22  exp_max=-147
```

| lane | sign | scaled_exp | rsft | mant_a * mant_b | Product hex/bin | Product << 22 bin | aligned product bin |
|---:|:---:|---:|---:|---:|---|---|---|
| 0 | + | -171 | 24 | 2 * 4 | 0x08 / 00001000 | 0010000000000000000000000000 | 0000000000000000000000000010 |
| 1 | - | -171 | 24 | 2 * 4 | 0x08 / 00001000 | 0010000000000000000000000000 | 0000000000000000000000000010 |
| 6 | - | -147 | 0 | 3 * 6 | 0x12 / 00010010 | 0100100000000000000000000000 | 0100100000000000000000000000 |
| 16 | + | -160 | 13 | 3 * 4 | 0x0c / 00001100 | 0011000000000000000000000000 | 0000000000000001100000000000 |
| 17 | + | -160 | 13 | 3 * 4 | 0x0c / 00001100 | 0011000000000000000000000000 | 0000000000000001100000000000 |
| 18 | + | -170 | 23 | 3 * 4 | 0x0c / 00001100 | 0011000000000000000000000000 | 0000000000000000000000000110 |
| 19 | - | -170 | 23 | 3 * 4 | 0x0c / 00001100 | 0011000000000000000000000000 | 0000000000000000000000000110 |

```text
16 active aligned product signed sum = -0x47fd000 (-75485184)
signed 28b two's complement        = 1011100000000011000000000000
abs_sum                            = 0x47fd000
abs_sum_bin                        = 0100011111111101000000000000
positive sum                       = 0x3008
negative sum                       = 0x4800008
```

### 9.3 256b dtype59, `abs(nv)=abs(my)-1`

当前已有 `-1ULP` 样本只在 256b 中复现；把同一 32-lane 输入拆成若干 128b 子集后都不再 mismatch。因此它暂时更像 256b 跨半片/跨 group reduction 相关行为，而不是单个 128b 半片内部规则。

```text
dtype=59  MXF8F6F4_e4m3_X_e2m3
scale_total=-153  a_scale=0x65  b_scale=0x00  C=0x00000000
mymac=0x80000009  nvmac=0x80000008
a=b1a5941426a6c2420d8d30b056d6e56557d71055a727d8582cac4ece37e8f070
b=19193c3c360a282819193c0803032c2c232325050606070734141e1e11113232
product_shift=19  exp_max=-145
```

| lane | sign | scaled_exp | rsft | mant_a * mant_b | Product hex/bin | Product << 19 bin | aligned product bin |
|---:|:---:|---:|---:|---:|---|---|---|
| 0 | - | -152 | 7 | 9 * 9 | 0x51 / 01010001 | 0010100010000000000000000000 | 0000000001010001000000000000 |
| 1 | - | -154 | 9 | 13 * 9 | 0x75 / 01110101 | 0011101010000000000000000000 | 0000000000011101010000000000 |
| 2 | + | -156 | 11 | 12 * 12 | 0x90 / 10010000 | 0100100000000000000000000000 | 0000000000001001000000000000 |
| 3 | - | -156 | 11 | 12 * 12 | 0x90 / 10010000 | 0100100000000000000000000000 | 0000000000001001000000000000 |
| 4 | - | -155 | 10 | 14 * 14 | 0xc4 / 11000100 | 0110001000000000000000000000 | 0000000000011000100000000000 |
| 5 | - | -156 | 11 | 14 * 10 | 0x8c / 10001100 | 0100011000000000000000000000 | 0000000000001000110000000000 |
| 6 | + | -152 | 7 | 10 * 8 | 0x50 / 01010000 | 0010100000000000000000000000 | 0000000001010000000000000000 |
| 7 | - | -152 | 7 | 10 * 8 | 0x50 / 01010000 | 0010100000000000000000000000 | 0000000001010000000000000000 |
| 8 | + | -157 | 12 | 13 * 9 | 0x75 / 01110101 | 0011101010000000000000000000 | 0000000000000011101010000000 |
| 9 | - | -157 | 12 | 13 * 9 | 0x75 / 01110101 | 0011101010000000000000000000 | 0000000000000011101010000000 |
| 10 | - | -152 | 7 | 8 * 12 | 0x60 / 01100000 | 0011000000000000000000000000 | 0000000001100000000000000000 |
| 11 | - | -154 | 9 | 8 * 8 | 0x40 / 01000000 | 0010000000000000000000000000 | 0000000000010000000000000000 |
| 12 | + | -150 | 5 | 14 * 3 | 0x2a / 00101010 | 0001010100000000000000000000 | 0000000010101000000000000000 |
| 13 | - | -150 | 5 | 14 * 3 | 0x2a / 00101010 | 0001010100000000000000000000 | 0000000010101000000000000000 |
| 14 | + | -148 | 3 | 13 * 12 | 0x9c / 10011100 | 0100111000000000000000000000 | 0000100111000000000000000000 |
| 15 | - | -148 | 3 | 13 * 12 | 0x9c / 10011100 | 0100111000000000000000000000 | 0000100111000000000000000000 |
| 16 | - | -150 | 5 | 15 * 3 | 0x2d / 00101101 | 0001011010000000000000000000 | 0000000010110100000000000000 |
| 17 | + | -150 | 5 | 15 * 3 | 0x2d / 00101101 | 0001011010000000000000000000 | 0000000010110100000000000000 |
| 18 | - | -158 | 13 | 8 * 5 | 0x28 / 00101000 | 0001010000000000000000000000 | 0000000000000000101000000000 |
| 19 | + | -150 | 5 | 13 * 5 | 0x41 / 01000001 | 0010000010000000000000000000 | 0000000100000100000000000000 |
| 20 | - | -156 | 11 | 15 * 6 | 0x5a / 01011010 | 0010110100000000000000000000 | 0000000000000101101000000000 |
| 21 | + | -156 | 11 | 15 * 6 | 0x5a / 01011010 | 0010110100000000000000000000 | 0000000000000101101000000000 |
| 22 | - | -149 | 4 | 8 * 7 | 0x38 / 00111000 | 0001110000000000000000000000 | 0000000111000000000000000000 |
| 23 | + | -149 | 4 | 8 * 7 | 0x38 / 00111000 | 0001110000000000000000000000 | 0000000111000000000000000000 |
| 24 | - | -154 | 9 | 12 * 12 | 0x90 / 10010000 | 0100100000000000000000000000 | 0000000000100100000000000000 |
| 25 | - | -154 | 9 | 12 * 12 | 0x90 / 10010000 | 0100100000000000000000000000 | 0000000000100100000000000000 |
| 26 | + | -149 | 4 | 14 * 14 | 0xc4 / 11000100 | 0110001000000000000000000000 | 0000011000100000000000000000 |
| 27 | - | -149 | 4 | 14 * 14 | 0xc4 / 11000100 | 0110001000000000000000000000 | 0000011000100000000000000000 |
| 28 | + | -153 | 8 | 15 * 9 | 0x87 / 10000111 | 0100001110000000000000000000 | 0000000001000011100000000000 |
| 29 | - | -146 | 1 | 8 * 9 | 0x48 / 01001000 | 0010010000000000000000000000 | 0001001000000000000000000000 |
| 30 | + | -145 | 0 | 8 * 10 | 0x50 / 01010000 | 0010100000000000000000000000 | 0010100000000000000000000000 |
| 31 | - | -145 | 0 | 8 * 10 | 0x50 / 01010000 | 0010100000000000000000000000 | 0010100000000000000000000000 |

```text
32 aligned product signed sum = -0x1200a00 (-18876928)
signed 28b two's complement  = 1110110111111111011000000000
abs_sum                      = 0x1200a00
abs_sum_bin                  = 0001001000000000101000000000
positive sum                 = 0x3ca5c80
negative sum                 = 0x4ea6680
```

### 9.4 128b BF16, `abs(nv)=abs(my)+1`

这是 2026-04-30 从 128b 10M 测试日志中发现并最小化后的 BF16 case。原始日志 first mismatch：

```text
First mismatch summary: dtype=BF16(2) vec=2694985 tag=BF16 my=0x80000fb9 nv=0x80000fba seed_mode=legacy base_seed=113
```

原始 vec 的逻辑输入为：

```text
a_logical=8004004200420046804900588031806c
b_logical=8036ba74000f006900762f3b805b0055
C=0x00000000
```

把无贡献项删除后，可压缩成 2 个非零 BF16 product：

```text
dtype=BF16(2)
C=0x00000000
mymac=0x80000fb9
nvmac=0x80000fba
a_logical=00000042000000000000005800000000
b_logical=0000ba740000000000002f3b00000000
a_physical=0000004200000000000000000000000000000058000000000000000000000000
b_physical=0000ba7400000000000000000000000000002f3b000000000000000000000000
product_shift=11
exp_max=-137
```

| lane | sign | scaled_exp | rsft | mant_a * mant_b | Product hex/bin | Product << 11 bin | aligned product bin |
|---:|:---:|---:|---:|---:|---|---|---|
| 1 | - | -137 | 0 | 66 * 244 | 0x3ee8 / 0011111011101000 | 0001111101110100000000000000 | 0001111101110100000000000000 |
| 9 | + | -159 | 22 | 88 * 187 | 0x4048 / 0100000001001000 | 0010000000100100000000000000 | 0000000000000000000000001000 |

```text
16 active aligned product signed sum = -0x1f73ff8 (-32980984)
signed 28b two's complement        = 1110000010001100000000001000
abs_sum                            = 0x1f73ff8
abs_sum_bin                        = 0001111101110011111111111000
positive sum                       = 0x8
negative sum                       = 0x1f74000
```

这个 BF16 case 的结构比 MXF8F6F4 代表样本更简单：一个 dominant negative product 决定 `exp_max=-137`，另一个 positive product 右移 22 bit 后只剩 `+8`。`mymac` exact RZ 得到 `0x80000fb9`，tested Blackwell-class GPU `nvmac` 给出 `0x80000fba`，即 negative subnormal magnitude 多 1 ulp。

## 10. 对 signed sum 的当前理解

只看 `abs_sum_bin` 容易丢掉一个关键信息：多数代表样本的 signed sum 是负数，最终输出是先由 signed reduction 得到负尾数和，再取绝对值做 FP32 subnormal pack。

例如 dtype58 样本：

```text
signed sum = -0x5827fec
abs sum    =  0x5827fec
```

硬件和 `mymac` 差 1 ulp 时，可能不是“abs_sum_bin 某些低位直接触发 +1”这么简单，而是 signed reduction 中正负项抵消、borrow/carry 链、局部 partial sum 归并方式共同决定了最终进入 pack 的 magnitude。

这也解释了为什么单 product 和简单三 product 构造没有复现：它们没有形成足够复杂的 signed reduction residue。

## 11. 最可能的硬件位置

目前更可能出问题或行为分歧的位置：

1. MX scale 后 product 进入局部 reduction tree。
2. 多个 signed significand 在固定内部精度下对阶。
3. 正负项抵消后产生非常小的 signed residual。
4. residual 进入 final FP32 tiny-subnormal writeback。
5. 某个局部 low-bit cut、sticky 汇总、carry-save 到 CPA 边界、lane/group partial-sum 汇合点，没有完全等价于数学 exact RZ。

不太可能是：

- operand decode 系统性错误；
- scale exponent 简单加法错误；
- 所有 FP32 subnormal pack 都有统一 sticky/RNE；
- BF16/TF32 共用 pack 路径存在无条件错误；
- 单 product tiny subnormal 错误。

## 12. 对 RTL 设计的解释

从 RTL 角度看，当前现象更像低位实现 artifact，而不是一个有意设计的复杂舍入策略。

合理动机包括：

- MXF8F6F4 的低精度 Tensor Core 路径需要高吞吐、低面积、低功耗，不太可能为了极罕见 FP32 tiny-subnormal 输出保留完整跨 lane sticky/carry 信息。
- scale 让 exponent 动态范围显著变宽，大量 product 需要远距离右移；完整保留所有被移出的低位代价高。
- 局部 carry-save tree、分段 CPA、lane group partial sum 可能只保证 normal 主区间精度，而在 underflow 边界出现 1 ulp 级差异。
- 真实 RTL 可能只有一个简单的局部截断或 sticky 汇总结构；软件上看到的复杂条件只是该结构叠加随机输入后的表象。

因此，不建议把复杂 if 条件写进主线模拟器。若要为 tested Blackwell-class GPU 做 bit compatibility，应作为独立 compatibility mode，并继续用 directed corpus 证明每条规则。

## 13. 维护建议

主线 `mymac` 建议继续保持 exact RZ：

- 与论文主算法一致。
- 与 exact rational/RZ 复算一致。
- 对 10M 预算下的 FP16、TF32、非 MX F8F6F4、MXFP4/NVFP4、integer 回归稳定；但 BF16 已发现 tiny-subnormal 反例，因此 BF16 不能再作为“完全无差异”类别。
- 逻辑简单，可指导 RTL 主设计。

后续如果继续追求 tested Blackwell-class GPU bit-match，建议：

1. 使用 `FMAC_DUMP_MISMATCH_INPUT=1` 建立更多 128b/256b payload corpus。
2. 每个候选规则必须同时解释 `+1` 与 `-1`。
3. 不再只看 `abs_sum_bin`，必须记录 signed aligned product sum、positive sum、negative sum、各 lane partial sum。
4. 优先按 lane group / partial reduction tree / carry-save boundary 设计 directed case。
5. compatibility mode 必须默认关闭，不能污染当前 exact RZ 主线。

## 14. 当前最终判断

当前最强结论是：

- `mymac` 默认模型实现的是和论文 FDA/RZ 主路径一致的 MXF8F6F4 数学模型。
- tested Blackwell-class GPU `nvmac` 在 MXF8F6F4 scale path 且结果落入 FP32 tiny-subnormal/underflow 边界时，存在少量 1 ulp 级硬件行为差异；BF16 也已发现同类 1 ulp negative subnormal case。
- 对 MXF8F6F4，这个差异和多 product signed reduction/cancellation 关系更强，单 product 或简单 `+P-P+R` 不能触发；对 BF16，当前最小反例是 2 个非零 product 的 dominant term + tiny residual。
- 差异方向既可能是 `abs(nv)=abs(my)+1`，也可能是 `abs(nv)=abs(my)-1`；全量随机日志目前主要提供 first vec/count，方向判断仍依赖带 payload 的 directed corpus。
- TF32 目前仍未复现 mismatch；已有 128b 10M random、10,000 个 unique-subnormal directed 候选和 1 个手工 BF16-like 构造均 match RZ。
- MMA-Sim 不是当前 tested Blackwell-class GPU MXF8F6F4 的最终 oracle。
- 目前不应合入复杂拟合逻辑；如果要 bit-match tested Blackwell-class GPU，需要继续以 RTL 结构假设驱动 directed 实验。
