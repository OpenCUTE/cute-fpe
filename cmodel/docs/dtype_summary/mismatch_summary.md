# 测试中发现无法对齐的 dtype

这里的“无法对齐”指 `mymac` / `nvmac` 输出出现 mismatch。

## README 中记录的 mismatch 结论

| 测试 / 实验 | 发现无法对齐的 dtype | 说明 |
| --- | --- | --- |
| 128b，`57-81`，1M 扫描 | `58, 61, 62, 63, 64, 65, 66, 68, 73, 77, 78` | 36 个 mymac/nvmac mismatch；共同特征是 `C=±0`，最终结果落入 FP32 tiny subnormal/underflow 边界，通常为 1 ulp 差异。 |
| 2026-04-27，128b/256b 全量回归，每 dtype 1,000,000 vec | `58, 62, 63, 64, 65, 66, 68, 73, 78` | 256b 剩余 32 个 mismatch；均位于 FP32 subnormal/zero 区间，`abs(nv) - abs(my) = 1` 个 final subnormal ulp。 |
| 对随机未失败的 MX dtype 做 directed 构造 | `57, 59, 60, 67, 69, 70, 71, 72, 74, 75, 76, 79, 80` | 这些 dtype 也能触发 tiny-subnormal 1 ulp mismatch；说明 MX scale 路径整体存在边界差异，只是随机命中概率不同。 |
| directed 构造暂未触发 | `81` | `81 = e2m1_X_e2m1` 在 200,000 个 directed 硬件候选内未触发。 |

## 当前日志中可见的 128b 10M mismatch

本地 `fmac_test_nv_fp32_128b_logs` 中还记录了 10,000,000 vec 级别的结果：

| dtype | 数据格式 | first vec | mismatch_count | first my | first nv |
| --- | --- | ---: | ---: | --- | --- |
| `2` | `BF16` | `2694985` | `3` | `0x80000fb9` | `0x80000fba` |
| `57` | `MXF8F6F4_e4m3_X_e4m3` | `1451552` | `6` | `0x8000000c` | `0x8000000b` |
| `58` | `MXF8F6F4_e4m3_X_e5m2` | `394630` | `34` | `0x80000005` | `0x80000006` |
| `59` | `MXF8F6F4_e4m3_X_e2m3` | `5953803` | `1` | `0x00000018` | `0x00000017` |
| `60` | `MXF8F6F4_e4m3_X_e3m2` | `3219883` | `7` | `0x00000029` | `0x0000002a` |
| `61` | `MXF8F6F4_e4m3_X_e2m1` | `570998` | `6` | `0x80000004` | `0x80000005` |
| `62` | `MXF8F6F4_e5m2_X_e4m3` | `868445` | `33` | `0x00000001` | `0x00000002` |
| `63` | `MXF8F6F4_e5m2_X_e5m2` | `13555` | `55` | `0x8000007f` | `0x80000080` |
| `64` | `MXF8F6F4_e5m2_X_e2m3` | `225897` | `45` | `0x00000037` | `0x00000038` |
| `65` | `MXF8F6F4_e5m2_X_e3m2` | `225897` | `50` | `0x0000000d` | `0x0000000e` |
| `66` | `MXF8F6F4_e5m2_X_e2m1` | `225897` | `57` | `0x0000006f` | `0x00000070` |
| `67` | `MXF8F6F4_e2m3_X_e4m3` | `5746871` | `1` | `0x00000002` | `0x00000003` |
| `68` | `MXF8F6F4_e2m3_X_e5m2` | `394630` | `44` | `0x8000017f` | `0x80000180` |
| `72` | `MXF8F6F4_e3m2_X_e4m3` | `1842055` | `3` | `0x80000005` | `0x80000006` |
| `73` | `MXF8F6F4_e3m2_X_e5m2` | `313657` | `46` | `0x8000003f` | `0x80000040` |
| `77` | `MXF8F6F4_e2m1_X_e4m3` | `927702` | `4` | `0x80000008` | `0x80000009` |
| `78` | `MXF8F6F4_e2m1_X_e5m2` | `372190` | `57` | `0x00000000` | `0x80000001` |

## 已通过或暂未发现 mismatch 的类别

| 类别 | README / 日志结论 |
| --- | --- |
| `FP16` | 128b 10M 日志中 `finished with no mismatches`。 |
| `TF32` | 128b 10M 日志中 `finished with no mismatches`；README 中 directed tiny-product 对照也未发现 mismatch。 |
| `F8F6F4 32-56` 非 scale | README 记录 directed 对照全部 match；本地 fp32 / fp16 128b 10M 日志也全部通过。 |
| `NVFP4/MXFP4 9/10` | README 记录全量回归通过；本地 128b 10M 日志也通过。 |
| 当前默认 integer dtype | README 记录通过；本地 128b 10M 日志中默认覆盖的 integer dtype 也通过。 |

