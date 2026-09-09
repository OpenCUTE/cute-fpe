# FMAC 测试与 Debug 使用指南

这个文档只讲怎么直接使用这份代码来测试某一个 `dtype`，发现哪里出错，以及后续怎么 debug。

## 0. 运行环境

当前目录里的 `all_fmac_test` 是 Linux 可执行文件，不是 Windows `.exe`。

如果你在 Windows PowerShell 里，不能直接这样跑：

```bash
./all_fmac_test
```

需要在 Linux / WSL / CUDA 服务器环境里运行。

进入项目目录：

```bash
cd cmodel/necessary_source
```

如果代码在 Linux 服务器上，就进入你实际存放项目的目录。

## 1. 编译

如果你要对比软件模型和 NVIDIA 硬件 MMA 结果，用这个：

```bash
make build NVIDIA_ON=1
```

先确认当前 shell 能访问 GPU：

```bash
make check_gpu
```

该命令必须输出 GPU 名称、compute capability 和驱动版本；若失败，当前容器或会话没有拿到 NVIDIA 设备，不能进行有效的硬件对比。

这种模式会比较：

- `mymac()` / `mymac_scale()`：软件参考模型
- `nvmac()` / `nvmac_scale()`：NVIDIA MMA 硬件路径

如果你只是想生成数据，不跑 NVIDIA 硬件对比：

```bash
make build NVIDIA_ON=0
```

如果要打开完整输出文件，方便 debug：

```bash
make build NVIDIA_ON=1 FMAC_DUMP_ALL=1
```

## 2. 查看支持哪些 dtype

```bash
./all_fmac_test --list-dtypes
```

常用 dtype 范围：

```text
1      FP16
2      BF16
3      TF32
9      NVFP4
10     MXFP4
32-56  F8F6F4 mixed，非 scale
57-81  MXF8F6F4 block-scale
96-119 s8/u8 integer MMA
128-151 s4/u4 integer MMA
```

## 3. 测试某一个 dtype

例如测试 `dtype=58`，跑 1000 个向量，遇到第一个 mismatch 就停：

```bash
FMAC_DUMP_MISMATCH_INPUT=1 ./all_fmac_test \
  --dtype 58 \
  --vec-count 1000 \
  --seed 1 \
  --seed-mode per_vec \
  --stop-on-mismatch
```

这是最推荐的单 dtype debug 命令。

参数含义：

- `--dtype 58`：只测试 dtype 58
- `--vec-count 1000`：测试 1000 个随机向量
- `--seed 1`：固定随机种子
- `--seed-mode per_vec`：让每个 vec 都能稳定复现
- `--stop-on-mismatch`：发现第一个 mismatch 立即停止
- `FMAC_DUMP_MISMATCH_INPUT=1`：出错时打印完整复现输入

## 4. 判断是否出错

如果没有错误，会看到类似：

```text
MXF8F6F4_e4m3_X_e5m2 Format test vec[0, 1000) finished with no mismatches
```

如果有错误，会看到类似：

```text
First mismatch summary: dtype=MXF8F6F4_e4m3_X_e5m2(58) vec=12345 tag=MXF8F6F4_e4m3_X_e5m2 my=0x00000001 nv=0x00000002 seed_mode=per_vec base_seed=1
Direct repro: ./all_fmac_test --dtype 58 --seed 1 --seed-mode per_vec --vec-start 12345 --vec-count 1 --stop-on-mismatch
MXF8F6F4_e4m3_X_e5m2 Format test vec[0, 1000) first mismatch at vec 12345, mismatch_count=1
```

重点看：

- `vec=12345`：第几个测试向量出错
- `my=0x...`：软件模型结果
- `nv=0x...`：NVIDIA 硬件结果
- `Direct repro`：可以直接复现这个错误的命令

## 5. 复现某一个出错 vec

把程序打印的 `Direct repro` 命令拿来跑。

例如出错的是 `vec=12345`：

```bash
FMAC_DUMP_MISMATCH_INPUT=1 ./all_fmac_test \
  --dtype 58 \
  --seed 1 \
  --seed-mode per_vec \
  --vec-start 12345 \
  --vec-count 1 \
  --stop-on-mismatch
```

这一步只跑一个向量，日志最干净，适合后续逐步 debug。

## 6. 查看完整 mismatch 输入

如果打开了：

```bash
FMAC_DUMP_MISMATCH_INPUT=1
```

出错时会打印：

```text
FMAC_MISMATCH_INPUT_BEGIN
dtype=58 name=MXF8F6F4_e4m3_X_e5m2 vec=12345 a_kind=e4m3 b_kind=e5m2 shape=m16n8k32
c=0x00000000 my=0x00000001 nv=0x00000002
a=[...]
b=[...]
a_payload=[...]
b_payload=[...]
a_physical=[...]
b_physical=[...]
a_scale=[...]
b_scale=[...]
FMAC_MISMATCH_INPUT_END
```

含义：

- `a/b`：输入 payload
- `c`：累加初值
- `my`：软件模型输出
- `nv`：NVIDIA 硬件输出
- `a_scale/b_scale`：scale 类 dtype 的 scale 输入
- `a_physical/b_physical`：实际送入计算路径的物理 payload

这个块可以保存成日志，后面用脚本分析。

## 7. 打开中间 trace

如果要看软件模型内部到底哪一步不一样，先重新编译并打开完整输出：

```bash
make build NVIDIA_ON=1 FMAC_DUMP_ALL=1
```

然后跑单个出错 vec：

```bash
FMAC_TRACE_INTERMEDIATE=1 FMAC_DUMP_MISMATCH_INPUT=1 ./all_fmac_test \
  --dtype 58 \
  --seed 1 \
  --seed-mode per_vec \
  --vec-start 12345 \
  --vec-count 1 \
  --stop-on-mismatch
```

trace 文件会写到：

```text
debug_data/
```

重点看两个文件：

```text
debug_data/mant_align_*.txt.log
debug_data/mant_sum_*.txt.log
```

`mant_align` 用来看对阶过程：

- `exp_max`：当前累加选择的最大指数
- `exp_ab`：乘积项指数
- `exp_c`：C 输入指数
- `rsft_amt`：右移对阶位数
- `mant_mul`：乘积 mantissa
- `mant_shifted`：对阶后的 mantissa
- `discarded`：右移时丢掉的 bit
- `guard/sticky/lsb`：舍入相关信息
- `signed_mant`：带符号后的参与累加值

`mant_sum` 用来看累加结果：

- partial sum
- complete sum
- 最后 pack 前后的中间值

一般 debug 顺序是：

1. 先看 `my` 和 `nv` 差多少
2. 再看 `mant_align` 中哪一项对阶时丢了 bit
3. 再看 `mant_sum` 中 partial sum 是否符合预期
4. 最后看 pack/输出阶段是否出现 subnormal、underflow、rounding 差异

## 8. 查看落盘输入输出

如果编译时没有关闭输出，程序会写这些目录：

```text
output_data/
scale_output_data/
debug_data/
```

主输入输出在：

```text
output_data/
```

常见文件：

```text
*_a_double.bin.log
*_b_double.bin.log
*_c_double.bin.log
*_d_double.bin.log
*_a_rawhex.txt.log
*_b_rawhex.txt.log
*_c_rawhex.txt.log
*_d_rawhex.txt.log
```

scale 输入在：

```text
scale_output_data/
```

trace 在：

```text
debug_data/
```

如果只是想人工看数据，优先看 `*_rawhex.txt.log`。

如果是脚本消费或 replay，优先用 `*_double.bin.log`。

## 9. replay 已经落盘的 vec

如果之前已经生成过对应 dtype 的落盘文件，可以 replay 某一个 vec：

```bash
./all_fmac_test --dtype 58 --replay-vec 12345
```

注意：

- `--replay-vec` 必须带 `--dtype`
- replay 依赖 `output_data/` 和 `scale_output_data/` 里已有对应文件
- 如果之前没有落盘文件，replay 会失败

## 10. 测试 FP16 输出

部分 dtype 支持 `fp16` 输出，例如 `FP16` 和 `F8F6F4 32-56`。

例如测试 `dtype=34` 的 FP16 输出：

```bash
FMAC_DUMP_MISMATCH_INPUT=1 ./all_fmac_test \
  --dtype 34 \
  --vec-count 1000 \
  --seed 1 \
  --seed-mode per_vec \
  --output-fp16 \
  --stop-on-mismatch
```

注意：

```text
57-81 这组 MXF8F6F4 block-scale 当前不支持 --output-fp16
```

## 11. 128bit 输入模式

如果你要测 128bit topology 输入，加环境变量：

```bash
TOPOLOGY_128_INPUT=1 FMAC_DUMP_MISMATCH_INPUT=1 ./all_fmac_test \
  --dtype 58 \
  --vec-count 1000 \
  --seed 1 \
  --seed-mode per_vec \
  --stop-on-mismatch
```

如果还要 trace：

```bash
TOPOLOGY_128_INPUT=1 FMAC_TRACE_INTERMEDIATE=1 FMAC_DUMP_MISMATCH_INPUT=1 ./all_fmac_test \
  --dtype 58 \
  --seed 1 \
  --seed-mode per_vec \
  --vec-start 12345 \
  --vec-count 1 \
  --stop-on-mismatch
```

## 12. 批量测试

跑多个 dtype 并生成数据：

```bash
make gen_data DTYPES="1 34 58 98" VEC_COUNT=1000
```

跑 NVIDIA FP32 128bit 回归：

```bash
make test_nv_fp32_128b VEC_COUNT=100000
```

跑 NVIDIA FP32 256bit 回归：

```bash
make test_nv_fp32_256b VEC_COUNT=100000
```

该目标默认使用 `--bitsize 256`。其中 `MXFP4` 每个操作数写两枚外部 scale，并在 NVIDIA 的四个 scale 槽位中展开为 `[s0, s0, s1, s1]`。

跑 NVIDIA FP16 128bit 回归：

```bash
make test_nv_fp16_128b VEC_COUNT=100000
```

汇总回归日志：

```bash
make show_test_logs
```

默认 NV 回归只写 group 日志，不写完整数据文件。

如果需要保留输入输出文件用于 debug，加：

```bash
make test_nv_fp32_128b VEC_COUNT=100000 TEST_NV_DUMP=1
```

如果还要完整 `txt/rawhex/bin/trace` 文件，加：

```bash
make test_nv_fp32_128b VEC_COUNT=100000 TEST_NV_DUMP=1 FMAC_DUMP_ALL=1
```

## 14. 最常用命令清单

编译 NVIDIA 对比版本：

```bash
make build NVIDIA_ON=1
```

测试一个 dtype：

```bash
FMAC_DUMP_MISMATCH_INPUT=1 ./all_fmac_test --dtype 58 --vec-count 1000 --seed 1 --seed-mode per_vec --stop-on-mismatch
```

复现一个出错 vec：

```bash
FMAC_DUMP_MISMATCH_INPUT=1 ./all_fmac_test --dtype 58 --seed 1 --seed-mode per_vec --vec-start 12345 --vec-count 1 --stop-on-mismatch
```

打开 trace debug：

```bash
make build NVIDIA_ON=1 FMAC_DUMP_ALL=1
FMAC_TRACE_INTERMEDIATE=1 FMAC_DUMP_MISMATCH_INPUT=1 ./all_fmac_test --dtype 58 --seed 1 --seed-mode per_vec --vec-start 12345 --vec-count 1 --stop-on-mismatch
```

看回归日志：

```bash
make show_test_logs
```

## 15. 建议的 debug 顺序

推荐按这个顺序排查：

1. 先用 `--dtype` 跑一个短测试。
2. 加 `--stop-on-mismatch` 找第一个出错 vec。
3. 用 `Direct repro` 或 `--vec-start 出错vec --vec-count 1` 稳定复现。
4. 加 `FMAC_DUMP_MISMATCH_INPUT=1` 保存完整输入。
5. 加 `FMAC_TRACE_INTERMEDIATE=1` 打开 `mant_align` 和 `mant_sum`。
6. 先比较 `my` / `nv`，再看 `mant_align`，最后看 `mant_sum`。
