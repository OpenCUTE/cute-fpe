# MMA 模拟器使用说明

> 如果你拿到的是整理后的源码包，建议先看 [PORTABLE_NOTES.md](PORTABLE_NOTES.md)，里面有跨机器编译所需的最短说明。

本项目用于对比同一个 MMA 输出元素的两条计算路径：

- `mymac()` / `mymac_scale()`：软件参考模型，定义在 [fmac.h](fmac.h)
- `nvmac()` / `nvmac_scale()`：NVIDIA MMA 硬件执行路径，定义在 [nv_mac.h](nv_mac.h)

主程序为 [all_fmac_test.cu](all_fmac_test.cu)。

它可以完成：

- 生成 `a/b/c/d` 与 scale 数据
- 比较软件模型与 NVIDIA 路径结果
- 在 mismatch 时输出复现信息
- 从已落盘文件中 replay 单个向量
- 输出中间 `mant_*` trace 供人工分析或 RTL 对照

## 0. 最近行为更新

当前版本相对早期版本，已同步到代码中的主要变化如下：

- 新增 `--output-fp16`
  对支持 `fp16` 输出的 dtype，可直接切到 `fp16` 累加/输出。

- `fp16` 输出时，`c/d` 仍按 32-bit 落盘
  当前规则是：高 16 bit 补零，低 16 bit 为有效 `fp16` 数据。
  replay 也已按这个格式适配。

- `NVIDIA_ON=0` 时默认不再强制执行第二遍 reference
  如需显式打开 reference 校验，请使用 `--validate-reference`。

- 普通输出文件与 trace 文件写句柄已复用
  降低了每个 vec 的 `fopen/fclose` 开销。

- `fmac.h` 中 legacy/generic 浮点路径的中间缓冲改为复用
  修复了长跑时的中间数组泄漏，并减少了频繁 `malloc/free`。

- `decodeadd()` 已删除临时运行时实验开关
  不再支持 `DECODEADD_ALIGN_RNE`、`DECODEADD_LATE_RNE_BITS`、`DECODEADD_FP32_SUBNORMAL_RNE` 这几条环境变量实验路径；当前 FP32 输出路径固定为 RZ，不再对正 subnormal 做 RNE pack。
  FDA 路径（`decimal_num=25`）默认按有效非零 mantissa 项选择 `exp_max`。MXF8F6F4 tiny-subnormal 相关的 zero-C exponent、subnormal RNE 和硬件边界拟合逻辑均已从主代码删除；相关结论只保留在第 11 节作为实验记录。

- `debug_data/` 中已删除 `fp32_pack_*`
  当前只保留硬件验证需要的：
  `mant_align_*`、`mant_sum_*`。

- 整数路径的 `mant_sum_*` 已补齐
  当前语义为：
  `partial sum` 按原始乘积顺序均分四组，不包含 `c`；
  `complete sum` 才包含 `c`。

- `S4/U4` 的 `*_double.bin.log` 改为紧密打包
  一个字节中存两个连续 4bit 元素；replay 与覆盖检查已同步适配。

- Mixed `f8f6f4` 非 scale 的 NVIDIA 硬件路径已补齐 `fp16` 输出验证
  现在 `32-56` 这组在 `NVIDIA_ON=1` 且带 `--output-fp16` 时，会走 `f16.<A>.<B>.f16` 的硬件指令并继续与 `mymac()` 对比。

- `fp16` 输出打包对 zero 做 canonical +0
  当 RNE 后 FP16 subnormal/underflow 尾数变为 0 时，`mymac` 不再保留负号，而是输出 `0x0000`，与 NVIDIA MMA 的 `fp16` 输出行为一致。

- `TOPOLOGY_128_INPUT=1` 下，`FP16/BF16/TF32` 会优先走原生 128bit 输入的 MMA 指令
  具体为：
  - `FP16`：`m16n8k8`
  - `BF16`：`m16n8k8`
  - `TF32`：`m16n8k4`
  同时输出文件前缀也会改成对应的 `k8/k4` 形状。

- `NVFP4/MXFP4/MXINT4` 的 scale 语义已固定为当前实现
  - `NVFP4`（dtype `9`）保持每 16 个 FP4 元素一个 scale，默认 256b 输入写 `4` 个 scale
  - `MXFP4`（dtype `10`）与 `MXINT4`（dtype `19`）改为每 32 个 4-bit 元素一个外部 scale：256b 输入写 `2` 个，512b 输入写 `4` 个
  - CModel/NVIDIA 的四个内部 16-element 槽位按 `[s0, s0, s1, s1]` 接收这两个 256b 外部 scale；RTL 只在 ScaleBus 低位传入有效 scale，其余位为 `0`
  - `TOPOLOGY_128_INPUT=1` 时，MXFP4/MXINT4 仅有一个有效外部 scale；NVFP4 仍遵循原有 blocksize16 策略
  - `mxf8f6f4 scale_vec::1X` 仍然是每个 vec 只写 `1` 个有效 scale

- 已删除旧的 deprecated dtype 入口
  下面这些旧编号不再保留：
  - `7 -> 57`
  - `8 -> 63`
  - `11 -> 69`
  - `12 -> 75`
  - `17 -> 81`
  - `13 -> 32`
  - `14 -> 38`
  - `15 -> 44`
  - `16 -> 50`
  - `18 -> 56`

- 新增 `make gen_data`
  可批量循环多个 dtype，生成单独目录下的数据输出。

- `make test_nv_fp32_128b` / `make test_nv_fp32_256b` 默认不再写 `rawhex/txt/bin`
  默认 `TEST_NV_DUMP=0`，只保留 group 日志，避免完整回归产生超大输出目录。需要调试落盘文件时显式加 `TEST_NV_DUMP=1`。
  这两个目标默认使用 `TEST_NV_FP32_128B_JOBS=27`、`TEST_NV_FP32_256B_JOBS=27` 并行跑 group；当前默认 `VEC_COUNT=10000000`，可按需要显式覆盖。

- 新增 `make test_nv_fp16_128b` / `make test_nv_fp16_256b`
  只跑 `DTYPES_FP16_*` 分组，并强制传入 `--output-fp16`。日志分别写入 `fmac_test_nv_fp16_128b_logs` 与 `fmac_test_nv_fp16_256b_logs`，不会覆盖 `fmac_test_nv_fp32_128b_logs` / `fmac_test_nv_fp32_256b_logs`。

- scale 文件新增 raw-order 输出
  原有 scale 文件仍保留历史反序/调试格式；新增的 `*_raw_order.*` 文件按外部消费者看到的原始 scale 顺序输出，写入时复用文件句柄以降低百万向量生成时的 `fopen/fclose` 开销。

- `FMAC_DUMP_MISMATCH_INPUT=1` 可在 mismatch 时打印复现输入
  输出块包含 dtype、vec、`c/my/nv`、`a/b` logical payload、physical payload 和 `a_scale/b_scale`。无 scale 的 dtype 会打印空 scale 数组。`make test_nv_fp32_128b` / `make test_nv_fp32_256b` 及 FP16 输出专用 test 目标会自动打开该环境变量，使 mismatch 输入直接进入 group 日志。

- MXF8F6F4 与 MMA-Sim / nvmac 的当前结论已明确
  57-81 的剩余差异集中在 `C=±0` 且最终结果落入 FP32 tiny subnormal/underflow 边界。已尝试的 zero-C exponent、普通 subnormal RNE、全局 final-ulp 截断和复杂边界拟合模型都不是可作为 RTL 设计依据的通用规则；主代码保持 exact RZ，实验拟合逻辑已删除。

## 1. 编译

默认编译：

```bash
cd cmodel/necessary_source
make build
```

如果不希望编译 NVIDIA MMA 路径，只保留软件模拟器：

```bash
make build NVIDIA_ON=0
```

说明：

- `NVIDIA_ON=1`：编译完整版本，包含 NVIDIA MMA 指令路径
- `NVIDIA_ON=0`：只保留软件模型与 fallback 路径
- 默认 `CXXFLAGS` 为 `-x c++ -std=c++11 -O2`
- `STATIC_CXX_RUNTIME=1`：链接时静态带上 `libstdc++/libgcc`，用于降低跨机器运行时对目标机 C++ 运行库版本的依赖
- `FMAC_DUMP_OUTPUT=0`：编译出的程序不写主输出、scale 输出与 trace 输出文件；默认 build 为 `1`
- `FMAC_DUMP_ALL=1`：写完整输出，包括 `txt.log`、`rawhex.txt.log`、`double.bin.log`、scale raw-order 和 trace；默认 `0` 时只写 `a/b/c` 的 `double.bin.log`、`a_scale/b_scale` 的 `double.bin.log` 和 `d` 的 `txt.log`
- `EXTRA_DEFINES="..."`：给 `build`、`build_256b`、`build_gen_data` 追加编译期实验宏

可执行文件：

- `all_fmac_test`

### 1.1 跨机器运行

如果在本机编译出的 `all_fmac_test` 拿到另一台机器运行时出现：

```text
/lib64/libstdc++.so.6: version `GLIBCXX_3.4.xx' not found
```

说明目标机的 `libstdc++` 比编译机旧。可以先在编译机使用静态 C++ 运行库版本：

```bash
make build_static_runtime NVIDIA_ON=1
```

等价写法：

```bash
make build NVIDIA_ON=1 STATIC_CXX_RUNTIME=1
```

这个开关只解决 `libstdc++/libgcc` 的版本依赖。如果错误里出现 `GLIBC_2.33`、`GLIBC_2.34` 这类 `glibc` 版本缺失，说明编译机系统本身比目标机新；这种情况应在目标机或更老的兼容系统/容器中重新编译，不能只靠静态 `libstdc++` 解决。

## 2. 命令行接口

查看帮助：

```bash
./all_fmac_test --help
```

查看所有支持的 dtype：

```bash
./all_fmac_test --list-dtypes
```

常用参数：

- `--dtype N`：只运行一个 dtype
- `--all-dtypes`：运行所有支持的 dtype
- `--all-128b-dtypes`：在 128bit 输入模式下对每类数据格式跑一个代表指令
- `--vec-start N`：从第几个向量开始
- `--vec-count N`：运行多少个向量
- `--replay-vec N`：从输出的文件中对其中某个向量重新跑一次
- `--seed N`：设置随机种子
- `--seed-mode legacy|per_dtype|per_vec`：随机数模式
- `--output-fp16`：对支持的浮点 dtype 切到 `fp16`
- `--blocksize16`：仅对 `mxf4nvf4` 且 `TOPOLOGY_128_INPUT=1` 生效；加上这个参数表示前两个 block 各自使用一个 scale；不加则表示前两个 block 共享同一个 scale
- `--validate-reference`：在 `NVIDIA_ON=0` 时强制打开 reference 对照
- `--stop-on-mismatch`：发现 mismatch 立即停止
- `--print-every N`：进度打印间隔

推荐调试时使用：

```bash
--seed-mode per_vec --seed <seed>
```

这样更容易稳定复现某一个 vec。

## 3. 默认运行行为

如果直接执行：

```bash
./all_fmac_test
```

默认跑的浮点 dtype 集合为：

- `1`：`FP16`
- `2`：`BF16`
- `3`：`TF32`
- `9`：`NVFP4`
- `10`：`MXFP4`
- `19`：`MXINT4`
- `32`：`F8F6F4_e4m3_X_e4m3`
- `38`：`F8F6F4_e5m2_X_e5m2`
- `44`：`F8F6F4_e2m3_X_e2m3`
- `50`：`F8F6F4_e3m2_X_e3m2`
- `56`：`F8F6F4_e2m1_X_e2m1`

如果你要验证其它 mixed 或整数 dtype，建议显式使用 `--dtype` 或 `--all-dtypes`。

## 4. 当前支持的 dtype

### 4.1 原始 legacy 浮点 / 特殊浮点

- `1`：`FP16`
- `2`：`BF16`
- `3`：`TF32`
- `9`：`NVFP4`
- `10`：`MXFP4`
- `19`：`MXINT4`

### 4.2 Mixed `f8f6f4`

- 编号范围：`32-56`
- shape：`m16n8k32`
- 支持 `fp32/fp16` 输出切换

示例：

- `34`：`F8F6F4_e4m3_X_e2m3`
- `38`：`F8F6F4_e5m2_X_e5m2`
- `56`：`F8F6F4_e2m1_X_e2m1`

### 4.3 Mixed `mxf8f6f4` block-scale

- 编号范围：`57-81`
- shape：`m16n8k32`
- `scale_vec::1X`
- `stype=ue8m0`
- 当前不支持 `fp16` 输出

示例：

- `58`：`MXF8F6F4_e4m3_X_e5m2`
- `69`：`MXF8F6F4_e2m3_X_e2m3`
- `81`：`MXF8F6F4_e2m1_X_e2m1`

### 4.4 整数 MMA

- `15`：`s4 x s4 -> s32`（RTL opcode-compatible）
- `16`：`s4 x u4 -> s32`（RTL opcode-compatible）
- `17`：`u4 x s4 -> s32`（RTL opcode-compatible）
- `18`：`u4 x u4 -> s32`（RTL opcode-compatible）
- `96-119`：`s8/u8`
- `128-151`：`s4/u4`

示例：

- `98`：`m8n8k16_s8_X_u8`
- `114`：`m16n8k32_s8_X_u8`
- `146`：`m16n8k64_s4_X_u4`

这些编号中的奇数项通常是对应的 `satfinite` 版本。

`15-18` 使用无 scale 的整数累加语义。C-model 为了便于生成、打印和复现，
会把每个逻辑 int4 元素存到一个字节的低 4 bit；落盘的 `*_double.bin.log`
则把相邻两个 nibble 紧密打包成一个字节：元素 `0` 在低 4 bit，元素 `1`
在高 4 bit，和 RTL 输入流及 PTX 寄存器打包顺序一致。

默认 `--bitsize 512` 对应 64 个逻辑 int4。也可以使用
`--int4-k N` 指定精确的点积长度，支持 `32`、`64`、`128` 或任意正整数，
例如：

```bash
./all_fmac_test --dtype 15 --int4-k 32  --vec-count 1
./all_fmac_test --dtype 15 --int4-k 64  --vec-count 1
./all_fmac_test --dtype 15 --int4-k 128 --vec-count 1
./all_fmac_test --dtype 15 --int4-k 37  --vec-count 1
```

这个选项只允许 dtype `15-18`，且优先于 `--bitsize`。自定义 K 的输出文件会
附加 `_k<N>`，replay 时也必须带相同的 `--int4-k N`。NVIDIA 硬件路径会将该
点积切成原生 `m8n8k32` / `m16n8k64` MMA tile，最后一个 tile 自动补零；每个
tile 的 `D[0]` 作为下一 tile 的 `C[0]`，因此最终结果就是所请求 K 个元素的
一个输出点。

## 4.5 当前支持验证的 NV MMA 硬件指令

下面这部分描述的是：当 `NVIDIA_ON=1` 且运行环境支持对应 MMA 指令时，当前项目里实际接了 NVIDIA 硬件执行路径的指令族。

| 指令类别 | 覆盖 dtype | 主要硬件指令族 | NV `fp16` 输出验证 | 当前验证入口 |
| --- | --- | --- | --- | --- |
| 同构 `FP16` | `1` | 默认：`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` / `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`；`TOPOLOGY_128_INPUT=1`：`mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` / `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16` | 支持 | `nvmac()` |
| 同构 `BF16` | `2` | 默认：`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`；`TOPOLOGY_128_INPUT=1`：`mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32` | 不支持 | `nvmac()` |
| 同构 `TF32` | `3` | 默认：`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`；`TOPOLOGY_128_INPUT=1`：`mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32` | 不支持 | `nvmac()` |
| Mixed `f8f6f4` 非 scale | `32-56` | `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.<A>.<B>.f32` / `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.<A>.<B>.f16` | 支持 | `nvmac()` |
| Mixed `mxf8f6f4` block-scale | `57-81` | `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.<A>.<B>.f32.ue8m0` | 不支持 | `nvmac_scale()` |
| `NVFP4` | `9` | `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`；`TOPOLOGY_128_INPUT=1` 时仍保留 `4X` 运行时布局，但只让前两个 block 有效并把后两个 block 归零 | 不支持 | `nvmac_scale()` |
| `MXFP4` | `10` | `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0`；外部每 32 个元素一个 scale，进入硬件前扩展为 `[s0,s0,s1,s1]` | 不支持 | `nvmac_scale()` |
| `MXINT4` | `19` | 无原生 NVIDIA MMA；使用 CModel/RTL 的 S4 packed 计算路径，外部每 32 个元素一个 scale | 不适用 | `mymac_scale()` |
| 整数 MMA | `15-18`, `96-119`, `128-151` | `mma.sync.aligned.*.s32.<a_kind>.<b_kind>.s32` | 不适用 | `nvmac()` |

补充说明：

- Mixed `f8f6f4` 非 scale 覆盖 `e4m3/e5m2/e2m3/e3m2/e2m1` 的全部 25 种 `<A>/<B>` 组合。
- Mixed `mxf8f6f4` block-scale 同样覆盖这 25 种组合，固定为 `scale_vec::1X`、`stype=ue8m0`。
- 整数 MMA 当前覆盖：
  - dtype `15-18` 的 RTL int4 标量输出点；`--int4-k N` 时依次执行 `m8n8k32` 或 `m16n8k64`，尾部补零
  - `m8n8k16`、`m16n8k16`、`m16n8k32` 的 `s8/u8`
  - `m8n8k32`、`m16n8k32`、`m16n8k64` 的 `s4/u4`
  - 每组都覆盖 `s×s / s×u / u×s / u×u` 以及 `satfinite` / 非 `satfinite`

## 5. 常见用法

跑一个 dtype 的一个向量：

```bash
./all_fmac_test --dtype 58 --vec-count 1 --seed 1 --seed-mode per_vec --stop-on-mismatch
```

跑一个短回归：

```bash
./all_fmac_test --dtype 34 --vec-count 100 --seed 1 --seed-mode per_vec --stop-on-mismatch --output-fp16
```

replay 一个已落盘向量：

```bash
./all_fmac_test --dtype 58 --replay-vec 0
```

打开中间 trace：

```bash
FMAC_TRACE_INTERMEDIATE=1 ./all_fmac_test --dtype 34 --vec-count 1 --seed 1 --seed-mode per_vec --output-fp16
```

## 6. 输出文件与命名规则

如果编译时设置 `FMAC_DUMP_OUTPUT=0`，本节中的主输出、scale 输出和 trace 文件都不会写入；程序仍会在 stdout/stderr 打印 mismatch 摘要。`make test_nv_fp32_128b` 与 `make test_nv_fp32_256b` 默认使用该模式。
默认 `FMAC_DUMP_OUTPUT=1 FMAC_DUMP_ALL=0` 是精简落盘模式，只生成 `a/b/c` 的 `double.bin.log`、scale 类 dtype 的 `a_scale/b_scale` 的 `double.bin.log`，以及 `d` 的 `txt.log`。需要恢复历史完整文件集时使用 `FMAC_DUMP_ALL=1`。

### 6.1 主输出目录

默认目录：

- `output_data/`

每个 `a/b/c/d` 会生成：

- `*.txt.log`
- `*_rawhex.txt.log`
- `*_double.bin.log`

### 6.2 文件 stem 规则

当前主输出采用统一 descriptor 风格命名：

```text
<shape>_<a_token>_<b_token>_to_<accumulator>
```

例如：

- `m16n8k16_fp16_fp16_to_fp16_*`
- `m16n8k32_e4m3_e2m3_to_fp16_*`
- `m16n8k32_mxe4m3_mxe5m2_to_fp32_*`
- `m8n8k16_s8_u8_to_s32_*`

补充说明：

- `TOPOLOGY_128_INPUT=1` 且 `dtype=1/2/3` 时，文件前缀会按实际 native MMA 指令形状写成：
  - `FP16/BF16`：`m16n8k8_*`
  - `TF32`：`m16n8k4_*`
- 对 `mxf4nvf4`（`dtype 9/10`）：
  - 输出文件前缀会显式带上 `--blocksize16` 的运行时策略：打开时写 `_b16`，未打开时写 `_b32`，例如 `m16n8k64_mxfp4_mxfp4_to_fp32_b16_a_double.bin.log` 和 `m16n8k64_mxfp4_mxfp4_to_fp32_b32_a_double.bin.log`
  - `a.txt.log / b.txt.log` 现在按一个 fp4 数据一行输出，方便人工核对
  - `a_rawhex.txt.log / b_rawhex.txt.log` 维持“第一列 hex、第二列 raw bits”的习惯；`mant:` 字段继续显示解码后的尾数
  - `*_double.bin.log` 仍保持现有二进制打包格式，不因文本输出格式调整而变化

### 6.3 Scale 文件

默认目录：

- `scale_output_data/`

默认 build 下，scale 文件与主 `a/b` 使用相同 stem，但单独写到 `scale_output_data/`。
因此 `mxf4nvf4` 的 scale 文件同样会继承 `_b16/_b32` 后缀，例如 `m16n8k64_nvfp4_nvfp4_to_fp32_b32_a_scale_double.bin.log`；`*_raw_order.*` 文件也沿用同一 stem。

scale 文件当前按“实际有效 scale 数”落盘，而不是按物理 scale 槽位数落盘。
例如：

- `NVFP4 (dtype 9)`：256b 模式每个 vec 写 `4` 个 scale；`TOPOLOGY_128_INPUT=1` 时每个 vec 写 `2` 个 scale
- `MXFP4 (dtype 10)`：外部接口每 32 个 FP4 元素使用一个 scale；因此 256b 模式每个 vec 写 `2` 个 scale，512b 模式写 `4` 个 scale。256b 的两枚 scale 在 CModel/NVIDIA 的四个 16-element 内部槽位中展开为 `[s0, s0, s1, s1]`。`TOPOLOGY_128_INPUT=1` 时只有一个有效外部 scale，并展开为 `[s0, s0, 0, 0]`；`--blocksize16` 不改变 MXFP4 的该规则。
- `mxf8f6f4 scale_vec::1X`：每个 vec 写 `1` 个 scale

另外，对于不需要 scale 的格式，会额外生成一个占位文件：

- `scale_output_data/data_for_no_scale.bin.log`

每个 scale 文件还会同步生成一份 `*_raw_order.*` 文件。`*_raw_order.*` 不做历史反序处理，按外部消费者/硬件输入侧看到的原始 scale 顺序写出；优先给后续工具链和外部脚本消费。

### 6.4 Trace 文件

默认目录：

- `debug_data/`

当前保留：

- `mant_align_*`
- `mant_sum_*`

`fp32_pack_*` 与 `mant_mul_*` 已删除。

## 7. 批量生成数据

`Makefile` 中新增了：

- `build_gen_data`
- `gen_data`

可用于批量生成多个 dtype 的数据。

示例：

```bash
make gen_data DTYPES="1 34 57 98" VEC_COUNT=1000 GEN_ARGS="--output-fp16"
```

默认变量：

- `GEN_OUTPUT_DATA_DIR=test_output_data`
- `GEN_TRACE_FILE=test_debug_data`
- `GEN_SCALE_OUTPUT_DATA_DIR=test_scale_output_data`
- `DTYPES=...`
- `VEC_COUNT=1000`
- `ACCUM_OUTPUT=fp32|fp16`

注意：

- `GEN_OUTPUT_DATA_DIR` 与 `GEN_TRACE_FILE` 会直接影响主输出目录与 trace 目录
- `GEN_SCALE_OUTPUT_DATA_DIR` 会统一影响：
  - 真正的 `a_scale/b_scale` 文件
  - `data_for_no_scale.bin.log` 占位文件

统一的累加输出切换方式：

```bash
make gen_data ACCUM_OUTPUT=fp16
make test_nv_fp16_128b
make test_nv_fp16_256b
make test_128b_trace ACCUM_OUTPUT=fp16
make test_128b_reply ACCUM_OUTPUT=fp16
```

说明：

- `ACCUM_OUTPUT=fp32`：默认行为，不额外添加参数
- `ACCUM_OUTPUT=fp16`：自动附加 `--output-fp16`
- `test_nv_fp32_128b/test_nv_fp32_256b` 是 FP32 输出 NV 回归入口，只使用 `TEST_NV_FP32_GROUPS`
- `test_nv_fp16_128b/test_nv_fp16_256b` 是 FP16 输出 NV 回归入口，只使用 `DTYPES_FP16_*`，避免全量 dtype 中不支持 FP16 输出的格式混入统计。

也可以覆盖输出目录：

```bash
make gen_data \
  GEN_OUTPUT_DATA_DIR=my_output_data \
  GEN_TRACE_FILE=my_debug_data \
  GEN_SCALE_OUTPUT_DATA_DIR=my_scale_output_data \
  DTYPES="32 34 38"
```

批量 NV 回归：

```bash
make test_nv_fp32_128b VEC_COUNT=1000000
make test_nv_fp32_256b VEC_COUNT=1000000
make test_nv_fp16_128b VEC_COUNT=1000000
make test_nv_fp16_256b VEC_COUNT=1000000
```

所有 `test_nv_*` 目标会先执行 `make check_gpu`。它必须打印实际可见 GPU 的名称、compute capability 和驱动版本；若失败，说明当前 shell 或容器没有拿到 NVIDIA 设备节点，不能将输出的 mismatch 视为 CModel/NVIDIA 对比结果。

`test_nv_*_256b` 默认传入 `--bitsize 256`。以 `MXFP4` 为例，这会生成两枚外部 scale，并在四个 NVIDIA MMA scale 槽位中使用 `[s0, s0, s1, s1]`。如需有意测试其它位宽，可覆盖 `TEST_NV_256B_ARGS`。

默认 `TEST_NV_DUMP=0`，因此不会写 `output_data*`、`scale_output_data*`、`debug_data*` 下的文件。需要保留精简落盘数据时显式打开：

```bash
make test_nv_fp32_128b VEC_COUNT=1000000 TEST_NV_DUMP=1 \
  OUTPUT_DATA_DIR=debug_output_128b \
  TRACE_FILE=debug_trace_128b \
  SCALE_OUTPUT_DATA_DIR=debug_scale_128b
```

如果需要历史完整输出文件集，再额外加 `FMAC_DUMP_ALL=1`。

批量 NV 回归日志可以用 `make show_test_logs` 汇总。`show_test_logs` 会同时检查 `fmac_test_nv_fp32_128b_logs`、`fmac_test_nv_fp32_256b_logs`、`fmac_test_nv_fp16_128b_logs`、`fmac_test_nv_fp16_256b_logs`。按 group 日志判定：只要同一个 `group_*.log` 中任意 dtype 出现 `first mismatch at vec`、`results mismatch` 或非零 `mismatch_count`，该 group 就判为 `FAIL`；只有没有这些失败标记且存在 `finished with no mismatches` 时才判为 `PASS`。

## 8. 环境变量

- `FMAC_TRACE_INTERMEDIATE=1`
  打开中间 `mant_*` trace

- `TOPOLOGY_128_INPUT=1`
  打开 128bit logical payload 适配模式

- `FP4_TOPOLOGY_128=1`
  打开 FP4 相关的 128bit 拓扑输入模式

- `CONTIGUOUS_128_INPUT=1`
  实验性调试开关，仅用于 A/B 实验与根因定位

- `FMAC_DUMP_MISMATCH_INPUT=1`
  出现 mismatch 时，在 stdout 打印 `FMAC_MISMATCH_INPUT_BEGIN/END` 包围的复现输入块。scale 类 dtype 会包含 `a/b/c/a_scale/b_scale`；非 scale dtype 会包含 `a/b/c` 并打印空 `a_scale/b_scale`。NV test 的 Makefile 目标会自动打开该变量，因此复现输入会写入对应 group 日志。

## 9. 推荐调试流程

1. 用 `--stop-on-mismatch` 跑一个短测试
2. 记录程序打印的复现信息
3. 用 `--replay-vec` 回放该向量
4. 如有需要，打开 `FMAC_TRACE_INTERMEDIATE=1`，会在 `debug_data/` 中生成 `mant_align_*`、`mant_sum_*` 两类调试数据，写入的是软件模型内部的中间运算结果。
5. 对任意 dtype，可打开 `FMAC_DUMP_MISMATCH_INPUT=1` 获取完整输入块；scale 类 dtype 还会带上 `a_scale/b_scale`，可用于 MMA-Sim 或独立脚本重放。
6. 人工查看 `output_data/`、`scale_output_data/`、`debug_data/`

示例：

```bash
./all_fmac_test --dtype 58 --seed 1 --seed-mode per_vec --vec-count 100 --stop-on-mismatch
FMAC_DUMP_MISMATCH_INPUT=1 TOPOLOGY_128_INPUT=1 ./all_fmac_test --dtype 58 --vec-count 100000 --stop-on-mismatch
./all_fmac_test --dtype 58 --replay-vec 0
```

## 10. 当前注意事项

- 默认输出仍是 `fp32`；支持的 dtype 如需切到 `fp16`，请显式使用 `--output-fp16`
- `57-81` 这组 `mxf8f6f4` block-scale 当前不支持 `fp16` 输出
- `--replay-vec` 依赖目标 dtype 的落盘文件已经存在
- 普通 `make build` / `make gen_data` 默认会写输出文件；`make test_nv_fp32_128b` / `make test_nv_fp32_256b` 默认不写输出文件，只写 group 日志
- `NVIDIA_ON=0` 时默认不会强制执行第二遍 reference；如需保留对照，请使用 `--validate-reference`
- 旧的 deprecated dtype 已删除，不再保留兼容入口
- 每次修改代码、Makefile、输出格式、调试开关或验证结论后，都要同步更新本文档，并删除与当前代码冲突的旧描述

## 11. MXF8F6F4 精度收敛实验小结

当前主代码保持 exact RZ 行为：`decodeadd()` 不再包含 zero-C exponent、正/负 subnormal RNE 或 tested Blackwell-class GPU tiny-subnormal 边界拟合分支。之前用于收敛 nvmac 输出的实验逻辑已删除，避免把“能拟合当前样本”的复杂条件误当成可指导 RTL 的正式规则。

更完整的全量回归数据、数值范围分析、MMA-Sim/论文对照和 RTL 解释见 [precision.md](precision.md)。

已完成的关键实验结论：

- MMA-Sim 不能作为 dtype `57-81` 的最终 oracle。部分 dtype58/62/63 case 中 MMA-Sim match nvmac、不 match mymac；但 128b 1M 支持子集复验显示 dtype57/58/62/63 中 MMA-Sim 也会出现少量 nvmac mismatch。论文验证硬件是 `NVIDIA RTX PRO 6000 Blackwell`，当前对照硬件是 `tested Blackwell-class GPU`，二者不是同一 SKU；本机 MMA-Sim 对 `e3m2/e2m3/e2m1` 组合也不完整。
- 128b dtype `57-81` 1M 扫描得到 36 个 mymac/nvmac mismatch，分布在 dtype `58/61/62/63/64/65/66/68/73/77/78`。共同特征是 `C=±0`，最终结果落入 FP32 tiny subnormal/underflow 边界，通常表现为 1 ulp 差异。
- 2026-04-27 的 128b/256b 全量回归各覆盖默认 63 个 dtype、每 dtype 1,000,000 个向量。非 MX F8F6F4、BF16、TF32、MXFP4/NVFP4 和当前默认 integer dtype 均通过；256b 剩余 32 个 mismatch，分布在 MXF8F6F4 dtype `58/62/63/64/65/66/68/73/78`。本轮 first mismatch summary 全部位于 FP32 subnormal/zero 区间，且 `abs(nv) - abs(my) = 1` 个 final subnormal ulp。
- 对全量随机未失败的 MX dtype 继续做 directed 构造后，`57/59/60/67/69/70/71/72/74/75/76/79/80` 均能触发 tiny-subnormal 1 ulp mismatch；`81 = e2m1_X_e2m1` 在 200,000 个 directed 硬件候选内未触发。结论应表述为：MX scale 路径整体存在边界差异，只是不同 dtype 在随机 1M 中的命中概率不同。
- 2026-04-28 的 subnormal bucket directed 实验进一步确认 mismatch 方向不只有 `abs(nv)=abs(my)+1`，也存在 `abs(nv)=abs(my)-1`；新实验已把观测到的最高 mismatch magnitude 推到 `my_mag=5641`，但最大 FP32 subnormal `0x007fffff` 附近 2552 个硬件候选未发现 mismatch。详见 [precision.md](precision.md) 第 9.2 节。
- 2026-04-28 的 BF16/TF32 tiny-product directed 对照覆盖 `a*b ~= 2^-160` 及 K 项累加接近 `2^-149` 的区域，128b/256b 共验证 BF16 8892 个、TF32 4788 个候选，均未发现 mismatch。详见 [precision.md](precision.md) 第 9.3 节。
- 2026-04-29 的 MX 单 product tiny-residual 对照把其它 lane 全部置零，只保留一个 scaled product；dtype `57-81` 共验证 75,000 个硬件候选，其中 25,000 个为非零 subnormal 分层样本，未发现 mismatch。说明已知 MX mismatch 需要多个 product 的 signed reduction/cancellation 参与，而不是单个 product 被 scale 搬到 subnormal 区间即可触发。详见 [precision.md](precision.md) 第 9.4 节。
- 2026-04-29 的 MX 三 product cancellation 对照只保留 `+P - P + R` 三个非零 product，并扫 scale 到非零 subnormal；dtype `57-81` 共验证 12,500 个 per-dtype 硬件候选，仍未发现 mismatch。说明触发条件可能需要更复杂的多项 reduction，而不是一个完全抵消 pair 加一个 residual 即可触发。详见 [precision.md](precision.md) 第 9.5 节。
- 2026-04-29 的 subnormal sticky / near-carry pack 对照显示：低位窗口 carry-in 可以解释 dtype58/dtype78 手工最小 case 以及现有 JSON mismatch corpus 的 45/47 个 `+1` 样本，但作为无条件 pack 规则会产生大量 false positive。继续尝试固定 bit 子窗口后，`mantissa[14:10]` 与 `mantissa[14]` 的硬件 false-positive sweep 分别得到 `998/1000` 与 `999/1000` 个反例；因此 sticky/near-carry 只能视为局部现象，还缺少 signed cancellation 或 partial-sum 边界一类启用条件。详见 [precision.md](precision.md) 第 9.6 节。
- 对 36 个 case 重新计算 exact rational RZ，结果全部与 mymac 默认输出一致，而不是与 nvmac 一致。因此当前 mymac 没有漏掉数学 exact RZ；差异更像是 tested Blackwell-class GPU 在 MX scale 路径某个局部 reduction/writeback 边界上丢失或提前截断了极低位贡献。
- 无条件采用 MMA-Sim 的 `C=0 -> exponent -126`、普通 subnormal RNE、全局 final-ulp 截断、mod8 lane-pair 或 16-lane group 截断都会引入回归。曾经收敛出的复杂硬件边界模型可以 match 已知样本，但需要多层条件保护，RTL 合理性不足，因此不保留在主代码。
- 非 MX directed 对照已验证：用临时 `/tmp` harness 对 dtype `32-56` 构造 product 抵消和 tiny FP32 C 扰动，共 `25 * 3 * 24 = 1800` 个 case，mymac 与 nvmac 全部 match。说明非 MX 与 MX 可以复用主 datapath，但 MX scale 更容易把 product 本身推到 FP32 tiny-subnormal 边界；非 MX 中 tiny C 往往在 product 对阶时被 RZ 截掉，不触发同类边界现象。

当前最可信的 RTL 假设是：tested Blackwell-class GPU Tensor Core 的 MX scale 路径在结果接近 FP32 tiny subnormal/underflow 时，某个局部 carry-save / 分段 CPA / writeback 边界没有把所有低位 contribution 和 sticky 信息完整保留到最终 pack。这可以降低低位 carry propagation、sign-magnitude cancellation 和 sticky 汇总成本，但只在 `C=±0` 且结果贴着 final subnormal ulp 边界时暴露为少量 `±1 ulp` 差异。这个假设仍是对硬件现象的解释，不应直接作为新 RTL 的目标行为。

后续如需继续追 nvmac 兼容，应把兼容逻辑放在独立实验分支或显式 compatibility mode 中，并继续使用 `FMAC_DUMP_MISMATCH_INPUT=1` 建 corpus；主线实现保持可解释的 exact RZ。
