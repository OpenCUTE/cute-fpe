# C model (MMA simulator) guide

This is a bit-accurate software model of one MMA output element. The main
program, `all_fmac_test`, computes each element on two paths and compares them:

- `mymac()` / `mymac_scale()`: the software reference model, defined in [fmac.h](necessary_source/fmac.h)
- `nvmac()` / `nvmac_scale()`: the NVIDIA MMA hardware path, defined in [nv_mac.h](necessary_source/nv_mac.h)

It can:

- generate `a/b/c/d` and scale data
- compare the software model against the NVIDIA hardware path
- print reproduction information on a mismatch
- replay a single vector from files already written to disk
- write intermediate `mant_*` traces for manual analysis or RTL comparison

Vector-by-vector comparison between the RTL and this model is done in
[rtl_cmodel_compare](../rtl_cmodel_compare/README.md). RTL opcodes and this
model's dtype ids are numbered differently (for example RTL opcodes `20/21` are
model dtypes `50/44`); [config/modes.json](../config/modes.json) is the
authoritative mapping.

## 1. Source files

All sources are in [necessary_source/](necessary_source/):

| File | Contents |
|---|---|
| `all_fmac_test.cu` | main program: command-line parsing, dtype selection, seeds, test loop and mismatch reporting |
| `mma_descriptor.h` | dtype table: id, name, shape, operand formats, scale type, accumulator |
| `gendata.h` | per-dtype data generation, the comparison loop, replay, and the mismatch input dump |
| `fmac.h` | software model: operand decode, `decodemul()` products, `decodeadd()` alignment and accumulation, output packing, `mymac()` / `mymac_scale()`, intermediate traces |
| `FloatDecode.h` | floating-point field decode, leading-zero count, exception codes |
| `nv_mac.h` | `nvmac()` / `nvmac_scale()`: packs operands into MMA register layouts and calls the kernels |
| `mma_kernel.h` | CUDA kernels with inline-PTX MMA instructions (built only with `NVIDIA_ON=1`) |
| `FileManager.h` | dump-file writing and reading |
| `myrandom.h`, `random_allinone.h` | random value generators for every format, including subnormal, NaN, Inf and zero |
| `Makefile` | build, batch data generation and NV regression targets |

## 2. Building

`all_fmac_test` is a Linux executable; run it on Linux, WSL or the CUDA server,
not in Windows PowerShell.

Default build (software model only, compiled with `c++`, no CUDA/GPU needed):

```bash
cd cmodel/necessary_source
make build
```

To compare against NVIDIA MMA hardware, build with the hardware path. It is
compiled with `nvcc` (default `CUDA_ARCH=120a`) and needs the CUTLASS headers at
`CUTLASS_DIR` (default `~/cutlass`):

```bash
make build NVIDIA_ON=1
make check_gpu
```

`make check_gpu` must print the GPU name, compute capability and driver
version. If it fails, the current shell or container cannot see the NVIDIA
device, and any comparison result is invalid.

Build options:

- `NVIDIA_ON=0` (default): software model and fallback paths only
- `NVIDIA_ON=1`: full build including the NVIDIA MMA instruction path
- The default `CXXFLAGS` is `-x c++ -std=c++11 -O2`
- `FMAC_DUMP_OUTPUT=0`: write no main, scale or trace files; the default is `1`
- `FMAC_DUMP_ALL=1`: write the full output set, including `txt.log`, `rawhex.txt.log`, `double.bin.log`, scale raw-order files and traces; the default `0` writes only `double.bin.log` for `a/b/c`, `double.bin.log` for `a_scale/b_scale`, and `txt.log` for `d`
- `STATIC_CXX_RUNTIME=1`: link `libstdc++/libgcc` statically
- `EXTRA_DEFINES="..."`: append compile-time macros to `build`, `build_256b` and `build_gen_data`

### 2.1 Running on another machine

If an `all_fmac_test` built here fails on another machine with:

```text
/lib64/libstdc++.so.6: version `GLIBCXX_3.4.xx' not found
```

the target's `libstdc++` is older than the build machine's. Build with the
static C++ runtime:

```bash
make build_static_runtime NVIDIA_ON=1
# same as: make build NVIDIA_ON=1 STATIC_CXX_RUNTIME=1
```

This only removes the `libstdc++/libgcc` version dependency. If the error names
a missing `glibc` version such as `GLIBC_2.33`, rebuild on the target or in an
older compatible system or container.

## 3. Command-line interface

```bash
./all_fmac_test --help
./all_fmac_test --list-dtypes
```

Options:

- `--dtype N`: run only one dtype
- `--all-dtypes`: run all supported dtypes
- `--all-128b-dtypes`: in 128-bit input mode, run one representative instruction per data format
- `--bitsize N`: input vector width in bits, default `512`
- `--int4-k N`: exact dot-product length for dtypes `15-18`, see section 4.4
- `--vec-start N`: first vector index to run
- `--vec-count N`: number of vectors to run, default `1000000`
- `--replay-vec N`: rerun one vector from the dump files
- `--seed N`: base random seed
- `--seed-mode legacy|per_dtype|per_vec`: how seeds are derived; `per_vec` gives every vector its own seed so any single vector can be reproduced
- `--output-fp16`: switch supported floating-point dtypes to `fp16` accumulation and output
- `--blocksize16`: only for `mxf4nvf4` with `TOPOLOGY_128_INPUT=1`; with it, the first two blocks each use their own scale, without it they share one scale
- `--validate-reference`: force the `nvmac` reference comparison when `NVIDIA_ON=0`
- `--stop-on-mismatch`: stop at the first mismatch
- `--print-every N`: progress print interval, default `50000`, `0` disables

Running `./all_fmac_test` without `--dtype` runs the default floating-point set:
`1` FP16, `2` BF16, `3` TF32, `9` NVFP4, `10` MXFP4, `19` MXINT4, and
`32/38/44/50/56` (the five homogeneous `f8f6f4` combinations). Use `--dtype` or
`--all-dtypes` for the others.

## 4. Supported dtypes

| ids | Format | Shape / scale | `fp16` output |
|---|---|---|---|
| `1` | FP16 | `m16n8k16` | yes |
| `2` | BF16 | `m16n8k16` | no |
| `3` | TF32 | `m16n8k8` | no |
| `9` | NVFP4 | `m16n8k64`, one `ue4m3` scale per 16 elements | no |
| `10` | MXFP4 | `m16n8k64`, one `ue8m0` scale per 32 elements | no |
| `19` | MXINT4 | S4 payload, one scale per 32 elements | no |
| `15-18` | S4/U4 × S4/U4 → S32, RTL opcode-compatible | see 4.4 | no |
| `32-56` | mixed `f8f6f4` without scale | `m16n8k32` | yes |
| `57-81` | mixed `mxf8f6f4` block-scale | `m16n8k32`, `scale_vec::1X`, `ue8m0` | no |
| `96-119` | `s8/u8` integer MMA | `m8n8k16`, `m16n8k16`, `m16n8k32` | no |
| `128-151` | `s4/u4` integer MMA | `m8n8k32`, `m16n8k32`, `m16n8k64` | no |

### 4.1 Mixed `f8f6f4` and `mxf8f6f4`

`32-56` and `57-81` each cover all 25 `<A>/<B>` combinations of
`e4m3/e5m2/e2m3/e3m2/e2m1`, for example `34` = `F8F6F4_e4m3_X_e2m3`,
`50` = `F8F6F4_e3m2_X_e3m2`, `58` = `MXF8F6F4_e4m3_X_e5m2`,
`69` = `MXF8F6F4_e2m3_X_e2m3`. Run `--list-dtypes` for the full table.

### 4.2 Integer MMA

In `96-119` and `128-151`, odd ids are usually the `satfinite` variant of the
id before them. Every group covers `s×s / s×u / u×s / u×u`.

### 4.3 Scale semantics

- `NVFP4` (dtype `9`): one scale per 16 FP4 elements; 256-bit input writes `4` scales.
- `MXFP4` (dtype `10`) and `MXINT4` (dtype `19`): one external scale per 32 4-bit elements, so 256-bit input writes `2` scales and 512-bit input writes `4`. The model and the NVIDIA path receive the two 256-bit scales in their four internal 16-element slots as `[s0, s0, s1, s1]`. The RTL receives the valid scales only in the low bits of ScaleBus, with all other bits `0`.
- With `TOPOLOGY_128_INPUT=1`, MXFP4/MXINT4 have one valid external scale (`[s0, s0, 0, 0]`); NVFP4 follows the `--blocksize16` policy.
- `mxf8f6f4 scale_vec::1X` (dtypes `57-81`): one scale per vec.

### 4.4 RTL int4 dtypes `15-18`

These use integer accumulation without scale. The model stores each logical
int4 element in the low 4 bits of a byte. The `*_double.bin.log` dump packs two
adjacent nibbles into one byte, element `0` in the low 4 bits and element `1` in
the high 4 bits, the same order as the RTL input stream and the PTX register
packing.

The default `--bitsize 512` gives 64 logical int4 elements. `--int4-k N` sets
the exact dot-product length (any positive integer):

```bash
./all_fmac_test --dtype 15 --int4-k 32  --vec-count 1
./all_fmac_test --dtype 15 --int4-k 128 --vec-count 1
./all_fmac_test --dtype 15 --int4-k 37  --vec-count 1
```

`--int4-k` is only allowed for `15-18` and takes precedence over `--bitsize`.
Dump files get an `_k<N>` suffix, and replay must pass the same `--int4-k N`.
The NVIDIA path cuts the dot product into native `m8n8k32` / `m16n8k64` tiles,
zero-pads the last tile, and feeds each tile's `D[0]` into the next tile's
`C[0]`.

### 4.5 Output arithmetic

- The FP32 output path rounds toward zero (RZ), not RNE. For example a value exactly at `0x7f.8` is truncated to `0x7f`.
- The FP16 output path (`--output-fp16`) uses RNE. When an FP16 subnormal/underflow mantissa becomes 0 after rounding, the result is canonical `+0` (`0x0000`).
- With `fp16` output, `c/d` are still dumped as 32 bits: the upper 16 bits are zero and the lower 16 bits hold the `fp16` value.
- With `TOPOLOGY_128_INPUT=1`, FP16/BF16 use `m16n8k8` and TF32 uses `m16n8k4`, and the dump-file prefixes use the same shapes.

### 4.6 NVIDIA hardware instructions

With `NVIDIA_ON=1`, these instruction families are wired to `nvmac()` / `nvmac_scale()`:

| Class | dtypes | Hardware instruction family | NV `fp16` output | Entry point |
| --- | --- | --- | --- | --- |
| `FP16` | `1` | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` / `...f16.f16.f16.f16`; `TOPOLOGY_128_INPUT=1`: `m16n8k8` | yes | `nvmac()` |
| `BF16` | `2` | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`; `TOPOLOGY_128_INPUT=1`: `m16n8k8` | no | `nvmac()` |
| `TF32` | `3` | `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`; `TOPOLOGY_128_INPUT=1`: `m16n8k4` | no | `nvmac()` |
| `f8f6f4` without scale | `32-56` | `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.<A>.<B>.f32` / `...f16.<A>.<B>.f16` | yes | `nvmac()` |
| `mxf8f6f4` block-scale | `57-81` | `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.<A>.<B>.f32.ue8m0` | no | `nvmac_scale()` |
| `NVFP4` | `9` | `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`; with `TOPOLOGY_128_INPUT=1` only the first two blocks are valid | no | `nvmac_scale()` |
| `MXFP4` | `10` | `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0`; scales expanded to `[s0,s0,s1,s1]` | no | `nvmac_scale()` |
| `MXINT4` | `19` | no native NVIDIA MMA; compared against the model's S4 packed path | n/a | `mymac_scale()` |
| Integer | `15-18`, `96-119`, `128-151` | `mma.sync.aligned.*.s32.<a_kind>.<b_kind>.s32` | n/a | `nvmac()` |

## 5. Testing one dtype

Recommended single-dtype command: 1000 random vectors, stop at the first
mismatch, print the full input on failure:

```bash
FMAC_DUMP_MISMATCH_INPUT=1 ./all_fmac_test \
  --dtype 58 --vec-count 1000 --seed 1 --seed-mode per_vec --stop-on-mismatch
```

A passing run prints:

```text
MXF8F6F4_e4m3_X_e5m2 Format test vec[0, 1000) finished with no mismatches
```

A failing run prints:

```text
First mismatch summary: dtype=MXF8F6F4_e4m3_X_e5m2(58) vec=12345 tag=MXF8F6F4_e4m3_X_e5m2 my=0x00000001 nv=0x00000002 seed_mode=per_vec base_seed=1
Direct repro: ./all_fmac_test --dtype 58 --seed 1 --seed-mode per_vec --vec-start 12345 --vec-count 1 --stop-on-mismatch
MXF8F6F4_e4m3_X_e5m2 Format test vec[0, 1000) first mismatch at vec 12345, mismatch_count=1
```

`vec` is the failing vector, `my` the software model result, `nv` the NVIDIA
result, and `Direct repro` a command that reruns only that vector.

### 5.1 Mismatch input block

With `FMAC_DUMP_MISMATCH_INPUT=1`, a mismatch also prints:

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

`a/b` are the input payloads, `c` the accumulator input, `a_physical/b_physical`
the payload actually fed to the datapath, and `a_scale/b_scale` the scale
inputs (empty for non-scale dtypes).

### 5.2 FP16 output and 128-bit input

```bash
# fp16 output (FP16 and f8f6f4 32-56; not 57-81)
./all_fmac_test --dtype 34 --vec-count 1000 --seed 1 --seed-mode per_vec --output-fp16 --stop-on-mismatch

# 128-bit topology input
TOPOLOGY_128_INPUT=1 ./all_fmac_test --dtype 58 --vec-count 1000 --seed 1 --seed-mode per_vec --stop-on-mismatch
```

## 6. Output files

When built with `FMAC_DUMP_OUTPUT=0`, none of these files are written; the
program still prints mismatch summaries.

| Directory | Contents |
|---|---|
| `output_data/` | `a/b/c/d` inputs and outputs |
| `scale_output_data/` | `a_scale/b_scale` inputs, plus the `data_for_no_scale.bin.log` placeholder for non-scale dtypes |
| `debug_data/` | intermediate traces (section 7) |

### 6.1 File names

File stems follow:

```text
<shape>_<a_token>_<b_token>_to_<accumulator>
```

for example `m16n8k16_fp16_fp16_to_fp16_*`, `m16n8k32_e4m3_e2m3_to_fp16_*`,
`m16n8k32_mxe4m3_mxe5m2_to_fp32_*`, `m8n8k16_s8_u8_to_s32_*`. Each of `a/b/c/d`
can produce:

- `*_double.bin.log`: binary payload; use this for scripts and replay
- `*.txt.log`: decoded text
- `*_rawhex.txt.log`: hex and raw bits; easiest to read by hand

With the default `FMAC_DUMP_ALL=0`, only `a/b/c` `*_double.bin.log`, the scale
`*_double.bin.log` files and `d` `*.txt.log` are written.

Special cases:

- `mxf4nvf4` (dtypes `9/10`) prefixes record the `--blocksize16` policy: `_b16` when on, `_b32` when off, for example `m16n8k64_mxfp4_mxfp4_to_fp32_b32_a_double.bin.log`. Their `a.txt.log / b.txt.log` write one fp4 value per line.
- Custom `--int4-k N` adds `_k<N>`.
- `TOPOLOGY_128_INPUT=1` with dtypes `1/2/3` uses the `m16n8k8` / `m16n8k4` shape in the prefix.

### 6.2 Scale files

Scale files use the same stem as the `a/b` files and are written by the number
of valid scales, not the number of physical slots (see section 4.3). Every scale
file also has a `*_raw_order.*` copy in the original scale order seen by the
hardware input side; downstream tools should prefer it. The plain scale files
keep the older reversed order.

## 7. Intermediate traces

Traces need a build with `FMAC_DUMP_ALL=1` and the `FMAC_TRACE_INTERMEDIATE=1`
environment variable:

```bash
make build NVIDIA_ON=1 FMAC_DUMP_ALL=1
FMAC_TRACE_INTERMEDIATE=1 FMAC_DUMP_MISMATCH_INPUT=1 ./all_fmac_test \
  --dtype 58 --seed 1 --seed-mode per_vec --vec-start 12345 --vec-count 1 --stop-on-mismatch
```

They are written to `debug_data/` (or `FMAC_TRACE_DIR`):

- `mant_align_*.txt.log`: the alignment step
  - `exp_max`: the maximum exponent chosen for the accumulation
  - `exp_ab`: product exponent
  - `exp_c`: C input exponent
  - `rsft_amt`: right-shift amount for alignment
  - `mant_mul`: product mantissa
  - `mant_shifted`: mantissa after alignment
  - `discarded`: bits shifted out
  - `guard/sticky/lsb`: rounding information
  - `signed_mant`: the signed value entering the accumulation
- `mant_sum_*.txt.log`: the accumulation
  - `partial sum`: the products split evenly into four groups in their original order, without `c`
  - `complete sum`: the full sum including `c`
  - intermediate values before and after the final pack

## 8. Replay

Replay one vector from existing dump files:

```bash
./all_fmac_test --dtype 58 --replay-vec 12345
```

`--replay-vec` needs `--dtype`, and the dtype's files must already exist in
`output_data/` and `scale_output_data/`.

## 9. Batch generation and NV regression

Generate data for several dtypes, each into its own output directory:

```bash
make gen_data DTYPES="1 34 57 98" VEC_COUNT=1000 GEN_ARGS="--output-fp16"
```

Default variables: `GEN_OUTPUT_DATA_DIR=test_output_data`,
`GEN_TRACE_FILE=test_debug_data`, `GEN_SCALE_OUTPUT_DATA_DIR=test_scale_output_data`,
`ACCUM_OUTPUT=fp32` (or `fp16`), `GEN_ARGS` (extra arguments), and
`GEN_ENV=TOPOLOGY_128_INPUT=1`, so `gen_data` generates 128-bit topology input
unless you override `GEN_ENV=`. `VEC_COUNT` defaults to `10000000` for every
Makefile target; set it explicitly for short runs. `GEN_SCALE_OUTPUT_DATA_DIR` sets the
location of both the scale files and the `data_for_no_scale.bin.log`
placeholder. `ACCUM_OUTPUT=fp16` appends `--output-fp16` automatically:

```bash
make gen_data ACCUM_OUTPUT=fp16
make test_128b_trace ACCUM_OUTPUT=fp16
make test_128b_reply ACCUM_OUTPUT=fp16
```

NV regressions (need a visible GPU; every `test_nv_*` target runs
`make check_gpu` first and then builds with `NVIDIA_ON=1` itself):

```bash
make test_nv_fp32_128b VEC_COUNT=1000000
make test_nv_fp32_256b VEC_COUNT=1000000
make test_nv_fp16_128b VEC_COUNT=1000000
make test_nv_fp16_256b VEC_COUNT=1000000
make show_test_logs
```

- `test_nv_fp32_*` use `TEST_NV_FP32_GROUPS`; `test_nv_fp16_*` use only the `DTYPES_FP16_*` groups and always pass `--output-fp16`.
- The groups run in parallel (`TEST_NV_FP32_128B_JOBS=27`, `TEST_NV_FP32_256B_JOBS=27`).
- `test_nv_*_128b` run with `TOPOLOGY_128_INPUT=1`.
- `test_nv_*_256b` pass `--bitsize 256`; override `TEST_NV_256B_ARGS` for another width.
- Logs go to `fmac_test_nv_fp32_128b_logs`, `fmac_test_nv_fp32_256b_logs`, `fmac_test_nv_fp16_128b_logs` and `fmac_test_nv_fp16_256b_logs`. These targets set `FMAC_DUMP_MISMATCH_INPUT=1`, so mismatch inputs go into the group logs.
- By default (`TEST_NV_DUMP=0`) only group logs are written. To keep dump files, add `TEST_NV_DUMP=1`, optionally with custom directories and `FMAC_DUMP_ALL=1`:

```bash
make test_nv_fp32_128b VEC_COUNT=1000000 TEST_NV_DUMP=1 \
  OUTPUT_DATA_DIR=debug_output_128b \
  TRACE_FILE=debug_trace_128b \
  SCALE_OUTPUT_DATA_DIR=debug_scale_128b
```

`make show_test_logs` judges each `group_*.log` as a whole: any
`first mismatch at vec`, `results mismatch` or nonzero `mismatch_count` makes
the group `FAIL`; it is `PASS` only when none appear and
`finished with no mismatches` is present.

## 10. Environment variables

- `FMAC_TRACE_INTERMEDIATE=1`: write intermediate `mant_*` traces (needs a `FMAC_DUMP_ALL=1` build)
- `FMAC_TRACE_DIR=<dir>`: trace output directory; defaults to the compile-time `TRACE_FILE` (`debug_data/`)
- `FMAC_DUMP_MISMATCH_INPUT=1`: print the `FMAC_MISMATCH_INPUT_BEGIN/END` block on a mismatch
- `FMAC_VALIDATE_REFERENCE=1`: same as `--validate-reference`
- `TOPOLOGY_128_INPUT=1`: 128-bit logical payload mode
- `FP4_TOPOLOGY_128=1`: 128-bit topology input mode for FP4

## 11. Debug workflow

1. Run a short test of one dtype with `--stop-on-mismatch` and `FMAC_DUMP_MISMATCH_INPUT=1`.
2. Rerun only the failing vector with the printed `Direct repro` command (`--vec-start <vec> --vec-count 1`).
3. Save the mismatch input block if you need to analyze it with a script.
4. Rebuild with `FMAC_DUMP_ALL=1` and rerun with `FMAC_TRACE_INTERMEDIATE=1`.
5. Compare `my` and `nv`, then check in `mant_align` which term lost bits during alignment, then check the partial sums in `mant_sum`, and finally look at the pack stage for subnormal, underflow or rounding effects.

## 12. Notes

- The default output is `fp32`; `fp16` needs `--output-fp16` and is not supported for `57-81`.
- `--replay-vec` needs the target dtype's dump files.
- With `NVIDIA_ON=0` the `nvmac` reference pass is skipped unless `--validate-reference` is given.
- Plain `make build` / `make gen_data` write dump files by default; `make test_nv_*` write only group logs unless `TEST_NV_DUMP=1`.
