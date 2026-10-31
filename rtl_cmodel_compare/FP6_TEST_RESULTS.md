# Non-scale FP6 integration test results

Date: 2026-10-31

Result: **22/22 RTL data types passed the smoke test and the 10-million-vector
random regression per type (220,000,000 regression vectors total)**.

## Changes verified

- `top.sv` carries all five opcode bits between pipe0 and pipe1, including
  non-scale FP6 E3M2 (20) and E2M3 (21).
- The modified pipe0/pipe1 sources are synchronized into `rtl/fpe/rtl`.
- All nine `rtl/pipe*/defines.sv` copies match `rtl/fpe/rtl/defines.sv`.
- RTL opcodes 20/21 use the existing C-model mixed-float implementations
  50/44, respectively; the C-model arithmetic source is unchanged.
- Default smoke/regression and parallel-run dtype lists cover opcodes 0-21.

## Smoke configuration and reproduction

Run from the repository root:

```bash
python3 rtl/sync_pipe_to_fpe.py --check
make -C rtl_cmodel_compare smoke SMOKE_COUNT=10 SEED=1 \
  TIMEOUT_CYCLES=3000 ULP_TOLERANCE=0
```

- RTL `ReduceWidth`: 512 bits; C-model input: 512 bits.
- Random seed: 1, with `--seed-mode per_vec`.
- Ten vectors per opcode; zero ULP tolerance; stop on first mismatch.
- Verilator 5.048; Apple clang 21.0.0; C++11, `-O2`, `NVIDIA_ON=0`.
- Golden values come from the custom C model; this run does not invoke
  NVIDIA MMA or require CUDA.
- For all non-scale formats, the existing testbench keeps both scale-valid
  signals low and randomizes the scale buses during input transfers. This
  also exercises the new FP6 formats without a scale handshake.

## Smoke results

| RTL opcode | Format | C-model dtype | Vectors | Result |
| --- | --- | --- | --- | --- |
| 0 | I8I8I32 | 0 | 10 | PASS |
| 1 | F16F16F32 | 1 | 10 | PASS |
| 2 | BF16BF16F32 | 2 | 10 | PASS |
| 3 | TF32TF32F32 | 3 | 10 | PASS |
| 4 | I8U8I32 | 4 | 10 | PASS |
| 5 | U8I8I32 | 5 | 10 | PASS |
| 6 | U8U8I32 | 6 | 10 | PASS |
| 7 | MXFP8E4M3F32 | 57 | 10 | PASS |
| 8 | MXFP8E5M2F32 | 63 | 10 | PASS |
| 9 | NVFP4F32 | 9 | 10 | PASS |
| 10 | MXFP4F32 | 10 | 10 | PASS |
| 11 | FP8E4M3F32 | 32 | 10 | PASS |
| 12 | FP8E5M2F32 | 38 | 10 | PASS |
| 13 | MXFP6E3M2F32 | 75 | 10 | PASS |
| 14 | MXFP6E2M3F32 | 69 | 10 | PASS |
| 15 | I4I4I32 | 15 | 10 | PASS |
| 16 | I4U4I32 | 16 | 10 | PASS |
| 17 | U4I4I32 | 17 | 10 | PASS |
| 18 | U4U4I32 | 18 | 10 | PASS |
| 19 | MXINT4F32 | 19 | 10 | PASS |
| 20 | FP6E3M2F32, no scale | 50 | 10 | PASS |
| 21 | FP6E2M3F32, no scale | 44 | 10 | PASS |

Synchronization check: PASS. Diff whitespace check, accounting for the
existing CRLF RTL files (`git -c core.whitespace=cr-at-eol diff --check`): PASS.

## 10-million-vector random regression

Run from the repository root:

```bash
make -C rtl_cmodel_compare debug-mismatch-dtypes \
  VEC_COUNT=10000000 BATCH_COUNT=100000 SEED=1 \
  CMODE_BITSIZE=512 RTL_BITSIZE=512 \
  TIMEOUT_CYCLES=3000 ULP_TOLERANCE=0 DEBUG_DIR=mismatch_debug_10m
```

- All opcodes 0-21 passed 10,000,000 vectors each, using seed 1, 512-bit
  operands, and zero ULP tolerance.
- The completed log was checked for all 2,200 consecutive batch intervals
  and their RTL PASS counts: 100 batches of 100,000 vectors per opcode,
  covering `[0, 10000000)` without gaps or repeated batch intervals.
- Each opcode had a final PASS marker; the overall final marker was:

```text
PASS no mismatch found for DTYPES=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
```

The verified `regression_10m.log` was 4,517,162 bytes, with SHA-256:

```text
5cba9933e0177d0730317fe593a9b36430b33c0021caa230754739670016da33
```

## Cleanup and scope

- Generated simulator builds, smoke/run vectors, regression search data, and
  logs were removed after checking the completed regression. This report
  retains the results and reproduction commands; generated files are not
  included in the source commit.
- The smoke build emitted non-fatal width warnings in unchanged arithmetic
  modules.
- These tests cover the 512-bit standalone RTL/custom C-model comparison.
  They do not establish exhaustive numerical coverage or verify the external
  CUTE/SDK integration.
