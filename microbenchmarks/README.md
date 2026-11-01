# RTX 5090 Tensor Core microbenchmarks

Directed microbenchmarks that characterize the FP32-output floating-point MMA arithmetic of the NVIDIA RTX 5090
(Blackwell, sm_120). They provide the GPU evidence for the *Arithmetic Characterization* section of

> **CUTE-TensorCore: A Bit-Accurate and Hardware-Efficient Tensor Core Design Aligned with NVIDIA Blackwell GPU
> Arithmetic**

Every probe issues the MMA instruction directly as inline PTX, builds operands from raw bit patterns, and checks
the raw output bits of D against the value predicted by the rule under test. No library (cuBLAS, CUTLASS) is
involved, and no reference model is used to produce expected values.

## Requirements

- NVIDIA RTX 5090 or another sm_120 GPU. The `kind::*` and `block_scale` instructions need the
  architecture-specific target `sm_120a`.
- CUDA Toolkit 13.1 or newer (`nvcc`), GNU make, bash, Linux.
- No other dependencies.

The reference results were produced with driver 590.44.01 and CUDA 13.1
([environment](results/rtx5090/environment.txt)).

## Quick start

```sh
make                # builds build/selfcheck and build/test1 ... build/test9
./run_all.sh        # runs everything, writes logs to results/local/
```

`run_all.sh` prints one line per program and exits with 0 only if every check passes:

```
selfcheck  PASS    19s  selfcheck: 16 checks, 16 passed, 0 failed -> PASS
test1      PASS    10s  test1: 48 checks, 48 passed, 0 failed -> PASS
test2      PASS     1s  test2: 26 checks, 26 passed, 0 failed -> PASS
test3      PASS     0s  test3: 15 checks, 15 passed, 0 failed -> PASS
test4      PASS     0s  test4: 32 checks, 32 passed, 0 failed -> PASS
test5      PASS     0s  test5: 372 checks, 372 passed, 0 failed -> PASS
test6      PASS     1s  test6: 6 checks, 6 passed, 0 failed -> PASS
test7      PASS     0s  test7: 40 checks, 40 passed, 0 failed -> PASS
test8      PASS     0s  test8: 22 checks, 22 passed, 0 failed -> PASS
test9      PASS     2s  test9: 6 checks, 6 passed, 0 failed -> PASS
```

A single program can be run on its own, for example `./build/test7`. `test1` and `test9` take the number of
random rounds as an optional argument (defaults 1,000,000 and 100,000). All random inputs use fixed seeds, so a
rerun on the same hardware reproduces the reference logs in [`results/rtx5090/`](results/rtx5090/).

`selfcheck` runs first. It verifies the register layout, slot placement and scale-byte mapping of all 16
instructions against an exact reference on random matrices. If it fails, `run_all.sh` stops, because every later
result depends on it.

## Tests

| Test | Question | Paper statement | Result on RTX 5090 |
|---|---|---|---|
| [test1](tests/test1_output_uniformity/) | Do all 128 output elements follow the same rule? | "After confirming that the output elements of an MMA operation are computed independently, we reduce the characterization to a single output element" | All 16 instructions: 128 elements bit-identical in 2 × 10^6 random rounds each |
| [test2](tests/test2_accumulation_order/) | How are the K products summed? | "Cancellation cases expose the reduction organization"; on the dedicated path "products are first reduced within scale-associated groups" | Common path fully fused; MXFP4 and NVFP4 sum in groups of 16 |
| [test3](tests/test3_hidden_precision/) | How many fractional bits survive alignment? | "truncated to 25 fractional bits"; the dedicated path "uses 35 fractional bits" | p = 25 on I1–I13, p = 35 on MXFP4 and NVFP4 |
| [test4](tests/test4_rounding/) | How is the FP32 output rounded? | rounding-boundary inputs isolate rounding | RZ on all 16 instructions |
| [test5](tests/test5_special_values/) | How are NaN, ±∞, overflow and ±0 handled? | special values isolate exception behavior | NaN rules confirmed; canonical NaN 0x7FFFFFFF; overflow → +∞; never −0 |
| [test6](tests/test6_scale_order/) | Is the NVFP4 scale applied before the cross-group alignment? | group sums are "multiplied by scale mantissas; the group sums and accumulator are aligned" | Scale first, then alignment (all 10 distinguishing cases) |
| [test7](tests/test7_exponent_floor/) | Is there a lower bound on the alignment exponent? | "Blackwell lower-bounds the effective alignment exponent at −133 on the common path and at −139 on the MXFP4/NVFP4 dedicated path" | −133 (TF32, BF16, `mxf8f6f4`), −139 (`mxf4`, `mxf4nvf4`) |
| [test8](tests/test8_zero_exponent/) | Is a structural zero aligned like a cancellation? | (rule of the reference model) | Structural zeros and zero UE4M3 scales get the minimum exponent; cancellations keep theirs |
| [test9](tests/test9_fp4_fp6_low_bits/) | Are bits [1:0] of an E2M1 slot ignored? | (rule of the reference model) | No: E2M1 is decoded as E2M3 `00_e2m1_00` |

[CLAIMS.md](CLAIMS.md) maps each statement of the paper to the cases and log lines that support it.
[docs/arithmetic_flow.md](docs/arithmetic_flow.md) puts all inferred rules together in computation order.

## Instructions under test

16 FP32-output floating-point instructions, listed with their PTX, register layout and encodings in
[docs/instructions.md](docs/instructions.md):

| IDs | Instructions |
|---|---|
| I1–I3 | TF32 m16n8k8, FP16 m16n8k16, BF16 m16n8k16 |
| I4–I8 | `kind::f8f6f4` m16n8k32: E5M2, E4M3, E3M2, E2M3, E2M1 |
| I9–I13 | `kind::mxf8f6f4` `scale_vec::1X` m16n8k32 (UE8M0): E5M2, E4M3, E3M2, E2M3, E2M1 |
| I14 | `kind::mxf4` `scale_vec::2X` m16n8k64 (MXFP4, UE8M0) |
| I15 | `kind::mxf4nvf4` `scale_vec::2X` m16n8k64 (MXFP4, UE8M0) |
| I16 | `kind::mxf4nvf4` `scale_vec::4X` m16n8k64 (NVFP4, UE4M3) |

I1–I13 form the common path, I14–I16 the dedicated FP4 path. [docs/data_types.md](docs/data_types.md) explains the
choice of shapes and how the instructions map to the formats of the RTL.

## Layout

```text
microbenchmarks/
├── README.md               this file
├── CLAIMS.md               paper statement → test → evidence
├── Makefile                builds build/selfcheck and build/test1 ... build/test9
├── run_all.sh              runs all programs, writes logs, prints PASS/FAIL
├── common/
│   ├── formats.hpp         encode/decode of element and scale formats (host)
│   ├── mma.cuh             the 16 PTX instructions, fragment layout, probe kernels
│   └── harness.cuh         problem construction, batched execution, checks
├── selfcheck/              layout and scale-mapping self-check
├── tests/
│   └── testN_<name>/
│       ├── README.md       method, inputs, decision rule, results, conclusion
│       └── testN.cu        the probe
├── docs/
│   ├── instructions.md     PTX, register layout, encodings, scale registers
│   ├── data_types.md       scope, K dimension, mapping to the RTL formats
│   └── arithmetic_flow.md  the inferred arithmetic rules in computation order
└── results/rtx5090/        reference logs and environment of the published run
```

## Scope and limits

- Only floating-point inputs with FP32 output are covered. Integer modes and FP16 output are not part of these
  microbenchmarks.
- The tests characterize observable outputs. They do not prove a particular internal circuit; the conclusions are
  the simplest rules consistent with every observation.
- Some statements cannot be probed on every instruction. For example, the exponent floor is unreachable on
  instructions whose products cannot go below −133 (test7); each test document lists such cases.
- The randomized comparisons of the paper (100M tests per format between the RTX 5090 and the C reference model,
  1M per format between the C model and the RTL) are separate validation steps and live in `../cmodel/` and
  `../rtl_cmodel_compare/`.
