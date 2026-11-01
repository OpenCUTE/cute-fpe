# Data types under test

## Scope

- Only floating-point inputs are characterized. Integer modes are out of scope for the microbenchmarks.
- The output (accumulator) format is always FP32.
- The input formats are the floating-point formats supported by the RTL, plus low-precision formats that NVIDIA
  `mma` supports and the RTL does not.

## K dimension

Every instruction uses the shape whose K dimension satisfies

```
K × storage width = 256 bit, i.e. K = 256 / storage width
```

The storage width is the number of register bits one element occupies, which can differ from the width of the
format itself:

| Format width | Storage | Storage width | K |
|---|---|---|---|
| 32 bit | TF32 stored as 32 bit | 32 | 8 |
| 16 bit | packed | 16 | 16 |
| 8 bit | packed | 8 | 32 |
| 6 bit | one FP6 element per 8-bit container | 8 | 32 |
| 4 bit | packed (`kind::mxf4`, `kind::mxf4nvf4`) | 4 | 64 |
| 4 bit | one E2M1 element in the middle 4 bits of an 8-bit container (`kind::f8f6f4`, `kind::mxf8f6f4`) | 8 | 32 |

## RTL formats and the instructions that test them

The RTL encodes the operation type with the opcodes below (`DataType*` defines in the RTL). The instruction IDs
are listed in [instructions.md](instructions.md).

| RTL opcode | RTL name | A × B → C | NVIDIA instruction |
|---|---|---|---|
| 1 | `DataTypeF16F16F32` | FP16 × FP16 → FP32 | I2 |
| 2 | `DataTypeBF16BF16F32` | BF16 × BF16 → FP32 | I3 |
| 3 | `DataTypeTF32TF32F32` | TF32 × TF32 → FP32 | I1 |
| 7 | `DataTypeMxfp8e4m3F32` | MXFP8 E4M3 → FP32 | I10 |
| 8 | `DataTypeMxfp8e5m2F32` | MXFP8 E5M2 → FP32 | I9 |
| 9 | `DataTypenvfp4F32` | NVFP4 → FP32 | I16 |
| 10 | `DataTypemxfp4F32` | MXFP4 → FP32 | I14, I15 (packed, K = 64); I13 (8-bit container, K = 32) |
| 11 | `DataTypefp8e4m3F32` | FP8 E4M3 → FP32 | I5 |
| 12 | `DataTypefp8e5m2F32` | FP8 E5M2 → FP32 | I4 |
| 13 | `DataTypeMxfp6e3m2F32` | MXFP6 E3M2 → FP32 | I11 |
| 14 | `DataTypeMxfp6e2m3F32` | MXFP6 E2M3 → FP32 | I12 |

RTL opcodes that are not tested:

| RTL opcode | RTL name | Reason |
|---|---|---|
| 0, 4, 5, 6 | `DataTypeI8I8I32`, `DataTypeI8U8I32`, `DataTypeU8I8I32`, `DataTypeU8U8I32` | integer |
| 15–18 | `DataTypeI4I4I32`, `DataTypeI4U4I32`, `DataTypeU4I4I32`, `DataTypeU4U4I32` | integer |
| 19 | `DataTypemxint4F32` | block-scaled integer, not supported by NVIDIA `mma` |

## Additional NVIDIA formats

These formats have no RTL opcode and are tested as additional reference points:

| Format | Instruction |
|---|---|
| FP6 E3M2 / E2M3 without scale | I6, I7 (`kind::f8f6f4`) |
| FP4 E2M1 without scale | I8 (`kind::f8f6f4`) |
| MXFP4 on the dedicated path with `kind::mxf4nvf4` | I15 |

Notes on the correspondence:

- FP8, MXFP8 and MXFP6 match the RTL formats one-to-one, with the same storage width and K.
- NVFP4 matches `DataTypenvfp4F32`. Both sides pack the elements and use K = 64.
- MXFP4 exists in both, but the storage differs. The RTL packs it (K = 64), like I14 and I15. In
  `kind::mxf8f6f4` (I13) NVIDIA stores it in an 8-bit container (K = 32).
