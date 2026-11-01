# Instructions, register layout and encodings

All probes use the 16 FP32-output floating-point `mma.sync` instructions below, issued directly as inline PTX by
[`common/mma.cuh`](../common/mma.cuh). Every probe runs one warp (32 threads) per MMA. The test documents refer to
the instructions by their ID.

## Instructions

| ID | Input format | Scale | Shape | PTX |
|---|---|---|---|---|
| I1 | TF32 | none | m16n8k8 | `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` |
| I2 | FP16 | none | m16n8k16 | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |
| I3 | BF16 | none | m16n8k16 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` |
| I4–I8 | E5M2, E4M3, E3M2, E2M3, E2M1 | none | m16n8k32 | `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.<t>.<t>.f32` |
| I9–I13 | E5M2, E4M3, E3M2, E2M3, E2M1 | UE8M0, 1 per 32 | m16n8k32 | `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.<t>.<t>.f32.ue8m0` |
| I14 | E2M1 (MXFP4) | UE8M0, 1 per 32 | m16n8k64 | `mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0` |
| I15 | E2M1 (MXFP4) | UE8M0, 1 per 32 | m16n8k64 | `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0` |
| I16 | E2M1 (NVFP4) | UE4M3, 1 per 16 | m16n8k64 | `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` |

`<t>` is `e5m2`, `e4m3`, `e3m2`, `e2m3` or `e2m1`. I1–I13 form the **common path**; I14–I16 (`kind::mxf4` and
`kind::mxf4nvf4`) form the **dedicated FP4 path** (see [arithmetic_flow.md](arithmetic_flow.md)).

The `a` suffix of the target is required: `kind::*` and `block_scale` instructions exist only on the
architecture-specific feature set.

```
nvcc -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a ...
```

## Fragment layout

Let `g = lane >> 2` and `t = lane % 4`. Each 32-bit register holds `e` elements: 1 for TF32, 2 for 16-bit formats,
4 for 8-bit slots (FP8, FP6 and E2M1 on I4–I13), 8 for packed E2M1 (I14–I16). Element `i` of a register occupies
bits `[(32/e)(i+1) − 1 : (32/e)i]`.

| Operand | Registers per lane | Register `r`, element `i` holds |
|---|---|---|
| A (16×K) | `a[0..3]` | row `g + 8·(r % 2)`, column `e·t + i + (K/2)·(r / 2)` |
| B (K×8) | `b[0..1]` | row `e·t + i + (K/2)·r`, column `g` |
| C, D (16×8) | `c[0..3]` / `d[0..3]` | row `g + 8·(r / 2)`, column `2t + (r % 2)` |

`A[0][k]` and `B[k][0]` live only in lanes 0–3. D[0][0] is `d[0]` of lane 0. The layout is verified for every
instruction by [`selfcheck`](../selfcheck/selfcheck.cu): random exactly-representable matrices must reproduce an
exact double-precision reference in every element.

## Element slots

| Format | Slot width | Position of the raw encoding |
|---|---|---|
| TF32 | 32 | FP32 bit pattern; the low 13 bits are ignored |
| FP16, BF16 | 16 | bits [15:0] of the slot |
| E5M2, E4M3 | 8 | bits [7:0] |
| E3M2, E2M3 | 8 | bits [5:0]; bits [7:6] are 0 |
| E2M1 on I8, I13 | 8 | bits [5:2]; the other bits are 0 (see test9 for bits [1:0]) |
| E2M1 on I14–I16 | 4 | packed, bits [3:0] of a nibble |

## Element encodings used by the probes

| Value | TF32 | FP16 | BF16 | E5M2 | E4M3 | E3M2 | E2M3 | E2M1 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0x3F000000 | 0x3800 | 0x3F00 | 0x38 | 0x30 | 0x08 | 0x04 | 0x1 |
| 1.0 | 0x3F800000 | 0x3C00 | 0x3F80 | 0x3C | 0x38 | 0x0C | 0x08 | 0x2 |
| 1.5 | 0x3FC00000 | 0x3E00 | 0x3FC0 | 0x3E | 0x3C | 0x0E | 0x0C | 0x3 |
| 4.0 | 0x40800000 | 0x4400 | 0x4080 | 0x44 | 0x48 | 0x14 | 0x18 | 0x6 |
| −0 | 0x80000000 | 0x8000 | 0x8000 | 0x80 | 0x80 | 0x20 | 0x20 | 0x8 |
| +Inf | 0x7F800000 | 0x7C00 | 0x7F80 | 0x7C | – | – | – | – |
| NaN | 0x7FC00000 | 0x7E00 | 0x7FC0 | 0x7E | 0x7F | – | – | – |

Negative values set the sign bit (bit 31, 15, 7, 5, 5 or 3). On I8 and I13 the E2M1 code is shifted left by 2
before it is written into the slot (for example 1.0 becomes 0x08). E2M3 0.5 and E2M1 0.5 are subnormal.

All other values are encoded exactly by [`common/formats.hpp`](../common/formats.hpp), which aborts if a value is
not representable.

## Scale registers

Every thread writes the same 32-bit scale register for A and for B, and both `byte-id` and `thread-id` selectors
are 0. Byte `j` of the register holds the scale of K-group `j`:

| Instructions | Group width | Bytes used |
|---|---|---|
| I9–I13 (1X) | 32 | byte 0 (the probes write the same value into all four bytes) |
| I14, I15 (2X) | 32 | bytes 0 and 1 (bytes 2 and 3 repeat them) |
| I16 (4X) | 16 | bytes 0–3 |

| Scale format | 1.0 | Other values | NaN |
|---|---|---|---|
| UE8M0 | 0x7F | 2^x = 127 + x (2^-127 = 0x00 … 2^127 = 0xFE) | 0xFF |
| UE4M3 | 0x38 | 2^8 = 0x78, 448 = 0x7E, 2^-6 = 0x08, 2^-9 = 0x01, 1.5·2^-6 = 0x0C, 0 = 0x00 | 0x7F (and 0xFF) |

A scale register of 1.0 is therefore 0x7F7F7F7F (UE8M0) or 0x38383838 (UE4M3). "scale_A0 = x" in the test
documents means that only byte 0 is replaced by x.

## FP32 codes for C and D

| Value | Bits |
|---|---|
| +0 / −0 | 0x00000000 / 0x80000000 |
| +Inf / −Inf | 0x7F800000 / 0xFF800000 |
| quiet NaN written into C | 0x7FC00000 |
| NaN produced by the hardware | 0x7FFFFFFF |
