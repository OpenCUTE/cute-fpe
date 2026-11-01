# test6: order of scale application (NVFP4)

Source: [`test6.cu`](test6.cu) · Reference log: [`results/rtx5090/test6.log`](../../results/rtx5090/test6.log)

## Goal

NVFP4 shares one scale per 16 elements. The test determines when the hardware multiplies by the scale:

| Model | Order |
|---|---|
| M1: scale first, then align | the exact value group dot product × scale_A × scale_B is aligned with the other groups and C, and the shifted-out bits are truncated |
| M2: align first, then scale significand | alignment and truncation use only the scale exponents; the scale significands are multiplied afterwards |

The models differ only when a scale is not a power of two. UE4M3 has 3 mantissa bits, so it can build such inputs.
UE8M0 is a pure power of two, so the test applies to NVFP4 only.

## Prerequisites

From test2 and test3:

- `kind::mxf4nvf4` `scale_vec::4X` sums in groups of 16 consecutive elements: k = 0–15, 16–31, 32–47, 48–63.
- Values within 35 bits below the largest exponent are kept; lower bits are truncated.

When the largest value is 2^24, the smallest unit after alignment is LSB = 2^(24−35) = 2^-11, and the fraction is
truncated toward zero to multiples of 2^-11.

## Instruction

I16: `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`

## Input

### Main experiment

Only groups 1 and 4 of B[:][0] are non-zero. D[0][0] is observed.

| Part | Setting | Contribution |
|---|---|---|
| C | C[0][0] = −2^24, other C = 0 | −2^24 |
| Group 1 (k = 0–15) | A[0][k] = 4, B[k][0] = 4, scale_A0 = scale_B0 = 2^8 | 16 × 16 × 2^16 = +2^24, cancels C |
| Groups 2, 3 (k = 16–47) | A, B = 0, scale 1.0 | 0 |
| Group 4 (k = 48–63) | A[0][48] = A[0][49] = 1, B[48][0] = B[49][0] = 1, other entries 0; scale_A3 = 1.5 × 2^-6, scale_B3 = 2^-6 | 2 × 1.5 × 2^-12 = 1.5 × 2^-11 |

After group 1 and C cancel, the largest exponent is still 2^24, so LSB = 2^-11 and group 4 contributes exactly 1.5
LSB.

Codes (layout and register mapping in [instructions.md](../../docs/instructions.md)):

| Value | Code |
|---|---|
| E2M1 1 / 4 | 0x2 / 0x6 |
| UE4M3 2^8 / 1.0 / 2^-6 / 1.5 × 2^-6 | 0x78 / 0x38 / 0x08 / 0x0C |

Scale registers (byte j = group j+1): scale_A = 0x0C383878, scale_B = 0x08383878.

### Reference run

The same input without group 1 and C must give exactly 1.5 × 2^-11. This confirms the scale encodings.

## Decision

| Model | Computation | Predicted D[0][0] |
|---|---|---|
| M1 | 1.5 × 2^-11, truncated to a multiple of 2^-11 | 2^-11 (one LSB kept) |
| M2 | 2 × 2^-12 = 2^-11 (exactly one LSB, nothing truncated), then × 1.5 | 1.5 × 2^-11 |

Neither model predicts 0, so the comparison must use the exact value.

## Scan

To rule out a coincidence, group 1 and C stay fixed and group 4 varies:

- group dot product n ∈ {1, 2}: the first n elements of group 4 have A = B = 1;
- scale_A3 = m × 2^ea with m ∈ {1, 1.25, 1.5, 1.75} and ea ∈ {−4, −5, −6, −7};
- scale_B3 = 2^-6.

That gives 32 parameter sets. With ref = n × scale_A3 × scale_B3 the models predict

- M1: D = ⌊ref / 2^-11⌋ × 2^-11
- M2: D = ⌊n × 2^ea × 2^-6 / 2^-11⌋ × 2^-11 × m

## Results (RTX 5090)

| Run | D[0][0] |
|---|---|
| reference (group 4 only) | 1.5 × 2^-11 (0x1.8p-11) |
| main experiment | 2^-11 (0x1p-11) |

Scan: all 32 references are exact. M1 matches all 32 sets, M2 matches 22. The 10 sets M2 misses are exactly the 10
sets where the models disagree, so every distinguishing set supports M1. Examples:

| n | scale_A3 | ref | M1 | M2 | Measured |
|---|---|---|---|---|---|
| 2 | 1.5 × 2^-6 | 1.5 × 2^-11 | 2^-11 | 1.5 × 2^-11 | 2^-11 |
| 1 | 1.75 × 2^-4 | 1.75 × 2^-10 | 1.5 × 2^-10 | 1.75 × 2^-10 | 1.5 × 2^-10 |
| 1 | 1.25 × 2^-4 | 1.25 × 2^-10 | 2^-10 | 1.25 × 2^-10 | 2^-10 |

The full table is in the reference log.

## Conclusion

On NVFP4 the scale is multiplied exactly into the group dot product before the group is aligned with the other
groups and C (M1). The scale significand takes part in the truncation; it is not applied after it.
