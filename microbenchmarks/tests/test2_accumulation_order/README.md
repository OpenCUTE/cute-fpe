# test2: summation order and grouping

Source: [`test2.cu`](test2.cu) · Reference log: [`results/rtx5090/test2.log`](../../results/rtx5090/test2.log)

## Goal

Determine how the K products and C are summed: fully fused, in groups (and how large), or sequentially.

| Method | Instructions |
|---|---|
| Method 1 | TF32, FP16, BF16, E5M2 / E4M3 with and without MX (I1–I5, I9, I10) |
| Method 2 | E3M2, E2M3, E2M1 with and without MX (I6–I8, I11–I13) |
| Method 3 | MXFP4 on the dedicated path (I14, I15) |

In all methods only A[0][:], B[:][0] and C[0][0] matter; D[0][0] is observed. Encodings are listed in
[instructions.md](../../docs/instructions.md).

## Method 1 (I1–I5, I9, I10)

### 1.1 Coarse filter

Setup: A = B = 1 everywhere, so all K products are 1. C starts at 2^(23+log2 K).

1. Run the MMA and compare D[0][0] with C.
2. If D ≠ C, record g = 2^(log2(C) − 23). g = K would mean fully fused summation; g < K needs 1.2.
3. If D = C, halve C and repeat. If C drops below 2^24 without a change, g = 1 and 1.2 is needed.

FP32 has 24 significand bits, so the ulp of C is 2^(log2(C) − 23). g is the ulp at which the products start to be
added, which is not necessarily the group size:

- If only FP32 output precision mattered, a group sum would be added exactly once ulp ≤ group sum, and g would be
  the group size.
- But each product is first aligned to the largest term (here C), and anything more than p bits below it is
  truncated (p is the accumulation precision, see test3). As long as log2(C) > p, a product of 1 is lost during
  alignment and D = C, whatever the grouping.

The first D ≠ C therefore appears at C = 2^min(23 + log2 K, p). When 23 + log2 K > p, 1.1 measures p, not the
group size.

### 1.2 Grouped summation versus fused summation with limited alignment

Setup:

- X = 2^8, y = 2^-6.
- A[0][k1] = X, A[0][k2] = −X, B[k1][0] = B[k2][0] = X: two products ±2^16.
- All other A and B entries are y, so every other product is y² = 2^-12.
- C[0][0] = y² = 2^-12.
- All ordered pairs k1 ≠ k2.

Decision:

- D = 0 for every pair: fully fused. The small products are 28 bits below 2^16, beyond p, and are truncated
  during alignment. The result of 1.1 is caused by the alignment precision.
- D ≠ 0 for some pair: grouped. Groups without ±X are summed first and survive. n = D / y² is the number of kept
  small products, and the group size follows from K − n.

### Results (RTX 5090)

| Instruction | K | 1.1: first D ≠ C | 1.1: g | 1.2: pairs with D = 0 | Conclusion |
|---|---|---|---|---|---|
| I1 TF32 | 8 | C = 2^25, D = 2^25 + 8 | 4 | 56 / 56 | fully fused |
| I2 FP16 | 16 | C = 2^25, D = 2^25 + 16 | 4 | 240 / 240 | fully fused |
| I3 BF16 | 16 | C = 2^25, D = 2^25 + 16 | 4 | 240 / 240 | fully fused |
| I4 E5M2 | 32 | C = 2^25, D = 2^25 + 32 | 4 | 992 / 992 | fully fused |
| I5 E4M3 | 32 | C = 2^25, D = 2^25 + 32 | 4 | 992 / 992 | fully fused |
| I9 MX E5M2 | 32 | C = 2^25, D = 2^25 + 32 | 4 | 992 / 992 | fully fused |
| I10 MX E4M3 | 32 | C = 2^25, D = 2^25 + 32 | 4 | 992 / 992 | fully fused |

1.1 always stops at C = 2^25, which is set by p = 25, not by the grouping.

## Method 2 (I6–I8, I11–I13)

Setup:

- B[:][0] = 1, other columns 0.
- C[0][0] = 2^-24.
- Scaled instructions use scale 1.0.
- A[0][k1] = 1, A[0][k2] = −1, all other A = 0; all ordered pairs k1 ≠ k2.

The exact result is 1 − 1 + 2^-24 = 2^-24. If C were first added to one of the ±1 and rounded to FP32, 1 + 2^-24
would round to 1 and the result would be 0.

| Instruction | Pairs with D[0][0] = 2^-24 | Conclusion |
|---|---|---|
| I6, I7, I8 (`kind::f8f6f4`) | 992 / 992 each | fully fused |
| I11, I12, I13 (`kind::mxf8f6f4` 1X) | 992 / 992 each | fully fused |

Strictly, this only shows that ±1 cancel before they meet C. A design that sums all products first and adds C at
the end would also pass.

## Method 3 (I14, I15)

### Background

- NVFP4 (`scale_vec::4X`) has one scale per 16 elements, which suggests four groups of 16.
- MXFP4 (`scale_vec::2X`) has one scale per 32 elements. The question is whether it sums in two groups of 32 or in
  four groups of 16.
- From test3, values up to 35 bits below the largest exponent are kept, values 36 bits below are lost. The
  parameters below are chosen at this boundary.

### Setup

| Operand | Setting |
|---|---|
| B | column 0 is 1, other columns 0; scale_B = 1.0 |
| C | 0 |
| A | A[0][0] = 1, A[0][1] = −1; A[0][p1] = A[0][p2] = 1 for every 32 ≤ p1 < p2 ≤ 63 (496 pairs); other entries 0 |
| scale_A | k = 0–31: 2^36; k = 32–63: 2^0 |

### Principle

- The first block gives ±2^36. They cancel, but alignment still uses 2^36.
- In the second block every product is 2^0, which alone is 36 bits below 2^36 and is lost.
- If both 1s are in the same group, they are summed first to 2^1, which is 35 bits below and survives.

So D[0][0] = 2 means p1 and p2 are in the same group, and D[0][0] = 0 means they are not.

### Results (RTX 5090)

| Position of p1, p2 | D[0][0] | Pairs |
|---|---|---|
| both in 32–47 | 2 | 120 |
| both in 48–63 | 2 | 120 |
| one in 32–47, one in 48–63 | 0 | 256 |

- Control: with scale_A = 2^37 for k = 0–31, even a same-group sum 2^1 is 36 bits below; all 496 pairs give 0. The
  result above is an alignment-window effect.
- Reversed: with the pair at A[0][32] = 1, A[0][33] = −1, scale 2^36 on k = 32–63 and probes in k = 0–31, the
  result is the same: D = 2 for 240 pairs inside 0–15 or 16–31, and D = 0 for the 256 pairs across.
- I15 (`kind::mxf4nvf4` `scale_vec::2X`) gives exactly the same counts as I14.

## Conclusion

- The common path (I1–I13) sums the K products fully fused.
- On the dedicated path, MXFP4 (I14 and I15) sums in groups of 16 consecutive elements, k = 0–15, 16–31, 32–47 and
  48–63, like NVFP4.
