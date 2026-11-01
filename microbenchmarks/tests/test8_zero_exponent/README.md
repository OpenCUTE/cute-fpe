# test8: structural zero versus cancellation to zero

Source: [`test8.cu`](test8.cu) · Reference log: [`results/rtx5090/test8.log`](../../results/rtx5090/test8.log)

## Goal

Two dot-product terms can both equal 0 in different ways. The test checks whether their exponents are treated the
same during alignment:

| Name | Construction | Example |
|---|---|---|
| structural zero | in every pair a × b at least one factor is 0, but a and b are not both all zero | a = X, b = 0 |
| cancellation to zero | the products are non-zero and only their sum is 0 | X·X + (−X)·X |

Hypotheses:

- **H1 (no scale)**: a product with a zero factor has a zero significand and its exponent is forced to the minimum,
  so it does not take part in alignment. A cancellation keeps the exponents of its products.
- **H2 (UE8M0 scale)**: if all products of a block are structural zeros, the scale does not take part in alignment,
  however large it is; the exponent is set to the minimum. A cancellation such as 1 + (−1) keeps the scale.
- **H3 (NVFP4, groups of 16)**:
  - a group whose products are all structural zeros gets the minimum exponent, so it cannot shift other data out;
  - a group that cancels, such as 1 and −1, keeps the exponent given by its scales;
  - a UE4M3 scale can be 0. If the low 7 bits of scale_A or scale_B are all 0, the group is 0 and is handled like a
    structural-zero group (minimum exponent).

## Principle

From test3: alignment keeps values within p bits below the largest exponent and truncates the rest (p = 35 on
`kind::mxf4` / `kind::mxf4nvf4`, 25 elsewhere).

Build a zero "large term" that is more than p bits above C and check whether C survives:

- if the exponent of the large term takes part in alignment, C is shifted out and D = 0;
- if it is set to the minimum, C is unaffected and D = C.

Structural zeros and cancellations use the same large term; only the way it becomes zero changes.

## Instructions

| Part | Instructions |
|---|---|
| (1) no scale | I1–I8 |
| (2) UE8M0 scale | I9–I15 (`mxf8f6f4` 1X, `mxf4` 2X, `mxf4nvf4` 2X) |
| (3) UE4M3 scale | I16 (NVFP4) |

Only A[0][:], B[:][0] and C[0][0] are set and D[0][0] is observed. Layout, codes and scale registers are in
[instructions.md](../../docs/instructions.md).

## (1) No scale (I1–I8)

The formats have different ranges (E2M1 only reaches 6), so a product of 2^40 is not always possible. Parameters:

- X = 2^e, where e is the largest power-of-two exponent of the format, capped at 20. The large product is
  X·X = 2^(2e).
- C[0][0] = 2^(2e−30), 30 bits below the large product and beyond p = 25.

| Instruction | X | X code | −X code | C[0][0] |
|---|---|---|---|---|
| I1 TF32 | 2^20 | 0x49800000 | 0xC9800000 | 2^10 |
| I2 FP16 | 2^15 | 0x7800 | 0xF800 | 2^0 |
| I3 BF16 | 2^20 | 0x4980 | 0xC980 | 2^10 |
| I4 E5M2 | 2^15 | 0x78 | 0xF8 | 2^0 |
| I5 E4M3 | 2^8 | 0x78 | 0xF8 | 2^-14 |
| I6 E3M2 | 2^4 | 0x1C | 0x3C | 2^-22 |
| I7 E2M3 | 2^2 | 0x18 | 0x38 | 2^-26 |
| I8 E2M1 (slot [5:2]) | 2^2 | 0x18 | 0x38 | 2^-26 |

| Case | A[0][k] | B[k][0] | D[0][0] if H1 holds |
|---|---|---|---|
| Z1 structural zero | a0 = 0 | b0 = X | C |
| Z2 structural zero | a0 = X | b0 = 0 | C |
| Z3 structural zero over all K | even k: a = X; odd k: a = 0 | even k: b = 0; odd k: b = X | C |
| N cancellation | a0 = X, a1 = −X | b0 = b1 = X | 0 |

Results (RTX 5090): for all 8 instructions Z1, Z2 and Z3 return C and N returns 0. H1 holds.

## (2) UE8M0 scale (I9–I15)

- Every byte of scale_A is 2^40 (0xA7), so the register is 0xA7A7A7A7; scale_B is 1.0 (0x7F7F7F7F).
- C[0][0] = 1, 40 bits below 2^40 and beyond p (25 or 35).

| Case | A[0][k], k = 0 … K−1 | B[k][0] | D[0][0] if H2 holds |
|---|---|---|---|
| Z structural zero | all 1 | all 0 | 1 |
| Zmix structural zero | even k: 1, odd k: 0 | even k: 0, odd k: 1 | 1 |
| N cancellation | a1 = −1, all others 1 | b0 = b1 = 1, others 0 | 0 |
| control | only a0 = 1 | only b0 = 1 | 2^40 |

The control confirms that the 2^40 scale is active: 2^40 + 1 rounds (RZ) to 2^40.

Results (RTX 5090): for all 7 instructions Z = 1, Zmix = 1, N = 0 and the control = 2^40. H2 holds.

## (3) NVFP4 (I16)

UE4M3 scales only reach about 2^8, so instead of a 2^40 scale the test uses a tiny C[0][0] = 2^-40. A group whose
exponent is more than 35 bits above C shifts C out.

All cases use group 1 only (k = 0–15). Groups 2–4 have a = b = 0 and scale 1.0 (0x38). Codes: E2M1 1 / −1 / 4 =
0x2 / 0xA / 0x6; UE4M3 1.0 / 2^8 / 2^-9 / 0 / zero with the top bit set = 0x38 / 0x78 / 0x01 / 0x00 / 0x80.

| Case | Group 1 a, b | scale_A | scale_B | Group exponent if it took part | D[0][0] if H3 holds |
|---|---|---|---|---|---|
| R reference | a = b = 0 | 0x38383878 | 0x38383878 | none | 2^-40 |
| Z structural zero | a = 1, b = 0 | 0x38383878 | 0x38383878 | about 2^16, 56 bits above C | 2^-40 |
| N cancellation | a0 = 1, a1 = −1, other a = 1; b0 = b1 = 1, other b = 0 | 0x38383878 | 0x38383878 | 2^16 | 0 |
| S1 scale_A0 = 0 | a = b = 4, dot product 256 | 0x38383800 | 0x38383878 | about 2^7 even if 0 counted as 2^-9, 47 bits above C | 2^-40 |
| S2 scale_A0 = 0x80 | as S1 | 0x38383880 | 0x38383878 | as S1 | 2^-40 |
| S3 scale_B0 = 0 | as S1 | 0x38383878 | 0x38383800 | as S1 | 2^-40 |
| control scale_A0 = 2^-9 | as S1 | 0x38383801 | 0x38383878 | 256 × 2^-9 × 2^8 = 128 | 128 |

The control shows that a small but non-zero scale takes part in alignment and shifts C out. S1–S3 keep C because
the scale is 0, not because it is small.

Results (RTX 5090): R, Z, S1, S2 and S3 return 2^-40, N returns 0 and the control returns 128, as predicted. S2
shows that UE4M3 ignores the top bit: a byte whose low 7 bits are 0 is treated as 0.

## Conclusion

| Instructions | Structural zero | Cancellation to zero |
|---|---|---|
| no scale (I1–I8) | exponent set to the minimum, no effect on alignment | exponents take part in alignment |
| UE8M0 scale (I9–I15) | scale ignored, exponent set to the minimum | scale takes part in alignment |
| NVFP4 (I16) | group exponent set to the minimum; the same for a scale whose low 7 bits are 0 | group aligned with its products and scales |

A NaN scale is the exception: it makes the result NaN even for a structural-zero group (test5).
