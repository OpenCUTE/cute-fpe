# test4: FP32 output rounding

Source: [`test4.cu`](test4.cu) · Reference log: [`results/rtx5090/test4.log`](../../results/rtx5090/test4.log)

## Goal

Identify how the MMA rounds the result when it writes FP32 D. The test has two stages:

1. distinguish RU, RD, RZ, RA and RN;
2. only if stage 1 gives RN, use midpoint inputs to distinguish RNU, RND, RNZ, RNA, RNE and RNO.

## Instructions

All 16 instructions I1–I16 ([instructions.md](../../docs/instructions.md)).

## Principle

Only D[0][0] keeps a non-zero product, so the exact result is

```
x = c + a0 · b0
```

With c = ±C0 and C0 = 2^23 = 8388608, FP32 (24 significand bits) has a spacing of u = 1 between C0 and C0 + 2u.
The product a0 · b0 places x between two neighbours (stage 1) or exactly on the midpoint (stage 2), and the output
shows which neighbour was chosen.

The spacing below C0 is 0.5, so the perturbation must point away from zero: C0 + δ on the positive side and
−C0 − δ on the negative side.

## Input

- All A, B, C are 0 except A[0][0] = a0, B[0][0] = b0 and C[0][0] = c. Register positions and element codes are in
  [instructions.md](../../docs/instructions.md): A[0][0] and B[0][0] are element 0 of `a[0]` and `b[0]` of lane 0,
  C[0][0] is `c[0]` of lane 0.
- Scaled instructions use scale 1.0 (0x7F7F7F7F for UE8M0, 0x38383838 for UE4M3). A scale of 0 would make every
  product 0, and D = C would look exactly like RZ.
- a0, b0 and c are passed to the MMA separately; c + a0·b0 is never computed on the host.
- All values below are exactly representable in E2M1, so every format uses the same numbers.

## Cases

| Case | c | a0 | b0 | Exact x | Use |
|---|---|---|---|---|---|
| P1–P8 | 0 | (0.5, 1.5), (0.5, 0.5), (−0.5, 1.5), (−0.5, 0.5), (1, 0.5), (1, 1.5), (1, −0.5), (1, −1.5) | | a0·b0 | controls |
| S1 | +C0 | 0.5 | 1.5 | C0 + 0.75 | stage 1 |
| S2 | +C0 | 0.5 | 0.5 | C0 + 0.25 | stage 1 |
| S3 | −C0 | −0.5 | 1.5 | −C0 − 0.75 | stage 1 |
| S4 | −C0 | −0.5 | 0.5 | −C0 − 0.25 | stage 1 |
| T1 | +C0 | 1.0 | 0.5 | C0 + 0.5 | stage 2 |
| T2 | +C0 | 1.0 | 1.5 | C0 + 1.5 | stage 2 |
| T3 | −C0 | 1.0 | −0.5 | −C0 − 0.5 | stage 2 |
| T4 | −C0 | 1.0 | −1.5 | −C0 − 1.5 | stage 2 |

P1–P8 must return the exact product. They confirm that the encodings, element positions and scales are correct. If
any control fails, the other results of that instruction are invalid.

## Decision

### Stage 1

| Mode | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| RU (toward +∞) | C0 + 1 | C0 + 1 | −C0 | −C0 |
| RD (toward −∞) | C0 | C0 | −C0 − 1 | −C0 − 1 |
| RZ (toward zero) | C0 | C0 | −C0 | −C0 |
| RA (away from zero) | C0 + 1 | C0 + 1 | −C0 − 1 | −C0 − 1 |
| RN (nearest) | C0 + 1 | C0 | −C0 − 1 | −C0 |

The measured [S1, S2, S3, S4] is matched exactly against each row. No match is reported as UNCLASSIFIED.

### Stage 2 (only after RN)

| Mode | Midpoint rule | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|
| RNU | toward +∞ | C0 + 1 | C0 + 2 | −C0 | −C0 − 1 |
| RND | toward −∞ | C0 | C0 + 1 | −C0 − 1 | −C0 − 2 |
| RNZ | toward zero | C0 | C0 + 1 | −C0 | −C0 − 1 |
| RNA | away from zero | C0 + 1 | C0 + 2 | −C0 − 1 | −C0 − 2 |
| RNE | to even | C0 | C0 + 2 | −C0 | −C0 − 2 |
| RNO | to odd | C0 + 1 | C0 + 1 | −C0 − 1 | −C0 − 1 |

C0 is even, so RNE picks C0 or C0 + 2 and RNO picks C0 + 1.

## Results (RTX 5090)

All 16 instructions give the same output:

| Cases | Measured |
|---|---|
| P1–P8 | equal to the exact product |
| [S1, S2, S3, S4] | [8388608, 8388608, −8388608, −8388608] |
| [T1, T2, T3, T4] (printed for reference) | [8388608, 8388609, −8388608, −8388609] |

Stage 1 matches the RZ row for every instruction, so stage 2 does not apply. The T outputs are plain truncation,
consistent with RZ.

## Conclusion

The FP32 output of all 16 instructions is rounded toward zero (RZ).

## Notes

- RZ and "every product is 0" give the same S1–S4, so the P1–P8 controls must not be skipped.
- This test only covers the rounding of "one product + C" written to D. How the multi-term sum is truncated is
  covered by test2 and test3. Overflow beyond the FP32 range is covered by test5: it produces +∞, not the largest
  finite value that IEEE 754 RZ would give.
