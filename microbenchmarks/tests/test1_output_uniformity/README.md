# test1: output uniformity

Source: [`test1.cu`](test1.cu) · Reference log: [`results/rtx5090/test1.log`](../../results/rtx5090/test1.log)

## Goal

Check that every output position of an MMA instruction follows the same observable arithmetic rule. If all
positions receive identical dot-product inputs, the same C and the same scales, their raw output bits must be
identical. Once this holds, the other probes only need to study D[0][0].

## Instructions

All 16 instructions I1–I16 ([instructions.md](../../docs/instructions.md)).

## Procedure

### 1. Write-back check (once per instruction)

Set A = B = 0, C[0][1] = 1 and all other C to 0. D[0][1] must be 1 and D[0][0] must be 0. This rules out a false
match caused by a broken write-back, for example all lanes writing the same position.

### 2. Random rounds

Each round:

1. **A, B**: draw A[0][:] and B[:][0] (length K) from the distribution below. Copy A[0][:] to all 16 rows and
   B[:][0] to all 8 columns.
2. **C**: draw one FP32 value and copy it to the whole 16×8 C.
3. **Scales** (scaled instructions only): draw the four bytes of scale_A and scale_B independently. The scale
   registers are shared by the whole tile, so every row and column sees the same scales.
4. **Execute and compare**: run the MMA and compare every D element bit by bit with D[0][0]. +0 and −0 count as
   different.

### 3. Distributions

Each instruction runs both distributions.

**Distribution 1: raw random bits.** Every element code, the 32 bits of C and every scale byte are uniform. This
covers NaN, Inf, subnormals and ±0. It is a weak test of the arithmetic, because many rounds return NaN or are
dominated by C, so it is not used alone.

**Distribution 2: finite values of moderate magnitude.** This is the main distribution.

| Operand | Generation |
|---|---|
| A, B (FP16 and 8/6/4-bit formats) | uniform raw code, NaN and Inf excluded; ±0 and subnormals kept |
| A, B (TF32, BF16) | random sign; exponent uniform in [−20, 20]; random mantissa bits |
| C | random sign; exponent uniform in [−8, 8]; random 23-bit mantissa |
| UE8M0 scale | each byte uniform in 124–130 (2^-3 to 2^3) |
| UE4M3 scale | each byte a normal value with exponent 2^-3 to 2^3 and a random 3-bit mantissa |

To show how much arithmetic a distribution exercises, every round also records whether D[0][0] ≠ C (the products
changed the output) and whether D[0][0] is NaN.

### 4. Scale

1,000,000 rounds per instruction and distribution, with a fixed seed. The instruction passes only if all 128
elements equal D[0][0] in every round.

```
./build/test1 [rounds]     # default 1000000
```

## Results (RTX 5090)

All 16 write-back checks pass. In both distributions, all 128 D elements equal D[0][0] in every round.

Rounds (out of 1,000,000) in which D[0][0] ≠ C or D[0][0] is NaN:

| Instruction | Dist. 1: D ≠ C | Dist. 1: NaN | Dist. 2: D ≠ C | Dist. 2: NaN |
|---|---|---|---|---|
| I1 TF32 | 974,745 | 64,030 | 1,000,000 | 0 |
| I2 FP16 | 885,390 | 638,739 | 1,000,000 | 0 |
| I3 BF16 | 997,240 | 120,681 | 1,000,000 | 0 |
| I4 E5M2 | 959,672 | 792,896 | 1,000,000 | 0 |
| I5 E4M3 | 788,410 | 397,183 | 1,000,000 | 0 |
| I6 E3M2 | 626,547 | 3,884 | 999,995 | 0 |
| I7 E2M3 | 614,691 | 3,974 | 999,963 | 0 |
| I8 E2M1 | 613,117 | 3,893 | 998,988 | 0 |
| I9 MX E5M2 | 953,354 | 794,503 | 1,000,000 | 0 |
| I10 MX E4M3 | 767,144 | 401,529 | 1,000,000 | 0 |
| I11 MX E3M2 | 597,411 | 11,619 | 1,000,000 | 0 |
| I12 MX E2M3 | 589,502 | 11,691 | 999,952 | 0 |
| I13 MX E2M1 | 587,970 | 11,628 | 998,999 | 0 |
| I14 `mxf4` 2X | 790,050 | 19,210 | 999,515 | 0 |
| I15 `mxf4nvf4` 2X | 790,067 | 19,444 | 999,532 | 0 |
| I16 NVFP4 | 686,065 | 64,326 | 999,996 | 0 |

Under distribution 1, about 79% of the E5M2 rounds are NaN and about 40% of the FP6/FP4 rounds have D = C. Under
distribution 2, almost every round depends on the products.

## Conclusion

For all 16 instructions every output position follows the same rule, so the other tests analyze D[0][0] only.

## Limits

This is uniformity under broadcast inputs. It does not change non-target rows or columns, so it does not on its
own show that D[m][n] is independent of other rows of A and other columns of B. The layout self-check
([`selfcheck`](../../selfcheck/selfcheck.cu)) uses fully random matrices and an exact reference, which covers that
case for exactly representable sums.
