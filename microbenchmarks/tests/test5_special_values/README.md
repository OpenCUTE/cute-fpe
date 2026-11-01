# test5: special values and signed zero

Source: [`test5.cu`](test5.cu) · Reference log: [`results/rtx5090/test5.log`](../../results/rtx5090/test5.log)

## Goal

Let p_i = a_i × b_i. This test checks how the MMA handles NaN, ±∞, out-of-range intermediates and ±0. Hypotheses:

**Common path (I1–I13)**

- H1a: an input 0 × ∞, or any NaN input, gives NaN.
- H1b: if +∞ and −∞ both occur among the products and C, the result is NaN.
- H1c: if only one sign of ∞ occurs, the result is that ∞.
- H1d: special values are detected only on the inputs. A product beyond the range of the input format is not a
  special value; the decision is made when the result is normalized to FP32. For example, E5M2 0x7C is ∞, but
  0x7B × 0x7B is an ordinary number.
- H1e: a UE8M0 scale of 0xFF is NaN and makes the result NaN whatever the other inputs are. Only a single scale byte
  of 0xFF (true exponent 128) is NaN; a sum of scale exponents of 128 (for example 2^127 × 2^1) is not.

**Dedicated FP4 path (I14–I16)**

- H2a: FP4 has no ∞ or NaN. A NaN scale (UE8M0 0xFF, or UE4M3 with the low 7 bits all 1) gives NaN.
- H2b: if neither elements nor scales are special, a NaN or ∞ in C is propagated (D = C).
- H2c: the product of two scales is an ordinary value however large it is; it never produces NaN.

**Signed zero**

- H3: whether the p_i are all −0, all +0 or mixed, the output is +0; −0 never occurs.

## Instructions

All 16 instructions I1–I16 ([instructions.md](../../docs/instructions.md)).

## Input

- Only A[0][:], B[:][0] and C[0][0] are set; every element not listed is +0. D[0][0] is observed.
- Scales are 1.0 unless stated (0x7F7F7F7F for UE8M0, 0x38383838 for UE4M3). "scale_A0 = x" replaces byte 0 only.
- Special element codes (before the slot shift):

  | Format | +∞ | −∞ | NaN | −0 | max (as used here) |
  |---|---|---|---|---|---|
  | TF32 | 0x7F800000 | 0xFF800000 | 0x7FC00000 | 0x80000000 | 0x7F000000 (2^127) |
  | FP16 | 0x7C00 | 0xFC00 | 0x7E00 | 0x8000 | 0x7BFF (65504) |
  | BF16 | 0x7F80 | 0xFF80 | 0x7FC0 | 0x8000 | 0x7F00 (2^127) |
  | E5M2 | 0x7C | 0xFC | 0x7E | 0x80 | 0x7B (57344) |
  | E4M3 | – | – | 0x7F | 0x80 | 0x7E (448) |
  | E3M2 | – | – | – | 0x20 | 0x1F (28) |
  | E2M3 | – | – | – | 0x20 | 0x1F (7.5) |
  | E2M1 | – | – | – | 0x8 | 0x7 (6) |

  −max is max with the sign bit set. On I8 and I13 the E2M1 code is shifted left by 2.

- Scale codes: UE8M0 NaN 0xFF, 2^127 0xFE, 2^1 0x80; UE4M3 NaN 0x7F (0xFF also tested), 448 0x7E.
- FP32 codes for C: +∞ 0x7F800000, −∞ 0xFF800000, qNaN 0x7FC00000, −0 0x80000000.

Each case has an expected bit pattern for D[0][0]; the program reports PASS only on an exact match.

## Cases and results (RTX 5090)

### 4.1 Special element values

E1–E11 run on the instructions whose element format has ∞ or NaN codes: I1–I4 and I9. I5 and I10 (E4M3 has NaN
only) run E9–E11.

| Case | Input | Hypothesis | Expected | Measured |
|---|---|---|---|---|
| E1 | a0 = +∞, b0 = 1 | H1c | +∞ | +∞ |
| E2 | a0 = −∞, b0 = 1 | H1c | −∞ | −∞ |
| E3 | a0 = +∞, b0 = 0 | H1a | NaN | NaN |
| E4 | a0 = +∞, a1 = −∞, b0 = b1 = 1 | H1b | NaN | NaN |
| E5 | a0 = a1 = +∞, b0 = b1 = 1 | H1c | +∞ | +∞ |
| E6 | a0 = +∞, b0 = 1, C = −∞ | H1b | NaN | NaN |
| E7 | a0 = +∞, b0 = 1, C = +∞ | H1c | +∞ | +∞ |
| E8 | a0 = +∞, b0 = 1; a1 = −max, b1 = max | H1c | +∞ | +∞ |
| E9 | a0 = NaN, b0 = 1 | H1a | NaN | NaN |
| E10 | a0 = NaN, b0 = 0 | H1a | NaN | NaN |
| E11 | a0 = NaN, b0 = 1, C = +∞ | H1a | NaN | NaN |

### 4.2 Special C with finite A, B (all 16 instructions)

| Case | Input | Expected | Measured D |
|---|---|---|---|
| C1 | C = 0x7FC00000 (qNaN), a = b = 0 | NaN | 0x7FFFFFFF |
| C2 | C = 0x7F800001 (sNaN), a = b = 0 | NaN | 0x7FFFFFFF |
| C3 | C = 0xFFC01234 (negative NaN with payload), a0 = b0 = 1 | NaN | 0x7FFFFFFF |
| C4 | C = +∞, a0 = b0 = 1 | +∞ | 0x7F800000 |
| C5 | C = −∞, a0 = b0 = 1 | −∞ | 0xFF800000 |

### 4.3 Out-of-range intermediates (H1d, H1e, H2c)

| Case | Input | Instructions | Measured |
|---|---|---|---|
| V1 | a0 = b0 = max | all | finite: FP16 4290774016, E5M2 3288334336, E4M3 200704, E3M2 784, E2M3 56.25, E2M1 36, the same with and without MX; TF32 and BF16 (2^127 · 2^127) +∞ |
| V2 | TF32 / BF16: a0 = b0 = 2^127, product 2^254 beyond FP32 | I1, I3 | +∞ |
| V2 | UE8M0: a0 = b0 = max, scale_A = scale_B = 2^127 | I9–I15 | +∞ |
| V2 | UE4M3: a0 = b0 = 6, scale_A0 = scale_B0 = 448 | I16 | 7225344 (finite) |
| V3 | V2 plus a1 = −max, b1 = max (the two out-of-range products cancel) | I1, I3, I9–I15 | +0 |
| V4 | V2 plus C = −∞ | I1, I3, I9–I15 | −∞ |
| V5 | a0 = b0 = 1, scale_A = 2^127, scale_B = 2^1 (exponent sum 128) | I9–I15 | +∞ (not NaN) |

V3 and V4 decide H1d. If an out-of-range product were already ∞ inside, V3 (+∞ − ∞) would be NaN; it is +0. V4
(+∞ + C = −∞) would also be NaN; it is −∞, so the only special value is the input C.

### 4.4 NaN scale (H1e, H2a)

I9–I15 use the NaN code 0xFF, I16 uses 0x7F.

| Case | Input | Instructions | Measured |
|---|---|---|---|
| N1 | scale_A0 = NaN, a0 = b0 = 1 | I9–I16 | NaN |
| N2 | scale_B0 = NaN, a0 = b0 = 1 | I9–I16 | NaN |
| N3 | scale_A0 = NaN; k = 0–15: a = 1, b = 0 (structural zero) | I9–I16 | NaN |
| N4 | scale_A0 = NaN; a = b = 0 | I9–I16 | NaN |
| N5 | scale_A0 = NaN, a0 = b0 = 1, C = +∞ | I9–I16 | NaN |
| N6 | scale_A0 = NaN; only the second scale group is non-zero (a = b = 1 at k = K / vec) | I14–I16 | NaN |
| N7 | scale_A0 = 0xFF (top bit set, low 7 bits all 1) | I16 | NaN |
| N8 | scale_A0 = 0x7E (448, largest normal), a0 = b0 = 1 | I16 | 448 |
| N9 | scale_A0 = 0xFE (2^127), a0 = b0 = 1 | I9–I15 | 2^127 |

### 4.5 Signed zero (H3, all 16 instructions)

"all p = −0" means every k = 0 … K−1 is −0.

| Case | Input | IEEE 754 result | Measured |
|---|---|---|---|
| Z1 | all p = −0 (a = −0, b = 1), C = −0 | −0 | +0 |
| Z2 | all p = −0 (a = −1, b = +0), C = −0 | −0 | +0 |
| Z3 | all p = −0, C = +0 | +0 | +0 |
| Z4 | all p = +0, C = −0 | +0 | +0 |
| Z5 | odd k: p = −0, even k: p = +0, C = −0 | +0 | +0 |
| Z6 | as Z5, C = +0 | +0 | +0 |
| Z7 | a0·b0 = 1, a1·b1 = −1, C = −0 | +0 | +0 |
| Z8 | a0·b0 = −1, C = +1 | +0 | +0 |

In Z1 and Z2 every term is −0, so IEEE 754 would give −0; the hardware gives +0.

## Conclusion

- H1a–H1e, H2a–H2c and H3 all hold (372 checks).
- Every NaN result is the canonical 0x7FFFFFFF. The sign and payload of a NaN in C are not kept, sNaN included. So
  "D = C" in H2b holds for the value class, not for the bit pattern.
- A result beyond the FP32 range becomes +∞ (V2, V5), although test4 shows RZ rounding. IEEE 754 RZ would give the
  largest finite value 0x7F7FFFFF.
- A NaN scale always gives NaN, even if its group is all structural zeros or a = b = 0 (N3, N4, N6). This is the
  exception to the test8 rule "a group whose products are all zero ignores its scale".
- UE4M3 ignores the top bit when it checks for NaN: 0x7F and 0xFF are both NaN (N7). This matches test8, where
  0x80 is treated as 0.
