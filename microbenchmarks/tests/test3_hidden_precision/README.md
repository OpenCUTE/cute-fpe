# test3: hidden accumulation precision

Source: [`test3.cu`](test3.cu) · Reference log: [`results/rtx5090/test3.log`](../../results/rtx5090/test3.log)

## Definition

Accumulation precision p: when the products and C are aligned to the largest term (the "large term"), values
within p bits below its exponent are kept, and values p + 1 or more bits below are lost.

All methods follow the same idea:

- build a large term that cancels during the summation, so that only a small term can reach D;
- shrink the small term until D becomes 0;
- if the exponent gap at that point is g, then p = g − 1.

| Method | Instructions | Large term | Small term |
|---|---|---|---|
| 1 | TF32, FP16, BF16, E5M2 / E4M3 with and without MX (I1–I5, I9, I10) | a cancelling pair ±2^16 | the other products and C |
| 2 | E3M2, E2M3, E2M1 with and without MX (I6–I8, I11–I13) | a cancelling pair ±P | C |
| 3.1 | MXFP4 (I14) | a cancelling pair ±2^Sa | C = 1 |
| 3.2 | NVFP4 (I16) | C = 2^24, cancelled by one group sum | one product in group 1 |

Only A[0][:], B[:][0] and C[0][0] are set; D[0][0] is observed. Encodings are in
[instructions.md](../../docs/instructions.md).

## Method 1 (I1–I5, I9, I10)

test2 shows these instructions are fully fused, so only 1.1 is used.

### 1.1 Grouped or fused summation (g > 2)

- A[0][0] = 2^8, A[0][1] = −2^8, B[0][0] = B[1][0] = 2^8: large term ±2^16.
- All other A[0][k] = 2^t and B[k][0] = 2^-6, so every other product is 2^(t−6).
- C[0][0] = 2^(t−6), all other C = 0.
- t starts at 0 and decreases by 1 (down to −6) until D[0][0] = 0.

The exponent gap is 16 − (t − 6) = 22 − t, so p = 21 − t.

### 1.2 Near-sequential summation (g ≤ 2)

For an instruction whose group size is at most 2: A = B = 1, C = 2^q, q increases from 0 until D[0][0] = C (the
products are fully lost). Then p = q − 1. No tested instruction needs this branch.

### Results (RTX 5090)

| Instruction | First t with D = 0 | p |
|---|---|---|
| I1 TF32, I2 FP16, I3 BF16 | −4 | 25 |
| I4 E5M2, I5 E4M3 (`kind::f8f6f4`) | −4 | 25 |
| I9 E5M2, I10 E4M3 (`kind::mxf8f6f4` 1X) | −4 | 25 |

## Method 2 (I6–I8, I11–I13)

- A[0][0] and A[0][1] are opposite, all other A are 0, giving a cancelling pair ±P.
- C = 2^c everywhere, c starts at 0 and is halved until D[0][0] = 0.

| Version | Construction | P |
|---|---|---|
| without MX (I6–I8) | A = ±4, B = 4 everywhere | 2^4 |
| with MX (I11–I13) | A = ±1, B[:][0] = 1, scale_A = scale_B = 2^12 | 2^24 |

p = log2(P) − c − 1.

| Instruction | P | First C with D = 0 | p |
|---|---|---|---|
| I6 E3M2, I7 E2M3, I8 E2M1 | 2^4 | 2^-22 | 25 |
| I11 E3M2, I12 E2M3, I13 E2M1 | 2^24 | 2^-2 | 25 |

## Method 3

### 3.1 MXFP4 (I14)

- A[0][0] = 1, A[0][63] = −1, other A = 0; B = 1 everywhere.
- Both A scale bytes are 2^Sa and the B scale is 1, giving a cancelling pair ±2^Sa.
- C = 1.
- Sa starts at 0 and increases until D[0][0] = 0.

p = Sa − 1. Measured: the first D = 0 is at Sa = 36, so **p = 35**.

### 3.2 NVFP4 (I16)

The UE4M3 scale only reaches about 2^8, so "± products + C = 1" cannot create a gap of more than 35 bits. C is the
large term instead.

- C = 2^24 everywhere.
- Group 4: A[0][48..63] = −4, B[48..63][0] = 4, scale_A3 = scale_B3 = 2^8. Each product is −2^20, and the 16 of
  them sum to −2^24, which cancels C.
- Group 1: A[0][0] = B[0][0] = 1, scale_B0 = 2^-7, scale_A0 = 2^n: the small product 2^(n−7).
- Groups 2 and 3: A = 0.
- n starts at 0 and decreases until D[0][0] = 0.

The exponent gap is 24 − (n − 7) = 31 − n, so p = 30 − n. Measured: the first D = 0 is at n = −5, so **p = 35**.

## Conclusion

| Instructions | p |
|---|---|
| TF32, FP16, BF16, `kind::f8f6f4` (all formats), `kind::mxf8f6f4` 1X (all formats) | 25 |
| MXFP4 (`kind::mxf4` 2X), NVFP4 (`kind::mxf4nvf4` 4X) | 35 |

For MXFP4 the result agrees with the boundary in test2 method 3: a value 35 bits below the cancelled maximum is
kept, 36 bits below it is lost.
