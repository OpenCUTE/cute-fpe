# test9: FP4 low-bit perturbation versus FP6 E2M3

Source: [`test9.cu`](test9.cu) · Reference log: [`results/rtx5090/test9.log`](../../results/rtx5090/test9.log)

## Goal

In `kind::f8f6f4` and `kind::mxf8f6f4` an E2M1 element sits in bits [5:2] of an 8-bit container, and bits [1:0]
are specified as 0. The test asks whether the hardware decodes E2M1 by ignoring [1:0], or uses the same 6-bit
decoder as E2M3 and treats [1:0] as the two low mantissa bits.

## Principle

E2M1 (s ee m) and E2M3 (s ee mmm) both have bias 1. After the E2M1 code is shifted left by 2 into [5:2], its sign,
exponent and top mantissa bit line up with E2M3:

```
bit:    7  6  5  4  3  2  1  0
E2M1:   0  0  s  e  e  m  0  0
E2M3:   0  0  s  e  e  m  m  m
```

With [1:0] = 0 the byte decodes to the same value as E2M1 and as E2M3, for normal and subnormal values. The two
decodings differ only when [1:0] ≠ 0. A random value r in [1:0] separates two hypotheses:

| Hypothesis | Result of the E2M1 instruction with [1:0] = r |
|---|---|
| H1: [1:0] is ignored | equal to the E2M1 result with [1:0] = 0 |
| H2: shared decoder with E2M3 | equal to the E2M3 instruction on the same bytes |

## Instructions

| Path | E2M1 instruction | E2M3 instruction |
|---|---|---|
| no scale | I8 `kind::f8f6f4 … e2m1.e2m1` | I7 `kind::f8f6f4 … e2m3.e2m3` |
| 1X scale | I13 `kind::mxf8f6f4 … scale_vec::1X … e2m1.e2m1 … ue8m0` | I12 `kind::mxf8f6f4 … scale_vec::1X … e2m3.e2m3 … ue8m0` |

On the 1X path both scale registers are 0x7F7F7F7F (1.0) for all three runs. Full PTX strings are in
[instructions.md](../../docs/instructions.md).

## Input

Each round draws new values:

- every element of A (16×32) and B (32×8) gets a 6-bit raw value `raw6 = (h4 << 2) | r2`, where h4 is uniform in
  0–15 (the E2M1 code s ee m, including −0 = 0x8) and r2 is uniform in 0–3 (the low-bit perturbation);
- a clean copy `raw6_clean = h4 << 2` (r2 = 0) is kept;
- every C element is ±n × 2^-4 with n uniform in 0–63, so |C| < 4 and C is exact in FP32. C is small so that the
  products are not dominated by C.

Bytes are written with bits [7:6] = 0, using the 8-bit slot layout of m16n8k32 ([instructions.md](../../docs/instructions.md)).
The random generator has a fixed seed.

## Procedure

Each round runs three MMAs per path with the same C and scales:

| Run | Instruction | A, B | Result |
|---|---|---|---|
| R1 | E2M1 | raw6 (perturbed) | D1 |
| R2 | E2M3 | raw6 (perturbed) | D2 |
| R3 | E2M1 | raw6_clean (r2 = 0) | D3 |

The full 16×8 results are compared bit by bit (raw FP32 bits; +0 and −0 differ): D1 against D2 and D1 against D3.
100,000 rounds per path:

```
./build/test9 [rounds]     # default 100000
```

Self-check before the main run: with all r2 = 0, D1, D2 and D3 must be bit-identical (2,000 rounds).

## Decision

| Observation | Conclusion |
|---|---|
| D1 = D3 always and D1 ≠ D2 at least once | H1: E2M1 is decoded from 4 bits; [1:0] is ignored |
| D1 = D2 always and D1 ≠ D3 at least once | H2: E2M1 and E2M3 share the 6-bit decoder; [1:0] acts as mantissa |
| D1 differs from both D2 and D3 | neither hypothesis; the failing rounds need analysis |

## Results (RTX 5090)

| Path | Rounds | Rounds with D1 ≠ D2 | Rounds with D1 ≠ D3 | Conclusion |
|---|---|---|---|---|
| no scale (`kind::f8f6f4`) | 100,000 | 0 | 100,000 | H2 |
| 1X scale (`kind::mxf8f6f4`) | 100,000 | 0 | 100,000 | H2 |

Both self-checks pass (2,000 rounds, all three results identical).

## Conclusion

In `kind::f8f6f4` and `kind::mxf8f6f4`, E2M1 and E2M3 share one 6-bit decoder. Bits [1:0] of an E2M1 slot are not
ignored; they act as the two low mantissa bits of E2M3: every round D1 equals D2 bit for bit and differs from D3.
Software that writes E2M1 data for these instructions must therefore keep [1:0] at 0.
