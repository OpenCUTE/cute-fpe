# test7: alignment exponent floor (−133 / −139)

Source: [`test7.cu`](test7.cu) · Reference log: [`results/rtx5090/test7.log`](../../results/rtx5090/test7.log)

## Goal

The paper (*Bit-Accurate Reference Model*) and [arithmetic_flow.md](../../docs/arithmetic_flow.md) state two rules:

- **H1** (all instructions except `scale_vec::2X` / `4X`): the largest exponent used for alignment has a lower bound
  of −133. When all exponents, including scales, are below −133, alignment uses −133.
- **H2** (`scale_vec::2X` / `4X`): the same lower bound exists with the value −139.

With emax_eff the exponent actually used for alignment,

```
emax_eff = max(e_C, e_0, e_1, ..., floor)
```

This test measures emax_eff.

## Principle

From test3: after alignment, values within p bits below emax_eff are kept and lower values are truncated, with
p = 35 on `kind::mxf4` / `kind::mxf4nvf4` and p = 25 elsewhere.

Use a large product +2^m and a small product −2^e, with C = +0:

- if the small product is kept, D = RZ(2^m − 2^e), one ulp below 2^m;
- if it is truncated, D = 2^m, the same as a reference run without the small product.

e decreases from m − 1. The first truncated e is e_drop, and since e_drop = emax_eff − p − 1,

```
emax_eff = e_drop + p + 1
```

| Case | No floor | Floor L |
|---|---|---|
| m ≥ L | emax_eff = m | emax_eff = m |
| m < L | emax_eff = m | emax_eff = L |

m must be below the floor to distinguish the hypotheses, and not below −148: otherwise D is below the smallest FP32
subnormal 2^-149, rounds to 0 under RZ, and cannot be observed.

C cannot serve as the large term. A C below 2^-126 is an FP32 subnormal, and a subnormal C is aligned with exponent
−126 (see the side result), so C never falls below the floor.

## Instructions

Only instructions whose products can go below −148 can show the floor:

| Instruction | Measurable | Reason |
|---|---|---|
| I1 TF32, I3 BF16 | yes | normal values reach 2^-126, products 2^-252 |
| I9 E5M2, I10 E4M3 (`mxf8f6f4` 1X) | yes | the UE8M0 scale (down to 2^-127) lowers the product |
| I11 E3M2 (`mxf8f6f4` 1X) | only m = −148 | 1X has one scale, so the gap between two products comes from element values only; E3M2 reaches 12 bits |
| I14 `mxf4` 2X, I15 `mxf4nvf4` 2X | yes | the two products sit in different scale groups, each with its own UE8M0 scale |
| I2 FP16, I4–I8 (`f8f6f4`) | no | no scale; the smallest element exponents keep products far above −133 (FP16: not below −28) |
| I12 E2M3, I13 E2M1 (`mxf8f6f4` 1X) | no | element values separate two products by at most 4 bits |
| I16 NVFP4 (UE4M3) | no | UE4M3 is at least 2^-9, so products stay above about 2^-20 |

## Input

Only A[0][:], B[:][0] are used; C = +0; D[0][0] is observed. Codes: 2^x is (127 + x) << 23 in TF32 and
(127 + x) << 7 in BF16 (sign bit for negative values); UE8M0 is 127 + x; E5M2 / E4M3 / E3M2 powers of two use each
format's bias, and every value used is normal. Register mapping is in [instructions.md](../../docs/instructions.md).

| Instructions | Large product +2^m | Small product −2^e |
|---|---|---|
| I1, I3 | k = 0: a = 2^⌊m/2⌋, b = 2^(m − ⌊m/2⌋) | k = 1: a = −2^⌊e/2⌋, b = 2^(e − ⌊e/2⌋) (⌊·⌋ truncates toward zero) |
| I14, I15 | k = 0: a = b = 1; group 1 scale_A × scale_B = 2^m | k = 32: a = −1, b = 1; group 2 scale_A × scale_B = 2^e |
| I9, I10, I11 | k = 0: a = b = 2^u | k = 1: a = −2^x, b = 2^y, x + y = e − s, x = ⌊(e − s)/2⌋ |

A scale exponent t is split as ⌊t/2⌋ for scale_A and t − ⌊t/2⌋ for scale_B, both inside UE8M0 [−127, 127].

I9–I11 have one scale S = 2^s with s = m − 2u, where u is the largest power-of-two exponent of the format and emin
the smallest normal exponent:

| Instruction | u | emin | smallest reachable e |
|---|---|---|---|
| I9 E5M2 | 15 | −14 | s − 28 |
| I10 E4M3 | 8 | −6 | s − 12 |
| I11 E3M2 | 4 | −2 | s − 4 |

## Procedure

1. m ∈ {−130, −136, −140, −145, −148}. m = −130 is above both floors and serves as a control.
2. Run the reference without the small product: D_ref = 2^m.
3. Decrease e from m − 1 to the smallest reachable value, one run each, and compare D with D_ref.
4. Record the first e with D = D_ref (e_drop) and check that every smaller e also gives D = D_ref.
5. emax_eff = e_drop + p + 1.

## Results (RTX 5090)

emax_eff for each m; every row is monotonic.

| Instruction | p | m = −130 | m = −136 | m = −140 | m = −145 | m = −148 |
|---|---|---|---|---|---|---|
| I1 TF32 | 25 | −130 | −133 | −133 | −133 | −133 |
| I3 BF16 | 25 | −130 | −133 | −133 | −133 | −133 |
| I9 E5M2 1X | 25 | −130 | −133 | −133 | −133 | −133 |
| I10 E4M3 1X | 25 | −130 | −133 | −133 | −133 | −133 |
| I11 E3M2 1X | 25 | not reachable | not reachable | not reachable | not reachable | −133 |
| I14 `mxf4` 2X | 35 | −130 | −136 | −139 | −139 | −139 |
| I15 `mxf4nvf4` 2X | 35 | −130 | −136 | −139 | −139 | −139 |

For example, I1 with m = −148 has e_drop = −159. Without a floor it would be −174.

## Side result: a subnormal C is aligned at −126

To rule out C as the large term: C[0][0] = 2^c is the largest term and a small product −2^e sits at k = 0; emax_eff
is measured as above.

| c | −148 | −140 | −136 | −134 | −133 | −132 | −130 | −126 | −120 |
|---|---|---|---|---|---|---|---|---|---|
| emax_eff | −126 | −126 | −126 | −126 | −126 | −126 | −126 | −126 | −120 |

The nine instructions I1, I3 and I9–I15 give identical rows. For c ≤ −126, emax_eff stays at −126: a subnormal C is
aligned at the FP32 minimum exponent and is not renormalized, the same treatment the inputs receive (an E5M2
subnormal is 0.5 · 2^-14, not 1.0 · 2^-15).

## Conclusion

- H1 holds: TF32, BF16 and `mxf8f6f4` 1X (measured on E5M2, E4M3, E3M2) have an alignment floor of −133.
- H2 holds: `mxf4` 2X and `mxf4nvf4` 2X have a floor of −139.
- Above the floor (m = −130, and m = −136 on the FP4 path) emax_eff equals the real maximum.
- A subnormal C is aligned at −126 and never triggers the floor.
- On I2, I4–I8, I12, I13 and I16 the products cannot go below the floor, so the floor cannot be observed and does
  not affect their results.
