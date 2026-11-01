# Inferred MMA arithmetic flow

This document summarizes the arithmetic rules inferred from the microbenchmarks. Each step names the tests that
support it. The paper describes the same rules in *Arithmetic Characterization* and *Bit-Accurate Reference Model*
(Figs. `fig-nv-float` and `fig-nv-block-scale-float`).

The output element is

```
d = Σ_k a_k · b_k + c
```

There are two paths:

- the **common path**: TF32, FP16, BF16, FP8, FP6, FP4 and `kind::mxf8f6f4` (I1–I13);
- the **dedicated FP4 path**: `kind::mxf4` and `kind::mxf4nvf4` (I14–I16).

## 1. Common path (I1–I13)

### 1.1 Special values (test5)

- If an input is NaN, or a product is 0 × ∞, D is NaN.
- If the products and C contain both +∞ and −∞, D is NaN.
- If only one sign of infinity occurs, D is that infinity.
- Special values are detected **only on the inputs**. A product that exceeds the range of the input format (for
  example E5M2 0x7B × 0x7B) is an ordinary finite value. Two products that exceed even the FP32 range cancel to
  +0 instead of producing NaN. Overflow is only decided when the result is normalized to FP32.
- A UE8M0 scale of 0xFF is NaN, and D is NaN regardless of the other inputs. A scale exponent sum of 128 (for
  example 2^127 × 2^1) is not NaN.
- Every NaN result is the canonical 0x7FFFFFFF. The sign and payload of a NaN in C are not propagated.

### 1.2 Fixed-point products (test3, test8, test9)

- Each product takes the significand and exponent from the raw encodings, without renormalizing subnormals. For
  example, an E5M2 subnormal is treated as 0.5 · 2^-14, not as 1.0 · 2^-15.
- Significands are multiplied and exponents added. The product is not renormalized: 1.5·2^1 × 1.5·2^2 is
  2.25·2^3, not 1.125·2^4. This affects the alignment below.
- The product is exact: from E2M1 to TF32 no precision is lost and nothing overflows or underflows.
- UE8M0 scales (I9–I13) only add to the exponent.
- A **structural zero** (a product with at least one factor equal to 0) has its exponent set to the minimum, so
  that it never shifts other terms out during alignment. A sum such as `1 + (−1)` is not a structural zero: its
  exponents still take part in alignment (test8). For the same reason, a block of `kind::mxf8f6f4` whose
  products are all structural zeros does not let its scale take part in alignment (test8), except that a NaN
  scale still produces NaN (test5).
- E2M1 in an 8-bit slot is decoded as the E2M3 byte `00_e2m1_00`. Bits [1:0] of the slot act as the two low
  mantissa bits of E2M3 (test9).

### 1.3 Alignment (test3, test7)

- All products and C are aligned to the largest exponent `emax`. Each significand is shifted right by
  `emax − e_k` and truncated to **25 fractional bits** (test3).
- `emax` has a lower bound of **−133**. When every exponent, including the scale, is below −133, alignment
  uses −133 (test7).
- A subnormal C is aligned with exponent −126, the FP32 minimum, and is not renormalized (test7).

Examples: `1.0·1.0 + (−1.0)·1.0 + 2^-25 = 2^-25`, and `2.0·1.0 + (−2.0)·1.0 + 2^-25 = 0`.

### 1.4 Fixed-point summation (test1, test2)

- The aligned terms are summed in fixed point. The summation is exact, so the order does not matter: the K
  dimension is fully fused (test2).
- Every output element is computed by the same rule (test1).

### 1.5 Normalization (test4, test5)

- The sum is normalized to FP32 with **round toward zero** (test4).
- If no NaN occurred before, normalization does not create one. A result beyond the FP32 range becomes +∞ even
  though rounding is RZ (test5).
- A zero result is always +0. −0 never occurs, even if every product and C are −0 (test5).

## 2. Dedicated FP4 path (I14–I16)

### 2.1 Special values (test5)

- E2M1 has no ∞ or NaN. If a scale is NaN (UE8M0 0xFF; UE4M3 with the low 7 bits all 1, i.e. 0x7F or 0xFF), D is
  NaN.
- If neither the elements nor the scales are special, a NaN or ∞ in C is propagated (NaN as 0x7FFFFFFF).
- The product of two scales is an ordinary value however large it is; it never becomes NaN.

### 2.2 Fixed-point products

The products a_k · b_k (magnitudes 0.25 to 36) are exact fixed-point values.

### 2.3 Group sums (test2)

Products are summed in **groups of 16 consecutive elements**, even when the scale block is 32 elements
(`scale_vec::2X`). K = 64 gives four group sums σ0–σ3 for k = 0–15, 16–31, 32–47 and 48–63.

### 2.4 Scaling the group sums (test6, test8)

- Each group sum is multiplied by the significands of its two scales. Group j uses scale byte j. The exponent is
  the sum of the two scale exponents. A UE4M3 subnormal scale such as 0.5 · 2^-6 keeps significand 0.5 and
  exponent −6.
- The scale is applied **before** the cross-group alignment: the scale significand takes part in the truncation
  (test6).
- A group whose products are all structural zeros has its exponent set to the minimum. A group that cancels to
  zero, such as `1 + (−1)`, keeps the exponent given by its scales (test8).
- A UE4M3 scale whose low 7 bits are 0 (0x00 or 0x80) makes the group zero, and the group exponent is set to the
  minimum (test8).

### 2.5 Alignment (test3, test7)

- σ0–σ3 and C are aligned to `emax = max(e_C, e_0, e_1, e_2, e_3)`. Bits shifted out of the **35 fractional
  bits** are truncated (round toward zero) (test3).
- `emax` has a lower bound of **−139** (test7).

### 2.6 Fixed-point summation and 2.7 normalization (test4)

The aligned values are summed in fixed point and normalized to FP32 with round toward zero.

## Limits of these conclusions

The rules are inferred from black-box outputs of one GPU (RTX 5090, driver 590.44.01, CUDA 13.1). They describe the
observable behavior for the inputs in the tests; they do not prove a particular internal circuit. Statements that
some tests cannot reach are marked in each test document, for example the exponent floor on instructions whose
products cannot get below −133 (test7).
