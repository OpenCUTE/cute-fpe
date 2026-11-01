# Paper statements and their GPU evidence

Each row links a statement of the paper (*Arithmetic Characterization* and *Bit-Accurate Reference Model*) to the
test that probes it and to the measured outcome. All outcomes come from the reference run in
[`results/rtx5090/`](results/rtx5090/); every listed case is a PASS line in the corresponding log.

"Supported" means the cases distinguish the stated rule from the alternatives listed in the test document, and the
GPU output matches the stated rule.

| Paper statement | Test | Deciding cases | Outcome | Status |
|---|---|---|---|---|
| The output elements of an MMA operation are computed independently, so the characterization can be reduced to one output element. | [test1](tests/test1_output_uniformity/), [selfcheck](selfcheck/selfcheck.cu) | broadcast rounds, 2 distributions × 10^6 per instruction; random full matrices against an exact reference | 128 elements identical in every round for I1–I16; full matrices exact in 38,400 / 38,400 elements per instruction | Supported |
| Cancellation cases expose the reduction organization. On the common path all products are reduced together. | [test2](tests/test2_accumulation_order/) methods 1, 2 | ±2^16 pair at every (k1, k2) with small products; ±1 pair with C = 2^-24 | D = 0 for every pair (I1–I5, I9, I10); C kept for every pair (I6–I8, I11–I13) | Supported |
| On the dedicated path, products are first reduced within scale-associated groups. | [test2](tests/test2_accumulation_order/) method 3 | two probes at the 36-bit edge, all 496 pairs, both directions | D = 2 exactly when both probes are in the same block of 16 (240 pairs), 0 otherwise (256); also for `scale_vec::2X` | Supported (group width 16 for MXFP4 too) |
| The common path keeps 25 fractional bits after alignment. | [test3](tests/test3_hidden_precision/) methods 1, 2 | small term halved until truncated | p = 25 on all 13 common-path instructions | Supported |
| The dedicated path uses 35 fractional bits. | [test3](tests/test3_hidden_precision/) method 3 | MXFP4 scale sweep; NVFP4 with C as large term | MXFP4 first D = 0 at Sa = 36; NVFP4 at n = −5; p = 35 for both | Supported |
| Rounding-boundary inputs isolate the rounding of the output. | [test4](tests/test4_rounding/) | S1–S4 around 2^23 with controls P1–P8 | [S1..S4] = [2^23, 2^23, −2^23, −2^23] on I1–I16: RZ | Supported |
| Special values isolate exception behavior. | [test5](tests/test5_special_values/) | E1–E11, C1–C5, V1–V5, N1–N9, Z1–Z8 | 0 × ∞, NaN inputs, opposite infinities and NaN scales give 0x7FFFFFFF; out-of-range intermediates are finite until normalization; overflow gives +∞; results are never −0 | Supported |
| On the common path, E8M0 scales contribute only exponent offsets. | [selfcheck](selfcheck/selfcheck.cu), [test8](tests/test8_zero_exponent/) part (2) | random scales 0.5/1/2 against an exact reference; scale 2^40 control | exact results for I9–I13; control gives 2^40 | Supported |
| On the dedicated path, group sums are multiplied by the scale mantissas before the group sums and the accumulator are aligned. | [test6](tests/test6_scale_order/) | main experiment with scale 1.5 · 2^-6; 32-case scan | D = 2^-11 (scale first) instead of 1.5 · 2^-11; scale-first model matches 32/32, including all 10 distinguishing cases | Supported (NVFP4) |
| The effective alignment exponent is lower-bounded at −133 on the common path and at −139 on the dedicated path. | [test7](tests/test7_exponent_floor/) | product sweep with m ∈ {−130 … −148} | emax_eff = −133 for m ≤ −136 on I1, I3, I9, I10 (and I11 at m = −148); −139 for m ≤ −140 on I14, I15; m = −130 control equals m | Supported |
| (Reference-model rule) A structural zero has the minimum exponent; a cancellation keeps its exponent. | [test8](tests/test8_zero_exponent/) | Z / Zmix / N cases for I1–I15; R / Z / N / S1–S3 / control for I16 | structural zeros keep C; cancellations shift C out; zero UE4M3 scales (0x00, 0x80) behave as structural zeros | Supported |
| (Reference-model rule) E2M1 in an 8-bit slot is the E2M3 byte `00_e2m1_00`. | [test9](tests/test9_fp4_fp6_low_bits/) | random bits [1:0], 10^5 rounds per path | E2M1 result equals E2M3 in 100,000 / 100,000 rounds and differs from the clean E2M1 in 100,000 / 100,000 | Supported |

## Additional observations

These are not stated in the paper but are measured by the tests:

- A NaN result is always 0x7FFFFFFF; the sign and payload of a NaN in C are lost (test5 C1–C3).
- FP32 overflow produces +∞ even though rounding is RZ (test5 V2, V5).
- A subnormal C is aligned at exponent −126 and is not renormalized (test7 side result).
- A NaN scale makes the result NaN even when its group is all zeros (test5 N3, N4, N6).
- UE4M3 ignores the top bit of a scale byte for both zero (0x80) and NaN (0xFF) (test8 S2, test5 N7).

## Not covered here

- The floor cannot be observed on I2, I4–I8, I12, I13 and I16, because their products cannot reach −133 (test7).
- The 100M-per-format GPU ↔ reference-model and 1M-per-format reference-model ↔ RTL comparisons belong to the
  validation flow in `../cmodel/` and `../rtl_cmodel_compare/`.
- Integer formats and FP16 output are outside the scope of these microbenchmarks.
