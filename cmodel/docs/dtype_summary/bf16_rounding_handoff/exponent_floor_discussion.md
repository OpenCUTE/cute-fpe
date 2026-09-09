# Tensor Core Exponent Floor Discussion

Date: 2026-06-04

## Background

We observed a low-exponent behavior:

```text
If all lanes have max exponent lower than -133,
the effective max exponent appears to be clamped to -133.
```

This note records the discussion and the current best explanation, so we can continue from here later.

## Initial Hypothesis

The first idea was that `-133` might be connected to BF16:

```text
BF16 min normal exponent = -126
BF16 fraction bits = 7
-126 - 7 = -133
```

So `-133` is exactly the minimum subnormal exponent / minimum non-zero quantum of BF16.

This made the BF16-specific explanation tempting:

```text
Tensor Core may clamp very small BF16 products to the BF16 representable floor.
```

There was also a higher-level training hypothesis:

```text
Maybe hardware intentionally introduces a quantization-like effect because BF16 is widely used in LLM training,
and training might adapt to this effect similarly to quantization-aware training.
```

## Problem With That Hypothesis

The key counter-observation is:

```text
This behavior is not limited to BF16.
Other dtypes also appear to use the same -133 threshold.
```

That makes the BF16-only explanation insufficient.

It also weakens the QAT explanation:

1. QAT is an algorithm/training procedure, not usually a hidden hardware behavior.
2. Tensor Core is a deterministic arithmetic unit; it is more likely implementing a fixed numerical datapath rule.
3. If many input dtypes share `-133`, the threshold is probably tied to a shared internal accumulator/alignment path, not BF16 input format alone.
4. Hardware behavior may be something training can adapt to, but that does not mean hardware is intentionally doing QAT.

## Revised Explanation

The more reasonable current explanation is:

```text
-133 is a Tensor Core internal exponent floor.
```

Possible rule:

```text
if real_max_product_exp < -133:
    internal_align_exp = -133
else:
    internal_align_exp = real_max_product_exp
```

This floor may be part of the shared Tensor Core accumulation path, especially for FP32 accumulation.

The value `-133` may still be meaningful:

```text
-133 = -126 - 7
```

But after observing the same behavior across dtypes, we should interpret this as:

```text
A shared internal floor near the FP32/BF16 normal boundary plus a small subnormal/guard region,
not a BF16-only format rule.
```

## Refined Training-Dataflow Interpretation

A more precise version of the BF16-related hypothesis is not that Tensor Core is doing QAT, and not that
the rule only applies because the current input dtype is BF16.

The refined idea is:

```text
Even when the MMA output/accumulator is FP32, that FP32 result is often stored back to a lower-precision
training format before it is consumed by the next MAC.
```

In mainstream deep learning training, BF16 is a common storage/activation/weight format. If an FP32
Tensor Core result is later stored as BF16 and then reloaded for another low-precision MAC, values below
the BF16 minimum non-zero scale cannot survive that store/reload path.

That makes `-133` meaningful:

```text
BF16 min normal exponent = -126
BF16 fraction bits = 7
BF16 min subnormal exponent = -126 - 7 = -133
```

So the threshold can be interpreted as:

```text
-133 is the lowest exponent that can still be observed after a BF16 store/reload path.
```

This gives a stronger reason for why the threshold is exactly `-133` instead of another value:

```text
-126 : would correspond to flushing all BF16 subnormal-scale values.
-133 : preserves the full BF16 subnormal tail.
-149 : would preserve full FP32 subnormal range, which is much wider than the common BF16 low-precision path needs.
```

A concise working hypothesis is:

```text
Tensor Core's low-precision matrix path may use a BF16-observable lower bound even when the immediate
destination is FP32, because the common training dataflow stores/reloads intermediate values in BF16.
```

This is a hardware/dataflow optimization hypothesis, not a strict IEEE-style semantic rule. A single MMA
instruction does not know for certain whether its FP32 output will later be stored as BF16. The claim is
only that the microarchitecture may have been designed around the precision that is most useful and
observable in common low-precision training pipelines.

## Current Best Statement

The safest current wording is:

```text
Tensor Core appears to use a shared internal alignment exponent floor at -133.
When all product exponents are below -133, the hardware does not continue lowering the alignment exponent;
instead, it clamps the internal max exponent to -133.

This is likely an underflow/subnormal handling or datapath simplification rule in the Tensor Core accumulator.
The exact value -133 may be chosen because it is the BF16 store/reload observable floor.
It is not strong evidence that Tensor Core is intentionally performing quantization-aware training.
```

## Why This Could Exist In Hardware

Possible hardware reasons:

1. Limit the width or range of the alignment shifter.
2. Simplify guard/sticky generation for extremely tiny products.
3. Avoid expensive denormal handling deep below the useful FP32 accumulation range.
4. Make Tensor Core behavior deterministic across input dtypes through a shared accumulator path.
5. Flush or effectively quantize ultra-small products that are below an internal usefulness threshold.

## Relationship To QAT

A careful distinction:

```text
The exponent floor can create a deterministic quantization-like numerical effect.
Training may adapt to such deterministic hardware behavior.
But that is not the same as hardware intentionally implementing QAT.
```

So a more precise statement is:

```text
This behavior may be one of the low-level numerical effects that training can absorb,
but it should be described as Tensor Core exponent-floor / underflow behavior,
not as built-in quantization-aware training.
```

With the refined dataflow interpretation:

```text
The behavior can be described as being compatible with BF16 training pipelines:
the Tensor Core path may avoid preserving FP32-only tiny values that would disappear after BF16
store/reload anyway.

That is still different from QAT.
QAT explicitly trains a model with simulated quantization effects.
This Tensor Core behavior is a fixed arithmetic/datapath rule that training may adapt to.
```

## Things To Verify Next

### 1. Check Whether The Threshold Is Really Global

Test the same low-exponent setup for:

```text
BF16
FP16
TF32
FP8 variants
other supported Tensor Core input formats
```

Question:

```text
Is the threshold always exactly -133?
```

### 2. Compare Tensor Core MMA With CUDA Core FMA

Run an equivalent computation using normal CUDA scalar/vector FMA rather than Tensor Core MMA.

Interpretation:

```text
If only Tensor Core shows -133, it is a Tensor Core datapath rule.
If CUDA core FMA also shows it, the explanation may involve compiler/lib/device-wide FTZ or denormal policy.
```

### 3. Test Accumulator Type

Compare:

```text
dst=fp32
dst=fp16
other accumulation/output modes if available
```

Question:

```text
Does -133 follow the input dtype, the accumulator dtype, or the MMA instruction family?
```

### 4. Add Nonzero C Input

Test whether `c` participates in the same clamp.

Cases:

```text
all a*b exponents < -133, c exponent < -133
all a*b exponents < -133, c exponent = -133
all a*b exponents < -133, c exponent > -133
```

Question:

```text
Is the floor applied to only product terms, or to the entire add/accumulate set?
```

### 5. Boundary Test

Run very focused cases:

```text
real max exponent = -135
real max exponent = -134
real max exponent = -133
real max exponent = -132
```

Question:

```text
Is the transition exactly at -133?
```

### 6. Lane Count And Distribution

Test:

```text
one nonzero lane
two nonzero lanes
all lanes nonzero
same final sum but different lane distribution
```

Question:

```text
Is the exponent floor independent of lane distribution?
```

## Working Conclusion

Current working conclusion:

```text
-133 should be treated as a shared Tensor Core internal exponent floor.
It is probably related to underflow/subnormal handling in the accumulator path.
It is not BF16-input-specific, even though the value numerically matches BF16's minimum subnormal exponent.
The exact value may be chosen because -133 is the lowest exponent that can survive a BF16 store/reload path,
which is a common dataflow in low-precision training.
It is not best described as hardware QAT.
```

Open question:

```text
Why exactly -133, and is the floor tied to an internal BF16-like datapath, a BF16 store/reload design assumption,
or another shared Tensor Core accumulation rule?
```
