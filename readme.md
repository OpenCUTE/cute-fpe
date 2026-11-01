# CUTE-TensorCore RTL and arithmetic verification

This repository separates arithmetic characterization, the reference model,
the RTL designs, and RTL verification:

- `config/modes.json`: shared mapping between instruction modes, C-model dtype
  values, and RTL opcodes.
- `microbenchmarks/`: directed RTX 5090 MMA probes for the arithmetic characterization, with reference logs and [paper-claim-to-evidence mapping](microbenchmarks/CLAIMS.md).
- `cmodel/`: bit-oriented C reference model and the existing NVIDIA MMA code.
  See [cmodel/README.md](cmodel/README.md).
- `rtl/`: final RTL. `pipe0`-`pipe8` hold the per-pipeline sources and
  `fpe/rtl` is the flattened compile directory. See [rtl/readme.md](rtl/readme.md).
- `comparative_code/`: the three PPA ablation RTL variants (`add_tree_reuse`,
  `mul_reduce`, `reduce_reg`), each laid out like `rtl/`.
- `rtl_cmodel_compare/`: C-model/RTL vector conversion and the Verilator flow.
  It tests `rtl/` and every `comparative_code/` variant against the C model for
  RTL opcodes 0-21. See [rtl_cmodel_compare/README.md](rtl_cmodel_compare/README.md).

The microbenchmark directory has a layout self-check and nine directed RTX 5090
tests for the 16 FP32-output floating-point MMA instructions: output
uniformity, accumulation order, hidden precision, rounding, special values,
scale order, exponent floors, zero-exponent semantics, and FP4/FP6 low-bit
decoding. It is plain CUDA: `make && ./run_all.sh` inside `microbenchmarks/`
builds and runs everything; see [microbenchmarks/README.md](microbenchmarks/README.md).

## RTL verification quick start

```bash
cd rtl_cmodel_compare
make run DTYPE=1 VEC_COUNT=1000                  # main RTL, one dtype
make run RTL=mul_reduce DTYPE=20 VEC_COUNT=1000  # one ablation variant
./scripts/run_regression.sh                      # all trees x dtypes 0-21
```
