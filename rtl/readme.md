This directory describes how to reproduce the RTL optimization part of the ICCD paper.

## Layout

- `pipe0/` - `pipe8/`: sources split by the 9 pipeline stages. Edit RTL here. Each stage's top is `pipeN_stage.sv`.
- `fpe/rtl/`: flat compile directory that holds every `pipe*` source side by side; simulation and synthesis use it. The top module is `top.sv`.
  - `defines.sv` here is the single source of truth: width parameters (`ReduceWidth` defaults to 512) and the RTL opcode 0-21 definitions.
  - `CuteFpeSv.sv` is the IO adapter that CUTE instantiates as a BlackBox; it has no `pipe*` source counterpart.
- `sync_pipe_to_fpe.py`: synchronizes `pipe*` and `fpe/rtl`.

Main function of each stage:

| Stage | Function |
|---|---|
| pipe0 | opcode decode, A/B data decode, exception and zero detection |
| pipe1 | scale decode, mantissa multiplication |
| pipe2 | C decode, maximum-exponent comparison |
| pipe3 | exponent alignment shift |
| pipe4 - pipe6 | three-level adder tree, C shift and accumulation |
| pipe7 | leading-zero count and normalization |
| pipe8 | exception handling and FP32 output |

## Syncing after an RTL edit

`.sv` sources are copied from `pipe*` into `fpe/rtl`. `defines.sv` goes the other way: `fpe/rtl/defines.sv` is copied into every `pipe*`.

```bash
cd ICCD_comparative
python3 rtl/sync_pipe_to_fpe.py --check   # only report differences
python3 rtl/sync_pipe_to_fpe.py --apply   # copy files
```

## Verifying against the C model

Run from `rtl_cmodel_compare/`; see [rtl_cmodel_compare/README.md](../rtl_cmodel_compare/README.md):

```bash
cd rtl_cmodel_compare
make run DTYPE=1 VEC_COUNT=1000
RTLS=rtl VEC_COUNT=100000000 ./scripts/run_regression.sh
```

Every test first checks that `pipe*` and `fpe/rtl` are in sync and fails immediately if they are not.

## Ablation variants

`../comparative_code/` holds three ablation variants, `add_tree_reuse`, `mul_reduce` and `reduce_reg`. Each has the same layout as this directory and its own `sync_pipe_to_fpe.py`. Compared with this directory, they change:

- `mul_reduce`: pipe1 mantissa multiplication (`sig_mul`, `sig_deal_unit`, `product_sum`).
- `reduce_reg`: data decode, multiplication and alignment in pipe0 - pipe4.
- `add_tree_reuse`: the adder tree in pipe2 - pipe6, plus separate `p2_nvfp4_add.sv` and `p5_int4_add.sv` adders.

Pass `RTL=<variant>` to verify any variant in `rtl_cmodel_compare`, for example `make run RTL=mul_reduce DTYPE=1`.
