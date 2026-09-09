# BF16 Custom Rounding Experiments

This file records custom BF16 replay experiments in a table-first format.

Common setup:

| Item | Value |
| --- | --- |
| Repo | `${C_MODEL_REPO}` |
| Runner | `./all_fmac_test_custom_nv --dtype 2 --replay-vec 0` |
| Required env | `FMAC_TRACE_INTERMEDIATE=1` |
| Live A input | `custom_output/m16n8k16_bf16_bf16_to_fp32_a_double.bin.log` |
| Live B input | `custom_output/m16n8k16_bf16_bf16_to_fp32_b_double.bin.log` |
| Live align trace | `custom_debug/mant_align_bf16_to_fp32.txt.log` |
| Live sum trace | `custom_debug/mant_sum_bf16_to_fp32.txt.log` |
| Note | `stored_d=0x800002b7` is stale; replay exit code 1 is expected when stored D differs. Compare `my` and `nv`. |

Column meanings:

| Column | Meaning |
| --- | --- |
| Time | Time when the experiment/trace was produced. |
| Lane | Logical lane index in `mant_align_bf16_to_fp32.txt.log`; lane line number is `lane + 2` after the header. |
| Input | BF16 input words used for that lane. |
| exp_ab | Product exponent before alignment. |
| rsft | Alignment right shift amount from trace. |
| mant_pre | Mantissa before alignment, printed as `mant_mul(point alignment)`. |
| mant_shifted | Mantissa after alignment. |
| sign | Sign of the lane contribution after `a*b`. |

## Experiment Index

| ID | Time | Description | Trace dir | Non-zero lane construction | Partial sum | Complete sum | my | nv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E001 | 2026-05-31 21:49:11 CST | split negative terms, net `-0x20000` | `custom_debug/exp_split_neg_0x20000/` | `-(0xff00 + 0xfffe + 0x00f2 + 0x0010)` plus lane9 `+0x2000000` | `fffffffffffe0000` | `0000000001fe0000` | `0x00000000` | `0x00000001` |
| E002 | 2026-05-31 21:49:45 CST | large negative plus positive compensation, net `-0x20000` | `custom_debug/exp_large_cancel_0x20000/` | `-0x40000 + 0x20000` plus lane9 `+0x2000000` | `fffffffffffe0000` | `0000000001fe0000` | `0x00000000` | `0x00000000` |
| E003 | 2026-06-01 16:04:34 CST | only lane1 negative `-0x10000`, lane9 nonzero | `custom_debug/` | `-0x10000` plus lane9 `+0x2000000` | `ffffffffffff0000` | `0000000001ff0000` | `0x00000000` | `0x00000000` |
| E004 | 2026-06-01 16:19:23 CST | split lane0/lane1 negative, net `-0xffff`, lane9 nonzero | `custom_debug/` | `-(0x0001 + 0xfffe)` plus lane9 `+0x2000000` | `ffffffffffff0001` | `0000000001ff0001` | `0x00000000` | `0x00000001` |
| E005 | 2026-06-01 16:23:32 CST | lane0 and lane1 both negative `-0x8000`, lane9 nonzero | `custom_debug/` | `-(0x8000 + 0x8000)` plus lane9 `+0x2000000` | `ffffffffffff0000` | `0000000001ff0000` | `0x00000000` | `0x00000001` |
| E006 | 2026-06-01 16:26:51 CST | lane0 and lane1 both negative `-0x10000`, lane9 nonzero | `custom_debug/` | `-(0x10000 + 0x10000)` plus lane9 `+0x2000000` | `fffffffffffe0000` | `0000000001fe0000` | `0x00000000` | `0x00000000` |
| E007 | 2026-06-01 16:29:47 CST | two logical `-0xffff` values via four-lane split, lane9 nonzero | `custom_debug/` | `-((0x0001 + 0xfffe) + (0x0001 + 0xfffe))` plus lane9 `+0x2000000` | `fffffffffffe0002` | `0000000001fe0002` | `0x00000000` | `0x00000001` |
| E008 | 2026-06-01 16:33:25 CST | E007 signs flipped for all non-zero lanes | `custom_debug/` | `+(0x0001 + 0xfffe + 0x0001 + 0xfffe)` and lane9 `-0x2000000` | `000000000001fffe` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| E009 | 2026-06-01 16:47:24 CST | E008 non-zero values rotated through lane positions 0..15 | `custom_debug/position_sweep_E008/` | same five `(a,b)` pairs as E008, circular shift `0..15` | varies by group, complete sum constant | `fffffffffe01fffe` for all shifts | `0x00000000` for all shifts | `0x80000001` for all shifts |
| E010 | 2026-06-01 16:53:49 CST | four logical `-0xffff` values rotated through lane positions 0..15 | `custom_debug/position_sweep_four_neg_ffff/` | four copies of `-(0x0001 + 0xfffe)` plus lane `+0x2000000`, circular shift `0..15` | varies by group, complete sum constant | `0000000001fc0004` for all shifts | `0x00000000` for all shifts | `0x00000001` for all shifts |
| E011 | 2026-06-01 17:18:39 CST | all lanes except lane9 are negative `-0xfffe`, lane9 nonzero | `custom_debug/all_except_lane9_neg_fffe/` | 15 lanes `-0xfffe` plus lane9 `+0x2000000` | groups: `fffffffffffc0008`, `fffffffffffc0008`, `0000000001fd0006`, `fffffffffffc0008` | `0000000001f1001e` | `0x00000000` | `0x00000001` |
| E012 | 2026-06-01 17:51:35 CST | lane0 positive `+0x0001`, lane1 negative `-0x10000`, lane9 nonzero | `custom_debug/lane0_pos1_lane1_neg10000_lane9_pos2000000/` | `+0x0001 - 0x10000` plus lane9 `+0x2000000` | groups: `ffffffffffff0001`, `0000000000000000`, `0000000002000000`, `0000000000000000` | `0000000001ff0001` | `0x00000000` | `0x00000000` |
| E013 | 2026-06-01 19:37:10 CST | E012 with max exponent raised from `-149` to `-148` | `custom_debug/lane0_pos1_lane1_neg10000_lane9_expmax_neg148/` | lane0 guard-only `+0x0001`, lane1 aligned to `-0x8000`, lane9 `+0x2000000` at `exp_ab=-148` | groups: `ffffffffffff8000`, `0000000000000000`, `0000000002000000`, `0000000000000000` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| E014 | 2026-06-01 19:43:16 CST | E013 with lane1 shifted mantissa changed from `0x8000` to `0x7fff` | `custom_debug/lane1_neg7fff_expmax_neg148/` | lane0 guard-only `+0x0001`, lane1 aligned to `-0x7fff`, lane9 `+0x2000000` at `exp_ab=-148` | groups: `ffffffffffff8001`, `0000000000000000`, `0000000002000000`, `0000000000000000` | `0000000001ff8001` | `0x00000001` | `0x00000002` |
| E015 | 2026-06-01 19:45:53 CST | only lane1 negative `-0x8000`, lane9 unchanged at `exp_ab=-148` | `custom_debug/lane1_neg8000_lane9_expmax_neg148_only/` | lane1 aligned to `-0x8000`, lane9 `+0x2000000`, all other lanes zero | groups: `ffffffffffff8000`, `0000000000000000`, `0000000002000000`, `0000000000000000` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| E016 | 2026-06-01 19:50:00 CST | E015 two-lane combination rotated through all lane positions | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/` | `-0x8000` and `+0x2000000` circular shift `0..15`, relative offset 8 | varies by group, complete sum constant | `0000000001ff8000` for all shifts | `0x00000001` for all shifts | `0x00000001` for all shifts |
| E017 | 2026-06-01 19:59:29 CST | all lanes except lane9 are negative `-0x7fff`, lane9 unchanged at `exp_ab=-148` | `custom_debug/all_except_lane9_neg7fff_expmax_neg148/` | 15 lanes `-0x7fff` plus lane9 `+0x2000000` | groups: `fffffffffffe0004`, `fffffffffffe0004`, `0000000001fe8003`, `fffffffffffe0004` | `0000000001f8800f` | `0x00000001` | `0x00000002` |
| E018 | 2026-06-01 20:07:47 CST | E017 with max exponent raised from `-148` to `-147` | `custom_debug/all_except_lane9_neg7fff_expmax_neg147/` | 15 lanes from same `a=0x0097 b=0xaf59`, now align to `0x3fff` with `guard=1`, lane9 `+0x2000000` at `exp_ab=-147` | groups: `ffffffffffff0004`, `ffffffffffff0004`, `0000000001ff4003`, `ffffffffffff0004` | `0000000001fc400f` | `0x00000003` | `0x00000004` |
| E019 | 2026-06-01 20:16:32 CST | E018 with non-lane9 shifted mantissas changed to `0x4000` | `custom_debug/all_except_lane9_neg4000_expmax_neg147/` | 15 lanes `-0x4000` plus lane9 `+0x2000000` at `exp_ab=-147` | groups: `ffffffffffff0000`, `ffffffffffff0000`, `0000000001ff4000`, `ffffffffffff0000` | `0000000001fc4000` | `0x00000003` | `0x00000003` |
| E020 | 2026-06-01 20:23:03 CST | E019 with lane9 changed from `0x2000000` to `0x3000000` | `custom_debug/all_except_lane9_neg4000_lane9_pos3000000_expmax_neg147/` | 15 lanes `-0x4000` plus lane9 `+0x3000000` at `exp_ab=-147` | groups: `ffffffffffff0000`, `ffffffffffff0000`, `0000000002ff4000`, `ffffffffffff0000` | `0000000002fc4000` | `0x00000005` | `0x00000005` |
| E021 | 2026-06-01 20:44:57 CST | E020 with max exponent changed from `-147` to `-148` | `custom_debug/all_except_lane9_neg8000_lane9_pos3000000_expmax_neg148/` | 15 lanes `-0x8000` plus lane9 `+0x3000000` at `exp_ab=-148` | groups: `fffffffffffe0000`, `fffffffffffe0000`, `0000000002fe8000`, `fffffffffffe0000` | `0000000002f88000` | `0x00000002` | `0x00000002` |
| E022 | 2026-06-01 21:14:44 CST | exponent sweep for lane9 mantissa `0x3000000` and four non-lane cases | `custom_debug/exp_sweep_max_exp_lane9_mant3000000_nonlane_cases/` | exp_max `-147..-120`, lane9 mantissa fixed, non-lane exp fixed at `-158`, cases a/b/c/d | varies | varies; see `results.csv` | all 112 cases match nv | all 112 cases match my |
| E023 | 2026-06-02 16:59:29 CST | GPU recheck of E021 with byte-identical saved inputs and identical align/sum traces | `custom_debug/recheck_E021_gpu_20260602/` | 15 lanes `-0x8000` plus lane9 `+0x3000000` at `exp_ab=-148`; same as E021 | groups: `fffffffffffe0000`, `fffffffffffe0000`, `0000000002fe8000`, `fffffffffffe0000` | `0000000002f88000` | `0x00000002` | `0x00000002` |
| E024 | 2026-06-02 17:06:58 CST | E021 with every non-lane9 shifted mantissa changed from `0x8000` to `0x7fff` | `custom_debug/all_except_lane9_neg7fff_lane9_pos3000000_expmax_neg148/` | 15 lanes `-0x7fff` plus lane9 `+0x3000000` at `exp_ab=-148` | groups: `fffffffffffe0004`, `fffffffffffe0004`, `0000000002fe8003`, `fffffffffffe0004` | `0000000002f8800f` | `0x00000002` | `0x00000003` |
| E025 | 2026-06-02 17:13:24 CST | E024 with lane9 product exponent raised from `-148` to `-147` | `custom_debug/all_except_lane9_neg7fff_lane9_pos3000000_expmax_neg147/` | 15 lanes align as `-0x3fff` with `guard=1`, plus lane9 `+0x3000000` at `exp_ab=-147` | groups: `ffffffffffff0004`, `ffffffffffff0004`, `0000000002ff4003`, `ffffffffffff0004` | `0000000002fc400f` | `0x00000005` | `0x00000006` |
| E026 | 2026-06-02 17:18:25 CST | E025 with every non-lane9 shifted mantissa changed from `0x3fff` to exact `0x4000` | `custom_debug/recheck_E020_after_E025_neg4000_lane9_pos3000000_expmax_neg147/` | 15 lanes exact `-0x4000` plus lane9 `+0x3000000` at `exp_ab=-147`; rechecks E020 | groups: `ffffffffffff0000`, `ffffffffffff0000`, `0000000002ff4000`, `ffffffffffff0000` | `0000000002fc4000` | `0x00000005` | `0x00000005` |
| E027 | 2026-06-02 17:28:04 CST | E026 with lane9 mantissa changed from `0x3000000` to `0x2c00000` | `custom_debug/all_except_lane9_neg4000_lane9_pos2c00000_expmax_neg147/` | 15 lanes exact `-0x4000` plus lane9 `+0x2c00000` at `exp_ab=-147` | groups: `ffffffffffff0000`, `ffffffffffff0000`, `0000000002bf4000`, `ffffffffffff0000` | `0000000002bc4000` | `0x00000005` | `0x00000005` |
| E028 | 2026-06-02 17:37:38 CST | E027 with every non-lane9 shifted mantissa changed from exact `0x4000` to `0x3fff` | `custom_debug/all_except_lane9_neg3fff_lane9_pos2c00000_expmax_neg147/` | 15 lanes align as `-0x3fff` with `guard=1`, plus lane9 `+0x2c00000` at `exp_ab=-147` | groups: `ffffffffffff0004`, `ffffffffffff0004`, `0000000002bf4003`, `ffffffffffff0004` | `0000000002bc400f` | `0x00000005` | `0x00000005` |
| E029 | 2026-06-02 17:47:34 CST | E028 with lane9 mantissa changed from `0x2c00000` to `0x2800000` | `custom_debug/all_except_lane9_neg3fff_lane9_pos2800000_expmax_neg147/` | 15 lanes align as `-0x3fff` with `guard=1`, plus lane9 `+0x2800000` at `exp_ab=-147` | groups: `ffffffffffff0004`, `ffffffffffff0004`, `00000000027f4003`, `ffffffffffff0004` | `00000000027c400f` | `0x00000004` | `0x00000005` |
| E030 | 2026-06-02 17:53:52 CST | E029 with every non-lane9 shifted mantissa changed from `0x3fff` to exact `0x4000` | `custom_debug/all_except_lane9_neg4000_lane9_pos2800000_expmax_neg147/` | 15 lanes exact `-0x4000` plus lane9 `+0x2800000` at `exp_ab=-147` | groups: `ffffffffffff0000`, `ffffffffffff0000`, `00000000027f4000`, `ffffffffffff0000` | `00000000027c4000` | `0x00000004` | `0x00000004` |
| E031 | 2026-06-02 18:19:13 CST | E030 with every non-lane9 product exponent changed from `-158` to `-159` | `custom_debug/all_except_lane9_expab_neg159_mant2000_lane9_pos2800000_expmax_neg147/` | 15 lanes exact `-0x2000` plus lane9 `+0x2800000` at `exp_ab=-147` | groups: `ffffffffffff8000`, `ffffffffffff8000`, `00000000027fa000`, `ffffffffffff8000` | `00000000027e2000` | `0x00000004` | `0x00000005` |
| E032 | 2026-06-02 18:24:49 CST | E031 with every non-lane9 product exponent changed from `-159` to `-157` | `custom_debug/all_except_lane9_expab_neg157_mant8000_lane9_pos2800000_expmax_neg147/` | 15 lanes exact `-0x8000` plus lane9 `+0x2800000` at `exp_ab=-147` | groups: `fffffffffffe0000`, `fffffffffffe0000`, `00000000027e8000`, `fffffffffffe0000` | `0000000002788000` | `0x00000004` | `0x00000004` |
| E033 | 2026-06-02 18:33:16 CST | E032 with every non-lane9 shifted mantissa changed from `0x8000` to exact `0x4000` while keeping `exp_ab=-157` | `custom_debug/all_except_lane9_expab_neg157_mant4000_lane9_pos2800000_expmax_neg147/` | 15 lanes exact `-0x4000` via subnormal `a=0x0040 b=0xb000`, plus lane9 `+0x2800000` at `exp_ab=-147` | groups: `ffffffffffff0000`, `ffffffffffff0000`, `00000000027f4000`, `ffffffffffff0000` | `00000000027c4000` | `0x00000004` | `0x00000004` |
| E034 | 2026-06-02 18:45:52 CST | E033 request for non-lane9 `0x3fff` while keeping `exp_ab=-157`; exact odd shifted value is impossible, so use maximum lower `0x3ffc` | `custom_debug/all_except_lane9_expab_neg157_mant3ffc_lane9_pos2800000_expmax_neg147/` | 15 lanes exact `-0x3ffc` via subnormal `a=0x003f b=0xb002`, plus lane9 `+0x2800000` at `exp_ab=-147` | groups: `ffffffffffff0010`, `ffffffffffff0010`, `00000000027f400c`, `ffffffffffff0010` | `00000000027c403c` | `0x00000004` | `0x00000005` |
| E035 | 2026-06-02 21:41:25 CST | E034 with lane9 exponent changed from `-147` to `-146` and lane9 mantissa from `0x2800000` to `0x2000000` | `custom_debug/all_except_lane9_expab_neg157_mant3ffc_lane9_pos2000000_expmax_neg146/` | 15 lanes become `-0x1ffe` after exp_max rises to `-146`, plus lane9 `+0x2000000` at `exp_ab=-146` | groups: `ffffffffffff8008`, `ffffffffffff8008`, `0000000001ffa006`, `ffffffffffff8008` | `0000000001fe201e` | `0x00000007` | `0x00000008` |
| E036 | 2026-06-02 21:46:47 CST | E035 with every non-lane9 shifted mantissa changed from `0x1ffe` to exact `0x2000` | `custom_debug/all_except_lane9_expab_neg157_mant2000_lane9_pos2000000_expmax_neg146/` | 15 lanes exact `-0x2000` via subnormal `a=0x0040 b=0xb000`, plus lane9 `+0x2000000` at `exp_ab=-146` | groups: `ffffffffffff8000`, `ffffffffffff8000`, `0000000001ffa000`, `ffffffffffff8000` | `0000000001fe2000` | `0x00000007` | `0x00000007` |
| E037 | 2026-06-02 21:55:38 CST | Experiment 1 sweep: keep all non-lane9 shifted mantissas at exact `0x2000` while changing non-lane9 `exp_ab` | `custom_debug/sweep_nonlane_shift2000_expab_lane9_pos2000000_expmax_neg146/` | lane9 fixed `+0x2000000` at `exp_ab=-146`; non-lane `exp_ab=-158..-151`, all exact `-0x2000` | same for all cases: groups `ffffffffffff8000`, `ffffffffffff8000`, `0000000001ffa000`, `ffffffffffff8000` | same for all cases: `0000000001fe2000` | all `0x00000007` | all `0x00000007` |
| E038 | 2026-06-02 21:55:38 CST | Experiment 2 sweep: keep all non-lane9 shifted mantissas at `0x1fff` while changing non-lane9 `exp_ab` | `custom_debug/sweep_nonlane_shift1fff_expab_lane9_pos2000000_expmax_neg146/` | lane9 fixed `+0x2000000` at `exp_ab=-146`; non-lane `exp_ab=-159,-158`, both floor to `-0x1fff` with guard=1 | same for both cases: groups `ffffffffffff8004`, `ffffffffffff8004`, `0000000001ffa003`, `ffffffffffff8004` | same for both cases: `0000000001fe200f` | all `0x00000007` | all `0x00000008` |
| E039 | 2026-06-02 22:16:10 CST | Experiment 1 requested sweep: keep lane9 mantissa `0x2000000`, sweep lane9 exponent from `-146` through `-127`; non-lane exponent fixed and mant_pre fixed at `0x40000` | `custom_debug/sweep_lane9_exp_neg146_to_neg127_nonlane_expab_neg151_mantpre40000/` | lane9 `a=0x0080`, `b=0x3580..0x3f00`; non-lane fixed `a=0x0001 b=0xb300`, `exp_ab=-151`, `mant_pre=0x40000` | varies; see `summary.md` | varies; final `-127` case is `0000000002000000` | all match nv | all match my |
| E040 | 2026-06-02 22:16:10 CST | Experiment 2 requested construction check: keep non-lane exponent fixed and set non-lane mant_pre to `0x3ffff`, or maximum lower value | `custom_debug/nonlane_expab_neg151_mantpre3ffff_representability/` | impossible: with fixed non-lane `exp_ab=-151`, smallest non-zero BF16 product has `mant_pre=0x40000`; no lower non-zero value exists | not run | not run | n/a | n/a |
| E041 | 2026-06-02 22:32:51 CST | Requested sweep: set every non-lane9 lane to `exp_ab=-159`, `mant_pre=0x3fff800`; keep lane9 mantissa `0x2000000` and sweep lane9 exponent from `-146` through `-127` | `custom_debug/sweep_lane9_exp_neg146_to_neg127_nonlane_expab_neg159_mantpre3fff800/` | lane9 `a=0x0080`, `b=0x3580..0x3f00`; non-lane fixed `a=0x0097 b=0xaf59`, `exp_ab=-159`, `mant_pre=0x3fff800` | varies; see `summary.md` | varies; final `-127` case is `0000000002000000` | `-146..-134`: my one less than nv; `-133..-127`: match | see summary |
| E042 | 2026-06-03 13:57:03 CST | lane9 set to `a=0x1c11 b=0x1ce1`; every non-lane9 lane is positive with exponent sum `-155` and raw mantissa product `1.0` | `custom_debug/lane9_1c11_1ce1_nonlane_expab_neg155_mant1_pos/` | 15 positive non-lane lanes `a=0x0080 b=0x3100`, plus positive lane9 `a=0x1c11 b=0x1ce1` | groups: `0000000000002000`, `0000000000002000`, `0000000003fba000`, `0000000000002000` | `0000000003fc0000` | `0x000001fe` | `0x000001fe` |
| E043 | 2026-06-03 14:15:35 CST | E042 with every lane product exponent sum decreased by 1 while preserving mantissas and positive signs | `custom_debug/lane9_1c11_1c61_nonlane_expab_neg156_mant1_pos_all_exp_minus1/` | 15 positive non-lane lanes `a=0x0080 b=0x3080`, plus positive lane9 `a=0x1c11 b=0x1c61` | groups: `0000000000002000`, `0000000000002000`, `0000000003fba000`, `0000000000002000` | `0000000003fc0000` | `0x000000ff` | `0x000000ff` |
| E044 | 2026-06-03 14:20:57 CST | E043 with every lane product exponent sum decreased by 1 again while preserving mantissas and positive signs | `custom_debug/lane9_1c11_1be1_nonlane_expab_neg157_mant1_pos_all_exp_minus2/` | 15 positive non-lane lanes `a=0x0080 b=0x3000`, plus positive lane9 `a=0x1c11 b=0x1be1` | groups: `0000000000002000`, `0000000000002000`, `0000000003fba000`, `0000000000002000` | `0000000003fc0000` | `0x0000007f` | `0x0000007f` |
| E045 | 2026-06-03 14:49:36 CST | E044 with every lane product exponent sum increased by 1 while preserving mantissas and positive signs | `custom_debug/lane9_1c11_1c61_nonlane_expab_neg156_mant1_pos_from_E044_all_exp_plus1/` | 15 positive non-lane lanes `a=0x0080 b=0x3080`, plus positive lane9 `a=0x1c11 b=0x1c61`; same input values as E043 | groups: `0000000000002000`, `0000000000002000`, `0000000003fba000`, `0000000000002000` | `0000000003fc0000` | `0x000000ff` | `0x000000ff` |
| E046 | 2026-06-03 14:57:35 CST | E045 with every lane product exponent sum increased by 1 again while preserving mantissas and positive signs | `custom_debug/lane9_1c11_1ce1_nonlane_expab_neg155_mant1_pos_from_E045_all_exp_plus1/` | 15 positive non-lane lanes `a=0x0080 b=0x3100`, plus positive lane9 `a=0x1c11 b=0x1ce1`; same input values as E042 | groups: `0000000000002000`, `0000000000002000`, `0000000003fba000`, `0000000000002000` | `0000000003fc0000` | `0x000001fe` | `0x000001fe` |
| E047 | 2026-06-03 15:13:55 CST | E046 with lane9 mantissa product changed from `0x3fb8800` to exact `0x2000000`, keeping lane9 `exp_ab=-141` | `custom_debug/lane9_mant2000000_expab_neg141_nonlane_expab_neg155_mant1_pos_from_E046/` | 15 positive non-lane lanes `a=0x0080 b=0x3100`; lane9 changed to positive `a=0x1c00 b=0x1c80` | groups: `0000000000002000`, `0000000000002000`, `0000000002001800`, `0000000000002000` | `0000000002007800` | `0x00000100` | `0x00000100` |
| E048 | 2026-06-03 15:22:41 CST | E047 with every non-lane9 product sign changed from positive to negative; lane9 stays positive | `custom_debug/nonlane_neg_lane9_pos_mant2000000_expab_neg141_from_E047/` | 15 negative non-lane lanes `a=0x0080 b=0xb100`; lane9 positive `a=0x1c00 b=0x1c80` | groups: `ffffffffffffe000`, `ffffffffffffe000`, `0000000001ffe800`, `ffffffffffffe000` | `0000000001ff8800` | `0x000000ff` | `0x000000ff` |
| E049 | 2026-06-03 15:33:18 CST | Sweep from E048: repeatedly decrease all non-lane9 `exp_ab` while keeping non-lane `mant_pre=0x2000000` and negative sign until first `my/nv` mismatch | `custom_debug/sweep_nonlane_neg_mantpre2000000_expdown_lane9_pos2000000_expab_neg141_from_E048/` | lane9 fixed positive `a=0x1c00 b=0x1c80`; non-lane fixed `a=0x0080`, negative `b=0xb080,0xb000,0xaf80,0xaf00` | first mismatch groups: `fffffffffffffe00`, `fffffffffffffe00`, `0000000001fffe80`, `fffffffffffffe00` | first mismatch `0000000001fff880` | `0x000000ff` | `0x00000100` |
| E050 | 2026-06-03 15:52:50 CST | E046 with every lane product exponent sum decreased by 4 while preserving mantissas and positive signs | `custom_debug/e046_all_exp_minus4_lane9_1c11_1ae1_nonlane_2f00/` | 15 positive non-lane lanes `a=0x0080 b=0x2f00`; lane9 positive `a=0x1c11 b=0x1ae1` | groups: `0000000000002000`, `0000000000002000`, `0000000003fba000`, `0000000000002000` | `0000000003fc0000` | `0x0000001f` | `0x0000001f` |
| E051 | 2026-06-03 16:23:24 CST | From E050-style live state: make lane8 and lane9 identical, set lanes0-6 to `exp_ab=-158`, lane7 to `-155`, lane10 to `-151`, and zero lanes11-15 while preserving input mantissas | `custom_debug/lane8_eq_lane9_lane0_6_exp158_lane7_exp155_lane10_exp151_rest_zero_from_E050/` | lane0-6 `a=0x0080 b=0x2f80`; lane7 `a=0x0080 b=0x3100`; lane8-9 `a=0x1c11 b=0x1ae1`; lane10 `a=0x0080 b=0x3300`; lane11-15 zero | groups: `0000000000004000`, `000000000000b000`, `0000000007ff1000`, `0000000000000000` | `0000000008000000` | `0x00000040` | `0x0000003f` |
| E052 | 2026-06-03 16:41:54 CST | From E051: change lane9 mantissa product from `0x3fb8800` to exact `0x2000000`, zero every lane except lane0/lane9, and flip lane0 product sign negative while preserving lane0 exponent/mantissa magnitude | `custom_debug/lane9_mant2000000_lane0_neg_rest_zero_from_E051/` | lane0 negative `a=0x0080 b=0xaf80`; lane9 positive `a=0x1c00 b=0x1a80`; all other lanes zero | groups: `fffffffffffff000`, `0000000000000000`, `0000000002000000`, `0000000000000000` | `0000000001fff000` | `0x0000000f` | `0x0000000f` |

## E001 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/exp_split_neg_0x20000 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000000 nv=0x00000001
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-31 21:49:11 CST | 0 | `a=0x0080 b=0xaf7f` | `-159` | `10` | `0x0000000003fc0000` | `0x000000000000ff00` | `-` |
| 2026-05-31 21:49:11 CST | 1 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-05-31 21:49:11 CST | 2 | `a=0x0001 b=0xaef2` | `-160` | `11` | `0x0000000000079000` | `0x00000000000000f2` | `-` |
| 2026-05-31 21:49:11 CST | 3 | `a=0x0001 b=0xad00` | `-163` | `14` | `0x0000000000040000` | `0x0000000000000010` | `-` |
| 2026-05-31 21:49:11 CST | 9 | `a=0x0080 b=0x3400` | `-149` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` |

Sum:

```text
partial sum:  ffffffffffffe0000
complete sum: 0000000001fe0000
```

## E002 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/exp_large_cancel_0x20000 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000000 nv=0x00000000
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-31 21:49:45 CST | 0 | `a=0x0080 b=0xb080` | `-156` | `7` | `0x0000000002000000` | `0x0000000000040000` | `-` |
| 2026-05-31 21:49:45 CST | 1 | `a=0x0080 b=0x3000` | `-157` | `8` | `0x0000000002000000` | `0x0000000000020000` | `+` |
| 2026-05-31 21:49:45 CST | 9 | `a=0x0080 b=0x3400` | `-149` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` |

Sum:

```text
partial sum:  ffffffffffffe0000
complete sum: 0000000001fe0000
```

## E003 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000000 nv=0x00000000
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 16:04:34 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `9` | `0x0000000002000000` | `0x0000000000010000` | `-` |
| 2026-06-01 16:04:34 CST | 9 | `a=0x0080 b=0x3400` | `-149` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` |

Sum:

```text
partial sum:  ffffffffffff0000
complete sum: 0000000001ff0000
```

## E004 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000000 nv=0x00000001
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 16:19:23 CST | 0 | `a=0x0080 b=0xa780` | `-174` | `25` | `0x0000000002000000` | `0x0000000000000001` | `-` |
| 2026-06-01 16:19:23 CST | 1 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 16:19:23 CST | 9 | `a=0x0080 b=0x3400` | `-149` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` |

Sum:

```text
partial sum:  ffffffffffff0001
complete sum: 0000000001ff0001
```

## E005 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000000 nv=0x00000001
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 16:23:32 CST | 0 | `a=0x0080 b=0xaf00` | `-159` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 16:23:32 CST | 1 | `a=0x0080 b=0xaf00` | `-159` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 16:23:32 CST | 9 | `a=0x0080 b=0x3400` | `-149` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` |

Sum:

```text
partial sum:  ffffffffffff0000
complete sum: 0000000001ff0000
```

## E006 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000000 nv=0x00000000
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 16:26:51 CST | 0 | `a=0x0080 b=0xaf80` | `-158` | `9` | `0x0000000002000000` | `0x0000000000010000` | `-` |
| 2026-06-01 16:26:51 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `9` | `0x0000000002000000` | `0x0000000000010000` | `-` |
| 2026-06-01 16:26:51 CST | 9 | `a=0x0080 b=0x3400` | `-149` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` |

Sum:

```text
partial sum:  fffffffffffe0000
complete sum: 0000000001fe0000
```

## E007 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000000 nv=0x00000001
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 16:29:47 CST | 0 | `a=0x0080 b=0xa780` | `-174` | `25` | `0x0000000002000000` | `0x0000000000000001` | `-` |
| 2026-06-01 16:29:47 CST | 1 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 16:29:47 CST | 2 | `a=0x0080 b=0xa780` | `-174` | `25` | `0x0000000002000000` | `0x0000000000000001` | `-` |
| 2026-06-01 16:29:47 CST | 3 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 16:29:47 CST | 9 | `a=0x0080 b=0x3400` | `-149` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` |

Sum:

```text
partial sum:  fffffffffffe0002
complete sum: 0000000001fe0002
```

## E008 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000000 nv=0x80000001
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 16:33:25 CST | 0 | `a=0x0080 b=0x2780` | `-174` | `25` | `0x0000000002000000` | `0x0000000000000001` | `+` |
| 2026-06-01 16:33:25 CST | 1 | `a=0x0097 b=0x2f59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `+` |
| 2026-06-01 16:33:25 CST | 2 | `a=0x0080 b=0x2780` | `-174` | `25` | `0x0000000002000000` | `0x0000000000000001` | `+` |
| 2026-06-01 16:33:25 CST | 3 | `a=0x0097 b=0x2f59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `+` |
| 2026-06-01 16:33:25 CST | 9 | `a=0x0080 b=0xb400` | `-149` | `0` | `0x0000000002000000` | `0x0000000002000000` | `-` |

Sum:

```text
partial sum:  000000000001fffe
complete sum: fffffffffe01fffe
```

## E009 Position Sweep

Sweep summary:

```text
custom_debug/position_sweep_E008/summary.md
```

Method:

```text
Keep the five non-zero E008 (a,b) pairs fixed.
Apply circular lane shift 0..15.
Run each case into custom_debug/position_sweep_E008/shift_XX/.
Restore live custom_output/custom_debug to shift 0 after the sweep.
```

Result:

| Time | Sweep | Lane range | Complete sum | my | nv | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 16:47:24 CST | E008 circular shift | `0..15` | `0xfffffffffe01fffe` for every shift | `0x00000000` for every shift | `0x80000001` for every shift | No position effect observed for this rotation sweep. |

Per-shift result:

| shift | trace dir | complete sum | my | nv |
| --- | --- | --- | --- | --- |
| 0 | `custom_debug/position_sweep_E008/shift_00/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 1 | `custom_debug/position_sweep_E008/shift_01/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 2 | `custom_debug/position_sweep_E008/shift_02/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 3 | `custom_debug/position_sweep_E008/shift_03/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 4 | `custom_debug/position_sweep_E008/shift_04/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 5 | `custom_debug/position_sweep_E008/shift_05/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 6 | `custom_debug/position_sweep_E008/shift_06/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 7 | `custom_debug/position_sweep_E008/shift_07/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 8 | `custom_debug/position_sweep_E008/shift_08/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 9 | `custom_debug/position_sweep_E008/shift_09/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 10 | `custom_debug/position_sweep_E008/shift_10/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 11 | `custom_debug/position_sweep_E008/shift_11/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 12 | `custom_debug/position_sweep_E008/shift_12/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 13 | `custom_debug/position_sweep_E008/shift_13/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 14 | `custom_debug/position_sweep_E008/shift_14/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |
| 15 | `custom_debug/position_sweep_E008/shift_15/` | `fffffffffe01fffe` | `0x00000000` | `0x80000001` |

## E010 Position Sweep

Sweep summary:

```text
custom_debug/position_sweep_four_neg_ffff/summary.md
```

Method:

```text
Four logical -0xffff values are represented as four copies of -(0x0001 + 0xfffe).
A positive +0x2000000 lane is kept to set exp_max=-149.
The nine non-zero (a,b) pairs are circularly shifted through logical product lanes 0..15.
Live custom_output/custom_debug were restored to shift 0 after the sweep.
```

Result:

| Time | Sweep | Lane range | Complete sum | my | nv | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 16:53:49 CST | four logical `-0xffff` circular shift | `0..15` | `0x0000000001fc0004` for every shift | `0x00000000` for every shift | `0x00000001` for every shift | No position effect observed for this rotation sweep. |

Per-shift result:

| shift | trace dir | complete sum | my | nv |
| --- | --- | --- | --- | --- |
| 0 | `custom_debug/position_sweep_four_neg_ffff/shift_00/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 1 | `custom_debug/position_sweep_four_neg_ffff/shift_01/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 2 | `custom_debug/position_sweep_four_neg_ffff/shift_02/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 3 | `custom_debug/position_sweep_four_neg_ffff/shift_03/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 4 | `custom_debug/position_sweep_four_neg_ffff/shift_04/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 5 | `custom_debug/position_sweep_four_neg_ffff/shift_05/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 6 | `custom_debug/position_sweep_four_neg_ffff/shift_06/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 7 | `custom_debug/position_sweep_four_neg_ffff/shift_07/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 8 | `custom_debug/position_sweep_four_neg_ffff/shift_08/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 9 | `custom_debug/position_sweep_four_neg_ffff/shift_09/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 10 | `custom_debug/position_sweep_four_neg_ffff/shift_10/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 11 | `custom_debug/position_sweep_four_neg_ffff/shift_11/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 12 | `custom_debug/position_sweep_four_neg_ffff/shift_12/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 13 | `custom_debug/position_sweep_four_neg_ffff/shift_13/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 14 | `custom_debug/position_sweep_four_neg_ffff/shift_14/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |
| 15 | `custom_debug/position_sweep_four_neg_ffff/shift_15/` | `0000000001fc0004` | `0x00000000` | `0x00000001` |

## E011 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg_fffe/
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000000 nv=0x00000001
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 17:18:39 CST | 0 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 1 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 2 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 3 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 4 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 5 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 6 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 7 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 8 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 9 | `a=0x0080 b=0x3400` | `-149` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` |
| 2026-06-01 17:18:39 CST | 10 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 11 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 12 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 13 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 14 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |
| 2026-06-01 17:18:39 CST | 15 | `a=0x0097 b=0xaf59` | `-159` | `10` | `0x0000000003fff800` | `0x000000000000fffe` | `-` |

Sum:

```text
partial sum:  fffffffffffc0008
partial sum:  fffffffffffc0008
partial sum:  0000000001fd0006
partial sum:  fffffffffffc0008
complete sum: 0000000001f1001e
```

Interpretation:

```text
15 * -0xfffe = -0xeffe2
0x2000000 - 0xeffe2 = 0x1f1001e
```

## E012 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved trace dir:

```text
custom_debug/lane0_pos1_lane1_neg10000_lane9_pos2000000/
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000000 nv=0x00000000
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 17:51:35 CST | 0 | `a=0x0080 b=0x2780` | `-174` | `25` | `0x0000000002000000` | `0x0000000000000001` | `+` |
| 2026-06-01 17:51:35 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `9` | `0x0000000002000000` | `0x0000000000010000` | `-` |
| 2026-06-01 17:51:35 CST | 9 | `a=0x0080 b=0x3400` | `-149` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` |

Sum:

```text
partial sum:  ffffffffffff0001
partial sum:  0000000000000000
partial sum:  0000000002000000
partial sum:  0000000000000000
complete sum: 0000000001ff0001
```

Interpretation:

```text
0x2000000 - 0x10000 + 0x0001 = 0x1ff0001
```

## E013 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved trace dir:

```text
custom_debug/lane0_pos1_lane1_neg10000_lane9_expmax_neg148/
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000001 nv=0x00000001
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 19:37:10 CST | 0 | `a=0x0080 b=0x2780` | `-174` | `26` | `0x0000000002000000` | `0x0000000000000000` | `+` | `1` | `0` |
| 2026-06-01 19:37:10 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` |
| 2026-06-01 19:37:10 CST | 9 | `a=0x0080 b=0x3480` | `-148` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff8000
partial sum:  0000000000000000
partial sum:  0000000002000000
partial sum:  0000000000000000
complete sum: 0000000001ff8000
```

Interpretation:

```text
Raising exp_max from -149 to -148 shifts lane0 one more bit right.
lane0 +0x0001 becomes mant_shifted=0 with guard=1, sticky=0.
lane1 -0x10000 becomes -0x8000.
lane9 remains +0x2000000 at the new exp_max.
```

## E014 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved trace dir:

```text
custom_debug/lane1_neg7fff_expmax_neg148/
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000001 nv=0x00000002
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 19:43:16 CST | 0 | `a=0x0080 b=0x2780` | `-174` | `26` | `0x0000000002000000` | `0x0000000000000000` | `+` | `1` | `0` |
| 2026-06-01 19:43:16 CST | 1 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` |
| 2026-06-01 19:43:16 CST | 9 | `a=0x0080 b=0x3480` | `-148` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff8001
partial sum:  0000000000000000
partial sum:  0000000002000000
partial sum:  0000000000000000
complete sum: 0000000001ff8001
```

Observation:

```text
Changing lane1 from -0x8000 to -0x7fff changed my from 0x00000001 to 0x00000001,
but changed nv from 0x00000001 to 0x00000002.
```

## E015 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved trace dir:

```text
custom_debug/lane1_neg8000_lane9_expmax_neg148_only/
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000001 nv=0x00000001
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 19:45:53 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` |
| 2026-06-01 19:45:53 CST | 9 | `a=0x0080 b=0x3480` | `-148` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff8000
partial sum:  0000000000000000
partial sum:  0000000002000000
partial sum:  0000000000000000
complete sum: 0000000001ff8000
```

## E016 Position Sweep

Sweep summary:

```text
custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/summary.md
```

Method:

```text
The two non-zero products from E015 were circularly shifted through lanes 0..15.
The negative product is -0x8000 at exp_max=-148.
The positive product is +0x2000000 at exp_ab=-148.
The relative offset between the two products was kept at 8 lanes.
Live custom_output/custom_debug were restored to shift 0 after the sweep.
```

Result:

| Time | Sweep | Lane range | Complete sum | my | nv | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 19:50:00 CST | E015 two-lane circular shift | `0..15` | `0x0000000001ff8000` for every shift | `0x00000001` for every shift | `0x00000001` for every shift | No position effect observed for this two-lane rotation sweep. |

Per-shift result:

| shift | negative lane | positive lane | trace dir | complete sum | my | nv |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 9 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_00/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 1 | 2 | 10 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_01/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 2 | 3 | 11 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_02/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 3 | 4 | 12 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_03/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 4 | 5 | 13 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_04/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 5 | 6 | 14 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_05/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 6 | 7 | 15 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_06/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 7 | 8 | 0 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_07/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 8 | 9 | 1 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_08/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 9 | 10 | 2 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_09/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 10 | 11 | 3 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_10/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 11 | 12 | 4 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_11/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 12 | 13 | 5 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_12/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 13 | 14 | 6 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_13/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 14 | 15 | 7 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_14/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |
| 15 | 0 | 8 | `custom_debug/position_sweep_lane1_neg8000_lane9_expmax_neg148/shift_15/` | `0000000001ff8000` | `0x00000001` | `0x00000001` |

## E017 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg7fff_expmax_neg148/
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000001 nv=0x00000002
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 19:59:29 CST | 0 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 1 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 2 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 3 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 4 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 5 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 6 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 7 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 8 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 9 | `a=0x0080 b=0x3480` | `-148` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` |
| 2026-06-01 19:59:29 CST | 10 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 11 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 12 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 13 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 14 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |
| 2026-06-01 19:59:29 CST | 15 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` |

Sum:

```text
partial sum:  fffffffffffe0004
partial sum:  fffffffffffe0004
partial sum:  0000000001fe8003
partial sum:  fffffffffffe0004
complete sum: 0000000001f8800f
```

Interpretation:

```text
15 * -0x7fff = -0x77ff1
0x2000000 - 0x77ff1 = 0x1f8800f
```

## E018 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg7fff_expmax_neg147/
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000003 nv=0x00000004
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 20:07:47 CST | 0 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 1 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 2 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 3 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 4 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 5 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 6 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 7 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 8 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 9 | `a=0x0080 b=0x3500` | `-147` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` | `0` | `0` | `0` |
| 2026-06-01 20:07:47 CST | 10 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 11 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 12 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 13 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 14 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-01 20:07:47 CST | 15 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |

Sum:

```text
partial sum:  ffffffffffff0004
partial sum:  ffffffffffff0004
partial sum:  0000000001ff4003
partial sum:  ffffffffffff0004
complete sum: 0000000001fc400f
```

Observation:

```text
Raising exp_max from -148 to -147 shifts all negative lanes one more bit right.
Each negative lane has mant_shifted=0x3fff, guard=1, sticky=0, lsb=1, rne_inc=1.
Replay result changed from E017 my=0x00000001 nv=0x00000002
to E018 my=0x00000003 nv=0x00000004.
```

## E019 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg4000_expmax_neg147/
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000003 nv=0x00000003
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 20:16:32 CST | 0 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 2 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 3 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 4 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 5 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 6 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 7 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 8 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 9 | `a=0x0080 b=0x3500` | `-147` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 10 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 11 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 12 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 13 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 14 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-01 20:16:32 CST | 15 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff0000
partial sum:  ffffffffffff0000
partial sum:  0000000001ff4000
partial sum:  ffffffffffff0000
complete sum: 0000000001fc4000
```

Observation:

```text
Changing non-lane9 shifted mantissas from the E018 rounded `0x3fff` case to exact `0x4000`
removed the my/nv difference: E019 my=0x00000003 nv=0x00000003.
```

## E020 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg4000_lane9_pos3000000_expmax_neg147/
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000005 nv=0x00000005
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 20:23:03 CST | 0 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 2 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 3 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 4 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 5 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 6 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 7 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 8 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 9 | `a=0x0080 b=0x3540` | `-147` | `0` | `0x0000000003000000` | `0x0000000003000000` | `+` |
| 2026-06-01 20:23:03 CST | 10 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 11 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 12 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 13 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 14 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |
| 2026-06-01 20:23:03 CST | 15 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` |

Sum:

```text
partial sum:  ffffffffffff0000
partial sum:  ffffffffffff0000
partial sum:  0000000002ff4000
partial sum:  ffffffffffff0000
complete sum: 0000000002fc4000
```

Interpretation:

```text
0x3000000 - 15 * 0x4000 = 0x2fc4000
```

## E021 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg8000_lane9_pos3000000_expmax_neg148/
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000002 nv=0x00000002
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-01 20:44:57 CST | 0 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 2 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 3 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 4 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 5 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 6 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 7 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 8 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 9 | `a=0x0080 b=0x34c0` | `-148` | `0` | `0x0000000003000000` | `0x0000000003000000` | `+` |
| 2026-06-01 20:44:57 CST | 10 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 11 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 12 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 13 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 14 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |
| 2026-06-01 20:44:57 CST | 15 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` |

Sum:

```text
partial sum:  fffffffffffe0000
partial sum:  fffffffffffe0000
partial sum:  0000000002fe8000
partial sum:  fffffffffffe0000
complete sum: 0000000002f88000
```

Interpretation:

```text
0x3000000 - 15 * 0x8000 = 0x2f88000
```

## E022 Exponent Sweep

Artifacts:

```text
custom_debug/exp_sweep_max_exp_lane9_mant3000000_nonlane_cases/summary.md
custom_debug/exp_sweep_max_exp_lane9_mant3000000_nonlane_cases/results.csv
```

Method:

```text
Starting from current exp_max=-148, increment exp_max by 1 and test -147..-120 inclusive.
Stop condition is exp_max > -121, so -120 is included.
lane9 mantissa is fixed at 0x3000000 for every run.
Non-lane exponent remains exp_ab=-158 for every non-zero non-lane product.
Non-lane non-zero products keep negative sign.
Cases a/b use random non-lane lane 3, selected with seed 20260601.
0x1ffffff is not directly constructible as a BF16 product here, so the nearest lower constructible value 0x1fff800 is used.
```

Case definitions:

| Case | Non-lane distribution | Requested pre mantissa | Actual pre mantissa |
| --- | --- | --- | --- |
| a | one random lane `3`, rest zero | `0x2000000` | `0x2000000` |
| b | one random lane `3`, rest zero | `0x1ffffff` | `0x1fff800` |
| c | all 15 non-lane9 lanes | `0x2000000` | `0x2000000` |
| d | all 15 non-lane9 lanes | `0x1ffffff` | `0x1fff800` |

Result:

```text
Total replay cases: 112
my != nv cases: 0
```

Compact my/nv map:

| exp_max | a my/nv | b my/nv | c my/nv | d my/nv |
| --- | --- | --- | --- | --- |
| -147 | `0x00000005/0x00000005` | `0x00000005/0x00000005` | `0x00000005/0x00000005` | `0x00000005/0x00000005` |
| -146 | `0x0000000b/0x0000000b` | `0x0000000b/0x0000000b` | `0x0000000b/0x0000000b` | `0x0000000b/0x0000000b` |
| -145 | `0x00000017/0x00000017` | `0x00000017/0x00000017` | `0x00000017/0x00000017` | `0x00000017/0x00000017` |
| -144 | `0x0000002f/0x0000002f` | `0x0000002f/0x0000002f` | `0x0000002f/0x0000002f` | `0x0000002f/0x0000002f` |
| -143 | `0x0000005f/0x0000005f` | `0x0000005f/0x0000005f` | `0x0000005f/0x0000005f` | `0x0000005f/0x0000005f` |
| -142 | `0x000000bf/0x000000bf` | `0x000000bf/0x000000bf` | `0x000000bf/0x000000bf` | `0x000000bf/0x000000bf` |
| -141 | `0x0000017f/0x0000017f` | `0x0000017f/0x0000017f` | `0x0000017f/0x0000017f` | `0x0000017f/0x0000017f` |
| -140 | `0x000002ff/0x000002ff` | `0x000002ff/0x000002ff` | `0x000002ff/0x000002ff` | `0x000002ff/0x000002ff` |
| -139 | `0x000005ff/0x000005ff` | `0x000005ff/0x000005ff` | `0x000005ff/0x000005ff` | `0x000005ff/0x000005ff` |
| -138 | `0x00000bff/0x00000bff` | `0x00000bff/0x00000bff` | `0x00000bff/0x00000bff` | `0x00000bff/0x00000bff` |
| -137 | `0x000017ff/0x000017ff` | `0x000017ff/0x000017ff` | `0x000017ff/0x000017ff` | `0x000017ff/0x000017ff` |
| -136 | `0x00002fff/0x00002fff` | `0x00002fff/0x00002fff` | `0x00002fff/0x00002fff` | `0x00002fff/0x00002fff` |
| -135 | `0x00005fff/0x00005fff` | `0x00005fff/0x00005fff` | `0x00005fff/0x00005fff` | `0x00005fff/0x00005fff` |
| -134 | `0x0000bfff/0x0000bfff` | `0x0000bfff/0x0000bfff` | `0x0000bfff/0x0000bfff` | `0x0000bfff/0x0000bfff` |
| -133 | `0x00017fff/0x00017fff` | `0x00017fff/0x00017fff` | `0x00017fff/0x00017fff` | `0x00017fff/0x00017fff` |
| -132 | `0x00030000/0x00030000` | `0x00030000/0x00030000` | `0x00030000/0x00030000` | `0x00030000/0x00030000` |
| -131 | `0x00060000/0x00060000` | `0x00060000/0x00060000` | `0x00060000/0x00060000` | `0x00060000/0x00060000` |
| -130 | `0x000c0000/0x000c0000` | `0x000c0000/0x000c0000` | `0x000c0000/0x000c0000` | `0x000c0000/0x000c0000` |
| -129 | `0x00180000/0x00180000` | `0x00180000/0x00180000` | `0x00180000/0x00180000` | `0x00180000/0x00180000` |
| -128 | `0x00300000/0x00300000` | `0x00300000/0x00300000` | `0x00300000/0x00300000` | `0x00300000/0x00300000` |
| -127 | `0x00600000/0x00600000` | `0x00600000/0x00600000` | `0x00600000/0x00600000` | `0x00600000/0x00600000` |
| -126 | `0x00c00000/0x00c00000` | `0x00c00000/0x00c00000` | `0x00c00000/0x00c00000` | `0x00c00000/0x00c00000` |
| -125 | `0x01400000/0x01400000` | `0x01400000/0x01400000` | `0x01400000/0x01400000` | `0x01400000/0x01400000` |
| -124 | `0x01c00000/0x01c00000` | `0x01c00000/0x01c00000` | `0x01c00000/0x01c00000` | `0x01c00000/0x01c00000` |
| -123 | `0x02400000/0x02400000` | `0x02400000/0x02400000` | `0x02400000/0x02400000` | `0x02400000/0x02400000` |
| -122 | `0x02c00000/0x02c00000` | `0x02c00000/0x02c00000` | `0x02c00000/0x02c00000` | `0x02c00000/0x02c00000` |
| -121 | `0x03400000/0x03400000` | `0x03400000/0x03400000` | `0x03400000/0x03400000` | `0x03400000/0x03400000` |
| -120 | `0x03c00000/0x03c00000` | `0x03c00000/0x03c00000` | `0x03c00000/0x03c00000` | `0x03c00000/0x03c00000` |

## E023 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/recheck_E021_gpu_20260602 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000002 nv=0x00000002
```

Preserved trace dir:

```text
custom_debug/recheck_E021_gpu_20260602/
```

Equivalence checks against E021 artifacts:

```text
a_double.bin.log vs E021: identical
b_double.bin.log vs E021: identical
mant_align_bf16_to_fp32.txt.log vs E021: identical
mant_sum_bf16_to_fp32.txt.log vs E021: identical
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 16:59:29 CST | 0 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 2 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 3 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 4 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 5 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 6 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 7 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 8 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 9 | `a=0x0080 b=0x34c0` | `-148` | `0` | `0x0000000003000000` | `0x0000000003000000` | `+` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 10 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 11 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 12 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 13 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 14 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 16:59:29 CST | 15 | `a=0x0080 b=0xaf80` | `-158` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffffe0000
partial sum:  ffffffffffffe0000
partial sum:  0000000002fe8000
partial sum:  ffffffffffffe0000
complete sum: 0000000002f88000
```

Observation:

```text
The GPU recheck input files and produced align/sum traces are byte-identical to E021.
The replay result confirms the E021 record: my=0x00000002 and nv=0x00000002.
A prior sandbox-only run without visible NVIDIA driver produced nv=0x00000000;
that run is invalid and should not be treated as hardware evidence.
```

## E024 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_neg7fff_lane9_pos3000000_expmax_neg148 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000002 nv=0x00000003
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg7fff_lane9_pos3000000_expmax_neg148/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 17:06:58 CST | 0 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 1 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 2 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 3 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 4 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 5 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 6 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 7 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 8 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 9 | `a=0x0080 b=0x34c0` | `-148` | `0` | `0x0000000003000000` | `0x0000000003000000` | `+` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 10 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 11 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 12 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 13 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 14 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:06:58 CST | 15 | `a=0x0097 b=0xaf59` | `-159` | `11` | `0x0000000003fff800` | `0x0000000000007fff` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffffe0004
partial sum:  ffffffffffffe0004
partial sum:  0000000002fe8003
partial sum:  ffffffffffffe0004
complete sum: 0000000002f8800f
```

Observation:

```text
This is the direct counterpart of E021/E023 with lane9 unchanged at +0x3000000.
Changing the 15 non-lane9 aligned mantissas from exact -0x8000 to -0x7fff
adds 0x000f to the complete sum and changes NVIDIA from nv=0x00000002 to
nv=0x00000003 while my remains 0x00000002.
```

## E025 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_neg7fff_lane9_pos3000000_expmax_neg147 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000005 nv=0x00000006
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg7fff_lane9_pos3000000_expmax_neg147/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 17:13:24 CST | 0 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 1 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 2 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 3 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 4 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 5 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 6 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 7 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 8 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 9 | `a=0x0080 b=0x3540` | `-147` | `0` | `0x0000000003000000` | `0x0000000003000000` | `+` | `0` | `0` | `0` |
| 2026-06-02 17:13:24 CST | 10 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 11 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 12 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 13 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 14 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:13:24 CST | 15 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |

Sum:

```text
partial sum:  ffffffffffff0004
partial sum:  ffffffffffff0004
partial sum:  0000000002ff4003
partial sum:  ffffffffffff0004
complete sum: 0000000002fc400f
```

Observation:

```text
Raising lane9 from exp_ab=-148 to exp_ab=-147 makes exp_max=-147.
The non-lane9 products now align as mant_shifted=0x3fff with guard=1 and
rne_inc=1, while lane9 stays at mant_shifted=0x3000000.
Compared with E020, which has exact non-lane9 -0x4000 at the same exp_max and
my=nv=0x00000005, this guard-bearing -0x3fff case produces nv=0x00000006.
```

## E026 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/recheck_E020_after_E025_neg4000_lane9_pos3000000_expmax_neg147 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000005 nv=0x00000005
```

Preserved trace dir:

```text
custom_debug/recheck_E020_after_E025_neg4000_lane9_pos3000000_expmax_neg147/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 17:18:25 CST | 0 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 2 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 3 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 4 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 5 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 6 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 7 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 8 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 9 | `a=0x0080 b=0x3540` | `-147` | `0` | `0x0000000003000000` | `0x0000000003000000` | `+` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 10 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 11 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 12 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 13 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 14 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:18:25 CST | 15 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff0000
partial sum:  ffffffffffff0000
partial sum:  0000000002ff4000
partial sum:  ffffffffffff0000
complete sum: 0000000002fc4000
```

Observation:

```text
This is the exact-mantissa counterpart to E025 and a GPU recheck of E020.
Changing the 15 non-lane9 aligned mantissas from guard-bearing -0x3fff to
exact -0x4000 removes the my/nv difference: E025 had my=0x00000005,
nv=0x00000006, while E026 has my=0x00000005, nv=0x00000005.
```

## E027 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_neg4000_lane9_pos2c00000_expmax_neg147 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000005 nv=0x00000005
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg4000_lane9_pos2c00000_expmax_neg147/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 17:28:04 CST | 0 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 2 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 3 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 4 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 5 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 6 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 7 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 8 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 9 | `a=0x0080 b=0x3530` | `-147` | `0` | `0x0000000002c00000` | `0x0000000002c00000` | `+` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 10 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 11 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 12 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 13 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 14 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:28:04 CST | 15 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff0000
partial sum:  ffffffffffff0000
partial sum:  0000000002bf4000
partial sum:  ffffffffffff0000
complete sum: 0000000002bc4000
```

Observation:

```text
This is E026 with lane9 reduced from +0x3000000 to +0x2c00000.
The 15 exact -0x4000 lanes are unchanged, so the complete sum decreases by
0x400000 from 0x2fc4000 to 0x2bc4000. The replay remains matched:
my=0x00000005 and nv=0x00000005.
```

## E028 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_neg3fff_lane9_pos2c00000_expmax_neg147 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000005 nv=0x00000005
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg3fff_lane9_pos2c00000_expmax_neg147/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 17:37:38 CST | 0 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 1 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 2 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 3 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 4 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 5 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 6 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 7 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 8 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 9 | `a=0x0080 b=0x3530` | `-147` | `0` | `0x0000000002c00000` | `0x0000000002c00000` | `+` | `0` | `0` | `0` |
| 2026-06-02 17:37:38 CST | 10 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 11 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 12 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 13 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 14 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:37:38 CST | 15 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |

Sum:

```text
partial sum:  ffffffffffff0004
partial sum:  ffffffffffff0004
partial sum:  0000000002bf4003
partial sum:  ffffffffffff0004
complete sum: 0000000002bc400f
```

Observation:

```text
This is E027 with the 15 non-lane9 exact -0x4000 lanes changed to the
guard-bearing -0x3fff construction. The complete sum increases by 0x000f from
0x2bc4000 to 0x2bc400f, but unlike E025 at lane9=0x3000000, the replay remains
matched: my=0x00000005 and nv=0x00000005.
```

## E029 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_neg3fff_lane9_pos2800000_expmax_neg147 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000004 nv=0x00000005
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg3fff_lane9_pos2800000_expmax_neg147/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 17:47:34 CST | 0 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 1 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 2 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 3 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 4 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 5 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 6 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 7 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 8 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 9 | `a=0x0080 b=0x3520` | `-147` | `0` | `0x0000000002800000` | `0x0000000002800000` | `+` | `0` | `0` | `0` |
| 2026-06-02 17:47:34 CST | 10 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 11 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 12 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 13 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 14 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |
| 2026-06-02 17:47:34 CST | 15 | `a=0x0097 b=0xaf59` | `-159` | `12` | `0x0000000003fff800` | `0x0000000000003fff` | `-` | `1` | `0` | `1` |

Sum:

```text
partial sum:  ffffffffffff0004
partial sum:  ffffffffffff0004
partial sum:  00000000027f4003
partial sum:  ffffffffffff0004
complete sum: 00000000027c400f
```

Observation:

```text
This is E028 with lane9 reduced from +0x2c00000 to +0x2800000.
The non-lane9 guard-bearing -0x3fff lanes are unchanged. The replay now differs
again by one ulp: my=0x00000004 and nv=0x00000005.
```

## E030 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_neg4000_lane9_pos2800000_expmax_neg147 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000004 nv=0x00000004
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_neg4000_lane9_pos2800000_expmax_neg147/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 17:53:52 CST | 0 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 1 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 2 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 3 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 4 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 5 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 6 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 7 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 8 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 9 | `a=0x0080 b=0x3520` | `-147` | `0` | `0x0000000002800000` | `0x0000000002800000` | `+` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 10 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 11 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 12 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 13 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 14 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 17:53:52 CST | 15 | `a=0x0080 b=0xaf80` | `-158` | `11` | `0x0000000002000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff0000
partial sum:  ffffffffffff0000
partial sum:  00000000027f4000
partial sum:  ffffffffffff0000
complete sum: 00000000027c4000
```

Observation:

```text
This is E029 with the 15 non-lane9 guard-bearing -0x3fff lanes changed to
exact -0x4000. The complete sum decreases by 0x000f from 0x27c400f to
0x27c4000, and the my/nv difference disappears: my=0x00000004 and
nv=0x00000004.
```

## E031 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_expab_neg159_mant2000_lane9_pos2800000_expmax_neg147 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000004 nv=0x00000005
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_expab_neg159_mant2000_lane9_pos2800000_expmax_neg147/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 18:19:13 CST | 0 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 1 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 2 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 3 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 4 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 5 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 6 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 7 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 8 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 9 | `a=0x0080 b=0x3520` | `-147` | `0` | `0x0000000002800000` | `0x0000000002800000` | `+` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 10 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 11 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 12 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 13 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 14 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:19:13 CST | 15 | `a=0x0080 b=0xaf00` | `-159` | `12` | `0x0000000002000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff8000
partial sum:  ffffffffffff8000
partial sum:  00000000027fa000
partial sum:  ffffffffffff8000
complete sum: 00000000027e2000
```

Observation:

```text
This changes the non-lane9 product exponent from E030's -158 to -159 by using
`a=0x0080 b=0xaf00`. Since exp_max stays -147, the non-lane9 shifted mantissa
also changes from exact 0x4000 to exact 0x2000. The replay differs by one ulp:
my=0x00000004 and nv=0x00000005.
```

## E032 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_expab_neg157_mant8000_lane9_pos2800000_expmax_neg147 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000004 nv=0x00000004
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_expab_neg157_mant8000_lane9_pos2800000_expmax_neg147/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 18:24:49 CST | 0 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 1 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 2 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 3 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 4 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 5 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 6 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 7 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 8 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 9 | `a=0x0080 b=0x3520` | `-147` | `0` | `0x0000000002800000` | `0x0000000002800000` | `+` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 10 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 11 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 12 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 13 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 14 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:24:49 CST | 15 | `a=0x0080 b=0xb000` | `-157` | `10` | `0x0000000002000000` | `0x0000000000008000` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  fffffffffffe0000
partial sum:  fffffffffffe0000
partial sum:  00000000027e8000
partial sum:  fffffffffffe0000
complete sum: 0000000002788000
```

Observation:

```text
This changes the non-lane9 product exponent from E031's -159 to -157 by using
`a=0x0080 b=0xb000`. Since exp_max stays -147, the non-lane9 shifted mantissa
changes from exact 0x2000 to exact 0x8000. The replay is matched:
my=0x00000004 and nv=0x00000004.
```

## E033 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_expab_neg157_mant4000_lane9_pos2800000_expmax_neg147 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000004 nv=0x00000004
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_expab_neg157_mant4000_lane9_pos2800000_expmax_neg147/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 18:33:16 CST | 0 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 1 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 2 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 3 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 4 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 5 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 6 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 7 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 8 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 9 | `a=0x0080 b=0x3520` | `-147` | `0` | `0x0000000002800000` | `0x0000000002800000` | `+` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 10 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 11 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 12 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 13 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 14 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:33:16 CST | 15 | `a=0x0040 b=0xb000` | `-157` | `10` | `0x0000000001000000` | `0x0000000000004000` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff0000
partial sum:  ffffffffffff0000
partial sum:  00000000027f4000
partial sum:  ffffffffffff0000
complete sum: 00000000027c4000
```

Observation:

```text
This keeps every non-lane9 product exponent at E032's exp_ab=-157 while changing
the shifted mantissa from 0x8000 to exact 0x4000. The construction uses the
subnormal `a=0x0040` with `b=0xb000`: BF16 decode keeps exponent -126 for
`a=0x0040`, so exp_ab remains -157, while the mantissa product is halved from
0x2000000 to 0x1000000 before alignment. The final complete sum matches E030,
but reaches it through a different non-lane exponent path. The replay is matched:
my=0x00000004 and nv=0x00000004.
```

## E034 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_expab_neg157_mant3ffc_lane9_pos2800000_expmax_neg147 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000004 nv=0x00000005
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_expab_neg157_mant3ffc_lane9_pos2800000_expmax_neg147/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 18:45:52 CST | 0 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 1 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 2 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 3 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 4 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 5 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 6 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 7 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 8 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 9 | `a=0x0080 b=0x3520` | `-147` | `0` | `0x0000000002800000` | `0x0000000002800000` | `+` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 10 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 11 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 12 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 13 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 14 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |
| 2026-06-02 18:45:52 CST | 15 | `a=0x003f b=0xb002` | `-157` | `10` | `0x0000000000fff000` | `0x0000000000003ffc` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff0010
partial sum:  ffffffffffff0010
partial sum:  00000000027f400c
partial sum:  ffffffffffff0010
complete sum: 00000000027c403c
```

Observation:

```text
Exact non-lane9 mant_shifted=0x3fff is not reachable while keeping exp_ab=-157
and exp_max=-147. BF16 product mant_pre is `(decoded_mant_a * decoded_mant_b)
<< 11`, and rsft=10, so the shifted value is always
`2 * decoded_mant_a * decoded_mant_b`, an even integer. The next odd target
0x3fff is therefore impossible; 0x3ffe would require product 0x1fff, which is
not factorizable with one subnormal mantissa in 1..127 and one normal mantissa
in 128..255 at exp_ab=-157. The maximum lower constructible product is 0x1ffe
and the maximum lower shifted mantissa is 0x3ffc. This run uses
`0x3f * 0x82 = 0x1ffe`, i.e. `a=0x003f b=0xb002`. The replay differs by one ulp:
my=0x00000004 and nv=0x00000005.
```

## E035 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_expab_neg157_mant3ffc_lane9_pos2000000_expmax_neg146 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000007 nv=0x00000008
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_expab_neg157_mant3ffc_lane9_pos2000000_expmax_neg146/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 21:41:25 CST | 0 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 1 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 2 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 3 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 4 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 5 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 6 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 7 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 8 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 9 | `a=0x0080 b=0x3580` | `-146` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 10 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 11 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 12 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 13 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 14 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:41:25 CST | 15 | `a=0x003f b=0xb002` | `-157` | `11` | `0x0000000000fff000` | `0x0000000000001ffe` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff8008
partial sum:  ffffffffffff8008
partial sum:  0000000001ffa006
partial sum:  ffffffffffff8008
complete sum: 0000000001fe201e
```

Observation:

```text
This changes only lane9 from E034: lane9 becomes `a=0x0080 b=0x3580`, so
exp_ab is raised from -147 to -146 and mant_pre/mant_shifted changes from
0x2800000 to 0x2000000. Since lane9 now sets exp_max=-146, every non-lane9 lane
keeps exp_ab=-157 but shifts by 11 instead of 10; the non-lane9 shifted mantissa
therefore changes from 0x3ffc to 0x1ffe. The replay differs by one ulp:
my=0x00000007 and nv=0x00000008.
```

## E036 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/all_except_lane9_expab_neg157_mant2000_lane9_pos2000000_expmax_neg146 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000007 nv=0x00000007
```

Preserved trace dir:

```text
custom_debug/all_except_lane9_expab_neg157_mant2000_lane9_pos2000000_expmax_neg146/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 21:46:47 CST | 0 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 1 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 2 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 3 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 4 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 5 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 6 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 7 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 8 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 9 | `a=0x0080 b=0x3580` | `-146` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 10 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 11 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 12 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 13 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 14 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |
| 2026-06-02 21:46:47 CST | 15 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `-` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffff8000
partial sum:  ffffffffffff8000
partial sum:  0000000001ffa000
partial sum:  ffffffffffff8000
complete sum: 0000000001fe2000
```

Observation:

```text
This changes every non-lane9 lane from E035's maximum-lower `a=0x003f b=0xb002`
construction to `a=0x0040 b=0xb000`. The non-lane9 product exponent stays
exp_ab=-157 and exp_max stays -146, but mant_pre changes from 0x0fff000 to
0x1000000, so the shifted mantissa changes from 0x1ffe to exact 0x2000.
The replay is matched again: my=0x00000007 and nv=0x00000007.
```

## E037 Sweep Table

Replay command pattern:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=<case trace dir> \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved sweep dir:

```text
custom_debug/sweep_nonlane_shift2000_expab_lane9_pos2000000_expmax_neg146/
```

Summary files:

```text
custom_debug/sweep_nonlane_shift2000_expab_lane9_pos2000000_expmax_neg146/summary.md
custom_debug/sweep_nonlane_shift2000_expab_lane9_pos2000000_expmax_neg146/results.csv
```

Lane9 is fixed as `a=0x0080 b=0x3580`, `exp_ab=-146`, `mant_shifted=0x2000000`.
Every non-lane9 lane is negative and keeps exact `mant_shifted=0x2000`.

| Time | Case | Non-lane input | exp_ab | rsft | mant_pre | mant_shifted | partial groups | complete sum | my | nv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 21:55:38 CST | expab_-158 | `a=0x0080 b=0xaf80` | `-158` | `12` | `0x0000000002000000` | `0x0000000000002000` | `ffffffffffff8000`, `ffffffffffff8000`, `0000000001ffa000`, `ffffffffffff8000` | `0000000001fe2000` | `0x00000007` | `0x00000007` |
| 2026-06-02 21:55:38 CST | expab_-157 | `a=0x0040 b=0xb000` | `-157` | `11` | `0x0000000001000000` | `0x0000000000002000` | `ffffffffffff8000`, `ffffffffffff8000`, `0000000001ffa000`, `ffffffffffff8000` | `0000000001fe2000` | `0x00000007` | `0x00000007` |
| 2026-06-02 21:55:38 CST | expab_-156 | `a=0x0020 b=0xb080` | `-156` | `10` | `0x0000000000800000` | `0x0000000000002000` | `ffffffffffff8000`, `ffffffffffff8000`, `0000000001ffa000`, `ffffffffffff8000` | `0000000001fe2000` | `0x00000007` | `0x00000007` |
| 2026-06-02 21:55:38 CST | expab_-155 | `a=0x0010 b=0xb100` | `-155` | `9` | `0x0000000000400000` | `0x0000000000002000` | `ffffffffffff8000`, `ffffffffffff8000`, `0000000001ffa000`, `ffffffffffff8000` | `0000000001fe2000` | `0x00000007` | `0x00000007` |
| 2026-06-02 21:55:38 CST | expab_-154 | `a=0x0008 b=0xb180` | `-154` | `8` | `0x0000000000200000` | `0x0000000000002000` | `ffffffffffff8000`, `ffffffffffff8000`, `0000000001ffa000`, `ffffffffffff8000` | `0000000001fe2000` | `0x00000007` | `0x00000007` |
| 2026-06-02 21:55:38 CST | expab_-153 | `a=0x0004 b=0xb200` | `-153` | `7` | `0x0000000000100000` | `0x0000000000002000` | `ffffffffffff8000`, `ffffffffffff8000`, `0000000001ffa000`, `ffffffffffff8000` | `0000000001fe2000` | `0x00000007` | `0x00000007` |
| 2026-06-02 21:55:38 CST | expab_-152 | `a=0x0002 b=0xb280` | `-152` | `6` | `0x0000000000080000` | `0x0000000000002000` | `ffffffffffff8000`, `ffffffffffff8000`, `0000000001ffa000`, `ffffffffffff8000` | `0000000001fe2000` | `0x00000007` | `0x00000007` |
| 2026-06-02 21:55:38 CST | expab_-151 | `a=0x0001 b=0xb300` | `-151` | `5` | `0x0000000000040000` | `0x0000000000002000` | `ffffffffffff8000`, `ffffffffffff8000`, `0000000001ffa000`, `ffffffffffff8000` | `0000000001fe2000` | `0x00000007` | `0x00000007` |

Observation:

```text
All tested non-lane9 exponent values from -158 through -151 can keep the aligned
mantissa exactly at 0x2000. The complete sum and NVIDIA result are invariant
across these exponent decompositions: my=0x00000007 and nv=0x00000007 for all
eight cases.
```

## E038 Sweep Table

Replay command pattern:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=<case trace dir> \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved sweep dir:

```text
custom_debug/sweep_nonlane_shift1fff_expab_lane9_pos2000000_expmax_neg146/
```

Summary files:

```text
custom_debug/sweep_nonlane_shift1fff_expab_lane9_pos2000000_expmax_neg146/summary.md
custom_debug/sweep_nonlane_shift1fff_expab_lane9_pos2000000_expmax_neg146/results.csv
```

Lane9 is fixed as `a=0x0080 b=0x3580`, `exp_ab=-146`, `mant_shifted=0x2000000`.
Every non-lane9 lane is negative and keeps floor `mant_shifted=0x1fff`.

| Time | Case | Non-lane input | exp_ab | rsft | mant_pre | mant_shifted | discarded | guard | sticky | rne_inc | partial groups | complete sum | my | nv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 21:55:38 CST | expab_-159 | `a=0x0081 b=0xaf7e` | `-159` | `13` | `0x0000000003fff000` | `0x0000000000001fff` | `0x0000000000001000` | `1` | `0` | `1` | `ffffffffffff8004`, `ffffffffffff8004`, `0000000001ffa003`, `ffffffffffff8004` | `0000000001fe200f` | `0x00000007` | `0x00000008` |
| 2026-06-02 21:55:38 CST | expab_-158 | `a=0x007f b=0xaf81` | `-158` | `12` | `0x0000000001fff800` | `0x0000000000001fff` | `0x0000000000000800` | `1` | `0` | `1` | `ffffffffffff8004`, `ffffffffffff8004`, `0000000001ffa003`, `ffffffffffff8004` | `0000000001fe200f` | `0x00000007` | `0x00000008` |

Observation:

```text
The requested non-lane9 shifted mantissa 0x1fff is constructible for exp_ab=-159
and exp_ab=-158, but both constructions carry guard=1 and rne_inc=1. Both cases
produce the same complete sum 0x1fe200f and the same one-ulp my/nv difference:
my=0x00000007 and nv=0x00000008.
```

## E039 Sweep Summary

Replay command pattern:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=<case trace dir> \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved sweep dir:

```text
custom_debug/sweep_lane9_exp_neg146_to_neg127_nonlane_expab_neg151_mantpre40000/
```

Summary files:

```text
custom_debug/sweep_lane9_exp_neg146_to_neg127_nonlane_expab_neg151_mantpre40000/summary.md
custom_debug/sweep_lane9_exp_neg146_to_neg127_nonlane_expab_neg151_mantpre40000/results.csv
```

Lane9 keeps mantissa `0x2000000` using `a=0x0080` and exponent-changing normal `b`.
Non-lane9 lanes keep `a=0x0001 b=0xb300`, so `exp_ab=-151` and `mant_pre=0x40000`.

| Time | Sweep range | Fixed non-lane input | Non-lane exp_ab | Non-lane mant_pre | Lane9 mantissa | Result summary |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 22:16:10 CST | lane9 `exp_ab=-146..-127` | `a=0x0001 b=0xb300` | `-151` | `0x0000000000040000` | `0x0000000002000000` | all 20 GPU cases have `my == nv` |

Selected cases:

| lane9 exp_ab | lane9 b | non-lane rsft | non-lane shifted | discarded | guard | sticky | complete sum | my | nv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `-146` | `0x3580` | `5` | `0x0000000000002000` | `0x0000000000000000` | `0` | `0` | `0000000001fe2000` | `0x00000007` | `0x00000007` |
| `-145` | `0x3600` | `6` | `0x0000000000001000` | `0x0000000000000000` | `0` | `0` | `0000000001ff1000` | `0x0000000f` | `0x0000000f` |
| `-140` | `0x3880` | `11` | `0x0000000000000080` | `0x0000000000000000` | `0` | `0` | `0000000001fff880` | `0x000001ff` | `0x000001ff` |
| `-133` | `0x3c00` | `18` | `0x0000000000000001` | `0x0000000000000000` | `0` | `0` | `0000000001fffff1` | `0x0000ffff` | `0x0000ffff` |
| `-132` | `0x3c80` | `19` | `0x0000000000000000` | `0x0000000000040000` | `1` | `0` | `0000000002000000` | `0x00020000` | `0x00020000` |
| `-127` | `0x3f00` | `24` | `0x0000000000000000` | `0x0000000000040000` | `0` | `1` | `0000000002000000` | `0x00400000` | `0x00400000` |

Observation:

```text
Raising lane9 exp_ab from -146 to -127 while keeping lane9 mantissa fixed makes
the fixed non-lane mant_pre=0x40000 shift progressively farther right. It remains
non-zero through lane9 exp_ab=-133, then shifts out at -132 and above. Every
tested case matches NVIDIA.
```

## E040 Representability Check

Requested construction:

```text
lane9: keep mantissa 0x2000000 and sweep exponent as in E039
non-lane9: keep exponent unchanged and set mant_pre to 0x3ffff, or maximum lower
```

Preserved check dir:

```text
custom_debug/nonlane_expab_neg151_mantpre3ffff_representability/
```

Summary file:

```text
custom_debug/nonlane_expab_neg151_mantpre3ffff_representability/summary.md
```

Result:

```text
Exact 0x3ffff is impossible. There is also no smaller non-zero BF16 product
while keeping non-lane9 exp_ab=-151.
```

Reason:

```text
mant_pre = decoded_mant_a * decoded_mant_b << 11.
With non-lane9 exp_ab fixed at -151, the smallest non-zero decoded mantissa
product is 1 * 128 = 0x80, represented by a=0x0001 b=0xb300. Therefore the
smallest non-zero mant_pre is 0x80 << 11 = 0x40000. Since 0x3ffff is below that,
no non-zero lower constructible value exists. A zero product would not preserve
the requested non-zero exponent/contribution, so no GPU replay was run.
```

## E041 Sweep Summary

Replay command pattern:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=<case trace dir> \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved sweep dir:

```text
custom_debug/sweep_lane9_exp_neg146_to_neg127_nonlane_expab_neg159_mantpre3fff800/
```

Summary files:

```text
custom_debug/sweep_lane9_exp_neg146_to_neg127_nonlane_expab_neg159_mantpre3fff800/summary.md
custom_debug/sweep_lane9_exp_neg146_to_neg127_nonlane_expab_neg159_mantpre3fff800/results.csv
```

Lane9 keeps mantissa `0x2000000` using `a=0x0080` and exponent-changing normal `b`.
Non-lane9 lanes keep `a=0x0097 b=0xaf59`, so `exp_ab=-159` and `mant_pre=0x3fff800`.

| Time | Sweep range | Fixed non-lane input | Non-lane exp_ab | Non-lane mant_pre | Lane9 mantissa | Result summary |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 22:32:51 CST | lane9 `exp_ab=-146..-127` | `a=0x0097 b=0xaf59` | `-159` | `0x0000000003fff800` | `0x0000000002000000` | `-146..-134` differ by one ulp; `-133..-127` match |

Selected cases:

| lane9 exp_ab | lane9 b | non-lane rsft | non-lane shifted | discarded | guard | sticky | rne_inc | complete sum | my | nv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `-146` | `0x3580` | `13` | `0x0000000000001fff` | `0x0000000000001800` | `1` | `1` | `1` | `0000000001fe200f` | `0x00000007` | `0x00000008` |
| `-145` | `0x3600` | `14` | `0x0000000000000fff` | `0x0000000000003800` | `1` | `1` | `1` | `0000000001ff100f` | `0x0000000f` | `0x00000010` |
| `-140` | `0x3880` | `19` | `0x000000000000007f` | `0x000000000007f800` | `1` | `1` | `1` | `0000000001fff88f` | `0x000001ff` | `0x00000200` |
| `-134` | `0x3b80` | `25` | `0x0000000000000001` | `0x0000000001fff800` | `1` | `1` | `1` | `0000000001fffff1` | `0x00007fff` | `0x00008000` |
| `-133` | `0x3c00` | `26` | `0x0000000000000000` | `0x0000000003fff800` | `1` | `1` | `1` | `0000000002000000` | `0x00010000` | `0x00010000` |
| `-127` | `0x3f00` | `32` | `0x0000000000000000` | `0x0000000003fff800` | `0` | `1` | `0` | `0000000002000000` | `0x00400000` | `0x00400000` |

Observation:

```text
Compared with E039's small exact mant_pre=0x40000 non-lane terms, this
guard-bearing non-lane construction (`mant_pre=0x3fff800`) produces a one-ulp
my/nv difference for lane9 exp_ab=-146 through -134. Once non-lane shifted
mantissa becomes zero at lane9 exp_ab=-133 and above, the replay returns to
my==nv even though discarded/sticky information remains in the trace.
```

## E042 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/lane9_1c11_1ce1_nonlane_expab_neg155_mant1_pos \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x000001fe nv=0x000001fe
```

Preserved trace dir:

```text
custom_debug/lane9_1c11_1ce1_nonlane_expab_neg155_mant1_pos/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-03 13:57:03 CST | 0 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 1 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 2 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 3 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 4 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 5 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 6 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 7 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 8 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 9 | `a=0x1c11 b=0x1ce1` | `-141` | `0` | `0x0000000003fb8800` | `0x0000000003fb8800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 10 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 11 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 12 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 13 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 14 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 13:57:03 CST | 15 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |

Sum:

```text
partial sum:  0000000000002000
partial sum:  0000000000002000
partial sum:  0000000003fba000
partial sum:  0000000000002000
complete sum: 0000000003fc0000
```

Observation:

```text
This uses positive signs for every lane. Lane9 decodes as A=0x1c11
(exp=-71, mant=0x91) and B=0x1ce1 (exp=-70, mant=0xe1), giving exp_ab=-141
and mant_pre=(0x91*0xe1)<<11=0x3fb8800. Every non-lane9 product uses
`a=0x0080 b=0x3100`, so exp_ab=-155 and the decoded mantissa product is exactly
1.0. The replay is matched: my=0x000001fe and nv=0x000001fe.
```

## E043 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/lane9_1c11_1c61_nonlane_expab_neg156_mant1_pos_all_exp_minus1 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x000000ff nv=0x000000ff
```

Preserved trace dir:

```text
custom_debug/lane9_1c11_1c61_nonlane_expab_neg156_mant1_pos_all_exp_minus1/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-03 14:15:35 CST | 0 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 1 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 2 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 3 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 4 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 5 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 6 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 7 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 8 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 9 | `a=0x1c11 b=0x1c61` | `-142` | `0` | `0x0000000003fb8800` | `0x0000000003fb8800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 10 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 11 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 12 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 13 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 14 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:15:35 CST | 15 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |

Sum:

```text
partial sum:  0000000000002000
partial sum:  0000000000002000
partial sum:  0000000003fba000
partial sum:  0000000000002000
complete sum: 0000000003fc0000
```

Observation:

```text
This is E042 with every product exponent sum decreased by one. Non-lane9 B
changes from 0x3100 to 0x3080, so non-lane exp_ab changes from -155 to -156
while the decoded mantissa product remains exactly 1.0. Lane9 B changes from
0x1ce1 to 0x1c61, so lane9 exp_ab changes from -141 to -142 while preserving
mant_pre=0x3fb8800. The aligned integer sums match E042, but exp_max is one
lower; replay is matched with my=0x000000ff and nv=0x000000ff.
```

## E044 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/lane9_1c11_1be1_nonlane_expab_neg157_mant1_pos_all_exp_minus2 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x0000007f nv=0x0000007f
```

Preserved trace dir:

```text
custom_debug/lane9_1c11_1be1_nonlane_expab_neg157_mant1_pos_all_exp_minus2/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-03 14:20:57 CST | 0 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 1 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 2 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 3 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 4 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 5 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 6 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 7 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 8 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 9 | `a=0x1c11 b=0x1be1` | `-143` | `0` | `0x0000000003fb8800` | `0x0000000003fb8800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 10 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 11 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 12 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 13 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 14 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:20:57 CST | 15 | `a=0x0080 b=0x3000` | `-157` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |

Sum:

```text
partial sum:  0000000000002000
partial sum:  0000000000002000
partial sum:  0000000003fba000
partial sum:  0000000000002000
complete sum: 0000000003fc0000
```

Observation:

```text
This is E043 with every product exponent sum decreased by one again, or E042
with all product exponent sums decreased by two. Non-lane9 B changes from
0x3080 to 0x3000, so non-lane exp_ab changes from -156 to -157 while preserving
mant_pre=0x2000000. Lane9 B changes from 0x1c61 to 0x1be1, so lane9 exp_ab
changes from -142 to -143 while preserving mant_pre=0x3fb8800. The aligned
integer sums still match E042/E043, but exp_max is lower again; replay is
matched with my=0x0000007f and nv=0x0000007f.

Audit note, 2026-06-03 14:28:25 CST: E042/E043/E044 were replayed again from
their saved input snapshots using real GPU access. The rerun outputs match the
recorded values, and the rerun align/sum traces are byte-identical to the
preserved traces. E044 may look surprising if one expects FP32 subnormal RNE:
halving E043's `0xff` lands at a half-ulp point where RNE would produce
`0x80`. The current `fmac.h` FP32 output path is RZ, so the replay truncates to
`0x7f`; NVIDIA matches this value in the real-GPU replay.
```

## E045 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/lane9_1c11_1c61_nonlane_expab_neg156_mant1_pos_from_E044_all_exp_plus1 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x000000ff nv=0x000000ff
```

Preserved trace dir:

```text
custom_debug/lane9_1c11_1c61_nonlane_expab_neg156_mant1_pos_from_E044_all_exp_plus1/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-03 14:49:36 CST | 0 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 1 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 2 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 3 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 4 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 5 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 6 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 7 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 8 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 9 | `a=0x1c11 b=0x1c61` | `-142` | `0` | `0x0000000003fb8800` | `0x0000000003fb8800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 10 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 11 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 12 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 13 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 14 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:49:36 CST | 15 | `a=0x0080 b=0x3080` | `-156` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |

Sum:

```text
partial sum:  0000000000002000
partial sum:  0000000000002000
partial sum:  0000000003fba000
partial sum:  0000000000002000
complete sum: 0000000003fc0000
```

Observation:

```text
This starts from E044 and increases every product exponent sum by one.
Non-lane9 B changes from 0x3000 to 0x3080, so non-lane exp_ab changes from
-157 to -156 while preserving mant_pre=0x2000000. Lane9 B changes from 0x1be1
to 0x1c61, so lane9 exp_ab changes from -143 to -142 while preserving
mant_pre=0x3fb8800. This reconstructs the same input values and trace behavior
as E043, and the real-GPU replay again matches: my=0x000000ff and
nv=0x000000ff.
```

## E046 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/lane9_1c11_1ce1_nonlane_expab_neg155_mant1_pos_from_E045_all_exp_plus1 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x000001fe nv=0x000001fe
```

Preserved trace dir:

```text
custom_debug/lane9_1c11_1ce1_nonlane_expab_neg155_mant1_pos_from_E045_all_exp_plus1/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-03 14:57:35 CST | 0 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 1 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 2 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 3 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 4 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 5 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 6 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 7 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 8 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 9 | `a=0x1c11 b=0x1ce1` | `-141` | `0` | `0x0000000003fb8800` | `0x0000000003fb8800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 10 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 11 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 12 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 13 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 14 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 14:57:35 CST | 15 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |

Sum:

```text
partial sum:  0000000000002000
partial sum:  0000000000002000
partial sum:  0000000003fba000
partial sum:  0000000000002000
complete sum: 0000000003fc0000
```

Observation:

```text
This starts from E045 and increases every product exponent sum by one again.
Non-lane9 B changes from 0x3080 to 0x3100, so non-lane exp_ab changes from
-156 to -155 while preserving mant_pre=0x2000000. Lane9 B changes from 0x1c61
to 0x1ce1, so lane9 exp_ab changes from -142 to -141 while preserving
mant_pre=0x3fb8800. This reconstructs the same input values and trace behavior
as E042, and the real-GPU replay again matches: my=0x000001fe and
nv=0x000001fe.
```

## E047 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/lane9_mant2000000_expab_neg141_nonlane_expab_neg155_mant1_pos_from_E046 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000100 nv=0x00000100
```

Preserved trace dir:

```text
custom_debug/lane9_mant2000000_expab_neg141_nonlane_expab_neg155_mant1_pos_from_E046/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-03 15:13:55 CST | 0 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 1 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 2 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 3 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 4 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 5 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 6 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 7 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 8 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 9 | `a=0x1c00 b=0x1c80` | `-141` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 10 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 11 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 12 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 13 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 14 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:13:55 CST | 15 | `a=0x0080 b=0x3100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |

Sum:

```text
partial sum:  0000000000002000
partial sum:  0000000000002000
partial sum:  0000000002001800
partial sum:  0000000000002000
complete sum: 0000000002007800
```

Observation:

```text
Only lane9 was changed from E046. The previous lane9 pair `a=0x1c11 b=0x1ce1`
had decoded mantissas `0x91` and `0xe1`, giving
`(0x91 * 0xe1) << 11 = 0x3fb8800`. E047 uses `a=0x1c00 b=0x1c80`, so both
decoded mantissas are `0x80`; with the same exponent sum `-141`, lane9 becomes
exact `0x2000000`. The replay is matched: my=0x00000100 and nv=0x00000100.
```

## E048 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/nonlane_neg_lane9_pos_mant2000000_expab_neg141_from_E047 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x000000ff nv=0x000000ff
```

Preserved trace dir:

```text
custom_debug/nonlane_neg_lane9_pos_mant2000000_expab_neg141_from_E047/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | signed_mant | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-03 15:22:41 CST | 0 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 1 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 2 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 3 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 4 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 5 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 6 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 7 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 8 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 9 | `a=0x1c00 b=0x1c80` | `-141` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` | `0x0000000002000000` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 10 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 11 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 12 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 13 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 14 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |
| 2026-06-03 15:22:41 CST | 15 | `a=0x0080 b=0xb100` | `-155` | `14` | `0x0000000002000000` | `0x0000000000000800` | `-` | `0xfffffffffffff800` | `0` | `0` | `0` |

Sum:

```text
partial sum:  ffffffffffffe000
partial sum:  ffffffffffffe000
partial sum:  0000000001ffe800
partial sum:  ffffffffffffe000
complete sum: 0000000001ff8800
```

Observation:

```text
Only the non-lane9 signs were changed from E047. Their magnitudes remain
`mant_shifted=0x800`, but each signed contribution becomes `-0x800`. Lane9
stays positive at `0x2000000`. The replay is matched:
my=0x000000ff and nv=0x000000ff.
```

## E049 Non-Lane Exp Sweep

Sweep command pattern:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=<case trace dir> \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Preserved sweep dir:

```text
custom_debug/sweep_nonlane_neg_mantpre2000000_expdown_lane9_pos2000000_expab_neg141_from_E048/
```

Fixed lane9:

```text
a=0x1c00 b=0x1c80  # positive; exp_ab=-141, mant_pre=0x2000000
```

Non-lane construction:

```text
a=0x0080               # exp=-126, mant=0x80
b sign is negative     # sign bit set
b exponent decreases   # keeps decoded mantissa at 0x80
mant_pre=(0x80 * 0x80) << 11 = 0x2000000
```

| non-lane exp_ab | non-lane b | rsft | non-lane signed | complete sum | my | nv | match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `-156` | `0xb080` | `15` | `-0x400` | `0000000001ffc400` | `0x000000ff` | `0x000000ff` | yes |
| `-157` | `0xb000` | `16` | `-0x200` | `0000000001ffe200` | `0x000000ff` | `0x000000ff` | yes |
| `-158` | `0xaf80` | `17` | `-0x100` | `0000000001fff100` | `0x000000ff` | `0x000000ff` | yes |
| `-159` | `0xaf00` | `18` | `-0x80` | `0000000001fff880` | `0x000000ff` | `0x00000100` | no |

First mismatch case:

```text
custom_debug/sweep_nonlane_neg_mantpre2000000_expdown_lane9_pos2000000_expab_neg141_from_E048/case_nonlane_expab_-159/
```

First mismatch trace summary:

```text
non-lane:
exp_max=-141 exp_ab=-159 rsft=18 mant_pre=0x2000000
mant_shifted=0x80 signed_mant=-0x80 guard=0 sticky=0 rne_inc=0

partial sums:
fffffffffffffe00
fffffffffffffe00
0000000001fffe80
fffffffffffffe00
complete sum: 0000000001fff880

Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x000000ff nv=0x00000100
```

Observation:

```text
The first mismatch appears exactly when each negative non-lane contribution has
shifted down to `-0x80`. The three previous cases (`-0x400`, `-0x200`,
`-0x100`) match NVIDIA. There is no per-lane guard/sticky in this construction;
the difference is tied to the lane-level reduction/packing behavior around this
near-boundary complete sum.
```

## E050 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/e046_all_exp_minus4_lane9_1c11_1ae1_nonlane_2f00 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x0000001f nv=0x0000001f
```

Preserved trace dir:

```text
custom_debug/e046_all_exp_minus4_lane9_1c11_1ae1_nonlane_2f00/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-03 15:52:50 CST | 0 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 1 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 2 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 3 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 4 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 5 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 6 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 7 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 8 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 9 | `a=0x1c11 b=0x1ae1` | `-145` | `0` | `0x0000000003fb8800` | `0x0000000003fb8800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 10 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 11 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 12 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 13 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 14 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |
| 2026-06-03 15:52:50 CST | 15 | `a=0x0080 b=0x2f00` | `-159` | `14` | `0x0000000002000000` | `0x0000000000000800` | `+` | `0` | `0` | `0` |

Sum:

```text
partial sum:  0000000000002000
partial sum:  0000000000002000
partial sum:  0000000003fba000
partial sum:  0000000000002000
complete sum: 0000000003fc0000
```

Observation:

```text
This is E046 with every product exponent sum decreased by four. Non-lane9 B
changes from 0x3100 to 0x2f00, so non-lane exp_ab changes from -155 to -159.
Lane9 B changes from 0x1ce1 to 0x1ae1, so lane9 exp_ab changes from -141 to
-145 while preserving mant_pre=0x3fb8800. The relative alignment remains the
same, so the traced integer sum is still 0x3fc0000. The output scales down to
my=0x0000001f and nv=0x0000001f.
```

## E051 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/lane8_eq_lane9_lane0_6_exp158_lane7_exp155_lane10_exp151_rest_zero_from_E050 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x00000040 nv=0x0000003f
```

Preserved trace dir:

```text
custom_debug/lane8_eq_lane9_lane0_6_exp158_lane7_exp155_lane10_exp151_rest_zero_from_E050/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-03 16:23:24 CST | 0-6 | `a=0x0080 b=0x2f80` | `-158` | `13` | `0x0000000002000000` | `0x0000000000001000` | `+` | `0` | `0` | `0` |
| 2026-06-03 16:23:24 CST | 7 | `a=0x0080 b=0x3100` | `-155` | `10` | `0x0000000002000000` | `0x0000000000008000` | `+` | `0` | `0` | `0` |
| 2026-06-03 16:23:24 CST | 8-9 | `a=0x1c11 b=0x1ae1` | `-145` | `0` | `0x0000000003fb8800` | `0x0000000003fb8800` | `+` | `0` | `0` | `0` |
| 2026-06-03 16:23:24 CST | 10 | `a=0x0080 b=0x3300` | `-151` | `6` | `0x0000000002000000` | `0x0000000000080000` | `+` | `0` | `0` | `0` |
| 2026-06-03 16:23:24 CST | 11-15 | `a=0x0000 b=0x0000` | zero | n/a | `0x0000000000000000` | `0x0000000000000000` | `+` | `0` | `0` | `0` |

Sum:

```text
partial sum:  0000000000004000
partial sum:  000000000000b000
partial sum:  0000000007ff1000
partial sum:  0000000000000000
complete sum: 0000000008000000
```

Observation:

```text
This requested distribution makes lane8 and lane9 identical and zeros lanes11-15.
The active lanes all have exact alignment shifts with guard=0 and sticky=0. The
complete sum lands on 0x08000000, where the current model returns my=0x00000040
but NVIDIA returns nv=0x0000003f.
```

## E052 Lane Table

Replay command:

```bash
FMAC_TRACE_INTERMEDIATE=1 \
FMAC_TRACE_DIR=custom_debug/lane9_mant2000000_lane0_neg_rest_zero_from_E051 \
./all_fmac_test_custom_nv --dtype 2 --replay-vec 0
```

Replay summary:

```text
Replay summary: dtype=2 vec=0 tag=BF16 stored_d=0x800002b7 my=0x0000000f nv=0x0000000f
```

Preserved trace dir:

```text
custom_debug/lane9_mant2000000_lane0_neg_rest_zero_from_E051/
```

| Time | Lane | Input | exp_ab | rsft | mant_pre | mant_shifted | sign | guard | sticky | rne_inc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-03 16:41:54 CST | 0 | `a=0x0080 b=0xaf80` | `-158` | `13` | `0x0000000002000000` | `0x0000000000001000` | `-` | `0` | `0` | `0` |
| 2026-06-03 16:41:54 CST | 1-8 | `a=0x0000 b=0x0000` | zero | n/a | `0x0000000000000000` | `0x0000000000000000` | `+` | `0` | `0` | `0` |
| 2026-06-03 16:41:54 CST | 9 | `a=0x1c00 b=0x1a80` | `-145` | `0` | `0x0000000002000000` | `0x0000000002000000` | `+` | `0` | `0` | `0` |
| 2026-06-03 16:41:54 CST | 10-15 | `a=0x0000 b=0x0000` | zero | n/a | `0x0000000000000000` | `0x0000000000000000` | `+` | `0` | `0` | `0` |

Sum:

```text
partial sum:  fffffffffffff000
partial sum:  0000000000000000
partial sum:  0000000002000000
partial sum:  0000000000000000
complete sum: 0000000001fff000
```

Observation:

```text
Lane9 keeps exp_ab=-145 but changes from the E051 mant_pre=0x3fb8800 input pair
to exact 1.0 mantissas (`a=0x1c00 b=0x1a80`), giving mant_pre=0x2000000.
Lane0 keeps its E051 exponent and mantissa magnitude but flips the product sign
negative via `b=0xaf80`. With every other lane zeroed, both implementations
return 0x0000000f.
```

## Main Observation

E001 and E002 have the same traced final integer sums:

```text
partial sum  = ffffffffffffe0000  # -0x20000
complete sum = 0000000001fe0000
```

but they produce different NVIDIA results:

```text
E001: nv=0x00000001
E002: nv=0x00000000
```

Therefore the NVIDIA result is not determined only by the final aligned integer mantissa sum printed by our current trace. It is sensitive to lane-level distribution, signs, cancellation pattern, and likely the hardware internal reduction/rounding path.

## Notes For Future Records

When adding a new experiment:

1. Add one row to `Experiment Index`.
2. Add a lane table with only non-zero lanes.
3. Include one timestamp per experiment.
4. For every non-zero lane, record:
   - BF16 input hex words
   - `exp_ab`
   - `rsft`
   - `mant_pre`
   - `mant_shifted`
   - sign
5. Include `partial sum`, `complete sum`, `my`, and `nv`.
6. Use a named `FMAC_TRACE_DIR` for important comparisons so traces do not overwrite each other.
