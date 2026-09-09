#!/usr/bin/env python3
"""Find the first RTL/C-model mismatch, then replay only that vector with trace."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


MAX_TB_VEC_COUNT = 100000


def parse_fail_log(path: Path) -> dict[str, str]:
    if not path.exists():
        raise RuntimeError(f"fail log was not created: {path}")

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 9:
            raise RuntimeError(f"unexpected fail log format: {line}")
        return {
            "local_vec": fields[0],
            "a": fields[1],
            "b": fields[2],
            "c": fields[3],
            "a_scale": fields[4],
            "b_scale": fields[5],
            "got": fields[6],
            "expected": fields[7],
            "ulp": fields[8],
        }

    raise RuntimeError(f"no mismatch entry found in {path}")


def run_make(make_bin: str, args: list[str], log_path: Optional[Path] = None) -> int:
    cmd = [make_bin, *args]
    print("+ " + " ".join(cmd), flush=True)
    if log_path is None:
        return subprocess.run(cmd).returncode

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout)
    return result.returncode


def write_summary(path: Path, data: dict[str, str]) -> None:
    lines = [
        f"dtype={data['dtype']}",
        f"seed={data['seed']}",
        f"cmode_bitsize={data['cmode_bitsize']}",
        f"rtl_bitsize={data['rtl_bitsize']}",
        f"original_vec={data['global_vec']}",
        f"batch_start={data['batch_start']}",
        f"batch_local_vec={data['local_vec']}",
        f"got=0x{data['got']}",
        f"expected=0x{data['expected']}",
        f"ulp={data['ulp']}",
        f"a=0x{data['a']}",
        f"b=0x{data['b']}",
        f"c=0x{data['c']}",
        f"a_scale=0x{data['a_scale']}",
        f"b_scale=0x{data['b_scale']}",
        f"wave={data['wave_file']}",
        f"wave_fail_log={data['wave_fail_log']}",
        f"cmodel_trace_dir={data['cmodel_trace_dir']}",
        f"cmodel_trace_log={data['cmodel_trace_log']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def safe_reset_debug_dir(repo_dir: Path, debug_dir: Path) -> None:
    resolved_repo = repo_dir.resolve()
    resolved_debug = debug_dir.resolve()
    if resolved_debug == resolved_repo or resolved_repo not in resolved_debug.parents:
        raise RuntimeError(f"refusing to reset unsafe debug directory: {resolved_debug}")
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", default="make")
    parser.add_argument("--dtype", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--vec-start", type=int, required=True)
    parser.add_argument("--vec-count", type=int, required=True)
    parser.add_argument("--batch-count", type=int, required=True)
    parser.add_argument("--cmode-bitsize", type=int, default=512)
    parser.add_argument("--rtl-bitsize", type=int, default=512)
    parser.add_argument("--timeout-cycles", type=int, required=True)
    parser.add_argument("--ulp-tolerance", type=int, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.vec_count <= 0:
        raise SystemExit("--vec-count must be positive")
    if args.batch_count <= 0 or args.batch_count > MAX_TB_VEC_COUNT:
        raise SystemExit(f"--batch-count must be in [1, {MAX_TB_VEC_COUNT}]")
    if args.cmode_bitsize <= 0 or args.cmode_bitsize % 32 != 0:
        raise SystemExit("--cmode-bitsize must be a positive multiple of 32")
    if args.rtl_bitsize <= 0 or args.rtl_bitsize % 64 != 0:
        raise SystemExit("--rtl-bitsize must be a positive multiple of 64")

    repo_dir = Path.cwd()
    debug_dir = args.debug_dir
    search_dir = debug_dir / "search"
    wave_dir = debug_dir / "wave"
    cmodel_trace_dir = debug_dir / "cmodel_trace"
    summary_path = debug_dir / "first_mismatch.txt"

    safe_reset_debug_dir(repo_dir, debug_dir)

    searched = 0
    while searched < args.vec_count:
        batch_start = args.vec_start + searched
        batch_count = min(args.batch_count, args.vec_count - searched)
        print(
            f"== searching dtype={args.dtype} seed={args.seed} "
            f"vec[{batch_start}, {batch_start + batch_count}) ==",
            flush=True,
        )
        search_timeout = max(args.timeout_cycles, args.timeout_cycles + batch_count)
        search_log = search_dir / "run.log"

        rc = run_make(
            args.make,
            [
                "run",
                f"DTYPE={args.dtype}",
                f"SEED={args.seed}",
                f"VEC_START={batch_start}",
                f"VEC_COUNT={batch_count}",
                f"CMODE_BITSIZE={args.cmode_bitsize}",
                f"RTL_BITSIZE={args.rtl_bitsize}",
                f"TIMEOUT_CYCLES={search_timeout}",
                f"ULP_TOLERANCE={args.ulp_tolerance}",
                "STOP_ON_FAIL=1",
                "VERBOSE_PASS=0",
                "TRACE=0",
                "DUMP_WAVE=0",
                f"RUN_DIR={search_dir}",
            ],
            log_path=search_log,
        )
        if rc == 0:
            searched += batch_count
            continue

        fail_log = search_dir / "fail_log.txt"
        try:
            fail = parse_fail_log(fail_log)
        except RuntimeError as exc:
            summary_path.write_text(
                "status=search_failed\n"
                f"dtype={args.dtype}\n"
                f"seed={args.seed}\n"
                f"cmode_bitsize={args.cmode_bitsize}\n"
                f"rtl_bitsize={args.rtl_bitsize}\n"
                f"batch_start={batch_start}\n"
                f"batch_count={batch_count}\n"
                f"return_code={rc}\n"
                f"fail_log={fail_log}\n"
                f"search_log={search_log}\n"
                "hint=Search failed before writing a mismatch entry; check for timeout, compile, or runtime errors.\n"
            )
            raise RuntimeError(
                f"search batch failed without a mismatch entry; see {search_log}. "
                "If this was a timeout, reduce BATCH_COUNT or increase TIMEOUT_CYCLES."
            ) from exc
        local_vec = int(fail["local_vec"])
        global_vec = batch_start + local_vec
        print(
            f"== first mismatch found: original_vec={global_vec} "
            f"got=0x{fail['got']} expected=0x{fail['expected']} ulp={fail['ulp']} ==",
            flush=True,
        )

        print(f"== replaying original_vec={global_vec} from saved search vectors ==", flush=True)
        replay_log = wave_dir / "run.log"
        replay_rc = run_make(
            args.make,
            [
                "run-existing-vectors",
                f"DTYPE={args.dtype}",
                f"SEED={args.seed}",
                "VEC_COUNT=1",
                f"RTL_VEC_START={local_vec}",
                f"EXISTING_VECTOR_DIR={search_dir / 'vectors'}",
                f"CMODE_BITSIZE={args.cmode_bitsize}",
                f"RTL_BITSIZE={args.rtl_bitsize}",
                f"TIMEOUT_CYCLES={args.timeout_cycles}",
                f"ULP_TOLERANCE={args.ulp_tolerance}",
                "STOP_ON_FAIL=1",
                "TRACE=1",
                "TRACE_DETAIL=0",
                "DUMP_WAVE=1",
                f"RUN_DIR={wave_dir}",
            ],
            log_path=replay_log,
        )

        wave_fail_log = wave_dir / "fail_log.txt"
        if replay_rc != 0:
            if wave_fail_log.exists():
                parse_fail_log(wave_fail_log)
            else:
                raise RuntimeError(
                    f"wave replay failed before writing a mismatch entry; see {replay_log}. "
                    "This usually means the replay simulator failed before it could report the mismatch."
                )

        print(f"== tracing C model for original_vec={global_vec} from saved inputs ==", flush=True)
        cmodel_trace_log = cmodel_trace_dir / "run.log"
        trace_rc = run_make(
            args.make,
            [
                "cmodel-replay",
                f"DTYPE={args.dtype}",
                f"SEED={args.seed}",
                f"CMODE_BITSIZE={args.cmode_bitsize}",
                f"RTL_BITSIZE={args.rtl_bitsize}",
                f"CMODE_REPLAY_VEC={local_vec}",
                f"CMODE_REPLAY_INPUT_DIR={search_dir / 'cmode_output'}",
                f"CMODE_REPLAY_SCALE_DIR={search_dir / 'cmode_scale'}",
                "CMODE_TRACE_INTERMEDIATE=1",
                f"RUN_DIR={cmodel_trace_dir}",
            ],
            log_path=cmodel_trace_log,
        )
        if trace_rc != 0:
            raise RuntimeError(f"C-model trace run failed; see {cmodel_trace_log}")

        fail.update(
            {
                "dtype": str(args.dtype),
                "seed": str(args.seed),
                "cmode_bitsize": str(args.cmode_bitsize),
                "rtl_bitsize": str(args.rtl_bitsize),
                "batch_start": str(batch_start),
                "global_vec": str(global_vec),
                "wave_file": str(wave_dir / "tb_top.fst"),
                "wave_fail_log": str(wave_fail_log),
                "cmodel_trace_dir": str(cmodel_trace_dir / "cmode_trace"),
                "cmodel_trace_log": str(cmodel_trace_log),
            }
        )
        write_summary(summary_path, fail)
        print(f"== summary: {summary_path} ==", flush=True)
        print(f"== waveform: {wave_dir / 'tb_top.fst'} ==", flush=True)
        print(f"== C-model trace: {cmodel_trace_dir / 'cmode_trace'} ==", flush=True)
        return 0

    print(
        f"PASS no mismatch found for dtype={args.dtype} seed={args.seed} "
        f"vec[{args.vec_start}, {args.vec_start + args.vec_count})",
        flush=True,
    )
    summary_path.write_text(
        "status=pass\n"
        f"dtype={args.dtype}\n"
        f"seed={args.seed}\n"
        f"cmode_bitsize={args.cmode_bitsize}\n"
        f"rtl_bitsize={args.rtl_bitsize}\n"
        f"vec_start={args.vec_start}\n"
        f"vec_count={args.vec_count}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
