#!/usr/bin/env python3
"""Check or sync RTL sources between rtl/pipe* and rtl/fpe/rtl.

Normal stage/module sources flow from rtl/pipe* into rtl/fpe/rtl.
defines.sv is special: rtl/fpe/rtl/defines.sv is the single source of truth
and is checked or synced back into every rtl/pipe* directory.
"""

import argparse
import filecmp
import shutil
import sys
from pathlib import Path


def same_file(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and filecmp.cmp(a, b, shallow=False)


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def collect_pipe_sources(rtl_root: Path) -> tuple[dict[str, Path], list[str]]:
    sources: dict[str, Path] = {}
    errors: list[str] = []

    for pipe_dir in sorted(rtl_root.glob("pipe*")):
        if not pipe_dir.is_dir():
            continue
        for src in sorted(pipe_dir.glob("*.sv")):
            if src.name == "defines.sv":
                continue
            old = sources.get(src.name)
            if old is not None:
                errors.append(f"duplicate module filename: {rel(old, rtl_root)} and {rel(src, rtl_root)}")
                continue
            sources[src.name] = src

    return sources, errors


def check_defines(rtl_root: Path, apply: bool) -> list[str]:
    errors: list[str] = []
    reference = rtl_root / "fpe" / "rtl" / "defines.sv"
    if not reference.exists():
        errors.append(f"missing defines reference: {rel(reference, rtl_root)}")
        return errors

    pipe_dirs = [pipe_dir for pipe_dir in sorted(rtl_root.glob("pipe*")) if pipe_dir.is_dir()]
    if not pipe_dirs:
        errors.append("no rtl/pipe* directories found")
        return errors

    for pipe_dir in pipe_dirs:
        dst = pipe_dir / "defines.sv"
        if not same_file(reference, dst):
            if apply:
                shutil.copy2(reference, dst)
                print(f"[COPY] {rel(reference, rtl_root)} -> {rel(dst, rtl_root)}")
            else:
                errors.append(f"defines mismatch: {rel(reference, rtl_root)} -> {rel(dst, rtl_root)}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync rtl/pipe*/*.sv into rtl/fpe/rtl, with defines.sv sourced from rtl/fpe/rtl."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="only report mismatches")
    mode.add_argument("--apply", action="store_true", help="copy changed pipe files into fpe/rtl")
    args = parser.parse_args()

    rtl_root = Path(__file__).resolve().parent
    fpe_rtl = rtl_root / "fpe" / "rtl"
    if not fpe_rtl.is_dir():
        print(f"missing destination directory: {fpe_rtl}", file=sys.stderr)
        return 2

    sources, errors = collect_pipe_sources(rtl_root)
    errors.extend(check_defines(rtl_root, args.apply))

    for name, src in sorted(sources.items()):
        dst = fpe_rtl / name
        if not dst.exists():
            if args.apply:
                shutil.copy2(src, dst)
                print(f"[COPY] {rel(src, rtl_root)} -> {rel(dst, rtl_root)}")
            else:
                errors.append(f"missing destination: {rel(src, rtl_root)} -> {rel(dst, rtl_root)}")
            continue

        if not same_file(src, dst):
            if args.apply:
                shutil.copy2(src, dst)
                print(f"[COPY] {rel(src, rtl_root)} -> {rel(dst, rtl_root)}")
            else:
                errors.append(f"out of sync: {rel(src, rtl_root)} -> {rel(dst, rtl_root)}")

    expected_names = set(sources)
    # CUTE instantiates this adapter as a BlackBox.  It belongs beside the
    # flattened FPE sources but intentionally has no pipe* source counterpart.
    expected_names.update({"defines.sv", "top.sv", "CuteFpeSv.sv"})
    for dst in sorted(fpe_rtl.glob("*.sv")):
        if dst.name not in expected_names:
            errors.append(f"extra fpe RTL file has no pipe source: {rel(dst, rtl_root)}")

    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        return 1

    if args.apply:
        print("OK: rtl/fpe/rtl is synced from rtl/pipe*, and pipe defines.sv files are synced from rtl/fpe/rtl.")
    else:
        print("OK: rtl/pipe* and rtl/fpe/rtl are in sync, using rtl/fpe/rtl/defines.sv as the defines reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
