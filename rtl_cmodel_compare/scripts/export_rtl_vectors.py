#!/usr/bin/env python3
"""Export one stored directed case to rtl_cmodel_compare readmemh files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "config" / "modes.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_hex(value: str) -> str:
    text = value[2:] if value.startswith("0x") else value
    int(text, 16)
    return text.lower()


def fit_hex(value: str, width_bits: int, label: str) -> str:
    text = clean_hex(value)
    width_hex = (width_bits + 3) // 4
    if len(text) > width_hex:
        raise ValueError(f"{label} is wider than {width_bits} bits")
    return text.rjust(width_hex, "0")


def pack_elements(values: list[str], element_bits: int, width_bits: int, label: str) -> str:
    packed = 0
    limit = 1 << element_bits
    for index, value in enumerate(values):
        raw = int(clean_hex(value), 16)
        if raw >= limit:
            raise ValueError(f"{label}[{index}] exceeds {element_bits} bits")
        packed |= raw << (index * element_bits)
    return f"{packed:0{(width_bits + 3) // 4}x}"


def pack_operand(case: dict[str, Any], mode: dict[str, Any], operand: str, rtl_bits: int) -> str:
    explicit = case.get("rtl_inputs", {}).get(f"{operand}_hex")
    if explicit is not None:
        return fit_hex(explicit, rtl_bits, f"rtl_inputs.{operand}_hex")
    values = case.get(f"{operand}_bits")
    if values is not None:
        return pack_elements(values, mode[f"{operand}_element_bits"], rtl_bits, operand)
    packed = case.get(f"{operand}_packed_hex")
    if packed is None:
        raise ValueError(f"case has no {operand} payload")
    return fit_hex(packed, rtl_bits, f"{operand}_packed_hex")


def pack_scales(case: dict[str, Any], operand: str, scale_bits: int) -> str:
    explicit = case.get("rtl_inputs", {}).get(f"{operand}_scale_hex")
    if explicit is not None:
        return fit_hex(explicit, scale_bits, f"rtl_inputs.{operand}_scale_hex")
    values = case.get(f"{operand}_scale_bits", [])
    return pack_elements(values, 8, scale_bits, f"{operand}_scale_bits")


def choose_expected(case: dict[str, Any], source: str) -> str:
    if source == "model":
        value = case.get("observed", {}).get("model_d_bits")
    elif source == "gpu":
        value = case.get("observed", {}).get("gpu_d_bits")
    else:
        value = case.get("expected_math_bits")
    if value is None:
        raise ValueError(f"case has no {source} expected result")
    return fit_hex(value, 32, f"{source} expected result")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rtl-bits", type=int, default=512)
    parser.add_argument("--scale-bits", type=int, default=64)
    parser.add_argument("--expected-source", choices=["model", "gpu", "math"], default="model")
    args = parser.parse_args()

    case = load_json(args.case)
    manifest = load_json(MANIFEST_PATH)
    modes = {mode["id"]: mode for mode in manifest["modes"]}
    mode = modes.get(case["mode_id"])
    if mode is None:
        raise SystemExit(f"unknown mode_id: {case['mode_id']}")
    if mode["rtl_opcode"] is None:
        raise SystemExit(f"mode has no RTL opcode: {mode['id']}")

    vectors = {
        "A.txt": pack_operand(case, mode, "a", args.rtl_bits),
        "B.txt": pack_operand(case, mode, "b", args.rtl_bits),
        "C.txt": fit_hex(
            case.get("rtl_inputs", {}).get("c_hex", case["c_bits"]), 32, "C"
        ),
        "A_scale.txt": pack_scales(case, "a", args.scale_bits),
        "B_scale.txt": pack_scales(case, "b", args.scale_bits),
        "D.txt": choose_expected(case, args.expected_source),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in vectors.items():
        (args.out_dir / filename).write_text(value + "\n", encoding="ascii")
    metadata = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "mode_id": mode["id"],
        "rtl_opcode": mode["rtl_opcode"],
        "rtl_bits": args.rtl_bits,
        "scale_bits": args.scale_bits,
        "expected_source": args.expected_source,
        "source_case": str(args.case.resolve()),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WROTE RTL vectors for {case['case_id']} to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
