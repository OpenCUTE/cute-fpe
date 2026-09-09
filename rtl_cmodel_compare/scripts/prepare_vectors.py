#!/usr/bin/env python3
"""Convert C-model dumps into simple $readmemh files for the RTL testbench."""

import argparse
import json
from pathlib import Path


SUPPORTED_DTYPES = {
    0: {
        "name": "I8I8I32",
        "prefix": "m16n8k32_s8_s8_to_fp32",
        "cmodel_dtype": 0,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": False,
    },
    1: {
        "name": "F16F16F32",
        "prefix": "m16n8k16_fp16_fp16_to_fp32",
        "cmodel_dtype": 1,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 2,
        "b_elem_bytes": 2,
        "scale": False,
    },
    2: {
        "name": "BF16BF16F32",
        "prefix": "m16n8k16_bf16_bf16_to_fp32",
        "cmodel_dtype": 2,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 2,
        "b_elem_bytes": 2,
        "scale": False,
    },
    3: {
        "name": "TF32TF32F32",
        "prefix": "m16n8k8_tf32_tf32_to_fp32",
        "cmodel_dtype": 3,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 4,
        "b_elem_bytes": 4,
        "scale": False,
    },
    4: {
        "name": "I8U8I32",
        "prefix": "m16n8k32_s8_u8_to_fp32",
        "cmodel_dtype": 4,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": False,
    },
    5: {
        "name": "U8I8I32",
        "prefix": "m16n8k32_u8_s8_to_fp32",
        "cmodel_dtype": 5,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": False,
    },
    6: {
        "name": "U8U8I32",
        "prefix": "m16n8k32_u8_u8_to_fp32",
        "cmodel_dtype": 6,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": False,
    },
    7: {
        "name": "MXFP8E4M3F32",
        "prefix": "m16n8k32_mxe4m3_mxe4m3_to_fp32",
        "cmodel_dtype": 57,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": True,
    },
    8: {
        "name": "MXFP8E5M2F32",
        "prefix": "m16n8k32_mxe5m2_mxe5m2_to_fp32",
        "cmodel_dtype": 63,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": True,
    },
    9: {
        "name": "NVFP4F32",
        "prefix": "m16n8k64_nvfp4_nvfp4_to_fp32_b32",
        "cmodel_dtype": 9,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": True,
    },
    10: {
        "name": "MXFP4F32",
        "prefix": "m16n8k64_mxfp4_mxfp4_to_fp32_b32",
        "cmodel_dtype": 10,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": True,
        "scale_per_32_elements": True,
    },
    11: {
        "name": "FP8E4M3F32",
        "prefix": "m16n8k32_e4m3_e4m3_to_fp32",
        "cmodel_dtype": 32,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": False,
    },
    12: {
        "name": "FP8E5M2F32",
        "prefix": "m16n8k32_e5m2_e5m2_to_fp32",
        "cmodel_dtype": 38,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": False,
    },
    13: {
        "name": "MXFP6E3M2F32",
        "prefix": "m16n8k32_mxe3m2_mxe3m2_to_fp32",
        "cmodel_dtype": 75,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": True,
    },
    14: {
        "name": "MXFP6E2M3F32",
        "prefix": "m16n8k32_mxe2m3_mxe2m3_to_fp32",
        "cmodel_dtype": 69,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": True,
    },
    15: {
        "name": "I4I4I32",
        "prefix": "m16n8k64_s4_s4_to_s32",
        "cmodel_dtype": 15,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": False,
        "int4": True,
    },
    16: {
        "name": "I4U4I32",
        "prefix": "m16n8k64_s4_u4_to_s32",
        "cmodel_dtype": 16,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": False,
        "int4": True,
    },
    17: {
        "name": "U4I4I32",
        "prefix": "m16n8k64_u4_s4_to_s32",
        "cmodel_dtype": 17,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": False,
        "int4": True,
    },
    18: {
        "name": "U4U4I32",
        "prefix": "m16n8k64_u4_u4_to_s32",
        "cmodel_dtype": 18,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": False,
        "int4": True,
    },
    19: {
        "name": "MXINT4F32",
        "prefix": "m16n8k64_mxint4_mxint4_to_fp32_b32",
        "cmodel_dtype": 19,
        "a_bytes": 64,
        "b_bytes": 64,
        "a_elem_bytes": 1,
        "b_elem_bytes": 1,
        "scale": True,
        "scale_per_32_elements": True,
    },
}


def read_exact(path: Path, expected_size: int) -> bytes:
    data = path.read_bytes()
    if len(data) < expected_size:
        raise SystemExit(f"{path} is too small: got {len(data)} bytes, need {expected_size}")
    return data


def reverse_chunks(data: bytes, chunk_size: int) -> bytes:
    if len(data) % chunk_size != 0:
        raise ValueError(f"data size {len(data)} is not divisible by chunk size {chunk_size}")
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    return b"".join(reversed(chunks))


def bytes_to_sv_hex(data: bytes, width_bytes: int, elem_bytes: int) -> str:
    if len(data) > width_bytes:
        raise ValueError(f"vector has {len(data)} bytes, width is only {width_bytes}")
    padded = data + bytes(width_bytes - len(data))
    return reverse_chunks(padded, elem_bytes).hex()


def infer_payload_bytes(path: Path, vec_count: int, max_bytes: int, label: str) -> int:
    size = path.stat().st_size
    if size % vec_count != 0:
        raise SystemExit(
            f"{path} size is not divisible by vec-count for {label}: "
            f"{size} bytes / {vec_count} vectors"
        )
    payload_bytes = size // vec_count
    if payload_bytes <= 0 or payload_bytes > max_bytes:
        raise SystemExit(
            f"{path} has unsupported {label} payload width: "
            f"{payload_bytes} bytes per vector, max is {max_bytes}"
        )
    return payload_bytes


def scale_bytes_to_sv_hex(data: bytes, width_bytes: int) -> str:
    if len(data) > width_bytes:
        raise ValueError(f"scale vector has {len(data)} bytes, width is only {width_bytes}")
    return data.hex().rjust(width_bytes * 2, "0")


def read_text_bits(path: Path, count: int) -> list[str]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < count:
        raise SystemExit(f"{path} has {len(lines)} values, need {count}")
    out = []
    for idx, line in enumerate(lines[:count]):
        if len(line) != 32 or any(ch not in "01" for ch in line):
            raise SystemExit(f"{path}:{idx + 1} is not a 32-bit binary string: {line!r}")
        out.append(f"{int(line, 2):08x}")
    return out


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=int, required=True)
    parser.add_argument("--vec-count", type=int, required=True)
    parser.add_argument("--cmodel-output-dir", type=Path, required=True)
    parser.add_argument("--cmodel-scale-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cmodel-bitsize", type=int, default=512)
    parser.add_argument("--rtl-bitsize", type=int, default=512)
    parser.add_argument(
        "--int4-k",
        type=int,
        default=0,
        help="exact packed S4/U4 dot-product length for dtype 15-18",
    )
    args = parser.parse_args()

    if args.cmodel_bitsize <= 0 or args.cmodel_bitsize % 32 != 0:
        raise SystemExit(f"--cmodel-bitsize must be a positive multiple of 32, got {args.cmodel_bitsize}")
    if args.rtl_bitsize <= 0 or args.rtl_bitsize % 64 != 0:
        raise SystemExit(f"--rtl-bitsize must be a positive multiple of 64, got {args.rtl_bitsize}")
    if args.vec_count <= 0:
        raise SystemExit(f"--vec-count must be positive, got {args.vec_count}")

    cmodel_width_bytes = args.cmodel_bitsize // 8
    rtl_width_bytes = args.rtl_bitsize // 8
    rtl_scale_width_bits = args.rtl_bitsize // 8
    rtl_scale_width_bytes = rtl_scale_width_bits // 8
    if cmodel_width_bytes > rtl_width_bytes:
        raise SystemExit(
            f"C-model vector width ({args.cmodel_bitsize} bits) is wider than RTL "
            f"input width ({args.rtl_bitsize} bits)"
        )
    if args.dtype not in SUPPORTED_DTYPES:
        raise SystemExit(
            f"Unsupported dtype {args.dtype}. First-stage flow supports: "
            f"{', '.join(str(k) for k in sorted(SUPPORTED_DTYPES))}"
        )

    cfg = SUPPORTED_DTYPES[args.dtype]
    prefix = cfg["prefix"]
    is_int4 = bool(cfg.get("int4", False))
    if is_int4:
        if args.int4_k <= 0:
            raise SystemExit(f"dtype {args.dtype} requires --int4-k > 0")
        if args.int4_k > args.cmodel_bitsize // 4:
            raise SystemExit(
                f"--int4-k {args.int4_k} exceeds C-model input capacity "
                f"({args.cmodel_bitsize // 4} packed 4-bit elements)"
            )
        if args.int4_k > args.rtl_bitsize // 4:
            raise SystemExit(
                f"--int4-k {args.int4_k} exceeds RTL input capacity "
                f"({args.rtl_bitsize // 4} packed 4-bit elements)"
            )
        prefix = f"{prefix}_k{args.int4_k}"
        a_bytes = (args.int4_k + 1) // 2
        b_bytes = (args.int4_k + 1) // 2
    else:
        if args.int4_k != 0:
            raise SystemExit("--int4-k is valid only for dtype 15-18")
        a_bytes = min(cmodel_width_bytes, int(cfg["a_bytes"]))
        b_bytes = min(cmodel_width_bytes, int(cfg["b_bytes"]))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    a_path = args.cmodel_output_dir / f"{prefix}_a_double.bin.log"
    b_path = args.cmodel_output_dir / f"{prefix}_b_double.bin.log"
    c_path = args.cmodel_output_dir / f"{prefix}_c_double.bin.log"
    d_path = args.cmodel_output_dir / f"{prefix}_d.txt.log"

    a_data = read_exact(a_path, a_bytes * args.vec_count)
    b_data = read_exact(b_path, b_bytes * args.vec_count)
    c_data = read_exact(c_path, 4 * args.vec_count)
    expected = read_text_bits(d_path, args.vec_count)

    a_lines = []
    b_lines = []
    c_lines = []
    if cfg["scale"]:
        a_scale_path = args.cmodel_scale_dir / f"{prefix}_a_double.bin.log"
        b_scale_path = args.cmodel_scale_dir / f"{prefix}_b_double.bin.log"
        a_scale_payload_bytes = infer_payload_bytes(
            a_scale_path, args.vec_count, rtl_scale_width_bytes, "A scale"
        )
        b_scale_payload_bytes = infer_payload_bytes(
            b_scale_path, args.vec_count, rtl_scale_width_bytes, "B scale"
        )
        if cfg.get("scale_per_32_elements"):
            expected_scale_bytes = (args.cmodel_bitsize + 127) // 128
            if (a_scale_payload_bytes != expected_scale_bytes or
                    b_scale_payload_bytes != expected_scale_bytes):
                raise SystemExit(
                    f"dtype {args.dtype} requires {expected_scale_bytes} scale bytes per operand "
                    f"for {args.cmodel_bitsize}-bit input, got "
                    f"A={a_scale_payload_bytes}, B={b_scale_payload_bytes}"
                )
        a_scale_data = read_exact(a_scale_path, a_scale_payload_bytes * args.vec_count)
        b_scale_data = read_exact(b_scale_path, b_scale_payload_bytes * args.vec_count)
        a_scale_lines = []
        b_scale_lines = []
    else:
        a_scale_path = None
        b_scale_path = None
        a_scale_data = b_scale_data = b""
        a_scale_payload_bytes = 0
        b_scale_payload_bytes = 0
        zero_scale = "0" * (rtl_scale_width_bytes * 2)
        a_scale_lines = [zero_scale for _ in range(args.vec_count)]
        b_scale_lines = [zero_scale for _ in range(args.vec_count)]

    for vec in range(args.vec_count):
        a_start = vec * a_bytes
        b_start = vec * b_bytes
        c_start = vec * 4

        a_lines.append(
            bytes_to_sv_hex(
                a_data[a_start : a_start + a_bytes],
                rtl_width_bytes,
                cfg["a_elem_bytes"],
            )
        )
        b_lines.append(
            bytes_to_sv_hex(
                b_data[b_start : b_start + b_bytes],
                rtl_width_bytes,
                cfg["b_elem_bytes"],
            )
        )
        c_lines.append(c_data[c_start : c_start + 4].hex())
        if cfg["scale"]:
            a_scale_start = vec * a_scale_payload_bytes
            b_scale_start = vec * b_scale_payload_bytes
            a_scale_payload = a_scale_data[a_scale_start : a_scale_start + a_scale_payload_bytes]
            b_scale_payload = b_scale_data[b_scale_start : b_scale_start + b_scale_payload_bytes]
            a_scale_lines.append(
                scale_bytes_to_sv_hex(
                    a_scale_payload,
                    rtl_scale_width_bytes,
                )
            )
            b_scale_lines.append(
                scale_bytes_to_sv_hex(
                    b_scale_payload,
                    rtl_scale_width_bytes,
                )
            )

    write_lines(out_dir / "A.txt", a_lines)
    write_lines(out_dir / "B.txt", b_lines)
    write_lines(out_dir / "C.txt", c_lines)
    write_lines(out_dir / "A_scale.txt", a_scale_lines)
    write_lines(out_dir / "B_scale.txt", b_scale_lines)
    write_lines(out_dir / "D.txt", expected)

    notes = [
        "C-model binary dumps are written in big-endian element order.",
        "A/B text files reverse element order so $readmemh maps element 0 into the low bits of the RTL bus.",
        "Shorter C-model payloads are zero-padded up to the RTL input bus before element-order reversal.",
        "Scale inputs are zero-filled for non-scale opcodes; RTL decode_scale ignores them when scale_en=0.",
        "MXFP8/MXFP6 scale payloads are placed in RTL raw scale slots in C-model block order.",
        "NVFP4 scale payloads use one scale byte per 8 payload bytes.",
    ]
    if is_int4:
        notes.append("S4/U4 payload bytes retain low-nibble-first packing: element 0 is bits [3:0].")
    if cfg.get("scale_per_32_elements"):
        notes.append("MXFP4/MXINT4 use one external scale byte per 32 four-bit elements; unused RTL scale-bus bytes are zero.")

    metadata = {
        "dtype": args.dtype,
        "rtl_dtype": args.dtype,
        "cmodel_dtype": cfg["cmodel_dtype"],
        "name": cfg["name"],
        "prefix": prefix,
        "vec_count": args.vec_count,
        "cmodel_bitsize": args.cmodel_bitsize,
        "rtl_reduce_width_bits": args.rtl_bitsize,
        "rtl_scale_width_bits": rtl_scale_width_bits,
        "a_bytes_per_vec": a_bytes,
        "b_bytes_per_vec": b_bytes,
        "int4_k": args.int4_k if is_int4 else None,
        "scale_bytes_per_vec": a_scale_payload_bytes if cfg["scale"] else 0,
        "rtl_scale_bus_bytes": rtl_scale_width_bytes,
        "a_scale_payload_bytes_per_vec": a_scale_payload_bytes,
        "b_scale_payload_bytes_per_vec": b_scale_payload_bytes,
        "source_files": {
            "a": str(a_path),
            "b": str(b_path),
            "c": str(c_path),
            "d": str(d_path),
            "a_scale": str(a_scale_path) if a_scale_path else None,
            "b_scale": str(b_scale_path) if b_scale_path else None,
        },
        "notes": notes,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {args.vec_count} vectors for dtype {args.dtype} ({cfg['name']}) in {out_dir}")


if __name__ == "__main__":
    main()
