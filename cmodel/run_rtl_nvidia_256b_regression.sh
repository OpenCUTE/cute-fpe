#!/usr/bin/env bash

set -uo pipefail

CMODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$CMODE_DIR/necessary_source"
VECTOR_COUNT="${VECTOR_COUNT:-100000000}"
SEED="${SEED:-1}"
PRINT_EVERY="${PRINT_EVERY:-1000000}"
RUN_DIR="${RUN_DIR:-$CMODE_DIR/rtl_nvidia_256b_$(date +%Y%m%d_%H%M%S)}"
BINARY="$SOURCE_DIR/all_fmac_test_rtl_nv_256b"

mkdir -p "$RUN_DIR"
STATUS_FILE="$RUN_DIR/status.tsv"
MANIFEST_FILE="$RUN_DIR/mapping.tsv"

if ! nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader > "$RUN_DIR/gpu.txt"; then
    echo "GPU preflight failed; no NVIDIA device is available." >&2
    exit 1
fi

make -C "$SOURCE_DIR" build_256b \
    NVIDIA_ON=1 \
    FMAC_DUMP_OUTPUT=0 \
    TARGET_256B=all_fmac_test_rtl_nv_256b

cat > "$MANIFEST_FILE" <<'EOF'
rtl_opcode	cmode_dtype	name
0	0	I8I8I32
1	1	F16F16F32
2	2	BF16BF16F32
3	3	TF32TF32F32
4	4	I8U8I32
5	5	U8I8I32
6	6	U8U8I32
7	57	MXFP8E4M3F32
8	63	MXFP8E5M2F32
9	9	NVFP4F32
10	10	MXFP4F32
11	32	FP8E4M3F32
12	38	FP8E5M2F32
13	75	MXFP6E3M2F32
14	69	MXFP6E2M3F32
15	15	I4I4I32
16	16	I4U4I32
17	17	U4I4I32
18	18	U4U4I32
EOF

printf 'rtl_opcode\tcmode_dtype\tname\tresult\texit_code\tlog\n' > "$STATUS_FILE"

while IFS=$'\t' read -r rtl_opcode cmode_dtype name; do
    if [[ "$rtl_opcode" == "rtl_opcode" ]]; then
        continue
    fi

    log="$RUN_DIR/rtl_${rtl_opcode}_cmode_${cmode_dtype}_${name}.log"
    "$BINARY" \
        --dtype "$cmode_dtype" \
        --bitsize 256 \
        --vec-count "$VECTOR_COUNT" \
        --seed "$SEED" \
        --seed-mode per_vec \
        --print-every "$PRINT_EVERY" \
        > "$log" 2>&1
    exit_code=$?

    if ! grep -q '^CUDA device 0: NVIDIA' "$log"; then
        result="INVALID_NO_NVIDIA_MMA"
    elif [[ "$exit_code" -eq 0 ]]; then
        result="PASS"
    elif grep -q 'first mismatch at vec' "$log"; then
        result="MISMATCH"
    else
        result="ERROR"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$rtl_opcode" "$cmode_dtype" "$name" "$result" "$exit_code" "$log" \
        >> "$STATUS_FILE"
done < "$MANIFEST_FILE"
