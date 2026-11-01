#!/usr/bin/env bash
# Compare one or more RTL trees against the C model for every requested dtype.
# Each (rtl, dtype) job runs `make debug-mismatch` and stops at that dtype's first
# mismatch; the other jobs keep running. Results are collected in one summary file.

set -uo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
COMPARE_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

RTLS="${RTLS:-rtl add_tree_reuse mul_reduce reduce_reg}"
DTYPES="${DTYPES:-$(seq -s ' ' 0 21)}"
MAX_JOBS="${MAX_JOBS:-24}"
VEC_COUNT="${VEC_COUNT:-10000000}"
BATCH_COUNT="${BATCH_COUNT:-100000}"
SEED="${SEED:-1}"
TIMEOUT_CYCLES="${TIMEOUT_CYCLES:-3000}"
RUN_NAME="${RUN_NAME:-seed_${SEED}_count_${VEC_COUNT}}"
LOG_ROOT="$COMPARE_DIR/logs/$RUN_NAME"
SUMMARY="$LOG_ROOT/summary.txt"

if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer, got: $MAX_JOBS" >&2
    exit 1
fi

# Verilator must be on PATH, or CHIPYARD_ENV must point at an env.sh that provides it.
if ! command -v verilator >/dev/null && [[ -n "${CHIPYARD_ENV:-}" ]]; then
    [[ -f "$CHIPYARD_ENV" ]] || { echo "CHIPYARD_ENV not found: $CHIPYARD_ENV" >&2; exit 1; }
    set +u
    source "$CHIPYARD_ENV" >/dev/null 2>&1
    set -u
fi
command -v verilator >/dev/null || { echo "verilator not found; put it on PATH or set CHIPYARD_ENV=<path>/env.sh" >&2; exit 1; }

run_job() {
    local rtl="$1" dtype="$2"
    local debug_dir="results/$RUN_NAME/$rtl/dtype_$dtype"
    local log="$LOG_ROOT/$rtl/dtype_$dtype.log"
    local summary_file="$COMPARE_DIR/$debug_dir/first_mismatch.txt"
    local start rc status

    start=$(date +%s)
    echo "== $rtl dtype $dtype started at $(date '+%F %T') ==" >"$log"
    (cd "$COMPARE_DIR" && make -s debug-mismatch RTL="$rtl" \
        DTYPE="$dtype" VEC_COUNT="$VEC_COUNT" BATCH_COUNT="$BATCH_COUNT" SEED="$SEED" \
        TIMEOUT_CYCLES="$TIMEOUT_CYCLES" DEBUG_DIR="$debug_dir") >>"$log" 2>&1
    rc=$?

    if [[ "$rc" -ne 0 ]]; then
        status="ERROR(rc=$rc)"
    elif grep -q '^status=pass$' "$summary_file" 2>/dev/null; then
        status="PASS"
    elif grep -q '^original_vec=' "$summary_file" 2>/dev/null; then
        status="MISMATCH($(grep '^original_vec=' "$summary_file"))"
    else
        status="UNKNOWN"
    fi
    echo "== $rtl dtype $dtype finished at $(date '+%F %T'): $status ==" >>"$log"
    printf '%-15s dtype=%-2s %-40s %ss  %s\n' "$rtl" "$dtype" "$status" "$(( $(date +%s) - start ))" \
        "$summary_file" >>"$SUMMARY"
}

mkdir -p "$LOG_ROOT"
for rtl in $RTLS; do mkdir -p "$LOG_ROOT/$rtl"; done
: >"$SUMMARY"
echo "start $(date '+%F %T') rtls=[$RTLS] dtypes=[$DTYPES] vec_count=$VEC_COUNT seed=$SEED max_jobs=$MAX_JOBS"

# Build every simulator up front so concurrent jobs never rebuild the same obj_dir.
for rtl in $RTLS; do
    (cd "$COMPARE_DIR" && make -s sim RTL="$rtl" && make -s sim RTL="$rtl" TRACE=1 TRACE_DETAIL=0) \
        >"$LOG_ROOT/$rtl/build.log" 2>&1 || { echo "build failed for $rtl; see $LOG_ROOT/$rtl/build.log" >&2; exit 1; }
done

active=0
for dtype in $DTYPES; do
    for rtl in $RTLS; do
        while (( active >= MAX_JOBS )); do
            wait -n
            active=$((active - 1))
        done
        run_job "$rtl" "$dtype" &
        active=$((active + 1))
    done
done
wait

sort -k1,1 -k2,2V -o "$SUMMARY" "$SUMMARY"
echo "done $(date '+%F %T')"
echo "PASS: $(grep -c ' PASS ' "$SUMMARY")  MISMATCH: $(grep -c 'MISMATCH' "$SUMMARY")  ERROR: $(grep -c 'ERROR\|UNKNOWN' "$SUMMARY")"
cat "$SUMMARY"
