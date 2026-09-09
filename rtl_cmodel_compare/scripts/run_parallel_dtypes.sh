#!/usr/bin/env bash
# Run independent dtype comparisons concurrently and retain per-dtype artifacts.

set -uo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
COMPARE_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
WORKSPACE_DIR="$(cd "$COMPARE_DIR/../.." && pwd)"
CHIPYARD_ENV="$WORKSPACE_DIR/third_party/CUTE/chipyard/env.sh"

DTYPES="${DTYPES:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
MAX_JOBS="${MAX_JOBS:-8}"
VEC_COUNT="${VEC_COUNT:-100000000}"
BATCH_COUNT="${BATCH_COUNT:-100000}"
SEED="${SEED:-1}"
CMODE_BITSIZE="${CMODE_BITSIZE:-512}"
RTL_BITSIZE="${RTL_BITSIZE:-512}"
TIMEOUT_CYCLES="${TIMEOUT_CYCLES:-3000}"
ULP_TOLERANCE="${ULP_TOLERANCE:-0}"
RUN_NAME="${RUN_NAME:-rtl_cmodel_512_parallel}"
LOG_DIR="${LOG_DIR:-$COMPARE_DIR/logs/$RUN_NAME}"
DEBUG_ROOT="${DEBUG_ROOT:-$COMPARE_DIR/$RUN_NAME}"

if [[ ! -f "$CHIPYARD_ENV" ]]; then
    echo "Chipyard environment was not found: $CHIPYARD_ENV" >&2
    exit 1
fi

if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer, got: $MAX_JOBS" >&2
    exit 1
fi

load_chipyard_env() {
    # Chipyard's activation hook references optional variables such as RISCV.
    set +u
    source "$CHIPYARD_ENV"
    set -u
}

run_worker() {
    local dtype="$1"
    local debug_dir="$DEBUG_ROOT/dtype_$dtype"
    local summary="$debug_dir/first_mismatch.txt"

    load_chipyard_env
    cd "$COMPARE_DIR"

    echo "== dtype $dtype started at $(date '+%F %T') =="
    if ! make -s debug-mismatch \
        DTYPE="$dtype" \
        VEC_COUNT="$VEC_COUNT" \
        BATCH_COUNT="$BATCH_COUNT" \
        SEED="$SEED" \
        CMODE_BITSIZE="$CMODE_BITSIZE" \
        RTL_BITSIZE="$RTL_BITSIZE" \
        TIMEOUT_CYCLES="$TIMEOUT_CYCLES" \
        ULP_TOLERANCE="$ULP_TOLERANCE" \
        DEBUG_DIR="$debug_dir"; then
        echo "ERROR: dtype $dtype stopped due to a runtime or build failure"
        return 1
    fi

    if [[ -f "$summary" ]] && grep -q '^status=pass$' "$summary"; then
        echo "== dtype $dtype passed at $(date '+%F %T') =="
        return 0
    fi

    echo "MISMATCH: dtype $dtype wrote $summary"
    [[ -f "$summary" ]] && cat "$summary"
    return 2
}

if [[ "${1:-}" == "--worker" ]]; then
    [[ -n "${2:-}" ]] || { echo "missing dtype for --worker" >&2; exit 1; }
    run_worker "$2"
    exit $?
fi

command -v setsid >/dev/null || { echo "setsid is required" >&2; exit 1; }
load_chipyard_env
cd "$COMPARE_DIR"
mkdir -p "$LOG_DIR" "$DEBUG_ROOT"

# Build once before workers start so they never race on the shared simulator build.
echo "== prebuilding simulator =="
make -s sim RTL_BITSIZE="$RTL_BITSIZE"

export DTYPES MAX_JOBS VEC_COUNT BATCH_COUNT SEED CMODE_BITSIZE RTL_BITSIZE
export TIMEOUT_CYCLES ULP_TOLERANCE RUN_NAME LOG_DIR DEBUG_ROOT

declare -A worker_dtype=()
active=0
failed=0

stop_workers() {
    local pid
    for pid in "${!worker_dtype[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
}

collect_one() {
    local finished_pid status dtype

    if wait -n -p finished_pid; then
        status=0
    else
        status=$?
    fi

    dtype="${worker_dtype[$finished_pid]:-unknown}"
    unset 'worker_dtype[$finished_pid]'
    active=$((active - 1))

    if [[ "$status" -ne 0 ]]; then
        echo "== dtype $dtype failed with status $status; stopping remaining workers ==" >&2
        failed="$status"
        stop_workers
        return 1
    fi

    echo "== dtype $dtype completed =="
    return 0
}

for dtype in $DTYPES; do
    while (( active >= MAX_JOBS )); do
        collect_one || break 2
    done

    echo "== launching dtype $dtype =="
    setsid "$SCRIPT_PATH" --worker "$dtype" >"$LOG_DIR/dtype_$dtype.log" 2>&1 &
    pid=$!
    worker_dtype[$pid]="$dtype"
    active=$((active + 1))
done

while (( active > 0 )); do
    collect_one || break
done

if [[ "$failed" -ne 0 ]]; then
    for pid in "${!worker_dtype[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$failed"
fi

echo "PASS all requested dtypes completed: $DTYPES"
