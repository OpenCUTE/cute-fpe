#!/usr/bin/env bash
# Runs the self-check and test1-test9, writes one log per program, and prints a PASS/FAIL summary.
# Usage: ./run_all.sh [log directory, default results/local]
set -u
cd "$(dirname "$0")"
OUT="${1:-results/local}"
mkdir -p "$OUT"

{
  echo "date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:   $(uname -srm)"
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | sed 's/^/gpu:    /'
  nvcc --version | tail -n 2 | sed 's/^/nvcc:   /'
  echo "flags:  -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a"
} > "$OUT/environment.txt"

status=0
for prog in selfcheck test1 test2 test3 test4 test5 test6 test7 test8 test9; do
  bin="build/$prog"
  if [ ! -x "$bin" ]; then echo "missing $bin (run make first)"; exit 2; fi
  start=$(date +%s)
  "$bin" > "$OUT/$prog.log" 2>&1
  rc=$?
  secs=$(( $(date +%s) - start ))
  summary=$(tail -n 1 "$OUT/$prog.log")
  printf '%-10s %-5s %4ss  %s\n' "$prog" "$([ $rc -eq 0 ] && echo PASS || echo FAIL)" "$secs" "$summary"
  [ $rc -eq 0 ] || status=1
  # every later test relies on the harness self-check
  if [ "$prog" = selfcheck ] && [ $rc -ne 0 ]; then echo "selfcheck failed, stopping"; exit 1; fi
done
exit $status
