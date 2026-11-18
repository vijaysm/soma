#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
#
# The Williamson benchmarks of the shallow-water atmosphere:
#   case 2: convergence in p at ne 4, 8 and 16 (p 2 to 8), 5 days, no viscosity;
#   cases 5 and 6: ne30 p = 3 for 15 days against REFDIR/case5.nc and
#   REFDIR/case6.nc, passing when the normalised l2 height error is at most twice
#   the mark in REFDIR/marks.txt (lines "case5 <l2> <citation>").
#
# usage: run_cases.sh REFDIR NP
# environment: SWE_ATM (the executable; default build/apps/swe_atm/swe_atm),
#              MPIEXEC (default mpiexec), OMP_NUM_THREADS (threads per rank).
set -euo pipefail
if [ $# -ne 2 ]; then
  echo "usage: $0 REFDIR NP" >&2
  exit 2
fi
refdir=$1
np=$2
exe=${SWE_ATM:-build/apps/swe_atm/swe_atm}
mpiexec=${MPIEXEC:-mpiexec}
run() { "$mpiexec" -n "$np" "$exe" "$@"; }

echo "# case 2: l2 height error after 5 days"
printf "%4s" "ne"
for p in 2 3 4 5 6 7 8; do printf " %12s" "p=$p"; done
echo
for ne in 4 8 16; do
  printf "%4d" "$ne"
  for p in 2 3 4 5 6 7 8; do
    l2=$(run --case w2 --ne "$ne" --order "$p" --days 5 --nu 0 |
      awk '/^t / { for (i = 1; i < NF; ++i) if ($i == "l2") v = $(i + 1) } END { print v }')
    printf " %12s" "$l2"
  done
  echo
done

status=0
for c in 5 6; do
  echo "# case $c: ne30 p = 3, 15 days"
  file="$refdir/case$c.nc"
  args=(--case "w$c" --ne 30 --order 3 --days 15 --report-every 24)
  if [ -f "$file" ]; then args+=(--reference "$file"); fi
  out=$(run "${args[@]}")
  echo "$out" | grep -E "^(t |reference|simulated)"
  if [ ! -f "$file" ]; then
    echo "case $c: no reference ($file); conservation only"
    continue
  fi
  l2=$(echo "$out" | awk '/^reference/ { for (i = 1; i < NF; ++i) if ($i == "l2") print $(i + 1) }')
  mark=$(awk -v k="case$c" '$1 == k { print $2 }' "$refdir/marks.txt" 2>/dev/null || true)
  if [ -z "$mark" ]; then
    echo "case $c: l2 $l2; no mark in $refdir/marks.txt"
  elif awk -v a="$l2" -v b="$mark" 'BEGIN { exit !(a <= 2 * b) }'; then
    echo "case $c: PASS (l2 $l2 <= 2 x $mark)"
  else
    echo "case $c: FAIL (l2 $l2 > 2 x $mark)"
    status=1
  fi
done
exit $status
