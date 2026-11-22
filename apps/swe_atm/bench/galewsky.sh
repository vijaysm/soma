#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
#
# Galewsky et al. (2004): the unperturbed jet at ne30 p = 3 for 6 days (its
# drift from the initial state), then the perturbed jet at ne30, ne60 and ne120,
# p = 3, to day 6, with vorticity sampled on a 361 x 720 grid; the lower
# resolutions' differences from ne120 must fall.
#
# usage: galewsky.sh NP [OUTDIR]
# environment: SWE_ATM, MPIEXEC, OMP_NUM_THREADS as run_cases.sh.
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "usage: $0 NP [OUTDIR]" >&2
  exit 2
fi
np=$1
out=${2:-galewsky}
exe=${SWE_ATM:-build/apps/swe_atm/swe_atm}
mpiexec=${MPIEXEC:-mpiexec}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"

echo "# steady jet, ne30 p = 3, 6 days"
"$mpiexec" -n "$np" "$exe" --case galewsky-steady --ne 30 --order 3 --days 6 --report-every 24 |
  grep -E "^(t |simulated)"

for ne in 30 60 120; do
  echo "# perturbed jet, ne$ne p = 3, 6 days"
  "$mpiexec" -n "$np" "$exe" --case galewsky --ne "$ne" --order 3 --days 6 --report-every 24 \
    --latlon "361,720,$out/ne$ne" | grep -E "^(t |simulated)"
done
python3 "$here/compare_npy.py" "$out/ne30_vorticity.npy" "$out/ne60_vorticity.npy" "$out/ne120_vorticity.npy"
