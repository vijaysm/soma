#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Mesh-quality statistics and stereographic renders of a regionally refined SOMA mesh:
#
#   PARAVIEW_BIN=<ParaView bin dir> [NP=16] tools/somaview/examples/conus/conus_job.sh FILE.soma OUTDIR
#
# conus_prepass.py (mpiexec, somaview's core with mpi4py and an MPI h5py) writes OUTDIR/prepass.json;
# conus_stereo.py and conus_telescope.py (ParaView's pvbatch, under its own mpiexec) render into
# OUTDIR/img. level_spacing.py writes the cell size of each refinement level.
set -x
set +e +o pipefail
F=${1:?usage: conus_job.sh FILE.soma OUTDIR}
OUT=${2:?usage: conus_job.sh FILE.soma OUTDIR}
PV=${PARAVIEW_BIN:?set PARAVIEW_BIN to the ParaView bin directory}
NP=${NP:-16}
HERE=$(cd "$(dirname "$0")" && pwd)
SV=$(cd "$HERE/../.." && pwd)
export PYTHONPATH=$SV/src${PYTHONPATH:+:$PYTHONPATH} OMP_NUM_THREADS=1
mkdir -p "$OUT/img" "$OUT/logs"
mpiexec -n $NP python "$HERE/conus_prepass.py" "$F" "$OUT/prepass.json" > "$OUT/logs/prepass.txt" 2>&1
mpiexec -n $NP python "$HERE/level_spacing.py" "$F" "$OUT/levels.json" > "$OUT/logs/levels.txt" 2>&1
for s in conus_stereo conus_telescope; do
  "$PV/mpiexec" -n $NP "$PV/pvbatch" "$HERE/$s.py" "$SV/paraview/SomaReader.py" "$F" "$OUT/prepass.json" "$OUT/img" \
    > "$OUT/logs/$s.txt" 2>&1
done
grep -h rendered "$OUT"/logs/conus_*.txt
