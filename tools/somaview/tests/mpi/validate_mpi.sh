#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# somaview's distributed checks on a machine with MPI, an MPI-enabled h5py and mpi4py:
#
#   SOMA_PREFIX=<SOMA install, built with MPI and HDF5> WORK=<scratch dir> \
#   [PARAVIEW_BIN=<ParaView bin dir>] [CONUS=<large SOMA file>] tools/somaview/tests/mpi/validate_mpi.sh
#
# 1. writes multi-part files with SOMA's own MPI writer (tests/make_data.sh);
# 2. runs the core and front-end suites; 3. reads every file under mpiexec at 1 to 8 ranks with
# both halo methods (mpi_validate.py); 4. writes VTKHDF collectively and reads it back with VTK;
# 5. times a large file (CONUS) at 8 to 48 ranks; 6. with ParaView, runs the plugin suite and
# pvbatch piece checks; 7. renders the ParaView and pyvista galleries. Checks are read from
# $WORK/logs; every step runs even when an earlier one fails.
set -x
set +e +o pipefail
: "${SOMA_PREFIX:?set SOMA_PREFIX}" "${WORK:?set WORK}"
HERE=$(cd "$(dirname "$0")" && pwd)
SV=$(cd "$HERE/../.." && pwd)            # tools/somaview
J=$WORK; W=$WORK
S=$WORK/src; mkdir -p "$S" && ln -sfn "$SV" "$S/somaview"
cp "$HERE"/mpi_validate.py "$HERE"/pv_mpi_check.py "$S"/
cp "$SV"/paraview/examples/pv_gallery.py "$S"/
export PYTHONPATH=$SV/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$J/img" "$J/logs"
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false

# -- 1. SOMA's MPI writer: true multi-part files ---------------------------------------------
"$SV/tests/make_data.sh" "$SOMA_PREFIX" "$W/data" > "$J/logs/make_data.txt" 2>&1
M=$W/data/mpi
ls -la "$M"
python -c "
import somaview as sv, glob
for f in sorted(glob.glob('$M/*.soma')):
    o = sv.open(f); print(o); o.close()
" | tee "$J/logs/mpidata_files.txt"

# -- 2. the core and front-end suites on Linux (VTK 9.7, pyvista with EGL) --------------------
cd $SV && SOMAVIEW_DATA=$W/data python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -15 | tee $J/logs/pytest_linux.txt
cd $J

# -- 3. distributed reading under mpiexec, SOMA-written files ---------------------------------
for f in hex_4p hex_7p tet_4p tet_7p quad_4p quad_7p hexderived_4p hexderived_7p; do
  for P in 1 2 3 4 5 8; do
    mpiexec -n $P python $S/mpi_validate.py $M/$f.soma --analytic >> logs/mpi_results.jsonl 2>> logs/mpi_err.txt || echo "{\"file\": \"$f\", \"ranks\": $P, \"ok\": false, \"crash\": true}" >> logs/mpi_results.jsonl
  done
done
for f in bricks_4p ocn_3p; do
  for P in 1 2 3 4 6; do
    mpiexec -n $P python $S/mpi_validate.py $M/$f.soma >> logs/mpi_results.jsonl 2>> logs/mpi_err.txt || echo "{\"file\": \"$f\", \"ranks\": $P, \"ok\": false, \"crash\": true}" >> logs/mpi_results.jsonl
  done
done
for f in amr3d_4p amr3d_7p; do
  for P in 1 3 4 8; do
    mpiexec -n $P python $S/mpi_validate.py $M/$f.soma >> logs/mpi_results.jsonl 2>> logs/mpi_err.txt || echo "{\"file\": \"$f\", \"ranks\": $P, \"ok\": false, \"crash\": true}" >> logs/mpi_results.jsonl
  done
done

# -- 4. parallel VTKHDF writer under mpiexec, read back by VTK's vtkHDFReader -----------------
for f in hex_4p hex_7p; do
  rm -f $W/$f.vtkhdf
  mpiexec -n 4 python -c "from mpi4py import MPI; import somaview as sv; sv.write_vtkhdf('$M/$f.soma', '$W/$f.vtkhdf', comm=MPI.COMM_WORLD)" 2>> logs/vtkhdf_err.txt || echo "write failed $f" >> logs/vtkhdf.txt
  python - >> logs/vtkhdf.txt 2>&1 <<EOF
import numpy as np, somaview as sv
from vtkmodules.vtkIOHDF import vtkHDFReader
from vtkmodules.vtkFiltersCore import vtkAppendFilter
from vtkmodules.util.numpy_support import vtk_to_numpy
r = vtkHDFReader(); r.SetFileName('$W/$f.vtkhdf'); r.Update()
out = r.GetOutputDataObject(0)
a = vtkAppendFilter(); a.AddInputData(out) if out.IsA('vtkDataSet') else None
if not out.IsA('vtkDataSet'):
    it = out.NewIterator(); it.InitTraversal()
    while not it.IsDoneWithTraversal():
        a.AddInputData(it.GetCurrentDataObject()); it.GoToNextItem()
a.Update(); ug = a.GetOutput()
p = sv.read('$M/$f.soma')
cg = vtk_to_numpy(ug.GetCellData().GetArray('soma_gid'))
pg = vtk_to_numpy(ug.GetPointData().GetArray('soma_gid'))
xyz = vtk_to_numpy(ug.GetPoints().GetData())
pr = vtk_to_numpy(ug.GetPointData().GetArray('pressure'))
o = np.argsort(p.point_gids); k = o[np.searchsorted(p.point_gids[o], pg)]
print('VTKHDF $f', out.GetClassName(), 'cells', ug.GetNumberOfCells(), 'expect', len(p.cells),
      'cells_ok', bool(np.array_equal(np.sort(cg), np.sort(p.cells.gids))),
      'coords_ok', bool(np.allclose(xyz, p.points[k])), 'pressure_ok', bool(np.allclose(pr, p.point_data['pressure'][k])))
EOF
done

# -- 5. a large file (CONUS): one part alone, then 8 to 48 ranks -----------------------------
C=${CONUS:-}
if [ -n "$C" ]; then
python - >> logs/conus.jsonl 2>> logs/conus_err.txt <<EOF
import json, time, resource, somaview as sv
t = time.perf_counter(); p = sv.read('$C', parts=[17], cell_fields=['spacing_km']); dt = time.perf_counter() - t
print(json.dumps({"file": "$C", "mode": "one part, one process", "part": 17, "cells": len(p.cells), "halo": p.halo_points,
                  "seconds": round(dt, 2), "rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}))
EOF
for P in 8 16; do
  mpiexec -n $P python $S/mpi_validate.py $C --cell-fields spacing_km >> logs/conus.jsonl 2>> logs/conus_err.txt || echo "{\"conus\": $P, \"crash\": true}" >> logs/conus.jsonl
done
for P in 24 48; do
  mpiexec -n $P --oversubscribe python $S/mpi_validate.py $C --cell-fields spacing_km --timing-only >> logs/conus.jsonl 2>> logs/conus_err.txt \
    || mpiexec -n $P python $S/mpi_validate.py $C --cell-fields spacing_km --timing-only >> logs/conus.jsonl 2>> logs/conus_err.txt \
    || echo "{\"conus\": $P, \"crash\": true}" >> logs/conus.jsonl
done
fi

# -- 6. ParaView 6.1.1: plugin suite, then the reader under MPI pvbatch -----------------------
PV=${PARAVIEW_BIN:-}
if [ -n "$PV" ]; then
cd $S/somaview/paraview
SOMAVIEW_DATA=$W/data SOMAVIEW_NE20=$M/quad_7p.soma $PV/pvpython --no-mpi --force-offscreen-rendering tests/test_paraview.py > $J/logs/pv_suite.txt 2>&1 || true
grep -E "FAIL|All checks|checks" $J/logs/pv_suite.txt | tail -5
cd $J
ls $PV | grep -i mpi
for f in hex_4p hex_7p quad_7p bricks_4p ocn_3p; do
  for P in 1 2 4 7; do
    $PV/mpiexec -n $P $PV/pvbatch --symmetric $S/pv_mpi_check.py $S/somaview/paraview/SomaReader.py $M/$f.soma 2>> logs/pv_mpi_err.txt | grep PVMPI >> logs/pv_mpi.txt || echo "PVMPI {\"file\": \"$f\", \"ranks\": $P, \"crash\": true}" >> logs/pv_mpi.txt
  done
done

# -- 7. renders: ParaView (IceT compositing over MPI ranks) and pyvista (EGL) ----------------
$PV/mpiexec -n 4 $PV/pvbatch $S/pv_gallery.py $S/somaview/paraview/SomaReader.py $J/img \
  hexmpi=$M/hex_4p.soma bricks=$M/bricks_4p.soma highorder=$W/data/highorder3d.soma amr=$M/amr3d_4p.soma > logs/pv_gallery.txt 2>&1 || true
[ -n "$C" ] && $PV/mpiexec -n 8 $PV/pvbatch $S/pv_gallery.py $S/somaview/paraview/SomaReader.py $J/img conus=$C > logs/pv_gallery_conus.txt 2>&1 || true
fi
cd $SV && python examples/gallery.py --data $W/data --out $J/img_pyvista > $J/logs/pyvista_gallery.txt 2>&1 || true
cd $J
grep -v "^\s*$" logs/pv_gallery.txt | grep -v -i "egl\|warn" | tail -20 || true
cat logs/mpi_results.jsonl | python -c "
import sys, json
rows = [json.loads(l) for l in sys.stdin if l.strip()]
print(len(rows), 'mpi runs;', sum(r.get('ok') is True for r in rows), 'ok')
for r in rows:
    if r.get('ok') is not True: print('NOT OK', r)
"
cat logs/conus.jsonl logs/pv_mpi.txt logs/vtkhdf.txt || true
ls -la img img_pyvista
