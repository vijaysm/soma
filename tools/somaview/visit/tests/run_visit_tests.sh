#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Builds the SOMA database plugin against a VisIt binary install and checks it in VisIt itself:
#
#   VISIT=<VisIt install, holding bin/visit> WORK=<scratch dir> [DATA=<tests/data>] \
#   [LARGE=<SOMA file>:<cell field>] tools/somaview/visit/tests/run_visit_tests.sh
#
# The plugin is built with VisIt's own HDF5 and the system compiler VisIt was built with
# (CXX, default /usr/bin/g++), and installed into ~/.visit. The expected zone counts and field
# ranges come from somaview's Python core (visit_expected.py, run by `python`, which needs
# somaview's dependencies). Then visit_test.py runs every file on a serial engine and on a
# 4-rank parallel engine, and LARGE on an 8-rank engine; results are VISITRESULT lines in
# $WORK/logs, renders in $WORK/img.
set -x
set +e +o pipefail
: "${VISIT:?set VISIT}" "${WORK:?set WORK}"
HERE=$(cd "$(dirname "$0")" && pwd)
SV=$(cd "$HERE/../.." && pwd)
D=${DATA:-$SV/tests/data}
M=$D/mpi
VB=$VISIT/bin
export PYTHONPATH=$SV/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$WORK/img" "$WORK/logs"

# -- the plugin --------------------------------------------------------------------------------
P=$WORK/plugin
rm -rf "$P" && mkdir -p "$P"
cp "$SV"/visit/avtSOMAFileFormat.C "$SV"/visit/avtSOMAFileFormat.h "$SV"/visit/soma_reader.hpp "$SV"/visit/SOMA.xml \
  "$SV"/visit/visit_macos.cmake "$P"/
cd "$P"
env -u CC -u CXX "$VB/xml2info" -clobber SOMA.xml > "$WORK/logs/xml2info.txt" 2>&1
env -u CC -u CXX "$VB/xml2cmake" -clobber SOMA.xml > "$WORK/logs/xml2cmake.txt" 2>&1
env -u CC -u CXX cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${CXX:-/usr/bin/g++}" \
  -DCMAKE_PROJECT_INCLUDE="$P/visit_macos.cmake" . > "$WORK/logs/cmake.txt" 2>&1
env -u CC -u CXX make -j8 > "$WORK/logs/make.txt" 2>&1
grep -E "error|Error" "$WORK/logs/make.txt" | head -30
cd "$WORK"

# -- expectations from the Python core ---------------------------------------------------------
python "$HERE/visit_expected.py" expected.json \
  $D/hex_fields.soma:pressure $D/mixed3d.soma:vx $D/highorder3d.soma:radius $D/highorder2d.soma:x \
  $D/polygons.soma:sides $D/64bricks_1khex.soma:set:MATERIAL_SET $D/recMeshOcn128.soma:GLOBAL_ID \
  $D/amr2d.soma:phi > logs/expected.txt 2>&1
if [ -d "$M" ]; then
  python "$HERE/visit_expected.py" expected_mpi.json \
    $M/hex_4p.soma:pressure $M/hex_7p.soma:pressure $M/quad_7p.soma:pressure $M/tet_4p.soma:pressure \
    $M/tet_7p.soma:pressure $M/bricks_4p.soma:set:MATERIAL_SET $M/ocn_3p.soma:GLOBAL_ID \
    $M/amr3d_4p.soma:phi >> logs/expected.txt 2>&1
  python - <<PYEOF
import json
a = json.load(open("expected.json")); a.update(json.load(open("expected_mpi.json")))
json.dump(a, open("expected.json", "w"), indent=1)
PYEOF
fi
[ -n "${LARGE:-}" ] && python "$HERE/visit_expected.py" expected_large.json "$LARGE" >> logs/expected.txt 2>&1

# -- VisIt: serial engine, a 4-rank engine, and the large file on 8 ranks ----------------------
for np in 1 4; do
  VISIT_TEST_EXPECTED=$WORK/expected.json VISIT_TEST_OUT=$WORK/img VISIT_TEST_NP=$np \
    "$VB/visit" -nowin -cli -s "$HERE/visit_test.py" > logs/visit_np$np.txt 2>&1
done
[ -n "${LARGE:-}" ] && VISIT_TEST_EXPECTED=$WORK/expected_large.json VISIT_TEST_OUT=$WORK/img VISIT_TEST_NP=8 \
  "$VB/visit" -nowin -cli -s "$HERE/visit_test.py" > logs/visit_large.txt 2>&1
grep -h VISITRESULT logs/visit_*.txt | cut -c1-300
