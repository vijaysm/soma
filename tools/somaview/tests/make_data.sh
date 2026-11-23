#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Writes the ground-truth SOMA files somaview's tests read into tests/data/ (or $1):
#
#   tools/somaview/tests/make_data.sh <SOMA prefix> [<data dir>] [<Kokkos prefix>]
#
# <SOMA prefix> is an install of SOMA built with HDF5 (cmake --install build --prefix P); its
# bin/soma-convert (or $SOMA_CONVERT) converts the bundled MOAB meshes. With SOMA built with MPI, set MPIEXEC
# (default mpiexec) and the multi-part files are written at 4 and 7 ranks into <data dir>/mpi.
set -euo pipefail
prefix=${1:?usage: make_data.sh <SOMA prefix> [<data dir>] [<Kokkos prefix>]}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
out=${2:-$here/data}
kokkos=${3:-}
build=$out/.gen-build
convert=${SOMA_CONVERT:-$prefix/bin/soma-convert}
mkdir -p "$out"
cmake -S "$here/gen" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$prefix${kokkos:+;$kokkos}" > "$build.log"
cmake --build "$build" >> "$build.log"
"$build/make_test_data" "$out"
for m in hexes_mixed 64bricks_1khex recMeshOcn128 hybrid_3quad_1tri; do
  "$convert" "$repo/data/$m.h5m" "$out/$m.soma"
done
if [ -x "$build/make_mpi_test_data" ]; then
  mkdir -p "$out/mpi"
  for p in 4 7; do "${MPIEXEC:-mpiexec}" -n $p "$build/make_mpi_test_data" "$out/mpi"; done
  "${MPIEXEC:-mpiexec}" -n 4 "$convert" "$repo/data/64bricks_1khex.h5m" "$out/mpi/bricks_4p.soma"
  "${MPIEXEC:-mpiexec}" -n 3 "$convert" "$repo/data/recMeshOcn128.h5m" "$out/mpi/ocn_3p.soma"
fi
ls "$out"
