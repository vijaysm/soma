#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
#
# Runs bench_parallel (distributed meshes against MOAB's ParallelComm) over
# rank counts and box sizes, one JSON line per timing, into OUT/<P>_<n>_t<T>.json.
#
#   bench/run_parallel.sh BUILD OUT "1 2 4 8" "32 64" [repeats] [mpiexec]
#
# Each case runs at one thread per rank (the data-structure comparison) and
# at the threads a rank would get from the cores (SOMA_BENCH_THREADS, default
# 1). A run whose results disagree with MOAB's exits 2 and is reported.
#
# At scale (a cluster): build with -DSOMA_ENABLE_MPI=ON -DSOMA_ENABLE_HDF5=ON
# (a parallel HDF5) -DSOMA_ENABLE_MOAB_REFERENCE=ON and Release; run from a
# job script with MPIEXEC set to the launcher (srun, or mpiexec with its
# binding options) and OUT on the parallel file system, since the box and the
# written files go there; record the file system's striping beside the
# results. The box takes O(n^3) memory on rank 0 while it is written.
set -u
BUILD=$1
OUT=$2
RANKS=$3
SIZES=$4
REPEATS=${5:-5}
MPIEXEC=${6:-${MPIEXEC:-mpiexec}}
THREADS=${SOMA_BENCH_THREADS:-1}
mkdir -p "$OUT"
status=0
for n in $SIZES; do
  for p in $RANKS; do
    for t in $(printf '%s\n' 1 "$THREADS" | sort -u); do
      f="$OUT/${p}_${n}_t${t}.json"
      OMP_NUM_THREADS=$t OMP_PROC_BIND=false $MPIEXEC -n "$p" "$BUILD/bench/bench_parallel" --json --repeats "$REPEATS" \
        --n "$n" --dir "$OUT" > "$f" 2> "$f.err"
      rc=$?
      echo "P=$p n=$n threads=$t rc=$rc lines=$(grep -c '"benchmark"' "$f")"
      [ $rc -eq 0 ] || status=$rc
    done
  done
done
exit $status
