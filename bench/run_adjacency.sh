#!/bin/sh
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Runs bench_adjacency at threads = 1, 2, 4, ..., up to the core count (or
# MAX_THREADS), one process per count: Kokkos fixes its thread count at
# initialization. JSON lines go to stdout.
#
#   bench/run_adjacency.sh <build-dir> [bench_adjacency arguments]

set -eu
build=${1:?usage: run_adjacency.sh <build-dir> [args]}
shift
max=${MAX_THREADS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}
t=1
while [ "$t" -le "$max" ]; do
  "$build/bench/bench_adjacency" --json "$@" --kokkos-num-threads="$t"
  if [ "$t" -lt "$max" ] && [ $((t * 2)) -gt "$max" ]; then
    t=$max
  else
    t=$((t * 2))
  fi
done
