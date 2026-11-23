#!/bin/sh
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
# Builds cpp_driver, the standalone correctness-checking tool (not the VisIt plugin itself --
# see ../README.md for that). Needs an HDF5 C install; defaults to /opt/hdf5-serial.
set -e
HDF5_ROOT="${HDF5_ROOT:-/opt/hdf5-serial}"
HERE="$(cd "$(dirname "$0")" && pwd)"
c++ -std=c++17 -O2 -Wall -Wextra -I"$HDF5_ROOT/include" "$HERE/cpp_driver.cpp" \
    -L"$HDF5_ROOT/lib" -lhdf5 -Wl,-rpath,"$HDF5_ROOT/lib" -o "$HERE/cpp_driver"
echo "built $HERE/cpp_driver"
