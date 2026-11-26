# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# The ground-truth SOMA files somaview's tests read, written by SOMA itself, built against an
# installed SOMA (as examples/app is):
#
#   cmake -S tools/somaview/tests/gen -B gen-build -DCMAKE_PREFIX_PATH=<SOMA prefix>[;<Kokkos prefix>]
#   cmake --build gen-build
#
# make_test_data writes the single-part files; make_mpi_test_data (when SOMA was built with MPI)
# writes multi-part files, one part per rank. tools/somaview/tests/make_data.sh runs both.

cmake_minimum_required(VERSION 3.20...3.24)
project(somaview_test_data LANGUAGES CXX)

find_package(soma REQUIRED)

add_executable(make_test_data make_test_data.cpp)
target_link_libraries(make_test_data PRIVATE soma::soma)

if(SOMA_ENABLE_MPI)
  add_executable(make_mpi_test_data make_mpi_test_data.cpp)
  target_link_libraries(make_mpi_test_data PRIVATE soma::soma)
endif()
