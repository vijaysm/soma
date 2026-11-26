# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# An application built against an installed SOMA, outside SOMA's build:
#
#   cmake -S examples/app -B app-build -DCMAKE_PREFIX_PATH=<SOMA prefix>[;<Kokkos prefix>]
#   cmake --build app-build
#   mpiexec -n 4 app-build/soma_app out.soma
#
# find_package(soma) brings in Kokkos and, when SOMA was built with them, MPI
# and HDF5; nothing else is needed.

cmake_minimum_required(VERSION 3.20...3.24)
project(soma_app LANGUAGES CXX)

find_package(soma REQUIRED)
message(STATUS "soma_app: soma ${soma_VERSION} from ${soma_DIR} (MPI ${SOMA_ENABLE_MPI}, HDF5 ${SOMA_ENABLE_HDF5})")

add_executable(soma_app main.cpp)
target_link_libraries(soma_app PRIVATE soma::soma)
