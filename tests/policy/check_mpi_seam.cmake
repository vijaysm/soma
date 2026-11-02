# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Fails if SOMA_ENABLE_MPI is tested anywhere outside the Comm seam. Keeps
# MPI-only code paths from hiding where a shared-memory build never compiles them.
# Run with cmake -P: the policy level is needed for IN_LIST (CMP0057).
cmake_minimum_required(VERSION 3.24)
set(allowed
  include/soma/parallel/comm.hpp
  src/parallel/comm_mpi.cpp
  src/parallel/comm_serial.cpp
  src/runtime/scope_guard.cpp)
file(GLOB_RECURSE files RELATIVE "${SOURCE_DIR}"
  "${SOURCE_DIR}/include/*.hpp" "${SOURCE_DIR}/src/*.hpp" "${SOURCE_DIR}/src/*.cpp"
  "${SOURCE_DIR}/bench/*.hpp" "${SOURCE_DIR}/bench/*.cpp")
list(LENGTH files scanned)
if(scanned EQUAL 0)
  message(FATAL_ERROR "MPI seam policy: no sources found under ${SOURCE_DIR}")
endif()
set(violations "")
foreach(f IN LISTS files)
  if(NOT f IN_LIST allowed)
    file(STRINGS "${SOURCE_DIR}/${f}" hits REGEX "SOMA_ENABLE_MPI")
    if(hits)
      list(APPEND violations "${f}")
      message(STATUS "SOMA_ENABLE_MPI outside the Comm seam: ${f}")
    endif()
  endif()
endforeach()
if(violations)
  message(FATAL_ERROR "SOMA_ENABLE_MPI used outside the Comm seam (${violations})")
endif()
message(STATUS "MPI seam policy: OK (${scanned} files scanned)")
