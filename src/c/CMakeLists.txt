# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# The C API: opaque handles over the C++ library, status codes, 0-based ids.

if(SOMA_ENABLE_MPI)
  set(SOMA_C_HAVE_MPI 1)
else()
  set(SOMA_C_HAVE_MPI 0)
endif()
if(SOMA_LO_64)
  set(SOMA_C_LO_BYTES 8)
else()
  set(SOMA_C_LO_BYTES 4)
endif()
if(SOMA_OFFSET_64)
  set(SOMA_C_OFFSET_BYTES 8)
else()
  set(SOMA_C_OFFSET_BYTES 4)
endif()
configure_file("${PROJECT_SOURCE_DIR}/include/soma/c/soma_config.h.in"
  "${PROJECT_BINARY_DIR}/generated/soma/c/soma_config.h" @ONLY)

add_library(soma_c runtime.cpp mesh.cpp edit.cpp fields.cpp io.cpp parallel.cpp algorithms.cpp)
add_library(soma::soma_c ALIAS soma_c)
# soma links MPI::MPI_CXX publicly, which carries mpi.h for the C header.
target_link_libraries(soma_c PUBLIC soma PRIVATE $<BUILD_INTERFACE:soma_warnings>)
