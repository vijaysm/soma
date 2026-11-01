# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Benchmarks against MOAB, the only baseline. Built with SOMA_ENABLE_MOAB_REFERENCE.
add_executable(bench_adjacency bench_adjacency.cpp)
target_include_directories(bench_adjacency PRIVATE ${PROJECT_SOURCE_DIR}/tests/reference)
target_link_libraries(bench_adjacency PRIVATE soma soma_warnings ${SOMA_MOAB_TARGET})
target_compile_definitions(bench_adjacency PRIVATE
  SOMA_BENCH_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
  SOMA_BENCH_LARGE_DATA_DIR="${SOMA_LARGE_DATA_DIR}"
  SOMA_BENCH_BUILD_TYPE="$<IF:$<CONFIG:>,unset,$<CONFIG>>"
  SOMA_BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
