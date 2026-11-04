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

if(SOMA_ENABLE_HDF5)
  add_executable(bench_h5m_read bench_h5m_read.cpp)
  target_link_libraries(bench_h5m_read PRIVATE soma soma_warnings ${SOMA_MOAB_TARGET})
  target_compile_definitions(bench_h5m_read PRIVATE
    SOMA_BENCH_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
    SOMA_BENCH_LARGE_DATA_DIR="${SOMA_LARGE_DATA_DIR}"
    SOMA_BENCH_BUILD_TYPE="$<IF:$<CONFIG:>,unset,$<CONFIG>>"
    SOMA_BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  add_executable(bench_h5m_write bench_h5m_write.cpp)
  target_link_libraries(bench_h5m_write PRIVATE soma soma_warnings ${SOMA_MOAB_TARGET})
  target_compile_definitions(bench_h5m_write PRIVATE
    SOMA_BENCH_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
    SOMA_BENCH_LARGE_DATA_DIR="${SOMA_LARGE_DATA_DIR}"
    SOMA_BENCH_BUILD_TYPE="$<IF:$<CONFIG:>,unset,$<CONFIG>>"
    SOMA_BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  if(SOMA_ENABLE_MOAB_COMPAT)
    add_executable(bench_edit bench_edit.cpp)
    target_link_libraries(bench_edit PRIVATE soma soma_compat soma_warnings ${SOMA_MOAB_TARGET})
    target_compile_definitions(bench_edit PRIVATE
      SOMA_BENCH_BUILD_TYPE="$<IF:$<CONFIG:>,unset,$<CONFIG>>"
      SOMA_BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
endif()
