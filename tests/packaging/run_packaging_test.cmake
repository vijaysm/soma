# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Installs the build tree into an empty scratch prefix, then configures, builds
# and runs an out-of-tree consumer that can reach SOMA only through
# find_package. Arguments (-D):
#   SOMA_BINARY_DIR SOMA_SOURCE_DIR SCRATCH_DIR BUILD_TYPE GENERATOR
#   MAKE_PROGRAM CXX_COMPILER
#   PREFIX_PATH       the build's CMAKE_PREFIX_PATH, with '|' for ';' so it
#                     survives CTest's argument handling
#   KOKKOS_SOURCE     installed | fetched
#   KOKKOS_DIR        passed through for an installed Kokkos
#   ENABLE_MPI MPI_HOME MPIEXEC

foreach(var SOMA_BINARY_DIR SOMA_SOURCE_DIR SCRATCH_DIR GENERATOR KOKKOS_SOURCE)
  if(NOT ${var})
    message(FATAL_ERROR "packaging test: ${var} is required")
  endif()
endforeach()

string(REPLACE "|" ";" consumer_prefix_path "${PREFIX_PATH}")
set(prefix "${SCRATCH_DIR}/prefix")
set(consumer_build "${SCRATCH_DIR}/consumer")
file(REMOVE_RECURSE "${SCRATCH_DIR}")
file(MAKE_DIRECTORY "${SCRATCH_DIR}")

function(run_step name)
  execute_process(COMMAND ${ARGN}
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "packaging test: ${name} failed (${rc})\n${out}")
  endif()
  set(step_output "${out}" PARENT_SCOPE)
endfunction()

set(install_cmd ${CMAKE_COMMAND} --install "${SOMA_BINARY_DIR}" --prefix "${prefix}")
if(BUILD_TYPE)
  list(APPEND install_cmd --config "${BUILD_TYPE}")
endif()
run_step("install" ${install_cmd})

set(expected
  include/soma/config.hpp
  lib/cmake/soma/somaConfig.cmake
  lib/cmake/soma/somaConfigVersion.cmake
  lib/cmake/soma/somaTargets.cmake)
file(GLOB_RECURSE headers RELATIVE "${SOMA_SOURCE_DIR}/include"
  "${SOMA_SOURCE_DIR}/include/soma/*.hpp")
foreach(h IN LISTS headers)
  list(APPEND expected "include/${h}")
endforeach()
if(KOKKOS_SOURCE STREQUAL "fetched")
  list(APPEND expected lib/cmake/Kokkos/KokkosConfig.cmake
    lib/cmake/Kokkos/KokkosTargets.cmake include/kokkos/Kokkos_Core.hpp)
endif()
foreach(f IN LISTS expected)
  if(NOT EXISTS "${prefix}/${f}")
    message(FATAL_ERROR "packaging test: ${f} was not installed")
  endif()
endforeach()
file(GLOB libs "${prefix}/lib/libsoma.*")
if(NOT libs)
  message(FATAL_ERROR "packaging test: libsoma was not installed")
endif()

set(configure_cmd ${CMAKE_COMMAND}
  -S "${SOMA_SOURCE_DIR}/tests/packaging" -B "${consumer_build}" -G "${GENERATOR}"
  "-DCMAKE_PREFIX_PATH=${prefix};${consumer_prefix_path}")
if(MAKE_PROGRAM)
  list(APPEND configure_cmd "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
if(CXX_COMPILER)
  list(APPEND configure_cmd "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
endif()
if(BUILD_TYPE)
  list(APPEND configure_cmd "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
endif()
if(KOKKOS_DIR AND NOT KOKKOS_SOURCE STREQUAL "fetched")
  list(APPEND configure_cmd "-DKokkos_DIR=${KOKKOS_DIR}")
endif()
if(MPI_HOME)
  list(APPEND configure_cmd "-DMPI_HOME=${MPI_HOME}")
endif()
run_step("consumer configure" ${configure_cmd})

# The imported Kokkos must come from the install prefix, not from a Kokkos
# that happens to be reachable.
if(KOKKOS_SOURCE STREQUAL "fetched")
  file(STRINGS "${consumer_build}/CMakeCache.txt" kokkos_dir REGEX "^Kokkos_DIR:")
  string(FIND "${kokkos_dir}" "=${prefix}/" in_prefix)
  if(in_prefix EQUAL -1)
    message(FATAL_ERROR "packaging test: Kokkos was not found in the install prefix (${kokkos_dir})")
  endif()
endif()

set(build_cmd ${CMAKE_COMMAND} --build "${consumer_build}")
if(BUILD_TYPE)
  list(APPEND build_cmd --config "${BUILD_TYPE}")
endif()
run_step("consumer build" ${build_cmd})

if(BUILD_TYPE)
  set(consumer "${consumer_build}/${BUILD_TYPE}/consumer")
endif()
if(NOT BUILD_TYPE OR NOT EXISTS "${consumer}")
  set(consumer "${consumer_build}/consumer")
endif()
run_step("consumer run" "${consumer}")
if(NOT step_output MATCHES "OK soma consumer: hex has 6 faces, 1 rank\\(s\\)")
  message(FATAL_ERROR "packaging test: unexpected consumer output: ${step_output}")
endif()
message(STATUS "${step_output}")

if(ENABLE_MPI AND MPIEXEC)
  run_step("consumer run (2 ranks)" "${MPIEXEC}" -n 2 "${consumer}")
  if(NOT step_output MATCHES "2 rank\\(s\\)")
    message(FATAL_ERROR "packaging test: expected 2 ranks: ${step_output}")
  endif()
  message(STATUS "2-rank run: OK")
endif()
