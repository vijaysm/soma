# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Installs the build tree into an empty scratch prefix, then configures, builds
# and runs examples/app, an application that reaches SOMA only through
# find_package. Arguments (-D):
#   SOMA_BINARY_DIR SOMA_SOURCE_DIR SCRATCH_DIR BUILD_TYPE GENERATOR
#   MAKE_PROGRAM CXX_COMPILER MPI_HOME
#   KOKKOS_SOURCE installed | fetched (a fetched Kokkos is installed with SOMA)
#   KOKKOS_DIR    passed through for an installed Kokkos
#   PREFIX_PATH  the build's CMAKE_PREFIX_PATH, with '|' for ';'
#   OUTPUT       the file the app writes (its extension picks the format)
#   ENABLE_MPI MPIEXEC

foreach(var SOMA_BINARY_DIR SOMA_SOURCE_DIR SCRATCH_DIR GENERATOR OUTPUT)
  if(NOT ${var})
    message(FATAL_ERROR "example app test: ${var} is required")
  endif()
endforeach()

string(REPLACE "|" ";" search_path "${PREFIX_PATH}")
set(prefix "${SCRATCH_DIR}/prefix")
set(app_build "${SCRATCH_DIR}/app")
file(REMOVE_RECURSE "${SCRATCH_DIR}")
file(MAKE_DIRECTORY "${SCRATCH_DIR}")

function(run_step name)
  execute_process(COMMAND ${ARGN} WORKING_DIRECTORY "${SCRATCH_DIR}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "example app test: ${name} failed (${rc})\n${out}")
  endif()
  set(step_output "${out}" PARENT_SCOPE)
endfunction()

set(install_cmd ${CMAKE_COMMAND} --install "${SOMA_BINARY_DIR}" --prefix "${prefix}")
if(BUILD_TYPE)
  list(APPEND install_cmd --config "${BUILD_TYPE}")
endif()
run_step("install" ${install_cmd})

# An initial-cache file keeps the search path's semicolons intact.
set(init_cache "${SCRATCH_DIR}/app.cache.cmake")
file(WRITE "${init_cache}" "set(CMAKE_PREFIX_PATH \"${prefix};${search_path}\" CACHE STRING \"\")\n")
set(cmd ${CMAKE_COMMAND} -S "${SOMA_SOURCE_DIR}/examples/app" -B "${app_build}" -G "${GENERATOR}" -C "${init_cache}")
foreach(pair "CMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}" "CMAKE_CXX_COMPILER=${CXX_COMPILER}"
             "CMAKE_BUILD_TYPE=${BUILD_TYPE}" "MPI_HOME=${MPI_HOME}")
  if(NOT pair MATCHES "=$")
    list(APPEND cmd "-D${pair}")
  endif()
endforeach()
if(KOKKOS_DIR AND NOT KOKKOS_SOURCE STREQUAL "fetched")
  list(APPEND cmd "-DKokkos_DIR=${KOKKOS_DIR}")
endif()
run_step("configure" ${cmd})
set(build_cmd ${CMAKE_COMMAND} --build "${app_build}")
if(BUILD_TYPE)
  list(APPEND build_cmd --config "${BUILD_TYPE}")
endif()
run_step("build" ${build_cmd})

set(app "${app_build}/soma_app")
if(BUILD_TYPE AND EXISTS "${app_build}/${BUILD_TYPE}/soma_app")
  set(app "${app_build}/${BUILD_TYPE}/soma_app")
endif()
run_step("run" "${app}" --n 6 "${SCRATCH_DIR}/one${OUTPUT}")
if(NOT step_output MATCHES "OK soma app: 1 rank\\(s\\), cells 216 -> 1728")
  message(FATAL_ERROR "example app test: unexpected output: ${step_output}")
endif()
message(STATUS "${step_output}")
if(ENABLE_MPI)
  run_step("run (2 ranks)" "${MPIEXEC}" -n 2 "${app}" --n 6 "${SCRATCH_DIR}/two${OUTPUT}")
  if(NOT step_output MATCHES "OK soma app: 2 rank\\(s\\), cells 216 -> 1728")
    message(FATAL_ERROR "example app test: unexpected output at 2 ranks: ${step_output}")
  endif()
  message(STATUS "${step_output}")
endif()
