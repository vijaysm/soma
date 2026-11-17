# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Installs the build tree into an empty scratch prefix, then builds and runs a
# C and a Fortran consumer (tests/packaging/c_fortran) that reach SOMA only
# through find_package(soma COMPONENTS c fortran). Arguments (-D):
#   SOMA_BINARY_DIR SOMA_SOURCE_DIR SCRATCH_DIR BUILD_TYPE GENERATOR MAKE_PROGRAM
#   C_COMPILER CXX_COMPILER Fortran_COMPILER PREFIX_PATH ('|' for ';') MPI_HOME

set(prefix "${SCRATCH_DIR}/prefix")
set(build "${SCRATCH_DIR}/consumer")
file(REMOVE_RECURSE "${SCRATCH_DIR}")
file(MAKE_DIRECTORY "${SCRATCH_DIR}")

function(run_step name)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "c/fortran packaging test: ${name} failed (${rc})\n${out}")
  endif()
  set(step_output "${out}" PARENT_SCOPE)
endfunction()

run_step("install" ${CMAKE_COMMAND} --install "${SOMA_BINARY_DIR}" --prefix "${prefix}" --config "${BUILD_TYPE}")
foreach(f include/soma/c/soma.h include/soma/c/soma_config.h include/soma/fortran/soma.mod)
  if(NOT EXISTS "${prefix}/${f}")
    message(FATAL_ERROR "c/fortran packaging test: ${f} was not installed")
  endif()
endforeach()

string(REPLACE "|" ";" search "${PREFIX_PATH}")
set(init_cache "${build}.cache.cmake")
file(WRITE "${init_cache}" "set(CMAKE_PREFIX_PATH \"${prefix};${search}\" CACHE STRING \"\")\n")
set(cmd ${CMAKE_COMMAND} -S "${SOMA_SOURCE_DIR}/tests/packaging/c_fortran" -B "${build}" -G "${GENERATOR}"
  -C "${init_cache}" "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" "-DCMAKE_C_COMPILER=${C_COMPILER}"
  "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}" "-DCMAKE_Fortran_COMPILER=${Fortran_COMPILER}")
if(MAKE_PROGRAM)
  list(APPEND cmd "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
if(MPI_HOME)
  list(APPEND cmd "-DMPI_HOME=${MPI_HOME}")
endif()
run_step("consumer configure" ${cmd})
run_step("consumer build" ${CMAKE_COMMAND} --build "${build}")
run_step("c consumer" "${build}/consumer_c")
message(STATUS "${step_output}")
run_step("fortran consumer" "${build}/consumer_fortran")
message(STATUS "${step_output}")
