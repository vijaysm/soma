# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Configures SOMA with a parallel HDF5 and MPI off, and fails unless the
# configure step refuses it with the message that says what to do.
file(REMOVE_RECURSE "${SCRATCH_DIR}")
string(REPLACE "|" ";" prefix_path "${PREFIX_PATH}")
execute_process(
  COMMAND ${CMAKE_COMMAND} -S "${SOURCE_DIR}" -B "${SCRATCH_DIR}" -G "${GENERATOR}"
          "-DCMAKE_PREFIX_PATH=${prefix_path}" -DSOMA_ENABLE_HDF5=ON -DSOMA_ENABLE_MPI=OFF
          "-DHDF5_ROOT=${PARALLEL_HDF5_ROOT}" -DSOMA_INSTALL=OFF
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
file(REMOVE_RECURSE "${SCRATCH_DIR}")
if(rc EQUAL 0)
  message(FATAL_ERROR "HDF5 guard: a parallel HDF5 with SOMA_ENABLE_MPI=OFF was accepted")
endif()
if(NOT err MATCHES "found a parallel HDF5" OR NOT err MATCHES "SOMA_ENABLE_MPI=ON")
  message(FATAL_ERROR "HDF5 guard: configure failed for another reason:\n${err}")
endif()
message(STATUS "HDF5 guard: OK")
