# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# How the tests are launched on the node at hand.
#
#   soma_test_gpus(<out>)          the GPUs the tests may use (SOMA_TEST_GPUS)
#   soma_mpi_test(<test> <np>)     properties of a test that runs <np> ranks
#
# The unit tests run every rank count they name on whatever the node has:
# their meshes are small, so the ranks of a GPU build share its GPUs (Kokkos
# gives rank i of a node device i modulo the devices: all of them the one
# GPU of a single-GPU node), and a count above the node's cores is
# oversubscribed. The examples are the other policy, one rank for each GPU
# (examples/CMakeLists.txt).

include_guard(GLOBAL)

# The cores of this node: ranks beyond them are oversubscribed.
cmake_host_system_information(RESULT SOMA_TEST_CORES QUERY NUMBER_OF_PHYSICAL_CORES)
if(NOT SOMA_TEST_CORES GREATER 0)
  set(SOMA_TEST_CORES 1)
endif()

# What a run of more ranks than cores is given: leave to oversubscribe (the
# variable of Open MPI 5 and that of Open MPI 4; other launchers ignore
# both), and one unbound host thread for each rank, so that the node carries
# as many threads as ranks.
set(SOMA_TEST_OVERSUBSCRIBED_ENVIRONMENT
  "PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe"
  "OMPI_MCA_rmaps_base_oversubscribe=1"
  "OMP_NUM_THREADS=1"
  "OMP_PROC_BIND=false")

# PROCESSORS for ctest's own scheduling and, above the node's cores, the
# environment above.
function(soma_mpi_test test np)
  set_property(TEST ${test} PROPERTY PROCESSORS ${np})
  if(np GREATER SOMA_TEST_CORES)
    set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT ${SOMA_TEST_OVERSUBSCRIBED_ENVIRONMENT})
  endif()
endfunction()

# SOMA_TEST_GPUS: a count, or AUTO for the GPUs the vendor's tool lists on
# this node (one, said aloud, when it cannot tell).
function(soma_test_gpus out)
  set(_n "${SOMA_TEST_GPUS}")
  if(_n MATCHES "^[1-9][0-9]*$")
    set(${out} ${_n} PARENT_SCOPE)
    return()
  endif()
  string(TOUPPER "${_n}" _mode)
  if(NOT _mode STREQUAL "AUTO")
    message(FATAL_ERROR "SOMA_TEST_GPUS must be AUTO or a count of GPUs, 1 or more (got '${SOMA_TEST_GPUS}')")
  endif()
  set(_n 0)
  if(SOMA_KOKKOS_GPU_BACKEND STREQUAL "CUDA")
    find_program(SOMA_NVIDIA_SMI nvidia-smi)
    mark_as_advanced(SOMA_NVIDIA_SMI)
    if(SOMA_NVIDIA_SMI)
      execute_process(COMMAND ${SOMA_NVIDIA_SMI} --query-gpu=index --format=csv,noheader
        OUTPUT_VARIABLE _listed RESULT_VARIABLE _status ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(_status EQUAL 0 AND NOT _listed STREQUAL "")
        string(REGEX MATCHALL "[^\n]+" _lines "${_listed}")
        list(LENGTH _lines _n)
      endif()
    endif()
  endif()
  if(_n EQUAL 0)
    message(STATUS "SOMA: cannot count the ${SOMA_KOKKOS_GPU_BACKEND} GPUs of this node; the tests "
      "take it to have one (set SOMA_TEST_GPUS to the number there are)")
    set(_n 1)
  endif()
  set(${out} ${_n} PARENT_SCOPE)
endfunction()
