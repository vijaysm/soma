# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Runs swe_atm (EXE) with ARGS at each rank count of RANKS through MPIEXEC and
# compares the "digest" lines it prints last: they must agree.
separate_arguments(args UNIX_COMMAND "${ARGS}")
set(digests "")
foreach(np IN LISTS RANKS)
  set(ENV{OMP_NUM_THREADS} 1)
  execute_process(COMMAND ${MPIEXEC} -n ${np} ${EXE} ${args} OUTPUT_VARIABLE out ERROR_VARIABLE err
                  RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "swe_atm at ${np} ranks failed (${rc}):\n${out}\n${err}")
  endif()
  string(REGEX MATCH "digest ([0-9a-f]+)" _ "${out}")
  if(CMAKE_MATCH_1 STREQUAL "")
    message(FATAL_ERROR "swe_atm at ${np} ranks printed no digest:\n${out}")
  endif()
  list(APPEND digests "${CMAKE_MATCH_1}")
endforeach()
list(REMOVE_DUPLICATES digests)
list(LENGTH digests distinct)
if(NOT distinct EQUAL 1)
  message(FATAL_ERROR "digests differ across rank counts (${RANKS}): ${digests}")
endif()
message(STATUS "digest ${digests} at ${RANKS} ranks")
