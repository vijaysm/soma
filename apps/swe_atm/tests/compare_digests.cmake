# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Runs swe_atm (EXE) with ARGS at each rank count of RANKS through MPIEXEC and
# compares the "digest" lines it prints last: they must agree. VARIANTS, when
# given, is a "|"-separated list of further arguments: each is run in turn at
# every rank count, and all the digests must agree.
separate_arguments(args UNIX_COMMAND "${ARGS}")
if(NOT DEFINED VARIANTS)
  set(VARIANTS "")
endif()
string(REPLACE "|" ";" variants "${VARIANTS}")
if(variants STREQUAL "")
  set(variants " ")
endif()
set(digests "")
foreach(variant IN LISTS variants)
  foreach(np IN LISTS RANKS)
    separate_arguments(extra UNIX_COMMAND "${variant}")
    set(ENV{OMP_NUM_THREADS} 1)
    # Bounded: a hung run is killed here, not left behind when ctest stops cmake.
    execute_process(COMMAND ${MPIEXEC} -n ${np} ${EXE} ${args} ${extra} OUTPUT_VARIABLE out ERROR_VARIABLE err
                    RESULT_VARIABLE rc TIMEOUT 100)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "swe_atm ${variant} at ${np} ranks failed (${rc}):\n${out}\n${err}")
    endif()
    string(REGEX MATCH "digest ([0-9a-f]+)" _ "${out}")
    if(CMAKE_MATCH_1 STREQUAL "")
      message(FATAL_ERROR "swe_atm ${variant} at ${np} ranks printed no digest:\n${out}")
    endif()
    list(APPEND digests "${CMAKE_MATCH_1}")
    list(APPEND runs "${variant} at ${np}: ${CMAKE_MATCH_1}")
  endforeach()
endforeach()
list(REMOVE_DUPLICATES digests)
list(LENGTH digests distinct)
if(NOT distinct EQUAL 1)
  string(REPLACE ";" "\n  " listing "${runs}")
  message(FATAL_ERROR "digests differ across runs:\n  ${listing}")
endif()
message(STATUS "digest ${digests} at ${RANKS} ranks, variants: ${VARIANTS}")
