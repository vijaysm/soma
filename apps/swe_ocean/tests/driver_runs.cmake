# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Runs swe_ocean (EXE) through MPIEXEC once per entry of RUNS, a "|"-separated list of "NP ARGS..." (the rank count
# first), in order, and compares the "digest" lines the runs print last: they must agree. PREPARE, when given, is a
# command run first (a grid file written for the runs). Each run is bounded, so a hung run is killed here, not
# left behind when ctest stops cmake.
if(DEFINED PREPARE AND NOT PREPARE STREQUAL "")
  separate_arguments(prepare UNIX_COMMAND "${PREPARE}")
  execute_process(COMMAND ${prepare} OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc TIMEOUT 60)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${PREPARE} failed (${rc}):\n${out}\n${err}")
  endif()
endif()
string(REPLACE "|" ";" runs "${RUNS}")
set(digests "")
set(listing "")
foreach(run IN LISTS runs)
  separate_arguments(words UNIX_COMMAND "${run}")
  list(POP_FRONT words np)
  set(ENV{OMP_NUM_THREADS} 1)
  execute_process(COMMAND ${MPIEXEC} -n ${np} ${EXE} ${words} OUTPUT_VARIABLE out ERROR_VARIABLE err
                  RESULT_VARIABLE rc TIMEOUT 100)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "swe_ocean at ${np} ranks (${words}) failed (${rc}):\n${out}\n${err}")
  endif()
  string(REGEX MATCH "digest ([0-9a-f]+)" _ "${out}")
  if(CMAKE_MATCH_1 STREQUAL "")
    message(FATAL_ERROR "swe_ocean at ${np} ranks (${words}) printed no digest:\n${out}")
  endif()
  list(APPEND digests "${CMAKE_MATCH_1}")
  list(APPEND listing "${run}: ${CMAKE_MATCH_1}")
endforeach()
list(REMOVE_DUPLICATES digests)
list(LENGTH digests distinct)
if(NOT distinct EQUAL 1)
  string(REPLACE ";" "\n  " text "${listing}")
  message(FATAL_ERROR "digests differ across runs:\n  ${text}")
endif()
message(STATUS "digest ${digests} in every run")
