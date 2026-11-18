# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# The shallow-water atmosphere: a library of the model's parts (swe_atm_core),
# its driver (swe_atm) and, with tests on, its tests.
add_library(swe_atm_core STATIC cases.cpp diagnostics.cpp model.cpp)
target_include_directories(swe_atm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(swe_atm_core PUBLIC soma PRIVATE soma_warnings)

if(SOMA_ENABLE_TESTS)
  # soma_add_test comes from tests/; an installed GoogleTest's targets are
  # visible only in the directory that found it.
  include(GoogleTest)
  if(NOT TARGET GTest::gtest)
    find_package(GTest REQUIRED)
  endif()
  # A test per rank count, one host thread per rank: the model's kernels are
  # small at test sizes, and a full set of threads on every rank oversubscribes
  # the cores (the .threads tests compare thread counts).
  function(swe_add_test name)
    cmake_parse_arguments(T "" "" "SOURCES;RANKS" ${ARGN})
    soma_add_test(${name} SOURCES ${T_SOURCES} LIBS swe_atm_core RANKS ${T_RANKS})
    # The library's test meshes (sem_meshes.hpp).
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit)
    if(SOMA_ENABLE_MPI AND T_RANKS)
      foreach(np IN LISTS T_RANKS)
        set_tests_properties(${name}.np${np} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
      endforeach()
    endif()
  endfunction()
  swe_add_test(test_swe_cases SOURCES tests/test_swe_cases.cpp)
  swe_add_test(test_swe_diagnostics SOURCES tests/test_swe_diagnostics.cpp RANKS 1 2 3 4)
  swe_add_test(test_swe_model SOURCES tests/test_swe_model.cpp RANKS 1 2 3 4 8)
  # A run is the same bit for bit at any number of threads.
  add_test(NAME test_swe_model.threads COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_swe_model>
    -DFILTER=SweModel.Digests -P ${PROJECT_SOURCE_DIR}/tests/unit/compare_threads.cmake)
  set_tests_properties(test_swe_model.threads PROPERTIES TIMEOUT 300)
endif()
