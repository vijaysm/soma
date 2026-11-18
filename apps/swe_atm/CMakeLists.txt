# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# The shallow-water atmosphere: a library of the model's parts (swe_atm_core),
# its driver (swe_atm) and, with tests on, its tests.
add_library(swe_atm_core STATIC cases.cpp diagnostics.cpp latlon.cpp model.cpp output.cpp reference.cpp)
add_executable(swe_atm main.cpp)
target_link_libraries(swe_atm PRIVATE swe_atm_core soma_warnings)
if(SOMA_ENABLE_HDF5)
  target_link_libraries(swe_atm_core PRIVATE HDF5::HDF5)
  target_compile_definitions(swe_atm_core PUBLIC SWE_HAVE_HDF5)
endif()
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
  swe_add_test(test_swe_latlon SOURCES tests/test_swe_latlon.cpp RANKS 1 2 3 4 8)
  swe_add_test(test_swe_output SOURCES tests/test_swe_output.cpp RANKS 1 2)
  if(SOMA_ENABLE_HDF5)
    swe_add_test(test_swe_reference SOURCES tests/test_swe_reference.cpp RANKS 1 2)
    target_link_libraries(test_swe_reference PRIVATE HDF5::HDF5)
  endif()
  # The driver: case 2's state the same at 1, 2 and 4 ranks, and a bad option
  # refused with the case names.
  if(SOMA_ENABLE_MPI)
    add_test(NAME swe_atm.case2.digests COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:swe_atm>
      -DMPIEXEC=${MPIEXEC_EXECUTABLE} "-DRANKS=1;2;4" "-DARGS=--case w2 --ne 2 --order 4 --days 0.25 --nu 0"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_digests.cmake)
    set_tests_properties(swe_atm.case2.digests PROPERTIES TIMEOUT 120 PROCESSORS 4)
  endif()
  add_test(NAME swe_atm.bad_option COMMAND swe_atm --case w7)
  set_tests_properties(swe_atm.bad_option PROPERTIES PASS_REGULAR_EXPRESSION "w2, w5, w6")
  # A run is the same bit for bit at any number of threads.
  add_test(NAME test_swe_model.threads COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_swe_model>
    -DFILTER=SweModel.Digests -P ${PROJECT_SOURCE_DIR}/tests/unit/compare_threads.cmake)
  set_tests_properties(test_swe_model.threads PROPERTIES TIMEOUT 300)
endif()
