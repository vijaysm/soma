# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# The shallow-water atmosphere: a library of the model's parts (swe_atm_core),
# its driver (swe_atm) and, with tests on, its tests.
add_library(swe_atm_core STATIC cases.cpp)
target_include_directories(swe_atm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(swe_atm_core PUBLIC soma PRIVATE soma_warnings)

if(SOMA_ENABLE_TESTS)
  # soma_add_test comes from tests/; an installed GoogleTest's targets are
  # visible only in the directory that found it.
  include(GoogleTest)
  if(NOT TARGET GTest::gtest)
    find_package(GTest REQUIRED)
  endif()
  soma_add_test(test_swe_cases SOURCES tests/test_swe_cases.cpp LIBS swe_atm_core)
endif()
