# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# The barotropic ocean: a library of the model's parts (swe_ocean_core), its
# driver (swe_ocean) and, with tests on, its tests.
add_library(swe_ocean_core STATIC vertical.cpp grid_file.cpp bathymetry.cpp coastline.cpp ocean_mesh.cpp)
add_executable(swe_ocean main.cpp)
target_link_libraries(swe_ocean PRIVATE swe_ocean_core soma_warnings)
if(SOMA_ENABLE_HDF5)
  target_link_libraries(swe_ocean_core PRIVATE HDF5::HDF5)
  target_compile_definitions(swe_ocean_core PUBLIC SWE_OCEAN_HAVE_HDF5)
endif()
target_include_directories(swe_ocean_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(swe_ocean_core PUBLIC soma PRIVATE soma_warnings)

if(SOMA_ENABLE_TESTS)
  # soma_add_test comes from tests/; an installed GoogleTest's targets are
  # visible only in the directory that found it.
  include(GoogleTest)
  if(NOT TARGET GTest::gtest)
    find_package(GTest REQUIRED)
  endif()
  # A test per rank count, one host thread per rank (as swe_add_test).
  function(ocean_add_test name)
    cmake_parse_arguments(T "" "" "SOURCES;RANKS" ${ARGN})
    soma_add_test(${name} SOURCES ${T_SOURCES} LIBS swe_ocean_core RANKS ${T_RANKS})
    # The library's test meshes (sem_meshes.hpp); the grid-file writer of tests/ needs HDF5.
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/tests/unit ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    if(SOMA_ENABLE_HDF5)
      target_link_libraries(${name} PRIVATE HDF5::HDF5)
    endif()
    if(SOMA_ENABLE_MPI AND T_RANKS)
      foreach(np IN LISTS T_RANKS)
        set_tests_properties(${name}.np${np} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
      endforeach()
    endif()
  endfunction()
  ocean_add_test(test_ocean_vertical SOURCES tests/test_ocean_vertical.cpp)
  ocean_add_test(test_ocean_grid_file SOURCES tests/test_ocean_grid_file.cpp RANKS 1 2 4)
  ocean_add_test(test_ocean_bathymetry SOURCES tests/test_ocean_bathymetry.cpp RANKS 1 2 3 4)
  ocean_add_test(test_ocean_coastline SOURCES tests/test_ocean_coastline.cpp RANKS 1 2 4)
  ocean_add_test(test_ocean_mesh SOURCES tests/test_ocean_mesh.cpp RANKS 1 2 3 4)
endif()
