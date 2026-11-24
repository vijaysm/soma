# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# The barotropic ocean: a library of the model's parts (swe_ocean_core), its
# driver (swe_ocean) and, with tests on, its tests.
add_library(swe_ocean_core STATIC vertical.cpp grid_file.cpp bathymetry.cpp coastline.cpp ocean_mesh.cpp model.cpp
  tracers.cpp cases.cpp sample.cpp output.cpp)
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
  ocean_add_test(test_ocean_model SOURCES tests/test_ocean_model.cpp RANKS 1 2 3 4 8)
  ocean_add_test(test_ocean_tracers SOURCES tests/test_ocean_tracers.cpp RANKS 1 2 4)
  ocean_add_test(test_ocean_cases SOURCES tests/test_ocean_cases.cpp RANKS 1 4)
  ocean_add_test(test_ocean_sample SOURCES tests/test_ocean_sample.cpp RANKS 1 2 3 4 8)
  # A run is the same bit for bit at any number of threads: in a plane box always, on the synthetic Earth (a grid
  # file written with HDF5) when HDF5 is on.
  add_test(NAME test_ocean_model.plane_threads COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_ocean_model>
    -DFILTER=OceanModel.plane_box_thread_independent "-DTHREADS=1;2;4"
    -P ${PROJECT_SOURCE_DIR}/tests/unit/compare_threads.cmake)
  set_tests_properties(test_ocean_model.plane_threads PROPERTIES TIMEOUT 180)
  if(SOMA_ENABLE_HDF5)
    add_test(NAME test_ocean_model.threads COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_ocean_model>
      -DFILTER=OceanModelTest.rank_and_thread_independent "-DTHREADS=1;2;4"
      -P ${PROJECT_SOURCE_DIR}/tests/unit/compare_threads.cmake)
    set_tests_properties(test_ocean_model.threads PROPERTIES TIMEOUT 180)
  endif()
  # The driver: Kelvin's state the same at 1, 2 and 4 ranks; a mesh built at 2 ranks and saved, read back at 3,
  # gives the same run; a bad option is refused with the case names (exit 2).
  if(SOMA_ENABLE_MPI)
    set(kelvin "--case kelvin --ne 8 --order 4")
    add_test(NAME swe_ocean.kelvin.digests COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:swe_ocean>
      -DMPIEXEC=${MPIEXEC_EXECUTABLE} "-DRUNS=1 ${kelvin}|2 ${kelvin}|4 ${kelvin}"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/driver_runs.cmake)
    set_tests_properties(swe_ocean.kelvin.digests PROPERTIES TIMEOUT 120 PROCESSORS 4)
    if(SOMA_ENABLE_HDF5)
      add_executable(write_earth tests/write_earth.cpp)
      target_include_directories(write_earth PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
      target_link_libraries(write_earth PRIVATE swe_ocean_core HDF5::HDF5 soma_warnings)
      set(dir ${CMAKE_CURRENT_BINARY_DIR}/mesh_reuse)
      set(wind "--case wind --ne 4 --order 3 --coast-km 400 --days 0.02")
      add_test(NAME swe_ocean.mesh_reuse COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:swe_ocean>
        -DMPIEXEC=${MPIEXEC_EXECUTABLE} "-DPREPARE=$<TARGET_FILE:write_earth> ${dir}/earth.h5"
        "-DRUNS=2 ${wind} --bathymetry ${dir}/earth.h5 --save-mesh ${dir}/earth.soma|3 ${wind} --mesh ${dir}/earth.soma"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/driver_runs.cmake)
      set_tests_properties(swe_ocean.mesh_reuse PROPERTIES TIMEOUT 120 PROCESSORS 3 FIXTURES_SETUP ocean_earth)
      # The start-up report of a built mesh gives the coastline's size.
      add_test(NAME swe_ocean.built_report COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:swe_ocean>
        "-DARGS=--case wind --ne 4 --order 3 --coast-km 400 --days 0 --bathymetry ${dir}/earth.h5" -DCODE=0
        "-DMATCH=values, [1-9][0-9]* coastline points" -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_exit.cmake)
      set_tests_properties(swe_ocean.built_report PROPERTIES FIXTURES_REQUIRED ocean_earth)
    endif()
  endif()
  add_test(NAME swe_ocean.bad_option COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:swe_ocean> "-DARGS=--case tides"
    -DCODE=2 "-DMATCH=stommel\\|kelvin\\|rest\\|wind" -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_exit.cmake)
endif()
