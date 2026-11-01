# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

include(GoogleTest)

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  if(SOMA_FETCH_GTEST)
    include("${PROJECT_SOURCE_DIR}/cmake/CPM.cmake")
    CPMAddPackage(NAME googletest GITHUB_REPOSITORY google/googletest
      GIT_TAG v1.17.0 VERSION 1.17.0
      OPTIONS "INSTALL_GTEST OFF" "BUILD_GMOCK OFF")
    if(NOT TARGET GTest::gtest)
      add_library(GTest::gtest ALIAS gtest)
    endif()
  else()
    message(FATAL_ERROR
      "SOMA_ENABLE_TESTS=ON requires GoogleTest. Install it (e.g. brew install googletest) "
      "or configure with -DSOMA_FETCH_GTEST=ON to download it.")
  endif()
endif()

add_library(soma_test_main OBJECT soma_test_main.cpp)
target_link_libraries(soma_test_main PUBLIC soma GTest::gtest PRIVATE soma_warnings)

# soma_add_test(<name> SOURCES ... [RANKS 1 2 4] [LIBS ...])
# Without MPI (or without RANKS) tests are discovered individually. With MPI
# and RANKS, the binary is registered once per rank count.
function(soma_add_test name)
  cmake_parse_arguments(T "" "" "SOURCES;RANKS;LIBS" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE soma_test_main soma GTest::gtest soma_warnings ${T_LIBS})
  target_compile_definitions(${name} PRIVATE
    SOMA_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
    SOMA_LARGE_DATA_DIR="${SOMA_LARGE_DATA_DIR}")
  if(SOMA_ENABLE_MPI AND T_RANKS)
    foreach(np IN LISTS T_RANKS)
      add_test(NAME ${name}.np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(${name}.np${np} PROPERTIES TIMEOUT 60 PROCESSORS ${np})
    endforeach()
  else()
    gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST)
  endif()
endfunction()

soma_add_test(test_smoke SOURCES unit/test_smoke.cpp)
soma_add_test(test_comm SOURCES unit/test_comm.cpp RANKS 1 2 4)
soma_add_test(test_canonical SOURCES unit/test_canonical.cpp)
soma_add_test(test_node_tables SOURCES unit/test_node_tables.cpp)
soma_add_test(test_segments SOURCES unit/test_segments.cpp)
soma_add_test(test_data_type SOURCES unit/test_data_type.cpp)
soma_add_test(test_layout SOURCES unit/test_layout.cpp)
soma_add_test(test_domain SOURCES unit/test_domain.cpp)
soma_add_test(test_space SOURCES unit/test_space.cpp)
soma_add_test(test_field_registry SOURCES unit/test_field_registry.cpp)
soma_add_test(test_family SOURCES unit/test_family.cpp)
soma_add_test(test_field_commit SOURCES unit/test_field_commit.cpp)
soma_add_test(test_exception_safety SOURCES unit/test_exception_safety.cpp)
soma_add_test(test_device_structs SOURCES unit/test_device_structs.cpp)
soma_add_test(test_static_lifetime SOURCES unit/test_static_lifetime.cpp)
soma_add_test(test_move_noexcept SOURCES unit/test_move_noexcept.cpp)
soma_add_test(test_mesh_builder SOURCES unit/test_mesh_builder.cpp)

if(SOMA_ENABLE_MOAB_REFERENCE)
  find_package(MOAB REQUIRED)
  # MOAB's MOAB_LIBRARIES string breaks on CMake >= 4 ("-framework Accelerate"
  # is split); always link the imported target.
  if(TARGET MOAB::MOAB)
    set(SOMA_MOAB_TARGET MOAB::MOAB)
  elseif(TARGET MOAB)
    set(SOMA_MOAB_TARGET MOAB)
  else()
    message(FATAL_ERROR "SOMA_ENABLE_MOAB_REFERENCE=ON but MOAB exports no imported target.")
  endif()
  soma_add_test(test_canonical_moab SOURCES reference/test_canonical_moab.cpp
    LIBS ${SOMA_MOAB_TARGET})
  soma_add_test(test_node_tables_moab SOURCES reference/test_node_tables_moab.cpp
    LIBS ${SOMA_MOAB_TARGET})
endif()

add_test(NAME policy.mpi_seam
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/policy/check_mpi_seam.cmake)

if(SOMA_INSTALL)
  set(SOMA_PACKAGING_SHADOW_PREFIX "/opt/kokkos" CACHE PATH
    "Prefix of another Kokkos the fetched-Kokkos packaging test lists first (ignored if absent)")
  # '|' stands for ';' so the prefix list survives as one argument
  string(REPLACE ";" "|" SOMA_PACKAGING_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
  add_test(NAME packaging.find_package
    COMMAND ${CMAKE_COMMAND}
      -DSOMA_BINARY_DIR=${PROJECT_BINARY_DIR} -DSOMA_SOURCE_DIR=${PROJECT_SOURCE_DIR}
      -DSCRATCH_DIR=${PROJECT_BINARY_DIR}/packaging_test
      -DBUILD_TYPE=$<CONFIG> -DGENERATOR=${CMAKE_GENERATOR}
      -DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DPREFIX_PATH=${SOMA_PACKAGING_PREFIX_PATH}
      -DKOKKOS_SOURCE=${SOMA_KOKKOS_SOURCE} -DKOKKOS_DIR=${Kokkos_DIR}
      -DEXTRA_PREFIX=${SOMA_PACKAGING_SHADOW_PREFIX}
      -DENABLE_MPI=${SOMA_ENABLE_MPI} -DMPI_HOME=${MPI_HOME}
      -DMPIEXEC=${MPIEXEC_EXECUTABLE}
      -P ${PROJECT_SOURCE_DIR}/tests/packaging/run_packaging_test.cmake)
  set_tests_properties(packaging.find_package PROPERTIES TIMEOUT 300)
endif()
