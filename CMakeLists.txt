# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

cmake_minimum_required(VERSION 3.24)
project(soma VERSION 0.1.0
  DESCRIPTION "Scalable Operators for Mesh Abstractions" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

get_property(SOMA_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT SOMA_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
    "Build type: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependency policy: Kokkos >= 5.0 is the only required dependency. Every
# option below is OFF by default, and a requested dependency that cannot be
# found is a configure error. Nothing is silently disabled.
# ---------------------------------------------------------------------------
option(SOMA_ENABLE_MPI "Multi-rank distribution through MPI" OFF)
option(SOMA_ENABLE_TESTS "Build the test suite (requires GoogleTest)" OFF)
option(SOMA_FETCH_GTEST "Download GoogleTest when no installed copy is found" OFF)
option(SOMA_ENABLE_MOAB_REFERENCE "Build MOAB reference tests and benchmarks" OFF)
option(SOMA_LO_64 "64-bit local ordinals (ABI-breaking)" OFF)
option(SOMA_OFFSET_64 "64-bit CSR offsets (ABI-breaking)" OFF)
option(SOMA_DEBUG "Device-side bounds and contract checks" OFF)
# GPU-aware MPI is the default on LCF systems; the default is resolved after
# find_package(Kokkos) below, because it depends on whether Kokkos has a GPU backend.
set(SOMA_GPU_AWARE_MPI "AUTO" CACHE STRING "Pass device buffers to MPI: AUTO, ON, OFF")
set_property(CACHE SOMA_GPU_AWARE_MPI PROPERTY STRINGS AUTO ON OFF)
option(SOMA_INSTALL "Generate install and export rules" OFF)
set(SOMA_LARGE_DATA_DIR "" CACHE PATH "Directory of large meshes for scale tests")

include(GNUInstallDirs)
include(SomaAppleOpenMP)

soma_seed_apple_openmp()
find_package(Kokkos 5.0 REQUIRED)
soma_patch_apple_openmp_target()

if(SOMA_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

set(SOMA_KOKKOS_HAS_GPU OFF)
if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
  set(SOMA_KOKKOS_HAS_GPU ON)
endif()
string(TOUPPER "${SOMA_GPU_AWARE_MPI}" SOMA_GPU_AWARE_MPI_MODE)
if(NOT SOMA_GPU_AWARE_MPI_MODE MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR
    "SOMA_GPU_AWARE_MPI must be AUTO, ON or OFF (got '${SOMA_GPU_AWARE_MPI}')")
endif()
if(SOMA_GPU_AWARE_MPI_MODE STREQUAL "AUTO")
  if(SOMA_ENABLE_MPI AND SOMA_KOKKOS_HAS_GPU)
    set(SOMA_GPU_AWARE_MPI_RESOLVED ON)
  else()
    set(SOMA_GPU_AWARE_MPI_RESOLVED OFF)
  endif()
else()
  set(SOMA_GPU_AWARE_MPI_RESOLVED ${SOMA_GPU_AWARE_MPI_MODE})
endif()
if(SOMA_GPU_AWARE_MPI_RESOLVED AND NOT SOMA_ENABLE_MPI)
  message(FATAL_ERROR "SOMA_GPU_AWARE_MPI=ON requires SOMA_ENABLE_MPI=ON")
endif()
if(SOMA_GPU_AWARE_MPI_RESOLVED)
  set(SOMA_GPU_AWARE_MPI_DEFAULT 1)
else()
  set(SOMA_GPU_AWARE_MPI_DEFAULT 0)
endif()

configure_file(cmake/config.hpp.in
  "${PROJECT_BINARY_DIR}/generated/soma/config.hpp" @ONLY)

add_library(soma_warnings INTERFACE)
target_compile_options(soma_warnings INTERFACE
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,IntelLLVM,NVHPC>:-Wall;-Wextra>")

set(SOMA_SOURCES src/runtime/scope_guard.cpp)
if(SOMA_ENABLE_MPI)
  list(APPEND SOMA_SOURCES src/parallel/comm_mpi.cpp)
else()
  list(APPEND SOMA_SOURCES src/parallel/comm_serial.cpp)
endif()

add_library(soma ${SOMA_SOURCES})
add_library(soma::soma ALIAS soma)
target_compile_features(soma PUBLIC cxx_std_20)
target_include_directories(soma PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(soma
  PUBLIC Kokkos::kokkos
  PRIVATE $<BUILD_INTERFACE:soma_warnings>)
if(SOMA_ENABLE_MPI)
  target_link_libraries(soma PUBLIC MPI::MPI_CXX)
endif()

if(SOMA_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SOMA_INSTALL)
  include(SomaInstall)
endif()

message(STATUS "SOMA ${PROJECT_VERSION}: Kokkos ${Kokkos_VERSION} [${Kokkos_DEVICES}]"
  " MPI=${SOMA_ENABLE_MPI} tests=${SOMA_ENABLE_TESTS}"
  " GPU-aware-MPI=${SOMA_GPU_AWARE_MPI_RESOLVED}"
  " MOAB-reference=${SOMA_ENABLE_MOAB_REFERENCE} LO64=${SOMA_LO_64}"
  " build=${CMAKE_BUILD_TYPE}")
