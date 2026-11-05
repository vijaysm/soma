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
option(SOMA_ENABLE_HDF5 "MOAB .h5m input through HDF5 (set HDF5_ROOT)" OFF)
option(SOMA_ENABLE_TESTS "Build the test suite (requires GoogleTest)" OFF)
option(SOMA_FETCH_GTEST "Download GoogleTest when no installed copy is found" OFF)
option(SOMA_FETCH_KOKKOS "Download and build Kokkos when no installed copy is found" OFF)
set(SOMA_KOKKOS_VERSION "5.1.0" CACHE STRING "Kokkos tag to fetch with SOMA_FETCH_KOKKOS")
set(SOMA_KOKKOS_GPU "AUTO" CACHE STRING
  "GPU backend of a fetched Kokkos: AUTO, NONE, CUDA, HIP, SYCL")
set_property(CACHE SOMA_KOKKOS_GPU PROPERTY STRINGS AUTO NONE CUDA HIP SYCL)
option(SOMA_ENABLE_MOAB_REFERENCE "Build MOAB reference tests and benchmarks" OFF)
option(SOMA_ENABLE_MOAB_COMPAT "MOAB's API over SOMA meshes (soma::compat)" OFF)
option(SOMA_LO_64 "64-bit local ordinals (ABI-breaking)" OFF)
option(SOMA_OFFSET_64 "64-bit CSR offsets (ABI-breaking)" OFF)
option(SOMA_DEBUG "Device-side bounds and contract checks" OFF)
# GPU-aware MPI is the default on LCF systems; the default is resolved after
# find_package(Kokkos) below, because it depends on whether Kokkos has a GPU backend.
set(SOMA_GPU_AWARE_MPI "AUTO" CACHE STRING "Pass device buffers to MPI: AUTO, ON, OFF")
set_property(CACHE SOMA_GPU_AWARE_MPI PROPERTY STRINGS AUTO ON OFF)
# Installing is the default only for a top-level build; a parent project that
# embeds SOMA does not get SOMA's install rules unless it asks for them.
option(SOMA_INSTALL "Generate install and export rules (default: ON when SOMA is the top-level project)"
  ${PROJECT_IS_TOP_LEVEL})
set(SOMA_LARGE_DATA_DIR "" CACHE PATH "Directory of large meshes for scale tests")

include(GNUInstallDirs)
include(SomaKokkos)

soma_find_or_fetch_kokkos()

if(SOMA_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()
if(SOMA_ENABLE_HDF5)
  include(SomaHdf5)
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

set(SOMA_SOURCES src/runtime/scope_guard.cpp src/runtime/options.cpp src/core/segments.cpp src/fields/node_tables.cpp src/core/mesh_builder.cpp src/core/mesh_edit.cpp src/core/permute.cpp src/testing/box_mesh.cpp src/io/vtk_writer.cpp src/io/vtk_reader.cpp src/amr/box_list.cpp src/amr/box_array.cpp src/amr/halo.cpp src/amr/copy.cpp src/amr/interlevel.cpp src/amr/flux_register.cpp src/amr/cluster.cpp src/amr/hierarchy.cpp src/amr/vtk_amr.cpp src/parallel/transport/route.cpp src/parallel/transport/comm_plan.cpp src/parallel/transport/sort.cpp src/parallel/partition.cpp src/parallel/sharing.cpp src/parallel/distributed.cpp src/parallel/migrate.cpp src/parallel/ghosts.cpp)
if(SOMA_ENABLE_MPI)
  list(APPEND SOMA_SOURCES src/parallel/comm_mpi.cpp)
else()
  list(APPEND SOMA_SOURCES src/parallel/comm_serial.cpp)
endif()
if(SOMA_ENABLE_HDF5)
  list(APPEND SOMA_SOURCES src/io/h5m_reader.cpp src/io/h5m_parallel_reader.cpp src/io/h5m_writer.cpp)
else()
  list(APPEND SOMA_SOURCES src/io/h5m_unavailable.cpp)
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
if(SOMA_ENABLE_HDF5)
  target_link_libraries(soma PRIVATE HDF5::HDF5)
endif()

if(SOMA_ENABLE_MOAB_COMPAT)
  add_library(soma_compat src/compat/range.cpp src/compat/cn.cpp src/compat/error.cpp src/compat/core.cpp
    src/compat/core_tags.cpp src/compat/core_sets.cpp src/compat/core_io.cpp src/compat/scd.cpp src/compat/prog_options.cpp)
  add_library(soma::compat ALIAS soma_compat)
  target_link_libraries(soma_compat PUBLIC soma PRIVATE $<BUILD_INTERFACE:soma_warnings>)
  # Drop-in mode: MOAB's include paths (moab/Core.hpp, ...) name soma::compat.
  add_library(soma_compat_dropin INTERFACE)
  add_library(soma::compat_dropin ALIAS soma_compat_dropin)
  target_include_directories(soma_compat_dropin INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/soma/compat/dropin>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/soma/compat/dropin>)
  if(NOT SOMA_ENABLE_MPI)
    # The serial stand-in for <mpi.h> that MOAB applications include.
    target_include_directories(soma_compat_dropin INTERFACE
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/soma/compat/serial_mpi>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/soma/compat/serial_mpi>)
  endif()
  target_link_libraries(soma_compat_dropin INTERFACE soma_compat)
endif()

if(SOMA_ENABLE_MOAB_REFERENCE)
  include(SomaMoab)
endif()

if(SOMA_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SOMA_ENABLE_MOAB_REFERENCE)
  add_subdirectory(bench)
endif()

if(SOMA_INSTALL)
  include(SomaInstall)
endif()

message(STATUS "SOMA ${PROJECT_VERSION}: Kokkos ${SOMA_KOKKOS_VERSION_FOUND}"
  " (${SOMA_KOKKOS_SOURCE}) host=${SOMA_KOKKOS_HOST_BACKEND}"
  " gpu=${SOMA_KOKKOS_GPU_BACKEND} [${SOMA_KOKKOS_DEVICES}]"
  " MPI=${SOMA_ENABLE_MPI} HDF5=${SOMA_ENABLE_HDF5} tests=${SOMA_ENABLE_TESTS}"
  " GPU-aware-MPI=${SOMA_GPU_AWARE_MPI_RESOLVED}"
  " MOAB-reference=${SOMA_ENABLE_MOAB_REFERENCE} MOAB-compat=${SOMA_ENABLE_MOAB_COMPAT}"
  " LO64=${SOMA_LO_64}"
  " build=${CMAKE_BUILD_TYPE}")
