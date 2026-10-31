# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# AppleClang ships without OpenMP. A Kokkos built with its OpenMP backend calls
# find_dependency(OpenMP), which FindOpenMP cannot satisfy unaided on macOS, so
# every configure fails. Seed FindOpenMP from Homebrew's libomp first.

macro(soma_seed_apple_openmp)
  set(SOMA_APPLE_OPENMP_MANUAL OFF)
  if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    find_package(OpenMP COMPONENTS CXX QUIET)
    if(NOT OpenMP_CXX_FOUND)
      set(_soma_omp_prefixes /opt/homebrew/opt/libomp /usr/local/opt/libomp)
      execute_process(COMMAND brew --prefix libomp
        OUTPUT_VARIABLE _soma_brew_omp ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(_soma_brew_omp)
        list(PREPEND _soma_omp_prefixes "${_soma_brew_omp}")
      endif()
      find_path(SOMA_LIBOMP_INCLUDE_DIR omp.h
        PATHS ${_soma_omp_prefixes} PATH_SUFFIXES include NO_DEFAULT_PATH)
      find_library(SOMA_LIBOMP_LIBRARY NAMES omp
        PATHS ${_soma_omp_prefixes} PATH_SUFFIXES lib NO_DEFAULT_PATH)
      if(SOMA_LIBOMP_INCLUDE_DIR AND SOMA_LIBOMP_LIBRARY)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor;-fopenmp" CACHE STRING "" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_omp_LIBRARY "${SOMA_LIBOMP_LIBRARY}" CACHE FILEPATH "" FORCE)
        set(OpenMP_CXX_INCLUDE_DIR "${SOMA_LIBOMP_INCLUDE_DIR}" CACHE PATH "" FORCE)
        find_package(OpenMP COMPONENTS CXX QUIET)
        set(SOMA_APPLE_OPENMP_MANUAL ON)
        message(STATUS "SOMA: AppleClang libomp fallback: ${SOMA_LIBOMP_LIBRARY}")
      endif()
    endif()
  endif()
endmacro()

# The synthesised OpenMP::OpenMP_CXX target can lack libomp's include
# directory, so omp.h is not found by Kokkos headers. Append it.
macro(soma_patch_apple_openmp_target)
  if(SOMA_APPLE_OPENMP_MANUAL AND TARGET OpenMP::OpenMP_CXX)
    set_property(TARGET OpenMP::OpenMP_CXX APPEND PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES "${SOMA_LIBOMP_INCLUDE_DIR}")
  endif()
endmacro()
