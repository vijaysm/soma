# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# AppleClang ships without OpenMP. A Kokkos built with its OpenMP backend calls
# find_dependency(OpenMP), which FindOpenMP cannot satisfy unaided on macOS, so
# every configure fails. Seed FindOpenMP from Homebrew's libomp first.

# Seed a cache entry unless the user already supplied a usable value.
function(_soma_seed_cache var type value)
  if("${${var}}" STREQUAL "" OR "${${var}}" MATCHES "(^|-)NOTFOUND$")
    set(${var} "${value}" CACHE ${type} "" FORCE)
  endif()
endfunction()

# SOMA_APPLE_OPENMP_MANUAL is an INTERNAL cache entry set once, when the
# fallback is first used. On a reconfigure the seeded cache already satisfies
# find_package(OpenMP), so a plain variable would flip back to OFF.
macro(soma_seed_apple_openmp)
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
        _soma_seed_cache(OpenMP_CXX_FLAGS STRING "-Xpreprocessor;-fopenmp")
        _soma_seed_cache(OpenMP_CXX_LIB_NAMES STRING "omp")
        _soma_seed_cache(OpenMP_omp_LIBRARY FILEPATH "${SOMA_LIBOMP_LIBRARY}")
        _soma_seed_cache(OpenMP_CXX_INCLUDE_DIR PATH "${SOMA_LIBOMP_INCLUDE_DIR}")
        find_package(OpenMP COMPONENTS CXX QUIET)
        set(SOMA_APPLE_OPENMP_MANUAL ON CACHE INTERNAL "libomp fallback in use")
        message(STATUS "SOMA: AppleClang libomp fallback: ${SOMA_LIBOMP_LIBRARY}")
      endif()
    endif()
  endif()
endmacro()

# The synthesised OpenMP::OpenMP_CXX target can lack libomp's include
# directory, so omp.h is not found by Kokkos headers. Append it.
macro(soma_patch_apple_openmp_target)
  if(SOMA_APPLE_OPENMP_MANUAL AND TARGET OpenMP::OpenMP_CXX)
    get_target_property(_soma_omp_incs OpenMP::OpenMP_CXX INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT SOMA_LIBOMP_INCLUDE_DIR IN_LIST _soma_omp_incs
       AND NOT "$<BUILD_INTERFACE:${SOMA_LIBOMP_INCLUDE_DIR}>" IN_LIST _soma_omp_incs)
      set_property(TARGET OpenMP::OpenMP_CXX APPEND PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${SOMA_LIBOMP_INCLUDE_DIR}")
    endif()
  endif()
endmacro()
