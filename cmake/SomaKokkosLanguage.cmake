# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# A Kokkos built with Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON compiles its
# device code as the CMake CUDA (or HIP) language and reports that in
# Kokkos_COMPILE_LANGUAGE. Its device flags (-extended-lambda, the GPU
# architecture, ...) are exported as $<COMPILE_LANGUAGE:CUDA> options, so they
# reach a consumer's translation units only when those are compiled as that
# language too. Every SOMA source includes Kokkos, so every SOMA source must be.
# A Kokkos built the other way (nvcc_wrapper, hipcc or icpx as the C++ compiler)
# sets Kokkos_COMPILE_LANGUAGE to CXX, or not at all, and plain C++ compilation
# is right. This module is shared by the build tree and the installed config.
#
#   soma_kokkos_language_setup()          after find_package(Kokkos), at file scope
#   soma_target_kokkos_sources(<target>)  after the target has all its sources
#
# soma_kokkos_language_setup sets SOMA_KOKKOS_COMPILE_LANGUAGE to CXX, CUDA or
# HIP. soma_target_kokkos_sources is a no-op for CXX.

include_guard(GLOBAL)

# Sets <out> to ON when <compiler> is nvcc or nvcc_wrapper, also behind an MPI
# wrapper (CMake identifies nvcc_wrapper as its host compiler, so the compiler
# id cannot tell).
function(soma_compiler_is_nvcc out compiler)
  execute_process(COMMAND "${compiler}" --version
    OUTPUT_VARIABLE _v ERROR_VARIABLE _v RESULT_VARIABLE _rc)
  if(_rc EQUAL 0 AND _v MATCHES "nvcc")
    set(${out} ON PARENT_SCOPE)
  else()
    set(${out} OFF PARENT_SCOPE)
  endif()
endfunction()

# The CUDA host compiler defaults to the C++ compiler, so an MPI wrapper around
# a host compiler keeps supplying its include paths and libraries to the host
# pass of nvcc. A user value, or CUDAHOSTCXX in the environment, wins. Must run
# before the CUDA language is enabled, by SOMA or by an in-tree Kokkos.
function(soma_kokkos_default_cuda_host_compiler)
  if(NOT CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH
      "Host compiler for CUDA sources (SOMA default: the C++ compiler)")
  endif()
endfunction()

# Text of the configure error for nvcc as the C++ compiler in language mode.
function(soma_kokkos_nvcc_problem out)
  set(${out} "the C++ compiler '${CMAKE_CXX_COMPILER}' is nvcc or nvcc_wrapper, but \
this Kokkos compiles device code as the CMake CUDA language, which drives nvcc itself. \
Use a host C++ compiler (-DCMAKE_CXX_COMPILER=g++ or clang++; for an MPI wrapper point \
OMPI_CXX or MPICH_CXX at one). nvcc_wrapper is for a Kokkos built without \
Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE" PARENT_SCOPE)
endfunction()

# A macro, because enable_language() must run at file scope.
macro(soma_kokkos_language_setup)
  set(SOMA_KOKKOS_COMPILE_LANGUAGE CXX)
  if(DEFINED Kokkos_COMPILE_LANGUAGE AND NOT "${Kokkos_COMPILE_LANGUAGE}" STREQUAL "CXX"
     AND NOT "${Kokkos_COMPILE_LANGUAGE}" STREQUAL "")
    set(SOMA_KOKKOS_COMPILE_LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
  endif()
  if(SOMA_KOKKOS_COMPILE_LANGUAGE STREQUAL "CUDA")
    if(CMAKE_VERSION VERSION_LESS 3.25.2)
      message(FATAL_ERROR
        "Kokkos as the CMake CUDA language with C++20 needs CMake >= 3.25.2 (have ${CMAKE_VERSION})")
    endif()
    soma_compiler_is_nvcc(_soma_cxx_is_nvcc "${CMAKE_CXX_COMPILER}")
    if(_soma_cxx_is_nvcc)
      soma_kokkos_nvcc_problem(_soma_nvcc_problem)
      message(FATAL_ERROR "Kokkos (${Kokkos_DIR}): ${_soma_nvcc_problem}.")
    endif()
    unset(_soma_cxx_is_nvcc)
    unset(_soma_nvcc_problem)
    soma_kokkos_default_cuda_host_compiler()
  endif()
  if(NOT SOMA_KOKKOS_COMPILE_LANGUAGE STREQUAL "CXX")
    enable_language(${SOMA_KOKKOS_COMPILE_LANGUAGE})
  endif()
endmacro()

# Compiles the C++ sources of <target> as SOMA_KOKKOS_COMPILE_LANGUAGE and gives
# the target the standard and GPU architectures Kokkos was built with. Sources
# added to the target later are not covered; generator expressions are skipped.
function(soma_target_kokkos_sources target)
  if(NOT SOMA_KOKKOS_COMPILE_LANGUAGE OR SOMA_KOKKOS_COMPILE_LANGUAGE STREQUAL "CXX")
    return()
  endif()
  set(_lang ${SOMA_KOKKOS_COMPILE_LANGUAGE})
  get_target_property(_sources ${target} SOURCES)
  get_target_property(_dir ${target} SOURCE_DIR)
  set(_files)
  foreach(_s IN LISTS _sources)
    if(_s MATCHES "^\\$<")
      continue()
    endif()
    get_filename_component(_abs "${_s}" ABSOLUTE BASE_DIR "${_dir}")
    if(_abs MATCHES "\\.(cpp|cxx|cc|C)$")
      list(APPEND _files "${_abs}")
    endif()
  endforeach()
  if(_files)
    set_source_files_properties(${_files} TARGET_DIRECTORY ${target}
      PROPERTIES LANGUAGE ${_lang})
  endif()
  set(_std 20)
  if(Kokkos_CXX_STANDARD)
    set(_std ${Kokkos_CXX_STANDARD})
  endif()
  set_target_properties(${target} PROPERTIES
    ${_lang}_STANDARD ${_std} ${_lang}_STANDARD_REQUIRED ON ${_lang}_EXTENSIONS OFF)
  if(Kokkos_${_lang}_ARCHITECTURES)
    set_target_properties(${target} PROPERTIES
      ${_lang}_ARCHITECTURES "${Kokkos_${_lang}_ARCHITECTURES}")
  endif()
endfunction()
