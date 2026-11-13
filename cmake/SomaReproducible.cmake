# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# SOMA_REPRODUCIBLE=ON: the compile flags that keep floating-point arithmetic
# as written, for the compiler and the Kokkos backend at hand, so that a
# kernel rounds the same in a host space and in a device space.
#
# What differs between compilers is contraction (a * b + c as one fused
# multiply-add) and, for a few, a default that reorders arithmetic:
#
#   g++, clang++                 contract where the target has an fma
#   icpx                         -fp-model=fast by default: reorders too
#   nvc++                        contracts, and relaxes at -fast
#   nvcc, clang and hipcc        always contract device code
#     compiling CUDA or HIP
#
#   soma_reproducible_setup()             after soma_kokkos_language_setup()
#   soma_target_reproducible(<target>)    PUBLIC flags on the library
#
# soma_reproducible_setup sets SOMA_REPRODUCIBLE_BUILD to 1 or 0 and, when 1,
# SOMA_REPRODUCIBLE_<LANG>_FLAGS for CXX and the Kokkos compile language; it
# stops the configuration for a compiler it has no flags for, for flags the
# compiler does not take, and for fast-math flags in CMAKE_<LANG>_FLAGS.

include_guard(GLOBAL)
include(CheckCompilerFlag)

# The flags for C++ compiled by <id>, a list; empty for an unknown compiler.
function(_soma_reproducible_host_flags out id)
  if(id MATCHES "^(GNU|Clang|AppleClang)$")
    set(${out} -ffp-contract=off PARENT_SCOPE)
  elseif(id STREQUAL "IntelLLVM")
    set(${out} -fp-model=precise -ffp-contract=off PARENT_SCOPE)
  elseif(id STREQUAL "NVHPC")
    set(${out} -Kieee -Mnofma PARENT_SCOPE)
  else()
    set(${out} "" PARENT_SCOPE)
  endif()
endfunction()

# Fast-math flags undo what the option promises; whoever set them must choose.
function(_soma_reproducible_refuse_fast_math lang)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _type)
  set(_all "${CMAKE_${lang}_FLAGS} ${CMAKE_${lang}_FLAGS_${_type}}")
  if(_all MATCHES "(^|[ ,=])(-ffast-math|-Ofast|-funsafe-math-optimizations|-fassociative-math|--?use_fast_math|-fp-model[= ]fast|-Mfprelaxed)")
    message(FATAL_ERROR "SOMA_REPRODUCIBLE=ON, but the ${lang} flags hold '${CMAKE_MATCH_2}' "
      "(${_all}): it lets the compiler reorder floating-point arithmetic. Remove it, or configure with "
      "SOMA_REPRODUCIBLE=OFF")
  endif()
endfunction()

function(_soma_reproducible_check lang flags)
  foreach(_f IN LISTS flags)
    string(MAKE_C_IDENTIFIER "SOMA_REPRODUCIBLE_${lang}_TAKES${_f}" _var)
    check_compiler_flag(${lang} "${_f}" ${_var})
    if(NOT ${_var})
      message(FATAL_ERROR "SOMA_REPRODUCIBLE=ON: the ${lang} compiler '${CMAKE_${lang}_COMPILER}' "
        "(${CMAKE_${lang}_COMPILER_ID} ${CMAKE_${lang}_COMPILER_VERSION}) does not take '${_f}'")
    endif()
  endforeach()
endfunction()

# A macro: its results are the caller's variables.
macro(soma_reproducible_setup)
  set(SOMA_REPRODUCIBLE_BUILD 0)
  set(SOMA_REPRODUCIBLE_SUMMARY "OFF")
  if(SOMA_REPRODUCIBLE)
    _soma_reproducible_host_flags(SOMA_REPRODUCIBLE_CXX_FLAGS "${CMAKE_CXX_COMPILER_ID}")
    if(NOT SOMA_REPRODUCIBLE_CXX_FLAGS)
      message(FATAL_ERROR "SOMA_REPRODUCIBLE=ON has no flags for the ${CMAKE_CXX_COMPILER_ID} C++ compiler "
        "(known: GNU, Clang, AppleClang, IntelLLVM, NVHPC)")
    endif()
    # nvcc_wrapper as the C++ compiler is identified as its host compiler: it
    # hands -fmad to nvcc and the host's own flags to the host pass.
    if(SOMA_KOKKOS_GPU_BACKEND STREQUAL "CUDA" AND SOMA_KOKKOS_COMPILE_LANGUAGE STREQUAL "CXX")
      soma_compiler_is_nvcc(_soma_cxx_is_nvcc "${CMAKE_CXX_COMPILER}")
      if(_soma_cxx_is_nvcc)
        list(PREPEND SOMA_REPRODUCIBLE_CXX_FLAGS -fmad=false)
      endif()
      unset(_soma_cxx_is_nvcc)
    endif()
    _soma_reproducible_refuse_fast_math(CXX)
    _soma_reproducible_check(CXX "${SOMA_REPRODUCIBLE_CXX_FLAGS}")
    set(SOMA_REPRODUCIBLE_SUMMARY "ON [CXX ${SOMA_REPRODUCIBLE_CXX_FLAGS}]")

    if(SOMA_KOKKOS_COMPILE_LANGUAGE STREQUAL "CUDA")
      if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        # nvcc: -fmad for the device, the host compiler's flags for its host pass.
        _soma_reproducible_host_flags(_soma_cuda_host "${CMAKE_CXX_COMPILER_ID}")
        list(JOIN _soma_cuda_host "," _soma_cuda_host)
        set(SOMA_REPRODUCIBLE_CUDA_FLAGS -fmad=false "-Xcompiler=${_soma_cuda_host}")
        unset(_soma_cuda_host)
      elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
        set(SOMA_REPRODUCIBLE_CUDA_FLAGS -ffp-contract=off)
      else()
        message(FATAL_ERROR "SOMA_REPRODUCIBLE=ON has no flags for the ${CMAKE_CUDA_COMPILER_ID} CUDA compiler "
          "(known: NVIDIA, Clang)")
      endif()
      _soma_reproducible_refuse_fast_math(CUDA)
      _soma_reproducible_check(CUDA "${SOMA_REPRODUCIBLE_CUDA_FLAGS}")
      string(APPEND SOMA_REPRODUCIBLE_SUMMARY " [CUDA ${SOMA_REPRODUCIBLE_CUDA_FLAGS}]")
    elseif(SOMA_KOKKOS_COMPILE_LANGUAGE STREQUAL "HIP")
      if(NOT CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "SOMA_REPRODUCIBLE=ON has no flags for the ${CMAKE_HIP_COMPILER_ID} HIP compiler "
          "(known: Clang)")
      endif()
      set(SOMA_REPRODUCIBLE_HIP_FLAGS -ffp-contract=off)
      _soma_reproducible_refuse_fast_math(HIP)
      _soma_reproducible_check(HIP "${SOMA_REPRODUCIBLE_HIP_FLAGS}")
      string(APPEND SOMA_REPRODUCIBLE_SUMMARY " [HIP ${SOMA_REPRODUCIBLE_HIP_FLAGS}]")
    endif()
    if(SOMA_KOKKOS_GPU_BACKEND STREQUAL "SYCL")
      message(WARNING "SOMA_REPRODUCIBLE=ON with a SYCL Kokkos: the flags reach the host code; whether a device "
        "kernel, compiled again by the device's runtime, rounds as the host does has not been verified")
    endif()
    set(SOMA_REPRODUCIBLE_BUILD 1)
  endif()
endmacro()

# PUBLIC: SOMA's kernels are templates in its headers, compiled in the targets
# that use them, so those take the flags too (also from an installed SOMA).
function(soma_target_reproducible target)
  if(NOT SOMA_REPRODUCIBLE_BUILD)
    return()
  endif()
  foreach(_lang CXX CUDA HIP)
    if(SOMA_REPRODUCIBLE_${_lang}_FLAGS)
      target_compile_options(${target} PUBLIC "$<$<COMPILE_LANGUAGE:${_lang}>:${SOMA_REPRODUCIBLE_${_lang}_FLAGS}>")
    endif()
  endforeach()
endfunction()
