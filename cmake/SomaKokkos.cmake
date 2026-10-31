# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# soma_find_or_fetch_kokkos(): an installed Kokkos >= 5.0 always wins. When none
# is found, SOMA_FETCH_KOKKOS=ON downloads and builds one in-tree, otherwise the
# configure fails. Results, set in the caller's scope:
#   SOMA_KOKKOS_SOURCE   installed | fetched
#   SOMA_KOKKOS_VERSION_FOUND, SOMA_KOKKOS_DEVICES
#   SOMA_KOKKOS_HOST_BACKEND (OpenMP, Threads, HPX or Serial)
#   SOMA_KOKKOS_GPU_BACKEND (CUDA, HIP, SYCL or none), SOMA_KOKKOS_HAS_GPU

include_guard(GLOBAL)
include(SomaAppleOpenMP)

# Name=value pairs of the Kokkos options as they stood after the last fetch
# configure. An option that is defined and differs from this record (or is
# missing from it) was set by the user; this survives a reconfigure, when
# Kokkos has filled the cache with its own defaults.
if(NOT DEFINED SOMA_KOKKOS_SEEN)
  set(SOMA_KOKKOS_SEEN "" CACHE INTERNAL "Kokkos options not set by the user")
endif()

set(_SOMA_KOKKOS_OPTION_REGEX
  "^Kokkos_(ENABLE_(SERIAL|OPENMP|THREADS|HPX|CUDA|HIP|SYCL)|ARCH_.+)$")

# Sets <out> to ON or OFF when the user defined <name>, and to "" otherwise.
function(_soma_kokkos_user_state out name)
  set(${out} "" PARENT_SCOPE)
  if(DEFINED ${name} AND NOT "${name}=${${name}}" IN_LIST SOMA_KOKKOS_SEEN)
    if(${name})
      set(${out} ON PARENT_SCOPE)
    else()
      set(${out} OFF PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(_soma_kokkos_force name value)
  set(${name} ${value} CACHE BOOL "Set by SOMA for the fetched Kokkos" FORCE)
endfunction()

# Decision functions are pure: they only look at their arguments.

# out: OPENMP, THREADS (enable it) or NONE (leave the host backends alone).
function(_soma_host_decide out out_reason)
  cmake_parse_arguments(A "" "OPENMP_USABLE;USER_OPENMP;USER_THREADS" "" ${ARGN})
  if(A_USER_OPENMP STREQUAL "ON" OR A_USER_THREADS STREQUAL "ON")
    set(_r NONE "host backend chosen by the user")
  elseif(A_OPENMP_USABLE AND NOT A_USER_OPENMP STREQUAL "OFF")
    set(_r OPENMP "OpenMP is usable")
  elseif(A_USER_THREADS STREQUAL "OFF")
    set(_r NONE "OpenMP is not used and Kokkos_ENABLE_THREADS=OFF was set by the user")
  elseif(A_USER_OPENMP STREQUAL "OFF")
    set(_r THREADS "Kokkos_ENABLE_OPENMP=OFF was set by the user")
  else()
    set(_r THREADS "OpenMP was not found")
  endif()
  list(GET _r 0 _backend)
  list(GET _r 1 _why)
  set(${out} ${_backend} PARENT_SCOPE)
  set(${out_reason} "${_why}" PARENT_SCOPE)
endfunction()

# out: NONE, CUDA, HIP, SYCL (enable it) or USER (leave the GPU options alone).
# HAVE_NVCC and HAVE_HIP say that the matching toolchain was found.
function(_soma_gpu_decide out out_reason)
  cmake_parse_arguments(A "HAVE_NVCC;HAVE_HIP;USER_SET" "REQUEST;COMPILER_ID;COMPILER" "" ${ARGN})
  get_filename_component(_cc "${A_COMPILER}" NAME)
  if(A_USER_SET)
    set(_r USER "Kokkos_ENABLE_CUDA/HIP/SYCL set by the user")
  elseif(A_REQUEST STREQUAL "NONE")
    set(_r NONE "SOMA_KOKKOS_GPU=NONE")
  elseif(NOT A_REQUEST STREQUAL "AUTO")
    set(_r ${A_REQUEST} "SOMA_KOKKOS_GPU=${A_REQUEST}")
  elseif(A_HAVE_NVCC)
    set(_r CUDA "nvcc found")
  elseif(A_HAVE_HIP)
    set(_r HIP "hipcc, amdclang++, ROCM_PATH or HIP_PATH found")
  elseif(A_COMPILER_ID STREQUAL "IntelLLVM" OR _cc STREQUAL "icpx")
    set(_r SYCL "C++ compiler is icpx")
  else()
    set(_r NONE "no GPU toolchain found")
  endif()
  list(GET _r 0 _backend)
  list(GET _r 1 _why)
  set(${out} ${_backend} PARENT_SCOPE)
  set(${out_reason} "${_why}" PARENT_SCOPE)
endfunction()

# Sets <out> to a message when the C++ compiler cannot drive the backend, and
# to "" when it can.
function(_soma_gpu_compiler_problem out)
  cmake_parse_arguments(A "AS_LANGUAGE" "BACKEND;COMPILER;COMPILER_ID" "" ${ARGN})
  get_filename_component(_cc "${A_COMPILER}" NAME)
  set(_ok FALSE)
  if(A_BACKEND STREQUAL "CUDA")
    if(A_AS_LANGUAGE OR A_COMPILER_ID STREQUAL "Clang" OR _cc MATCHES "^nvcc")
      set(_ok TRUE)
    else()
      set(_fix "use a CUDA-capable clang++ (-DCMAKE_CXX_COMPILER=/path/to/clang++) or the \
bin/nvcc_wrapper of a Kokkos source tree or installation \
(-DCMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper). The nvcc_wrapper of the \
fetched Kokkos is not available when the compiler is chosen")
    endif()
  elseif(A_BACKEND STREQUAL "HIP")
    if(A_COMPILER_ID MATCHES "^(Clang|ROCMClang)$" OR _cc MATCHES "^(hipcc|amdclang)")
      set(_ok TRUE)
    else()
      set(_fix "use -DCMAKE_CXX_COMPILER=hipcc or amdclang++ (from ROCm)")
    endif()
  elseif(A_BACKEND STREQUAL "SYCL")
    if(A_COMPILER_ID STREQUAL "IntelLLVM" OR _cc STREQUAL "icpx")
      set(_ok TRUE)
    else()
      set(_fix "use -DCMAKE_CXX_COMPILER=icpx")
    endif()
  endif()
  if(_ok)
    set(${out} "" PARENT_SCOPE)
  else()
    set(${out} "the C++ compiler '${A_COMPILER}' cannot build the ${A_BACKEND} backend: ${_fix}"
      PARENT_SCOPE)
  endif()
endfunction()

# Sets <out_nvcc> and <out_hip>. SOMA_PROBE_NVCC and SOMA_PROBE_HIP, when
# defined, replace the search.
function(_soma_gpu_probe out_nvcc out_hip)
  if(DEFINED SOMA_PROBE_NVCC)
    set(_nvcc ${SOMA_PROBE_NVCC})
  else()
    find_program(_soma_nvcc nvcc
      HINTS ${CUDAToolkit_ROOT} $ENV{CUDA_HOME} $ENV{CUDA_PATH} PATH_SUFFIXES bin)
    set(_nvcc OFF)
    if(_soma_nvcc)
      set(_nvcc ON)
    endif()
    unset(_soma_nvcc CACHE)
  endif()
  if(DEFINED SOMA_PROBE_HIP)
    set(_hip ${SOMA_PROBE_HIP})
  else()
    find_program(_soma_hipcc NAMES hipcc amdclang++
      HINTS ${ROCM_PATH} $ENV{ROCM_PATH} $ENV{HIP_PATH} PATH_SUFFIXES bin)
    set(_hip OFF)
    if(_soma_hipcc OR DEFINED ENV{ROCM_PATH} OR DEFINED ENV{HIP_PATH} OR ROCM_PATH)
      set(_hip ON)
    endif()
    unset(_soma_hipcc CACHE)
  endif()
  set(${out_nvcc} ${_nvcc} PARENT_SCOPE)
  set(${out_hip} ${_hip} PARENT_SCOPE)
endfunction()

function(_soma_fetch_kokkos)
  string(TOUPPER "${SOMA_KOKKOS_GPU}" _request)
  if(NOT _request MATCHES "^(AUTO|NONE|CUDA|HIP|SYCL)$")
    message(FATAL_ERROR
      "SOMA_KOKKOS_GPU must be AUTO, NONE, CUDA, HIP or SYCL (got '${SOMA_KOKKOS_GPU}')")
  endif()

  get_cmake_property(_vars VARIABLES)
  list(FILTER _vars INCLUDE REGEX "${_SOMA_KOKKOS_OPTION_REGEX}")
  set(_user_names)
  foreach(_v IN LISTS _vars)
    if(NOT "${_v}=${${_v}}" IN_LIST SOMA_KOKKOS_SEEN)
      list(APPEND _user_names ${_v})
    endif()
  endforeach()
  set(_user_arch ${_user_names})
  list(FILTER _user_arch INCLUDE REGEX "^Kokkos_ARCH_")

  _soma_kokkos_user_state(_user_serial Kokkos_ENABLE_SERIAL)
  _soma_kokkos_user_state(_user_openmp Kokkos_ENABLE_OPENMP)
  _soma_kokkos_user_state(_user_threads Kokkos_ENABLE_THREADS)
  if(NOT _user_serial)
    _soma_kokkos_force(Kokkos_ENABLE_SERIAL ON)
  endif()

  find_package(OpenMP COMPONENTS CXX QUIET)
  set(_omp_usable OFF)
  if(OpenMP_CXX_FOUND)
    set(_omp_usable ON)
    # Kokkos compiles its OpenMP backend against this target
    soma_patch_apple_openmp_target()
  endif()
  _soma_host_decide(_host _host_why OPENMP_USABLE ${_omp_usable}
    USER_OPENMP "${_user_openmp}" USER_THREADS "${_user_threads}")
  if(_host STREQUAL "OPENMP")
    _soma_kokkos_force(Kokkos_ENABLE_OPENMP ON)
    if(NOT _user_threads)
      _soma_kokkos_force(Kokkos_ENABLE_THREADS OFF)
    endif()
  elseif(_host STREQUAL "THREADS")
    _soma_kokkos_force(Kokkos_ENABLE_THREADS ON)
    if(NOT _user_openmp)
      _soma_kokkos_force(Kokkos_ENABLE_OPENMP OFF)
    endif()
  endif()
  message(STATUS "SOMA: Kokkos host backend request: ${_host} (${_host_why})")

  set(_gpu_user_set OFF)
  set(_gpu_user_on OFF)
  foreach(_b CUDA HIP SYCL)
    _soma_kokkos_user_state(_user_${_b} Kokkos_ENABLE_${_b})
    if(_user_${_b})
      set(_gpu_user_on ON)
    endif()
    if(NOT "${_user_${_b}}" STREQUAL "")
      set(_gpu_user_set ON)
    endif()
    if(_request STREQUAL _b AND _user_${_b} STREQUAL "OFF")
      message(FATAL_ERROR
        "SOMA_KOKKOS_GPU=${_b} conflicts with Kokkos_ENABLE_${_b}=OFF")
    endif()
  endforeach()
  set(_have_nvcc OFF)
  set(_have_hip OFF)
  if(_request STREQUAL "AUTO" AND NOT _gpu_user_set)
    _soma_gpu_probe(_have_nvcc _have_hip)
  endif()
  set(_gpu_args REQUEST ${_request} COMPILER "${CMAKE_CXX_COMPILER}"
    COMPILER_ID "${CMAKE_CXX_COMPILER_ID}")
  if(_have_nvcc)
    list(APPEND _gpu_args HAVE_NVCC)
  endif()
  if(_have_hip)
    list(APPEND _gpu_args HAVE_HIP)
  endif()
  if(_gpu_user_set)
    list(APPEND _gpu_args USER_SET)
  endif()
  _soma_gpu_decide(_gpu _gpu_why ${_gpu_args})
  message(STATUS "SOMA: Kokkos GPU backend request: ${_gpu} (${_gpu_why})")
  if(_gpu MATCHES "^(CUDA|HIP|SYCL)$")
    set(_lang_args)
    if(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
      list(APPEND _lang_args AS_LANGUAGE)
    endif()
    _soma_gpu_compiler_problem(_problem BACKEND ${_gpu} COMPILER "${CMAKE_CXX_COMPILER}"
      COMPILER_ID "${CMAKE_CXX_COMPILER_ID}" ${_lang_args})
    if(_problem)
      message(FATAL_ERROR
        "SOMA_FETCH_KOKKOS: ${_gpu} was selected (${_gpu_why}) but ${_problem}. "
        "For a CPU-only build pass -DSOMA_KOKKOS_GPU=NONE.")
    endif()
    _soma_kokkos_force(Kokkos_ENABLE_${_gpu} ON)
  endif()
  if((_gpu MATCHES "^(CUDA|HIP|SYCL)$" OR _gpu_user_on) AND NOT _user_arch)
    message(STATUS "SOMA: Kokkos detects the GPU architecture only when a GPU is "
      "visible; otherwise set the matching Kokkos_ARCH_* option.")
  endif()

  if(NOT _user_arch)
    _soma_kokkos_force(Kokkos_ARCH_NATIVE ON)
  endif()

  include("${PROJECT_SOURCE_DIR}/cmake/CPM.cmake")
  CPMAddPackage(NAME Kokkos GITHUB_REPOSITORY kokkos/kokkos
    GIT_TAG ${SOMA_KOKKOS_VERSION} GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL YES SYSTEM YES)
  if(NOT TARGET Kokkos::kokkos)
    add_library(Kokkos::kokkos ALIAS kokkos)
  endif()

  get_cmake_property(_vars VARIABLES)
  list(FILTER _vars INCLUDE REGEX "${_SOMA_KOKKOS_OPTION_REGEX}")
  list(REMOVE_ITEM _vars ${_user_names})
  set(_seen)
  foreach(_v IN LISTS _vars)
    list(APPEND _seen "${_v}=${${_v}}")
  endforeach()
  set(SOMA_KOKKOS_SEEN "${_seen}" CACHE INTERNAL "Kokkos options not set by the user" FORCE)

  get_directory_property(_version DIRECTORY "${Kokkos_SOURCE_DIR}" DEFINITION Kokkos_VERSION)
  set(_devices)
  foreach(_d SERIAL OPENMP THREADS HPX CUDA HIP SYCL)
    if(Kokkos_ENABLE_${_d})
      list(APPEND _devices ${_d})
    endif()
  endforeach()
  set(SOMA_KOKKOS_VERSION_FOUND "${_version}" PARENT_SCOPE)
  set(SOMA_KOKKOS_DEVICES "${_devices}" PARENT_SCOPE)
endfunction()

function(soma_find_or_fetch_kokkos)
  soma_seed_apple_openmp()
  find_package(Kokkos 5.0 QUIET)
  if(Kokkos_FOUND)
    set(_source installed)
    set(_version "${Kokkos_VERSION}")
    set(_devices "${Kokkos_DEVICES}")
    if(SOMA_FETCH_KOKKOS)
      message(STATUS "SOMA: using the installed Kokkos ${Kokkos_VERSION} "
        "(${Kokkos_DIR}); SOMA_FETCH_KOKKOS is ignored")
    endif()
    if(NOT "OPENMP" IN_LIST _devices AND NOT "THREADS" IN_LIST _devices
       AND NOT "HPX" IN_LIST _devices)
      message(WARNING
        "The installed Kokkos (${Kokkos_DIR}) has no parallel host backend, so SOMA "
        "runs host code serially. Rebuild it with Kokkos_ENABLE_OPENMP=ON or "
        "Kokkos_ENABLE_THREADS=ON. To have SOMA download and build its own Kokkos "
        "instead, hide the installed one (remove it from CMAKE_PREFIX_PATH and "
        "Kokkos_ROOT, or pass -DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON) and pass "
        "-DSOMA_FETCH_KOKKOS=ON.")
    endif()
  elseif(NOT SOMA_FETCH_KOKKOS)
    message(FATAL_ERROR
      "Kokkos >= 5.0 was not found. Point CMAKE_PREFIX_PATH or Kokkos_ROOT at an "
      "installed Kokkos, or configure with -DSOMA_FETCH_KOKKOS=ON to download and "
      "build it automatically.")
  else()
    set(_source fetched)
    _soma_fetch_kokkos()
    set(_version "${SOMA_KOKKOS_VERSION_FOUND}")
    set(_devices "${SOMA_KOKKOS_DEVICES}")
  endif()
  soma_patch_apple_openmp_target()

  set(_host Serial)
  foreach(_h OpenMP Threads HPX)
    string(TOUPPER ${_h} _uh)
    if(_uh IN_LIST _devices)
      set(_host ${_h})
    endif()
  endforeach()
  set(_gpu none)
  foreach(_g CUDA HIP SYCL)
    if(_g IN_LIST _devices)
      set(_gpu ${_g})
    endif()
  endforeach()
  set(_has_gpu OFF)
  if(NOT _gpu STREQUAL "none")
    set(_has_gpu ON)
  endif()

  set(SOMA_KOKKOS_SOURCE ${_source} PARENT_SCOPE)
  set(SOMA_KOKKOS_VERSION_FOUND "${_version}" PARENT_SCOPE)
  set(SOMA_KOKKOS_DEVICES "${_devices}" PARENT_SCOPE)
  set(SOMA_KOKKOS_HOST_BACKEND ${_host} PARENT_SCOPE)
  set(SOMA_KOKKOS_GPU_BACKEND ${_gpu} PARENT_SCOPE)
  set(SOMA_KOKKOS_HAS_GPU ${_has_gpu} PARENT_SCOPE)
endfunction()
