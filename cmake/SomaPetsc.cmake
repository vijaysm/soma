# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# PETSc through its pkg-config file, under PETSC_DIR (an install prefix) or
# PETSC_DIR/PETSC_ARCH (an in-place build); either may come from the
# environment. Defines the imported target PkgConfig::PETSC. Needs PETSc
# 3.19 or later (COO assembly, PETSC_SUCCESS), real double scalars, and shared
# libraries: pkg-config's Libs.private, which a static PETSc needs, are not
# linked.

if(NOT PETSC_DIR AND DEFINED ENV{PETSC_DIR})
  set(PETSC_DIR "$ENV{PETSC_DIR}")
endif()
if(NOT PETSC_ARCH AND DEFINED ENV{PETSC_ARCH})
  set(PETSC_ARCH "$ENV{PETSC_ARCH}")
endif()
if(NOT PETSC_DIR)
  message(FATAL_ERROR "SOMA_ENABLE_PETSC needs PETSC_DIR (and PETSC_ARCH for an in-place build)")
endif()
set(PETSC_DIR "${PETSC_DIR}" CACHE PATH "PETSc prefix or source tree")
set(PETSC_ARCH "${PETSC_ARCH}" CACHE STRING "PETSc build directory under PETSC_DIR, if any")

find_package(PkgConfig REQUIRED)
set(_soma_petsc_pc "${PETSC_DIR}/lib/pkgconfig")
if(PETSC_ARCH)
  set(_soma_petsc_pc "${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig")
endif()
# pkg_check_modules caches its result whatever the search path; look again
# when PETSC_DIR or PETSC_ARCH names another PETSc.
if(NOT _soma_petsc_pc STREQUAL SOMA_PETSC_PKGCONFIG_DIR)
  unset(__pkg_config_checked_PETSC CACHE)
  unset(SOMA_PETSC_REAL_DOUBLE CACHE)
endif()
set(_soma_pkg_config_path "$ENV{PKG_CONFIG_PATH}")
set(ENV{PKG_CONFIG_PATH} "${_soma_petsc_pc}:$ENV{PKG_CONFIG_PATH}")
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc>=3.19)
set(ENV{PKG_CONFIG_PATH} "${_soma_pkg_config_path}")
set(SOMA_PETSC_PKGCONFIG_DIR "${_soma_petsc_pc}" CACHE INTERNAL "The pkg-config directory of the PETSc found")

# Real double scalars: SOMA's values pass to PETSc unconverted.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES PkgConfig::PETSC MPI::MPI_CXX)
set(CMAKE_REQUIRED_QUIET ON)
check_cxx_source_compiles("
#include <petscsys.h>
#if !defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_COMPLEX)
#error not real double
#endif
int main() { return 0; }" SOMA_PETSC_REAL_DOUBLE)
unset(CMAKE_REQUIRED_LIBRARIES)
unset(CMAKE_REQUIRED_QUIET)
if(NOT SOMA_PETSC_REAL_DOUBLE)
  message(FATAL_ERROR "The PETSc under ${_soma_petsc_pc} does not have real double scalars, which SOMA needs")
endif()
