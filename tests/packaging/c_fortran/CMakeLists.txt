# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# A C and a Fortran program built against an installed SOMA's C API and
# Fortran module, through find_package only.

cmake_minimum_required(VERSION 3.24)
project(soma_c_fortran_consumer LANGUAGES C CXX Fortran)

find_package(soma REQUIRED COMPONENTS c fortran)

add_executable(consumer_c main.c)
target_link_libraries(consumer_c PRIVATE soma::soma_c)

add_executable(consumer_fortran main.F90)
target_link_libraries(consumer_fortran PRIVATE soma::soma_fortran)
if(SOMA_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS Fortran)
  target_link_libraries(consumer_fortran PRIVATE MPI::MPI_Fortran)
  target_compile_definitions(consumer_fortran PRIVATE HAVE_MPI=1)
endif()
