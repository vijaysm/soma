# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Finds HDF5 for the .h5m reader. Point HDF5_ROOT at the installation.
# A parallel HDF5 links MPI, so it is accepted only together with
# SOMA_ENABLE_MPI.
# FindHDF5 probes the C compiler wrapper, which needs the C language.
enable_language(C)
find_package(HDF5 REQUIRED COMPONENTS C)
if(NOT TARGET HDF5::HDF5)
  message(FATAL_ERROR "SOMA_ENABLE_HDF5=ON but FindHDF5 provided no HDF5::HDF5 target")
endif()
if(HDF5_IS_PARALLEL AND NOT SOMA_ENABLE_MPI)
  message(FATAL_ERROR
    "SOMA_ENABLE_HDF5=ON found a parallel HDF5 (${HDF5_C_INCLUDE_DIRS}), which needs MPI. "
    "Configure with -DSOMA_ENABLE_MPI=ON, or set HDF5_ROOT to a serial HDF5.")
endif()
if(HDF5_VERSION VERSION_LESS 1.10)
  message(FATAL_ERROR "SOMA_ENABLE_HDF5=ON needs HDF5 1.10 or newer (found ${HDF5_VERSION})")
endif()
# Recorded in the package config so consumers find the same HDF5; only a root
# the user gave, since one guessed from the include directory can be wrong.
if(HDF5_ROOT)
  set(SOMA_HDF5_ROOT "${HDF5_ROOT}")
elseif(DEFINED ENV{HDF5_ROOT})
  set(SOMA_HDF5_ROOT "$ENV{HDF5_ROOT}")
else()
  set(SOMA_HDF5_ROOT "")
endif()
