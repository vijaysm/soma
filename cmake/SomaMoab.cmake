# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Sets SOMA_MOAB_TARGET for the MOAB reference tests and benchmarks.
find_package(MOAB REQUIRED)
# MOAB's MOAB_LIBRARIES string breaks on CMake >= 4 ("-framework Accelerate"
# is split); always link the imported target.
if(TARGET MOAB::MOAB)
  set(SOMA_MOAB_TARGET MOAB::MOAB)
elseif(TARGET MOAB)
  set(SOMA_MOAB_TARGET MOAB)
else()
  message(FATAL_ERROR "SOMA_ENABLE_MOAB_REFERENCE=ON but MOAB exports no imported target.")
endif()
# MOAB's target links the HDF5 imported targets by name (hdf5-shared,
# hdf5_hl-shared) but loads hdf5-config.cmake from one hard-coded path only.
# When that misses, load the config from the HDF5_DIR MOAB recorded.
if(MOAB_USE_HDF5 AND NOT TARGET hdf5-shared AND NOT TARGET hdf5-static)
  find_package(HDF5 CONFIG QUIET HINTS ${HDF5_DIR})
  if(NOT TARGET hdf5-shared AND NOT TARGET hdf5-static)
    message(FATAL_ERROR
      "SOMA_ENABLE_MOAB_REFERENCE=ON: MOAB (${MOAB_DIR}) was built with HDF5 but no HDF5 "
      "imported target was found. Set HDF5_DIR to the directory with hdf5-config.cmake.")
  endif()
endif()
