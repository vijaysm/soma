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
