// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_COMPAT_DROPIN_MOAB_MOABCONFIG_H_
#define SOMA_COMPAT_DROPIN_MOAB_MOABCONFIG_H_

// Drop-in mode: what MOAB applications test of their MOAB build. MPI calls
// resolve to the MPI SOMA was built with, or else to a serial stand-in.

#define MOAB_VERSION "5.6.0"
#define MOAB_VERSION_MAJOR 5
#define MOAB_VERSION_MINOR 6
#define MOAB_VERSION_PATCH 0
#define MOAB_HAVE_MPI 1

#endif  // SOMA_COMPAT_DROPIN_MOAB_MOABCONFIG_H_
