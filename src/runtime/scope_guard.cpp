// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/runtime/scope_guard.hpp"

#include <Kokkos_Core.hpp>

#include "soma/config.hpp"

#if SOMA_ENABLE_MPI
#include <mpi.h>
#endif

namespace soma {

ScopeGuard::ScopeGuard(int& argc, char**& argv) {
#if SOMA_ENABLE_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided = 0;
    // Kokkos host threads never call MPI, so FUNNELED suffices.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    owns_mpi_ = true;
  }
#endif
  // Under MPI, Kokkos maps one device per node-local rank from the launcher's
  // environment; no explicit binding is needed here.
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize(argc, argv);
    owns_kokkos_ = true;
  }
}

ScopeGuard::~ScopeGuard() {
  if (owns_kokkos_ && !Kokkos::is_finalized()) {
    Kokkos::finalize();
  }
#if SOMA_ENABLE_MPI
  if (owns_mpi_) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }
#endif
}

}  // namespace soma
