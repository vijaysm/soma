// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_RUNTIME_SCOPE_GUARD_HPP_
#define SOMA_RUNTIME_SCOPE_GUARD_HPP_

namespace soma {

// Owns runtime lifetime: MPI (when enabled), then Kokkos; torn down in
// reverse. Leaves alone anything the application already initialized.
//
// Kokkos forbids constructing a View before initialize(), so every SOMA
// program -- tests and benchmarks included -- starts with one of these.
class ScopeGuard {
 public:
  ScopeGuard(int& argc, char**& argv);
  ~ScopeGuard();

  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;

 private:
  [[maybe_unused]] bool owns_mpi_ = false;  // only set in MPI builds
  bool owns_kokkos_ = false;
};

}  // namespace soma

#endif  // SOMA_RUNTIME_SCOPE_GUARD_HPP_
