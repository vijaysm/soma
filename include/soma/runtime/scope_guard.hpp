// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_RUNTIME_SCOPE_GUARD_HPP_
#define SOMA_RUNTIME_SCOPE_GUARD_HPP_

/// \file
/// Starts what SOMA needs and stops what it started: reads SOMA's options from
/// the environment and the command line (parse_options, removing its own
/// arguments), then initializes MPI (in MPI builds) and Kokkos unless they run
/// already.

namespace soma {

/// Owns runtime lifetime: MPI (when enabled), then Kokkos; torn down in
/// reverse. Leaves alone anything the application already initialized.
///
/// Kokkos forbids constructing a View before initialize(), so every SOMA
/// program -- tests and benchmarks included -- starts with one of these.
///
/// Kokkos removes its own command-line options from argv, so any argument
/// parser (GoogleTest, for one) must run after the guard is constructed.
class [[nodiscard]] ScopeGuard {
 public:
  ScopeGuard(int& argc, char**& argv);
  ~ScopeGuard();

  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;

 private:
  [[maybe_unused]] bool owns_mpi_ = false;  ///< only set in MPI builds
  bool owns_kokkos_ = false;
};

}  // namespace soma

#endif  // SOMA_RUNTIME_SCOPE_GUARD_HPP_
