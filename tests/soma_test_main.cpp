// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Shared GoogleTest entry point. Owns the runtime scope so no test can
// construct a Kokkos::View before Kokkos::initialize(), and so a failure on any
// rank fails the run on every rank.

#include <gtest/gtest.h>

#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"

int main(int argc, char** argv) {
  // Kokkos strips its own options from argv, so the guard comes first.
  const soma::ScopeGuard guard(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  rc = soma::Comm::world().allreduce(rc, soma::Op::max);
  return rc;
}
