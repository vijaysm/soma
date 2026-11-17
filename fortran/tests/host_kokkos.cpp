// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// For test_stop: Kokkos initialized by the host program, not by SOMA.

#include <Kokkos_Core.hpp>

extern "C" void test_host_kokkos_initialize() {
  Kokkos::initialize(Kokkos::InitializationSettings().set_disable_warnings(true));
}
