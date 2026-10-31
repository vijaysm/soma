// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

namespace soma_test {

std::int64_t triangular_sum_on_default_space(std::int64_t n) {
  std::int64_t total = 0;
  Kokkos::parallel_reduce(
      "soma::test::triangular", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const std::int64_t i, std::int64_t& acc) { acc += i; }, total);
  return total;
}

}  // namespace soma_test

TEST(Smoke, KokkosIsInitialized) { EXPECT_TRUE(Kokkos::is_initialized()); }

TEST(Smoke, ParallelReduceOnDefaultExecutionSpace) {
  const std::int64_t n = std::int64_t{1} << 20;
  EXPECT_EQ(soma_test::triangular_sum_on_default_space(n), n * (n - 1) / 2);
}
