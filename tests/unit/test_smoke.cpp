// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <cstdio>
#include <type_traits>

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

TEST(Smoke, DefaultExecutionSpaceIsTheMostParallelBackend) {
  using Default = Kokkos::DefaultExecutionSpace;
  using Host = Kokkos::DefaultHostExecutionSpace;
#if defined(KOKKOS_ENABLE_CUDA)
  EXPECT_TRUE((std::is_same_v<Default, Kokkos::Cuda>));
#elif defined(KOKKOS_ENABLE_HIP)
  EXPECT_TRUE((std::is_same_v<Default, Kokkos::HIP>));
#elif defined(KOKKOS_ENABLE_SYCL)
  EXPECT_TRUE((std::is_same_v<Default, Kokkos::SYCL>));
#else
  EXPECT_TRUE((std::is_same_v<Default, Host>));
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  EXPECT_TRUE((std::is_same_v<Host, Kokkos::OpenMP>));
#elif defined(KOKKOS_ENABLE_THREADS)
  EXPECT_TRUE((std::is_same_v<Host, Kokkos::Threads>));
#else
  EXPECT_TRUE((std::is_same_v<Host, Kokkos::Serial>));
#endif
  std::printf("default execution space: %s, concurrency %d; host: %s, concurrency %d\n",
              Default::name(), static_cast<int>(Default().concurrency()), Host::name(),
              static_cast<int>(Host().concurrency()));
}
