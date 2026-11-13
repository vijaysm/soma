// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <bit>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "soma/core/hash3d_exact.hpp"
#include "soma/core/point_key.hpp"

using soma::Real;

namespace {

// Signed zeros, subnormals, extremes, infinities, and random bit patterns
// (NaN ones left out).
std::vector<Real> values() {
  constexpr Real inf = std::numeric_limits<Real>::infinity();
  std::vector<Real> v{0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1.0, -1.0, 0.1,
                      std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max(), inf, -inf};
  std::mt19937_64 gen(17);
  while (v.size() < 20000) {
    const Real x = std::bit_cast<Real>(gen());
    if (x == x) v.push_back(x);
  }
  return v;
}

// Keys of `v` computed on the default execution space.
std::vector<std::uint64_t> keys_on_device(const std::vector<Real>& v, bool& nan) {
  Kokkos::View<Real*> x("x", v.size());
  Kokkos::View<std::uint64_t*> k("k", v.size());
  Kokkos::deep_copy(x, Kokkos::View<const Real*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(v.data(), v.size()));
  nan = soma::point_keys(Kokkos::DefaultExecutionSpace{}, x, k);
  std::vector<std::uint64_t> out(v.size());
  Kokkos::deep_copy(Kokkos::View<std::uint64_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(out.data(), out.size()), k);
  return out;
}

}  // namespace

TEST(PointKey, EqualsTheHashersKeyAndInvertsIt) {
  std::size_t wrong = 0;
  for (const Real x : values()) {
    const std::uint64_t k = soma::point_key(x);
    wrong += k == spatial_exact::key_of(x) ? 0 : 1;
    wrong += std::bit_cast<std::uint64_t>(soma::point_of_key(k)) == std::bit_cast<std::uint64_t>(spatial_exact::value_of(k)) ? 0 : 1;
  }
  EXPECT_EQ(wrong, 0U);
  EXPECT_EQ(soma::point_key(-0.0), soma::point_key(0.0));
  EXPECT_LT(soma::point_key(-1.0), soma::point_key(-0.5));
  EXPECT_LT(soma::point_key(0.5), soma::point_key(std::numeric_limits<Real>::infinity()));
}

TEST(PointKey, KeysOnTheDefaultSpaceEqualTheHost) {
  const std::vector<Real> v = values();
  bool nan = true;
  const std::vector<std::uint64_t> got = keys_on_device(v, nan);
  EXPECT_FALSE(nan);
  std::size_t wrong = 0;
  for (std::size_t i = 0; i < v.size(); ++i) wrong += got[i] == spatial_exact::key_of(v[i]) ? 0 : 1;
  EXPECT_EQ(wrong, 0U);
  std::vector<Real> with_nan{1.0, std::numeric_limits<Real>::quiet_NaN()};
  (void)keys_on_device(with_nan, nan);
  EXPECT_TRUE(nan);
}
