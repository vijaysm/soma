// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "hilbert_reference.hpp"
#include "soma/core/sfc.hpp"
#include "soma/core/types.hpp"

using soma::Real;

namespace {

std::uint64_t index_2d(std::uint32_t x, std::uint32_t y, int bits) {
  const std::uint32_t c[2] = {x, y};
  return soma::sfc::hilbert_index(c, 2, bits);
}

std::uint64_t index_3d(std::uint32_t x, std::uint32_t y, std::uint32_t z, int bits) {
  const std::uint32_t c[3] = {x, y, z};
  return soma::sfc::hilbert_index(c, 3, bits);
}

int manhattan(const std::array<std::uint32_t, 3>& a, const std::array<std::uint32_t, 3>& b) {
  int d = 0;
  for (std::size_t i = 0; i < 3; ++i) {
    d += std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
  }
  return d;
}

}  // namespace

// The first-order curves, from the definition: the 2D curve is a U through the
// four cells, the 3D curve a path through the eight cells of a cube.
TEST(Hilbert, FirstOrderCurvesAreTheDefinition) {
  const std::vector<std::array<std::uint32_t, 2>> two{{0, 0}, {0, 1}, {1, 1}, {1, 0}};
  for (std::size_t k = 0; k < two.size(); ++k) {
    EXPECT_EQ(index_2d(two[k][0], two[k][1], 1), k);
  }
  const std::vector<std::array<std::uint32_t, 3>> three{{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0},
                                                         {1, 1, 0}, {1, 1, 1}, {1, 0, 1}, {1, 0, 0}};
  for (std::size_t k = 0; k < three.size(); ++k) {
    EXPECT_EQ(index_3d(three[k][0], three[k][1], three[k][2], 1), k);
  }
}

TEST(Hilbert, TwoDimensionsMatchTheIndependentReference) {
  for (int bits = 1; bits <= 6; ++bits) {
    const std::uint32_t n = std::uint32_t{1} << bits;
    for (std::uint32_t x = 0; x < n; ++x) {
      for (std::uint32_t y = 0; y < n; ++y) {
        ASSERT_EQ(index_2d(x, y, bits), soma_test::hilbert_reference_2d(n, x, y)) << "bits " << bits << " (" << x << "," << y << ")";
      }
    }
  }
}

// Two axes use all 32 bits; sampled cells, the corners among them, against
// the reference on the 2^32 x 2^32 grid.
TEST(Hilbert, TwoDimensionsMatchTheReferenceAtFullWidth) {
  ASSERT_EQ(soma::sfc::bits_per_axis(2), 32);
  const std::uint64_t n = std::uint64_t{1} << 32;
  const std::uint32_t m = soma::sfc::max_cell(2);
  EXPECT_EQ(m, 0xFFFFFFFFu);
  std::vector<std::uint32_t> samples{0U, 1U, 2U, m / 3, m / 2, m / 2 + 1, m - 1, m};
  std::uint64_t s = 0x2545F4914F6CDD1DULL;
  for (int i = 0; i < 64; ++i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    samples.push_back(static_cast<std::uint32_t>(s >> 32));
  }
  for (const std::uint32_t x : samples) {
    for (const std::uint32_t y : samples) {
      ASSERT_EQ(index_2d(x, y, 32), soma_test::hilbert_reference_2d(n, x, y))
          << "(" << x << "," << y << ")";
    }
  }
  EXPECT_EQ(index_2d(m, 0, 32), ~std::uint64_t{0});
}

TEST(Hilbert, KnownValuesOnAnEightByEightGrid) {
  EXPECT_EQ(index_2d(0, 0, 3), 0u);
  EXPECT_EQ(index_2d(7, 0, 3), 63u);
  EXPECT_EQ(index_2d(0, 7, 3), 21u);
  EXPECT_EQ(index_2d(7, 7, 3), 42u);
  EXPECT_EQ(index_2d(3, 4, 3), 31u);
  EXPECT_EQ(index_2d(4, 4, 3), 32u);
  EXPECT_EQ(index_2d(1, 0, 3), 3u);
  EXPECT_EQ(index_2d(0, 1, 3), 1u);
}

TEST(Hilbert, IsABijectionOnTheGrid) {
  for (int bits = 1; bits <= 4; ++bits) {
    const std::uint32_t n = std::uint32_t{1} << bits;
    std::vector<char> seen2(static_cast<std::size_t>(n) * n, 0);
    for (std::uint32_t x = 0; x < n; ++x) {
      for (std::uint32_t y = 0; y < n; ++y) {
        const std::uint64_t k = index_2d(x, y, bits);
        ASSERT_LT(k, static_cast<std::uint64_t>(n) * n);
        ASSERT_EQ(seen2[k], 0) << "2D key " << k << " repeated, bits " << bits;
        seen2[k] = 1;
      }
    }
    std::vector<char> seen3(static_cast<std::size_t>(n) * n * n, 0);
    for (std::uint32_t x = 0; x < n; ++x) {
      for (std::uint32_t y = 0; y < n; ++y) {
        for (std::uint32_t z = 0; z < n; ++z) {
          const std::uint64_t k = index_3d(x, y, z, bits);
          ASSERT_LT(k, static_cast<std::uint64_t>(n) * n * n);
          ASSERT_EQ(seen3[k], 0) << "3D key " << k << " repeated, bits " << bits;
          seen3[k] = 1;
        }
      }
    }
  }
}

TEST(Hilbert, ConsecutiveKeysAreAdjacentCells) {
  for (int bits = 1; bits <= 4; ++bits) {
    const std::uint32_t n = std::uint32_t{1} << bits;
    std::vector<std::array<std::uint32_t, 3>> by2(static_cast<std::size_t>(n) * n);
    for (std::uint32_t x = 0; x < n; ++x) {
      for (std::uint32_t y = 0; y < n; ++y) by2[index_2d(x, y, bits)] = {x, y, 0};
    }
    for (std::size_t k = 1; k < by2.size(); ++k) {
      ASSERT_EQ(manhattan(by2[k - 1], by2[k]), 1) << "2D, bits " << bits << ", key " << k;
    }
    std::vector<std::array<std::uint32_t, 3>> by3(static_cast<std::size_t>(n) * n * n);
    for (std::uint32_t x = 0; x < n; ++x) {
      for (std::uint32_t y = 0; y < n; ++y) {
        for (std::uint32_t z = 0; z < n; ++z) by3[index_3d(x, y, z, bits)] = {x, y, z};
      }
    }
    for (std::size_t k = 1; k < by3.size(); ++k) {
      ASSERT_EQ(manhattan(by3[k - 1], by3[k]), 1) << "3D, bits " << bits << ", key " << k;
    }
    EXPECT_EQ((by3[0]), (std::array<std::uint32_t, 3>{0, 0, 0}));
  }
}

TEST(Hilbert, OneDimensionIsTheIdentity) {
  for (std::uint32_t x = 0; x < 64; ++x) {
    const std::uint32_t c[1] = {x};
    EXPECT_EQ(soma::sfc::hilbert_index(c, 1, 6), x);
  }
  for (const std::uint32_t x : {0U, 1U, 0x80000000U, 0xFFFFFFFFU}) {
    const std::uint32_t c[1] = {x};
    EXPECT_EQ(soma::sfc::hilbert_index(c, 1, soma::sfc::bits_per_axis(1)), x);
  }
}

TEST(Hilbert, FullWidthKeysUseAllSixtyThreeBits) {
  ASSERT_EQ(soma::sfc::bits_per_axis(3), 21);
  ASSERT_EQ(soma::sfc::bits_per_axis(1), 32);
  const std::uint32_t m = soma::sfc::max_cell(3);
  std::uint64_t max_key = 0;
  for (const std::uint32_t x : {0U, 1U, m / 2, m - 1, m}) {
    for (const std::uint32_t y : {0U, 1U, m / 2, m - 1, m}) {
      for (const std::uint32_t z : {0U, 1U, m / 2, m - 1, m}) {
        max_key = std::max(max_key, index_3d(x, y, z, soma::sfc::bits_per_axis(3)));
      }
    }
  }
  EXPECT_LT(max_key, std::uint64_t{1} << 63);
  EXPECT_GT(max_key, std::uint64_t{1} << 61);
  EXPECT_EQ(index_3d(m, 0, 0, soma::sfc::bits_per_axis(3)), (std::uint64_t{1} << 63) - 1);
}

TEST(HilbertKey, QuantizesTheBoundingBox) {
  const Real lo[3] = {-1.0, 0.0, 5.0};
  const Real hi[3] = {1.0, 4.0, 5.0};
  const auto box = soma::sfc::KeyBox::make(lo, hi);
  const std::uint32_t last = soma::sfc::max_cell(2);
  EXPECT_EQ(box.n, 2);
  EXPECT_EQ(box.axis[0], 0);
  EXPECT_EQ(box.axis[1], 1);
  EXPECT_EQ(soma::sfc::hilbert_key(box, -1.0, 0.0, 5.0), 0u);
  EXPECT_EQ(soma::sfc::quantize(-1.0, -1.0, 0.5, last), 0u);
  EXPECT_EQ(soma::sfc::quantize(1.0, -1.0, 0.5, last), last);
  EXPECT_EQ(soma::sfc::quantize(5.0, -1.0, 0.5, last), last);
  EXPECT_EQ(soma::sfc::quantize(-7.0, -1.0, 0.5, last), 0u);
  EXPECT_GT(soma::sfc::quantize(0.0, -1.0, 0.5, last), 0u);
  EXPECT_LT(soma::sfc::quantize(0.0, -1.0, 0.5, last), last);
  const auto a = soma::sfc::hilbert_key(box, 0.25, 1.0, 5.0);
  EXPECT_EQ(a, soma::sfc::hilbert_key(box, 0.25, 1.0, 99.0));
  // A box that is flat in y keeps x and z: the curve runs over axes 0 and 2.
  const Real lo_xz[3] = {0.0, 5.0, 0.0};
  const Real hi_xz[3] = {1.0, 5.0, 1.0};
  const auto xz = soma::sfc::KeyBox::make(lo_xz, hi_xz);
  EXPECT_EQ(xz.n, 2);
  EXPECT_EQ(xz.axis[1], 2);
  const auto k = soma::sfc::hilbert_key(xz, 0.75, 5.0, 0.25);
  EXPECT_EQ(k, soma::sfc::hilbert_key(xz, 0.75, -3.0, 0.25));
  EXPECT_NE(k, soma::sfc::hilbert_key(xz, 0.75, 5.0, 0.75));
  const std::uint32_t cell[2] = {soma::sfc::quantize(0.75, 0.0, 1.0, last),
                                 soma::sfc::quantize(0.25, 0.0, 1.0, last)};
  EXPECT_EQ(cell[0], 3221225471u);
  EXPECT_EQ(cell[1], 1073741823u);
  EXPECT_EQ(k, soma_test::hilbert_reference_2d(std::uint64_t{1} << 32, cell[0], cell[1]));
  const Real pt[3] = {2.0, 2.0, 2.0};
  EXPECT_EQ(soma::sfc::hilbert_key(soma::sfc::KeyBox::make(pt, pt), 2.0, 3.0, 4.0), 0u);
}

namespace soma_test {

KOKKOS_INLINE_FUNCTION std::uint64_t key_of_cell(int k) {
  const std::uint32_t c[3] = {static_cast<std::uint32_t>(k % 8),
                              static_cast<std::uint32_t>((k / 8) % 8),
                              static_cast<std::uint32_t>(k / 64)};
  return soma::sfc::hilbert_index(c, 3, 3);
}

Kokkos::View<std::uint64_t*, Kokkos::HostSpace> keys_from_default_space() {
  Kokkos::View<std::uint64_t*> keys("soma::test::hilbert_keys", 512);
  Kokkos::parallel_for(
      "soma::test::hilbert_keys", 512, KOKKOS_LAMBDA(const int k) { keys(k) = key_of_cell(k); });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, keys);
}

KOKKOS_INLINE_FUNCTION std::uint64_t box_key_of(int k, soma::sfc::KeyBox box) {
  return soma::sfc::hilbert_key(box, 0.1 * (k % 7), 0.2 * (k % 5), 0.3 * (k % 3));
}

Kokkos::View<std::uint64_t*, Kokkos::HostSpace> box_keys_from_default_space(soma::sfc::KeyBox box) {
  Kokkos::View<std::uint64_t*> keys("soma::test::box_keys", 105);
  Kokkos::parallel_for(
      "soma::test::box_keys", 105, KOKKOS_LAMBDA(const int k) { keys(k) = box_key_of(k, box); });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, keys);
}

}  // namespace soma_test

TEST(Hilbert, DeviceEvaluationMatchesHost) {
  const auto device = soma_test::keys_from_default_space();
  for (int k = 0; k < 512; ++k) EXPECT_EQ(device(k), soma_test::key_of_cell(k)) << "cell " << k;
  const Real lo[3] = {0.0, 0.0, 0.0};
  const Real hi[3] = {0.6, 0.8, 0.6};
  const auto box = soma::sfc::KeyBox::make(lo, hi);
  const auto boxed = soma_test::box_keys_from_default_space(box);
  for (int k = 0; k < 105; ++k) EXPECT_EQ(boxed(k), soma_test::box_key_of(k, box)) << "point " << k;
}
