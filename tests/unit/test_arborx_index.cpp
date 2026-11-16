// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// ArborXIndex: the pairs GridIndex finds, in its order, for boxes of every
// shape, with and without a tolerance, far from the origin and at the edge
// of the tolerance; a query that is not finite meets nothing; bad input
// throws.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/arborx_index.hpp"
#include "soma/algorithms/grid_index.hpp"

using soma::Real;
using Grid = soma::GridIndex<Kokkos::HostSpace>;
using Tree = soma::ArborXIndex<Kokkos::HostSpace>;
using Boxes = soma::View2<Real, Kokkos::HostSpace>;

namespace {

// n boxes in [at, at + 1]^3 (z flat when `flat`), sizes up to `size`.
Boxes random_boxes(std::size_t n, Real size, bool flat, std::uint32_t seed, Real at = 0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> where(0, 1), extent(0, size);
  Boxes b("boxes", n, 6);
  for (std::size_t i = 0; i < n; ++i) {
    for (int a = 0; a < 3; ++a) {
      const Real lo = at + (flat && a == 2 ? 0.5 : where(gen));
      b(i, a) = lo;
      b(i, a + 3) = flat && a == 2 ? lo : lo + extent(gen);
    }
  }
  return b;
}

// The pairs of both, the same entry for entry.
void expect_grid(const Boxes& boxes, const Boxes& queries, Real tol) {
  const auto want = Grid(boxes, tol).query(queries);
  const auto got = Tree(boxes, tol).query(queries);
  ASSERT_EQ(got.offsets.extent(0), want.offsets.extent(0));
  ASSERT_EQ(got.boxes.extent(0), want.boxes.extent(0));
  std::size_t differ = 0;
  for (std::size_t q = 0; q < want.offsets.extent(0); ++q) differ += got.offsets(q) == want.offsets(q) ? 0 : 1;
  for (std::size_t k = 0; k < want.boxes.extent(0); ++k) differ += got.boxes(k) == want.boxes(k) ? 0 : 1;
  EXPECT_EQ(differ, 0u);
}

}  // namespace

TEST(ArborXIndex, FindsWhatGridIndexFinds) {
  for (const bool flat : {false, true}) {
    for (const Real size : {0.01, 0.1, 0.5}) {
      for (const Real tol : {0.0, 1e-3}) {
        SCOPED_TRACE(testing::Message() << "flat " << flat << " size " << size << " tol " << tol);
        expect_grid(random_boxes(800, size, flat, 7), random_boxes(300, size, flat, 11), tol);
      }
    }
  }
  // Far from the origin, a tolerance below the coordinates' spacing.
  expect_grid(random_boxes(800, 0.1, false, 3, 1e7), random_boxes(300, 0.1, false, 5, 1e7), 1e-12);
}

// A query that reaches a box by exactly the tolerance meets it; one a unit
// in the last place further does not; boxes that touch meet.
TEST(ArborXIndex, TheToleranceIsKeptToTheLastPlace) {
  Boxes box("box", 1, 6), q("q", 4, 6);
  for (int a = 0; a < 3; ++a) {
    box(0, a) = 0;
    box(0, a + 3) = 0.5;
    for (std::size_t i = 0; i < 4; ++i) {
      q(i, a) = 0.25;
      q(i, a + 3) = 0.25;
    }
  }
  q(0, 0) = 0.75;  // 0.5 + 0.25: exactly at the tolerance
  q(1, 0) = std::nextafter(0.75, 1.0);
  q(2, 0) = 0.5;  // touching, no tolerance needed
  q(2, 3) = 0.6;
  q(3, 3) = -0.25;  // below, by exactly the tolerance
  q(3, 0) = -0.5;
  for (const Real tol : {0.25, 0.0}) {
    SCOPED_TRACE(testing::Message() << "tol " << tol);
    expect_grid(box, q, tol);
  }
  const auto pairs = Tree(box, 0.25).query(q);
  EXPECT_EQ(pairs.offsets(1) - pairs.offsets(0), 1);
  EXPECT_EQ(pairs.offsets(2) - pairs.offsets(1), 0);
  EXPECT_EQ(pairs.offsets(3) - pairs.offsets(2), 1);
  EXPECT_EQ(pairs.offsets(4) - pairs.offsets(3), 1);
}

TEST(ArborXIndex, EmptyInputsBadBoxesAndQueriesThatAreNotFinite) {
  const Boxes none("none", 0, 6);
  EXPECT_EQ(Tree(none).query(random_boxes(5, 0.1, false, 1)).boxes.extent(0), 0u);
  EXPECT_EQ(Tree(random_boxes(5, 0.1, false, 1)).query(none).offsets.extent(0), 1u);
  Boxes nan = random_boxes(3, 0.5, false, 4);
  nan(1, 2) = std::numeric_limits<Real>::quiet_NaN();
  const auto pairs = Tree(random_boxes(50, 0.5, false, 6)).query(nan);
  EXPECT_EQ(pairs.offsets(2) - pairs.offsets(1), 0);
  EXPECT_GT(pairs.offsets(1) - pairs.offsets(0), 0);
  Boxes bad = random_boxes(4, 0.1, false, 2);
  bad(2, 1) = bad(2, 4) + 1;  // low above high
  EXPECT_THROW(Tree{bad}, std::invalid_argument);
  bad(2, 1) = std::numeric_limits<Real>::quiet_NaN();
  EXPECT_THROW(Tree{bad}, std::invalid_argument);
  EXPECT_THROW(Tree(random_boxes(4, 0.1, false, 2), -1), std::invalid_argument);
  EXPECT_THROW(Tree(Boxes("five", 3, 5)), std::invalid_argument);
}
