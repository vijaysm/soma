// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// GridIndex: the pairs it finds are those a brute-force search finds, each
// once and in order, for boxes of every shape (flat ones included), with and
// without a tolerance; bad input throws.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/grid_index.hpp"

using soma::LO;
using soma::Offset;
using soma::Real;
using Index = soma::GridIndex<Kokkos::HostSpace>;
using Boxes = soma::View2<Real, Kokkos::HostSpace>;

namespace {

// n boxes in [0, 1]^3 (z flat when `flat`), sizes up to `size`.
Boxes random_boxes(std::size_t n, Real size, bool flat, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> at(0, 1), extent(0, size);
  Boxes b("boxes", n, 6);
  for (std::size_t i = 0; i < n; ++i) {
    for (int a = 0; a < 3; ++a) {
      const Real lo = flat && a == 2 ? 0.5 : at(gen);
      b(i, a) = lo;
      b(i, a + 3) = flat && a == 2 ? 0.5 : lo + extent(gen);
    }
  }
  return b;
}

std::vector<std::vector<LO>> brute(const Boxes& boxes, const Boxes& queries, Real tol) {
  std::vector<std::vector<LO>> out(queries.extent(0));
  for (std::size_t q = 0; q < queries.extent(0); ++q) {
    for (std::size_t x = 0; x < boxes.extent(0); ++x) {
      bool meets = true;
      for (int a = 0; a < 3; ++a) {
        meets = meets && queries(q, a) <= boxes(x, a + 3) + tol && boxes(x, a) <= queries(q, a + 3) + tol;
      }
      if (meets) out[q].push_back(static_cast<LO>(x));
    }
  }
  return out;
}

void expect_brute(const Boxes& boxes, const Boxes& queries, Real tol) {
  const Index index(boxes, tol);
  const auto pairs = index.query(queries);
  const auto want = brute(boxes, queries, tol);
  ASSERT_EQ(pairs.offsets.extent(0), queries.extent(0) + 1);
  std::size_t differ = 0;
  for (std::size_t q = 0; q < queries.extent(0); ++q) {
    const std::vector<LO> got(pairs.boxes.data() + pairs.offsets(q), pairs.boxes.data() + pairs.offsets(q + 1));
    if (got != want[q] && differ++ == 0) ADD_FAILURE() << "query " << q << ": " << got.size() << " vs " << want[q].size();
  }
  EXPECT_EQ(differ, 0u);
}

}  // namespace

TEST(GridIndex, FindsWhatBruteForceFinds) {
  for (const bool flat : {false, true}) {
    for (const Real size : {0.01, 0.1, 0.5}) {
      for (const Real tol : {0.0, 1e-3}) {
        SCOPED_TRACE(testing::Message() << "flat " << flat << " size " << size << " tol " << tol);
        expect_brute(random_boxes(800, size, flat, 7), random_boxes(300, size, flat, 11), tol);
      }
    }
  }
}

// Boxes that only touch meet; a tolerance reaches across a gap no wider.
TEST(GridIndex, TouchingBoxesMeetAndTheToleranceSpansAGap) {
  Boxes grid("grid", 16, 6);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const std::size_t b = static_cast<std::size_t>(4 * i + j);
      grid(b, 0) = i * 0.25;
      grid(b, 1) = j * 0.25;
      grid(b, 3) = (i + 1) * 0.25;
      grid(b, 4) = (j + 1) * 0.25;
    }
  }
  expect_brute(grid, grid, 0);
  const auto self = Index(grid).query(grid);
  EXPECT_EQ(self.offsets(5 + 1) - self.offsets(5), 9);  // an interior box: itself and 8 around it
  Boxes apart("apart", 1, 6);
  apart(0, 0) = 1 + 1e-9;
  apart(0, 3) = 2;
  apart(0, 4) = 1;
  EXPECT_EQ(Index(grid).query(apart).boxes.extent(0), 0u);
  EXPECT_EQ(Index(grid, 2e-9).query(apart).boxes.extent(0), 4u);
}

// Few bins for few boxes; one across a flat axis; at most 2n in all.
TEST(GridIndex, BinsFollowTheBoxes) {
  const Index flat(random_boxes(1000, 0.05, true, 3));
  EXPECT_EQ(flat.bins()[2], 1);
  const auto b = Index(random_boxes(1000, 0.001, false, 5)).bins();
  EXPECT_LE(static_cast<long>(b[0]) * b[1] * b[2], 2000);
  EXPECT_GE(b[0], 2);
}

// Tiny boxes scattered over a plane: the bins stay within the budget.
TEST(GridIndex, SparseBoxesStayWithinTheBinBudget) {
  for (const std::size_t n : {1000u, 20000u}) {
    const auto b = Index(random_boxes(n, 1e-6, true, 13)).bins();
    EXPECT_LE(static_cast<double>(b[0]) * b[1] * b[2], 2.0 * static_cast<double>(n));
    EXPECT_EQ(b[2], 1);
  }
}

TEST(GridIndex, EmptyInputsAndBadBoxes) {
  const Boxes none("none", 0, 6);
  const auto pairs = Index(none).query(random_boxes(5, 0.1, false, 1));
  EXPECT_EQ(pairs.boxes.extent(0), 0u);
  EXPECT_EQ(Index(random_boxes(5, 0.1, false, 1)).query(none).offsets.extent(0), 1u);
  Boxes bad = random_boxes(4, 0.1, false, 2);
  bad(2, 1) = bad(2, 4) + 1;  // low above high
  EXPECT_THROW(Index{bad}, std::invalid_argument);
  bad(2, 1) = std::numeric_limits<Real>::quiet_NaN();
  EXPECT_THROW(Index{bad}, std::invalid_argument);
  EXPECT_THROW(Index(random_boxes(4, 0.1, false, 2), -1), std::invalid_argument);
  EXPECT_THROW(Index(Boxes("five", 3, 5)), std::invalid_argument);
}
