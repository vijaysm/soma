// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Berger–Rigoutsos clustering: every tag covered by disjoint, aligned boxes in
// the domain, each efficient enough or a single block, the same whatever the
// order of the tags; known shapes split where they should.

#include <algorithm>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "soma/amr/box_list.hpp"
#include "soma/amr/cluster.hpp"
#include "soma/amr/level_field.hpp"

namespace {

using soma::amr::Box;
using soma::amr::BoxList;
using soma::amr::IntVect;

std::vector<IntVect> blobs(std::mt19937& g, int dim, const Box& domain, int count) {
  std::set<std::tuple<int, int, int>> seen;
  std::vector<IntVect> out;
  std::uniform_int_distribution<int> radius(1, 5);
  for (int b = 0; b < count; ++b) {
    IntVect c;
    for (int d = 0; d < dim; ++d) c[d] = std::uniform_int_distribution<int>(domain.lo(d), domain.hi(d))(g);
    const int rr = radius(g);
    for (int k = dim == 3 ? -rr : 0; k <= (dim == 3 ? rr : 0); ++k) {
      for (int j = -rr; j <= rr; ++j) {
        for (int i = -rr; i <= rr; ++i) {
          const IntVect p{{c[0] + i, c[1] + j, c[2] + k}};
          if (i * i + j * j + k * k > rr * rr || !domain.contains(p)) continue;
          if (seen.emplace(p[0], p[1], p[2]).second) out.push_back(p);
        }
      }
    }
  }
  return out;
}

}  // namespace

TEST(AmrCluster, CoversEveryTagEfficiently) {
  std::mt19937 g(17);
  for (const int dim : {2, 3}) {
    const Box domain = dim == 2 ? Box(IntVect{{-16, 0, 0}}, IntVect{{47, 31, 0}}, 2) : Box(IntVect{}, IntVect{{23, 15, 15}});
    for (const int block : {1, 2, 4}) {
      for (const double eff : {0.5, 0.7, 0.9}) {
        for (int trial = 0; trial < 4; ++trial) {
          const std::vector<IntVect> tags = blobs(g, dim, domain, 1 + trial);
          const BoxList boxes = soma::amr::cluster(tags, domain, {eff, block});
          EXPECT_TRUE(soma::amr::disjoint(boxes));
          std::set<std::tuple<int, int, int>> blocks;
          for (const IntVect& t : tags) {
            int n = 0;
            for (const Box& b : boxes) n += b.contains(t) ? 1 : 0;
            ASSERT_EQ(n, 1);
            blocks.emplace(soma::amr::floor_div(t[0], block), soma::amr::floor_div(t[1], block), soma::amr::floor_div(t[2], block));
          }
          for (const Box& b : boxes) {
            EXPECT_TRUE(domain.contains(b));
            EXPECT_EQ(b.dim(), dim);
            const Box c = b.coarsen(block);
            EXPECT_EQ(c.refine(block), b) << "aligned to the block";
            std::int64_t tagged = 0;
            for (std::int64_t q = 0; q < c.num_pts(); ++q) {
              const IntVect p = soma::amr::detail::point_of(c, q);
              tagged += blocks.count({p[0], p[1], dim == 3 ? p[2] : 0});
            }
            EXPECT_TRUE(static_cast<double>(tagged) >= eff * static_cast<double>(c.num_pts()) || c.num_pts() == 1) << tagged << " of " << c.num_pts();
          }
          // The order of the tags does not matter.
          std::vector<IntVect> shuffled = tags;
          std::shuffle(shuffled.begin(), shuffled.end(), g);
          EXPECT_EQ(soma::amr::cluster(shuffled, domain, {eff, block}), boxes);
        }
      }
    }
  }
}

TEST(AmrCluster, SplitsAtHolesAndInflections) {
  const Box domain(IntVect{}, IntVect{{31, 31, 0}}, 2);
  const auto fill = [](const Box& b, std::vector<IntVect>& out) {
    for (std::int64_t q = 0; q < b.num_pts(); ++q) out.push_back(soma::amr::detail::point_of(b, q));
  };
  // Two separate rectangles: a hole between them, one box each.
  std::vector<IntVect> two;
  const Box a(IntVect{{2, 3, 0}}, IntVect{{6, 9, 0}}, 2), b(IntVect{{12, 1, 0}}, IntVect{{20, 4, 0}}, 2);
  fill(a, two);
  fill(b, two);
  const BoxList got = soma::amr::cluster(two, domain, {0.7, 1});
  ASSERT_EQ(got.size(), 2u);
  EXPECT_TRUE((got[0] == a && got[1] == b) || (got[0] == b && got[1] == a));
  // An L (28 of 64 cells of its bounding box) has no hole: it splits at the
  // inflection into two full boxes.
  std::vector<IntVect> ell;
  fill(Box(IntVect{{0, 0, 0}}, IntVect{{7, 1, 0}}, 2), ell);
  fill(Box(IntVect{{0, 2, 0}}, IntVect{{1, 7, 0}}, 2), ell);
  const BoxList l = soma::amr::cluster(ell, domain, {0.7, 1});
  EXPECT_EQ(l.size(), 2u);
  EXPECT_EQ(soma::amr::num_pts(l), 28);
  // Efficiency 1 takes only full boxes; a loose efficiency takes one box.
  EXPECT_EQ(soma::amr::num_pts(soma::amr::cluster(ell, domain, {1.0, 1})), 28);
  EXPECT_EQ(soma::amr::cluster(ell, domain, {0.4, 1}).size(), 1u);
  // A box exactly as efficient as asked is taken.
  EXPECT_EQ(soma::amr::cluster({IntVect{}, IntVect{{3, 0, 0}}}, domain, {0.5, 1}).size(), 1u);
  // Of two holes the one nearer the middle cuts first: x = 0, 2..6, 8, 9 at
  // efficiency 0.85 gives [0, 6] and [8, 9], not [0, 0] and [2, 9].
  std::vector<IntVect> row;
  for (const int x : {0, 2, 3, 4, 5, 6, 8, 9}) row.push_back(IntVect{{x, 4, 0}});
  const BoxList cut = soma::amr::cluster(row, domain, {0.85, 1});
  ASSERT_EQ(cut.size(), 2u);
  EXPECT_EQ(cut[0], Box(IntVect{{0, 4, 0}}, IntVect{{6, 4, 0}}, 2));
  EXPECT_EQ(cut[1], Box(IntVect{{8, 4, 0}}, IntVect{{9, 4, 0}}, 2));
  // Blocks of 4: a single tag gives one aligned block.
  EXPECT_EQ(soma::amr::cluster({IntVect{{5, 9, 0}}}, domain, {0.7, 4}), (BoxList{Box(IntVect{{4, 8, 0}}, IntVect{{7, 11, 0}}, 2)}));
  EXPECT_TRUE(soma::amr::cluster({}, domain, {0.7, 4}).empty());
}

TEST(AmrCluster, RejectsBadInput) {
  const Box domain(IntVect{}, IntVect{{31, 31, 0}}, 2);
  EXPECT_THROW(soma::amr::cluster({IntVect{}}, domain, {0.0, 1}), std::invalid_argument);
  EXPECT_THROW(soma::amr::cluster({IntVect{}}, domain, {1.5, 1}), std::invalid_argument);
  EXPECT_THROW(soma::amr::cluster({IntVect{}}, domain, {0.7, 0}), std::invalid_argument);
  EXPECT_THROW(soma::amr::cluster({IntVect{}}, Box(IntVect{}, IntVect{{30, 31, 0}}, 2), {0.7, 4}), std::invalid_argument);
  EXPECT_THROW(soma::amr::cluster({IntVect{}}, Box(IntVect{{2, 0, 0}}, IntVect{{33, 31, 0}}, 2), {0.7, 4}), std::invalid_argument);
  EXPECT_THROW(soma::amr::cluster({IntVect{{5, 5, 0}}}, Box(IntVect{{2, 0, 0}}, IntVect{{33, 31, 0}}, 2), {0.7, 4}), std::invalid_argument);
  EXPECT_THROW(soma::amr::cluster({IntVect{{32, 0, 0}}}, domain, {0.7, 1}), std::invalid_argument);
}
