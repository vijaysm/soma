// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Box algebra against brute force: every operation is checked point by point
// over random boxes in a small window, in 2D and 3D, cell and nodal.

#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "soma/amr/box.hpp"
#include "soma/amr/box_list.hpp"

namespace {

using soma::amr::Box;
using soma::amr::BoxList;
using soma::amr::IndexType;
using soma::amr::IntVect;

constexpr int W = 7;  // the window is [-W, W] per direction

Box random_box(std::mt19937& g, int dim, IndexType t) {
  std::uniform_int_distribution<int> c(-W + 1, W - 1), len(0, 4);
  IntVect lo, hi;
  for (int d = 0; d < dim; ++d) {
    lo[d] = c(g);
    hi[d] = lo[d] + len(g);
  }
  return Box(lo, hi, dim, t);
}

// Every point of the window, 2D or 3D.
std::vector<IntVect> window(int dim, int w = W) {
  std::vector<IntVect> out;
  const int kw = dim == 3 ? w : 0;
  for (int k = -kw; k <= kw; ++k) {
    for (int j = -w; j <= w; ++j) {
      for (int i = -w; i <= w; ++i) out.push_back(IntVect{{i, j, k}});
    }
  }
  return out;
}

int in_list(const BoxList& l, const IntVect& p) {
  int n = 0;
  for (const Box& b : l) n += b.contains(p) ? 1 : 0;
  return n;
}

std::vector<IndexType> types(int dim) {
  return {IndexType::cell(), IndexType::node(dim), IndexType::face(0), IndexType::face(1)};
}

}  // namespace

TEST(AmrBox, CountsAndContainment) {
  const Box b(IntVect{{-1, 0, 2}}, IntVect{{2, 4, 2}});
  EXPECT_EQ(b.num_pts(), 4 * 5 * 1);
  EXPECT_TRUE(b.contains(IntVect{{2, 4, 2}}));
  EXPECT_FALSE(b.contains(IntVect{{3, 4, 2}}));
  EXPECT_TRUE(Box().empty());
  EXPECT_EQ(Box().num_pts(), 0);
  // 2D keeps k at 0 whatever is given, and leaves it alone.
  const Box q(IntVect{{0, 0, 5}}, IntVect{{3, 3, 9}}, 2);
  EXPECT_EQ(q.lo(2), 0);
  EXPECT_EQ(q.hi(2), 0);
  EXPECT_EQ(q.grow(1).num_pts(), 6 * 6);
  EXPECT_EQ(q.refine(2).num_pts(), 8 * 8);
  // n cells are n + 1 nodes, and back.
  EXPECT_EQ(b.convert(IndexType::node(3)).num_pts(), 5 * 6 * 2);
  EXPECT_EQ(b.convert(IndexType::face(1)).num_pts(), 4 * 6 * 1);
  EXPECT_EQ(b.convert(IndexType::node(3)).convert(IndexType::cell()), b);
  // Boxes of different kinds do not meet.
  EXPECT_TRUE((b & b.convert(IndexType::node(3))).empty());
  EXPECT_FALSE(b.contains(b.convert(IndexType::face(0))));
  EXPECT_FALSE(b.contains(Box(b.lo(), b.lo(), 3, IndexType::face(0))));
  EXPECT_TRUE(b.contains(Box(b.lo(), b.lo())));
}

TEST(AmrBox, IntersectionAndGrowthPointByPoint) {
  std::mt19937 g(7);
  for (const int dim : {2, 3}) {
    const auto pts = window(dim);
    for (const IndexType t : types(dim)) {
      for (int trial = 0; trial < 40; ++trial) {
        const Box a = random_box(g, dim, t), b = random_box(g, dim, t);
        const Box ab = a & b;
        const int n = trial % 3 - 1;
        const Box ga = a.grow(n);
        const Box gd = a.grow(trial % dim, 2, -1);
        for (const IntVect& p : pts) {
          ASSERT_EQ(ab.contains(p), a.contains(p) && b.contains(p));
          bool in = true, ind = true;
          for (int d = 0; d < dim; ++d) {
            in = in && p[d] >= a.lo(d) - n && p[d] <= a.hi(d) + n;
            const int lo = a.lo(d) - (d == trial % dim ? 2 : 0), hi = a.hi(d) + (d == trial % dim ? -1 : 0);
            ind = ind && p[d] >= lo && p[d] <= hi;
          }
          ASSERT_EQ(ga.contains(p), in);
          ASSERT_EQ(gd.contains(p), ind);
        }
        EXPECT_EQ(a.contains(ab), true);
        EXPECT_EQ(a.intersects(b), !ab.empty());
      }
    }
  }
}

// A cell of the coarse box holds some fine cell; a fine cell's coarse cell is
// in the coarsened box. Nodes: refined nodes are the coarse nodes scaled, and
// the coarsened box brackets every node.
TEST(AmrBox, RefineAndCoarsenPointByPoint) {
  std::mt19937 g(11);
  for (const int dim : {2, 3}) {
    for (const int r : {2, 4}) {
      for (const IndexType t : types(dim)) {
        for (int trial = 0; trial < 30; ++trial) {
          const Box a = random_box(g, dim, t);
          const Box f = a.refine(r), c = a.coarsen(r);
          EXPECT_TRUE(f.same_kind(a));
          EXPECT_TRUE(c.same_kind(a));
          for (const IntVect& p : window(dim, W * r)) {
            IntVect q;  // the coarse cell of p
            for (int d = 0; d < dim; ++d) q[d] = soma::amr::floor_div(p[d], r);
            // Refine: a cell is in f iff its coarse cell is in a; a node iff
            // it lies between the scaled end nodes of a.
            bool want = true;
            for (int d = 0; d < dim; ++d) {
              if (t.is_nodal(d)) {
                want = want && p[d] >= a.lo(d) * r && p[d] <= a.hi(d) * r;
              } else {
                want = want && q[d] >= a.lo(d) && q[d] <= a.hi(d);
              }
            }
            ASSERT_EQ(f.contains(p), want) << dim << " " << r << " " << int(t.nodal);
          }
          for (const IntVect& p : window(dim)) {
            if (!a.contains(p)) continue;
            IntVect lo, hi;
            for (int d = 0; d < dim; ++d) {
              lo[d] = soma::amr::floor_div(p[d], r);
              hi[d] = t.is_nodal(d) ? -soma::amr::floor_div(-p[d], r) : lo[d];
            }
            ASSERT_TRUE(c.contains(lo));
            ASSERT_TRUE(c.contains(hi));
          }
          // The coarsened box is the smallest such: its corners are hit.
          for (int d = 0; d < dim; ++d) {
            EXPECT_EQ(c.lo(d), soma::amr::floor_div(a.lo(d), r));
            EXPECT_EQ(c.hi(d), t.is_nodal(d) ? -soma::amr::floor_div(-a.hi(d), r) : soma::amr::floor_div(a.hi(d), r));
          }
          EXPECT_EQ(a.refine(r).coarsen(r), a);
        }
      }
    }
  }
}

TEST(AmrBoxList, SubtractAndComplementPointByPoint) {
  std::mt19937 g(5);
  for (const int dim : {2, 3}) {
    const auto pts = window(dim);
    for (const IndexType t : types(dim)) {
      for (int trial = 0; trial < 30; ++trial) {
        const Box a = random_box(g, dim, t), b = random_box(g, dim, t);
        const BoxList d1 = soma::amr::subtract(a, b);
        EXPECT_LE(d1.size(), static_cast<std::size_t>(2 * dim));
        EXPECT_TRUE(soma::amr::disjoint(d1));
        BoxList many, holes;
        for (int i = 0; i < 4; ++i) many.push_back(random_box(g, dim, t));
        for (int i = 0; i < 3; ++i) holes.push_back(random_box(g, dim, t));
        const Box domain(IntVect{{-W, -W, -W}}, IntVect{{W, W, W}}, dim, t);
        const BoxList rest = soma::amr::subtract(BoxList{a}, holes);
        const BoxList comp = soma::amr::complement(domain, many);
        EXPECT_TRUE(soma::amr::disjoint(rest));
        EXPECT_TRUE(soma::amr::disjoint(comp));
        for (const IntVect& p : pts) {
          ASSERT_EQ(in_list(d1, p), a.contains(p) && !b.contains(p) ? 1 : 0);
          ASSERT_EQ(in_list(rest, p), a.contains(p) && in_list(holes, p) == 0 ? 1 : 0);
          ASSERT_EQ(in_list(comp, p), domain.contains(p) && in_list(many, p) == 0 ? 1 : 0);
        }
        EXPECT_EQ(soma::amr::covers(many, a), soma::amr::subtract(BoxList{a}, many).empty());
        EXPECT_TRUE(soma::amr::covers(BoxList{a}, a & b));
      }
    }
  }
}

TEST(AmrBoxList, ListHelpers) {
  const Box a(IntVect{{0, 0, 0}}, IntVect{{3, 3, 0}}, 2), b(IntVect{{4, 0, 0}}, IntVect{{7, 3, 0}}, 2);
  const BoxList l{a, b};
  EXPECT_EQ(soma::amr::num_pts(l), 32);
  EXPECT_TRUE(soma::amr::disjoint(l));
  EXPECT_FALSE(soma::amr::disjoint(soma::amr::grow(l, 1)));
  EXPECT_EQ(soma::amr::num_pts(soma::amr::refine(l, 2)), 128);
  EXPECT_EQ(soma::amr::coarsen(soma::amr::refine(l, 2), 2), l);
  EXPECT_EQ(soma::amr::intersect(l, Box(IntVect{{3, 1, 0}}, IntVect{{4, 1, 0}}, 2)).size(), 2u);
  EXPECT_TRUE(soma::amr::grow(BoxList{a}, -2).empty());
}
