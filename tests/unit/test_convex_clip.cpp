// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Overlaps of whole convex cells: boxes against the product of their
// extents, affine hexes, prisms, pyramids and tets against the sums over
// their tets, a grid of cells recovering a cell inside it (shared faces leave
// no gaps), flat and convex cells recognised, the merge tolerance, and cells
// far from the origin.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "soma/algorithms/convex_clip.hpp"

using soma::Real;
using soma::detail::convex_overlap;
using soma::detail::is_convex;
using soma::detail::overlap;
using soma::detail::Point3;
using soma::detail::Tet;

namespace {

constexpr Real eps = 1e-12;

using Cell = std::vector<Point3>;

// The reference cells (MOAB corner order) on the unit cube.
Cell hex() { return {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}; }
Cell prism() { return {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}}; }
Cell pyramid() { return {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1}}; }
Cell tet() { return {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; }

// The cell's tets, by corner (each turned as the cell).
std::vector<std::vector<int>> tets_of(std::size_t corners) {
  if (corners == 8) {  // Kuhn: paths 0 -> 6 one axis at a time
    return {{0, 1, 2, 6}, {0, 1, 5, 6}, {0, 3, 2, 6}, {0, 3, 7, 6}, {0, 4, 5, 6}, {0, 4, 7, 6}};
  }
  if (corners == 6) return {{0, 1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}};
  if (corners == 5) return {{0, 1, 2, 4}, {0, 2, 3, 4}};
  return {{0, 1, 2, 3}};
}

std::vector<Tet> tets(const Cell& c) {
  std::vector<Tet> out;
  for (const auto& t : tets_of(c.size())) out.push_back({{c[t[0]], c[t[1]], c[t[2]], c[t[3]]}});
  return out;
}

// The overlap as the sum over the two cells' tets.
Real by_tets(const Cell& a, const Cell& b) {
  Real sum = 0;
  for (const Tet& s : tets(a)) {
    for (const Tet& t : tets(b)) sum += overlap(s, t, eps);
  }
  return sum;
}

Real convex(const Cell& a, const Cell& b) {
  return convex_overlap(a.data(), static_cast<int>(a.size()), b.data(), static_cast<int>(b.size()), eps);
}

// x -> m x + o, with m of positive determinant.
struct Affine {
  Real m[3][3];
  Point3 o;
  [[nodiscard]] Point3 operator()(const Point3& p) const {
    return {m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + o.x, m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + o.y,
            m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + o.z};
  }
  [[nodiscard]] Real det() const {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  }
};

Affine random_affine(std::mt19937& gen, Real size) {
  std::uniform_real_distribution<Real> u(0, 1);
  Affine a{};
  do {
    for (auto& row : a.m) {
      for (Real& x : row) x = size * (u(gen) - 0.5);
    }
  } while (a.det() < 0.05 * size * size * size);
  a.o = {0.3 + 0.4 * u(gen), 0.3 + 0.4 * u(gen), 0.3 + 0.4 * u(gen)};
  return a;
}

Cell mapped(const Cell& c, const Affine& a) {
  Cell out;
  for (const Point3& p : c) out.push_back(a(p));
  return out;
}

Real volume(const Cell& c) {
  Real v = 0;
  for (const Tet& t : tets(c)) v += std::abs(soma::detail::six_volume(t)) / 6;
  return v;
}

}  // namespace

TEST(ConvexClip, BoxesOverlapAsTheProductOfTheirExtents) {
  std::mt19937 gen(3);
  std::uniform_real_distribution<Real> u(0, 1);
  Real worst = 0;
  int some = 0;
  for (int trial = 0; trial < 1000; ++trial) {
    Real lo[2][3], hi[2][3];
    Cell box[2];
    for (int b = 0; b < 2; ++b) {
      for (int a = 0; a < 3; ++a) {
        lo[b][a] = std::ldexp(std::round(std::ldexp(u(gen), 10)), -10);
        hi[b][a] = lo[b][a] + std::ldexp(std::round(std::ldexp(0.05 + u(gen), 10)), -10);
      }
      for (const Point3& p : hex()) {
        box[b].push_back({p.x == 0 ? lo[b][0] : hi[b][0], p.y == 0 ? lo[b][1] : hi[b][1],
                          p.z == 0 ? lo[b][2] : hi[b][2]});
      }
    }
    Real want = 1;
    for (int a = 0; a < 3; ++a) want *= std::max<Real>(0, std::min(hi[0][a], hi[1][a]) - std::max(lo[0][a], lo[1][a]));
    some += want > 0 ? 1 : 0;
    worst = std::max(worst, std::abs(convex(box[0], box[1]) - want));
  }
  EXPECT_GT(some, 300);  // not vacuous
  EXPECT_LE(worst, 1e-15);
}

// Every pair of hexes, prisms, pyramids and tets, each the image of its
// reference cell by a random affine map: the same as over their tets.
TEST(ConvexClip, CellsOverlapAsTheirTetsDo) {
  std::mt19937 gen(7);
  const std::vector<Cell> kinds = {hex(), prism(), pyramid(), tet()};
  Real worst = 0;
  int some = 0;
  for (int trial = 0; trial < 300; ++trial) {
    for (const Cell& a : kinds) {
      for (const Cell& b : kinds) {
        const Cell s = mapped(a, random_affine(gen, 0.7)), t = mapped(b, random_affine(gen, 0.7));
        const Real want = by_tets(s, t), got = convex(s, t);
        ASSERT_GE(got, 0);
        some += want > 1e-3 ? 1 : 0;
        worst = std::max(worst, std::abs(got - want));
        worst = std::max(worst, std::abs(convex(t, s) - want));
      }
    }
  }
  EXPECT_GT(some, 500);  // not vacuous
  EXPECT_LE(worst, 1e-14);
}

// A cell inside an affine grid of hexes: its overlaps with them add up to it.
TEST(ConvexClip, AGridOfCellsRecoversACellInsideIt) {
  std::mt19937 gen(11);
  std::uniform_real_distribution<Real> u(0, 1);
  const int n = 4;
  Real worst = 0;
  for (int trial = 0; trial < 20; ++trial) {
    Affine grid = random_affine(gen, 1);
    grid.o = {0, 0, 0};
    for (const Cell& kind : {hex(), prism(), pyramid(), tet()}) {
      // A small cell well inside the grid's middle.
      Affine in{};
      for (int a = 0; a < 3; ++a) in.m[a][a] = 0.3 + 0.2 * u(gen);
      in.m[0][1] = 0.1 * u(gen);
      in.m[1][2] = 0.1 * u(gen);
      in.o = {0.1 + 0.1 * u(gen), 0.1 + 0.1 * u(gen), 0.1 + 0.1 * u(gen)};
      Cell c;
      for (const Point3& p : mapped(kind, in)) c.push_back(grid(p));
      Real sum = 0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          for (int k = 0; k < n; ++k) {
            Cell h;
            for (const Point3& p : hex()) {
              h.push_back(grid({(i + p.x) / n, (j + p.y) / n, (k + p.z) / n}));
            }
            sum += convex(c, h);
          }
        }
      }
      worst = std::max(worst, std::abs(sum - volume(c)) / volume(c));
    }
  }
  EXPECT_LE(worst, 1e-13);
}

TEST(ConvexClip, FlatConvexCellsAreRecognised) {
  std::mt19937 gen(5);
  for (const Cell& kind : {hex(), prism(), pyramid(), tet()}) {
    const Cell c = mapped(kind, random_affine(gen, 1));
    EXPECT_TRUE(is_convex(c.data(), static_cast<int>(c.size()), eps));
    // A corner moved off its faces' planes by more than eps, or less.
    for (const Real by : {1e-9, 1e-14}) {
      Cell d = c;
      d[1] = {d[1].x + by, d[1].y - by, d[1].z + by};
      EXPECT_EQ(is_convex(d.data(), static_cast<int>(d.size()), eps), by < eps || kind.size() == 4)
          << kind.size() << " corners, moved by " << by;
    }
  }
  // A hex with a corner pushed in: its faces are not flat.
  Cell dent = hex();
  dent[6] = {0.6, 0.6, 0.6};
  EXPECT_FALSE(is_convex(dent.data(), 8, eps));
}

// A copy moved by less than the tolerance overlaps the cell but for a band
// no thicker than the move; a sliver thinner than the tolerance is nothing.
TEST(ConvexClip, TheMergeToleranceRemovesSlivers) {
  const auto moved = [](Cell c, Real x, Real y, Real z) {
    for (Point3& p : c) p = {p.x + x, p.y + y, p.z + z};
    return c;
  };
  EXPECT_NEAR(convex(hex(), moved(hex(), 1e-14, -1e-14, 1e-14)), 1, 1e-13);
  EXPECT_EQ(convex(hex(), moved(hex(), 1 - 1e-13, 0, 0)), 0);
  EXPECT_EQ(convex(moved(hex(), 1 - 1e-13, 0, 0), hex()), 0);
  EXPECT_EQ(convex(prism(), moved(hex(), 0, 0, 1 - 1e-13)), 0);
}

TEST(ConvexClip, OverlapsFarFromTheOriginKeepTheirAccuracy) {
  std::mt19937 gen(13);
  // Corners on a 2^-24 grid, so that the moved ones are exact.
  const auto snapped = [](Cell c) {
    const auto snap = [](Real x) { return std::ldexp(std::round(std::ldexp(x, 24)), -24); };
    for (Point3& p : c) p = {snap(p.x), snap(p.y), snap(p.z)};
    return c;
  };
  const auto moved = [](Cell c, Real by) {
    for (Point3& p : c) p = {p.x + by, p.y - by, p.z + by};
    return c;
  };
  Real worst = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const Cell a = snapped(mapped(hex(), random_affine(gen, 0.6)));
    const Cell b = snapped(mapped(prism(), random_affine(gen, 0.6)));
    worst = std::max(worst, std::abs(convex(moved(a, 1e7), moved(b, 1e7)) - convex(a, b)));
  }
  EXPECT_LE(worst, 1e-14);
}
