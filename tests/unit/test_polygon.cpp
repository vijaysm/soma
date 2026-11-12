// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Plane polygons: areas, convexity, overlaps of convex polygons (known
// cases, either order, a polygon split in two), the merge tolerance, and
// ear clipping of non-convex polygons.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "soma/algorithms/polygon.hpp"

using soma::Real;
using soma::detail::clip;
using soma::detail::convex;
using soma::detail::Polygon;
using soma::detail::simple;
using soma::detail::triangulate;
using soma::detail::twice_area;

namespace {

constexpr Real eps = 1e-12;

Polygon poly(const std::vector<Real>& xy) {
  Polygon p;
  for (std::size_t i = 0; i + 1 < xy.size(); i += 2) p.push(xy[i], xy[i + 1]);
  return p;
}

Polygon square(Real x0, Real y0, Real h) { return poly({x0, y0, x0 + h, y0, x0 + h, y0 + h, x0, y0 + h}); }

Real overlap(const Polygon& a, const Polygon& b) {
  Polygon out;
  return clip(a, b, eps, out);
}

// A random convex polygon: k points on an ellipse about (cx, cy), in angle order.
Polygon random_convex(std::mt19937& gen, int k) {
  std::uniform_real_distribution<Real> u(0, 1);
  const Real cx = u(gen), cy = u(gen), rx = 0.1 + 0.4 * u(gen), ry = 0.1 + 0.4 * u(gen), phase = 6.28 * u(gen);
  std::vector<Real> t(static_cast<std::size_t>(k));
  for (auto& v : t) v = u(gen);
  std::sort(t.begin(), t.end());
  Polygon p;
  for (const Real v : t) p.push(cx + rx * std::cos(phase + 6.283185307179586 * v), cy + ry * std::sin(phase + 6.283185307179586 * v));
  return p;
}

}  // namespace

TEST(Polygon, AreasAndConvexity) {
  EXPECT_EQ(twice_area(square(0, 0, 1)), 2);
  EXPECT_EQ(twice_area(poly({0, 0, 1, 0, 0, 1})), 1);
  EXPECT_TRUE(convex(square(0, 0, 1), eps));
  EXPECT_TRUE(convex(poly({0, 0, 0.5, 0, 1, 0, 1, 1, 0, 1}), eps));  // a corner in a line
  EXPECT_FALSE(convex(poly({0, 0, 0, 1, 1, 1, 1, 0}), eps));         // clockwise
  EXPECT_FALSE(convex(poly({0, 0, 2, 0, 2, 1, 1, 1, 1, 2, 0, 2}), eps));  // an L
  EXPECT_FALSE(convex(poly({0, 0, 1, 0}), eps));
}

TEST(Polygon, OverlapsOfConvexPolygons) {
  EXPECT_EQ(overlap(square(0, 0, 1), square(0, 0, 1)), 1);
  EXPECT_EQ(overlap(square(0, 0, 1), square(0.5, 0.5, 1)), 0.25);
  EXPECT_EQ(overlap(square(0, 0, 1), square(0.25, 0.25, 0.5)), 0.25);  // inside
  EXPECT_EQ(overlap(square(0, 0, 1), square(2, 0, 1)), 0);              // apart
  EXPECT_EQ(overlap(square(0, 0, 1), square(1, 0, 1)), 0);              // sharing an edge
  EXPECT_EQ(overlap(square(0, 0, 1), square(1, 1, 1)), 0);              // sharing a corner
  // A triangle over a square: half the square.
  EXPECT_NEAR(overlap(square(0, 0, 1), poly({0, 0, 1, 0, 0, 1})), 0.5, 1e-16);
  // A diamond in a square: the diamond.
  EXPECT_NEAR(overlap(square(0, 0, 1), poly({0.5, 0, 1, 0.5, 0.5, 1, 0, 0.5})), 0.5, 1e-16);
}

// Random convex polygons: either order gives the area to rounding, and the
// overlap with b is the sum of the overlaps with b's two halves.
TEST(Polygon, OverlapIdentities) {
  std::mt19937 gen(42);
  Real worst_order = 0, worst_split = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    const Polygon a = random_convex(gen, 3 + trial % 9), b = random_convex(gen, 3 + trial % 7);
    const Real ab = overlap(a, b), ba = overlap(b, a);
    worst_order = std::max(worst_order, std::abs(ab - ba));
    // b cut along its diagonal from corner 0 to corner b.n / 2.
    Polygon h1, h2;
    const int mid = b.n / 2;
    for (int i = 0; i <= mid; ++i) h1.push(b.x[i], b.y[i]);
    for (int i = mid; i <= b.n; ++i) h2.push(b.x[i % b.n], b.y[i % b.n]);
    worst_split = std::max(worst_split, std::abs(ab - (overlap(a, h1) + overlap(a, h2))));
  }
  EXPECT_LE(worst_order, 1e-15);
  EXPECT_LE(worst_split, 1e-15);
}

// Corners closer than the tolerance merge; a sliver thinner than it is none.
TEST(Polygon, TheMergeToleranceRemovesDegeneratePieces) {
  Polygon out;
  EXPECT_NEAR(clip(square(0, 0, 1), square(1e-14, -1e-14, 1), eps, out), 1, 1e-13);
  EXPECT_EQ(out.n, 4);
  EXPECT_EQ(clip(square(0, 0, 1), square(1 - 1e-13, 0, 1), eps, out), 0);  // a sliver 1e-13 wide
  EXPECT_EQ(out.n, 0);
  // With a tolerance below the sliver's width it is a piece.
  EXPECT_NEAR(clip(square(0, 0, 1), square(1 - 1e-13, 0, 1), 1e-15, out), 1e-13, 1e-16);
}

TEST(Polygon, EarClippingSplitsNonConvexPolygons) {
  int tri[3 * soma::detail::polygon_capacity];
  // An L of area 3, and a star.
  const Polygon l = poly({0, 0, 2, 0, 2, 1, 1, 1, 1, 2, 0, 2});
  std::vector<Real> star;
  for (int i = 0; i < 10; ++i) {
    const Real r = i % 2 == 0 ? 1 : 0.4, t = 0.6283185307179586 * i;
    star.insert(star.end(), {r * std::cos(t), r * std::sin(t)});
  }
  for (const Polygon& p : {l, poly(star), square(0, 0, 1)}) {
    const int n = triangulate(p, eps, tri);
    ASSERT_EQ(n, p.n - 2);
    Real sum = 0;
    for (int t = 0; t < n; ++t) {
      const Polygon piece = poly({p.x[tri[3 * t]], p.y[tri[3 * t]], p.x[tri[3 * t + 1]], p.y[tri[3 * t + 1]],
                                  p.x[tri[3 * t + 2]], p.y[tri[3 * t + 2]]});
      EXPECT_GT(twice_area(piece), 0);
      sum += twice_area(piece);
    }
    EXPECT_NEAR(sum, twice_area(p), 1e-14);
  }
  // The L's overlap with a square, through its triangles.
  Real through = 0;
  const int n = triangulate(l, eps, tri);
  for (int t = 0; t < n; ++t) {
    through += overlap(poly({l.x[tri[3 * t]], l.y[tri[3 * t]], l.x[tri[3 * t + 1]], l.y[tri[3 * t + 1]],
                             l.x[tri[3 * t + 2]], l.y[tri[3 * t + 2]]}),
                       square(0.5, 0.5, 1));
  }
  EXPECT_NEAR(through, 0.75, 1e-15);  // the square less its quarter in the L's notch
  EXPECT_EQ(triangulate(poly({0, 0, 1, 0}), eps, tri), -1);
}

// Random star-shaped polygons (corners at random radii in angle order):
// every triangle positive, their areas the polygon's.
TEST(Polygon, EarClippingSplitsRandomStars) {
  std::mt19937 gen(9);
  std::uniform_real_distribution<Real> u(0, 1);
  int tri[3 * soma::detail::polygon_capacity];
  int failed = 0;
  for (int trial = 0; trial < 1000; ++trial) {
    const int k = 4 + trial % 28;
    Polygon p;
    for (int i = 0; i < k; ++i) {
      const Real r = 0.2 + u(gen), t = 6.283185307179586 * (i + 0.8 * u(gen)) / k;
      p.push(r * std::cos(t), r * std::sin(t));
    }
    const int n = triangulate(p, eps, tri);
    Real sum = 0;
    bool positive = n == k - 2;
    for (int t = 0; t < n; ++t) {
      const Polygon piece = poly({p.x[tri[3 * t]], p.y[tri[3 * t]], p.x[tri[3 * t + 1]], p.y[tri[3 * t + 1]],
                                  p.x[tri[3 * t + 2]], p.y[tri[3 * t + 2]]});
      positive = positive && twice_area(piece) > 0;
      sum += twice_area(piece);
    }
    if (!positive || std::abs(sum - twice_area(p)) > 1e-13) ++failed;
  }
  EXPECT_EQ(failed, 0);
}

// A bowtie and a pentagram turn left at every corner, or nearly, and are not
// simple; an L with corners in a line is.
TEST(Polygon, SelfCrossingPolygonsAreNotSimple) {
  EXPECT_FALSE(simple(poly({0.227, 0.319, 0.978, 0.456, 0.308, 0.264, 0.087, 0.419})));
  std::vector<Real> star;
  for (int i = 0; i < 5; ++i) {
    const Real t = 1.2566370614359172 * 2 * i;  // every second corner of a pentagon
    star.insert(star.end(), {std::cos(t), std::sin(t)});
  }
  const Polygon pentagram = poly(star);
  EXPECT_TRUE(convex(pentagram, eps));  // locally
  EXPECT_FALSE(simple(pentagram));
  EXPECT_TRUE(simple(poly({0, 0, 1, 0, 2, 0, 2, 1, 1, 1, 1, 2, 0, 2})));
  EXPECT_FALSE(simple(poly({0, 0, 2, 0, 2, 2, 1, 0, 0, 2})));  // a corner on an edge
}

// Two 64-gons (the most a cell may have) overlap in up to 128 corners: the
// overlap is kept, in either order.
TEST(Polygon, OverlapsOfManyCornersFit) {
  for (const int n : {40, 64}) {
    Polygon a, b;
    for (int i = 0; i < n; ++i) {
      const Real t = 6.283185307179586 * i / n;
      a.push(std::cos(t), std::sin(t));
      b.push(std::cos(t + 3.141592653589793 / n), std::sin(t + 3.141592653589793 / n));
    }
    Polygon out;
    const Real ab = clip(a, b, eps, out);
    EXPECT_EQ(out.n, 2 * n);
    EXPECT_GT(ab, 0.99 * twice_area(a) / 2);
    EXPECT_NEAR(ab, clip(b, a, eps, out), 1e-14);
  }
}

// Ears are found whatever the tolerance.
TEST(Polygon, EarClippingDoesNotDependOnTheTolerance) {
  std::mt19937 gen(9);
  std::uniform_real_distribution<Real> u(0, 1);
  int tri[3 * soma::detail::polygon_capacity];
  int failed = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    const int k = 4 + trial % 28;
    Polygon p;
    for (int i = 0; i < k; ++i) {
      const Real r = 0.2 + u(gen), t = 6.283185307179586 * (i + 0.8 * u(gen)) / k;
      p.push(r * std::cos(t), r * std::sin(t));
    }
    failed += triangulate(p, 1e-2, tri) == k - 2 ? 0 : 1;
  }
  EXPECT_EQ(failed, 0);
}
