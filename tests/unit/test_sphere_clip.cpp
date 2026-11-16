// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Polygons on the unit sphere: known areas, the two clippings agreeing on
// random pairs, a grid of cells recovering a cell inside it, the merge
// tolerance, and the gnomonic projection refusing a cell past its horizon.

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "soma/algorithms/sphere_clip.hpp"

using soma::Real;
using soma::detail::clip_gnomonic;
using soma::detail::clip_great_circle;
using soma::detail::Point3;
using soma::detail::sphere_area;
using soma::detail::SpherePolygon;
using soma::detail::unit;

namespace {

constexpr Real eps = 1e-12;
constexpr Real pi = std::numbers::pi;

SpherePolygon polygon(std::vector<Point3> corners) {
  SpherePolygon s;
  for (const Point3& p : corners) s.push(unit(p));
  return s;
}

// A spherical "rectangle" about direction (lon, lat): corners at +-du, +-dv
// along the tangent axes, counter-clockwise seen from outside.
SpherePolygon patch(Real lon, Real lat, Real du, Real dv, Real spin = 0) {
  const Point3 c{std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat)};
  const Point3 east{-std::sin(lon), std::cos(lon), 0};
  const Point3 north = soma::detail::cross(c, east);
  const Point3 u{std::cos(spin) * east.x + std::sin(spin) * north.x, std::cos(spin) * east.y + std::sin(spin) * north.y,
                 std::cos(spin) * east.z + std::sin(spin) * north.z};
  const Point3 v = soma::detail::cross(c, u);
  std::vector<Point3> corners;
  for (const auto& [a, b] : {std::pair{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}) {
    corners.push_back({c.x + a * du * u.x + b * dv * v.x, c.y + a * du * u.y + b * dv * v.y,
                       c.z + a * du * u.z + b * dv * v.z});
  }
  return polygon(corners);
}

Real gnomonic(const SpherePolygon& s, const SpherePolygon& c) {
  SpherePolygon out;
  return clip_gnomonic(s, c, eps, out);
}

Real great(const SpherePolygon& s, const SpherePolygon& c) {
  SpherePolygon out;
  return clip_great_circle(s, c, eps, out);
}

}  // namespace

TEST(SphereClip, KnownAreas) {
  const SpherePolygon octant = polygon({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
  EXPECT_NEAR(sphere_area(octant), pi / 2, 1e-15);
  const SpherePolygon face = polygon({{1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, -1, 1}});
  EXPECT_NEAR(sphere_area(face), 4 * pi / 6, 1e-14);
  // A quarter of the face (y, z >= 0).
  const SpherePolygon quarter = polygon({{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}});
  for (const auto& clip : {gnomonic, great}) {
    EXPECT_NEAR(clip(face, face), 4 * pi / 6, 1e-14);
    EXPECT_NEAR(clip(face, quarter), 4 * pi / 6 / 4, 1e-14);
    EXPECT_NEAR(clip(quarter, face), 4 * pi / 6 / 4, 1e-14);
  }
  // The octant reaches the +x face's horizon: only great circles take it.
  EXPECT_NEAR(great(face, octant), 4 * pi / 6 / 4, 1e-14);
}

TEST(SphereClip, TheTwoClippingsAgree) {
  std::mt19937 gen(5);
  std::uniform_real_distribution<Real> u(0, 1);
  Real worst = 0;
  int some = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    const Real lon = 2 * pi * u(gen), lat = pi * (u(gen) - 0.5) * 0.95;
    const SpherePolygon a = patch(lon, lat, 0.05 + 0.1 * u(gen), 0.05 + 0.1 * u(gen), u(gen));
    const SpherePolygon b = patch(lon + 0.1 * (u(gen) - 0.5), lat + 0.1 * (u(gen) - 0.5), 0.05 + 0.1 * u(gen),
                                  0.05 + 0.1 * u(gen), u(gen));
    const Real g = gnomonic(a, b), c = great(a, b);
    ASSERT_GE(g, 0);
    ASSERT_GE(c, 0);
    some += c > 1e-4 ? 1 : 0;
    worst = std::max(worst, std::abs(g - c) / std::min(sphere_area(a), sphere_area(b)));
    worst = std::max(worst, std::abs(great(b, a) - c) / std::min(sphere_area(a), sphere_area(b)));
  }
  EXPECT_GT(some, 1000);
  EXPECT_LE(worst, 1e-13);
}

// A cell over a cube face's grid of equiangular cells: its overlaps add up
// to it (shared edges leave no gaps or double counts).
TEST(SphereClip, AGridOfCellsRecoversACellInsideIt) {
  const int m = 8;
  const auto at = [&](int t) { return std::tan(pi / 4 * (2.0 * t / m - 1)); };
  for (const Real spin : {0.0, 0.3, 1.1}) {
    const SpherePolygon c = patch(0.1, 0.05, 0.3, 0.2, spin);
    for (const auto& clip : {gnomonic, great}) {
      Real sum = 0;
      for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
          sum += clip(c, polygon({{1, at(i), at(j)}, {1, at(i + 1), at(j)}, {1, at(i + 1), at(j + 1)},
                                  {1, at(i), at(j + 1)}}));
        }
      }
      EXPECT_NEAR(sum, sphere_area(c), 1e-14);
    }
  }
}

// A copy moved by less than the tolerance overlaps all of the cell; cells
// that only share an edge or a corner overlap by nothing.
TEST(SphereClip, CopiesOverlapWhollyAndNeighboursNotAtAll) {
  const SpherePolygon a = patch(0.4, 0.3, 0.1, 0.1);
  const SpherePolygon b = patch(0.4 + 1e-14, 0.3 - 1e-14, 0.1, 0.1);
  const auto cell = [](Real t0, Real t1, Real s0, Real s1) {
    return polygon({{1, t0, s0}, {1, t1, s0}, {1, t1, s1}, {1, t0, s1}});
  };
  for (const auto& clip : {gnomonic, great}) {
    EXPECT_NEAR(clip(a, b), sphere_area(a), 1e-13);
    EXPECT_EQ(clip(cell(0.1, 0.3, 0.2, 0.4), cell(0.3, 0.5, 0.2, 0.4)), 0);
    EXPECT_EQ(clip(cell(0.1, 0.3, 0.2, 0.4), cell(0.3, 0.5, 0.4, 0.6)), 0);
  }
}

TEST(SphereClip, GnomonicRefusesACellPastTheHorizon) {
  // A cell reaching more than 90 degrees from the face its partner points
  // into cannot be projected; great circles take it.
  const SpherePolygon big = polygon({{1, -1, -1}, {1, 1, -1}, {-1, 1, 0.5}, {-1, -1, 0.5}});
  const SpherePolygon small = patch(0, 0, 0.1, 0.1);
  EXPECT_LT(gnomonic(big, small), 0);
  EXPECT_GE(great(small, big), 0);
}
