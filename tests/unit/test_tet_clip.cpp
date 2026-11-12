// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Tet overlaps: known volumes, either order, a tet split in four about its
// centroid, a cube's six Kuhn tets recovering a tet inside it, and the
// merge tolerance.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "soma/algorithms/tet_clip.hpp"

using soma::Real;
using soma::detail::overlap;
using soma::detail::Point3;
using soma::detail::six_volume;
using soma::detail::Tet;

namespace {

constexpr Real eps = 1e-12;

Tet unit(Real dx = 0, Real dy = 0, Real dz = 0, Real s = 1) {
  return {{{dx, dy, dz}, {dx + s, dy, dz}, {dx, dy + s, dz}, {dx, dy, dz + s}}};
}

Real volume(const Tet& t) { return std::abs(six_volume(t)) / 6; }

Tet random_tet(std::mt19937& gen, Real size) {
  std::uniform_real_distribution<Real> u(0, 1);
  Tet t;
  const Point3 c{0.4 + 0.2 * u(gen), 0.4 + 0.2 * u(gen), 0.4 + 0.2 * u(gen)};  // mostly overlapping
  for (auto& p : t.p) p = {c.x + size * (u(gen) - 0.5), c.y + size * (u(gen) - 0.5), c.z + size * (u(gen) - 0.5)};
  return t;
}

// The cube [x0, x0 + h]^3 as six tets about its main diagonal (Kuhn).
std::vector<Tet> kuhn(Real x0, Real h) {
  const auto at = [&](int i, int j, int k) { return Point3{x0 + i * h, x0 + j * h, x0 + k * h}; };
  const Point3 o = at(0, 0, 0), f = at(1, 1, 1);
  const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  std::vector<Tet> out;
  for (const auto& p : perm) {
    int s[3] = {0, 0, 0};
    Tet t;
    t.p[0] = o;
    for (int q = 0; q < 2; ++q) {
      s[p[q]] = 1;
      t.p[q + 1] = at(s[0], s[1], s[2]);
    }
    t.p[3] = f;
    out.push_back(t);
  }
  return out;
}

}  // namespace

TEST(TetClip, KnownVolumes) {
  EXPECT_NEAR(overlap(unit(), unit(), eps), 1.0 / 6, 1e-16);
  EXPECT_NEAR(overlap(unit(), unit(0.5), eps), 1.0 / 48, 1e-16);  // a half-size copy
  EXPECT_NEAR(overlap(unit(0.25, 0.25, 0.25, 0.25), unit(), eps), 0.25 * 0.25 * 0.25 / 6, 1e-17);  // inside
  EXPECT_EQ(overlap(unit(), unit(2), eps), 0);  // apart
  // Sharing the face x + y + z = 1 from the other side.
  const Tet other{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}}};
  EXPECT_EQ(overlap(unit(), other, eps), 0);
  EXPECT_EQ(overlap(unit(), unit(1), eps), 0);  // sharing a corner
  // Six Kuhn tets fill the unit cube.
  Real cube = 0;
  for (const Tet& k : kuhn(0, 1)) cube += overlap(k, k, eps);
  EXPECT_NEAR(cube, 1, 1e-15);
}

// Random tets: either order agrees, and so does the sum over a tet split in
// four about its centroid.
TEST(TetClip, OverlapIdentities) {
  std::mt19937 gen(5);
  Real worst_order = 0, worst_split = 0;
  int some = 0;
  for (int trial = 0; trial < 3000; ++trial) {
    const Tet a = random_tet(gen, 0.6), b = random_tet(gen, 0.6);
    const Real ab = overlap(a, b, eps), ba = overlap(b, a, eps);
    ASSERT_GE(ab, 0);
    some += ab > 0 ? 1 : 0;
    worst_order = std::max(worst_order, std::abs(ab - ba));
    Point3 c{};
    for (const auto& p : b.p) c = {c.x + p.x / 4, c.y + p.y / 4, c.z + p.z / 4};
    Real sum = 0;
    for (int f = 0; f < 4; ++f) {
      Tet piece = b;
      piece.p[f] = c;
      sum += overlap(a, piece, eps);
    }
    worst_split = std::max(worst_split, std::abs(ab - sum));
  }
  EXPECT_GT(some, 1500);  // not vacuous
  EXPECT_LE(worst_order, 1e-15);
  EXPECT_LE(worst_split, 1e-15);
}

// A tet inside a cube made of small cubes' Kuhn tets: its pieces add up to it.
TEST(TetClip, AFineSplitRecoversATet) {
  const Tet t{{{0.13, 0.21, 0.08}, {0.91, 0.17, 0.22}, {0.31, 0.87, 0.15}, {0.4, 0.33, 0.95}}};
  const int n = 5;
  Real sum = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        for (Tet q : kuhn(0, 1.0 / n)) {
          for (auto& p : q.p) p = {p.x + static_cast<Real>(i) / n, p.y + static_cast<Real>(j) / n, p.z + static_cast<Real>(k) / n};
          sum += overlap(t, q, eps);
        }
      }
    }
  }
  EXPECT_NEAR(sum, volume(t), 1e-15);
}

// A copy moved by less than the tolerance is the tet; a sliver thinner than
// it is nothing.
TEST(TetClip, TheMergeToleranceRemovesSlivers) {
  EXPECT_NEAR(overlap(unit(), unit(1e-14, -1e-14, 1e-14), eps), 1.0 / 6, 1e-15);
  // A unit cube's Kuhn tets against the cube moved by 1e-13 along x: the
  // 1e-13 slab beyond it is a sliver.
  Real moved = 0;
  for (const Tet& a : kuhn(0, 1)) {
    for (Tet b : kuhn(0, 1)) {
      for (auto& p : b.p) p.x += 1 - 1e-13;
      moved += overlap(a, b, eps);
    }
  }
  EXPECT_EQ(moved, 0);
}
