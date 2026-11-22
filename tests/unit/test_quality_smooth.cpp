// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <set>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "distributed_fixture.hpp"
#include "refined_sphere.hpp"
#include "soma/algorithms/laplacian.hpp"
#include "soma/algorithms/quality_smooth.hpp"
#include "soma/testing/box_mesh.hpp"

using namespace soma_test;

namespace {

// Each vertex's colour by gid, gathered from every rank (owned vertices).
std::map<GO, std::int32_t> colours_by_gid(const soma::Comm& c, const soma::HostMesh& m,
                                          const std::vector<std::int32_t>& colour) {
  std::vector<Real> mine;
  for (LO v = 0; v < m.segment_table(0).owned_end; ++v) {
    mine.push_back(static_cast<Real>(m.gid(0)(v)));
    mine.push_back(static_cast<Real>(colour[static_cast<std::size_t>(v)]));
  }
  const auto all = c.allgatherv(std::span<const Real>(mine));
  std::map<GO, std::int32_t> out;
  for (std::size_t i = 0; i + 1 < all.size(); i += 2) {
    out[static_cast<GO>(all[i])] = static_cast<std::int32_t>(all[i + 1]);
  }
  return out;
}

}  // namespace

// No two corners of a cell share a colour, ghost cells included, and every
// vertex has one, with neighbours on other ranks.
TEST(QualitySmooth, ColoursAreIndependent) {
  const soma::Comm c = soma::Comm::world();
  const soma::HostMesh m = dealt_sphere(c);
  const auto colour = soma::detail::vertex_colours(c, m, soma::detail::vertex_cells(m));
  ASSERT_EQ(colour.size(), static_cast<std::size_t>(m.count(0)));
  const auto mv = m.view();
  Real bad = 0;
  for (LO v = 0; v < m.count(0); ++v) bad += colour[static_cast<std::size_t>(v)] < 0 ? 1 : 0;
  for (LO e = 0; e < m.count(2); ++e) {
    std::set<std::int32_t> seen;
    for (LO i = 0; i < mv.num_vertices(2, e); ++i) seen.insert(colour[static_cast<std::size_t>(mv.vertex(2, e, i))]);
    bad += static_cast<LO>(seen.size()) == mv.num_vertices(2, e) ? 0 : 1;
  }
  EXPECT_EQ(c.allreduce(bad, soma::Op::sum), 0);
}

// Colours follow from gids and connectivity alone.
TEST(QualitySmooth, ColoursSameOnAnyRankCount) {
  const soma::Comm c = soma::Comm::world();
  const soma::Comm self = soma::Comm::self();
  const soma::HostMesh serial = refined_sphere(self);
  const auto want = colours_by_gid(self, serial, soma::detail::vertex_colours(self, serial,
                                                                               soma::detail::vertex_cells(serial)));
  const soma::HostMesh m = dealt_sphere(c);
  const auto got = colours_by_gid(c, m, soma::detail::vertex_colours(c, m, soma::detail::vertex_cells(m)));
  EXPECT_EQ(got, want);
}

namespace {

soma::HostMesh grid(int n) { return soma::testing::box_mesh({n, n, 1, soma::Topology::quad}); }

// Inner vertices of the n x n unit grid moved by up to `scale` h, by gid.
void perturb_inner(soma::HostMesh& m, int n, Real scale) {
  const Real h = 1.0 / n;
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) {
    if (x(v, 0) <= 0 || x(v, 0) >= 1 || x(v, 1) <= 0 || x(v, 1) >= 1) continue;
    const auto t = static_cast<Real>(m.gid(0)(v));
    x(v, 0) += scale * h * std::sin(t);
    x(v, 1) += scale * h * std::cos(3 * t);
  }
}

// Mean condition number of the corners of a planar quad mesh (z = 0); a
// corner with det <= 0 counts as infinite.
Real mean_kappa(const soma::HostMesh& m) {
  const auto mv = m.view();
  const auto x = m.coords();
  Real s = 0, k = 0;
  for (LO e = 0; e < m.count(2); ++e) {
    for (LO i = 0; i < 4; ++i) {
      const LO q = mv.vertex(2, e, i), a = mv.vertex(2, e, (i + 3) % 4), b = mv.vertex(2, e, (i + 1) % 4);
      const Real e1[2] = {x(b, 0) - x(q, 0), x(b, 1) - x(q, 1)}, e2[2] = {x(a, 0) - x(q, 0), x(a, 1) - x(q, 1)};
      const Real det = e1[0] * e2[1] - e1[1] * e2[0];
      s += det > 0 ? (e1[0] * e1[0] + e1[1] * e1[1] + e2[0] * e2[0] + e2[1] * e2[1]) / (2 * det)
                   : std::numeric_limits<Real>::infinity();
      k += 1;
    }
  }
  return s / k;
}

// The least corner angle (degrees) over the owned cells of every rank.
Real least_angle(const soma::Comm& c, const soma::HostMesh& m) {
  const auto mv = m.view();
  Real least = 180;
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    const LO n = mv.num_vertices(2, e);
    for (LO i = 0; i < n; ++i) {
      const auto p = point(m, mv.vertex(2, e, (i + n - 1) % n)), q = point(m, mv.vertex(2, e, i)),
                 r = point(m, mv.vertex(2, e, (i + 1) % n));
      const Real a[3] = {p[0] - q[0], p[1] - q[1], p[2] - q[2]}, b[3] = {r[0] - q[0], r[1] - q[1], r[2] - q[2]};
      const Real cosine = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (std::hypot(a[0], a[1], a[2]) *
                                                                        std::hypot(b[0], b[1], b[2]));
      least = std::min(least, std::acos(std::clamp(cosine, Real{-1}, Real{1})) * 180 / std::numbers::pi);
    }
  }
  return c.allreduce(least, soma::Op::min);
}

Real largest_move(const soma::HostMesh& m, const std::map<GO, std::array<Real, 3>>& before) {
  const auto x = m.coords();
  Real worst = 0;
  for (LO v = 0; v < m.count(0); ++v) {
    const auto& p = before.at(m.gid(0)(v));
    for (int q = 0; q < 3; ++q) worst = std::max(worst, std::abs(x(v, q) - p[static_cast<std::size_t>(q)]));
  }
  return worst;
}

std::map<GO, std::array<Real, 3>> by_gid(const soma::HostMesh& m) {
  std::map<GO, std::array<Real, 3>> out;
  for (LO v = 0; v < m.count(0); ++v) out[m.gid(0)(v)] = point(m, v);
  return out;
}

}  // namespace

// Every corner of a uniform grid is a square's already.
TEST(QualitySmooth, UniformGridIsAFixedPoint) {
  soma::HostMesh m = grid(8);
  const auto before = by_gid(m);
  (void)soma::quality_smooth(soma::Comm::self(), m, {.angle_sweeps = 0, .polish_sweeps = 0});
  EXPECT_LE(largest_move(m, before), 1e-14);
}

TEST(QualitySmooth, ShapeImprovesAPerturbedGrid) {
  soma::HostMesh m = grid(12);
  perturb_inner(m, 12, 0.3);
  const soma::Comm self = soma::Comm::self();
  const Real k0 = mean_kappa(m), a0 = least_angle(self, m);
  const auto r = soma::quality_smooth(self, m, {.angle_sweeps = 0, .polish_sweeps = 0});
  const Real k1 = mean_kappa(m), a1 = least_angle(self, m);
  std::printf("shape: mean kappa %.4f -> %.4f, least angle %.2f -> %.2f\n", k0, k1, a0, a1);
  EXPECT_LT(k1, k0);
  EXPECT_TRUE(std::isfinite(k1));
  EXPECT_GE(a1, a0);
  EXPECT_EQ(r.shape_sweeps, 50);
  EXPECT_NEAR(r.least_angle_before, a0, 1e-9);
  EXPECT_NEAR(r.least_angle_after, a1, 1e-9);
}

// A grid perturbed close to folding: no corner inverts. On the refined
// sphere, whose transition cells the condition number alone would make
// worse, the least angle does not fall.
TEST(QualitySmooth, NoStepInvertsOrLowersTheLeast) {
  const soma::Comm self = soma::Comm::self();
  soma::HostMesh m = grid(12);
  perturb_inner(m, 12, 0.4);
  const auto r = soma::quality_smooth(self, m, {.angle_sweeps = 0, .polish_sweeps = 0});
  std::printf("near folding: least angle %.2f -> %.2f\n", r.least_angle_before, r.least_angle_after);
  EXPECT_LT(r.least_angle_before, 35);
  EXPECT_TRUE(std::isfinite(mean_kappa(m)));
  EXPECT_GE(r.least_angle_after, r.least_angle_before);
  const soma::Comm c = soma::Comm::world();
  soma::HostMesh s = refined_sphere(c);
  const auto q = soma::quality_smooth(c, s, {.angle_sweeps = 0, .sphere_radius = kRadius});
  if (c.rank() == 0) {
    std::printf("sphere, shape only: least angle %.3f -> %.3f\n", q.least_angle_before, q.least_angle_after);
  }
  EXPECT_GE(q.least_angle_after, q.least_angle_before);
}

// One free vertex in a 2 x 2 patch whose condition-number optimum lowers the
// least angle (50.8 to 40.1 degrees, found by search): the shape step moves
// it only as far as the least angle allows.
TEST(QualitySmooth, ShapeStopsAtTheLeastAngle) {
  soma::HostMesh m = grid(2);
  const std::map<std::pair<int, int>, std::array<Real, 2>> at{
      {{0, 0}, {0.0, 0.0}},     {{1, 0}, {0.883, 0.094}}, {{2, 0}, {2.0, 0.0}},   {{2, 1}, {1.562, 1.304}},
      {{2, 2}, {2.0, 2.0}},     {{1, 2}, {1.446, 1.973}}, {{0, 2}, {0.0, 2.0}},   {{0, 1}, {0.125, 0.686}},
      {{1, 1}, {1.081, 1.221}}};
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) {
    const auto p = at.at({static_cast<int>(std::lround(2 * x(v, 0))), static_cast<int>(std::lround(2 * x(v, 1)))});
    x(v, 0) = p[0];
    x(v, 1) = p[1];
  }
  const auto before = by_gid(m);
  const soma::Comm self = soma::Comm::self();
  const auto r = soma::quality_smooth(self, m, {.angle_sweeps = 0, .polish_sweeps = 0});
  std::printf("floor: least angle %.3f -> %.3f, moved %.3g\n", r.least_angle_before, r.least_angle_after,
              largest_move(m, before));
  EXPECT_NEAR(r.least_angle_before, 50.80, 0.01);
  EXPECT_GE(r.least_angle_after, r.least_angle_before);
  EXPECT_GT(largest_move(m, before), 1e-3);
}

namespace {

// Mean over the triangles of a planar mesh (z = 0) of their condition number
// against the equilateral triangle, sum of squared edges / (4 sqrt 3 area):
// 1 for an equilateral one, the same at each corner; infinite once inverted.
Real mean_tri_kappa(const soma::HostMesh& m) {
  const auto mv = m.view();
  const auto x = m.coords();
  Real s = 0;
  for (LO e = 0; e < m.count(2); ++e) {
    const LO a = mv.vertex(2, e, 0), b = mv.vertex(2, e, 1), c = mv.vertex(2, e, 2);
    const Real u[2] = {x(b, 0) - x(a, 0), x(b, 1) - x(a, 1)}, w[2] = {x(c, 0) - x(a, 0), x(c, 1) - x(a, 1)};
    const Real area = (u[0] * w[1] - u[1] * w[0]) / 2;
    const Real edges = u[0] * u[0] + u[1] * u[1] + w[0] * w[0] + w[1] * w[1] + (w[0] - u[0]) * (w[0] - u[0]) +
                       (w[1] - u[1]) * (w[1] - u[1]);
    s += area > 0 ? edges / (4 * std::sqrt(Real{3}) * area) : std::numeric_limits<Real>::infinity();
  }
  return s / static_cast<Real>(m.count(2));
}

soma::HostMesh tri_grid(int n) { return soma::testing::box_mesh({n, n, 1, soma::Topology::tri}); }

}  // namespace

// Triangles are measured against the equilateral one: a lattice of them is
// a fixed point (its sheared grid's diagonals are the short ones).
TEST(QualitySmooth, EquilateralLatticeIsAFixedPoint) {
  soma::HostMesh m = tri_grid(8);
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) {
    const Real y = x(v, 1);
    x(v, 0) -= y / 2;
    x(v, 1) = y * std::sqrt(Real{3}) / 2;
  }
  ASSERT_NEAR(mean_tri_kappa(m), 1, 1e-12);
  const auto before = by_gid(m);
  (void)soma::quality_smooth(soma::Comm::self(), m, {.angle_sweeps = 0, .polish_sweeps = 0});
  EXPECT_LE(largest_move(m, before), 1e-14);
}

TEST(QualitySmooth, ShapeImprovesPerturbedTriangles) {
  soma::HostMesh m = tri_grid(12);
  perturb_inner(m, 12, 0.3);
  const soma::Comm self = soma::Comm::self();
  const Real k0 = mean_tri_kappa(m);
  const auto r = soma::quality_smooth(self, m, {.angle_sweeps = 0, .polish_sweeps = 0});
  const Real k1 = mean_tri_kappa(m);
  std::printf("triangles: mean kappa %.4f -> %.4f, least angle %.2f -> %.2f\n", k0, k1, r.least_angle_before,
              r.least_angle_after);
  EXPECT_TRUE(std::isfinite(k1));
  EXPECT_LT(k1, k0);
  EXPECT_GE(r.least_angle_after, r.least_angle_before);
}

namespace {

// The cubed sphere ne 8 refined twice about a face's middle (cells within 0.5
// radians, then their children within 0.3), with a ghost layer through
// vertices on more than one rank: two levels and two bands of transitions.
soma::HostMesh two_level_sphere(const soma::Comm& c) {
  soma::sem::CubedSphereSpec spec;
  spec.ne = 8;
  spec.radius = kRadius;
  soma::HostMesh m = soma::sem::cubed_sphere_mesh(c, spec);
  const Real d[3] = {1 / std::sqrt(1.13), 0.3 / std::sqrt(1.13), 0.2 / std::sqrt(1.13)};
  for (const Real reach : {Real{0.5}, Real{0.3}}) {
    const auto mv = m.view();
    std::vector<std::int8_t> marks;
    for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
      Real g[3] = {0, 0, 0};
      for (LO i = 0; i < 4; ++i) {
        const auto p = point(m, mv.vertex(2, e, i));
        for (int q = 0; q < 3; ++q) g[q] += p[static_cast<std::size_t>(q)];
      }
      const Real cosine = (g[0] * d[0] + g[1] * d[1] + g[2] * d[2]) / std::hypot(g[0], g[1], g[2]);
      marks.push_back(std::acos(std::min(cosine, Real{1})) < reach ? 1 : 0);
    }
    m = soma::refine_quads(c, m, marks, {.sphere_radius = kRadius}).mesh;
  }
  if (c.size() > 1) m.add_ghosts(c, soma::GhostOptions{1, 0});
  return m;
}

// The sum over the owned cells' corners of their condition number against
// the square, each corner in its cell's plane.
Real total_kappa(const soma::Comm& c, const soma::HostMesh& m) {
  const auto mv = m.view();
  Real s = 0;
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    for (LO i = 0; i < 4; ++i) {
      const auto q = point(m, mv.vertex(2, e, i)), a = point(m, mv.vertex(2, e, (i + 3) % 4)),
                 b = point(m, mv.vertex(2, e, (i + 1) % 4));
      const Real u[3] = {b[0] - q[0], b[1] - q[1], b[2] - q[2]}, w[3] = {a[0] - q[0], a[1] - q[1], a[2] - q[2]};
      const Real n[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
      s += (u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) /
           (2 * std::hypot(n[0], n[1], n[2]));
    }
  }
  return c.allreduce(s, soma::Op::sum);
}

}  // namespace

TEST(QualitySmooth, StaysOnTheSphere) {
  const soma::Comm c = soma::Comm::world();
  soma::HostMesh m = refined_sphere(c);
  (void)soma::quality_smooth(c, m, {.shape_sweeps = 5, .angle_sweeps = 0, .sphere_radius = kRadius});
  EXPECT_LE(worst_off_sphere(c, m), kOnSphere);
}

namespace {

// Each vertex's coordinate bits after smoothing on `c` with `o`, compared
// with the serial run's by gid: the vertices that differ, over every rank.
Real differ_from_serial(const soma::Comm& c, const soma::QualitySmoothOptions& o) {
  const soma::Comm self = soma::Comm::self();
  soma::HostMesh serial = refined_sphere(self);
  (void)soma::quality_smooth(self, serial, o);
  std::map<GO, std::array<std::uint64_t, 3>> want;
  for (LO v = 0; v < serial.count(0); ++v) {
    const auto p = point(serial, v);
    want[serial.gid(0)(v)] = {std::bit_cast<std::uint64_t>(p[0]), std::bit_cast<std::uint64_t>(p[1]),
                              std::bit_cast<std::uint64_t>(p[2])};
  }
  soma::HostMesh m = dealt_sphere(c);
  (void)soma::quality_smooth(c, m, o);
  Real differ = 0;
  for (LO v = 0; v < m.count(0); ++v) {
    const auto p = point(m, v);
    const std::array<std::uint64_t, 3> got{std::bit_cast<std::uint64_t>(p[0]), std::bit_cast<std::uint64_t>(p[1]),
                                           std::bit_cast<std::uint64_t>(p[2])};
    const auto it = want.find(m.gid(0)(v));
    differ += it != want.end() && it->second == got ? 0 : 1;
  }
  return c.allreduce(differ, soma::Op::sum);
}

}  // namespace

TEST(QualitySmooth, SameBitsOnAnyRankCount) {
  const soma::Comm c = soma::Comm::world();
  EXPECT_EQ(differ_from_serial(c, {.shape_sweeps = 5, .angle_sweeps = 0, .sphere_radius = kRadius}), 0);
}

TEST(QualitySmooth, KeepsSkinAndMask) {
  const soma::Comm self = soma::Comm::self();
  soma::HostMesh m = grid(12);
  perturb_inner(m, 12, 0.3);
  const auto x = m.coords();
  // The skin stays.
  const auto start = by_gid(m);
  (void)soma::quality_smooth(self, m, {.shape_sweeps = 3, .angle_sweeps = 3});
  for (LO v = 0; v < m.count(0); ++v) {
    const bool side = x(v, 0) <= 0 || x(v, 0) >= 1 || x(v, 1) <= 0 || x(v, 1) >= 1;
    if (side) EXPECT_EQ(point(m, v), start.at(m.gid(0)(v)));
  }
  // A mask: every vertex fixed but the one of least gid inside.
  soma::HostMesh k = grid(12);
  perturb_inner(k, 12, 0.3);
  const auto before = by_gid(k);
  const auto y = k.coords();
  std::vector<std::uint8_t> fixed(static_cast<std::size_t>(k.count(0)), 1);
  LO free = -1;
  for (LO v = 0; v < k.count(0); ++v) {
    const bool inner = y(v, 0) > 0 && y(v, 0) < 1 && y(v, 1) > 0 && y(v, 1) < 1;
    if (inner && (free < 0 || k.gid(0)(v) < k.gid(0)(free))) free = v;
  }
  fixed[static_cast<std::size_t>(free)] = 0;
  (void)soma::quality_smooth(self, k, {.shape_sweeps = 3, .angle_sweeps = 3, .fixed = fixed});
  Real moved_free = 0;
  for (LO v = 0; v < k.count(0); ++v) {
    const auto p = point(k, v);
    if (v == free) {
      moved_free = std::hypot(p[0] - before.at(k.gid(0)(v))[0], p[1] - before.at(k.gid(0)(v))[1]);
    } else {
      EXPECT_EQ(p, before.at(k.gid(0)(v)));
    }
  }
  EXPECT_GT(moved_free, 0);
}

namespace {

// quality_smooth throws std::invalid_argument on every rank.
void expect_refused(const soma::Comm& c, soma::HostMesh& m, const soma::QualitySmoothOptions& o) {
  Real threw = 0;
  try {
    (void)soma::quality_smooth(c, m, o);
  } catch (const std::invalid_argument&) {
    threw = 1;
  }
  EXPECT_EQ(c.allreduce(threw, soma::Op::min), 1);
}

}  // namespace

TEST(QualitySmooth, OptionsAreChecked) {
  const soma::Comm c = soma::Comm::world();
  const Real inf = std::numeric_limits<Real>::infinity(), nan = std::numeric_limits<Real>::quiet_NaN();
  {
    soma::HostMesh hex = soma::testing::box_mesh(c, {2, 2, 2, soma::Topology::hex});
    if (c.size() > 1) hex.add_ghosts(c, soma::GhostOptions{1, 0});
    expect_refused(c, hex, {});
  }
  soma::HostMesh m = refined_sphere(c);
  const soma::QualitySmoothOptions ok{.sphere_radius = kRadius};
  for (const int s : {-1}) {
    expect_refused(c, m, {.shape_sweeps = s, .sphere_radius = kRadius});
    expect_refused(c, m, {.angle_sweeps = s, .sphere_radius = kRadius});
    expect_refused(c, m, {.polish_sweeps = s, .sphere_radius = kRadius});
  }
  for (const Real t : {Real{0}, Real{91}, nan, -Real{1}}) {
    expect_refused(c, m, {.angle_threshold = t, .sphere_radius = kRadius});
  }
  for (const Real t : {-Real{1}, inf, nan}) expect_refused(c, m, {.tolerance = t, .sphere_radius = kRadius});
  for (const Real r : {Real{0}, -Real{1}, inf, nan}) expect_refused(c, m, {.sphere_radius = r});
  const std::vector<std::uint8_t> short_mask(1, 0);
  expect_refused(c, m, {.fixed = short_mask, .sphere_radius = kRadius});
  if (c.size() > 1) {
    // One rank differs.
    expect_refused(c, m, {.shape_sweeps = c.rank() == 1 ? 3 : 10, .sphere_radius = kRadius});
    expect_refused(c, m, {.polish_sweeps = c.rank() == 1 ? 3 : 10, .sphere_radius = kRadius});
    expect_refused(c, m, {.angle_threshold = c.rank() == 1 ? Real{50} : Real{60}, .sphere_radius = kRadius});
    soma::QualitySmoothOptions one = ok;
    if (c.rank() == 1) one.sphere_radius.reset();
    expect_refused(c, m, one);
    soma::HostMesh bare = refined_sphere(c, false);
    expect_refused(c, bare, ok);
  }
  // An inverted cell: an inner vertex pushed past its neighbour.
  soma::HostMesh g = soma::testing::box_mesh(c, {4, 4, 1, soma::Topology::quad});
  if (c.size() > 1) g.add_ghosts(c, soma::GhostOptions{1, 0});
  const auto x = g.coords();
  for (LO v = 0; v < g.count(0); ++v) {
    if (std::abs(x(v, 0) - 0.25) < 1e-12 && std::abs(x(v, 1) - 0.25) < 1e-12) x(v, 0) = 0.6;
  }
  expect_refused(c, g, {});
  // Untouched by refusals: the mesh smooths afterwards.
  EXPECT_NO_THROW((void)soma::quality_smooth(c, m, ok));
}

// The angle phase alone raises the refined sphere's least angle; the
// defaults (both phases) never lower it.
TEST(QualitySmooth, AngleNeverFalls) {
  const soma::Comm c = soma::Comm::world();
  {
    soma::HostMesh m = refined_sphere(c);
    const Real aspect = quality(c, m).worst_aspect;
    const auto r = soma::quality_smooth(c, m, {.shape_sweeps = 0, .polish_sweeps = 0, .sphere_radius = kRadius});
    const Real after = quality(c, m).worst_aspect;
    if (c.rank() == 0) {
      std::printf("angle only: %.3f -> %.3f, worst aspect %.3f -> %.3f\n", r.least_angle_before, r.least_angle_after,
                  aspect, after);
    }
    EXPECT_GT(r.least_angle_after, r.least_angle_before);
    EXPECT_EQ(r.angle_sweeps, 5);
    // Angles do not bound a quad's aspect: steps keep each corner within the
    // patch's worst condition number, which bounds the stretching (measured
    // 3.26 -> 3.95; without the bound, 22.6).
    EXPECT_LE(after, 1.25 * aspect);
  }
  soma::HostMesh m = refined_sphere(c);
  const auto r = soma::quality_smooth(c, m, {.sphere_radius = kRadius});
  if (c.rank() == 0) std::printf("both: %.3f -> %.3f\n", r.least_angle_before, r.least_angle_after);
  EXPECT_GE(r.least_angle_after, r.least_angle_before);
  EXPECT_EQ(r.polish_sweeps, 10);
  // Polishing keeps what the angle phase reached.
  soma::HostMesh unpolished = refined_sphere(c);
  const auto u = soma::quality_smooth(c, unpolished, {.polish_sweeps = 0, .sphere_radius = kRadius});
  EXPECT_GE(r.least_angle_after, u.least_angle_after);
}

// Against 20 centroid sweeps: a higher least angle and mean least angle, and
// the level-1 cells' mean spacing within 5% of the unsmoothed mesh's.
TEST(QualitySmooth, BeatsCentroids) {
  const soma::Comm c = soma::Comm::world();
  const auto fine_spacing = [&](const soma::HostMesh& m) {
    const auto level = m.fields().get<std::int32_t>("refine:level", 2).view();
    Real s[2] = {0, 0};
    for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
      if (level(static_cast<std::size_t>(e), 0) != 1) continue;
      s[0] += std::sqrt(area(m, e));
      s[1] += 1;
    }
    c.allreduce(std::span<Real>(s), soma::Op::sum);
    return s[0] / s[1];
  };
  const soma::HostMesh start = refined_sphere(c);
  soma::HostMesh centroids = refined_sphere(c);
  (void)soma::laplacian_smooth(c, centroids,
                               {.iterations = 20, .method = soma::SmoothMethod::centroids, .sphere_radius = kRadius});
  soma::HostMesh optimized = refined_sphere(c);
  (void)soma::quality_smooth(c, optimized, {.sphere_radius = kRadius});
  const Real s0 = fine_spacing(start), sc = fine_spacing(centroids), so = fine_spacing(optimized);
  const Quality q0 = quality(c, start), qc = quality(c, centroids), qo = quality(c, optimized);
  const Real lc = least_angle(c, centroids), lo = least_angle(c, optimized);
  if (c.rank() == 0) {
    std::printf("least %.2f (centroids) %.2f (optimized); mean %.2f %.2f; fine spacing change %.4f %.4f; worst "
                "aspect %.3f (start) %.3f %.3f\n",
                lc, lo, qc.mean_angle, qo.mean_angle, sc / s0 - 1, so / s0 - 1, q0.worst_aspect, qc.worst_aspect,
                qo.worst_aspect);
  }
  EXPECT_GT(lo, lc);
  EXPECT_GT(qo.mean_angle, qc.mean_angle);
  // Cells are not stretched more than centroid smoothing stretches them.
  EXPECT_LE(qo.worst_aspect, qc.worst_aspect);
  EXPECT_LE(std::abs(so / s0 - 1), 0.05);
}

TEST(QualitySmooth, AngleSameBitsOnAnyRankCount) {
  const soma::Comm c = soma::Comm::world();
  EXPECT_EQ(differ_from_serial(c, {.shape_sweeps = 0, .sphere_radius = kRadius}), 0);
}



// Two levels of refinement relax slowly (improvement spreads a cell a sweep):
// from 20 centroid sweeps, the defaults raise the mean least angle by a degree
// or more (measured 81.71 to 83.01; 10 unrelaxed steepest-descent sweeps made
// 82.27) and the least angle, with no more cells under 45 degrees and no
// worse aspect than centroids leave.
TEST(QualitySmooth, DefaultsBeatCentroidsOnTwoLevels) {
  const soma::Comm c = soma::Comm::world();
  soma::HostMesh centroids = two_level_sphere(c);
  (void)soma::laplacian_smooth(c, centroids,
                               {.iterations = 20, .method = soma::SmoothMethod::centroids, .sphere_radius = kRadius});
  soma::HostMesh optimized = two_level_sphere(c);
  (void)soma::laplacian_smooth(c, optimized,
                               {.iterations = 20, .method = soma::SmoothMethod::centroids, .sphere_radius = kRadius});
  (void)soma::quality_smooth(c, optimized, {.sphere_radius = kRadius});
  const Quality qc = quality(c, centroids), qo = quality(c, optimized);
  const Real lc = least_angle(c, centroids), lo = least_angle(c, optimized);
  const Real kc = total_kappa(c, centroids), ko = total_kappa(c, optimized);  // collective: every rank
  if (c.rank() == 0) {
    std::printf("two levels: least %.2f -> %.2f, mean %.3f -> %.3f, under 45 %g -> %g, aspect %.3f -> %.3f, "
                "kappa %.1f -> %.1f\n",
                lc, lo, qc.mean_angle, qo.mean_angle, qc.under_45, qo.under_45, qc.worst_aspect, qo.worst_aspect, kc,
                ko);
  }
  EXPECT_GE(qo.mean_angle, qc.mean_angle + 1);
  EXPECT_GT(lo, lc + 5);
  EXPECT_LE(qo.under_45, qc.under_45);
  EXPECT_LE(qo.worst_aspect, qc.worst_aspect);
}

namespace {

// `m` with two ghost layers through vertices in place of one (more than one rank).
soma::HostMesh two_layers(const soma::Comm& c, soma::HostMesh m) {
  if (c.size() > 1) {
    m.remove_ghosts(c);
    m.add_ghosts(c, soma::GhostOptions{2, 0});
  }
  return m;
}

// The largest area ratio (larger over smaller) of an owned cell to a local cell
// sharing two of its corners, over every rank: found through the cells of each
// corner, apart from the library's edge tables.
Real brute_ratio(const soma::Comm& c, const soma::HostMesh& m) {
  const auto mv = m.view();
  const auto around = soma::detail::vertex_cells(m);
  Real most = 1;
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    const Real a = area(m, e);
    for (LO i = 0; i < mv.num_vertices(2, e); ++i) {
      const auto v = static_cast<std::size_t>(mv.vertex(2, e, i));
      for (auto k = around.offsets[v]; k < around.offsets[v + 1]; ++k) {
        const LO d = around.cells[static_cast<std::size_t>(k)];
        if (d == e) continue;
        int shared = 0;
        for (LO p = 0; p < mv.num_vertices(2, e); ++p) {
          for (LO q = 0; q < mv.num_vertices(2, d); ++q) shared += mv.vertex(2, e, p) == mv.vertex(2, d, q) ? 1 : 0;
        }
        if (shared < 2) continue;
        const Real b = area(m, d);
        most = std::max(most, a > b ? a / b : b / a);
      }
    }
  }
  return c.allreduce(most, soma::Op::max);
}

}  // namespace

// With the edge stencil, no two corners of two cells sharing an edge share a
// colour, ghost cells included, and every vertex has one.
TEST(QualitySmooth, StencilColoursSeparateEdgeNeighbours) {
  const soma::Comm c = soma::Comm::world();
  const soma::HostMesh m = two_layers(c, dealt_sphere(c));
  const auto cells = soma::detail::vertex_cells(m);
  const auto neighbours = soma::detail::edge_neighbours(m, cells);
  const auto colour = soma::detail::vertex_colours(c, m, soma::detail::edge_stencil(m, cells, neighbours));
  const auto mv = m.view();
  Real bad = 0;
  for (LO v = 0; v < m.count(0); ++v) bad += colour[static_cast<std::size_t>(v)] < 0 ? 1 : 0;
  for (LO e = 0; e < m.count(2); ++e) {
    const auto ei = static_cast<std::size_t>(e);
    for (auto k = neighbours.offsets[ei]; k < neighbours.offsets[ei + 1]; ++k) {
      const LO d = neighbours.cells[static_cast<std::size_t>(k)];
      if (d < 0) continue;
      std::set<LO> corners;
      std::set<std::int32_t> seen;
      for (const LO f : {e, d}) {
        for (LO i = 0; i < mv.num_vertices(2, f); ++i) corners.insert(mv.vertex(2, f, i));
      }
      for (const LO v : corners) seen.insert(colour[static_cast<std::size_t>(v)]);
      bad += seen.size() == corners.size() ? 0 : 1;
    }
  }
  EXPECT_EQ(c.allreduce(bad, soma::Op::sum), 0);
}

// Where two levels meet, shape sweeps trade size for shape (the unbounded run
// raises the ratio); with the bound at the mesh's own largest ratio the ratio
// does not grow and the summed condition number still falls; with the bound at
// 1 no pair grows past where its patch was.
TEST(QualitySmooth, AreaRatioBoundHolds) {
  const soma::Comm c = soma::Comm::world();
  const soma::HostMesh start = two_layers(c, two_level_sphere(c));
  const Real before = brute_ratio(c, start);
  const Real kappa_before = total_kappa(c, start);
  const soma::QualitySmoothOptions sweeps{.shape_sweeps = 10, .angle_sweeps = 3, .polish_sweeps = 3,
                                          .sphere_radius = kRadius};
  soma::HostMesh free = start.clone();
  (void)soma::quality_smooth(c, free, sweeps);
  const Real unbounded = brute_ratio(c, free);
  EXPECT_GT(unbounded, before);
  for (const Real bound : {before, Real{1}}) {
    soma::HostMesh m = start.clone();
    soma::QualitySmoothOptions o = sweeps;
    o.area_ratio_bound = bound;
    const auto r = soma::quality_smooth(c, m, o);
    const Real after = brute_ratio(c, m);
    const Real kappa_after = total_kappa(c, m);
    if (c.rank() == 0) {
      std::printf("area ratio: start %.4f, unbounded %.4f, bound %.4f -> %.4f; kappa %.1f -> %.1f\n", before,
                  unbounded, bound, after, kappa_before, kappa_after);
    }
    EXPECT_LE(after, std::max(bound, before) * (1 + 1e-12));
    EXPECT_NEAR(r.area_ratio_before, before, 1e-12 * before);
    EXPECT_NEAR(r.area_ratio_after, after, 1e-12 * after);
    EXPECT_LT(kappa_after, kappa_before);
    EXPECT_GE(r.least_angle_after, r.least_angle_before);
  }
}

// The bound's result is the same bit for bit on any number of ranks.
TEST(QualitySmooth, AreaRatioSameBitsOnAnyRankCount) {
  const soma::Comm c = soma::Comm::world();
  const soma::Comm self = soma::Comm::self();
  const soma::QualitySmoothOptions o{.shape_sweeps = 5, .angle_sweeps = 3, .polish_sweeps = 3,
                                     .area_ratio_bound = 1.5, .sphere_radius = kRadius};
  soma::HostMesh serial = refined_sphere(self, false);
  (void)soma::quality_smooth(self, serial, o);
  std::map<GO, std::array<std::uint64_t, 3>> want;
  for (LO v = 0; v < serial.count(0); ++v) {
    const auto p = point(serial, v);
    want[serial.gid(0)(v)] = {std::bit_cast<std::uint64_t>(p[0]), std::bit_cast<std::uint64_t>(p[1]),
                              std::bit_cast<std::uint64_t>(p[2])};
  }
  soma::HostMesh m = two_layers(c, dealt_sphere(c));
  (void)soma::quality_smooth(c, m, o);
  Real differ = 0;
  for (LO v = 0; v < m.count(0); ++v) {
    const auto p = point(m, v);
    const std::array<std::uint64_t, 3> got{std::bit_cast<std::uint64_t>(p[0]), std::bit_cast<std::uint64_t>(p[1]),
                                           std::bit_cast<std::uint64_t>(p[2])};
    const auto it = want.find(m.gid(0)(v));
    differ += it != want.end() && it->second == got ? 0 : 1;
  }
  EXPECT_EQ(c.allreduce(differ, soma::Op::sum), 0);
}

TEST(QualitySmooth, AreaRatioOptionsAreChecked) {
  const soma::Comm c = soma::Comm::world();
  const Real inf = std::numeric_limits<Real>::infinity(), nan = std::numeric_limits<Real>::quiet_NaN();
  soma::HostMesh m = two_layers(c, refined_sphere(c));
  for (const Real b : {Real{0.5}, -Real{1}, inf, nan}) {
    expect_refused(c, m, {.area_ratio_bound = b, .sphere_radius = kRadius});
  }
  if (c.size() > 1) {
    expect_refused(c, m, {.area_ratio_bound = c.rank() == 1 ? Real{2} : Real{3}, .sphere_radius = kRadius});
    soma::HostMesh one = refined_sphere(c);  // one ghost layer
    expect_refused(c, one, {.area_ratio_bound = 2, .sphere_radius = kRadius});
    EXPECT_NO_THROW((void)soma::quality_smooth(c, one, {.shape_sweeps = 1, .sphere_radius = kRadius}));
  }
  EXPECT_NO_THROW(
      (void)soma::quality_smooth(c, m, {.shape_sweeps = 1, .area_ratio_bound = 2, .sphere_radius = kRadius}));
}
