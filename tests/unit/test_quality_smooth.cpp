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
  (void)soma::quality_smooth(soma::Comm::self(), m, {.angle_sweeps = 0});
  EXPECT_LE(largest_move(m, before), 1e-14);
}

TEST(QualitySmooth, ShapeImprovesAPerturbedGrid) {
  soma::HostMesh m = grid(12);
  perturb_inner(m, 12, 0.3);
  const soma::Comm self = soma::Comm::self();
  const Real k0 = mean_kappa(m), a0 = least_angle(self, m);
  const auto r = soma::quality_smooth(self, m, {.angle_sweeps = 0});
  const Real k1 = mean_kappa(m), a1 = least_angle(self, m);
  std::printf("shape: mean kappa %.4f -> %.4f, least angle %.2f -> %.2f\n", k0, k1, a0, a1);
  EXPECT_LT(k1, k0);
  EXPECT_TRUE(std::isfinite(k1));
  EXPECT_GE(a1, a0);
  EXPECT_EQ(r.shape_sweeps, 10);
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
  const auto r = soma::quality_smooth(self, m, {.angle_sweeps = 0});
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
