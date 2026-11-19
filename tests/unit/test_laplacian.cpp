// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <cstdint>
#include <map>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "distributed_fixture.hpp"
#include "refined_sphere.hpp"
#include "soma/algorithms/laplacian.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/sem/cubed_sphere.hpp"
#include "soma/testing/box_mesh.hpp"
#if SOMA_TEST_HDF5
#include "soma/io/h5m.hpp"
#endif

using namespace soma_test;
using soma::LO;
using soma::Offset;
using soma::Topology;

namespace {

soma::HostMesh box(int n, Topology t) {
  return soma::testing::box_mesh({n, n, t == Topology::hex ? n : 1, t});
}

std::size_t skin_count(const soma::detail::SmoothingGraph& g) {
  std::size_t k = 0;
  for (const auto s : g.skin) k += s;
  return k;
}

// A vertex none of whose coordinates is on the box's faces [0, 1].
LO interior_vertex(const soma::HostMesh& m) {
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) {
    bool inside = true;
    for (int q = 0; q < m.dim(); ++q) inside = inside && x(v, q) > 0 && x(v, q) < 1;
    if (inside) return v;
  }
  return -1;
}

// A displacement of each vertex by its gid, the same on every rank.
std::array<Real, 3> offset_of(GO g) {
  const auto t = static_cast<Real>(g);
  return {0.2 * std::sin(t), 0.2 * std::cos(3 * t), 0.1 * std::sin(7 * t)};
}

void perturb(soma::HostMesh& m, Real scale) {
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) {
    const auto d = offset_of(m.gid(0)(v));
    for (int q = 0; q < m.dim(); ++q) x(v, q) += scale * d[static_cast<std::size_t>(q)];
  }
}

std::map<GO, std::array<Real, 3>> by_gid(const soma::HostMesh& m) {
  std::map<GO, std::array<Real, 3>> out;
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) out[m.gid(0)(v)] = {x(v, 0), x(v, 1), x(v, 2)};
  return out;
}

}  // namespace

// Rows hold the other corners of the entities of the bridge dimension
// around a vertex, by gid; the skin is the corners of sides one cell has.
TEST(Laplacian, NeighboursAndSkinOfBoxes) {
  struct Case {
    Topology t;
    int n;
    std::size_t skin;
    std::vector<std::pair<int, Offset>> rows;  // (bridge_dim, interior row length)
  };
  const std::vector<Case> cases{{Topology::hex, 4, 125 - 27, {{3, 26}, {2, 18}, {1, 6}}},
                                {Topology::quad, 3, 16 - 4, {{2, 8}, {1, 4}}},
                                {Topology::tri, 3, 16 - 4, {}}};
  for (const Case& c : cases) {
    SCOPED_TRACE(soma::name(c.t));
    const soma::HostMesh m = box(c.n, c.t);
    for (int b = 1; b <= m.dim(); ++b) {
      const auto g = soma::detail::smoothing_graph(m, b);
      EXPECT_EQ(skin_count(g), c.skin);
      const auto gid_at = [&](Offset k) { return m.gid(0)(g.targets[static_cast<std::size_t>(k)]); };
      for (std::size_t v = 0; v < static_cast<std::size_t>(m.count(0)); ++v) {
        for (Offset k = g.offsets[v] + 1; k < g.offsets[v + 1]; ++k) EXPECT_LT(gid_at(k - 1), gid_at(k));
      }
      for (const auto& [bd, len] : c.rows) {
        if (bd != b) continue;
        const auto v = static_cast<std::size_t>(interior_vertex(m));
        EXPECT_EQ(g.offsets[v + 1] - g.offsets[v], len) << "bridge " << b;
      }
    }
  }
}

// A uniform grid is a fixed point: perturbed inside, it comes back, while
// the skin stays.
TEST(Laplacian, PerturbedBoxesReturnToTheGrid) {
  const soma::Comm self = soma::Comm::self();
  for (const int b : {-1, 1}) {
    SCOPED_TRACE(testing::Message() << "bridge " << b);
    soma::HostMesh m = box(6, Topology::hex);
    const auto grid = by_gid(m);
    const auto g = soma::detail::smoothing_graph(m, 3);
    perturb(m, 0.15);
    const auto x = m.coords();
    for (LO v = 0; v < m.count(0); ++v) {
      if (g.skin[static_cast<std::size_t>(v)] == 0) continue;
      for (int q = 0; q < 3; ++q) x(v, q) = grid.at(m.gid(0)(v))[static_cast<std::size_t>(q)];
    }
    soma::SmoothOptions o;
    o.iterations = 2000;
    o.bridge_dim = b;
    EXPECT_EQ(soma::laplacian_smooth(self, m, o).iterations, 2000);
    for (const auto& [gid, p] : by_gid(m)) {
      for (std::size_t q = 0; q < 3; ++q) EXPECT_NEAR(p[q], grid.at(gid)[q], 1e-12);
    }
  }
}

TEST(Laplacian, ToleranceMaskAndZeroIterations) {
  const soma::Comm self = soma::Comm::self();
  soma::HostMesh m = box(5, Topology::hex);
  perturb(m, 0.1);
  const auto before = by_gid(m);
  soma::SmoothOptions o;
  o.iterations = 0;
  EXPECT_EQ(soma::laplacian_smooth(self, m, o).iterations, 0);
  EXPECT_EQ(by_gid(m), before);
  const std::vector<std::uint8_t> all(static_cast<std::size_t>(m.count(0)), 1);
  o.iterations = 5;
  o.fixed = all;
  EXPECT_EQ(soma::laplacian_smooth(self, m, o).iterations, 5);
  EXPECT_EQ(by_gid(m), before);
  o.fixed = {};
  o.iterations = 2000;
  o.tolerance = 1e-3;
  const soma::SmoothReport r = soma::laplacian_smooth(self, m, o);
  EXPECT_LT(r.iterations, 2000);
  EXPECT_GT(r.iterations, 1);
  EXPECT_LE(r.largest_move, 1e-3);
  EXPECT_GT(r.largest_move, 0);
}

TEST(Laplacian, BadOptionsThrow) {
  const soma::Comm self = soma::Comm::self();
  soma::HostMesh m = box(2, Topology::quad);
  const auto bad = [&](soma::SmoothOptions o) {
    EXPECT_THROW((void)soma::laplacian_smooth(self, m, o), std::invalid_argument);
  };
  bad({.iterations = -1});
  bad({.bridge_dim = 0});
  bad({.bridge_dim = 3});
  bad({.tolerance = -1});
  bad({.tolerance = std::nan("")});
  const std::vector<std::uint8_t> short_mask(3, 0);
  bad({.fixed = short_mask});
}

// On every partition, with one ghost layer, the smoothed coordinates are
// the serial ones, bit for bit; without ghosts, or with differing options,
// every rank throws.
TEST(Laplacian, DistributedSweepsAreTheSerialOnes) {
  const Comm c = Comm::world();
  std::vector<GO> all;
  for (int cell = 0; cell < nx * ny; ++cell) all.push_back(1000 + cell);
  soma::HostMesh whole = soma::build_local_part(Comm::self(), cells_builder(all));
  perturb(whole, 0.3);
  soma::SmoothOptions o;
  o.iterations = 25;
  (void)soma::laplacian_smooth(Comm::self(), whole, o);
  const auto want = by_gid(whole);
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(testing::Message() << "partition " << kind);
    soma::HostMesh m = soma::build_local_part(c, local_builder(partition(c, kind), c.rank()));
    if (c.size() > 1) EXPECT_THROW((void)soma::laplacian_smooth(c, m, o), std::invalid_argument);
    m.add_ghosts(c, soma::GhostOptions{1, 0, false});
    perturb(m, 0.3);
    (void)soma::laplacian_smooth(c, m, o);
    for (const auto& [gid, p] : by_gid(m)) EXPECT_EQ(p, want.at(gid)) << "vertex " << gid;
    soma::SmoothOptions other = o;
    other.iterations = c.rank() == 0 ? 3 : 4;
    if (c.size() > 1) EXPECT_THROW((void)soma::laplacian_smooth(c, m, other), std::invalid_argument);
  }
}

// With a tolerance, every rank stops after the sweep the serial run stops
// after, on the same largest move, at the same coordinates.
TEST(Laplacian, DistributedToleranceStopsWhereSerialDoes) {
  const Comm c = Comm::world();
  std::vector<GO> all;
  for (int cell = 0; cell < nx * ny; ++cell) all.push_back(1000 + cell);
  soma::HostMesh whole = soma::build_local_part(Comm::self(), cells_builder(all));
  perturb(whole, 0.3);
  soma::SmoothOptions o;
  o.iterations = 1000;
  o.tolerance = 1e-4;
  const soma::SmoothReport serial = soma::laplacian_smooth(Comm::self(), whole, o);
  ASSERT_LT(serial.iterations, o.iterations);
  const auto want = by_gid(whole);
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(testing::Message() << "partition " << kind);
    soma::HostMesh m = soma::build_local_part(c, local_builder(partition(c, kind), c.rank()));
    m.add_ghosts(c, soma::GhostOptions{1, 0, false});
    perturb(m, 0.3);
    const soma::SmoothReport r = soma::laplacian_smooth(c, m, o);
    EXPECT_EQ(r.iterations, serial.iterations);
    EXPECT_EQ(r.largest_move, serial.largest_move);
    for (const auto& [gid, p] : by_gid(m)) EXPECT_EQ(p, want.at(gid)) << "vertex " << gid;
  }
}

#if SOMA_TEST_HDF5
// A hex mesh read in parallel with its file's parts smooths as it does read
// whole on one rank.
TEST(Laplacian, HexFilesSmoothAsOnOneRank) {
  const Comm c = Comm::world();
  const std::string file = std::string(SOMA_TEST_DATA_DIR) + "/64bricks_1khex.h5m";
  soma::H5mReadOptions ro;
  ro.tags = false;
  soma::HostMesh whole = soma::read_h5m(Comm::self(), file, ro);
  perturb(whole, 0.01);
  soma::SmoothOptions o;
  o.iterations = 10;
  (void)soma::laplacian_smooth(Comm::self(), whole, o);
  const auto want = by_gid(whole);
  soma::HostMesh m = soma::read_h5m(c, file, ro);
  m.add_ghosts(c, soma::GhostOptions{1, 0, false});
  perturb(m, 0.01);
  (void)soma::laplacian_smooth(c, m, o);
  for (const auto& [gid, p] : by_gid(m)) EXPECT_EQ(p, want.at(gid)) << "vertex " << gid;
}
#endif

// Centroid smoothing and the sphere.

// Each inner vertex of a uniform grid is the area-weighted mean of its four
// cells' centroids already.
TEST(Laplacian, CentroidsKeepAUniformGrid) {
  soma::HostMesh m = box(8, Topology::quad);
  std::vector<Real> before(static_cast<std::size_t>(3 * m.count(0)));
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) {
    for (int q = 0; q < 3; ++q) before[static_cast<std::size_t>(3 * v + q)] = x(v, q);
  }
  soma::SmoothOptions o;
  o.iterations = 3;
  o.method = soma::SmoothMethod::centroids;
  (void)soma::laplacian_smooth(soma::Comm::self(), m, o);
  Real worst = 0;
  for (LO v = 0; v < m.count(0); ++v) {
    for (int q = 0; q < 3; ++q) {
      worst = std::max(worst, std::abs(x(v, q) - before[static_cast<std::size_t>(3 * v + q)]));
    }
  }
  EXPECT_LE(worst, 1e-15);
}

TEST(Laplacian, CentroidsRelaxAPerturbedGrid) {
  constexpr int n = 12;
  constexpr Real h = 1.0 / n;
  soma::HostMesh m = box(n, Topology::quad);
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) {
    if (x(v, 0) <= 0 || x(v, 0) >= 1 || x(v, 1) <= 0 || x(v, 1) >= 1) continue;
    const auto d = offset_of(m.gid(0)(v));
    x(v, 0) += 1.5 * h * d[0];  // up to 0.3 h
    x(v, 1) += 1.5 * h * d[1];
  }
  const auto spread = [&] {
    Real worst = 0;
    for (LO e = 0; e < m.count(2); ++e) worst = std::max(worst, std::abs(area(m, e) / (h * h) - 1));
    return worst;
  };
  const Real before = spread();
  soma::SmoothOptions o;
  o.iterations = 10;
  o.method = soma::SmoothMethod::centroids;
  (void)soma::laplacian_smooth(soma::Comm::self(), m, o);
  std::printf("centroids: max |area / h^2 - 1| %.4f before, %.4f after 10 sweeps\n", before, spread());
  EXPECT_LE(spread(), 0.05);
}

// A closed sphere has no skin: every vertex moves. Projected, they stay on
// the sphere; averaged without it, the mesh shrinks.
TEST(Laplacian, SmoothingStaysOnTheSphere) {
  const soma::Comm c = soma::Comm::world();
  for (const auto method : {soma::SmoothMethod::neighbours, soma::SmoothMethod::centroids}) {
    soma::HostMesh m = refined_sphere(c);
    (void)soma::laplacian_smooth(c, m, {.iterations = 5, .method = method, .sphere_radius = kRadius});
    EXPECT_LE(worst_off_sphere(c, m), kOnSphere) << static_cast<int>(method);
  }
  soma::HostMesh m = refined_sphere(c);
  (void)soma::laplacian_smooth(c, m, {.iterations = 5});
  Real least = std::numeric_limits<Real>::max();
  for (LO v = 0; v < m.count(0); ++v) {
    const auto p = point(m, v);
    least = std::min(least, std::hypot(p[0], p[1], p[2]));
  }
  EXPECT_LT(c.allreduce(least, soma::Op::min), kRadius * (1 - 1e-6));
}

TEST(Laplacian, CentroidsSameBitsOnAnyRankCount) {
  const soma::Comm c = soma::Comm::world();
  const soma::Comm self = soma::Comm::self();
  const soma::SmoothOptions o{.iterations = 5, .method = soma::SmoothMethod::centroids, .sphere_radius = kRadius};
  soma::HostMesh serial = refined_sphere(self);
  (void)soma::laplacian_smooth(self, serial, o);
  std::map<GO, std::array<std::uint64_t, 3>> want;
  for (LO v = 0; v < serial.count(0); ++v) {
    const auto p = point(serial, v);
    want[serial.gid(0)(v)] = {std::bit_cast<std::uint64_t>(p[0]), std::bit_cast<std::uint64_t>(p[1]),
                              std::bit_cast<std::uint64_t>(p[2])};
  }
  soma::HostMesh m = refined_sphere(c);
  (void)soma::laplacian_smooth(c, m, o);
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


// On the refined sphere, 20 sweeps of either method raise the mean least
// angle and halve the cells under 45 degrees; centroids stretch cells less
// than neighbours and keep the fine cells' size within 10%. (Weighting the
// centroids by area grew them by 87%: the regression this guards.)
TEST(Laplacian, CentroidsKeepTheRefinement) {
  const soma::Comm c = soma::Comm::world();
  const auto fine_area = [&](const soma::HostMesh& m) {
    const auto level = m.fields().get<std::int32_t>("refine:level", 2).view();
    Real s[2] = {0, 0};
    for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
      if (level(static_cast<std::size_t>(e), 0) != 1) continue;
      s[0] += area(m, e);
      s[1] += 1;
    }
    c.allreduce(std::span<Real>(s), soma::Op::sum);
    return s[0] / s[1];
  };
  const soma::HostMesh start = refined_sphere(c);
  const Real before = fine_area(start);
  const Quality q0 = quality(c, start);
  Real change[2];
  Quality q[2];
  for (const auto method : {soma::SmoothMethod::neighbours, soma::SmoothMethod::centroids}) {
    soma::HostMesh m = refined_sphere(c);
    (void)soma::laplacian_smooth(c, m, {.iterations = 20, .method = method, .sphere_radius = kRadius});
    const auto k = static_cast<std::size_t>(method);
    change[k] = std::abs(fine_area(m) / before - 1);
    q[k] = quality(c, m);
  }
  if (c.rank() == 0) {
    std::printf("before: mean least angle %.2f, under 45 %g, worst aspect %.3f\n", q0.mean_angle, q0.under_45,
                q0.worst_aspect);
    for (std::size_t k = 0; k < 2; ++k) {
      std::printf("%s: fine-cell area change %.4f, mean least angle %.2f, under 45 %g, worst aspect %.3f\n",
                  k == 0 ? "neighbours" : "centroids", change[k], q[k].mean_angle, q[k].under_45,
                  q[k].worst_aspect);
    }
  }
  for (std::size_t k = 0; k < 2; ++k) {
    EXPECT_GT(q[k].mean_angle, q0.mean_angle) << k;
    EXPECT_LE(2 * q[k].under_45, q0.under_45) << k;
  }
  EXPECT_LT(q[1].worst_aspect, q[0].worst_aspect);
  EXPECT_LE(change[1], 0.1);
}

TEST(Laplacian, CentroidOptionsAreChecked) {
  const soma::Comm c = soma::Comm::world();
  soma::HostMesh sphere = refined_sphere(c);
  const auto throws = [&](soma::HostMesh& m, const soma::SmoothOptions& o, const char* what) {
    EXPECT_THROW((void)soma::laplacian_smooth(c, m, o), std::invalid_argument) << what;
  };
  soma::HostMesh hex = box(3, Topology::hex);
  if (c.size() == 1) {
    EXPECT_THROW((void)soma::laplacian_smooth(c, hex, {.method = soma::SmoothMethod::centroids}),
                 std::invalid_argument);
  }
  for (const Real r : {0.0, -1.0, std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::quiet_NaN()}) {
    throws(sphere, {.sphere_radius = r}, "a radius");
  }
  if (c.size() > 1) {
    soma::SmoothOptions one;
    if (c.rank() == 0) one.sphere_radius = kRadius;
    throws(sphere, one, "a radius on one rank");
    soma::SmoothOptions method;
    if (c.rank() == 0) method.method = soma::SmoothMethod::centroids;
    throws(sphere, method, "a method on one rank");
  }
}
