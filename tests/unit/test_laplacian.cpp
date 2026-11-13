// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "distributed_fixture.hpp"
#include "soma/algorithms/laplacian.hpp"
#include "soma/config.hpp"
#include "soma/core/transfer.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
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
  bad({-1, -1, 0, {}});
  bad({1, 0, 0, {}});
  bad({1, 3, 0, {}});
  bad({1, -1, -1, {}});
  bad({1, -1, std::nan(""), {}});
  const std::vector<std::uint8_t> short_mask(3, 0);
  bad({1, -1, 0, short_mask});
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

// A mesh in the default execution space's memory is smoothed as the host
// one: the coordinates bit for bit in any build (a sweep adds and divides,
// which no compiler contracts), in as many sweeps; the largest move bit for
// bit where that space is a host one or the build keeps arithmetic as
// written (SOMA_REPRODUCIBLE=ON), to rounding on a device otherwise (its sum
// of squares is contracted there).
TEST(Laplacian, DefaultMemorySpaceSweepsAsTheHost) {
  using Mem = Kokkos::DefaultExecutionSpace::memory_space;
  constexpr bool bitwise =
      std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace> || SOMA_REPRODUCIBLE_BUILD;
  const Comm c = Comm::world();
  for (const Real tolerance : {0.0, 1e-3}) {
    SCOPED_TRACE(testing::Message() << "tolerance " << tolerance);
    soma::HostMesh host = soma::build_local_part(c, local_builder(partition(c, 1), c.rank()));
    host.add_ghosts(c, soma::GhostOptions{1, 0, false});
    perturb(host, 0.3);
    soma::Mesh<Mem> mesh = host.mirror<Mem>(soma::TransferOptions::everything());
    soma::SmoothOptions o;
    o.iterations = 25;
    o.tolerance = tolerance;
    const soma::SmoothReport want = soma::laplacian_smooth(c, host, o);
    const soma::SmoothReport got = soma::laplacian_smooth(c, mesh, o);
    EXPECT_EQ(got.iterations, want.iterations);
    if (bitwise) {
      EXPECT_EQ(got.largest_move, want.largest_move);
    } else {
      EXPECT_NEAR(got.largest_move, want.largest_move, 4 * std::numeric_limits<Real>::epsilon() * want.largest_move);
    }
    const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.coords());
    const auto y = host.coords();
    ASSERT_EQ(x.extent(0), y.extent(0));
    LO differ = 0;
    for (LO v = 0; v < host.count(0); ++v) {
      for (int q = 0; q < 3; ++q) differ += x(v, q) == y(v, q) ? 0 : 1;
    }
    EXPECT_EQ(differ, 0);
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
