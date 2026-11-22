// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The distributed equiangular cubed sphere: counts, geometry, orientation, and
// the same mesh on any number of ranks.

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/cubed_sphere.hpp"

using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Op;
using soma::Real;

namespace {

soma::sem::CubedSphereSpec spec(int ne, Real r = 1, int order = 1) {
  soma::sem::CubedSphereSpec s;
  s.ne = ne;
  s.radius = r;
  s.geometry_order = order;
  return s;
}

TEST(SemCubedSphere, CountsAndEuler) {
  const Comm c = Comm::world();
  for (int ne : {1, 2, 5}) {
    const HostMesh m = soma::sem::cubed_sphere_mesh(c, spec(ne));
    const auto quads = static_cast<std::int64_t>(m.segment_table(2).owned_end);
    const auto verts = static_cast<std::int64_t>(m.segment_table(0).owned_end);
    EXPECT_EQ(c.allreduce(quads, Op::sum), 6 * ne * ne) << "ne " << ne;
    EXPECT_EQ(c.allreduce(verts, Op::sum), 6 * ne * ne + 2) << "ne " << ne;
  }
}

TEST(SemCubedSphere, VerticesOnSphere) {
  const Comm c = Comm::world();
  for (Real r : {1.0, 6.371e6}) {
    const HostMesh m = soma::sem::cubed_sphere_mesh(c, spec(3, r));
    const auto x = m.coords();
    for (LO v = 0; v < m.count(0); ++v) {
      const Real l = std::sqrt(x(v, 0) * x(v, 0) + x(v, 1) * x(v, 1) + x(v, 2) * x(v, 2));
      EXPECT_LE(std::abs(l - r), 1e-14 * r) << "vertex " << v;
    }
  }
}

TEST(SemCubedSphere, OutwardOrientation) {
  const Comm c = Comm::world();
  const HostMesh m = soma::sem::cubed_sphere_mesh(c, spec(4));
  const auto x = m.coords();
  const auto conn = m.conn(2);
  const auto off = m.conn_offsets(2);
  for (LO e = 0; e < m.count(2); ++e) {
    ASSERT_EQ(off(e + 1) - off(e), 4);
    const LO v0 = conn(off(e)), v1 = conn(off(e) + 1), v3 = conn(off(e) + 3);
    std::array<Real, 3> a{}, b{}, n{}, g{};
    for (int k = 0; k < 3; ++k) {
      a[k] = x(v1, k) - x(v0, k);
      b[k] = x(v3, k) - x(v0, k);
      for (int q = 0; q < 4; ++q) g[k] += x(conn(off(e) + q), k);
    }
    n = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    EXPECT_GT(n[0] * g[0] + n[1] * g[1] + n[2] * g[2], 0) << "element gid " << m.gid(2)(e);
  }
}

TEST(SemCubedSphere, SameMeshOnAnyRankCount) {
  const Comm c = Comm::world();
  const HostMesh m = soma::sem::cubed_sphere_mesh(c, spec(3));
  const HostMesh s = soma::sem::cubed_sphere_mesh(Comm::self(), spec(3));
  std::map<GO, LO> serial;
  for (LO e = 0; e < s.count(2); ++e) serial[s.gid(2)(e)] = e;
  for (LO e = 0; e < m.count(2); ++e) {
    const LO f = serial.at(m.gid(2)(e));
    for (int q = 0; q < 4; ++q) {
      const LO v = m.conn(2)(m.conn_offsets(2)(e) + q), w = s.conn(2)(s.conn_offsets(2)(f) + q);
      EXPECT_EQ(m.gid(0)(v), s.gid(0)(w)) << "element gid " << m.gid(2)(e) << " corner " << q;
      for (int k = 0; k < 3; ++k) EXPECT_EQ(m.coords()(v, k), s.coords()(w, k));
    }
  }
}

TEST(SemCubedSphere, Quad9NodesOnSphere) {
  const Comm c = Comm::world();
  const Real r = 2.5;
  const HostMesh m = soma::sem::cubed_sphere_mesh(c, spec(3, r, 2));
  if (m.count(2) == 0) return;
  ASSERT_TRUE(m.has_nodes(2));
  const auto off = m.node_offsets(2);
  const auto nodes = m.nodes(2);
  const auto x = m.coords();
  for (LO e = 0; e < m.count(2); ++e) {
    ASSERT_EQ(off(e + 1) - off(e), 5) << "element " << e;
    for (auto k = off(e); k < off(e + 1); ++k) {
      const LO v = nodes(k);
      const Real l = std::sqrt(x(v, 0) * x(v, 0) + x(v, 1) * x(v, 1) + x(v, 2) * x(v, 2));
      EXPECT_LE(std::abs(l - r), 1e-14 * r);
    }
  }
}

TEST(SemCubedSphere, EmptyRanks) {
  const Comm c = Comm::world();
  const HostMesh m = soma::sem::cubed_sphere_mesh(c, spec(1));
  if (c.rank() >= 6) EXPECT_EQ(m.count(2), 0);
  EXPECT_EQ(c.allreduce(static_cast<std::int64_t>(m.segment_table(2).owned_end), Op::sum), 6);
}

TEST(SemCubedSphere, RejectsBadSpecs) {
  const Comm c = Comm::world();
  EXPECT_THROW((void)soma::sem::cubed_sphere_mesh(c, spec(0)), std::invalid_argument);
  EXPECT_THROW((void)soma::sem::cubed_sphere_mesh(c, spec(2, 0.0)), std::invalid_argument);
  EXPECT_THROW((void)soma::sem::cubed_sphere_mesh(c, spec(2, 1.0, 3)), std::invalid_argument);
}

}  // namespace
