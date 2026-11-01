// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_BOX_MESH_CHECK_HPP_
#define SOMA_TESTS_BOX_MESH_CHECK_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "soma/core/mesh.hpp"
#include "soma/testing/box_mesh.hpp"

namespace soma_test {

// The coordinate rule: lx*i/nx in that order, and exactly lx at the far face.
inline double box_coordinate(double length, std::int64_t i, std::int64_t n) {
  return i == n ? length : length * static_cast<double>(i) / static_cast<double>(n);
}

struct BoxShape {
  std::int64_t px = 0, py = 0, pz = 0;  // vertices per axis
  std::int64_t cells = 0;               // quads or hexes
  std::int64_t vertices = 0;
  std::int64_t elements = 0;
  int arity = 0;
  int dim = 0;
};

inline BoxShape shape_of(const soma::testing::BoxSpec& s) {
  BoxShape b;
  const bool solid = s.element == soma::Topology::hex;
  b.px = s.nx + 1;
  b.py = s.ny + 1;
  b.pz = solid ? s.nz + 1 : 1;
  b.cells = static_cast<std::int64_t>(s.nx) * s.ny * (solid ? s.nz : 1);
  b.vertices = b.px * b.py * b.pz;
  b.elements = s.element == soma::Topology::tri ? 2 * b.cells : b.cells;
  b.arity = soma::num_vertices(s.element);
  b.dim = solid ? 3 : 2;
  return b;
}

// Vertex gids of the element with this gid, in the specified order.
inline std::vector<soma::GO> expected_vertices(const soma::testing::BoxSpec& s, soma::GO gid) {
  const BoxShape b = shape_of(s);
  const bool tri = s.element == soma::Topology::tri;
  const std::int64_t cell = tri ? (gid - 1) / 2 : gid - 1;
  const std::int64_t i = cell % s.nx;
  const std::int64_t j = (cell / s.nx) % s.ny;
  const std::int64_t k = cell / (static_cast<std::int64_t>(s.nx) * s.ny);
  const auto v = [&](std::int64_t a, std::int64_t c, std::int64_t d) -> soma::GO {
    return 1 + a + b.px * (c + b.py * d);
  };
  if (s.element == soma::Topology::hex) {
    return {v(i, j, k),         v(i + 1, j, k),         v(i + 1, j + 1, k),     v(i, j + 1, k),
            v(i, j, k + 1),     v(i + 1, j, k + 1),     v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)};
  }
  if (!tri) return {v(i, j, 0), v(i + 1, j, 0), v(i + 1, j + 1, 0), v(i, j + 1, 0)};
  if ((gid - 1) % 2 == 0) return {v(i, j, 0), v(i + 1, j, 0), v(i + 1, j + 1, 0)};
  return {v(i, j, 0), v(i + 1, j + 1, 0), v(i, j + 1, 0)};
}

// Everything that does not depend on the local numbering: counts, gids,
// coordinates per gid, and the connectivity of each element read back as gids.
inline void expect_box_invariants(const soma::HostMesh& mesh, const soma::testing::BoxSpec& s) {
  const BoxShape b = shape_of(s);
  ASSERT_EQ(mesh.dim(), b.dim);
  ASSERT_EQ(mesh.count(0), b.vertices);
  ASSERT_EQ(mesh.count(b.dim), b.elements);
  for (int d = 1; d <= 3; ++d) {
    if (d != b.dim) EXPECT_EQ(mesh.count(d), 0) << "dimension " << d;
  }

  std::vector<char> gid_seen(static_cast<std::size_t>(b.vertices) + 1, 0);
  for (soma::LO v = 0; v < mesh.count(0); ++v) {
    const soma::GO g = mesh.gid(0)(v);
    ASSERT_GE(g, 1);
    ASSERT_LE(g, b.vertices);
    ASSERT_EQ(gid_seen[static_cast<std::size_t>(g)], 0) << "vertex gid " << g << " repeated";
    gid_seen[static_cast<std::size_t>(g)] = 1;
    const std::int64_t i = (g - 1) % b.px;
    const std::int64_t j = ((g - 1) / b.px) % b.py;
    const std::int64_t k = (g - 1) / (b.px * b.py);
    EXPECT_EQ(mesh.coords()(v, 0), box_coordinate(s.lx, i, s.nx)) << "vertex gid " << g;
    EXPECT_EQ(mesh.coords()(v, 1), box_coordinate(s.ly, j, s.ny)) << "vertex gid " << g;
    EXPECT_EQ(mesh.coords()(v, 2), b.dim == 3 ? box_coordinate(s.lz, k, s.nz) : 0.0)
        << "vertex gid " << g;
  }

  std::vector<char> used(static_cast<std::size_t>(b.vertices), 0);
  std::vector<char> element_seen(static_cast<std::size_t>(b.elements) + 1, 0);
  const auto offsets = mesh.conn_offsets(b.dim);
  const auto conn = mesh.conn(b.dim);
  for (soma::LO e = 0; e < mesh.count(b.dim); ++e) {
    const soma::GO g = mesh.gid(b.dim)(e);
    ASSERT_GE(g, 1);
    ASSERT_LE(g, b.elements);
    ASSERT_EQ(element_seen[static_cast<std::size_t>(g)], 0) << "element gid " << g << " repeated";
    element_seen[static_cast<std::size_t>(g)] = 1;
    EXPECT_EQ(mesh.topology(b.dim)(e), static_cast<std::uint8_t>(s.element));
    ASSERT_EQ(offsets(e + 1) - offsets(e), b.arity);
    const std::vector<soma::GO> want = expected_vertices(s, g);
    for (int k = 0; k < b.arity; ++k) {
      const soma::LO lid = conn(offsets(e) + k);
      ASSERT_GE(lid, 0);
      ASSERT_LT(lid, mesh.count(0));
      used[static_cast<std::size_t>(lid)] = 1;
      EXPECT_EQ(mesh.gid(0)(lid), want[static_cast<std::size_t>(k)])
          << "element gid " << g << " corner " << k;
    }
  }
  for (std::size_t v = 0; v < used.size(); ++v) EXPECT_EQ(used[v], 1) << "vertex lid " << v;

  ASSERT_EQ(mesh.segments(b.dim).size(), 1u);
  const soma::Segment& seg = mesh.segments(b.dim)[0];
  EXPECT_EQ(seg.begin, 0);
  EXPECT_EQ(seg.end, b.elements);
  EXPECT_EQ(seg.topology, s.element);
  EXPECT_EQ(seg.arity, b.arity);
  EXPECT_EQ(seg.ownership, soma::Ownership::interior);
  ASSERT_EQ(mesh.segments(0).size(), 1u);
  EXPECT_EQ(mesh.segments(0)[0].end, b.vertices);
}

}  // namespace soma_test

#endif  // SOMA_TESTS_BOX_MESH_CHECK_HPP_
