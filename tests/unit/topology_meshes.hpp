// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_TOPOLOGY_MESHES_HPP_
#define SOMA_TESTS_TOPOLOGY_MESHES_HPP_

// Small meshes for the topology tests, built in Ordering::file unless asked.

#include <utility>
#include <vector>

#include "adjacency_check.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"
#include "soma/testing/box_mesh.hpp"

namespace soma_test {

using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::Topology;

inline soma::HostMesh box(int nx, int ny, int nz, Topology t,
                   soma::Ordering ordering = soma::Ordering::file) {
  soma::testing::BoxSpec s;
  s.nx = nx;
  s.ny = ny;
  s.nz = nz;
  s.element = t;
  s.ordering = ordering;
  return soma::testing::box_mesh(s);
}

// One hex with a pyramid on its top face, a tet on a pyramid triangle, and a
// prism against its x = 1 face: every fixed 3D topology but the knife.
inline soma::HostMesh mixed_3d() {
  soma::MeshBuilder b(3);
  const std::vector<GO> vg{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::vector<Real> xyz{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1,
                              1, 1, 1, 0, 1, 1, 0.5, 0.5, 2, 0.5, -0.5, 2, 2, 0, 0, 2, 0, 1};
  b.add_vertices(vg, xyz);
  b.add_elements(Topology::hex, std::vector<GO>{1}, std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8});
  b.add_elements(Topology::pyramid, std::vector<GO>{2}, std::vector<GO>{5, 6, 7, 8, 9});
  b.add_elements(Topology::tet, std::vector<GO>{3}, std::vector<GO>{5, 6, 9, 10});
  // prism triangles (2, 11, 3) at the bottom and (6, 12, 7) at the top; its
  // quad (2, 3, 7, 6) is the hex face x = 1
  b.add_elements(Topology::prism, std::vector<GO>{4}, std::vector<GO>{2, 11, 3, 6, 12, 7});
  return std::move(b).build(soma::Ordering::file);
}

// A 5 x 4 quad grid with the quads of columns (0,1) and (2,3) merged into
// hexagons: arities 6, 6, 4 per row.
inline soma::HostMesh merged_polygons() {
  const int nx = 5, ny = 4;
  soma::MeshBuilder b(2);
  std::vector<GO> vg;
  std::vector<Real> xyz;
  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) {
      vg.push_back(1 + i + (nx + 1) * j);
      xyz.insert(xyz.end(), {Real(i), Real(j), 0});
    }
  }
  b.add_vertices(vg, xyz);
  const auto v = [&](int i, int j) { return GO{1 + i + (nx + 1) * j}; };
  std::vector<GO> gids;
  std::vector<Offset> offsets{0};
  std::vector<GO> conn;
  for (int j = 0; j < ny; ++j) {
    for (int i0 : {0, 2}) {
      conn.insert(conn.end(), {v(i0, j), v(i0 + 1, j), v(i0 + 2, j), v(i0 + 2, j + 1),
                               v(i0 + 1, j + 1), v(i0, j + 1)});
      gids.push_back(static_cast<GO>(gids.size() + 1));
      offsets.push_back(static_cast<Offset>(conn.size()));
    }
    conn.insert(conn.end(), {v(4, j), v(5, j), v(5, j + 1), v(4, j + 1)});
    gids.push_back(static_cast<GO>(gids.size() + 1));
    offsets.push_back(static_cast<Offset>(conn.size()));
  }
  b.add_polygons(gids, offsets, conn);
  return std::move(b).build(soma::Ordering::file);
}

// The box's hexes and vertices plus explicit faces and edges.
inline soma::HostMesh box_with_explicit(int n, const std::vector<GO>& quad_gids,
                                 const std::vector<GO>& quad_conn,
                                 const std::vector<GO>& edge_gids = {},
                                 const std::vector<GO>& edge_conn = {}) {
  const auto box3 = box(n, n, n, Topology::hex);
  soma::MeshBuilder b(3);
  const auto gid0 = to_vector(box3.gid(0));
  std::vector<Real> xyz;
  for (LO v = 0; v < box3.count(0); ++v) {
    for (int c = 0; c < 3; ++c) xyz.push_back(box3.coords()(v, c));
  }
  b.add_vertices(gid0, xyz);
  std::vector<GO> hconn;
  for (const LO v : to_vector(box3.conn(3))) hconn.push_back(gid0[static_cast<std::size_t>(v)]);
  b.add_elements(Topology::hex, to_vector(box3.gid(3)), hconn);
  if (!quad_gids.empty()) b.add_elements(Topology::quad, quad_gids, quad_conn);
  if (!edge_gids.empty()) b.add_elements(Topology::edge, edge_gids, edge_conn);
  return std::move(b).build(soma::Ordering::file);
}

}  // namespace soma_test

#endif  // SOMA_TESTS_TOPOLOGY_MESHES_HPP_
