// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Meshes for the spectral-element tests, ready for a SemSpace on any number of
// ranks: a ghost layer through vertices and distributed edges on more than one.

#ifndef SOMA_TESTS_UNIT_SEM_MESHES_HPP_
#define SOMA_TESTS_UNIT_SEM_MESHES_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/topology.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/sem/cubed_sphere.hpp"
#include "soma/testing/box_mesh.hpp"

namespace sem_test {

inline soma::HostMesh prepare(const soma::Comm& c, soma::HostMesh mesh) {
  if (c.size() > 1) {
    mesh.add_ghosts(c, soma::GhostOptions{1, 0});
    soma::build_distributed_adjacencies(c, mesh, soma::AdjacencyRequest{}.add(2, 1));
  }
  return mesh;
}

inline soma::HostMesh box(const soma::Comm& c, int nx, int ny, soma::Real lx, soma::Real ly,
                          soma::Topology t = soma::Topology::quad) {
  soma::testing::BoxSpec s;
  s.nx = nx;
  s.ny = ny;
  s.element = t;
  s.lx = lx;
  s.ly = ly;
  return prepare(c, soma::testing::box_mesh(c, s));
}

inline soma::HostMesh sphere(const soma::Comm& c, int ne, soma::Real r = 1, int geometry_order = 1) {
  soma::sem::CubedSphereSpec s;
  s.ne = ne;
  s.radius = r;
  s.geometry_order = geometry_order;
  return prepare(c, soma::sem::cubed_sphere_mesh(c, s));
}

// A mesh of elements given whole (gids, nodes per element, node gids, node
// coordinates by gid), each rank building a consecutive block of them.
inline soma::HostMesh distribute(const soma::Comm& c, const std::vector<soma::GO>& elements, int nodes_per_element,
                                 const std::vector<soma::GO>& nodes,
                                 const std::map<soma::GO, std::array<soma::Real, 3>>& xyz) {
  const auto n = static_cast<std::int64_t>(elements.size());
  std::vector<soma::GO> mine, conn, used;
  for (std::int64_t e = 0; e < n; ++e) {
    if (e * c.size() / n != c.rank()) continue;
    mine.push_back(elements[static_cast<std::size_t>(e)]);
    for (int k = 0; k < nodes_per_element; ++k) {
      conn.push_back(nodes[static_cast<std::size_t>(e * nodes_per_element + k)]);
    }
  }
  used = conn;
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  std::vector<soma::Real> coords;
  for (const soma::GO g : used) {
    for (const soma::Real x : xyz.at(g)) coords.push_back(x);
  }
  soma::MeshBuilder b(2);
  b.add_vertices(used, coords);
  if (nodes_per_element == 4) {
    b.add_elements(soma::Topology::quad, mine, conn);
  } else {
    b.add_elements(soma::Topology::quad, mine, conn, nodes_per_element);
  }
  return prepare(c, soma::build_local_part(c, std::move(b), soma::default_ordering));
}

}  // namespace sem_test

#endif  // SOMA_TESTS_UNIT_SEM_MESHES_HPP_
