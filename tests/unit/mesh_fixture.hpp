// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_MESH_FIXTURE_HPP_
#define SOMA_TESTS_MESH_FIXTURE_HPP_

#include <cstdint>
#include <utility>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"

namespace soma {

// Test-only access to a mesh's segment tables. The builder produces interior
// entities only, so this is how a test gets shared and ghost entities.
struct MeshTestAccess {
  template <class M>
  static void set_segments(Mesh<M>& mesh, int dim, SegmentTable table) {
    mesh.ents_[static_cast<std::size_t>(dim)].segments = std::move(table);
  }
  // Installs a graph directly, as if build_adjacencies had made it.
  template <class M>
  static void put_adjacency(Mesh<M>& mesh, int slot, Adjacency<M> graph) {
    mesh.adj_.put(slot, std::move(graph));
  }
  // Overwrites one gid of a host mesh, past the builder's checks.
  static void set_gid(Mesh<Kokkos::HostSpace>& mesh, int dim, LO lid, GO gid) {
    mesh.ents_[static_cast<std::size_t>(dim)].gid(static_cast<std::size_t>(lid)) = gid;
  }
};

}  // namespace soma

namespace soma_test {

// Two quads and one triangle in file-id space with unsorted gids.
//  v gids:  10(0,0) 20(1,0) 30(2,0) 40(0,1) 50(1,1) 60(2,1) 70(3,0.5)
//  quads:   gid 7 -> [10 20 50 40],  gid 3 -> [20 30 60 50]
//  tri:     gid 5 -> [30 70 60]
inline soma::HostMesh two_quads_and_a_triangle(soma::Ordering ordering = soma::Ordering::file) {
  using soma::GO;
  using soma::Real;
  soma::MeshBuilder b(2);
  const std::vector<GO> vg{50, 10, 70, 30, 20, 60, 40};
  const std::vector<Real> xyz{1, 1, 0, 0, 0, 0, 3, 0.5, 0, 2, 0, 0, 1, 0, 0, 2, 1, 0, 0, 1, 0};
  b.add_vertices(vg, xyz);
  b.add_elements(soma::Topology::quad, std::vector<GO>{7, 3},
                 std::vector<GO>{10, 20, 50, 40, 20, 30, 60, 50});
  b.add_elements(soma::Topology::tri, std::vector<GO>{5}, std::vector<GO>{30, 70, 60});
  return std::move(b).build(ordering);
}

// The fixture mesh with its three elements relabelled interior, shared, ghost
// (the vertices stay interior): interior_end = 1, owned_end = 2.
inline soma::HostMesh mesh_with_shared_and_ghost_elements() {
  soma::HostMesh mesh = two_quads_and_a_triangle();
  const std::vector<soma::Ownership> own{soma::Ownership::interior, soma::Ownership::shared,
                                         soma::Ownership::ghost};
  const std::vector<soma::Topology> topo{soma::Topology::tri, soma::Topology::quad,
                                         soma::Topology::quad};
  const std::vector<std::uint16_t> arity{3, 4, 4};
  soma::MeshTestAccess::set_segments(mesh, 2, soma::make_segments(own, topo, arity));
  return mesh;
}

// The fixture mesh with its seven vertices relabelled: three interior, two
// shared and owned here, one shared copy, one ghost.
inline soma::HostMesh mesh_with_every_vertex_ownership() {
  soma::HostMesh mesh = two_quads_and_a_triangle();
  using soma::Ownership;
  const std::vector<Ownership> own{Ownership::interior, Ownership::interior, Ownership::interior, Ownership::shared,
                                   Ownership::shared,   Ownership::shared_copy, Ownership::ghost};
  const std::vector<soma::Topology> topo(7, soma::Topology::vertex);
  const std::vector<std::uint16_t> arity(7, 1);
  soma::MeshTestAccess::set_segments(mesh, 0, soma::make_segments(own, topo, arity));
  return mesh;
}

}  // namespace soma_test

#endif  // SOMA_TESTS_MESH_FIXTURE_HPP_
