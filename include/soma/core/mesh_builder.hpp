// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_MESH_BUILDER_HPP_
#define SOMA_CORE_MESH_BUILDER_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/ordering.hpp"

namespace soma {

class MeshBuilder {
 public:
  explicit MeshBuilder(int dim);

  // Throws std::invalid_argument, naming the gid, for a NaN or infinite
  // coordinate, whatever the ordering; nothing is added then.
  void add_vertices(std::span<const GO> gids, std::span<const Real> xyz);
  void add_elements(Topology t, std::span<const GO> gids, std::span<const GO> vertex_gids);
  // High-order elements: nodes_per_element node gids an element, its corners
  // and then the other nodes in MOAB's order (soma/topology/nodes.hpp). The
  // count must be a configuration of t; num_vertices(t) adds linear elements.
  void add_elements(Topology t, std::span<const GO> gids, std::span<const GO> node_gids,
                    int nodes_per_element);
  void add_polygons(std::span<const GO> gids, std::span<const Offset> offsets,
                    std::span<const GO> vertex_gids);

  // The vertices added so far: gids and coordinates (x, y, z interleaved).
  [[nodiscard]] std::span<const GO> vertex_gids() const noexcept { return vertex_gids_; }
  [[nodiscard]] std::span<const Real> vertex_coordinates() const noexcept { return xyz_; }

  // Distributed meshes: the ownership of the entities of dimension `dim`
  // named by gid (unnamed ones are interior) and a neighbour-set group, equal
  // for entities shared with the same ranks. Within an ownership kind,
  // topology, arity and node count, entities are ordered by group. Throws
  // std::invalid_argument for a dimension outside [0, dim], spans of
  // different lengths, or (at build) a gid named twice in a dimension.
  void set_ownership(int dim, std::span<const GO> gids, std::span<const Ownership> kinds,
                     std::span<const std::uint32_t> groups);
  // The box of the curve keys, instead of that of the vertices here: the box
  // of every rank's vertices makes keys comparable across ranks. Throws
  // std::invalid_argument unless the corners are finite and lo <= hi.
  void set_key_box(const std::array<Real, 3>& lo, const std::array<Real, 3>& hi);

  // Orders every dimension and translates connectivity to local ids.
  //   Ordering::file            elements (topology, arity, nodes, gid);
  //                             vertices by gid
  //   Ordering::type_major_sfc  elements (topology, arity, nodes, hilbert key
  //                             of the corners' centroid, gid); vertices
  //                             (hilbert key, gid)
  // The key uses one box, that of the vertices, for every dimension. Vertices
  // that are nodes of high-order elements and no element's corner come after
  // the others, in the same order among themselves. Element
  // gids are identities within a dimension: two elements of one dimension may
  // not share a gid, whatever their topology. Ordering::tiled throws.
  // Ownership comes first in every key, then topology, arity and nodes, then
  // the group set by set_ownership, then the order above, so each ownership
  // kind is a contiguous range (soma/core/segments.hpp). Ownership other than
  // interior on vertices of a mesh with node-only vertices throws
  // std::invalid_argument: corners first and ownership first cannot both hold.
  HostMesh build(Ordering ordering = default_ordering) &&;

 private:
  struct Block {
    Topology topology;
    std::vector<GO> gids;
    std::vector<Offset> offsets;
    std::vector<GO> vertex_gids;
    std::uint16_t nodes = 0;      // an element's nodes beyond its corners
    std::vector<GO> node_gids;    // nodes per element, after the corners
  };
  int dim_;
  std::vector<GO> vertex_gids_;
  std::vector<Real> xyz_;
  std::array<std::vector<Block>, 4> blocks_;
  struct Owned {
    GO gid;
    Ownership kind;
    std::uint32_t group;
  };
  std::array<std::vector<Owned>, 4> owned_;
  std::optional<std::array<Real, 6>> key_box_;
};

}  // namespace soma

#endif  // SOMA_CORE_MESH_BUILDER_HPP_
