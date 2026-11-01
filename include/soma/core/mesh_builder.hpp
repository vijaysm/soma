// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_MESH_BUILDER_HPP_
#define SOMA_CORE_MESH_BUILDER_HPP_

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "soma/core/mesh.hpp"

namespace soma {

enum class Ordering : std::uint8_t { file, type_major_sfc, tiled };

// Ordering used when the caller does not choose one.
inline constexpr Ordering default_ordering = Ordering::type_major_sfc;

class MeshBuilder {
 public:
  explicit MeshBuilder(int dim);

  void add_vertices(std::span<const GO> gids, std::span<const Real> xyz);
  void add_elements(Topology t, std::span<const GO> gids, std::span<const GO> vertex_gids);
  void add_polygons(std::span<const GO> gids, std::span<const Offset> offsets,
                    std::span<const GO> vertex_gids);

  // Orders every dimension and translates connectivity to local ids.
  //   Ordering::file            elements (topology, arity, gid); vertices by gid
  //   Ordering::type_major_sfc  elements (topology, arity, hilbert key of the
  //                             centroid, gid); vertices (hilbert key, gid)
  // The key uses one box, that of the vertices, for every dimension. Element
  // gids are identities within a dimension: two elements of one dimension may
  // not share a gid, whatever their topology. Ordering::tiled throws.
  HostMesh build(Ordering ordering = default_ordering) &&;

 private:
  struct Block {
    Topology topology;
    std::vector<GO> gids;
    std::vector<Offset> offsets;
    std::vector<GO> vertex_gids;
  };
  int dim_;
  std::vector<GO> vertex_gids_;
  std::vector<Real> xyz_;
  std::array<std::vector<Block>, 4> blocks_;
};

}  // namespace soma

#endif  // SOMA_CORE_MESH_BUILDER_HPP_
