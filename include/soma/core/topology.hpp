// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_TOPOLOGY_HPP_
#define SOMA_CORE_TOPOLOGY_HPP_

#include <cstdint>

#include <Kokkos_Core.hpp>

namespace soma {

// Values equal the MOAB .h5m "/tstt/elemtypes" enumeration and moab::EntityType,
// so the reader maps element groups without a translation table. Never renumber.
enum class Topology : std::uint8_t {
  vertex = 0,
  edge = 1,
  tri = 2,
  quad = 3,
  polygon = 4,
  tet = 5,
  pyramid = 6,
  prism = 7,
  knife = 8,
  hex = 9,
  polyhedron = 10,
};

inline constexpr int topology_count = 11;

KOKKOS_INLINE_FUNCTION constexpr int dimension(Topology t) noexcept {
  switch (t) {
    case Topology::vertex: return 0;
    case Topology::edge: return 1;
    case Topology::tri:
    case Topology::quad:
    case Topology::polygon: return 2;
    case Topology::tet:
    case Topology::pyramid:
    case Topology::prism:
    case Topology::knife:
    case Topology::hex:
    case Topology::polyhedron: return 3;
  }
  return -1;
}

// Corner vertex count; 0 for variable-arity topologies.
KOKKOS_INLINE_FUNCTION constexpr int num_vertices(Topology t) noexcept {
  switch (t) {
    case Topology::vertex: return 1;
    case Topology::edge: return 2;
    case Topology::tri: return 3;
    case Topology::quad: return 4;
    case Topology::tet: return 4;
    case Topology::pyramid: return 5;
    case Topology::prism: return 6;
    case Topology::knife: return 7;
    case Topology::hex: return 8;
    case Topology::polygon:
    case Topology::polyhedron: return 0;
  }
  return 0;
}

KOKKOS_INLINE_FUNCTION constexpr bool has_fixed_topology(Topology t) noexcept {
  return t != Topology::polygon && t != Topology::polyhedron;
}

constexpr const char* name(Topology t) noexcept {
  switch (t) {
    case Topology::vertex: return "vertex";
    case Topology::edge: return "edge";
    case Topology::tri: return "tri";
    case Topology::quad: return "quad";
    case Topology::polygon: return "polygon";
    case Topology::tet: return "tet";
    case Topology::pyramid: return "pyramid";
    case Topology::prism: return "prism";
    case Topology::knife: return "knife";
    case Topology::hex: return "hex";
    case Topology::polyhedron: return "polyhedron";
  }
  return "unknown";
}

}  // namespace soma

#endif  // SOMA_CORE_TOPOLOGY_HPP_
