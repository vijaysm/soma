// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TOPOLOGY_NODES_HPP_
#define SOMA_TOPOLOGY_NODES_HPP_

// MOAB's high-order nodes. An element of `count` nodes lists its corners,
// then one node on each edge in canonical edge order, one on each face in
// canonical face order and one on the region, as each configuration has them
// (CN::HasMidNodes, CN::HONodeParent). For an edge the mid-edge node, and for
// a 2D element the mid-face node, is its centre.

#include "soma/core/topology.hpp"
#include "soma/topology/canonical.hpp"

namespace soma {

// Configuration bits, as CN::HasMidNodes: mid-edge, mid-face, mid-region.
inline constexpr int mid_edge_bit = 1 << 1;
inline constexpr int mid_face_bit = 1 << 2;
inline constexpr int mid_region_bit = 1 << 3;

// The node count of a configuration of t: its corners and the nodes the bits
// add. Bits a type has no sub-entities for (a region on a 2D element) add none.
KOKKOS_INLINE_FUNCTION constexpr int num_nodes(Topology t, int bits) noexcept {
  return num_vertices(t) + ((bits & mid_edge_bit) != 0 ? canonical::num_sides(t, 1) : 0) +
         ((bits & mid_face_bit) != 0 ? canonical::num_sides(t, 2) : 0) +
         ((bits & mid_region_bit) != 0 && dimension(t) == 3 ? 1 : 0);
}

// The bits of the configuration of t with `count` nodes: 0 for the corners
// alone, -1 when no configuration has that count. Edges, triangles, quads,
// tets, pyramids, prisms and hexes have mid-nodes; other types only corners.
KOKKOS_INLINE_FUNCTION constexpr int mid_node_bits(Topology t, int count) noexcept {
  const int corners = num_vertices(t);
  if (corners == 0 || t == Topology::vertex || t == Topology::knife) return count == corners && corners > 0 ? 0 : -1;
  // Bits a type has no sub-entities for add no nodes, so the smaller bits
  // without them match first.
  for (int bits = 0; bits <= (mid_edge_bit | mid_face_bit | mid_region_bit); bits += 2) {
    if (num_nodes(t, bits) == count) return bits;
  }
  return -1;
}

// The sub-entity node k of a `count`-node element of t sits on: dimension 0
// and the corner for a corner, else the dimension and canonical index of the
// edge, face or region; {-1, -1} for a k or count out of range.
struct NodeParent {
  int dim = -1;
  int index = -1;
};

KOKKOS_INLINE_FUNCTION constexpr NodeParent node_parent(Topology t, int count, int k) noexcept {
  const int bits = mid_node_bits(t, count);
  if (bits < 0 || k < 0) return {};  // past the last group for k >= count
  int first = num_vertices(t);
  if (k < first) return {0, k};
  for (int d = 1; d <= 3; ++d) {
    const int bit = d == 1 ? mid_edge_bit : d == 2 ? mid_face_bit : mid_region_bit;
    if ((bits & bit) == 0) continue;
    const int n = d == 3 ? 1 : canonical::num_sides(t, d);
    if (k < first + n) return {d, k - first};
    first += n;
  }
  return {};
}

}  // namespace soma

#endif  // SOMA_TOPOLOGY_NODES_HPP_
