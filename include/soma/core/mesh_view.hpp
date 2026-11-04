// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_MESH_VIEW_HPP_
#define SOMA_CORE_MESH_VIEW_HPP_

#include <cstddef>
#include <cstdint>

#include <Kokkos_Core.hpp>

#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/topology/adjacency.hpp"

namespace soma {

// Kernel handle. Copy it into lambdas; never capture a Mesh. All Views are
// unmanaged (no refcount traffic per launch). Valid only while the owning Mesh
// is alive and unmodified in topology.
template <class MemSpace>
struct MeshView {
  using Traits = Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>;
  using CoordsU = Kokkos::View<const Real* [3], Kokkos::LayoutLeft, MemSpace, Traits>;

  int dim = 0;
  Kokkos::Array<LO, 4> counts{};
  Kokkos::Array<LO, 4> interior_ends{};
  Kokkos::Array<LO, 4> owned_ends{};
  Kokkos::Array<LO, 4> shared_ends{};
  Kokkos::Array<UView1<Offset, MemSpace>, 4> offsets{};
  Kokkos::Array<UView1<LO, MemSpace>, 4> conn{};
  Kokkos::Array<UView1<std::uint8_t, MemSpace>, 4> topo{};
  Kokkos::Array<UView1<GO, MemSpace>, 4> gids{};
  // Empty for a dimension without high-order elements.
  Kokkos::Array<UView1<Offset, MemSpace>, 4> node_offsets{};
  Kokkos::Array<UView1<LO, MemSpace>, 4> nodes{};
  // Empty unless the mesh is edited and holds removed entities.
  Kokkos::Array<UView1<std::uint8_t, MemSpace>, 4> alive_mask{};
  CoordsU coords{};
  LO corners = 0;  // vertices [0, corners) are corners, the rest nodes only
  // Indexed by pair_slot and bridge_slot; empty where the mesh holds no graph.
  Kokkos::Array<UView1<Offset, MemSpace>, adjacency_slots> adj_offsets{};
  Kokkos::Array<UView1<LO, MemSpace>, adjacency_slots> adj_targets{};
  Kokkos::Array<UView1<std::uint8_t, MemSpace>, adjacency_slots> adj_code_values{};

  KOKKOS_INLINE_FUNCTION LO count(int d) const { return counts[static_cast<std::size_t>(d)]; }
  KOKKOS_INLINE_FUNCTION Topology topology(int d, LO e) const {
    return d == 0 ? Topology::vertex : static_cast<Topology>(topo[static_cast<std::size_t>(d)](e));
  }
  KOKKOS_INLINE_FUNCTION LO num_vertices(int d, LO e) const {
    const auto k = static_cast<std::size_t>(d);
    return d == 0 ? 1 : static_cast<LO>(offsets[k](e + 1) - offsets[k](e));
  }
  KOKKOS_INLINE_FUNCTION LO vertex(int d, LO e, LO k) const {
    const auto i = static_cast<std::size_t>(d);
    return d == 0 ? e : conn[i](offsets[i](e) + k);
  }
  // False for an entity MeshEdit removed and compress() has not dropped yet;
  // always true on a mesh that was not edited.
  KOKKOS_INLINE_FUNCTION bool alive(int d, LO e) const {
    const auto& m = alive_mask[static_cast<std::size_t>(d)];
    return m.extent(0) == 0 || m(e) != 0;
  }
  // The corners and then the other nodes of a high-order element, in MOAB's
  // order (soma/topology/nodes.hpp names the sub-entity each sits on).
  KOKKOS_INLINE_FUNCTION LO num_nodes(int d, LO e) const {
    const auto i = static_cast<std::size_t>(d);
    const LO n = num_vertices(d, e);
    return node_offsets[i].extent(0) == 0 ? n : n + static_cast<LO>(node_offsets[i](e + 1) - node_offsets[i](e));
  }
  KOKKOS_INLINE_FUNCTION LO node(int d, LO e, LO k) const {
    const LO n = num_vertices(d, e);
    if (k < n) return vertex(d, e, k);
    const auto i = static_cast<std::size_t>(d);
    return nodes[i](node_offsets[i](e) + (k - n));
  }
  // (d, 0) for d >= 1 is the connectivity. Any other graph must be held by the
  // mesh; under SOMA_DEBUG a missing one aborts.
  KOKKOS_INLINE_FUNCTION Row<const LO> adj(int from, int to, LO e) const {
    if (to == 0 && from > 0) {
      const auto i = static_cast<std::size_t>(from);
      return {conn[i].data() + offsets[i](e), static_cast<LO>(offsets[i](e + 1) - offsets[i](e))};
    }
    return row(pair_slot(from, to), e);
  }
  // Empty for (d, 0).
  KOKKOS_INLINE_FUNCTION Row<const std::uint8_t> adj_codes(int from, int to, LO e) const {
    if (to == 0 && from > 0) return {};
    const auto s = static_cast<std::size_t>(pair_slot(from, to));
    check_slot(s);
    const Offset b = adj_offsets[s](e);
    return {adj_code_values[s].data() + b, static_cast<LO>(adj_offsets[s](e + 1) - b)};
  }
  KOKKOS_INLINE_FUNCTION Row<const LO> bridge(int d, int via, LO e) const {
    return row(bridge_slot(d, via), e);
  }
  KOKKOS_INLINE_FUNCTION Real x(LO v, int k) const { return coords(v, k); }
  KOKKOS_INLINE_FUNCTION GO gid(int d, LO e) const { return gids[static_cast<std::size_t>(d)](e); }
  KOKKOS_INLINE_FUNCTION bool owned(int d, LO e) const {
    return e < owned_ends[static_cast<std::size_t>(d)];
  }
  KOKKOS_INLINE_FUNCTION bool interior(int d, LO e) const {
    return e < interior_ends[static_cast<std::size_t>(d)];
  }
  // Shared with another rank, whichever owns it.
  KOKKOS_INLINE_FUNCTION bool shared(int d, LO e) const {
    const auto k = static_cast<std::size_t>(d);
    return e >= interior_ends[k] && e < shared_ends[k];
  }

 private:
  KOKKOS_INLINE_FUNCTION void check_slot([[maybe_unused]] std::size_t s) const {
#if SOMA_DEBUG
    if (adj_offsets[s].extent(0) == 0) Kokkos::abort("soma::MeshView: adjacency not built");
#endif
  }
  KOKKOS_INLINE_FUNCTION Row<const LO> row(int slot, LO e) const {
    const auto s = static_cast<std::size_t>(slot);
    check_slot(s);
    const Offset b = adj_offsets[s](e);
    return {adj_targets[s].data() + b, static_cast<LO>(adj_offsets[s](e + 1) - b)};
  }
};

}  // namespace soma

#endif  // SOMA_CORE_MESH_VIEW_HPP_
