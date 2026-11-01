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
  Kokkos::Array<UView1<Offset, MemSpace>, 4> offsets{};
  Kokkos::Array<UView1<LO, MemSpace>, 4> conn{};
  Kokkos::Array<UView1<std::uint8_t, MemSpace>, 4> topo{};
  Kokkos::Array<UView1<GO, MemSpace>, 4> gids{};
  CoordsU coords{};

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
  KOKKOS_INLINE_FUNCTION Real x(LO v, int k) const { return coords(v, k); }
  KOKKOS_INLINE_FUNCTION GO gid(int d, LO e) const { return gids[static_cast<std::size_t>(d)](e); }
  KOKKOS_INLINE_FUNCTION bool owned(int d, LO e) const {
    return e < owned_ends[static_cast<std::size_t>(d)];
  }
  KOKKOS_INLINE_FUNCTION bool interior(int d, LO e) const {
    return e < interior_ends[static_cast<std::size_t>(d)];
  }
};

}  // namespace soma

#endif  // SOMA_CORE_MESH_VIEW_HPP_
