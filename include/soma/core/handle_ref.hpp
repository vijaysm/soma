// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_HANDLE_REF_HPP_
#define SOMA_CORE_HANDLE_REF_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"

namespace soma {

// What a handle names: a slot (a dimension, or set_dim for the entity sets)
// and a local id there. A default target names nothing.
struct HandleTarget {
  int slot = -1;
  LO lid = invalid_lo;
  KOKKOS_INLINE_FUNCTION bool valid() const { return slot >= 0; }
};

// Resolves the values of handle fields (gids, FieldOptions::handle) to
// entities and sets, on the host and in kernels. It is a snapshot of one
// mesh's gids; build it again after the mesh changes. 0, the null handle, and
// gids the mesh does not hold resolve to no target. build throws
// std::invalid_argument when one gid names entities of two slots, or an
// entity has gid 0, since handles could not tell them apart. box_mesh and VTK
// input number every dimension from the same start. .h5m file ids are unique
// across a file, but derived edges and faces are numbered within their
// dimension from 1 (or above its largest explicit gid), so after
// build_adjacencies derives entities the gids usually repeat and build throws.
template <class MemSpace>
class HandleRef {
 public:
  HandleRef() = default;

  static HandleRef build(const Mesh<MemSpace>& mesh) {
    std::vector<std::tuple<GO, int, LO>> all;
    const auto add = [&](int slot, const View1<const GO, MemSpace>& gids) {
      const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gids);
      for (std::size_t l = 0; l < h.extent(0); ++l) all.emplace_back(h(l), slot, static_cast<LO>(l));
    };
    for (int d = 0; d <= mesh.dim(); ++d) add(d, mesh.gid(d));
    add(set_dim, mesh.sets().gid());
    std::sort(all.begin(), all.end());
    for (std::size_t i = 0; i < all.size(); ++i) {
      const auto [gid, slot, lid] = all[i];
      if (gid == 0) {
        throw std::invalid_argument("HandleRef: " + slot_text(slot) + " has an entity of gid 0, the null handle");
      }
      if (i > 0 && std::get<0>(all[i - 1]) == gid) {
        throw std::invalid_argument("HandleRef: gid " + std::to_string(gid) + " names entities of " +
                                    slot_text(std::get<1>(all[i - 1])) + " and " + slot_text(slot) +
                                    "; handles need gids unique across slots");
      }
    }
    View1<GO, Kokkos::HostSpace> g(Kokkos::view_alloc("soma::handles::gid", Kokkos::WithoutInitializing),
                                   all.size());
    View1<std::int8_t, Kokkos::HostSpace> s(
        Kokkos::view_alloc("soma::handles::slot", Kokkos::WithoutInitializing), all.size());
    View1<LO, Kokkos::HostSpace> l(Kokkos::view_alloc("soma::handles::lid", Kokkos::WithoutInitializing),
                                   all.size());
    for (std::size_t i = 0; i < all.size(); ++i) {
      g(i) = std::get<0>(all[i]);
      s(i) = static_cast<std::int8_t>(std::get<1>(all[i]));
      l(i) = std::get<2>(all[i]);
    }
    HandleRef out;
    out.gids_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g);
    out.slots_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, s);
    out.lids_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, l);
    return out;
  }

  // Number of entities and sets it resolves.
  KOKKOS_INLINE_FUNCTION std::size_t size() const { return gids_.extent(0); }

  KOKKOS_INLINE_FUNCTION HandleTarget resolve(GO gid) const {
    std::size_t lo = 0;
    std::size_t hi = gids_.extent(0);
    while (lo < hi) {
      const std::size_t mid = lo + (hi - lo) / 2;
      if (gids_(mid) < gid) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if (gid == 0 || lo == gids_.extent(0) || gids_(lo) != gid) return HandleTarget{};
    return HandleTarget{static_cast<int>(slots_(lo)), lids_(lo)};
  }

 private:
  static std::string slot_text(int slot) {
    return slot == set_dim ? std::string("the entity sets") : "dimension " + std::to_string(slot);
  }

  View1<const GO, MemSpace> gids_;  // sorted
  View1<const std::int8_t, MemSpace> slots_;
  View1<const LO, MemSpace> lids_;
};

}  // namespace soma

#endif  // SOMA_CORE_HANDLE_REF_HPP_
