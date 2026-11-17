// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_HANDLE_REF_HPP_
#define SOMA_CORE_HANDLE_REF_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"

namespace soma {

/// A handle names an entity or set by its slot (a dimension, or set_dim for the
/// entity sets) and its gid, as a MOAB handle names one by type and id: gids
/// are unique within a slot only, so vertex 1 and edge 1 are two handles. The
/// slot + 1 is kept in the bits above handle_gid_bits and the gid below them,
/// so 0 is the null handle and any gid in [0, max_handle_gid] has one.
inline constexpr int handle_gid_bits = 60;
inline constexpr GO max_handle_gid = (GO{1} << handle_gid_bits) - 1;

/// The handle of (slot, gid), or 0 when no handle can hold them: a slot outside
/// [0, set_dim] or a gid outside [0, max_handle_gid].
KOKKOS_INLINE_FUNCTION constexpr GO make_handle(int slot, GO gid) {
  if (slot < 0 || slot > set_dim || gid < 0 || gid > max_handle_gid) return 0;
  return (static_cast<GO>(slot + 1) << handle_gid_bits) | gid;
}

/// The slot a handle names, or -1 for the null handle and values no
/// make_handle gives.
KOKKOS_INLINE_FUNCTION constexpr int handle_slot(GO h) {
  if (h <= 0) return -1;
  const auto s = static_cast<int>(h >> handle_gid_bits) - 1;
  return s <= set_dim ? s : -1;
}

/// The gid a handle names; meaningful when handle_slot(h) >= 0.
KOKKOS_INLINE_FUNCTION constexpr GO handle_gid(GO h) { return h & max_handle_gid; }

/// What a handle names: a slot and a local id there. A default target names
/// nothing.
struct HandleTarget {
  int slot = -1;
  LO lid = invalid_lo;
  KOKKOS_INLINE_FUNCTION bool valid() const { return slot >= 0; }
};

/// Resolves the values of handle fields (FieldOptions::handle) to entities and
/// sets, on the host and in kernels. It is a snapshot of one mesh's gids;
/// build it again after the mesh changes. The null handle and handles the mesh
/// does not hold resolve to no target. build throws std::invalid_argument when
/// a gid repeats within a slot or has no handle (negative, or beyond
/// max_handle_gid).
template <class MemSpace>
class HandleRef {
 public:
  HandleRef() = default;

  static HandleRef build(const Mesh<MemSpace>& mesh) {
    std::vector<std::pair<GO, LO>> all;
    Kokkos::Array<std::size_t, num_field_slots + 1> start{};
    const auto add = [&](int slot, const View1<const GO, MemSpace>& gids) {
      const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gids);
      const std::size_t first = all.size();
      for (std::size_t l = 0; l < h.extent(0); ++l) {
        if (make_handle(slot, h(l)) == 0) {
          throw std::invalid_argument("HandleRef: " + slot_text(slot) + " has the gid " + std::to_string(h(l)) +
                                      ", which no handle can hold");
        }
        all.emplace_back(h(l), static_cast<LO>(l));
      }
      const auto b = all.begin() + static_cast<std::ptrdiff_t>(first);
      if (!std::is_sorted(b, all.end())) std::sort(b, all.end());
      const auto twice = std::adjacent_find(b, all.end(), [](const auto& x, const auto& y) {
        return x.first == y.first;
      });
      if (twice != all.end()) {
        throw std::invalid_argument("HandleRef: " + slot_text(slot) + " has two entities of gid " +
                                    std::to_string(twice->first));
      }
    };
    for (int s = 0; s < num_field_slots; ++s) {
      start[static_cast<std::size_t>(s)] = all.size();
      if (s <= mesh.dim()) add(s, mesh.gid(s));
      if (s == set_dim) add(s, mesh.sets().gid());
    }
    start[num_field_slots] = all.size();

    View1<GO, Kokkos::HostSpace> g(Kokkos::view_alloc("soma::handles::gid", Kokkos::WithoutInitializing),
                                   all.size());
    View1<LO, Kokkos::HostSpace> l(Kokkos::view_alloc("soma::handles::lid", Kokkos::WithoutInitializing),
                                   all.size());
    for (std::size_t i = 0; i < all.size(); ++i) {
      g(i) = all[i].first;
      l(i) = all[i].second;
    }
    HandleRef out;
    out.gids_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, g);
    out.lids_ = Kokkos::create_mirror_view_and_copy(MemSpace{}, l);
    out.start_ = start;
    return out;
  }

  /// Number of entities and sets it resolves.
  KOKKOS_INLINE_FUNCTION std::size_t size() const { return gids_.extent(0); }

  KOKKOS_INLINE_FUNCTION HandleTarget resolve(GO handle) const {
    const int slot = handle_slot(handle);
    if (slot < 0) return HandleTarget{};
    const GO gid = handle_gid(handle);
    std::size_t lo = start_[static_cast<std::size_t>(slot)];
    const std::size_t end = start_[static_cast<std::size_t>(slot) + 1];
    std::size_t hi = end;
    while (lo < hi) {
      const std::size_t mid = lo + (hi - lo) / 2;
      if (gids_(mid) < gid) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if (lo == end || gids_(lo) != gid) return HandleTarget{};
    return HandleTarget{slot, lids_(lo)};
  }

 private:
  static std::string slot_text(int slot) {
    return slot == set_dim ? std::string("the entity sets") : "dimension " + std::to_string(slot);
  }

  View1<const GO, MemSpace> gids_;  ///< sorted within each slot's [start_[s], start_[s + 1])
  View1<const LO, MemSpace> lids_;
  Kokkos::Array<std::size_t, num_field_slots + 1> start_{};
};

}  // namespace soma

#endif  // SOMA_CORE_HANDLE_REF_HPP_
