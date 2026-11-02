// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_SETS_HPP_
#define SOMA_CORE_SETS_HPP_

// Entity sets: groups of entities and of other sets, as MOAB's meshsets.
// Membership is one CSR per field slot (dimensions 0 to 3, then the sets),
// each row sorted and unique, and is readable on the device through SetsView.
// Flags, the file order of ordered sets and the parent/child links stay on the
// host. Sets are made by SetBuilder (soma/core/set_builder.hpp).

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/topology/adjacency.hpp"

namespace soma {

// MOAB's meshset flags as a set keeps them; the file's ranged-storage bit
// (0x8) is not kept.
inline constexpr std::uint8_t set_track_owner = 0x1;
inline constexpr std::uint8_t set_unique = 0x2;
inline constexpr std::uint8_t set_ordered = 0x4;
inline constexpr std::uint8_t set_flag_bits = 0x7;

// A member: an entity (slot 0 to 3 and its local id) or a set (set_dim and its
// index).
struct SetEntry {
  int slot = 0;
  LO lid = invalid_lo;
  friend bool operator==(const SetEntry&, const SetEntry&) = default;
  friend auto operator<=>(const SetEntry&, const SetEntry&) = default;
};

// Members of a dimension a mirror did not take were asked for.
class MissingSetMembers : public std::logic_error {
 public:
  using std::logic_error::logic_error;
};

// Kernel handle; valid while the EntitySets it came from lives. Rows exist for
// the slots the sets hold (EntitySets::has_members).
template <class MemSpace>
struct SetsView {
  LO count = 0;
  UView1<GO, MemSpace> gid{};
  Kokkos::Array<UView1<Offset, MemSpace>, num_field_slots> offsets{};
  Kokkos::Array<UView1<LO, MemSpace>, num_field_slots> members{};

  KOKKOS_INLINE_FUNCTION Row<const LO> row(int slot, LO set) const {
    const auto k = static_cast<std::size_t>(slot);
    const Offset b = offsets[k](set);
    return {members[k].data() + b, static_cast<LO>(offsets[k](set + 1) - b)};
  }
  KOKKOS_INLINE_FUNCTION bool contains(int slot, LO set, LO lid) const {
    const Row<const LO> r = row(slot, set);
    LO lo = 0;
    LO hi = r.size;
    while (lo < hi) {
      const LO mid = lo + (hi - lo) / 2;
      if (r[mid] < lid) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo < r.size && r[lo] == lid;
  }
};

class SetBuilder;
template <class MemSpace>
class Mesh;

template <class MemSpace>
class EntitySets {
 public:
  using memory_space = MemSpace;

  EntitySets() = default;
  EntitySets(const EntitySets&) = delete;
  EntitySets& operator=(const EntitySets&) = delete;
  EntitySets(EntitySets&& o) noexcept { swap(o); }
  EntitySets& operator=(EntitySets&& o) noexcept {
    EntitySets tmp(std::move(o));
    swap(tmp);
    return *this;
  }

  void swap(EntitySets& o) noexcept {
    std::swap(count_, o.count_);
    std::swap(gid_, o.gid_);
    std::swap(host_gid_, o.host_gid_);
    std::swap(slots_, o.slots_);
    flags_.swap(o.flags_);
    seq_offsets_.swap(o.seq_offsets_);
    seq_.swap(o.seq_);
    child_offsets_.swap(o.child_offsets_);
    children_.swap(o.children_);
    parent_offsets_.swap(o.parent_offsets_);
    parents_.swap(o.parents_);
  }

  LO count() const noexcept { return count_; }
  // Strictly increasing; the file ids of sets read from a file.
  View1<const GO, MemSpace> gid() const noexcept { return gid_; }

  // False for a dimension that a mirror did not take; the sets slot is always
  // held.
  bool has_members(int slot) const {
    check_slot(slot);
    return slots_[static_cast<std::size_t>(slot)].present;
  }
  // CSR of one slot: count + 1 offsets into members. Throws MissingSetMembers
  // for a slot the sets do not hold.
  View1<const Offset, MemSpace> offsets(int slot) const { return held(slot).offsets; }
  View1<const LO, MemSpace> members(int slot) const { return held(slot).members; }

  // A view of temporary sets would dangle, so that overload is deleted.
  SetsView<MemSpace> view() const& {
    SetsView<MemSpace> v;
    v.count = count_;
    v.gid = gid_;
    for (std::size_t k = 0; k < slots_.size(); ++k) {
      v.offsets[k] = slots_[k].offsets;
      v.members[k] = slots_[k].members;
    }
    return v;
  }
  SetsView<MemSpace> view() const&& = delete;

  // Host queries. A set is named by its index, 0 to count() - 1.
  std::uint8_t flags(LO set) const { return flags_[index(set)]; }
  bool ordered(LO set) const { return (flags(set) & set_ordered) != 0; }

  // Sorted, unique local ids (set indices for set_dim) of one slot.
  std::span<const LO> members_of(LO set, int slot) const {
    const Slot& s = held(slot);
    const std::size_t i = index(set);
    const auto b = static_cast<std::size_t>(s.host_offsets(i));
    const auto e = static_cast<std::size_t>(s.host_offsets(i + 1));
    return {s.host_members.data() + b, e - b};
  }

  // Contents of an ordered set in their stored order, repeats included. Throws
  // std::logic_error for an unordered set, and MissingSetMembers when an entry
  // is on a dimension the sets do not hold.
  std::span<const SetEntry> sequence(LO set) const {
    const std::size_t i = index(set);
    if (!ordered(set)) {
      throw std::logic_error("EntitySets: set " + std::to_string(set) +
                             " is not ordered and keeps no sequence");
    }
    const auto b = static_cast<std::size_t>(seq_offsets_[i]);
    const auto e = static_cast<std::size_t>(seq_offsets_[i + 1]);
    const std::span<const SetEntry> out(seq_.data() + b, e - b);
    for (const SetEntry& x : out) (void)held(x.slot);
    return out;
  }

  std::span<const LO> children(LO set) const { return links(child_offsets_, children_, set); }
  std::span<const LO> parents(LO set) const { return links(parent_offsets_, parents_, set); }

  std::optional<LO> index_of(GO gid) const {
    const GO* first = host_gid_.data();
    const GO* last = first + count_;
    const GO* it = std::lower_bound(first, last, gid);
    if (it == last || *it != gid) return std::nullopt;
    return static_cast<LO>(it - first);
  }

  // Local ids of dimension dim in the set and in every set it contains,
  // directly or not; sorted and unique. Containment cycles are followed once.
  std::vector<LO> members_recursive(LO set, int dim) const {
    if (dim < 0 || dim > 3) {
      throw std::invalid_argument("EntitySets: dimension " + std::to_string(dim) +
                                  " is not in [0, 3]");
    }
    (void)held(dim);
    std::vector<char> seen(static_cast<std::size_t>(count_), 0);
    std::vector<LO> todo{set};
    seen[index(set)] = 1;
    std::vector<LO> out;
    while (!todo.empty()) {
      const LO s = todo.back();
      todo.pop_back();
      const auto mine = members_of(s, dim);
      out.insert(out.end(), mine.begin(), mine.end());
      for (const LO c : members_of(s, set_dim)) {
        if (seen[static_cast<std::size_t>(c)] == 0) {
          seen[static_cast<std::size_t>(c)] = 1;
          todo.push_back(c);
        }
      }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }

  // The same sets with the members of one dimension renumbered: lid l becomes
  // new_lid[l]. Rows are sorted again; sequences keep their order. Throws
  // std::logic_error when a member has no new lid (invalid_lo).
  EntitySets renumbered(int dim, const std::vector<LO>& new_lid) const {
    const Slot& s = held(dim);
    const auto map = [&](LO l) {
      const auto at = static_cast<std::size_t>(l);
      const LO m = at < new_lid.size() ? new_lid[at] : invalid_lo;
      if (m == invalid_lo) {
        throw std::logic_error("EntitySets: member " + std::to_string(l) + " of dimension " +
                               std::to_string(dim) + " has no counterpart after renumbering");
      }
      return m;
    };
    const auto n = static_cast<std::size_t>(count_);
    View1<Offset, Kokkos::HostSpace> offs("soma::sets::offsets", n + 1);
    View1<LO, Kokkos::HostSpace> mem("soma::sets::members", s.host_members.extent(0));
    std::vector<LO> row;
    for (std::size_t i = 0; i < n; ++i) {
      row.clear();
      const auto b = static_cast<std::size_t>(s.host_offsets(i));
      const auto e = static_cast<std::size_t>(s.host_offsets(i + 1));
      for (std::size_t k = b; k < e; ++k) row.push_back(map(s.host_members(k)));
      std::sort(row.begin(), row.end());
      for (std::size_t k = 0; k < row.size(); ++k) mem(b + k) = row[k];
      offs(i) = s.host_offsets(i);
    }
    offs(n) = s.host_offsets(n);
    std::vector<SetEntry> seq = seq_;
    for (SetEntry& e : seq) {
      if (e.slot == dim) e.lid = map(e.lid);
    }
    EntitySets out;
    out.count_ = count_;
    out.gid_ = gid_;
    out.host_gid_ = host_gid_;
    out.slots_ = slots_;
    auto& t = out.slots_[static_cast<std::size_t>(dim)];
    t.offsets = Kokkos::create_mirror_view_and_copy(MemSpace{}, offs);
    t.members = Kokkos::create_mirror_view_and_copy(MemSpace{}, mem);
    t.host_offsets = offs;
    t.host_members = mem;
    out.flags_ = flags_;
    out.seq_offsets_ = seq_offsets_;
    out.seq_ = std::move(seq);
    out.child_offsets_ = child_offsets_;
    out.children_ = children_;
    out.parent_offsets_ = parent_offsets_;
    out.parents_ = parents_;
    return out;
  }

  // keep[k] selects the slots to copy; the sets slot is always copied. With
  // alias the Views are shared where Dst can use them, as for the mesh.
  template <class Dst>
  EntitySets<Dst> mirror(const std::array<bool, 4>& keep, bool alias) const {
    EntitySets<Dst> out;
    out.count_ = count_;
    transfer(out.gid_, gid_, alias);
    out.host_gid_ = host_of(out.gid_);
    for (std::size_t k = 0; k < slots_.size(); ++k) {
      const Slot& s = slots_[k];
      auto& t = out.slots_[k];
      t.present = s.present && (k == static_cast<std::size_t>(set_dim) || keep[k]);
      if (!t.present) continue;
      transfer(t.offsets, s.offsets, alias);
      transfer(t.members, s.members, alias);
      t.host_offsets = host_of(t.offsets);
      t.host_members = host_of(t.members);
    }
    out.flags_ = flags_;
    out.seq_offsets_ = seq_offsets_;
    out.seq_ = seq_;
    out.child_offsets_ = child_offsets_;
    out.children_ = children_;
    out.parent_offsets_ = parent_offsets_;
    out.parents_ = parents_;
    return out;
  }

 private:
  template <class>
  friend class EntitySets;
  friend class SetBuilder;

  struct Slot {
    bool present = true;
    View1<Offset, MemSpace> offsets;
    View1<LO, MemSpace> members;
    View1<Offset, Kokkos::HostSpace> host_offsets;
    View1<LO, Kokkos::HostSpace> host_members;
  };

  static void check_slot(int slot) {
    if (slot < 0 || slot > set_dim) {
      throw std::invalid_argument("EntitySets: slot " + std::to_string(slot) + " is not in [0, 4]");
    }
  }
  const Slot& held(int slot) const {
    check_slot(slot);
    const Slot& s = slots_[static_cast<std::size_t>(slot)];
    if (!s.present) {
      throw MissingSetMembers("EntitySets: set members of dimension " + std::to_string(slot) +
                              " were not transferred");
    }
    return s;
  }
  std::size_t index(LO set) const {
    if (set < 0 || set >= count_) {
      throw std::out_of_range("EntitySets: set " + std::to_string(set) + " is not in [0, " +
                              std::to_string(count_) + ")");
    }
    return static_cast<std::size_t>(set);
  }
  std::span<const LO> links(const std::vector<Offset>& off, const std::vector<LO>& to,
                            LO set) const {
    const std::size_t i = index(set);
    return {to.data() + off[i], static_cast<std::size_t>(off[i + 1] - off[i])};
  }

  template <class DstView, class SrcView>
  static void transfer(DstView& dst, const SrcView& src, bool alias) {
    using Dst = typename DstView::memory_space;
    if (alias) {
      dst = Kokkos::create_mirror_view_and_copy(Dst{}, src);
    } else {
      auto out = Kokkos::create_mirror(Kokkos::view_alloc(Dst{}, Kokkos::WithoutInitializing), src);
      Kokkos::deep_copy(out, src);
      dst = out;
    }
  }
  template <class V>
  static auto host_of(const V& v) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  }

  LO count_ = 0;
  View1<GO, MemSpace> gid_;
  View1<GO, Kokkos::HostSpace> host_gid_;
  std::array<Slot, num_field_slots> slots_{};
  std::vector<std::uint8_t> flags_;
  std::vector<Offset> seq_offsets_;  // count + 1; rows of unordered sets are empty
  std::vector<SetEntry> seq_;
  std::vector<Offset> child_offsets_;
  std::vector<LO> children_;
  std::vector<Offset> parent_offsets_;
  std::vector<LO> parents_;
};

}  // namespace soma

#endif  // SOMA_CORE_SETS_HPP_
