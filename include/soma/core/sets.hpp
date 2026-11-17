// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_SETS_HPP_
#define SOMA_CORE_SETS_HPP_

/// \file
/// Entity sets: groups of entities and of other sets, as MOAB's meshsets.
/// Membership is one row per set and field slot (dimensions 0 to 3, then the
/// sets): sorted, unique ids held in the smallest of the forms of
/// soma/core/lid_runs.hpp (full, intervals or list), readable on the device
/// through SetsView.
/// Flags, the file order of ordered sets and the parent/child links stay on the
/// host. Sets are made by SetBuilder (soma/core/set_builder.hpp).

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/lid_runs.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/topology/adjacency.hpp"

namespace soma {

/// MOAB's meshset flags as a set keeps them; the file's ranged-storage bit
/// (0x8) is not kept.
inline constexpr std::uint8_t set_track_owner = 0x1;
inline constexpr std::uint8_t set_unique = 0x2;
inline constexpr std::uint8_t set_ordered = 0x4;
inline constexpr std::uint8_t set_flag_bits = 0x7;

/// A member: an entity (slot 0 to 3 and its local id) or a set (set_dim and its
/// index).
struct SetEntry {
  int slot = 0;
  LO lid = invalid_lo;
  friend bool operator==(const SetEntry&, const SetEntry&) = default;
  friend auto operator<=>(const SetEntry&, const SetEntry&) = default;
};

/// Members of a dimension a mirror did not take were asked for.
class MissingSetMembers : public std::logic_error {
 public:
  using std::logic_error::logic_error;
};

namespace detail {
/// Indexes a pointer with operator(), as the run search expects.
struct PtrAt {
  const LO* p = nullptr;
  KOKKOS_INLINE_FUNCTION LO operator()(LO i) const { return p[i]; }
};
}  // namespace detail

/// One set's members in one slot, read in place: size sorted, unique ids out of
/// [0, count). r[k] is the k-th member (O(log runs) for intervals), row(lid)
/// its index or invalid_lo; the iterator walks the runs in order. Valid while
/// the EntitySets it came from lives.
struct SetRow {
  DomainKind kind = DomainKind::list;
  LO count = 0;  ///< entities of the slot (sets for set_dim)
  LO size = 0;   ///< members
  LO runs = 0;
  const LO* a = nullptr;
  const LO* b = nullptr;

  /// The row whose stored ids, len of them, start at p.
  KOKKOS_INLINE_FUNCTION static SetRow of(DomainKind kind, LO count, const LO* p, LO len) {
    SetRow r;
    r.kind = kind;
    r.count = count;
    r.a = p;
    switch (kind) {
      case DomainKind::full: r.size = count; break;
      case DomainKind::list: r.size = len; break;
      case DomainKind::intervals:
        r.runs = (len - 1) / 2;
        r.b = p + r.runs;
        r.size = r.b[r.runs];
        break;
    }
    return r;
  }

  KOKKOS_INLINE_FUNCTION LO operator[](LO k) const {
    return detail::run_entity(kind, size, runs, detail::PtrAt{a}, detail::PtrAt{b}, k);
  }
  KOKKOS_INLINE_FUNCTION LO row(LO lid) const {
    return detail::run_row(kind, count, size, runs, detail::PtrAt{a}, detail::PtrAt{b}, lid);
  }
  KOKKOS_INLINE_FUNCTION bool contains(LO lid) const { return row(lid) != invalid_lo; }
  KOKKOS_INLINE_FUNCTION bool empty() const { return size == 0; }

  class iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = LO;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = LO;

    iterator() = default;
    KOKKOS_INLINE_FUNCTION iterator(DomainKind kind, LO runs, const LO* a, const LO* b, LO k)
        : kind_(kind), runs_(runs), a_(a), b_(b), k_(k) {}
    KOKKOS_INLINE_FUNCTION LO operator*() const {
      switch (kind_) {
        case DomainKind::full: return k_;
        case DomainKind::list: return a_[k_];
        case DomainKind::intervals: return a_[run_] + (k_ - b_[run_]);
      }
      return invalid_lo;
    }
    KOKKOS_INLINE_FUNCTION iterator& operator++() {
      ++k_;
      if (kind_ == DomainKind::intervals && run_ + 1 < runs_ && k_ == b_[run_ + 1]) ++run_;
      return *this;
    }
    KOKKOS_INLINE_FUNCTION iterator operator++(int) {
      iterator old = *this;
      ++*this;
      return old;
    }
    KOKKOS_INLINE_FUNCTION friend bool operator==(const iterator& x, const iterator& y) { return x.k_ == y.k_; }

   private:
    DomainKind kind_ = DomainKind::list;
    LO runs_ = 0;
    const LO* a_ = nullptr;
    const LO* b_ = nullptr;
    LO k_ = 0;
    LO run_ = 0;  ///< the run of k_ (intervals)
  };
  KOKKOS_INLINE_FUNCTION iterator begin() const { return iterator(kind, runs, a, b, 0); }
  KOKKOS_INLINE_FUNCTION iterator end() const { return iterator(kind, runs, a, b, size); }
};

namespace detail {
/// The runs of a row, read without listing its members.
inline std::vector<Interval> runs_of(const SetRow& r) {
  std::vector<Interval> out;
  switch (r.kind) {
    case DomainKind::full:
      if (r.size > 0) out.push_back(Interval{0, r.size});
      break;
    case DomainKind::intervals:
      for (LO i = 0; i < r.runs; ++i) out.push_back(Interval{r.a[i], r.a[i] + (r.b[i + 1] - r.b[i])});
      break;
    case DomainKind::list:
      out = runs_of(std::span<const LO>(r.a, static_cast<std::size_t>(r.size)));
      break;
  }
  return out;
}
}  // namespace detail

/// Kernel handle; valid while the EntitySets it came from lives. Rows exist for
/// the slots the sets hold (EntitySets::has_members).
template <class MemSpace>
struct SetsView {
  LO count = 0;
  UView1<GO, MemSpace> gid{};
  Kokkos::Array<LO, num_field_slots> entities{};
  Kokkos::Array<UView1<std::uint8_t, MemSpace>, num_field_slots> kinds{};
  Kokkos::Array<UView1<Offset, MemSpace>, num_field_slots> offsets{};
  Kokkos::Array<UView1<LO, MemSpace>, num_field_slots> data{};

  KOKKOS_INLINE_FUNCTION SetRow row(int slot, LO set) const {
    const auto k = static_cast<std::size_t>(slot);
    const Offset b = offsets[k](set);
    return SetRow::of(static_cast<DomainKind>(kinds[k](set)), entities[k], data[k].data() + b,
                      static_cast<LO>(offsets[k](set + 1) - b));
  }
  KOKKOS_INLINE_FUNCTION bool contains(int slot, LO set, LO lid) const { return row(slot, set).contains(lid); }
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
  /// Strictly increasing; the file ids of sets read from a file.
  View1<const GO, MemSpace> gid() const noexcept { return gid_; }

  /// False for a dimension that a mirror did not take; the sets slot is always
  /// held.
  bool has_members(int slot) const {
    check_slot(slot);
    return slots_[static_cast<std::size_t>(slot)].present;
  }
  /// Storage of one slot: the form of each row (a DomainKind), count + 1
  /// offsets into data, and the stored ids of every row; none at all when there
  /// are no sets. entities() is the count the rows are out of. Throws
  /// MissingSetMembers for a slot the sets do not hold.
  View1<const std::uint8_t, MemSpace> kinds(int slot) const { return held(slot).kinds; }
  View1<const Offset, MemSpace> offsets(int slot) const { return held(slot).offsets; }
  View1<const LO, MemSpace> data(int slot) const { return held(slot).data; }
  LO entities(int slot) const { return held(slot).entities; }

  /// A view of temporary sets would dangle, so that overload is deleted.
  SetsView<MemSpace> view() const& {
    SetsView<MemSpace> v;
    v.count = count_;
    v.gid = gid_;
    for (std::size_t k = 0; k < slots_.size(); ++k) {
      v.entities[k] = slots_[k].entities;
      v.kinds[k] = slots_[k].kinds;
      v.offsets[k] = slots_[k].offsets;
      v.data[k] = slots_[k].data;
    }
    return v;
  }
  SetsView<MemSpace> view() const&& = delete;

  /// Host queries. A set is named by its index, 0 to count() - 1.
  std::uint8_t flags(LO set) const { return flags_[index(set)]; }
  bool ordered(LO set) const { return (flags(set) & set_ordered) != 0; }

  /// Sorted, unique local ids (set indices for set_dim) of one slot, in host
  /// memory; std::vector<LO>(r.begin(), r.end()) lists them.
  SetRow members_of(LO set, int slot) const {
    const Slot& s = held(slot);
    const std::size_t i = index(set);
    const Offset b = s.host_offsets(i);
    return SetRow::of(static_cast<DomainKind>(s.host_kinds(i)), s.entities, s.host_data.data() + b,
                      static_cast<LO>(s.host_offsets(i + 1) - b));
  }

  /// Contents of an ordered set in their stored order, repeats included. Throws
  /// std::logic_error for an unordered set, and MissingSetMembers when an entry
  /// is on a dimension the sets do not hold.
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

  /// Local ids of dimension dim in the set and in every set it contains,
  /// directly or not; sorted and unique. Containment cycles are followed once.
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

  /// The same sets with the members of one dimension renumbered: lid l becomes
  /// new_lid[l], out of count entities. Rows are sorted and take their form
  /// again; sequences keep their order. A member with no new lid (invalid_lo)
  /// throws std::logic_error, or with drop leaves the set, sequences included.
  EntitySets renumbered(int dim, const std::vector<LO>& new_lid, LO count, bool drop = false) const {
    (void)held(dim);
    const auto map = [&](LO l) {
      const auto at = static_cast<std::size_t>(l);
      const LO m = at < new_lid.size() ? new_lid[at] : invalid_lo;
      if (m == invalid_lo && !drop) {
        throw std::logic_error("EntitySets: member " + std::to_string(l) + " of dimension " +
                               std::to_string(dim) + " has no counterpart after renumbering");
      }
      return m;
    };
    std::vector<std::vector<Interval>> rows(static_cast<std::size_t>(count_));
    std::vector<LO> row;
    for (LO i = 0; i < count_; ++i) {
      row.clear();
      for (const LO l : members_of(i, dim)) {
        if (const LO m = map(l); m != invalid_lo) row.push_back(m);
      }
      std::sort(row.begin(), row.end());
      rows[static_cast<std::size_t>(i)] = detail::runs_of(row);
    }
    std::vector<SetEntry> seq;
    std::vector<Offset> seq_offsets(seq_offsets_.size(), 0);
    seq.reserve(seq_.size());
    for (std::size_t i = 0; i + 1 < seq_offsets_.size(); ++i) {
      for (auto k = static_cast<std::size_t>(seq_offsets_[i]); k < static_cast<std::size_t>(seq_offsets_[i + 1]); ++k) {
        SetEntry e = seq_[k];
        if (e.slot == dim) e.lid = map(e.lid);
        if (e.lid != invalid_lo) seq.push_back(e);
      }
      seq_offsets[i + 1] = static_cast<Offset>(seq.size());
    }
    EntitySets out;
    out.count_ = count_;
    out.gid_ = gid_;
    out.host_gid_ = host_gid_;
    out.slots_ = slots_;
    out.slots_[static_cast<std::size_t>(dim)] = Slot::packed(rows, count);
    out.flags_ = flags_;
    out.seq_offsets_ = std::move(seq_offsets);
    out.seq_ = std::move(seq);
    out.child_offsets_ = child_offsets_;
    out.children_ = children_;
    out.parent_offsets_ = parent_offsets_;
    out.parents_ = parents_;
    return out;
  }

  /// keep[k] selects the slots to copy; the sets slot is always copied. With
  /// alias the Views are shared where Dst can use them, as for the mesh.
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
      t.entities = s.entities;
      transfer(t.kinds, s.kinds, alias);
      transfer(t.offsets, s.offsets, alias);
      transfer(t.data, s.data, alias);
      t.host_kinds = host_of(t.kinds);
      t.host_offsets = host_of(t.offsets);
      t.host_data = host_of(t.data);
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
    LO entities = 0;
    View1<std::uint8_t, MemSpace> kinds;
    View1<Offset, MemSpace> offsets;
    View1<LO, MemSpace> data;
    View1<std::uint8_t, Kokkos::HostSpace> host_kinds;
    View1<Offset, Kokkos::HostSpace> host_offsets;
    View1<LO, Kokkos::HostSpace> host_data;

    /// Rows given as sorted, disjoint, merged runs of ids out of count, each
    /// held in its smallest form.
    static Slot packed(const std::vector<std::vector<Interval>>& rows, LO count) {
      const std::size_t n = rows.size();
      std::vector<std::uint8_t> kinds(n);
      std::vector<Offset> offsets(n + 1, 0);
      std::vector<LO> data;
      for (std::size_t i = 0; i < n; ++i) {
        const auto& runs = rows[i];
        std::size_t size = 0;
        for (const Interval& r : runs) size += static_cast<std::size_t>(r.end - r.begin);
        const DomainKind k = detail::form_of(static_cast<LO>(size), count, static_cast<LO>(runs.size()));
        kinds[i] = static_cast<std::uint8_t>(k);
        const std::vector<LO> packed = detail::pack_runs(runs, k);
        data.insert(data.end(), packed.begin(), packed.end());
        offsets[i + 1] = detail::to_offset(data.size(), "set membership size");
      }
      Slot s;
      s.entities = count;
      s.host_kinds = host_view(kinds, "soma::sets::kinds");
      s.host_offsets = host_view(n == 0 ? std::vector<Offset>{} : offsets, "soma::sets::offsets");
      s.host_data = host_view(data, "soma::sets::data");
      s.kinds = Kokkos::create_mirror_view_and_copy(MemSpace{}, s.host_kinds);
      s.offsets = Kokkos::create_mirror_view_and_copy(MemSpace{}, s.host_offsets);
      s.data = Kokkos::create_mirror_view_and_copy(MemSpace{}, s.host_data);
      return s;
    }
  };
  template <class T>
  static View1<T, Kokkos::HostSpace> host_view(const std::vector<T>& v, const char* label) {
    View1<T, Kokkos::HostSpace> out(Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing),
                                    v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out(i) = v[i];
    return out;
  }

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
  std::vector<Offset> seq_offsets_;  ///< count + 1; rows of unordered sets are empty
  std::vector<SetEntry> seq_;
  std::vector<Offset> child_offsets_;
  std::vector<LO> children_;
  std::vector<Offset> parent_offsets_;
  std::vector<LO> parents_;
};

}  // namespace soma

#endif  // SOMA_CORE_SETS_HPP_
