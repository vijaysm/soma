// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_RESOLVER_HPP_
#define SOMA_FIELDS_RESOLVER_HPP_

// Entity-to-value lookup across the members of a family. For each dimension
// the family builds one index, chosen by how its members cover that dimension:
//   none             no member has entities of the dimension
//   single_full      one member over every entity: row = entity
//   single_subset    one member over a subset: that member's inverse map
//   multi_intervals  several members, all interval or full: binary search on
//                    one merged table of runs (begin, end, member, first row)
//   multi_owner      several members, general subsets: owner[lid] names the
//                    member (255 = none), then that member's inverse map
// Members of a family are disjoint (checked when they are added), so a lookup
// has at most one answer.
//
// Lifetime. FamilyView is a trivially copyable bundle of unmanaged Views, so a
// kernel can capture it, but it owns nothing. The registry replaces an index
// whenever the family or the mesh epoch changes, so a bare FamilyView would
// dangle after any add, remove, commit, or epoch change. FamilyHandle is the
// host-side owner: it keeps the indexes it was built from alive (shared
// ownership) and hands out the FamilyView. Keep the handle alive for as long as
// any launch that uses its view() can run (view() on a temporary handle does
// not compile); a handle never sees later changes to the family, it keeps
// answering from the state it was taken at.

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/field.hpp"
#include "soma/fields/search.hpp"

namespace soma {

enum class DimIndexKind : std::uint8_t {
  none,
  single_full,
  single_subset,
  multi_intervals,
  multi_owner
};

constexpr const char* name(DimIndexKind k) noexcept {
  switch (k) {
    case DimIndexKind::none: return "none";
    case DimIndexKind::single_full: return "single_full";
    case DimIndexKind::single_subset: return "single_subset";
    case DimIndexKind::multi_intervals: return "multi_intervals";
    case DimIndexKind::multi_owner: return "multi_owner";
  }
  return "unknown";
}

inline constexpr std::uint8_t no_member = 255;

// The owner map names a member by one byte, 255 meaning none.
inline constexpr int max_family_members = 255;

// Where an entity's values live: the member (slot in the family's per-dimension
// member list) and the first row of the entity in that member's field.
struct FamilyHit {
  int slot = -1;
  LO row = invalid_lo;
  KOKKOS_INLINE_FUNCTION bool found() const { return slot >= 0; }
};

namespace detail {

// One member's lookup record on one dimension. Offsets index the shared pools.
struct MemberRec {
  DomainKind kind = DomainKind::full;
  LO count = 0;
  LO size = 0;
  LO runs = 0;
  LO a_off = 0;
  LO b_off = 0;
  LO o_off = 0;
  LO row_base = 0;
  LO stride = 0;  // value points per entity; 0 means the pool_o prefix is used
};

template <class V>
KOKKOS_INLINE_FUNCTION LO lower_bound_at(const V& v, LO base, LO n, LO key) {
  LO lo = 0;
  LO hi = n;
  while (lo < hi) {
    const LO mid = lo + (hi - lo) / 2;
    if (v(base + mid) < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <class V>
KOKKOS_INLINE_FUNCTION LO upper_bound_at(const V& v, LO base, LO n, LO key) {
  LO lo = 0;
  LO hi = n;
  while (lo < hi) {
    const LO mid = lo + (hi - lo) / 2;
    if (v(base + mid) <= key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

}  // namespace detail

template <class MemSpace>
struct FamilyDimView {
  using Traits = Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>;
  template <class T>
  using U = Kokkos::View<const T*, Kokkos::LayoutLeft, MemSpace, Traits>;

  DimIndexKind kind = DimIndexKind::none;
  LO count = 0;
  LO table_size = 0;
  U<detail::MemberRec> recs{};
  U<LO> pool_a{};
  U<LO> pool_b{};
  U<LO> pool_o{};
  U<LO> tb_begin{};
  U<LO> tb_end{};
  U<LO> tb_mrow0{};
  U<std::uint8_t> tb_slot{};
  U<std::uint8_t> owner{};

  KOKKOS_INLINE_FUNCTION FamilyHit in_member(int slot, LO lid, LO known_member_row) const {
    const detail::MemberRec r = recs(static_cast<std::size_t>(slot));
    LO m = known_member_row;
    if (m == invalid_lo) {
      if (lid < 0 || lid >= r.count) return FamilyHit{};
      switch (r.kind) {
        case DomainKind::full: m = lid; break;
        case DomainKind::list: {
          const LO i = detail::lower_bound_at(pool_a, r.a_off, r.size, lid);
          if (i >= r.size || pool_a(r.a_off + i) != lid) return FamilyHit{};
          m = i;
          break;
        }
        case DomainKind::intervals: {
          const LO i = detail::upper_bound_at(pool_a, r.a_off, r.runs, lid) - 1;
          if (i < 0) return FamilyHit{};
          const LO off = lid - pool_a(r.a_off + i);
          if (off >= pool_b(r.b_off + i + 1) - pool_b(r.b_off + i)) return FamilyHit{};
          m = pool_b(r.b_off + i) + off;
          break;
        }
      }
    }
    const LO local = r.stride != 0 ? r.stride * m : pool_o(r.o_off + m);
    return FamilyHit{slot, r.row_base + local};
  }

  KOKKOS_INLINE_FUNCTION FamilyHit lookup(LO lid) const {
    switch (kind) {
      case DimIndexKind::none: return FamilyHit{};
      case DimIndexKind::single_full:
      case DimIndexKind::single_subset: return in_member(0, lid, invalid_lo);
      case DimIndexKind::multi_intervals: {
        const LO i = detail::upper_bound_at(tb_begin, 0, table_size, lid) - 1;
        if (i < 0 || lid >= tb_end(i)) return FamilyHit{};
        return in_member(tb_slot(i), lid, tb_mrow0(i) + (lid - tb_begin(i)));
      }
      case DimIndexKind::multi_owner: {
        if (lid < 0 || lid >= count) return FamilyHit{};
        const std::uint8_t s = owner(lid);
        return s == no_member ? FamilyHit{} : in_member(s, lid, invalid_lo);
      }
    }
    return FamilyHit{};
  }
};

template <class MemSpace>
struct FamilyView {
  Kokkos::Array<FamilyDimView<MemSpace>, num_field_slots> dims{};

  KOKKOS_INLINE_FUNCTION FamilyHit lookup(int dim, LO lid) const {
    if (dim < 0 || dim > set_dim) return FamilyHit{};
    return dims[static_cast<std::size_t>(dim)].lookup(lid);
  }
};

namespace detail {

// Host description of one member's block on the dimension being indexed.
struct MemberBlock {
  DomainBlockView<Kokkos::HostSpace> domain;
  LO row_base = 0;
  LO stride = 0;
  View1<const LO, Kokkos::HostSpace> offsets;
};

template <class MemSpace>
struct FamilyDimData {
  DimIndexKind kind = DimIndexKind::none;
  LO count = 0;
  LO table_size = 0;

  View1<const MemberRec, Kokkos::HostSpace> h_recs;
  View1<const LO, Kokkos::HostSpace> h_pool_a, h_pool_b, h_pool_o, h_tb_begin, h_tb_end, h_tb_mrow0;
  View1<const std::uint8_t, Kokkos::HostSpace> h_tb_slot, h_owner;
  View1<const MemberRec, MemSpace> d_recs;
  View1<const LO, MemSpace> d_pool_a, d_pool_b, d_pool_o, d_tb_begin, d_tb_end, d_tb_mrow0;
  View1<const std::uint8_t, MemSpace> d_tb_slot, d_owner;

  std::vector<FieldId> slots;  // member slot -> field

  template <class S, class R, class L, class B>
  FamilyDimView<S> make(const R& recs, const L& pa, const L& pb, const L& po, const L& tbb,
                        const L& tbe, const L& tbm, const B& tbs, const B& own) const {
    FamilyDimView<S> v;
    v.kind = kind;
    v.count = count;
    v.table_size = table_size;
    v.recs = recs;
    v.pool_a = pa;
    v.pool_b = pb;
    v.pool_o = po;
    v.tb_begin = tbb;
    v.tb_end = tbe;
    v.tb_mrow0 = tbm;
    v.tb_slot = tbs;
    v.owner = own;
    return v;
  }
  FamilyDimView<MemSpace> view() const {
    return make<MemSpace>(d_recs, d_pool_a, d_pool_b, d_pool_o, d_tb_begin, d_tb_end, d_tb_mrow0,
                          d_tb_slot, d_owner);
  }
  FamilyDimView<Kokkos::HostSpace> host_view() const {
    return make<Kokkos::HostSpace>(h_recs, h_pool_a, h_pool_b, h_pool_o, h_tb_begin, h_tb_end,
                                   h_tb_mrow0, h_tb_slot, h_owner);
  }
};

template <class T>
View1<const T, Kokkos::HostSpace> to_host_view(const std::vector<T>& v, const std::string& label) {
  View1<T, Kokkos::HostSpace> out(Kokkos::view_alloc(label, Kokkos::WithoutInitializing),
                                  v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out(i) = v[i];
  return out;
}

// Builds the index of one dimension from the members that have entities there
// (slot order = the order given).
template <class MemSpace>
FamilyDimData<MemSpace> build_dim_index(const std::vector<MemberBlock>& members) {
  FamilyDimData<MemSpace> d;
  std::vector<MemberRec> recs;
  std::vector<LO> pa, pb, po;
  bool all_runs = true;
  for (const MemberBlock& m : members) {
    MemberRec r;
    r.kind = m.domain.kind;
    r.count = m.domain.count;
    r.size = m.domain.size;
    r.runs = m.domain.runs;
    r.row_base = m.row_base;
    r.stride = m.stride;
    r.a_off = to_lo(pa.size(), "family pool offset");
    r.b_off = to_lo(pb.size(), "family pool offset");
    r.o_off = to_lo(po.size(), "family pool offset");
    if (m.domain.kind == DomainKind::list) {
      for (LO i = 0; i < m.domain.size; ++i) pa.push_back(m.domain.a(i));
    } else if (m.domain.kind == DomainKind::intervals) {
      for (LO i = 0; i < m.domain.runs; ++i) pa.push_back(m.domain.a(i));
      for (LO i = 0; i <= m.domain.runs; ++i) pb.push_back(m.domain.b(i));
    }
    if (m.stride == 0) {
      for (LO i = 0; i <= m.domain.size; ++i) po.push_back(m.offsets(i));
    }
    all_runs = all_runs && m.domain.kind != DomainKind::list;
    d.count = r.count;
    recs.push_back(r);
  }

  std::vector<LO> tbb, tbe, tbm;
  std::vector<std::uint8_t> tbs, owner;
  if (members.empty()) {
    d.kind = DimIndexKind::none;
  } else if (members.size() == 1) {
    d.kind = members[0].domain.kind == DomainKind::full ? DimIndexKind::single_full
                                                        : DimIndexKind::single_subset;
  } else if (members.size() > static_cast<std::size_t>(max_family_members)) {
    throw std::length_error("family has more than 255 members on one dimension");
  } else if (all_runs) {
    d.kind = DimIndexKind::multi_intervals;
    struct Run {
      LO begin, end, mrow0;
      std::uint8_t slot;
    };
    std::vector<Run> runs;
    for (std::size_t s = 0; s < members.size(); ++s) {
      const auto& dom = members[s].domain;
      if (dom.kind == DomainKind::full) {
        runs.push_back(Run{0, dom.size, 0, static_cast<std::uint8_t>(s)});
      } else {
        for (LO i = 0; i < dom.runs; ++i) {
          runs.push_back(Run{dom.a(i), dom.a(i) + (dom.b(i + 1) - dom.b(i)), dom.b(i),
                             static_cast<std::uint8_t>(s)});
        }
      }
    }
    std::sort(runs.begin(), runs.end(), [](const Run& x, const Run& y) { return x.begin < y.begin; });
    for (const Run& r : runs) {
      tbb.push_back(r.begin);
      tbe.push_back(r.end);
      tbm.push_back(r.mrow0);
      tbs.push_back(r.slot);
    }
    d.table_size = to_lo(runs.size(), "family run table size");
  } else {
    d.kind = DimIndexKind::multi_owner;
    owner.assign(static_cast<std::size_t>(d.count), no_member);
    for (std::size_t s = 0; s < members.size(); ++s) {
      const auto& dom = members[s].domain;
      for (LO r = 0; r < dom.size; ++r) {
        owner[static_cast<std::size_t>(dom.entity(r))] = static_cast<std::uint8_t>(s);
      }
    }
  }

  d.h_recs = to_host_view(recs, "soma::family::recs");
  d.h_pool_a = to_host_view(pa, "soma::family::pool_a");
  d.h_pool_b = to_host_view(pb, "soma::family::pool_b");
  d.h_pool_o = to_host_view(po, "soma::family::pool_o");
  d.h_tb_begin = to_host_view(tbb, "soma::family::tb_begin");
  d.h_tb_end = to_host_view(tbe, "soma::family::tb_end");
  d.h_tb_mrow0 = to_host_view(tbm, "soma::family::tb_mrow0");
  d.h_tb_slot = to_host_view(tbs, "soma::family::tb_slot");
  d.h_owner = to_host_view(owner, "soma::family::owner");
  d.d_recs = Kokkos::create_mirror_view_and_copy(MemSpace{}, d.h_recs);
  d.d_pool_a = Kokkos::create_mirror_view_and_copy(MemSpace{}, d.h_pool_a);
  d.d_pool_b = Kokkos::create_mirror_view_and_copy(MemSpace{}, d.h_pool_b);
  d.d_pool_o = Kokkos::create_mirror_view_and_copy(MemSpace{}, d.h_pool_o);
  d.d_tb_begin = Kokkos::create_mirror_view_and_copy(MemSpace{}, d.h_tb_begin);
  d.d_tb_end = Kokkos::create_mirror_view_and_copy(MemSpace{}, d.h_tb_end);
  d.d_tb_mrow0 = Kokkos::create_mirror_view_and_copy(MemSpace{}, d.h_tb_mrow0);
  d.d_tb_slot = Kokkos::create_mirror_view_and_copy(MemSpace{}, d.h_tb_slot);
  d.d_owner = Kokkos::create_mirror_view_and_copy(MemSpace{}, d.h_owner);
  return d;
}

}  // namespace detail

// Owner of the indexes behind a FamilyView, one per dimension. Copies share
// the indexes. The view and the member lists stay valid, and unchanged, for as
// long as any copy of the handle lives.
template <class MemSpace>
class FamilyHandle {
 public:
  using Index = std::shared_ptr<const detail::FamilyDimData<MemSpace>>;

  explicit FamilyHandle(std::array<Index, num_field_slots> index) : index_(std::move(index)) {
    for (std::size_t d = 0; d < index_.size(); ++d) view_.dims[d] = index_[d]->view();
  }

  // The view borrows from this handle: it holds unmanaged Views into the
  // indexes the handle owns and does not extend their lifetime. Copying it is
  // cheap, but a copy is valid only while a handle (or a copy of the handle)
  // lives. Asking a temporary handle for a view is deleted, because the index
  // behind it can be replaced by the next change to the family:
  //   auto v = registry.family_handle(f).view();   // does not compile
  //   const auto h = registry.family_handle(f);    // keep the handle
  //   const auto v = h.view();
  FamilyView<MemSpace> view() const& noexcept { return view_; }
  FamilyView<MemSpace> view() const&& = delete;

  // Fields behind the member slots that lookups report on `dim`. On a
  // temporary handle the list is returned as a copy, so a range-for over
  // `registry.family_handle(f).members(d)` is safe.
  const std::vector<FieldId>& members(int dim) const& {
    return index_.at(static_cast<std::size_t>(dim))->slots;
  }
  std::vector<FieldId> members(int dim) const&& {
    return index_.at(static_cast<std::size_t>(dim))->slots;
  }

 private:
  std::array<Index, num_field_slots> index_;
  FamilyView<MemSpace> view_{};
};

}  // namespace soma

#endif  // SOMA_FIELDS_RESOLVER_HPP_
