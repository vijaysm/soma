// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_DOMAIN_HPP_
#define SOMA_FIELDS_DOMAIN_HPP_

// Which entities a field lives on. A Domain is an immutable snapshot of a set
// of local entity ids, one block per dimension, built from raw counts and ids
// (it does not depend on Mesh). Three representations, chosen automatically:
//   full       every entity of the dimension; identity maps, no storage
//   intervals  few contiguous runs; binary search on the run starts
//   list       sorted local ids; the inverse map is a binary search
// There is no rank-bitset representation and no set algebra (union,
// intersection, difference) yet; blocks of distinct dimensions can be joined by
// combine().

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/search.hpp"

namespace soma {

enum class DomainKind : std::uint8_t { full, intervals, list };

constexpr const char* name(DomainKind k) noexcept {
  switch (k) {
    case DomainKind::full: return "full";
    case DomainKind::intervals: return "intervals";
    case DomainKind::list: return "list";
  }
  return "unknown";
}

// Half-open run [begin, end) of local ids.
struct Interval {
  LO begin = 0;
  LO end = 0;
  friend bool operator==(const Interval&, const Interval&) = default;
};

KOKKOS_INLINE_FUNCTION constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Structural identity of one block. The hash combines splitmix64 of every
// member's global id with a commutative sum and a xor, so it does not depend
// on member order or on local numbering. It only rejects: equality of domains
// is always confirmed on the members themselves.
struct Fingerprint {
  std::uint64_t epoch = 0;
  int dim = 0;
  LO size = 0;
  std::uint64_t sum = 0;
  std::uint64_t xor_hash = 0;
  friend bool operator==(const Fingerprint&, const Fingerprint&) = default;
};

// Device-copyable handle on one block. intervals: a = run begins (runs),
// b = row prefix (runs + 1, b(0) = 0); list: a = sorted lids.
template <class MemSpace>
struct DomainBlockView {
  int dim = 0;
  DomainKind kind = DomainKind::full;
  LO count = 0;  // entities of this dimension in the mesh
  LO size = 0;   // members
  LO runs = 0;
  UView1<LO, MemSpace> a{};
  UView1<LO, MemSpace> b{};

  // Row in [0, size) to local id; invalid_lo when out of range.
  KOKKOS_INLINE_FUNCTION LO entity(LO row) const {
    if (row < 0 || row >= size) return invalid_lo;
    switch (kind) {
      case DomainKind::full: return row;
      case DomainKind::list: return a(row);
      case DomainKind::intervals: {
        const LO i = detail::upper_bound(b, runs, row) - 1;
        return a(i) + (row - b(i));
      }
    }
    return invalid_lo;
  }

  // Local id to row; invalid_lo when the entity is not a member.
  KOKKOS_INLINE_FUNCTION LO row(LO lid) const {
    if (lid < 0 || lid >= count) return invalid_lo;
    switch (kind) {
      case DomainKind::full: return lid;
      case DomainKind::list: {
        const LO i = detail::lower_bound(a, size, lid);
        return (i < size && a(i) == lid) ? i : invalid_lo;
      }
      case DomainKind::intervals: {
        const LO i = detail::upper_bound(a, runs, lid) - 1;
        if (i < 0) return invalid_lo;
        const LO offset = lid - a(i);
        return offset < b(i + 1) - b(i) ? b(i) + offset : invalid_lo;
      }
    }
    return invalid_lo;
  }
};

template <class MemSpace>
struct DomainView {
  struct Location {
    int dim = -1;
    LO lid = invalid_lo;
  };

  int num_blocks = 0;
  LO size = 0;
  Kokkos::Array<DomainBlockView<MemSpace>, 4> blocks{};
  Kokkos::Array<LO, 4> row_begin{};

  KOKKOS_INLINE_FUNCTION int find_block(int dim) const {
    for (int i = 0; i < num_blocks; ++i) {
      if (blocks[static_cast<std::size_t>(i)].dim == dim) return i;
    }
    return -1;
  }

  // Row in the whole domain (blocks are stacked in dimension order).
  KOKKOS_INLINE_FUNCTION LO row(int dim, LO lid) const {
    const int i = find_block(dim);
    if (i < 0) return invalid_lo;
    const auto k = static_cast<std::size_t>(i);
    const LO r = blocks[k].row(lid);
    return r == invalid_lo ? invalid_lo : row_begin[k] + r;
  }

  KOKKOS_INLINE_FUNCTION Location locate(LO row) const {
    for (std::size_t k = 0; k < static_cast<std::size_t>(num_blocks); ++k) {
      if (row >= row_begin[k] && row < row_begin[k] + blocks[k].size) {
        return Location{blocks[k].dim, blocks[k].entity(row - row_begin[k])};
      }
    }
    return Location{};
  }
};

struct DomainBlockInfo {
  int dim = 0;
  DomainKind kind = DomainKind::full;
  LO count = 0;
  LO size = 0;
  LO runs = 0;
  LO row_begin = 0;
  Fingerprint fingerprint{};
};

namespace detail {

struct HashPair {
  std::uint64_t sum = 0;
  std::uint64_t xr = 0;
};

// One parallel reduction over the rows of a block. Without global ids the
// local ids stand in for them.
template <class MemSpace>
struct MemberHash {
  using execution_space = typename MemSpace::execution_space;
  using value_type = HashPair;

  DomainBlockView<MemSpace> block;
  UView1<GO, MemSpace> gid;
  bool has_gid;

  KOKKOS_INLINE_FUNCTION void operator()(const LO row, value_type& acc) const {
    const LO lid = block.entity(row);
    const std::uint64_t key =
        has_gid ? static_cast<std::uint64_t>(gid(lid)) : static_cast<std::uint64_t>(lid);
    const std::uint64_t h = splitmix64(key);
    acc.sum += h;
    acc.xr ^= h;
  }
  KOKKOS_INLINE_FUNCTION void init(value_type& v) const { v = value_type{}; }
  KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const {
    dst.sum += src.sum;
    dst.xr ^= src.xr;
  }
};

template <class MemSpace>
struct DomainBlockData {
  DomainBlockInfo info;
  View1<const LO, MemSpace> a;
  View1<const LO, MemSpace> b;
  View1<const LO, Kokkos::HostSpace> ha;
  View1<const LO, Kokkos::HostSpace> hb;

  template <class S, class VA, class VB>
  static DomainBlockView<S> make_view(const DomainBlockInfo& i, const VA& va, const VB& vb) {
    DomainBlockView<S> v;
    v.dim = i.dim;
    v.kind = i.kind;
    v.count = i.count;
    v.size = i.size;
    v.runs = i.runs;
    v.a = va;
    v.b = vb;
    return v;
  }
  DomainBlockView<MemSpace> view() const { return make_view<MemSpace>(info, a, b); }
  DomainBlockView<Kokkos::HostSpace> host_view() const {
    return make_view<Kokkos::HostSpace>(info, ha, hb);
  }
};

// Canonical runs of a sorted id list.
inline std::vector<Interval> runs_of(std::span<const LO> lids) {
  std::vector<Interval> runs;
  for (const LO l : lids) {
    if (!runs.empty() && runs.back().end == l) {
      ++runs.back().end;
    } else {
      runs.push_back(Interval{l, l + 1});
    }
  }
  return runs;
}

template <class T>
View1<const T, Kokkos::HostSpace> host_copy(const std::vector<T>& v, const std::string& label) {
  View1<T, Kokkos::HostSpace> out(Kokkos::view_alloc(label, Kokkos::WithoutInitializing),
                                  v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out(i) = v[i];
  return out;
}

}  // namespace detail

template <class MemSpace>
class Domain {
 public:
  using memory_space = MemSpace;
  using GidView = View1<const GO, MemSpace>;

  // The empty domain: no blocks, epoch 0.
  Domain() : data_(std::make_shared<const Data>()) {}

  // Every entity of `dim`. `count` is the number of entities of that dimension.
  // `gids` (one per entity) feed the fingerprint; local ids are used when empty.
  static Domain all(int dim, LO count, std::uint64_t epoch, GidView gids = {}) {
    check_dim(dim, count);
    return single(dim, count, epoch, {Interval{0, count}}, gids);
  }

  // Disjoint, sorted runs; adjacent runs are merged and empty ones dropped.
  static Domain from_intervals(int dim, LO count, std::uint64_t epoch,
                               std::span<const Interval> intervals, GidView gids = {}) {
    check_dim(dim, count);
    std::vector<Interval> runs;
    for (const Interval& iv : intervals) {
      if (iv.begin < 0 || iv.end > count || iv.begin > iv.end) {
        throw std::invalid_argument("Domain: interval [" + std::to_string(iv.begin) + ", " +
                                    std::to_string(iv.end) + ") is outside [0, " +
                                    std::to_string(count) + ")");
      }
      if (iv.begin == iv.end) continue;
      if (!runs.empty() && iv.begin < runs.back().end) {
        throw std::invalid_argument("Domain: intervals are not sorted and disjoint at " +
                                    std::to_string(iv.begin));
      }
      if (!runs.empty() && iv.begin == runs.back().end) {
        runs.back().end = iv.end;
      } else {
        runs.push_back(iv);
      }
    }
    return single(dim, count, epoch, std::move(runs), gids);
  }

  // Strictly increasing local ids in [0, count).
  static Domain from_list(int dim, LO count, std::uint64_t epoch,
                          std::span<const LO> sorted_lids, GidView gids = {}) {
    check_dim(dim, count);
    for (std::size_t i = 0; i < sorted_lids.size(); ++i) {
      const LO l = sorted_lids[i];
      if (l < 0 || l >= count) {
        throw std::invalid_argument("Domain: entity " + std::to_string(l) + " is outside [0, " +
                                    std::to_string(count) + ")");
      }
      if (i > 0 && l <= sorted_lids[i - 1]) {
        throw std::invalid_argument("Domain: list is not strictly increasing at entity " +
                                    std::to_string(l));
      }
    }
    return single(dim, count, epoch, detail::runs_of(sorted_lids), gids);
  }

  int num_blocks() const noexcept { return static_cast<int>(data_->blocks.size()); }
  LO size() const noexcept { return data_->size; }
  std::uint64_t epoch() const noexcept { return data_->epoch; }
  bool empty() const noexcept { return data_->blocks.empty(); }

  const DomainBlockInfo& info(int b) const { return data_->blocks.at(static_cast<std::size_t>(b)).info; }
  const Fingerprint& fingerprint(int b) const { return info(b).fingerprint; }

  // Index of the block of dimension `dim`, or -1.
  int find_block(int dim) const noexcept {
    for (int i = 0; i < num_blocks(); ++i) {
      if (info(i).dim == dim) return i;
    }
    return -1;
  }
  bool contains_dim(int dim) const noexcept { return find_block(dim) >= 0; }

  DomainBlockView<MemSpace> block(int b) const {
    return data_->blocks.at(static_cast<std::size_t>(b)).view();
  }
  // Host copy of the same block; an alias when MemSpace is host-accessible.
  DomainBlockView<Kokkos::HostSpace> host_block(int b) const {
    return data_->blocks.at(static_cast<std::size_t>(b)).host_view();
  }

  DomainView<MemSpace> view() const {
    DomainView<MemSpace> v;
    v.num_blocks = num_blocks();
    v.size = size();
    for (int i = 0; i < num_blocks(); ++i) {
      v.blocks[static_cast<std::size_t>(i)] = block(i);
      v.row_begin[static_cast<std::size_t>(i)] = info(i).row_begin;
    }
    return v;
  }

  // Host lookups. row() returns a row of the whole domain.
  LO row(int dim, LO lid) const {
    const int i = find_block(dim);
    if (i < 0) return invalid_lo;
    const LO r = host_block(i).row(lid);
    return r == invalid_lo ? invalid_lo : info(i).row_begin + r;
  }

  std::vector<LO> members(int b) const {
    const auto hv = host_block(b);
    std::vector<LO> out(static_cast<std::size_t>(hv.size));
    for (LO r = 0; r < hv.size; ++r) out[static_cast<std::size_t>(r)] = hv.entity(r);
    return out;
  }

  // Same epoch, same blocks, same fingerprints, and identical members. The
  // fingerprint only rejects early; a hash collision cannot make two different
  // domains equal.
  bool same_fingerprints(const Domain& o) const {
    if (epoch() != o.epoch() || num_blocks() != o.num_blocks()) return false;
    for (int i = 0; i < num_blocks(); ++i) {
      if (!(fingerprint(i) == o.fingerprint(i))) return false;
    }
    return true;
  }

  bool same_members(const Domain& o) const {
    if (num_blocks() != o.num_blocks()) return false;
    for (int i = 0; i < num_blocks(); ++i) {
      const auto& x = data_->blocks[static_cast<std::size_t>(i)];
      const auto& y = o.data_->blocks[static_cast<std::size_t>(i)];
      if (x.info.dim != y.info.dim || x.info.kind != y.info.kind || x.info.count != y.info.count ||
          x.info.size != y.info.size || x.info.runs != y.info.runs) {
        return false;
      }
      if (!std::equal(x.ha.data(), x.ha.data() + x.ha.extent(0), y.ha.data(), y.ha.data() + y.ha.extent(0)) ||
          !std::equal(x.hb.data(), x.hb.data() + x.hb.extent(0), y.hb.data(), y.hb.data() + y.hb.extent(0))) {
        return false;
      }
    }
    return true;
  }

  friend bool operator==(const Domain& a, const Domain& b) {
    return a.same_fingerprints(b) && a.same_members(b);
  }

  // True when both handles share one snapshot.
  bool shares_storage_with(const Domain& o) const noexcept { return data_ == o.data_; }
  const void* storage_id() const noexcept { return data_.get(); }

  // Domains are immutable, so the same memory space shares the snapshot.
  template <class Dst>
  Domain<Dst> mirror() const {
    if constexpr (std::is_same_v<Dst, MemSpace>) {
      return *this;
    } else {
      typename Domain<Dst>::Data d;
      d.epoch = data_->epoch;
      d.size = data_->size;
      for (const auto& blk : data_->blocks) {
        detail::DomainBlockData<Dst> out;
        out.info = blk.info;
        out.ha = blk.ha;
        out.hb = blk.hb;
        out.a = Kokkos::create_mirror_view_and_copy(Dst{}, blk.ha);
        out.b = Kokkos::create_mirror_view_and_copy(Dst{}, blk.hb);
        d.blocks.push_back(std::move(out));
      }
      return Domain<Dst>(std::make_shared<const typename Domain<Dst>::Data>(std::move(d)));
    }
  }

  // Joins two domains of distinct dimensions built against the same epoch.
  // General set algebra (union, intersection, difference of overlapping
  // blocks) is a later addition.
  static Domain combine(const Domain& x, const Domain& y) {
    if (x.empty()) return y;
    if (y.empty()) return x;
    if (x.epoch() != y.epoch()) {
      throw std::invalid_argument("Domain::combine: epochs differ (" + std::to_string(x.epoch()) +
                                  " vs " + std::to_string(y.epoch()) + ")");
    }
    Data d;
    d.epoch = x.epoch();
    d.blocks = x.data_->blocks;
    for (const auto& blk : y.data_->blocks) {
      if (x.contains_dim(blk.info.dim)) {
        throw std::invalid_argument("Domain::combine: both domains have a block of dimension " +
                                    std::to_string(blk.info.dim));
      }
      d.blocks.push_back(blk);
    }
    std::sort(d.blocks.begin(), d.blocks.end(),
              [](const auto& l, const auto& r) { return l.info.dim < r.info.dim; });
    LO row = 0;
    for (auto& blk : d.blocks) {
      blk.info.row_begin = row;
      row += blk.info.size;
    }
    d.size = row;
    return Domain(std::make_shared<const Data>(std::move(d)));
  }

 private:
  template <class>
  friend class Domain;

  struct Data {
    std::vector<detail::DomainBlockData<MemSpace>> blocks;
    LO size = 0;
    std::uint64_t epoch = 0;
  };

  explicit Domain(std::shared_ptr<const Data> d) : data_(std::move(d)) {}

  static void check_dim(int dim, LO count) {
    if (dim < 0 || dim > 3) {
      throw std::invalid_argument("Domain: dimension " + std::to_string(dim) + " is not in [0, 3]");
    }
    if (count < 0) throw std::invalid_argument("Domain: negative entity count");
  }

  // Picks the representation that needs the least storage, builds the block,
  // and fingerprints it.
  static Domain single(int dim, LO count, std::uint64_t epoch, std::vector<Interval> runs,
                       const GidView& gids) {
    if (gids.extent(0) != 0 && static_cast<LO>(gids.extent(0)) != count) {
      throw std::invalid_argument("Domain: " + std::to_string(gids.extent(0)) +
                                  " global ids for " + std::to_string(count) + " entities");
    }
    LO size = 0;
    for (const Interval& r : runs) size += r.end - r.begin;
    const auto nruns = static_cast<LO>(runs.size());

    detail::DomainBlockData<MemSpace> blk;
    blk.info.dim = dim;
    blk.info.count = count;
    blk.info.size = size;
    std::vector<LO> a;
    std::vector<LO> b;
    if (size == count) {
      blk.info.kind = DomainKind::full;
    } else if (2 * nruns + 1 < size) {
      blk.info.kind = DomainKind::intervals;
      blk.info.runs = nruns;
      LO row = 0;
      for (const Interval& r : runs) {
        a.push_back(r.begin);
        b.push_back(row);
        row += r.end - r.begin;
      }
      b.push_back(row);
    } else {
      blk.info.kind = DomainKind::list;
      for (const Interval& r : runs) {
        for (LO l = r.begin; l < r.end; ++l) a.push_back(l);
      }
    }
    blk.ha = detail::host_copy(a, "soma::domain::a");
    blk.hb = detail::host_copy(b, "soma::domain::b");
    blk.a = Kokkos::create_mirror_view_and_copy(MemSpace{}, blk.ha);
    blk.b = Kokkos::create_mirror_view_and_copy(MemSpace{}, blk.hb);

    detail::HashPair h;
    if (size > 0) {
      const detail::MemberHash<MemSpace> f{blk.view(), gids, gids.extent(0) != 0};
      Kokkos::parallel_reduce("soma::domain::fingerprint",
                              Kokkos::RangePolicy<typename MemSpace::execution_space>(0, size), f,
                              h);
    }
    blk.info.fingerprint = Fingerprint{epoch, dim, size, h.sum, h.xr};

    Data d;
    d.epoch = epoch;
    d.size = size;
    d.blocks.push_back(std::move(blk));
    return Domain(std::make_shared<const Data>(std::move(d)));
  }

  std::shared_ptr<const Data> data_;
};

}  // namespace soma

#endif  // SOMA_FIELDS_DOMAIN_HPP_
