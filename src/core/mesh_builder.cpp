// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/core/mesh_builder.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "soma/core/sfc.hpp"
#include "soma/fields/domain.hpp"
#include "soma/topology/nodes.hpp"

namespace soma {

MeshBuilder::MeshBuilder(int dim) : dim_(dim) {
  if (dim < 1 || dim > 3) throw std::invalid_argument("MeshBuilder: dimension must be 1, 2, or 3");
}

void MeshBuilder::add_vertices(std::span<const GO> gids, std::span<const Real> xyz) {
  if (xyz.size() != 3 * gids.size()) {
    throw std::invalid_argument("add_vertices: xyz must hold 3 values per vertex");
  }
  for (std::size_t i = 0; i < xyz.size(); ++i) {
    if (!std::isfinite(xyz[i])) {
      throw std::invalid_argument("add_vertices: vertex gid " + std::to_string(gids[i / 3]) +
                                  " has a non-finite coordinate");
    }
  }
  const std::size_t old = vertex_gids_.size();
  vertex_gids_.insert(vertex_gids_.end(), gids.begin(), gids.end());
  try {
    xyz_.insert(xyz_.end(), xyz.begin(), xyz.end());
  } catch (...) {
    vertex_gids_.resize(old);  // keep the two arrays paired
    throw;
  }
}

void MeshBuilder::add_elements(Topology t, std::span<const GO> gids,
                               std::span<const GO> vertex_gids) {
  add_elements(t, gids, vertex_gids, num_vertices(t));
}

void MeshBuilder::add_elements(Topology t, std::span<const GO> gids, std::span<const GO> node_gids,
                               int nodes_per_element) {
  if (t == Topology::polyhedron) {
    throw std::invalid_argument("polyhedra need face connectivity and are not supported yet");
  }
  if (t == Topology::polygon) throw std::invalid_argument("use add_polygons for polygons");
  if (dimension(t) < 1 || dimension(t) > dim_) {
    throw std::invalid_argument(std::string("add_elements: ") + name(t) +
                                " elements have dimension " + std::to_string(dimension(t)) +
                                ", outside [1, " + std::to_string(dim_) + "]");
  }
  const int nv = num_vertices(t);
  if (mid_node_bits(t, nodes_per_element) < 0) {
    throw std::invalid_argument(std::string("add_elements: no ") + name(t) + " element has " +
                                std::to_string(nodes_per_element) + " nodes");
  }
  const auto n = static_cast<std::size_t>(nodes_per_element);
  if (node_gids.size() != gids.size() * n) {
    throw std::invalid_argument("add_elements: connectivity size does not match the topology");
  }
  const auto stride = static_cast<std::size_t>(nv);
  (void)detail::to_offset(gids.size() * stride, "add_elements: connectivity size");
  Block b{t, {gids.begin(), gids.end()}, {}, {}, static_cast<std::uint16_t>(n - stride), {}};
  if (n == stride) {
    b.vertex_gids.assign(node_gids.begin(), node_gids.end());
  } else {
    b.vertex_gids.reserve(gids.size() * stride);
    b.node_gids.reserve(gids.size() * (n - stride));
    for (std::size_t i = 0; i < gids.size(); ++i) {
      const auto first = node_gids.begin() + static_cast<std::ptrdiff_t>(i * n);
      b.vertex_gids.insert(b.vertex_gids.end(), first, first + static_cast<std::ptrdiff_t>(stride));
      b.node_gids.insert(b.node_gids.end(), first + static_cast<std::ptrdiff_t>(stride),
                         first + static_cast<std::ptrdiff_t>(n));
    }
  }
  b.offsets.resize(gids.size() + 1);
  for (std::size_t i = 0; i <= gids.size(); ++i) b.offsets[i] = static_cast<Offset>(i * stride);
  blocks_[static_cast<std::size_t>(dimension(t))].push_back(std::move(b));
}

void MeshBuilder::add_polygons(std::span<const GO> gids, std::span<const Offset> offsets,
                               std::span<const GO> vertex_gids) {
  if (offsets.size() != gids.size() + 1) {
    throw std::invalid_argument("add_polygons: offsets must have one more entry than gids");
  }
  if (dim_ < 2) throw std::invalid_argument("add_polygons: polygons need dimension 2 or 3");
  if (offsets[0] != 0) throw std::invalid_argument("add_polygons: offsets must start at 0");
  constexpr Offset max_arity = std::numeric_limits<std::uint16_t>::max();
  for (std::size_t i = 0; i < gids.size(); ++i) {
    if (offsets[i + 1] < offsets[i]) {
      throw std::invalid_argument("add_polygons: offsets decrease at polygon " +
                                  std::to_string(gids[i]));
    }
    const Offset arity = offsets[i + 1] - offsets[i];
    if (arity < 3 || arity > max_arity) {
      throw std::invalid_argument("add_polygons: polygon " + std::to_string(gids[i]) +
                                  " has arity " + std::to_string(arity) + ", outside [3, " +
                                  std::to_string(max_arity) + "]");
    }
  }
  if (static_cast<std::size_t>(offsets.back()) != vertex_gids.size()) {
    throw std::invalid_argument("add_polygons: offsets end at " + std::to_string(offsets.back()) +
                                " but there are " + std::to_string(vertex_gids.size()) +
                                " vertex gids");
  }
  blocks_[2].push_back(Block{Topology::polygon,
                             {gids.begin(), gids.end()},
                             {offsets.begin(), offsets.end()},
                             {vertex_gids.begin(), vertex_gids.end()},
                             0,
                             {}});
}

std::vector<GO> MeshBuilder::element_gids(int dim) const {
  std::vector<GO> out;
  if (dim < 1 || dim > 3) return out;
  for (const Block& blk : blocks_[static_cast<std::size_t>(dim)]) out.insert(out.end(), blk.gids.begin(), blk.gids.end());
  return out;
}

void MeshBuilder::set_ownership(int dim, std::span<const GO> gids, std::span<const Ownership> kinds,
                                std::span<const std::uint32_t> groups) {
  if (dim < 0 || dim > dim_) {
    throw std::invalid_argument("set_ownership: dimension " + std::to_string(dim) + " is outside [0, " +
                                std::to_string(dim_) + "]");
  }
  if (kinds.size() != gids.size() || groups.size() != gids.size()) {
    throw std::invalid_argument("set_ownership: gids, kinds and groups differ in length");
  }
  auto& o = owned_[static_cast<std::size_t>(dim)];
  for (std::size_t i = 0; i < gids.size(); ++i) o.push_back(Owned{gids[i], kinds[i], groups[i]});
}

void MeshBuilder::set_key_box(const std::array<Real, 3>& lo, const std::array<Real, 3>& hi) {
  for (std::size_t d = 0; d < 3; ++d) {
    if (!std::isfinite(lo[d]) || !std::isfinite(hi[d]) || lo[d] > hi[d]) {
      throw std::invalid_argument("set_key_box: the corners are not finite with lo <= hi");
    }
  }
  key_box_ = std::array<Real, 6>{lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]};
}

namespace {

struct Keys {
  sfc::KeyBox box;
  bool active = false;

  std::uint64_t operator()(Real x, Real y, Real z) const {
    return active ? sfc::hilbert_key(box, x, y, z) : 0;
  }
};

// Input in file order is usually sorted already; checking is linear.
template <class It, class Less>
void sort_unless_sorted(It first, It last, Less less) {
  if (!std::is_sorted(first, last, less)) std::sort(first, last, less);
}

// Records are ordered through their positions: (key, position) pairs are
// sorted, and the records moved once at the end (permute).
struct Keyed {
  std::uint64_t key;
  std::size_t at;
};

// Stable LSD radix sort of the pairs by key, 11 bits a pass; a digit every
// key shares takes no pass, so equal keys cost one scan.
void radix_sort(std::vector<Keyed>& v) {
  if (v.size() < 2) return;
  std::uint64_t all_or = 0;
  std::uint64_t all_and = ~std::uint64_t{0};
  for (const Keyed& r : v) {
    all_or |= r.key;
    all_and &= r.key;
  }
  const std::uint64_t varying = all_or ^ all_and;
  constexpr int bits = 11;
  constexpr std::size_t buckets = std::size_t{1} << bits;
  std::vector<Keyed> tmp;
  std::vector<std::size_t> start(buckets);
  for (int shift = 0; shift < 64; shift += bits) {
    if (((varying >> shift) & (buckets - 1)) == 0) continue;
    if (tmp.empty()) tmp.resize(v.size());
    std::fill(start.begin(), start.end(), std::size_t{0});
    for (const Keyed& r : v) ++start[(r.key >> shift) & (buckets - 1)];
    std::size_t sum = 0;
    for (std::size_t& c : start) sum += std::exchange(c, sum);
    for (const Keyed& r : v) tmp[start[(r.key >> shift) & (buckets - 1)]++] = r;
    v.swap(tmp);
  }
}

// The positions of n records by (key, tie): by key, then each run of equal
// keys (all of them with Ordering::file, few on a curve) by `tie`, a strict
// order of positions. Input already in order costs a scan.
template <class KeyOf, class Tie>
std::vector<std::size_t> order_by_key(std::size_t n, KeyOf key_of, Tie tie) {
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  const auto before = [&](std::size_t a, std::size_t b) {
    const std::uint64_t ka = key_of(a), kb = key_of(b);
    return ka != kb ? ka < kb : tie(a, b);
  };
  if (std::is_sorted(order.begin(), order.end(), before)) return order;
  std::vector<Keyed> v(n);
  std::uint64_t all_or = 0;
  std::uint64_t all_and = ~std::uint64_t{0};
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = {key_of(i), i};
    all_or |= v[i].key;
    all_and &= v[i].key;
  }
  // One pass on the 16 highest bits that vary, then each bucket (a few keys
  // on a curve) sorted alone; stability is not needed, ties being by `tie`.
  const std::uint64_t varying = all_or ^ all_and;
  if (varying != 0) {
    const int top = 64 - std::countl_zero(varying);
    const int shift = std::max(0, top - 16);
    const auto bucket = [&](const Keyed& r) { return static_cast<std::size_t>((r.key >> shift) & 0xFFFF); };
    std::vector<std::size_t> start(std::size_t{1} << 16, 0);
    for (const Keyed& r : v) ++start[bucket(r)];
    std::size_t sum = 0;
    for (std::size_t& c : start) sum += std::exchange(c, sum);
    std::vector<Keyed> sorted(n);
    for (const Keyed& r : v) sorted[start[bucket(r)]++] = r;
    v.swap(sorted);
    const auto by_key = [](const Keyed& a, const Keyed& b) { return a.key < b.key; };
    for (std::size_t lo = 0, b = 0; b < start.size(); ++b) {
      if (start[b] - lo > 1) {
        std::sort(v.begin() + static_cast<std::ptrdiff_t>(lo), v.begin() + static_cast<std::ptrdiff_t>(start[b]),
                  by_key);
      }
      lo = start[b];
    }
  }
  for (std::size_t i = 0; i < n; ++i) order[i] = v[i].at;
  for (std::size_t lo = 0; lo < n;) {
    std::size_t hi = lo + 1;
    while (hi < n && v[hi].key == v[lo].key) ++hi;
    if (hi - lo > 1) {
      std::sort(order.begin() + static_cast<std::ptrdiff_t>(lo), order.begin() + static_cast<std::ptrdiff_t>(hi), tie);
    }
    lo = hi;
  }
  return order;
}

// Reorders the positions stably by key_of(position).
template <class KeyOf>
void stable_by(std::vector<std::size_t>& order, KeyOf key_of) {
  std::vector<Keyed> v(order.size());
  for (std::size_t i = 0; i < order.size(); ++i) v[i] = {key_of(order[i]), order[i]};
  radix_sort(v);
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = v[i].at;
}

// The records in the order of their positions.
template <class R>
void permute(std::vector<R>& recs, const std::vector<std::size_t>& order) {
  std::vector<R> out(recs.size());
  for (std::size_t i = 0; i < order.size(); ++i) out[i] = recs[order[i]];
  recs.swap(out);
}

// The segments of n entities sorted by (ownership, topology, arity) already:
// what make_segments returns for them, without per-entity key arrays.
template <class KeyOf>
SegmentTable sorted_segments(std::size_t n, KeyOf key_of) {
  SegmentTable out;
  out.count = static_cast<LO>(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto [own, t, arity] = key_of(i);
    const auto at = static_cast<LO>(i);
    const Segment* last = out.segments.empty() ? nullptr : &out.segments.back();
    if (!last || last->ownership != own || last->topology != t || last->arity != arity) {
      out.segments.push_back(Segment{at, at, t, arity, own});
    }
    out.segments.back().end = static_cast<LO>(i + 1);
  }
  for (const Segment& s : out.segments) {
    if (s.ownership == Ownership::interior) out.interior_end = s.end;
    if (s.ownership <= Ownership::shared) out.owned_end = s.end;
    if (s.ownership <= Ownership::shared_copy) out.shared_end = s.end;
  }
  out.owned_end = std::max(out.owned_end, out.interior_end);
  out.shared_end = std::max(out.shared_end, out.owned_end);
  return out;
}

Keys make_keys(Ordering ordering, const std::vector<Real>& xyz, const std::optional<std::array<Real, 6>>& box) {
  Keys k;
  if (ordering != Ordering::type_major_sfc) return k;
  if (box) {
    k.box = sfc::KeyBox::make(box->data(), box->data() + 3);
    k.active = true;
    return k;
  }
  if (xyz.empty()) return k;
  Real lo[3] = {xyz[0], xyz[1], xyz[2]};
  Real hi[3] = {xyz[0], xyz[1], xyz[2]};
  for (std::size_t i = 0; i < xyz.size(); i += 3) {
    for (std::size_t d = 0; d < 3; ++d) {
      lo[d] = std::min(lo[d], xyz[i + d]);
      hi[d] = std::max(hi[d], xyz[i + d]);
    }
  }
  k.box = sfc::KeyBox::make(lo, hi);
  k.active = true;
  return k;
}

}  // namespace

HostMesh MeshBuilder::build(Ordering ordering) && {
  if (ordering != Ordering::file && ordering != Ordering::type_major_sfc) {
    throw std::invalid_argument("MeshBuilder::build: ordering is not implemented");
  }
  HostMesh mesh;
  mesh.dim_ = dim_;
  mesh.ordering_ = ordering;
  const Keys keys = make_keys(ordering, xyz_, key_box_);
  // Per dimension, the ownership named by gid, sorted by gid; (kind, group)
  // packed in one word, kind high, so a word sorts by kind then group.
  // An array over the gids' range when it is at most twice their count (as
  // file gids usually are; positions + 1, 0 for none), else sorted by gid.
  std::array<std::vector<Owned>, 4> owned = std::move(owned_);
  std::array<GO, 4> owned_lo{};
  std::array<std::vector<std::size_t>, 4> owned_at;
  for (std::size_t d = 0; d < 4; ++d) {
    auto& o = owned[d];
    const auto twice = [&](GO g) {
      throw std::invalid_argument("set_ownership: gid " + std::to_string(g) + " of dimension " + std::to_string(d) +
                                  " is named twice");
    };
    if (o.empty()) continue;
    GO lo = o[0].gid, hi = o[0].gid;
    for (const Owned& x : o) {
      lo = std::min(lo, x.gid);
      hi = std::max(hi, x.gid);
    }
    // Unsigned, since hi - lo overflows GO for gids spanning its range.
    const std::uint64_t span = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo);
    if (span < 2 * static_cast<std::uint64_t>(o.size())) {
      owned_lo[d] = lo;
      auto& at = owned_at[d];
      at.assign(static_cast<std::size_t>(span) + 1, 0);
      bool again = false;
      for (std::size_t i = 0; i < o.size(); ++i) {
        std::size_t& a = at[static_cast<std::size_t>(o[i].gid - lo)];
        again = again || a != 0;
        a = i + 1;
      }
      if (!again) continue;
      owned_at[d].clear();  // the least gid named twice is reported below
    }
    std::sort(o.begin(), o.end(), [](const Owned& a, const Owned& b) { return a.gid < b.gid; });
    const auto dup = std::adjacent_find(o.begin(), o.end(), [](const Owned& a, const Owned& b) { return a.gid == b.gid; });
    if (dup != o.end()) twice(dup->gid);
  }
  std::array<std::size_t, 4> named{};  // entities found among those named
  const auto own_of = [&](std::size_t d, GO g) -> std::uint64_t {
    const auto& o = owned[d];
    const Owned* x = nullptr;
    if (!owned_at[d].empty()) {
      const auto& at = owned_at[d];
      if (g >= owned_lo[d] && static_cast<std::uint64_t>(g) - static_cast<std::uint64_t>(owned_lo[d]) < at.size()) {
        const std::size_t a = at[static_cast<std::size_t>(g - owned_lo[d])];
        if (a != 0) x = &o[a - 1];
      }
    } else {
      const auto it = std::lower_bound(o.begin(), o.end(), g, [](const Owned& a, GO v) { return a.gid < v; });
      if (it != o.end() && it->gid == g) x = &*it;
    }
    if (x == nullptr) return 0;
    ++named[d];
    return std::uint64_t{static_cast<std::uint8_t>(x->kind)} << 32 | x->group;
  };
  const auto check_named = [&](std::size_t d) {
    if (named[d] != owned[d].size()) {
      throw std::invalid_argument("set_ownership: " + std::to_string(owned[d].size() - named[d]) +
                                  " gids of dimension " + std::to_string(d) + " name no entity");
    }
  };
  const auto kind_of = [](std::uint64_t own) { return static_cast<Ownership>(own >> 32); };

  // Vertices: (key, gid). With Ordering::file every key is 0, so by gid.
  const std::size_t nv = vertex_gids_.size();
  struct VertexRec {
    std::uint64_t key;
    GO gid;
    std::size_t src;
    std::uint64_t own;
  };
  std::vector<VertexRec> vrecs(nv);
  using HostRange = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>;
  Kokkos::parallel_for("soma::mesh_builder::vertex_keys", HostRange(0, nv),
                       [&](const std::size_t i) {
                         const Real* x = &xyz_[3 * i];
                         vrecs[i] = {keys(x[0], x[1], x[2]), vertex_gids_[i], i, 0};
                       });
  {
    // By (key, gid); with Ordering::file every key is 0, so by gid.
    std::vector<std::size_t> order = order_by_key(
        nv, [&](std::size_t i) { return vrecs[i].key; },
        [&](std::size_t a, std::size_t b) { return vrecs[a].gid < vrecs[b].gid; });
    // Then stably by group and by kind: by (kind, group, key, gid).
    if (!owned[0].empty()) {
      for (VertexRec& r : vrecs) r.own = own_of(0, r.gid);
      check_named(0);
      stable_by(order, [&](std::size_t i) { return vrecs[i].own & 0xFFFFFFFFULL; });
      stable_by(order, [&](std::size_t i) { return vrecs[i].own >> 32; });
    }
    permute(vrecs, order);
  }
  mesh.coords_ = CoordView<Kokkos::HostSpace>(
      Kokkos::view_alloc(std::string("soma::mesh::coords"), Kokkos::WithoutInitializing), nv);
  auto& v = mesh.ents_[0];
  v.count = detail::to_lo(nv, "number of vertices");
  v.input = true;
  v.gid = View1<GO, Kokkos::HostSpace>(
      Kokkos::view_alloc(std::string("soma::mesh::gid0"), Kokkos::WithoutInitializing), nv);
  std::vector<LO> rank(nv);  // local id of each input vertex
  for (std::size_t i = 0; i < nv; ++i) {
    const std::size_t src = vrecs[i].src;
    v.gid(i) = vrecs[i].gid;
    rank[src] = static_cast<LO>(i);
    for (std::size_t k = 0; k < 3; ++k) mesh.coords_(i, k) = xyz_[3 * src + k];
  }
  // Unique gids that are also contiguous (file ids are) give a vertex's
  // local id by indexing instead of a search. Input gids that increase by one
  // are that already: the local ids by gid are the ranks.
  std::vector<std::pair<GO, LO>> local_of;  // (gid, local id) by gid, unless dense
  std::vector<LO> dense_local;
  GO gmin = 0;
  GO gmax = -1;
  // Unsigned, since gmax - gmin overflows GO for gids spanning its range.
  const auto contiguous = [&] {
    return static_cast<std::uint64_t>(gmax) - static_cast<std::uint64_t>(gmin) == nv - 1;
  };
  const bool increasing =
      nv > 0 && std::adjacent_find(vertex_gids_.begin(), vertex_gids_.end(),
                                   std::greater_equal<>()) == vertex_gids_.end();
  if (increasing) {
    gmin = vertex_gids_.front();
    gmax = vertex_gids_.back();
  }
  if (increasing && contiguous()) {
    dense_local = std::move(rank);
  } else {
    local_of.resize(nv);
    for (std::size_t i = 0; i < nv; ++i) local_of[i] = {vertex_gids_[i], rank[i]};
    sort_unless_sorted(local_of.begin(), local_of.end(), std::less<>());
    const auto same_gid = [](const auto& x, const auto& y) { return x.first == y.first; };
    const auto dup = std::adjacent_find(local_of.begin(), local_of.end(), same_gid);
    if (dup != local_of.end()) {
      throw std::invalid_argument("duplicate vertex gid " + std::to_string(dup->first));
    }
    if (nv > 0) {
      gmin = local_of.front().first;
      gmax = local_of.back().first;
    }
    if (nv > 0 && contiguous()) {
      dense_local.resize(nv);
      for (const auto& [g, l] : local_of) dense_local[static_cast<std::size_t>(g - gmin)] = l;
    }
  }
  v.segments = sorted_segments(nv, [&](std::size_t i) {
    return std::tuple(kind_of(vrecs[i].own), Topology::vertex, std::uint16_t{1});
  });

  const auto find_vertex = [&](GO g) -> LO {
    if (!dense_local.empty()) {
      return g < gmin || g > gmax ? invalid_lo : dense_local[static_cast<std::size_t>(g - gmin)];
    }
    const auto it = std::lower_bound(local_of.begin(), local_of.end(), std::pair<GO, LO>{g, 0});
    return (it == local_of.end() || it->first != g) ? invalid_lo : it->second;
  };
  // Within each ownership kind, vertices that are nodes of high-order
  // elements and no element's corner move after the others, keeping their
  // order, in segments of their own; unknown gids are reported with their
  // element below.
  const std::array<LO, 5> bounds = kind_bounds(v.segments);
  for (std::size_t k = 0; k < 4; ++k) mesh.corner_ends_[k] = bounds[k + 1];
  {
    std::vector<std::uint8_t> role(nv, 0);  // 1 a corner, 2 a node
    bool any_nodes = false;
    for (std::size_t d = 1; d <= 3; ++d) {
      for (const Block& blk : blocks_[d]) {
        if (blk.nodes == 0) continue;
        any_nodes = true;
        for (const GO g : blk.node_gids) {
          if (const LO l = find_vertex(g); l != invalid_lo) role[static_cast<std::size_t>(l)] |= 2;
        }
      }
    }
    for (std::size_t d = 1; any_nodes && d <= 3; ++d) {
      for (const Block& blk : blocks_[d]) {
        for (const GO g : blk.vertex_gids) {
          if (const LO l = find_vertex(g); l != invalid_lo) role[static_cast<std::size_t>(l)] |= 1;
        }
      }
    }
    if (std::find(role.begin(), role.end(), std::uint8_t{2}) != role.end()) {
      std::vector<LO> to(nv);
      for (std::size_t k = 0; k < 4; ++k) {
        LO next = bounds[k];
        for (LO i = bounds[k]; i < bounds[k + 1]; ++i) {
          if (role[static_cast<std::size_t>(i)] != 2) to[static_cast<std::size_t>(i)] = next++;
        }
        mesh.corner_ends_[k] = next;
        for (LO i = bounds[k]; i < bounds[k + 1]; ++i) {
          if (role[static_cast<std::size_t>(i)] == 2) to[static_cast<std::size_t>(i)] = next++;
        }
      }
      split_segments(v.segments, mesh.corner_ends_);
      CoordView<Kokkos::HostSpace> xyz(
          Kokkos::view_alloc(std::string("soma::mesh::coords"), Kokkos::WithoutInitializing), nv);
      View1<GO, Kokkos::HostSpace> gid(
          Kokkos::view_alloc(std::string("soma::mesh::gid0"), Kokkos::WithoutInitializing), nv);
      for (std::size_t i = 0; i < nv; ++i) {
        const auto j = static_cast<std::size_t>(to[i]);
        gid(j) = v.gid(i);
        for (std::size_t k = 0; k < 3; ++k) xyz(j, k) = mesh.coords_(i, k);
      }
      mesh.coords_ = xyz;
      v.gid = gid;
      for (LO& l : dense_local) l = to[static_cast<std::size_t>(l)];
      for (auto& gl : local_of) gl.second = to[static_cast<std::size_t>(gl.second)];
    }
  }

  for (std::size_t d = 1; d <= 3; ++d) {
    struct Rec {
      Topology t;
      std::uint16_t arity;
      std::uint16_t nodes;
      GO gid;
      std::uint64_t key;
      std::size_t block;
      std::size_t index;
      std::uint64_t own;
    };
    std::vector<Rec> recs;
    {
      std::size_t n = 0;
      for (const Block& blk : blocks_[d]) n += blk.gids.size();
      recs.reserve(n);
    }
    std::vector<std::vector<LO>> local(blocks_[d].size());  // per connectivity entry
    std::vector<std::vector<LO>> node_local(blocks_[d].size());
    std::size_t entries = 0;
    std::size_t node_entries = 0;
    for (std::size_t b = 0; b < blocks_[d].size(); ++b) {
      const Block& blk = blocks_[d][b];
      local[b].resize(blk.vertex_gids.size());
      entries += blk.vertex_gids.size();
      for (std::size_t i = 0; i < blk.gids.size(); ++i) {
        const Offset first = blk.offsets[i];
        const Offset last = blk.offsets[i + 1];
        for (Offset k = first; k < last; ++k) {
          const GO g = blk.vertex_gids[static_cast<std::size_t>(k)];
          const LO l = find_vertex(g);
          if (l == invalid_lo) {
            throw std::invalid_argument("unknown vertex gid " + std::to_string(g) + " in " +
                                        name(blk.topology) + " element " +
                                        std::to_string(blk.gids[i]));
          }
          local[b][static_cast<std::size_t>(k)] = l;
        }
        const auto arity = static_cast<std::uint16_t>(last - first);
        recs.push_back(Rec{blk.topology, arity, blk.nodes, blk.gids[i], 0, b, i, 0});
      }
      node_local[b].resize(blk.node_gids.size());
      node_entries += blk.node_gids.size();
      for (std::size_t k = 0; k < blk.node_gids.size(); ++k) {
        const GO g = blk.node_gids[k];
        const LO l = find_vertex(g);
        if (l == invalid_lo) {
          const int count = num_vertices(blk.topology) + blk.nodes;
          throw std::invalid_argument("unknown vertex gid " + std::to_string(g) + " among the nodes of " +
                                      name(blk.topology) + std::to_string(count) + " element " +
                                      std::to_string(blk.gids[k / blk.nodes]));
        }
        node_local[b][k] = l;
      }
    }
    if (keys.active) {
      Kokkos::parallel_for(
          "soma::mesh_builder::element_keys", HostRange(0, recs.size()), [&](const std::size_t r) {
            Rec& rec = recs[r];
            const Block& blk = blocks_[d][rec.block];
            const std::vector<LO>& ids = local[rec.block];
            const Real inv = rec.arity > 0 ? Real{1} / static_cast<Real>(rec.arity) : Real{0};
            Real c[3] = {0, 0, 0};
            for (Offset k = blk.offsets[rec.index]; k < blk.offsets[rec.index + 1]; ++k) {
              const auto src = static_cast<std::size_t>(ids[static_cast<std::size_t>(k)]);
              // Summing x / arity, not x, keeps centroids near the largest Real finite.
              for (std::size_t q = 0; q < 3; ++q) c[q] += mesh.coords_(src, q) * inv;
            }
            rec.key = keys(c[0], c[1], c[2]);
          });
    }
    // A gid given twice is reported at the least such gid, by the first two
    // of its elements in (topology, arity) order.
    const auto repeated = [&](GO g) {
      std::vector<std::pair<Topology, std::uint16_t>> same;
      for (const Rec& r : recs) {
        if (r.gid == g) same.emplace_back(r.t, r.arity);
      }
      std::sort(same.begin(), same.end());
      if (same[0].first != same[1].first) {
        throw std::invalid_argument("element gid " + std::to_string(g) + " is used by both " + name(same[0].first) +
                                    " and " + name(same[1].first) + " elements of dimension " + std::to_string(d));
      }
      throw std::invalid_argument("duplicate element gid " + std::to_string(g));
    };
    if (!recs.empty()) {
      GO lo = recs[0].gid, hi = recs[0].gid;
      for (const Rec& r : recs) {
        lo = std::min(lo, r.gid);
        hi = std::max(hi, r.gid);
      }
      // Unsigned, since hi - lo overflows GO for gids spanning its range.
      const std::uint64_t span = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo);
      if (span < 2 * static_cast<std::uint64_t>(recs.size())) {
        std::vector<std::uint8_t> seen(static_cast<std::size_t>(span) + 1, 0);  // 2: twice or more
        bool twice = false;
        for (const Rec& r : recs) {
          std::uint8_t& c = seen[static_cast<std::size_t>(r.gid - lo)];
          c = c == 0 ? 1 : 2;
          twice = twice || c == 2;
        }
        for (std::size_t k = 0; twice && k < seen.size(); ++k) {
          if (seen[k] == 2) repeated(lo + static_cast<GO>(k));
        }
      } else {
        std::vector<GO> gids(recs.size());
        for (std::size_t i = 0; i < recs.size(); ++i) gids[i] = recs[i].gid;
        sort_unless_sorted(gids.begin(), gids.end(), std::less<>());
        const auto dup = std::adjacent_find(gids.begin(), gids.end());
        if (dup != gids.end()) repeated(*dup);
      }
    }
    // By (key, gid), then stably by group, by (topology, arity, nodes) and by
    // kind: by (kind, topology, arity, nodes, group, key, gid).
    std::vector<std::size_t> order = order_by_key(
        recs.size(), [&](std::size_t i) { return recs[i].key; },
        [&](std::size_t a, std::size_t b) { return recs[a].gid < recs[b].gid; });
    if (!owned[d].empty()) {
      for (Rec& r : recs) r.own = own_of(d, r.gid);
      check_named(d);
      stable_by(order, [&](std::size_t i) { return recs[i].own & 0xFFFFFFFFULL; });
    }
    {
      std::vector<std::tuple<Topology, std::uint16_t, std::uint16_t>> groups;
      for (const Rec& r : recs) {
        const auto g = std::tuple(r.t, r.arity, r.nodes);
        if (groups.empty() || groups.back() != g) groups.push_back(g);
      }
      std::sort(groups.begin(), groups.end());
      groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
      if (groups.size() > 1) {
        const auto group_of = [&](std::size_t i) {
          const Rec& r = recs[i];
          const auto it =
              std::lower_bound(groups.begin(), groups.end(), std::tuple(r.t, r.arity, r.nodes));
          return static_cast<std::uint64_t>(it - groups.begin());
        };
        stable_by(order, group_of);
      }
    }
    if (!owned[d].empty()) stable_by(order, [&](std::size_t i) { return recs[i].own >> 32; });
    permute(recs, order);
    auto& e = mesh.ents_[d];
    e.count = detail::to_lo(recs.size(), "number of elements");
    e.input = !recs.empty();
    (void)detail::to_offset(entries, "connectivity size");  // so every prefix fits too
    const auto uninit = [](std::string label) {
      return Kokkos::view_alloc(std::move(label), Kokkos::WithoutInitializing);
    };
    const std::size_t ne = recs.size();
    e.conn_offsets = View1<Offset, Kokkos::HostSpace>(uninit("soma::mesh::offsets"), ne + 1);
    e.topology = View1<std::uint8_t, Kokkos::HostSpace>(uninit("soma::mesh::topology"), ne);
    e.gid = View1<GO, Kokkos::HostSpace>(uninit("soma::mesh::gid"), recs.size());
    e.conn = View1<LO, Kokkos::HostSpace>(uninit("soma::mesh::conn"), entries);
    std::size_t written = 0;
    // Elements still in block order (file order usually is) copy each block's
    // connectivity whole.
    bool in_block_order = true;
    {
      std::size_t b = 0;
      std::size_t i = 0;
      for (const Rec& rec : recs) {
        while (i == blocks_[d][b].gids.size()) {
          ++b;
          i = 0;
        }
        if (rec.block != b || rec.index != i) {
          in_block_order = false;
          break;
        }
        ++i;
      }
    }
    if (in_block_order) {
      for (const std::vector<LO>& ids : local) {
        std::copy(ids.begin(), ids.end(), e.conn.data() + written);
        written += ids.size();
      }
      written = 0;
    }
    e.conn_offsets(0) = 0;
    for (std::size_t r = 0; r < recs.size(); ++r) {
      const Block& blk = blocks_[d][recs[r].block];
      const Offset first = blk.offsets[recs[r].index];
      const Offset last = blk.offsets[recs[r].index + 1];
      if (in_block_order) {
        written += static_cast<std::size_t>(last - first);
      } else {
        const std::vector<LO>& ids = local[recs[r].block];
        for (Offset k = first; k < last; ++k) e.conn(written++) = ids[static_cast<std::size_t>(k)];
      }
      e.conn_offsets(r + 1) = static_cast<Offset>(written);
      e.topology(r) = static_cast<std::uint8_t>(recs[r].t);
      e.gid(r) = recs[r].gid;
    }
    if (node_entries > 0) {
      (void)detail::to_offset(node_entries, "number of high-order nodes");
      e.node_offsets = View1<Offset, Kokkos::HostSpace>(uninit("soma::mesh::node_offsets"), ne + 1);
      e.nodes = View1<LO, Kokkos::HostSpace>(uninit("soma::mesh::nodes"), node_entries);
      std::size_t at = 0;
      e.node_offsets(0) = 0;
      for (std::size_t r = 0; r < ne; ++r) {
        const std::size_t n = recs[r].nodes;
        const LO* from = node_local[recs[r].block].data() + recs[r].index * n;
        std::copy(from, from + n, e.nodes.data() + at);
        at += n;
        e.node_offsets(r + 1) = static_cast<Offset>(at);
      }
    }
    e.segments = sorted_segments(ne, [&](std::size_t r) { return std::tuple(kind_of(recs[r].own), recs[r].t, recs[r].arity); });
  }
  mesh.epoch_ = 1;
  mesh.attach_fields();
  return mesh;
}

}  // namespace soma
