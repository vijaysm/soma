// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TOPOLOGY_DERIVE_HPP_
#define SOMA_TOPOLOGY_DERIVE_HPP_

// Sub-entities of dimension k (edges; faces in 3D) from the elements of the
// mesh dimension D, by the lowest-creator rule: side s of element e has the
// vertex set S; among the elements around min(S) whose sides include S, the
// one with the smallest (element, side) creates the sub-entity. No hash table
// and no global sort; each side costs a scan of one vertex's elements.
//
// Numbering: by (topology, arity), then by (creator element, side), so it
// follows the element order and every segment is contiguous. Connectivity is
// the creator's side in the creator's canonical order.
//
// Explicit sub-entities already in the mesh (dimension k blocks from the
// input) are adopted: the derived entity with the same vertex set keeps the
// explicit gid and the explicit vertex order. New entities are numbered from
// the largest explicit gid + 1. An explicit entity that is no side of any
// element, or two explicit entities with one vertex set, throw
// std::invalid_argument naming the gid.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/mesh_view.hpp"
#include "soma/core/segments.hpp"
#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/topology/adjacency.hpp"
#include "soma/topology/orientation.hpp"
#include "soma/topology/sides.hpp"
#include "soma/topology/transpose.hpp"

namespace soma::topology {

template <class MemSpace>
struct Derived {
  EntityBlock<MemSpace> block;  // the dimension-k entities, derived = true
  Adjacency<MemSpace> down;     // D -> k, rows in side order, orientation codes
};

namespace detail {

// The lowest (element, side) around min(S) whose side has the vertex set S, as
// a slot side_off(c) + t; -1 if none.
template <class M, class SideOff>
KOKKOS_INLINE_FUNCTION Offset lowest_side(const MeshView<M>& m, int d, int k, const Side& s,
                                          const SideOff& side_off, const UView1<Offset, M>& up_off,
                                          const UView1<LO, M>& up_targets) {
  const LO v = min_vertex(s);
  for (Offset j = up_off(v); j < up_off(v + 1); ++j) {
    const LO c = up_targets(j);
    const int nc = side_count(m, d, c, k);
    for (int t = 0; t < nc; ++t) {
      const Side o = side_of(m, d, c, k, t);
      if (o.n == s.n && same_set(o.v, s.v, s.n)) return side_off(c) + t;
    }
  }
  return -1;
}

}  // namespace detail

// vertex_to_elements is the 0 -> D graph with sorted rows (transpose).
template <class ExecSpace>
Derived<typename ExecSpace::memory_space> derive(
    const ExecSpace& exec, const Mesh<typename ExecSpace::memory_space>& mesh, int k,
    const Adjacency<typename ExecSpace::memory_space>& vertex_to_elements) {
  using M = typename ExecSpace::memory_space;
  using Policy = Kokkos::RangePolicy<ExecSpace>;
  const int D = mesh.dim();
  if (k < 1 || k >= D) {
    throw std::invalid_argument("derive: dimension " + std::to_string(k) + " is not in [1, " +
                                std::to_string(D - 1) + "]");
  }
  const std::string base = "soma::topology::derive" + std::to_string(k);
  const MeshView<M> m = mesh.view();
  const LO ne = mesh.count(D);

  Derived<M> out;
  const UView1<Offset, M> up_off = vertex_to_elements.offsets;
  const UView1<LO, M> up_tgt = vertex_to_elements.targets;

  // Sides per element, and their slots.
  View1<Offset, M> nsides(base + "::nsides", static_cast<std::size_t>(ne));
  Kokkos::parallel_for(
      base + "::count_sides", Policy(exec, 0, ne),
      KOKKOS_LAMBDA(const LO e) { nsides(e) = side_count(m, D, e, k); });
  View1<Offset, M> side_off(base + "::side_offsets", static_cast<std::size_t>(ne) + 1);
  const Offset nslots = soma::topology::detail::exclusive_scan(exec, base + "::scan_sides", ne,
                                                               nsides, side_off);

  // The creator slot of every slot.
  View1<Offset, M> creator(Kokkos::view_alloc(exec, base + "::creator", Kokkos::WithoutInitializing),
                           static_cast<std::size_t>(nslots));
  Kokkos::parallel_for(
      base + "::find_creators", Policy(exec, 0, ne), KOKKOS_LAMBDA(const LO e) {
        const int ns = side_count(m, D, e, k);
        for (int s = 0; s < ns; ++s) {
          const Side side = side_of(m, D, e, k, s);
          creator(side_off(e) + s) = detail::lowest_side(m, D, k, side, side_off, up_off, up_tgt);
        }
      });

  // Creators per element and kind (kind 1 = quads, k = 2 only).
  View1<Offset, M> count0(base + "::count0", static_cast<std::size_t>(ne));
  View1<Offset, M> count1(base + "::count1", static_cast<std::size_t>(ne));
  Kokkos::parallel_for(
      base + "::count_creators", Policy(exec, 0, ne), KOKKOS_LAMBDA(const LO e) {
        Offset c0 = 0;
        Offset c1 = 0;
        const int ns = side_count(m, D, e, k);
        for (int s = 0; s < ns; ++s) {
          const Offset i = side_off(e) + s;
          if (creator(i) != i) continue;
          const Side side = side_of(m, D, e, k, s);
          if (k == 2 && side.n == 4) {
            ++c1;
          } else {
            ++c0;
          }
        }
        count0(e) = c0;
        count1(e) = c1;
      });
  View1<Offset, M> scan0(base + "::scan0", static_cast<std::size_t>(ne) + 1);
  View1<Offset, M> scan1(base + "::scan1", static_cast<std::size_t>(ne) + 1);
  const Offset n0 = soma::topology::detail::exclusive_scan(exec, base + "::scan0", ne, count0, scan0);
  const Offset n1 = soma::topology::detail::exclusive_scan(exec, base + "::scan1", ne, count1, scan1);
  const Offset total = n0 + n1;
  if (static_cast<std::int64_t>(total) > std::int64_t{Kokkos::Experimental::finite_max_v<LO>}) {
    throw std::length_error(base + ": " + std::to_string(total) + " entities do not fit LO");
  }
  const LO n = static_cast<LO>(total);
  const int a0 = k == 1 ? 2 : 3;  // arity of kind 0
  const int a1 = 4;
  const Topology t0 = k == 1 ? Topology::edge : Topology::tri;

  EntityBlock<M>& b = out.block;
  b.count = n;
  b.derived = true;
  b.conn_offsets = View1<Offset, M>(base + "::offsets", static_cast<std::size_t>(n) + 1);
  b.topology = View1<std::uint8_t, M>(base + "::topology", static_cast<std::size_t>(n));
  b.gid = View1<GO, M>(base + "::gid", static_cast<std::size_t>(n));
  const std::int64_t conn_entries = std::int64_t{n0} * a0 + std::int64_t{n1} * a1;
  if (conn_entries > std::int64_t{Kokkos::Experimental::finite_max_v<Offset>}) {
    throw std::length_error(base + ": " + std::to_string(conn_entries) +
                            " connectivity entries do not fit Offset");
  }
  const auto nconn = static_cast<Offset>(conn_entries);
  b.conn = View1<LO, M>(base + "::conn", static_cast<std::size_t>(nconn));
  out.down.offsets = side_off;
  out.down.targets = View1<LO, M>(base + "::down_targets", static_cast<std::size_t>(nslots));
  out.down.codes = View1<std::uint8_t, M>(base + "::down_codes", static_cast<std::size_t>(nslots));

  const auto offs = b.conn_offsets;
  const auto conn = b.conn;
  const auto topo = b.topology;
  const auto gid = b.gid;
  const auto down = out.down.targets;
  const auto codes = out.down.codes;
  Kokkos::parallel_for(
      base + "::offsets", Policy(exec, 0, n + 1), KOKKOS_LAMBDA(const LO id) {
        const auto i = static_cast<Offset>(id);  // fits: the connectivity size was checked
        offs(id) = i <= n0 ? i * a0 : n0 * a0 + (i - n0) * a1;
      });
  Kokkos::parallel_for(
      base + "::create", Policy(exec, 0, ne), KOKKOS_LAMBDA(const LO e) {
        Offset r0 = 0;
        Offset r1 = 0;
        const int ns = side_count(m, D, e, k);
        for (int s = 0; s < ns; ++s) {
          const Offset i = side_off(e) + s;
          if (creator(i) != i) continue;
          const Side side = side_of(m, D, e, k, s);
          const bool quad = k == 2 && side.n == 4;
          const auto id = static_cast<LO>(quad ? n0 + scan1(e) + r1++ : scan0(e) + r0++);
          down(i) = id;
          topo(id) = static_cast<std::uint8_t>(quad ? Topology::quad : t0);
          for (int q = 0; q < side.n; ++q) conn(offs(id) + q) = side.v[q];
        }
      });

  // Adoption of the explicit entities of dimension k.
  constexpr LO none = Kokkos::Experimental::finite_max_v<LO>;
  View1<LO, M> adopted(Kokkos::view_alloc(exec, base + "::adopted"), static_cast<std::size_t>(n));
  Kokkos::deep_copy(exec, adopted, none);
  const LO nx = mesh.count(k);
  GO max_gid = 0;
  if (nx > 0) {
    Kokkos::View<LO, M> orphan(base + "::orphan");
    Kokkos::View<LO, M> twin(base + "::twin");
    Kokkos::deep_copy(exec, orphan, none);
    Kokkos::deep_copy(exec, twin, none);
    Kokkos::parallel_for(
        base + "::adopt", Policy(exec, 0, nx), KOKKOS_LAMBDA(const LO x) {
          const Side side = entity_as_side(m, k, x);
          const Offset slot =
              side.n >= 2 && side.n <= 4
                  ? detail::lowest_side(m, D, k, side, side_off, up_off, up_tgt)
                  : Offset{-1};
          if (slot < 0) {
            Kokkos::atomic_min(&orphan(), x);
            return;
          }
          Kokkos::atomic_min(&adopted(down(slot)), x);
        });
    // The lowest explicit entity of a vertex set is adopted; any other one is
    // a repeat, and the lowest repeat is reported, whatever the thread order.
    Kokkos::parallel_for(
        base + "::repeats", Policy(exec, 0, nx), KOKKOS_LAMBDA(const LO x) {
          const Side side = entity_as_side(m, k, x);
          const Offset slot = side.n >= 2 && side.n <= 4
                                  ? detail::lowest_side(m, D, k, side, side_off, up_off, up_tgt)
                                  : Offset{-1};
          if (slot >= 0 && adopted(down(slot)) != x) Kokkos::atomic_min(&twin(), x);
        });
    LO bad[2] = {0, 0};
    {
      auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, orphan);
      auto ht = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, twin);
      bad[0] = ho();
      bad[1] = ht();
    }
    auto gid_of = [&](LO x) {
      auto h = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, Kokkos::subview(mesh.gid(k), std::pair<LO, LO>(x, x + 1)));
      return std::to_string(h(0));
    };
    if (bad[0] != none) {
      throw std::invalid_argument("derive: explicit entity gid " + gid_of(bad[0]) +
                                  " of dimension " + std::to_string(k) +
                                  " is no side of any element");
    }
    if (bad[1] != none) {
      throw std::invalid_argument("derive: explicit entity gid " + gid_of(bad[1]) +
                                  " of dimension " + std::to_string(k) +
                                  " repeats the vertices of another");
    }
    const auto xgid = mesh.gid(k);
    Kokkos::parallel_reduce(
        base + "::max_gid", Policy(exec, 0, nx),
        KOKKOS_LAMBDA(const LO x, GO& mx) { mx = xgid(x) > mx ? xgid(x) : mx; },
        Kokkos::Max<GO>(max_gid));
    if (max_gid < 0) max_gid = 0;
  }

  // Gids: explicit ones kept, new ones from max_gid + 1 in id order. Adopted
  // entities take the explicit vertex order.
  View1<Offset, M> fresh(base + "::fresh", static_cast<std::size_t>(n));
  Kokkos::parallel_for(
      base + "::mark_new", Policy(exec, 0, n),
      KOKKOS_LAMBDA(const LO id) { fresh(id) = adopted(id) == none ? 1 : 0; });
  View1<Offset, M> fresh_rank(base + "::fresh_rank", static_cast<std::size_t>(n) + 1);
  (void)soma::topology::detail::exclusive_scan(exec, base + "::scan_new", n, fresh, fresh_rank);
  const auto xoff = mesh.conn_offsets(k);
  const auto xconn = mesh.conn(k);
  const auto xgid = mesh.gid(k);
  Kokkos::parallel_for(
      base + "::gids", Policy(exec, 0, n), KOKKOS_LAMBDA(const LO id) {
        const LO x = adopted(id);
        if (x == none) {
          gid(id) = max_gid + 1 + static_cast<GO>(fresh_rank(id));
          return;
        }
        gid(id) = xgid(x);
        for (Offset q = 0; q < offs(id + 1) - offs(id); ++q) conn(offs(id) + q) = xconn(xoff(x) + q);
      });

  // Every slot: its entity and its orientation against the stored vertices.
  Kokkos::parallel_for(
      base + "::orient", Policy(exec, 0, ne), KOKKOS_LAMBDA(const LO e) {
        const int ns = side_count(m, D, e, k);
        for (int s = 0; s < ns; ++s) {
          const Offset i = side_off(e) + s;
          const LO id = down(creator(i));
          const Side side = side_of(m, D, e, k, s);
          const Match mt = connectivity_match(side.v, &conn(offs(id)), side.n);
          codes(i) = orientation_code(mt.offset, mt.reversed);
          if (creator(i) != i) down(i) = id;
        }
      });
  exec.fence(base);

  // Segments: one per kind present.
  SegmentTable& st = b.segments;
  if (n0 > 0) st.segments.push_back(Segment{0, static_cast<LO>(n0), t0, static_cast<std::uint16_t>(a0), Ownership::interior});
  if (n1 > 0) st.segments.push_back(Segment{static_cast<LO>(n0), n, Topology::quad, 4, Ownership::interior});
  st.count = n;
  st.interior_end = n;
  st.owned_end = n;
  return out;
}

}  // namespace soma::topology

#endif  // SOMA_TOPOLOGY_DERIVE_HPP_
