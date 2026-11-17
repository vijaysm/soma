// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TOPOLOGY_BRIDGE_HPP_
#define SOMA_TOPOLOGY_BRIDGE_HPP_

/// \file
/// Bridge adjacency: x -> the entities z != x of the same dimension that share
/// an entity of the bridge dimension, z in second(y) for y in first(x). Two
/// passes: gather every candidate into a buffer sized by an exact upper bound,
/// sort and deduplicate each row in place, then compact by a scan.

#include <cstdint>
#include <string>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/topology/adjacency.hpp"
#include "soma/topology/transpose.hpp"

namespace soma::topology {

namespace detail {

template <class Targets>
KOKKOS_INLINE_FUNCTION Offset sort_unique(const Targets& t, Offset b, Offset e) {
  for (Offset i = b + 1; i < e; ++i) {
    const LO ti = t(i);
    Offset j = i;
    while (j > b && t(j - 1) > ti) {
      t(j) = t(j - 1);
      --j;
    }
    t(j) = ti;
  }
  Offset w = b;
  for (Offset i = b; i < e; ++i) {
    if (w == b || t(w - 1) != t(i)) t(w++) = t(i);
  }
  return w - b;
}

}  // namespace detail

/// first: n rows (dimension dim -> via); second: via -> dim. codes is empty.
template <class ExecSpace>
Adjacency<typename ExecSpace::memory_space> bridge(
    const ExecSpace& exec, LO n,
    const View1<const Offset, typename ExecSpace::memory_space>& first_off,
    const View1<const LO, typename ExecSpace::memory_space>& first_tgt,
    const View1<const Offset, typename ExecSpace::memory_space>& second_off,
    const View1<const LO, typename ExecSpace::memory_space>& second_tgt, const std::string& label) {
  using M = typename ExecSpace::memory_space;
  using Policy = Kokkos::RangePolicy<ExecSpace>;
  const std::string base = "soma::topology::bridge::" + label;

  View1<Offset, M> bound(base + "::bound", static_cast<std::size_t>(n));
  Kokkos::parallel_for(
      base + "::count", Policy(exec, 0, n), KOKKOS_LAMBDA(const LO x) {
        Offset c = 0;
        for (Offset j = first_off(x); j < first_off(x + 1); ++j) {
          const LO y = first_tgt(j);
          c += second_off(y + 1) - second_off(y);
        }
        bound(x) = c;
      });
  View1<Offset, M> buf_off(base + "::buffer_offsets", static_cast<std::size_t>(n) + 1);
  const Offset nbuf = detail::exclusive_scan(exec, base + "::scan_bound", n, bound, buf_off);
  View1<LO, M> buf(Kokkos::view_alloc(exec, base + "::buffer", Kokkos::WithoutInitializing),
                   static_cast<std::size_t>(nbuf));
  View1<Offset, M> unique(base + "::unique", static_cast<std::size_t>(n));
  Kokkos::parallel_for(
      base + "::gather", Policy(exec, 0, n), KOKKOS_LAMBDA(const LO x) {
        Offset w = buf_off(x);
        for (Offset j = first_off(x); j < first_off(x + 1); ++j) {
          const LO y = first_tgt(j);
          for (Offset q = second_off(y); q < second_off(y + 1); ++q) {
            if (second_tgt(q) != x) buf(w++) = second_tgt(q);
          }
        }
        unique(x) = detail::sort_unique(buf, buf_off(x), w);
      });

  Adjacency<M> out;
  out.offsets = View1<Offset, M>(base + "::offsets", static_cast<std::size_t>(n) + 1);
  const Offset nnz = detail::exclusive_scan(exec, base + "::scan", n, unique, out.offsets);
  out.targets = View1<LO, M>(Kokkos::view_alloc(exec, base + "::targets", Kokkos::WithoutInitializing),
                             static_cast<std::size_t>(nnz));
  const auto o = out.offsets;
  const auto t = out.targets;
  Kokkos::parallel_for(
      base + "::compact", Policy(exec, 0, n), KOKKOS_LAMBDA(const LO x) {
        for (Offset i = 0; i < o(x + 1) - o(x); ++i) t(o(x) + i) = buf(buf_off(x) + i);
      });
  exec.fence(base);
  return out;
}

}  // namespace soma::topology

#endif  // SOMA_TOPOLOGY_BRIDGE_HPP_
