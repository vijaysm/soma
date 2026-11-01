// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TOPOLOGY_TRANSPOSE_HPP_
#define SOMA_TOPOLOGY_TRANSPOSE_HPP_

// Upward graphs as the transpose of a downward CSR: count with atomics, scan,
// fill with atomic cursors, then sort each row. The fill order depends on the
// threads; the sort by (source, local index) makes the result independent of it.

#include <cstdint>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/topology/adjacency.hpp"

namespace soma::topology {

namespace detail {

// Insertion sort of [b, e) by (target, code); rows are short.
template <class Targets, class Codes>
KOKKOS_INLINE_FUNCTION void sort_row(const Targets& t, const Codes& c, Offset b, Offset e) {
  for (Offset i = b + 1; i < e; ++i) {
    const LO ti = t(i);
    const std::uint8_t ci = c(i);
    Offset j = i;
    while (j > b && (t(j - 1) > ti || (t(j - 1) == ti && c(j - 1) > ci))) {
      t(j) = t(j - 1);
      c(j) = c(j - 1);
      --j;
    }
    t(j) = ti;
    c(j) = ci;
  }
}

// Exclusive scan of counts(0 .. n-1) into offsets(0 .. n), returning the total.
template <class ExecSpace, class Counts, class Offsets>
Offset exclusive_scan(const ExecSpace& exec, const std::string& label, LO n, const Counts& counts,
                      const Offsets& offsets) {
  Offset total = 0;
  Kokkos::parallel_scan(
      label, Kokkos::RangePolicy<ExecSpace>(exec, 0, n + 1),
      KOKKOS_LAMBDA(const LO i, Offset& acc, const bool final) {
        if (final) offsets(i) = acc;
        if (i < n) acc += static_cast<Offset>(counts(i));
      },
      total);
  exec.fence(label);
  return total;
}

}  // namespace detail

// Transpose of a graph from rows = offsets.extent(0) - 1 sources to ntargets
// targets. Row t of the result lists the sources s that list t, sorted by
// (s, k), and codes holds k, the position of t in row s. Throws
// std::length_error if a source row is longer than 256 (k must fit the code).
template <class ExecSpace>
Adjacency<typename ExecSpace::memory_space> transpose(
    const ExecSpace& exec, const View1<const Offset, typename ExecSpace::memory_space>& offsets,
    const View1<const LO, typename ExecSpace::memory_space>& targets, LO ntargets,
    const std::string& label) {
  using M = typename ExecSpace::memory_space;
  const LO nrows = offsets.extent(0) == 0 ? 0 : static_cast<LO>(offsets.extent(0) - 1);
  const std::string base = "soma::topology::transpose::" + label;

  Offset longest = 0;
  Kokkos::parallel_reduce(
      base + "::longest", Kokkos::RangePolicy<ExecSpace>(exec, 0, nrows),
      KOKKOS_LAMBDA(const LO s, Offset& m) {
        const Offset n = offsets(s + 1) - offsets(s);
        if (n > m) m = n;
      },
      Kokkos::Max<Offset>(longest));
  if (longest > 256) {
    throw std::length_error("transpose " + label + ": a row has " + std::to_string(longest) +
                            " entries; local indices above 255 do not fit the codes");
  }

  View1<Offset, M> counts(base + "::counts", static_cast<std::size_t>(ntargets));
  Kokkos::parallel_for(
      base + "::count", Kokkos::RangePolicy<ExecSpace>(exec, 0, nrows), KOKKOS_LAMBDA(const LO s) {
        for (Offset j = offsets(s); j < offsets(s + 1); ++j) Kokkos::atomic_inc(&counts(targets(j)));
      });

  Adjacency<M> out;
  out.offsets = View1<Offset, M>(base + "::offsets", static_cast<std::size_t>(ntargets) + 1);
  const Offset nnz = detail::exclusive_scan(exec, base + "::scan", ntargets, counts, out.offsets);
  out.targets =
      View1<LO, M>(Kokkos::view_alloc(exec, base + "::targets", Kokkos::WithoutInitializing),
                   static_cast<std::size_t>(nnz));
  out.codes = View1<std::uint8_t, M>(
      Kokkos::view_alloc(exec, base + "::codes", Kokkos::WithoutInitializing),
      static_cast<std::size_t>(nnz));

  const auto o = out.offsets;
  const auto t = out.targets;
  const auto c = out.codes;
  Kokkos::deep_copy(exec, counts, Offset{0});
  Kokkos::parallel_for(
      base + "::fill", Kokkos::RangePolicy<ExecSpace>(exec, 0, nrows), KOKKOS_LAMBDA(const LO s) {
        for (Offset j = offsets(s); j < offsets(s + 1); ++j) {
          const LO dst = targets(j);
          const Offset at = o(dst) + Kokkos::atomic_fetch_add(&counts(dst), Offset{1});
          t(at) = s;
          c(at) = static_cast<std::uint8_t>(j - offsets(s));
        }
      });
  Kokkos::parallel_for(
      base + "::sort", Kokkos::RangePolicy<ExecSpace>(exec, 0, ntargets),
      KOKKOS_LAMBDA(const LO r) { detail::sort_row(t, c, o(r), o(r + 1)); });
  exec.fence(base);
  return out;
}

}  // namespace soma::topology

#endif  // SOMA_TOPOLOGY_TRANSPOSE_HPP_
