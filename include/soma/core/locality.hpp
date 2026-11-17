// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_LOCALITY_HPP_
#define SOMA_CORE_LOCALITY_HPP_

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"

namespace soma {

/// How far apart in memory the vertices of one element are: max - min of its
/// vertex local ids. Small spans mean an element gathers coordinates from a
/// narrow window. An element without vertices has span 0 and counts towards
/// the mean. Metrics over element-to-element neighbours need adjacency and are
/// not provided yet.
struct VertexSpanReport {
  LO elements = 0;
  double mean = 0;
  LO max = 0;
};

namespace detail {

/// The report over `count` elements of a host CSR block.
template <class OffsetView, class ConnView>
VertexSpanReport vertex_span(LO count, const OffsetView& offsets, const ConnView& conn) {
  VertexSpanReport r;
  r.elements = count;
  if (count == 0) return r;
  double total = 0;
  for (LO e = 0; e < count; ++e) {
    if (offsets(e + 1) == offsets(e)) continue;
    LO lo = conn(offsets(e));
    LO hi = lo;
    for (auto k = offsets(e); k < offsets(e + 1); ++k) {
      lo = std::min(lo, conn(k));
      hi = std::max(hi, conn(k));
    }
    total += static_cast<double>(hi - lo);
    r.max = std::max(r.max, hi - lo);
  }
  r.mean = total / static_cast<double>(count);
  return r;
}

}  // namespace detail

/// Host only; the mesh must be host accessible. Throws std::out_of_range for a
/// dimension outside [1, 3].
template <class MemSpace>
VertexSpanReport vertex_span_report(const Mesh<MemSpace>& mesh, int dim) {
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible,
                "vertex_span_report reads on the host; mirror the mesh first");
  if (dim < 1 || dim > 3) {
    throw std::out_of_range("vertex_span_report: dimension " + std::to_string(dim) +
                            " is not in [1, 3]");
  }
  return detail::vertex_span(mesh.count(dim), mesh.conn_offsets(dim), mesh.conn(dim));
}

}  // namespace soma

#endif  // SOMA_CORE_LOCALITY_HPP_
