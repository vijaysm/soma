// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_LOCALITY_HPP_
#define SOMA_CORE_LOCALITY_HPP_

#include <algorithm>
#include <cstddef>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"

namespace soma {

// How far apart in memory the vertices of one element are: max - min of its
// vertex local ids. Small spans mean an element gathers coordinates from a
// narrow window. Metrics over element-to-element neighbours need adjacency and
// are not provided yet.
struct VertexSpanReport {
  LO elements = 0;
  double mean = 0;
  LO max = 0;
};

// Host only; the mesh must be host accessible.
template <class MemSpace>
VertexSpanReport vertex_span_report(const Mesh<MemSpace>& mesh, int dim) {
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible,
                "vertex_span_report reads on the host; mirror the mesh first");
  VertexSpanReport r;
  r.elements = mesh.count(dim);
  if (dim < 1 || r.elements == 0) return r;
  const auto offsets = mesh.conn_offsets(dim);
  const auto conn = mesh.conn(dim);
  double total = 0;
  for (LO e = 0; e < r.elements; ++e) {
    LO lo = conn(offsets(e));
    LO hi = lo;
    for (Offset k = offsets(e); k < offsets(e + 1); ++k) {
      lo = std::min(lo, conn(k));
      hi = std::max(hi, conn(k));
    }
    total += static_cast<double>(hi - lo);
    r.max = std::max(r.max, hi - lo);
  }
  r.mean = total / static_cast<double>(r.elements);
  return r;
}

}  // namespace soma

#endif  // SOMA_CORE_LOCALITY_HPP_
