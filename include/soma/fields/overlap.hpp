// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_OVERLAP_HPP_
#define SOMA_FIELDS_OVERLAP_HPP_

#include <utility>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/fields/domain.hpp"
#include "soma/core/search.hpp"

namespace soma::detail {

using HostBlock = DomainBlockView<Kokkos::HostSpace>;

// Smallest local id that two blocks of one dimension share, or invalid_lo.
//   full vs anything: O(1)
//   intervals vs intervals: merge-scan, O(k1 + k2)
//   intervals vs list: one binary search per interval, O(k log n)
//   list vs list: merge-scan, O(n1 + n2)
inline LO first_overlap(const HostBlock& x, const HostBlock& y) {
  if (x.size == 0 || y.size == 0) return invalid_lo;
  if (x.kind == DomainKind::full) return y.entity(0);
  if (y.kind == DomainKind::full) return x.entity(0);

  if (x.kind == DomainKind::intervals && y.kind == DomainKind::intervals) {
    LO i = 0;
    LO j = 0;
    while (i < x.runs && j < y.runs) {
      const LO xb = x.a(i);
      const LO xe = xb + (x.b(i + 1) - x.b(i));
      const LO yb = y.a(j);
      const LO ye = yb + (y.b(j + 1) - y.b(j));
      const LO lo = xb > yb ? xb : yb;
      const LO hi = xe < ye ? xe : ye;
      if (lo < hi) return lo;
      if (xe < ye) {
        ++i;
      } else {
        ++j;
      }
    }
    return invalid_lo;
  }

  if (x.kind == DomainKind::list && y.kind == DomainKind::list) {
    LO i = 0;
    LO j = 0;
    while (i < x.size && j < y.size) {
      const LO u = x.a(i);
      const LO v = y.a(j);
      if (u == v) return u;
      if (u < v) {
        ++i;
      } else {
        ++j;
      }
    }
    return invalid_lo;
  }

  const HostBlock& runs = x.kind == DomainKind::intervals ? x : y;
  const HostBlock& list = x.kind == DomainKind::intervals ? y : x;
  for (LO i = 0; i < runs.runs; ++i) {
    const LO begin = runs.a(i);
    const LO end = begin + (runs.b(i + 1) - runs.b(i));
    const LO k = lower_bound(list.a, list.size, begin);
    if (k < list.size && list.a(k) < end) return list.a(k);
  }
  return invalid_lo;
}

}  // namespace soma::detail

#endif  // SOMA_FIELDS_OVERLAP_HPP_
