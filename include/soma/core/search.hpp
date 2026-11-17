// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_SEARCH_HPP_
#define SOMA_CORE_SEARCH_HPP_

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"

namespace soma::detail {

/// First index i in [0, n) with v(i) >= key, or n.
template <class V>
KOKKOS_INLINE_FUNCTION LO lower_bound(const V& v, LO n, LO key) {
  LO lo = 0;
  LO hi = n;
  while (lo < hi) {
    const LO mid = lo + (hi - lo) / 2;
    if (v(mid) < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/// First index i in [0, n) with v(i) > key, or n.
template <class V>
KOKKOS_INLINE_FUNCTION LO upper_bound(const V& v, LO n, LO key) {
  LO lo = 0;
  LO hi = n;
  while (lo < hi) {
    const LO mid = lo + (hi - lo) / 2;
    if (v(mid) <= key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

}  // namespace soma::detail

#endif  // SOMA_CORE_SEARCH_HPP_
