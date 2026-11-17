// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TOPOLOGY_ORIENTATION_HPP_
#define SOMA_TOPOLOGY_ORIENTATION_HPP_

/// \file
/// Relative orientation of two vertex lists naming the same entity, with the
/// semantics of MOAB's CN::ConnectivityMatch(a, b): offset is the position of
/// b[0] in a, and reversed says that b runs through a backwards. For two
/// vertices, b is reversed whenever it is not identical to a.

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"

namespace soma::topology {

struct Match {
  bool same = false;  ///< the lists name the same cycle
  int offset = 0;
  bool reversed = false;
};

KOKKOS_INLINE_FUNCTION Match connectivity_match(const LO* a, const LO* b, int n) {
  Match m;
  int at = -1;
  for (int i = 0; i < n; ++i) {
    if (a[i] == b[0]) {
      at = i;
      break;
    }
  }
  if (at < 0) return m;
  bool forward = true;
  bool backward = true;
  for (int i = 1; i < n; ++i) {
    if (a[(at + i) % n] != b[i]) forward = false;
    if (a[(at - i + n) % n] != b[i]) backward = false;
  }
  // Both directions agree for two vertices; the position decides.
  if (n == 2) {
    forward = forward && at == 0;
    backward = backward && at == 1;
  }
  m.same = forward || backward;
  m.offset = at;
  m.reversed = !forward;
  return m;
}

KOKKOS_INLINE_FUNCTION constexpr std::uint8_t orientation_code(int offset, bool reversed) {
  return static_cast<std::uint8_t>((offset << 1) | (reversed ? 1 : 0));
}
KOKKOS_INLINE_FUNCTION constexpr int code_offset(std::uint8_t code) { return code >> 1; }
KOKKOS_INLINE_FUNCTION constexpr bool code_reversed(std::uint8_t code) { return (code & 1) != 0; }
/// +1 when the parent sees the sub-entity as stored, -1 when reversed.
KOKKOS_INLINE_FUNCTION constexpr int code_sense(std::uint8_t code) {
  return code_reversed(code) ? -1 : 1;
}

}  // namespace soma::topology

#endif  // SOMA_TOPOLOGY_ORIENTATION_HPP_
