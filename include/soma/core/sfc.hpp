// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_SFC_HPP_
#define SOMA_CORE_SFC_HPP_

// Hilbert curve keys, used to order entities so that neighbours in space are
// close in memory. The index of a cell follows Skilling's transpose algorithm
// (J. Skilling, "Programming the Hilbert curve", AIP Conf. Proc. 707, 2004):
// the cell coordinates are turned into the transposed Hilbert index by Gray
// coding in place, and the bits are then interleaved. Everything is a
// constexpr-friendly, allocation-free function that works in device code.

#include <cmath>
#include <cstdint>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"

namespace soma::sfc {

// Bits per axis of a curve over n axes: the most that fit both a 64-bit key
// and a 32-bit cell coordinate, so 32 for one or two axes and 21 for three.
KOKKOS_INLINE_FUNCTION constexpr int bits_per_axis(int n) { return n == 3 ? 21 : 32; }

// The last cell along an axis of a curve over n axes.
KOKKOS_INLINE_FUNCTION constexpr std::uint32_t max_cell(int n) {
  return static_cast<std::uint32_t>((std::uint64_t{1} << bits_per_axis(n)) - 1);
}

// Index along the Hilbert curve of the cell c[0..n), which has `bits` bits per
// axis (n in 1..3, bits in 1..32, n * bits <= 64; asserted in debug builds).
// The curve starts at the origin and consecutive indexes are cells that differ
// by one step along one axis.
KOKKOS_INLINE_FUNCTION std::uint64_t hilbert_index(const std::uint32_t* c, int n, int bits) {
  KOKKOS_ASSERT(n >= 1 && n <= 3);
  KOKKOS_ASSERT(bits >= 1 && bits <= 32);
  KOKKOS_ASSERT(n * bits <= 64);
  std::uint32_t x[3] = {0, 0, 0};
  for (int i = 0; i < n; ++i) x[i] = c[i];
  const std::uint32_t top = std::uint32_t{1} << (bits - 1);
  for (std::uint32_t q = top; q > 1; q >>= 1) {
    const std::uint32_t p = q - 1;
    for (int i = 0; i < n; ++i) {
      if ((x[i] & q) != 0) {
        x[0] ^= p;
      } else {
        const std::uint32_t t = (x[0] ^ x[i]) & p;
        x[0] ^= t;
        x[i] ^= t;
      }
    }
  }
  for (int i = 1; i < n; ++i) x[i] ^= x[i - 1];
  std::uint32_t t = 0;
  for (std::uint32_t q = top; q > 1; q >>= 1) {
    if ((x[n - 1] & q) != 0) t ^= q - 1;
  }
  for (int i = 0; i < n; ++i) x[i] ^= t;
  std::uint64_t key = 0;
  for (int bit = bits - 1; bit >= 0; --bit) {
    for (int i = 0; i < n; ++i) key = (key << 1) | ((x[i] >> bit) & 1U);
  }
  return key;
}

// Maps a point of a bounding box to a cell of a grid with bits_per_axis(n)
// bits per axis. Only the n axes on which the box has extent take part, so a
// planar mesh gets a 2D curve and a single point gets key 0. An axis whose
// extent is at most one cell of the widest axis, on the curve over every axis
// with extent, is dropped too: it could change no key, yet would stretch its
// noise over the whole grid.
struct KeyBox {
  int n = 0;
  int axis[3] = {0, 0, 0};
  Real lo[3] = {0, 0, 0};
  Real inv_extent[3] = {0, 0, 0};

  // lo and hi are the box corners.
  static KeyBox make(const Real* lo, const Real* hi) {
    Real extent[3] = {0, 0, 0};
    Real widest = 0;
    int with_extent = 0;
    for (int d = 0; d < 3; ++d) {
      if (hi[d] > lo[d]) {
        extent[d] = hi[d] - lo[d];
        widest = extent[d] > widest ? extent[d] : widest;
        ++with_extent;
      }
    }
    KeyBox b;
    if (with_extent == 0) return b;
    const Real cell = std::ldexp(widest, -bits_per_axis(with_extent));
    for (int d = 0; d < 3; ++d) {
      if (extent[d] > cell) {
        b.axis[b.n] = d;
        b.lo[b.n] = lo[d];
        b.inv_extent[b.n] = Real{1} / extent[d];
        ++b.n;
      }
    }
    return b;
  }
};

// floor(t * last) for t in [0, 1]; values outside the box (and NaN) clamp.
KOKKOS_INLINE_FUNCTION std::uint32_t quantize(Real v, Real lo, Real inv_extent,
                                              std::uint32_t last) {
  const Real t = (v - lo) * inv_extent;
  if (!(t > 0)) return 0;
  if (t >= 1) return last;
  return static_cast<std::uint32_t>(t * static_cast<Real>(last));
}

KOKKOS_INLINE_FUNCTION std::uint64_t hilbert_key(const KeyBox& box, Real x, Real y, Real z) {
  if (box.n == 0) return 0;
  const Real p[3] = {x, y, z};
  std::uint32_t c[3] = {0, 0, 0};
  const std::uint32_t last = max_cell(box.n);
  for (int i = 0; i < box.n; ++i) {
    c[i] = quantize(p[box.axis[i]], box.lo[i], box.inv_extent[i], last);
  }
  return hilbert_index(c, box.n, bits_per_axis(box.n));
}

}  // namespace soma::sfc

#endif  // SOMA_CORE_SFC_HPP_
