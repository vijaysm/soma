// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_SFC_HPP_
#define SOMA_CORE_SFC_HPP_

// Hilbert curve keys, used to order entities so that neighbours in space are
// close in memory. The index of a cell follows Skilling's transpose algorithm
// (J. Skilling, "Programming the Hilbert curve", AIP Conf. Proc. 707, 2004):
// the cell coordinates are turned into the transposed Hilbert index by Gray
// coding in place, and the bits are then interleaved. Everything is
// allocation-free, and everything but KeyBox::make, which runs on the host,
// works in device code.

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

namespace detail {

// Spreads the low 32 (21) bits of v so that bit b lands on bit 2b (3b).
KOKKOS_INLINE_FUNCTION std::uint64_t spread2(std::uint64_t v) {
  v &= 0xFFFFFFFFULL;
  v = (v | v << 16) & 0x0000FFFF0000FFFFULL;
  v = (v | v << 8) & 0x00FF00FF00FF00FFULL;
  v = (v | v << 4) & 0x0F0F0F0F0F0F0F0FULL;
  v = (v | v << 2) & 0x3333333333333333ULL;
  return (v | v << 1) & 0x5555555555555555ULL;
}

KOKKOS_INLINE_FUNCTION std::uint64_t spread3(std::uint64_t v) {
  v &= 0x1FFFFFULL;
  v = (v | v << 32) & 0x001F00000000FFFFULL;
  v = (v | v << 16) & 0x001F0000FF0000FFULL;
  v = (v | v << 8) & 0x100F00F00F00F00FULL;
  v = (v | v << 4) & 0x10C30C30C30C30C3ULL;
  return (v | v << 2) & 0x1249249249249249ULL;
}

// The transpose steps over N axes known at compile time, so that x stays in
// registers. They select with masks, not branches: the bits of a cell are
// close to random, so branches on them mispredict.
template <int N>
KOKKOS_INLINE_FUNCTION std::uint64_t hilbert_index(const std::uint32_t* c, int bits) {
  std::uint32_t x[N];
  for (int i = 0; i < N; ++i) x[i] = c[i];
  const std::uint32_t top = std::uint32_t{1} << (bits - 1);
  for (std::uint32_t q = top; q > 1; q >>= 1) {
    const std::uint32_t p = q - 1;
    for (int i = 0; i < N; ++i) {
      // set: x[0] ^= p; clear: swap the low bits of x[0] and x[i].
      const std::uint32_t set = 0U - ((x[i] & q) != 0 ? 1U : 0U);
      const std::uint32_t t = (x[0] ^ x[i]) & p & ~set;
      x[0] ^= (p & set) | t;
      x[i] ^= t;
    }
  }
  for (int i = 1; i < N; ++i) x[i] ^= x[i - 1];
  std::uint32_t t = 0;
  for (std::uint32_t q = top; q > 1; q >>= 1) {
    t ^= (q - 1) & (0U - ((x[N - 1] & q) != 0 ? 1U : 0U));
  }
  // Interleave the low `bits` bits, x[0] the most significant of each group.
  const std::uint64_t low = bits == 32 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1;
  if constexpr (N == 1) {
    return (x[0] ^ t) & low;
  } else if constexpr (N == 2) {
    return spread2((x[0] ^ t) & low) << 1 | spread2((x[1] ^ t) & low);
  } else {
    return spread3((x[0] ^ t) & low) << 2 | spread3((x[1] ^ t) & low) << 1 |
           spread3((x[2] ^ t) & low);
  }
}

}  // namespace detail

// Index along the Hilbert curve of the cell c[0..n), which has `bits` bits per
// axis (n in 1..3, bits in 1..32, n * bits <= 64; asserted in debug builds).
// The curve starts at the origin and consecutive indexes are cells that differ
// by one step along one axis.
KOKKOS_INLINE_FUNCTION std::uint64_t hilbert_index(const std::uint32_t* c, int n, int bits) {
  KOKKOS_ASSERT(n >= 1 && n <= 3);
  KOKKOS_ASSERT(bits >= 1 && bits <= 32);
  KOKKOS_ASSERT(n * bits <= 64);
  if (n == 1) return detail::hilbert_index<1>(c, bits);
  if (n == 2) return detail::hilbert_index<2>(c, bits);
  return detail::hilbert_index<3>(c, bits);
}

// Maps a point of a bounding box to a cell of a grid with bits_per_axis(n)
// bits per axis. Only the n axes on which the box has extent take part, so a
// planar mesh gets a 2D curve and a single point gets key 0. An axis whose
// extent is at most one cell of the widest axis, on the curve over every axis
// with extent, is dropped too: it could change no key. Every kept axis is
// divided by the widest extent, so the grid covers a cube and keeps the shape
// of the mesh: a thin axis is not stretched over the whole grid. When an
// extent overflows, every coordinate is halved first (`scale`), which is exact
// for values that large.
struct KeyBox {
  int n = 0;
  int axis[3] = {0, 0, 0};
  Real scale = 1;
  Real lo[3] = {0, 0, 0};  // scaled
  Real extent = 0;  // scaled; the widest axis, shared by every kept axis

  // lo and hi are the finite box corners. Host only.
  static KeyBox make(const Real* lo, const Real* hi) {
    KeyBox b;
    for (int d = 0; d < 3; ++d) {
      if (hi[d] > lo[d] && !std::isfinite(hi[d] - lo[d])) b.scale = Real{0.5};
    }
    Real extent[3] = {0, 0, 0};
    Real widest = 0;
    int with_extent = 0;
    for (int d = 0; d < 3; ++d) {
      if (hi[d] > lo[d]) {
        extent[d] = b.scale * hi[d] - b.scale * lo[d];
        widest = extent[d] > widest ? extent[d] : widest;
        ++with_extent;
      }
    }
    if (with_extent == 0) return b;
    b.extent = widest;
    const Real cell = std::ldexp(widest, -bits_per_axis(with_extent));
    for (int d = 0; d < 3; ++d) {
      if (extent[d] > cell) {
        b.axis[b.n] = d;
        b.lo[b.n] = b.scale * lo[d];
        ++b.n;
      }
    }
    return b;
  }
};

// floor(t * last) for t = (v - lo) / extent in [0, 1]; values outside the box
// (and NaN) clamp. Dividing, rather than multiplying by 1 / extent, keeps
// subnormal extents, whose reciprocal overflows, usable.
KOKKOS_INLINE_FUNCTION std::uint32_t quantize(Real v, Real lo, Real extent, std::uint32_t last) {
  const Real t = (v - lo) / extent;
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
    c[i] = quantize(box.scale * p[box.axis[i]], box.lo[i], box.extent, last);
  }
  return hilbert_index(c, box.n, bits_per_axis(box.n));
}

}  // namespace soma::sfc

#endif  // SOMA_CORE_SFC_HPP_
