// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_POINT_KEY_HPP_
#define SOMA_CORE_POINT_KEY_HPP_

/// \file
/// The exact keys of point coordinates (spatial_exact::key_of in
/// soma/core/hash3d_exact.hpp) in device code: a coordinate's IEEE bits made
/// an unsigned integer that sorts as the coordinate does, -0.0 as +0.0.
/// Integer operations only, so no floating-point flag changes them.

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"

namespace soma {

KOKKOS_INLINE_FUNCTION bool is_nan_bits(Real v) {
  const auto b = Kokkos::bit_cast<std::uint64_t>(v);
  return (b & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL && (b & 0x000FFFFFFFFFFFFFULL) != 0;
}

/// Meaningless for NaN.
KOKKOS_INLINE_FUNCTION std::uint64_t point_key(Real v) {
  auto b = Kokkos::bit_cast<std::uint64_t>(v);
  if (b == 0x8000000000000000ULL) b = 0;  // -0.0 as +0.0
  return (b >> 63) != 0 ? ~b : (b | (std::uint64_t{1} << 63));
}

/// The coordinate of a key.
KOKKOS_INLINE_FUNCTION Real point_of_key(std::uint64_t k) {
  return Kokkos::bit_cast<Real>((k >> 63) != 0 ? (k & ~(std::uint64_t{1} << 63)) : ~k);
}

/// keys(3 i + j) = point_key(xyz(3 i + j)) for every coordinate, in parallel
/// on `exec`; returns whether any coordinate is NaN.
template <class ExecSpace, class In, class Out>
bool point_keys(const ExecSpace& exec, const In& xyz, const Out& keys) {
  int nan = 0;
  Kokkos::parallel_reduce(
      "soma::point_keys", Kokkos::RangePolicy<ExecSpace>(exec, 0, xyz.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i, int& count) {
        keys(i) = point_key(xyz(i));
        if (is_nan_bits(xyz(i))) ++count;
      },
      nan);
  return nan != 0;
}

}  // namespace soma

#endif  // SOMA_CORE_POINT_KEY_HPP_
