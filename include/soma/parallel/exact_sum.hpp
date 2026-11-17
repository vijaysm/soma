// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_EXACT_SUM_HPP_
#define SOMA_PARALLEL_EXACT_SUM_HPP_

/// \file
/// Sums of doubles that do not depend on the order of the terms: each term is
/// added exactly to a fixed-point accumulator spanning every finite double
/// (68 signed 32-bit digits in int64 words), and the total is rounded to the
/// nearest double once, at the end. Integer addition is associative, so the
/// same terms give the same bits on any number of threads and ranks, however
/// they are split. A NaN or infinite term makes its sum NaN.
///
/// ```
///   ExactSums<2> s;                                   // or a kernel's reducer:
///   Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(LO i, ExactSums<2>& acc) {
///     acc.add(0, x(i) * y(i));
///     acc.add(1, x(i) * x(i));
///   }, ExactSumReducer<2>(s));
///   const auto [xy, xx] = allreduce_exact(comm, s);  // one allreduce of integers
/// ```
///
/// A term costs a few integer operations; a dot product by these sums the
/// rounded products exactly, so it is at least as accurate as a plain one.
/// One accumulator takes at most 2^28 terms between normalizations
/// (reduce_exact splits longer ranges); a reducer normalizes on every join.
/// The terms themselves must be the same bits wherever they are computed:
/// that holds for one build (each expression contracted alike, as compilers
/// do by default), not under -ffast-math, which also drops the NaN checks.
/// Tested on host execution spaces; on a GPU the 552-byte accumulator per sum
/// is a large reduction value.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

namespace exact {

inline constexpr int digits = 68;      ///< 32 bits each, from 2^-1074 up
inline constexpr int radix_bits = 32;  ///< a word holds 2^31 digits' worth of carries
inline constexpr std::int64_t terms_per_pass = std::int64_t{1} << 28;

/// Adds the finite double x to the digits d, exactly.
KOKKOS_INLINE_FUNCTION void add(std::int64_t* d, Real x) {
  const auto bits = Kokkos::bit_cast<std::uint64_t>(x);
  const auto field = static_cast<int>((bits >> 52) & 0x7ff);
  std::uint64_t m = bits & ((std::uint64_t{1} << 52) - 1);
  if (field != 0) m |= std::uint64_t{1} << 52;
  if (m == 0) return;
  // x = +-m 2^(s - 1074), s = 0 for subnormals.
  const int s = field == 0 ? 0 : field - 1;
  const int i = s / radix_bits, o = s % radix_bits;
  constexpr std::uint64_t low = 0xffffffffu;
  const std::uint64_t d0 = (m << o) & low;
  const std::uint64_t d1 = o == 0 ? (m >> 32) & low : (m >> (32 - o)) & low;
  const std::uint64_t d2 = o == 0 ? 0 : m >> (64 - o);
  const std::int64_t sign = (bits >> 63) != 0 ? -1 : 1;
  d[i] += sign * static_cast<std::int64_t>(d0);
  d[i + 1] += sign * static_cast<std::int64_t>(d1);
  d[i + 2] += sign * static_cast<std::int64_t>(d2);
}

/// Carries so that every digit but the last is in [0, 2^32); the last holds
/// the sign.
KOKKOS_INLINE_FUNCTION void normalize(std::int64_t* d) {
  std::int64_t carry = 0;
  for (int i = 0; i + 1 < digits; ++i) {
    const std::int64_t v = d[i] + carry;
    d[i] = v & 0xffffffff;
    carry = v >> radix_bits;  // arithmetic: the floor
  }
  d[digits - 1] += carry;
}

/// The digits' value rounded to the nearest double, ties to even (a total in
/// the subnormal range is a multiple of 2^-1074 and so exact).
KOKKOS_INLINE_FUNCTION Real to_real(const std::int64_t* in) {
  std::int64_t d[digits];
  for (int i = 0; i < digits; ++i) d[i] = in[i];
  normalize(d);
  const bool negative = d[digits - 1] < 0;
  if (negative) {
    for (int i = 0; i < digits; ++i) d[i] = -d[i];
    normalize(d);
  }
  int h = digits - 1;
  while (h >= 0 && d[h] == 0) --h;
  if (h < 0) return 0;
  // Past 2^(32 * 66 - 1074) = 2^1038: no double.
  if (h >= 66) return negative ? -Kokkos::Experimental::infinity_v<Real> : Kokkos::Experimental::infinity_v<Real>;
  const auto top = static_cast<std::uint64_t>(d[h]);
  const int length = 64 - Kokkos::countl_zero(top);  // 1 .. 32: the digit is normalized
  const auto at = [&](int k) { return k >= 0 ? static_cast<std::uint64_t>(d[k]) : std::uint64_t{0}; };
  // The 64 bits from the leading one down, and whether any bit below is set.
  const std::uint64_t m = (top << (64 - length)) | (at(h - 1) << (32 - length)) |
                          (length == 32 ? std::uint64_t{0} : at(h - 2) >> length);
  bool sticky = length != 32 && (at(h - 2) & ((std::uint64_t{1} << length) - 1)) != 0;
  for (int k = h - 3; k >= 0 && !sticky; --k) sticky = d[k] != 0;
  if (length == 32) sticky = sticky || at(h - 2) != 0;
  std::uint64_t keep = m >> 11;
  const std::uint64_t rest = m & 0x7ff;
  if (rest > 0x400 || (rest == 0x400 && (sticky || (keep & 1) != 0))) ++keep;
  const int exponent = radix_bits * h + length - 1 - 52 - 1074;
  const Real r = Kokkos::ldexp(static_cast<Real>(keep), exponent);
  return negative ? -r : r;
}

}  // namespace exact

/// K exact sums, and for each a count of the terms that were not finite.
template <int K>
struct ExactSums {
  std::int64_t d[K][exact::digits] = {};
  std::int64_t nonfinite[K] = {};

  KOKKOS_INLINE_FUNCTION void add(int k, Real x) {
    if (Kokkos::isfinite(x)) {
      exact::add(d[k], x);
    } else {
      ++nonfinite[k];
    }
  }
  KOKKOS_INLINE_FUNCTION void normalize() {
    for (int k = 0; k < K; ++k) exact::normalize(d[k]);
  }
  /// The sum rounded; NaN if a term was not finite.
  KOKKOS_INLINE_FUNCTION Real value(int k) const {
    return nonfinite[k] != 0 ? Kokkos::Experimental::quiet_NaN_v<Real> : exact::to_real(d[k]);
  }
  KOKKOS_INLINE_FUNCTION void join(const ExactSums& o) {
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < exact::digits; ++i) d[k][i] += o.d[k][i];
      nonfinite[k] += o.nonfinite[k];
    }
    normalize();
  }
};

/// A Kokkos reducer of ExactSums<K> into a host value.
template <int K>
struct ExactSumReducer {
  using reducer = ExactSumReducer;
  using value_type = ExactSums<K>;
  using result_view_type = Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

  value_type* value;

  explicit ExactSumReducer(value_type& v) : value(&v) {}

  KOKKOS_INLINE_FUNCTION void join(value_type& dest, const value_type& src) const { dest.join(src); }
  KOKKOS_INLINE_FUNCTION void init(value_type& v) const { v = value_type{}; }
  KOKKOS_INLINE_FUNCTION value_type& reference() const { return *value; }
  result_view_type view() const { return result_view_type(value); }
  KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
};

/// ExactSums<K> of body(i, sums) over [first, last), in passes of at most
/// exact::terms_per_pass indices (body adds at most one term per sum per
/// index); normalized.
template <int K, class Exec, class Body>
ExactSums<K> reduce_exact(const char* label, LO first, LO last, const Body& body) {
  ExactSums<K> total;
  for (std::int64_t at = first; at < last; at += exact::terms_per_pass) {
    const auto end = static_cast<LO>(std::min<std::int64_t>(last, at + exact::terms_per_pass));
    ExactSums<K> part;
    Kokkos::parallel_reduce(label, Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(static_cast<LO>(at), end), body,
                            ExactSumReducer<K>(part));
    part.normalize();
    total.join(part);
  }
  return total;
}

/// Collective: every rank's sums added (one allreduce of integers) and
/// rounded.
template <int K>
std::array<Real, K> allreduce_exact(const Comm& comm, ExactSums<K> sums) {
  sums.normalize();
  constexpr int width = exact::digits + 1;  // the digits, then the count
  std::array<std::int64_t, K * width> words{};
  for (int k = 0; k < K; ++k) {
    for (int i = 0; i < exact::digits; ++i) words[static_cast<std::size_t>(k * width + i)] = sums.d[k][i];
    words[static_cast<std::size_t>(k * width + exact::digits)] = sums.nonfinite[k];
  }
  comm.allreduce(std::span<std::int64_t>(words), Op::sum);
  for (int k = 0; k < K; ++k) {
    for (int i = 0; i < exact::digits; ++i) sums.d[k][i] = words[static_cast<std::size_t>(k * width + i)];
    sums.nonfinite[k] = words[static_cast<std::size_t>(k * width + exact::digits)];
  }
  sums.normalize();
  std::array<Real, K> out{};
  for (int k = 0; k < K; ++k) out[static_cast<std::size_t>(k)] = sums.value(k);
  return out;
}

}  // namespace soma

#endif  // SOMA_PARALLEL_EXACT_SUM_HPP_
