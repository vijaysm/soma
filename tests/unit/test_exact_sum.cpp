// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Exact sums: two terms round as IEEE addition does; any order, split or
// execution space gives the same bits; cancellation is exact; a term that is
// not finite makes NaN; summed over ranks, one allreduce gives the serial
// sum.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/parallel/comm.hpp"
#include "soma/parallel/exact_sum.hpp"

using soma::Comm;
using soma::ExactSums;
using soma::LO;
using soma::Real;

namespace {

std::uint64_t bits(Real x) { return std::bit_cast<std::uint64_t>(x); }

// A finite double of random sign, exponent and mantissa: subnormals, tiny
// and huge values alike (exponent fields in [lo, hi]).
Real random_double(std::mt19937_64& rng, int lo = 0, int hi = 2046) {
  std::uniform_int_distribution<int> field(lo, hi);
  const std::uint64_t mantissa = rng() & ((std::uint64_t{1} << 52) - 1);
  const std::uint64_t sign = rng() & 1;
  return std::bit_cast<Real>((sign << 63) | (static_cast<std::uint64_t>(field(rng)) << 52) | mantissa);
}

Real sum_of(const std::vector<Real>& terms) {
  ExactSums<1> s;
  for (std::size_t i = 0; i < terms.size(); ++i) {
    s.add(0, terms[i]);
    if (i % 4096 == 4095) s.normalize();
  }
  s.normalize();
  return s.value(0);
}

// sum of (x_i, x_i^2) over [0, n) in the execution space Exec.
template <class Exec>
ExactSums<2> sums_in(const Kokkos::View<const Real*, Kokkos::HostSpace>& x) {
  return soma::reduce_exact<2, Exec>("test::sums", 0, static_cast<LO>(x.extent(0)),
                                     KOKKOS_LAMBDA(const LO i, ExactSums<2>& acc) {
                                       acc.add(0, x(i));
                                       acc.add(1, x(i) * x(i));
                                     });
}

}  // namespace

// The exact sum of two doubles rounded once is IEEE addition.
TEST(ExactSum, TwoTermsRoundAsIeeeAddition) {
  std::mt19937_64 rng(5);
  for (int i = 0; i < 200000; ++i) {
    const Real a = random_double(rng);
    // Near exponents too, where rounding ties and carries happen.
    const int f = static_cast<int>((std::bit_cast<std::uint64_t>(a) >> 52) & 0x7ff);
    const Real b = i % 2 == 0 ? random_double(rng) : random_double(rng, std::max(0, f - 60), std::min(2046, f + 2));
    ASSERT_EQ(bits(sum_of({a, b})), bits(a + b)) << a << " + " << b;
  }
  // Ties to even, at the edge of the mantissa.
  const Real one = 1, ulp = std::ldexp(1.0, -52);
  EXPECT_EQ(sum_of({one, ulp / 2}), one);
  EXPECT_EQ(sum_of({one + ulp, ulp / 2}), one + 2 * ulp);
  EXPECT_EQ(sum_of({one, ulp / 2, std::ldexp(1.0, -200)}), one + ulp);
  // The ends of the range.
  const Real big = std::numeric_limits<Real>::max(), tiny = std::numeric_limits<Real>::denorm_min();
  EXPECT_EQ(sum_of({big, big}), std::numeric_limits<Real>::infinity());
  EXPECT_EQ(sum_of({-big, -big}), -std::numeric_limits<Real>::infinity());
  EXPECT_EQ(sum_of({tiny, tiny}), 2 * tiny);
  EXPECT_EQ(bits(sum_of({})), bits(0.0));
}

// Cancellation leaves exactly what is left.
TEST(ExactSum, CancellationIsExact) {
  EXPECT_EQ(sum_of({1e300, 1, -1e300}), 1);
  EXPECT_EQ(sum_of({1e-300, 1e300, -1e300}), 1e-300);
  std::mt19937_64 rng(7);
  std::vector<Real> terms;
  for (int i = 0; i < 20000; ++i) {
    const Real x = random_double(rng, 1, 2045);
    terms.push_back(x);
    terms.push_back(-x);
  }
  terms.push_back(0.1);
  std::shuffle(terms.begin(), terms.end(), rng);
  EXPECT_EQ(sum_of(terms), 0.1);
}

// Any order and any split, joined in any order, gives the same bits.
TEST(ExactSum, OrderAndSplitDoNotMatter) {
  std::mt19937_64 rng(11);
  std::vector<Real> terms;
  for (int i = 0; i < 20000; ++i) terms.push_back(random_double(rng, 900, 1150));
  const Real want = sum_of(terms);
  int differ_plainly = 0;
  for (int trial = 0; trial < 30; ++trial) {
    std::shuffle(terms.begin(), terms.end(), rng);
    std::uniform_int_distribution<std::size_t> cut(0, terms.size());
    std::vector<std::size_t> cuts{0, terms.size()};
    for (int k = 0; k < 6; ++k) cuts.push_back(cut(rng));
    std::sort(cuts.begin(), cuts.end());
    std::vector<ExactSums<1>> parts(cuts.size() - 1);
    for (std::size_t p = 0; p + 1 < cuts.size(); ++p) {
      for (std::size_t i = cuts[p]; i < cuts[p + 1]; ++i) parts[p].add(0, terms[i]);
      parts[p].normalize();
    }
    std::shuffle(parts.begin(), parts.end(), rng);
    ExactSums<1> total;
    for (const auto& p : parts) total.join(p);
    ASSERT_EQ(bits(total.value(0)), bits(want)) << "trial " << trial;
    Real plain = 0;
    for (const Real x : terms) plain += x;
    differ_plainly += bits(plain) != bits(want) ? 1 : 0;
  }
  // The plain sums do depend on the order: the test is not vacuous.
  EXPECT_GT(differ_plainly, 0);
}

// A NaN or infinite term makes its sum NaN, whatever else is added; the
// other sums stay.
TEST(ExactSum, TermsNotFiniteMakeNaN) {
  ExactSums<2> s;
  s.add(0, 1);
  s.add(0, std::numeric_limits<Real>::infinity());
  s.add(1, 2);
  s.normalize();
  EXPECT_TRUE(std::isnan(s.value(0)));
  EXPECT_EQ(s.value(1), 2);
  ExactSums<1> n;
  n.add(0, std::numeric_limits<Real>::quiet_NaN());
  n.normalize();
  EXPECT_TRUE(std::isnan(n.value(0)));
}

// The reducer gives the sequential sum, bit for bit, on one thread and on
// the default host space's threads.
TEST(ExactSum, ReductionsAreTheSequentialSumInEverySpace) {
  std::mt19937_64 rng(13);
  Kokkos::View<Real*, Kokkos::HostSpace> x("x", 300001);
  std::vector<Real> terms, squares;
  for (std::size_t i = 0; i < x.extent(0); ++i) {
    x(i) = random_double(rng, 800, 1250);
    terms.push_back(x(i));
    squares.push_back(x(i) * x(i));
  }
  const Kokkos::View<const Real*, Kokkos::HostSpace> cx = x;
#ifdef KOKKOS_ENABLE_SERIAL
  auto serial = sums_in<Kokkos::Serial>(cx);
#else
  auto serial = sums_in<Kokkos::DefaultHostExecutionSpace>(cx);
#endif
  auto threaded = sums_in<Kokkos::DefaultHostExecutionSpace>(cx);
  EXPECT_EQ(bits(serial.value(0)), bits(sum_of(terms)));
  EXPECT_EQ(bits(serial.value(1)), bits(sum_of(squares)));
  EXPECT_EQ(bits(threaded.value(0)), bits(serial.value(0)));
  EXPECT_EQ(bits(threaded.value(1)), bits(serial.value(1)));
}

// A reducer's result, joined or not, reads right without a normalize first.
void two_terms(Real t, ExactSums<1>& out) {
  Kokkos::parallel_reduce(
      "test::two", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, 2),
      KOKKOS_LAMBDA(const int, ExactSums<1>& acc) { acc.add(0, t); }, soma::ExactSumReducer<1>(out));
}

// sum over [0, n) of (i % 7) - 3.25, in passes.
Real long_sum(LO n) {
  const auto s = soma::reduce_exact<1, Kokkos::DefaultHostExecutionSpace>(
      "test::long", 0, n,
      KOKKOS_LAMBDA(const LO i, ExactSums<1>& acc) { acc.add(0, static_cast<Real>(i % 7) - 3.25); });
  return s.value(0);
}

// Each rank sums a share of the terms; one allreduce gives every rank the
// serial sum, and a term not finite on one rank makes its sum NaN on all.
TEST(ExactSum, AllreduceIsTheSerialSum) {
  const Comm c = Comm::world();
  std::mt19937_64 rng(17);
  std::vector<Real> terms;
  for (int i = 0; i < 50000; ++i) terms.push_back(random_double(rng, 700, 1300));
  ExactSums<2> mine;
  for (std::size_t i = 0; i < terms.size(); ++i) {
    if (static_cast<int>(i % static_cast<std::size_t>(c.size())) == c.rank()) {
      mine.add(0, terms[i]);
      mine.add(1, -terms[i] / 3);
    }
  }
  std::vector<Real> thirds;
  for (const Real t : terms) thirds.push_back(-t / 3);
  const auto before = soma::detail::comm_counts().collectives;
  const auto got = soma::allreduce_exact(c, mine);
  EXPECT_EQ(soma::detail::comm_counts().collectives - before, 1u);
  EXPECT_EQ(bits(got[0]), bits(sum_of(terms)));
  EXPECT_EQ(bits(got[1]), bits(sum_of(thirds)));
  if (c.rank() == c.size() - 1) mine.add(1, std::numeric_limits<Real>::quiet_NaN());
  const auto poisoned = soma::allreduce_exact(c, mine);
  EXPECT_EQ(bits(poisoned[0]), bits(sum_of(terms)));
  EXPECT_TRUE(std::isnan(poisoned[1]));
}

// value() needs no normalize first: a negative term alone, a reducer's own
// result.
TEST(ExactSum, ValuesNeedNoNormalizeFirst) {
  ExactSums<1> one;
  one.add(0, -1.0);
  EXPECT_EQ(one.value(0), -1.0);
  ExactSums<1> r;
  two_terms(-0.75, r);
  EXPECT_EQ(r.value(0), -1.5);
}

// A range past one pass (2^28 terms) sums in several, exactly.
TEST(ExactSum, LongRangesSumInPasses) {
  const LO n = static_cast<LO>(soma::exact::terms_per_pass + 12345);
  // Each full 7 of (i % 7) - 3.25 adds 21 - 22.75 = -1.75.
  const std::int64_t whole = n / 7;
  Real tail = 0;
  for (std::int64_t i = whole * 7; i < n; ++i) tail += static_cast<Real>(i % 7) - 3.25;
  EXPECT_EQ(long_sum(n), static_cast<Real>(whole) * -1.75 + tail);
}
