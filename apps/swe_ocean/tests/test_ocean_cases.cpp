// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The analytic cases of the barotropic ocean, in its linear mode without hyperviscosity: Stommel's wind-driven
// gyre on a beta plane converges with the element size, a Kelvin wave packet in a closed f-plane channel
// converges with the order, and both errors have the same bits on any number of ranks.

#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>

#include <gtest/gtest.h>
#include "cases.hpp"
#include "soma/parallel/comm.hpp"

namespace {

using ocean::CaseResult;
using soma::Comm;

void report(const Comm& c, const char* name, int nx, int order, const CaseResult& r) {
  if (c.rank() == 0) {
    std::printf("%s nx %d order %d: l2 %.6e, max %.6e, steps %ld\n", name, nx, order, r.error_l2, r.error_max,
                r.steps);
  }
}

// Stommel at order 4 on nx_coarse and nx_fine elements a side. The western boundary layer r / beta = 100 km is
// 0.4 of an element at nx = 4 (250 km) and 0.8 at nx = 8; 6 and 12 pass too (1.2e-3 and 8.2e-5, ratio 15) but
// take 44 s at 4 ranks on their own, over the 60 s budget with the rest of this file. At order 4 the error
// should fall as h^4 to h^5 (16 to 32 times per halving), so a ratio of 8 leaves room for the pre-asymptotic coarse
// mesh (13.7 measured); 1e-3 is 2.5 times the error measured at nx = 8 (4.0e-4), far above the steady-state
// tolerance's share (1e-7 relative change over 1 / r).
constexpr int stommel_order = 4, nx_coarse = 4, nx_fine = 8;
constexpr double stommel_ratio = 8, stommel_fine = 1e-3;

// Kelvin on 40 x 10 elements (100 km), the packet (sigma = 200 km) two elements wide.
constexpr int kelvin_nx = 40;

// The runs on every rank, each made once and shared by the tests.
const CaseResult& stommel(int nx) {
  static std::map<int, CaseResult> runs;
  if (runs.count(nx) == 0) runs[nx] = ocean::run_stommel(Comm::world(), nx, stommel_order, ocean::StommelSpec{});
  return runs[nx];
}

const CaseResult& kelvin(int order) {
  static std::map<int, CaseResult> runs;
  if (runs.count(order) == 0) runs[order] = ocean::run_kelvin(Comm::world(), kelvin_nx, order, ocean::KelvinSpec{});
  return runs[order];
}

TEST(Stommel, converges) {
  const Comm c = Comm::world();
  const CaseResult& coarse = stommel(nx_coarse);
  report(c, "stommel", nx_coarse, stommel_order, coarse);
  const CaseResult& fine = stommel(nx_fine);
  report(c, "stommel", nx_fine, stommel_order, fine);
  EXPECT_GE(coarse.error_l2, stommel_ratio * fine.error_l2);
  EXPECT_LT(fine.error_l2, stommel_fine);
}

// A run that cannot be steady within its step cap throws.
TEST(Stommel, step_cap_throws) {
  const Comm c = Comm::world();
  ocean::StommelSpec s;
  s.max_steps = 10;
  EXPECT_THROW(ocean::run_stommel(c, 2, 2, s), std::runtime_error);
}

// Spectral convergence: each step of 2 in the order gains at least a factor of 10 (520 and 111 measured); 1e-6 at
// order 6 (3.7e-7 measured, of which about 2e-7 spatial) bounds the time-stepping error too.
constexpr double kelvin_gain = 10, kelvin_order6 = 1e-6;

TEST(Kelvin, converges_in_p) {
  const Comm c = Comm::world();
  double last = 0;
  for (const int order : {2, 4, 6}) {
    const CaseResult& r = kelvin(order);
    report(c, "kelvin", kelvin_nx, order, r);
    EXPECT_GT(r.error_l2, 0);
    if (order > 2) EXPECT_LE(kelvin_gain * r.error_l2, last) << "order " << order;
    if (order == 6) EXPECT_LT(r.error_l2, kelvin_order6);
    last = r.error_l2;
  }
}

// The errors and step counts of the coarsest runs on every rank equal those of the same runs on one rank, bit for
// bit.
TEST(Cases, rank_independent) {
  const Comm c = Comm::world();
  const CaseResult &s = stommel(nx_coarse), &k = kelvin(2);
  if (c.size() > 1) {
    const CaseResult s1 = ocean::run_stommel(Comm::self(), nx_coarse, stommel_order, ocean::StommelSpec{});
    const CaseResult k1 = ocean::run_kelvin(Comm::self(), kelvin_nx, 2, ocean::KelvinSpec{});
    EXPECT_EQ(s.error_l2, s1.error_l2);
    EXPECT_EQ(s.error_max, s1.error_max);
    EXPECT_EQ(s.steps, s1.steps);
    EXPECT_EQ(k.error_l2, k1.error_l2);
    EXPECT_EQ(k.error_max, k1.error_max);
    EXPECT_EQ(k.steps, k1.steps);
  }
  if (c.rank() == 0) {
    std::printf("errors reproducible %.17g %.17g %.17g %.17g\n", s.error_l2, s.error_max, k.error_l2, k.error_max);
  }
}

TEST(Cases, reject_bad_specs) {
  const Comm c = Comm::world();
  EXPECT_THROW(ocean::run_stommel(c, 0, 4, ocean::StommelSpec{}), std::invalid_argument);
  EXPECT_THROW(ocean::run_kelvin(c, 4, 0, ocean::KelvinSpec{}), std::invalid_argument);
  ocean::StommelSpec s;
  s.r = 0;
  EXPECT_THROW(ocean::run_stommel(c, 2, 2, s), std::invalid_argument);
  ocean::KelvinSpec k;
  k.sigma = std::nan("");
  EXPECT_THROW(ocean::run_kelvin(c, 4, 2, k), std::invalid_argument);
}

}  // namespace
