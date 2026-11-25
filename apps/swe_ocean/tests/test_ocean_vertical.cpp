// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The vertical coordinate: reference thicknesses sum to the depth in either coordinate, the s-coordinate is
// ROMS's (Vtransform 2, Vstretching 4), z-star keeps partial and vanished layers, and the slope factors rx0
// and rx1.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include "vertical.hpp"

namespace {

constexpr double eps = std::numeric_limits<double>::epsilon();
// The telescoped sum of K thicknesses rounds in each of its K - 1 additions and in each difference of
// interfaces: 4 ulps per level of the depth bounds it.
double sum_tolerance(int K, double H) { return 4.0 * eps * K * H; }
// Test and code evaluate the same formulas in the same order but may contract them differently: a few ulps
// of the largest operand (cosh(5) ~ 74, exp(2) ~ 7) bound the difference.
constexpr double formula_tolerance = 1e-12;
// The s-coordinate at H = hc * 1e-6 differs from uniform sigma by (H / (hc + H)) times the stretching's
// departure from uniform, at most about the largest K dC (below 50 for the defaults): a bound of 1e-6 * 50.
constexpr double shallow_tolerance = 5e-5;
// The hand-computed z-star thicknesses are decimals (14.999, 1e-3) subtracted from interfaces near 100: a few
// hundred ulps of 100 bound them.
constexpr double zstar_tolerance = 1e-12;
// rx1 of the hand-computed columns is a ratio of integers, exact to a couple of ulps.
constexpr double ratio_tolerance = 1e-14;

std::vector<double> thicknesses(const ocean::VerticalSpec& spec, double H) {
  std::vector<double> h0(static_cast<std::size_t>(spec.levels));
  ocean::reference_thicknesses(spec, H, h0);
  return h0;
}

double sum(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0); }

}  // namespace

TEST(Vertical, s_sums_to_depth) {
  const ocean::VerticalSpec spec;
  for (const double H : {10.0, 100.0, 1000.0, 6000.0}) {
    const auto h0 = thicknesses(spec, H);
    EXPECT_NEAR(sum(h0), H, sum_tolerance(spec.levels, H)) << "H = " << H;
    for (const double h : h0) EXPECT_GT(h, 0.0) << "H = " << H;
  }
}

TEST(Vertical, s_matches_roms) {
  const ocean::VerticalSpec spec;  // K = 10, theta_s = 5, theta_b = 2, hc = 100
  const double H = 1000.0;
  const auto h0 = thicknesses(spec, H);
  const int K = spec.levels;
  auto interface = [&](int k) {
    const double s = -static_cast<double>(k) / K;
    double C = (1.0 - std::cosh(spec.theta_s * s)) / (std::cosh(spec.theta_s) - 1.0);
    C = (std::exp(spec.theta_b * C) - 1.0) / (1.0 - std::exp(-spec.theta_b));
    return H * (spec.hc * s + H * C) / (spec.hc + H);
  };
  for (int k = 1; k <= K; ++k) {
    EXPECT_NEAR(h0[static_cast<std::size_t>(k - 1)], interface(k - 1) - interface(k), formula_tolerance * H)
        << "level " << k;
  }
  // The interfaces run from the surface to the bottom.
  EXPECT_NEAR(interface(0), 0.0, formula_tolerance);
  EXPECT_NEAR(interface(K), -H, formula_tolerance * H);
}

TEST(Vertical, s_shallow_is_uniform_sigma) {
  const ocean::VerticalSpec spec;
  const double H = spec.hc * 1e-6;
  const auto h0 = thicknesses(spec, H);
  for (const double h : h0) EXPECT_NEAR(h, H / spec.levels, shallow_tolerance * H / spec.levels);
}

TEST(Vertical, zstar_partial_and_vanished) {
  ocean::VerticalSpec spec;
  spec.kind = ocean::Vertical::zstar;
  spec.levels = 4;
  spec.z = {0.0, -10.0, -30.0, -60.0, -100.0};
  const struct {
    double H;
    std::vector<double> expect;
  } cases[] = {
      {45.0, {10.0, 20.0, 14.999, 0.001}},
      {200.0, {10.0, 20.0, 30.0, 140.0}},  // the bottom replaces z_4: 200 - 60
      {10.0, {10.0 - 3e-3, 1e-3, 1e-3, 1e-3}},
  };
  for (const auto& c : cases) {
    const auto h0 = thicknesses(spec, c.H);
    for (std::size_t k = 0; k < 4; ++k) EXPECT_NEAR(h0[k], c.expect[k], zstar_tolerance) << "H = " << c.H << " k " << k;
    EXPECT_NEAR(sum(h0), c.H, sum_tolerance(4, c.H)) << "H = " << c.H;
  }
}

TEST(Vertical, zstar_default_profile_reaches_5500) {
  ocean::VerticalSpec spec;
  spec.kind = ocean::Vertical::zstar;
  const auto h0 = thicknesses(spec, 5500.0);
  EXPECT_NEAR(sum(h0), 5500.0, sum_tolerance(spec.levels, 5500.0));
  // Stretched: thin at the surface, thick at depth, every level positive.
  for (std::size_t k = 1; k < h0.size(); ++k) EXPECT_GT(h0[k], h0[k - 1]);
  EXPECT_GT(h0.front(), 0.0);
}

TEST(Vertical, invalid_specs_throw) {
  std::vector<double> h0(10);
  ocean::VerticalSpec spec;
  EXPECT_THROW(ocean::reference_thicknesses(spec, 0.0, h0), std::invalid_argument);
  EXPECT_THROW(ocean::reference_thicknesses(spec, -5.0, h0), std::invalid_argument);
  auto bad = [&](auto&& edit) {
    ocean::VerticalSpec s;
    edit(s);
    EXPECT_THROW(ocean::reference_thicknesses(s, 100.0, h0), std::invalid_argument);
  };
  bad([](ocean::VerticalSpec& s) { s.levels = 0; });
  bad([](ocean::VerticalSpec& s) { s.theta_s = 0.0; });
  bad([](ocean::VerticalSpec& s) { s.theta_s = 10.5; });
  bad([](ocean::VerticalSpec& s) { s.theta_b = -0.1; });
  bad([](ocean::VerticalSpec& s) { s.theta_b = 4.5; });
  bad([](ocean::VerticalSpec& s) { s.hc = -1.0; });
  bad([](ocean::VerticalSpec& s) {
    s.kind = ocean::Vertical::zstar;
    s.h_min = 20.0;  // H = 100 < K h_min = 200
  });
  bad([](ocean::VerticalSpec& s) {
    s.kind = ocean::Vertical::zstar;
    s.z = {0.0, -1.0, -2.0};  // not K + 1 interfaces
  });
  std::vector<double> wrong(3);
  EXPECT_THROW(ocean::reference_thicknesses(spec, 100.0, wrong), std::invalid_argument);
}

TEST(Vertical, rx0_is_the_slope_factor) {
  EXPECT_NEAR(ocean::rx0(100.0, 150.0), 0.2, ratio_tolerance);
  EXPECT_EQ(ocean::rx0(80.0, 80.0), 0.0);
}

TEST(Vertical, rx1_of_flat_levels_is_zero) {
  const std::vector<double> z = {0.0, -5.0, -20.0, -100.0};
  EXPECT_EQ(ocean::rx1(z, z), 0.0);
}

TEST(Vertical, rx1_known_value) {
  // Level 1: |(-1 + 1) + (0 - 0)| / |(-1 - 1) - (0 + 0)| = 0.
  // Level 2: |(-3 + 4) + (-1 + 1)| / |(-3 - 4) - (-1 - 1)| = 1 / 5.
  const std::vector<double> za = {0.0, -1.0, -3.0};
  const std::vector<double> zb = {0.0, -1.0, -4.0};
  EXPECT_NEAR(ocean::rx1(za, zb), 0.2, ratio_tolerance);
  EXPECT_THROW(ocean::rx1(za, std::vector<double>{0.0, -1.0}), std::invalid_argument);
}
