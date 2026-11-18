// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The shallow-water test cases: case 2 is in geostrophic balance, case 5's
// mountain, Galewsky's balanced jet and mean depth, case 6's height at two
// points against its closed form, and the case names.

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "cases.hpp"
#include "constants.hpp"

namespace {

using swe::Case;
using Vec = std::array<double, 3>;
constexpr double pi = std::numbers::pi;

Vec point(double lambda, double theta) {
  return {swe::radius * std::cos(theta) * std::cos(lambda), swe::radius * std::cos(theta) * std::sin(lambda),
          swe::radius * std::sin(theta)};
}

// Eastward and northward components of a velocity at (lambda, theta).
std::array<double, 2> east_north(const Vec& u, double lambda, double theta) {
  const Vec e = {-std::sin(lambda), std::cos(lambda), 0};
  const Vec n = {-std::sin(theta) * std::cos(lambda), -std::sin(theta) * std::sin(lambda), std::cos(theta)};
  return {u[0] * e[0] + u[1] * e[1] + u[2] * e[2], u[0] * n[0] + u[1] * n[1] + u[2] * n[2]};
}

TEST(SweCases, Case2IsGeostrophic) {
  std::mt19937_64 rng(20261117);
  std::uniform_real_distribution<double> lon(0, 2 * pi), lat(-1.4, 1.4);
  const double step = 1e-4;
  for (const double alpha : {0.0, pi / 4}) {
    // Per point: the residual of (f + zeta) n x u + grad(g h + |u|^2 / 2), and g |grad h|.
    double worst = 0, scale = 0;
    for (int n = 0; n < 200; ++n) {
      const double l = lon(rng), t = lat(rng);
      const auto at = [&](double dl, double dt) { return swe::initial(Case::williamson2, point(l + dl, t + dt), alpha); };
      const auto phi = [&](double dl, double dt) {
        const swe::PointState s = at(dl, dt);
        return swe::gravity * (s.h + s.hs) + 0.5 * (s.u[0] * s.u[0] + s.u[1] * s.u[1] + s.u[2] * s.u[2]);
      };
      const auto uv = [&](double dl, double dt) { return east_north(at(dl, dt).u, l + dl, t + dt); };
      const double c = std::cos(t);
      const double dphi_l = (phi(step, 0) - phi(-step, 0)) / (2 * step);
      const double dphi_t = (phi(0, step) - phi(0, -step)) / (2 * step);
      const double dv_l = (uv(step, 0)[1] - uv(-step, 0)[1]) / (2 * step);
      const double ducos_t =
          (uv(0, step)[0] * std::cos(t + step) - uv(0, -step)[0] * std::cos(t - step)) / (2 * step);
      const double zeta = (dv_l - ducos_t) / (swe::radius * c);
      const swe::PointState s = at(0, 0);
      const auto [u, v] = east_north(s.u, l, t);
      // n x u = (-v, u) in (east, north).
      const double re = (s.f + zeta) * -v + dphi_l / (swe::radius * c);
      const double rn = (s.f + zeta) * u + dphi_t / swe::radius;
      worst = std::max(worst, std::hypot(re, rn));
      const double dh_l = (at(step, 0).h - at(-step, 0).h) / (2 * step) / (swe::radius * c);
      const double dh_t = (at(0, step).h - at(0, -step).h) / (2 * step) / swe::radius;
      scale = std::max(scale, swe::gravity * std::hypot(dh_l, dh_t));
    }
    EXPECT_GT(scale, 0);
    EXPECT_LE(worst, 1e-6 * scale) << "alpha " << alpha;
  }
}

TEST(SweCases, Case5Mountain) {
  const double u0 = 20, h0 = 5960;
  const auto free_surface = [&](double theta) {
    return h0 - (swe::radius * swe::omega * u0 + u0 * u0 / 2) * std::sin(theta) * std::sin(theta) / swe::gravity;
  };
  const swe::PointState top = swe::initial(Case::williamson5, point(3 * pi / 2, pi / 6));
  EXPECT_NEAR(top.hs, 2000, 1e-9);
  // On the rim to rounding (the distance is computed), and beyond it exactly.
  for (const auto& [l, t] : {std::array<double, 2>{3 * pi / 2 + pi / 9, pi / 6}, {3 * pi / 2, pi / 6 - pi / 9}}) {
    EXPECT_NEAR(swe::initial(Case::williamson5, point(l, t)).hs, 0, 1e-9) << l << " " << t;
  }
  for (const auto& [l, t] : {std::array<double, 2>{3 * pi / 2 + 0.35, pi / 6}, {0.3, -0.7}, {pi / 2, pi / 6}}) {
    EXPECT_EQ(swe::initial(Case::williamson5, point(l, t)).hs, 0) << l << " " << t;
  }
  for (const auto& [l, t] : {std::array<double, 2>{3 * pi / 2, pi / 6}, {3 * pi / 2 + 0.1, pi / 6 - 0.05},
                             {1.0, 0.2}, {4.0, -1.2}}) {
    const swe::PointState s = swe::initial(Case::williamson5, point(l, t));
    EXPECT_NEAR(s.h + s.hs, free_surface(t), 1e-9) << l << " " << t;
  }
}

TEST(SweCases, GalewskyBalanceAndMean) {
  const int nlon = 720, nlat = 360;
  double sum = 0, area = 0;
  for (int j = 0; j < nlat; ++j) {
    const double t = -pi / 2 + (j + 0.5) * pi / nlat;
    for (int i = 0; i < nlon; ++i) {
      const double l = (i + 0.5) * 2 * pi / nlon;
      sum += swe::initial(Case::galewsky_steady, point(l, t)).h * std::cos(t);
      area += std::cos(t);
    }
  }
  EXPECT_NEAR(sum / area, 10000, 0.1);
  const double t0 = pi / 7, t1 = pi / 2 - t0;
  const swe::PointState peak = swe::initial(Case::galewsky_steady, point(1.0, (t0 + t1) / 2));
  EXPECT_NEAR(east_north(peak.u, 1.0, (t0 + t1) / 2)[0], 80, 1e-12);
  for (const double t : {-1.0, 0.0, t0, t1, 1.5}) {
    const swe::PointState s = swe::initial(Case::galewsky_steady, point(0.4, t));
    EXPECT_EQ(std::hypot(s.u[0], s.u[1], s.u[2]), 0) << t;
  }
  // The perturbation is the only difference, largest at (0, pi / 4).
  const swe::PointState a = swe::initial(Case::galewsky, point(0, pi / 4));
  const swe::PointState b = swe::initial(Case::galewsky_steady, point(0, pi / 4));
  EXPECT_NEAR(a.h - b.h, 120 * std::cos(pi / 4), 1e-9);
}

TEST(SweCases, Case6Values) {
  // Williamson et al. (1992), test case 6, typed again: omega = K = 7.848e-6,
  // R = 4, h0 = 8000.
  const double w = 7.848e-6, k = 7.848e-6, h0 = 8000, a = swe::radius, om = swe::omega;
  const auto height = [&](double l, double t) {
    const double c = std::cos(t), c2 = c * c;
    const double c8 = std::pow(c, 8), c4 = std::pow(c, 4);
    const double big_a = w / 2 * (2 * om + w) * c2 + k * k / 4 * c8 * (5 * c2 + 26 - 32 / c2);
    const double big_b = 2 * (om + w) * k / 30 * c4 * (26 - 25 * c2);
    const double big_c = k * k / 4 * c8 * (5 * c2 - 6);
    return h0 + a * a * (big_a + big_b * std::cos(4 * l) + big_c * std::cos(8 * l)) / swe::gravity;
  };
  for (const auto& [l, t] : {std::array<double, 2>{0, 0}, {pi / 4, pi / 3}}) {
    const swe::PointState s = swe::initial(Case::williamson6, point(l, t));
    EXPECT_NEAR(s.h, height(l, t), 1e-9 * h0) << l << " " << t;
    EXPECT_EQ(s.hs, 0);
  }
  // Velocity at the equator, lambda = 0: u = a w + a K (-1), v = 0.
  const swe::PointState s = swe::initial(Case::williamson6, point(0, 0));
  const auto uv = east_north(s.u, 0, 0);
  EXPECT_NEAR(uv[0], a * w - a * k, 1e-9);
  EXPECT_NEAR(uv[1], 0, 1e-9);
}

TEST(SweCases, NamesRoundTrip) {
  for (const Case c : {Case::williamson2, Case::williamson5, Case::williamson6, Case::galewsky,
                       Case::galewsky_steady}) {
    EXPECT_EQ(swe::case_from_name(swe::case_name(c)), c);
  }
  try {
    (void)swe::case_from_name("w7");
    FAIL() << "w7 parsed";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("w2, w5, w6, galewsky, galewsky-steady"), std::string::npos) << e.what();
  }
  EXPECT_THROW((void)swe::exact(Case::williamson5, point(0, 0), 0, 0), std::invalid_argument);
}

}  // namespace
