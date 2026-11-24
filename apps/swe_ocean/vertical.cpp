// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "vertical.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ocean {

namespace {

// The depth of the default z-star profile (m) and its stretching: z_k = -depth (exp(rate k / K) - 1) /
// (exp(rate) - 1).
constexpr double default_depth = 5500.0;
constexpr double default_rate = 3.0;

// ROMS's Vstretching 4 at s in [-1, 0]: C(0) = 0 and C(-1) = -1.
double stretching(const VerticalSpec& spec, double s) {
  double C = (1.0 - std::cosh(spec.theta_s * s)) / (std::cosh(spec.theta_s) - 1.0);
  if (spec.theta_b > 0.0) C = (std::exp(spec.theta_b * C) - 1.0) / (1.0 - std::exp(-spec.theta_b));
  return C;
}

}  // namespace

void reference_thicknesses(const VerticalSpec& spec, double H, std::span<double> h0) {
  const int K = spec.levels;
  if (K < 1) throw std::invalid_argument("reference_thicknesses: the number of levels must be at least 1");
  if (!(H > 0.0)) throw std::invalid_argument("reference_thicknesses: the depth must be positive");
  if (!(spec.theta_s > 0.0 && spec.theta_s <= 10.0))
    throw std::invalid_argument("reference_thicknesses: theta_s must be in (0, 10]");
  if (!(spec.theta_b >= 0.0 && spec.theta_b <= 4.0))
    throw std::invalid_argument("reference_thicknesses: theta_b must be in [0, 4]");
  if (!(spec.hc >= 0.0)) throw std::invalid_argument("reference_thicknesses: hc must not be negative");
  if (h0.size() != static_cast<std::size_t>(K))
    throw std::invalid_argument("reference_thicknesses: the output must hold one thickness per level");

  const auto nk = static_cast<std::size_t>(K);
  // The interfaces z0_0 = 0 above z0_K = -H, from which the thicknesses are differences: they telescope to H.
  double above = 0.0;
  if (spec.kind == Vertical::s) {
    for (int k = 1; k <= K; ++k) {
      double below = -H;
      if (k < K) {
        const double s = -static_cast<double>(k) / K;
        below = H * (spec.hc * s + H * stretching(spec, s)) / (spec.hc + H);
      }
      h0[static_cast<std::size_t>(k - 1)] = above - below;
      above = below;
    }
    return;
  }

  if (!(H >= K * spec.h_min))
    throw std::invalid_argument("reference_thicknesses: a z-star column must be at least levels * h_min deep");
  std::vector<double> z = spec.z;
  if (z.empty()) {
    z.resize(nk + 1);
    for (int k = 0; k <= K; ++k)
      z[static_cast<std::size_t>(k)] =
          -default_depth * std::expm1(default_rate * k / K) / std::expm1(default_rate);
  }
  if (z.size() != nk + 1 || z[0] != 0.0)
    throw std::invalid_argument("reference_thicknesses: z must hold levels + 1 interfaces from 0");
  for (std::size_t k = 1; k <= nk; ++k) {
    if (!(z[k] < z[k - 1])) throw std::invalid_argument("reference_thicknesses: z must descend");
  }
  for (int k = 1; k <= K; ++k) {
    const double below = k < K ? std::max(z[static_cast<std::size_t>(k)], -H + (K - k) * spec.h_min) : -H;
    h0[static_cast<std::size_t>(k - 1)] = above - below;
    above = below;
  }
}

double rx0(double Ha, double Hb) {
  const double total = Ha + Hb;
  return total == 0.0 ? 0.0 : std::abs(Ha - Hb) / total;
}

double rx1(std::span<const double> za, std::span<const double> zb) {
  if (za.size() != zb.size() || za.size() < 2)
    throw std::invalid_argument("rx1: the columns must have the same number, at least 2, of interfaces");
  double worst = 0.0;
  for (std::size_t k = 1; k < za.size(); ++k) {
    const double denominator = std::abs(za[k] + zb[k] - za[k - 1] - zb[k - 1]);
    if (denominator == 0.0) continue;
    worst = std::max(worst, std::abs(za[k] - zb[k] + za[k - 1] - zb[k - 1]) / denominator);
  }
  return worst;
}

}  // namespace ocean
