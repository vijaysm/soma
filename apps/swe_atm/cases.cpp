// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "cases.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.hpp"

namespace swe {

namespace {

constexpr double pi = std::numbers::pi;

struct Spherical {
  double lambda, theta;  // longitude in (-pi, pi], latitude
};

Spherical spherical(const std::array<double, 3>& x) {
  return {std::atan2(x[1], x[0]), std::atan2(x[2], std::hypot(x[0], x[1]))};
}

// The Cartesian velocity of eastward u and northward v at (lambda, theta).
std::array<double, 3> cartesian(double u, double v, const Spherical& p) {
  const double sl = std::sin(p.lambda), cl = std::cos(p.lambda), st = std::sin(p.theta), ct = std::cos(p.theta);
  return {-sl * u - st * cl * v, cl * u - st * sl * v, ct * v};
}

// Williamson et al. (1992), case 2: steady zonal flow about the axis
// (-sin alpha, 0, cos alpha), u0 = 2 pi a / 12 days, g h0 = 2.94e4.
PointState williamson2(const std::array<double, 3>& x, double alpha) {
  const double r = std::hypot(x[0], x[1], x[2]);
  const std::array<double, 3> n = {x[0] / r, x[1] / r, x[2] / r};
  const std::array<double, 3> w = {-std::sin(alpha), 0, std::cos(alpha)};
  const double u0 = 2 * pi * radius / (12 * 86400.0), h0 = 2.94e4 / gravity;
  const double mu = w[0] * n[0] + w[1] * n[1] + w[2] * n[2];
  PointState s;
  s.h = h0 - (radius * omega * u0 + u0 * u0 / 2) * mu * mu / gravity;
  s.u = {u0 * (w[1] * n[2] - w[2] * n[1]), u0 * (w[2] * n[0] - w[0] * n[2]), u0 * (w[0] * n[1] - w[1] * n[0])};
  s.f = 2 * omega * mu;
  return s;
}

// Case 5: case 2's flow at alpha = 0 with u0 = 20, h0 = 5960, over a conical
// mountain of height 2000 m and radius pi / 9 centred at (3 pi / 2, pi / 6);
// distances in (lambda, theta) as the paper's.
PointState williamson5(const Spherical& p) {
  const double u0 = 20, h0 = 5960, hs0 = 2000, r0 = pi / 9, lc = 3 * pi / 2, tc = pi / 6;
  const double lambda = p.lambda < 0 ? p.lambda + 2 * pi : p.lambda;
  const double r = std::min(r0, std::hypot(lambda - lc, p.theta - tc));
  const double st = std::sin(p.theta);
  PointState s;
  s.hs = hs0 * (1 - r / r0);
  s.h = h0 - (radius * omega * u0 + u0 * u0 / 2) * st * st / gravity - s.hs;
  s.u = cartesian(u0 * std::cos(p.theta), 0, p);
  s.f = 2 * omega * st;
  return s;
}

// Case 6: the Rossby-Haurwitz wave of wavenumber R = 4 with omega = K =
// 7.848e-6 1/s and h0 = 8000 m, as the paper's velocity and A, B, C:
//   u = a w cos t + a K cos^(R-1) t (R sin^2 t - cos^2 t) cos R l
//   v = -a K R cos^(R-1) t sin t sin R l
//   g h = g h0 + a^2 (A + B cos R l + C cos 2 R l)
//   A = w (2 Omega + w) cos^2 t / 2 + K^2 cos^2R t ((R + 1) cos^2 t + 2 R^2 - R - 2 - 2 R^2 / cos^2 t) / 4
//   B = 2 (Omega + w) K cos^R t ((R^2 + 2 R + 2) - (R + 1)^2 cos^2 t) / ((R + 1) (R + 2))
//   C = K^2 cos^2R t ((R + 1) cos^2 t - (R + 2)) / 4
PointState williamson6(const Spherical& p) {
  const double w = 7.848e-6, k = 7.848e-6, h0 = 8000, n = 4, a = radius;
  const double c = std::cos(p.theta), st = std::sin(p.theta), c2 = c * c;
  const double cn = std::pow(c, n), c2n = cn * cn, cn1 = std::pow(c, n - 1);
  const double big_a = w * (2 * omega + w) * c2 / 2 +
                       k * k * c2n * ((n + 1) * c2 + (2 * n * n - n - 2) - 2 * n * n / c2) / 4;
  const double big_b = 2 * (omega + w) * k * cn * ((n * n + 2 * n + 2) - (n + 1) * (n + 1) * c2) / ((n + 1) * (n + 2));
  const double big_c = k * k * c2n * ((n + 1) * c2 - (n + 2)) / 4;
  PointState s;
  s.h = h0 + a * a * (big_a + big_b * std::cos(n * p.lambda) + big_c * std::cos(2 * n * p.lambda)) / gravity;
  const double u = a * w * c + a * k * cn1 * (n * st * st - c2) * std::cos(n * p.lambda);
  const double v = -a * k * n * cn1 * st * std::sin(n * p.lambda);
  s.u = cartesian(u, v, p);
  s.f = 2 * omega * st;
  return s;
}

// Galewsky et al. (2004): a zonal jet between theta0 = pi / 7 and theta1 =
// pi / 2 - theta0 peaking at 80 m/s, in balance with
//   g h(theta) = g h0 - int_{-pi/2}^{theta} a u (f + tan(t) u / a) dt,
// h0 such that the mean depth is 10 km.
double galewsky_u(double theta) {
  const double umax = 80, t0 = pi / 7, t1 = pi / 2 - t0;
  if (theta <= t0 || theta >= t1) return 0;
  const double en = std::exp(-4 / ((t1 - t0) * (t1 - t0)));
  return umax / en * std::exp(1 / ((theta - t0) * (theta - t1)));
}

double galewsky_integrand(double theta) {
  const double u = galewsky_u(theta);
  if (u == 0) return 0;
  return radius * u * (2 * omega * std::sin(theta) + std::tan(theta) * u / radius);
}

double simpson(double a, double b) {
  return (b - a) / 6 * (galewsky_integrand(a) + 4 * galewsky_integrand((a + b) / 2) + galewsky_integrand(b));
}

// The integral from -pi/2 to each of 10^4 + 1 equally spaced latitudes (by
// Simpson's rule on each interval), and g h0.
struct GalewskyTable {
  static constexpr int intervals = 10000;
  double step = pi / intervals;
  std::vector<double> integral;
  double gh0 = 0;

  GalewskyTable() : integral(intervals + 1, 0.0) {
    for (int i = 0; i < intervals; ++i) {
      const double a = -pi / 2 + i * step;
      integral[i + 1] = integral[i] + simpson(a, a + step);
    }
    // Mean of g h = g h0 - mean of the integral, area-weighted by cos(theta)
    // (Simpson's rule on the same latitudes; the integral is smooth).
    double mean = 0;
    for (int i = 0; i < intervals; ++i) {
      const double a = -pi / 2 + i * step, m = a + step / 2;
      const double mid = integral[i] + simpson(a, m);
      mean += step / 6 * (integral[i] * std::cos(a) + 4 * mid * std::cos(m) + integral[i + 1] * std::cos(a + step));
    }
    gh0 = gravity * 10000 + mean / 2;
  }

  double gh(double theta) const {
    const int i = std::min(intervals - 1, std::max(0, static_cast<int>((theta + pi / 2) / step)));
    const double a = -pi / 2 + i * step;
    return gh0 - (integral[i] + simpson(a, theta));
  }
};

const GalewskyTable& galewsky_table() {
  static const GalewskyTable table;
  return table;
}

PointState galewsky(const Spherical& p, bool perturbed) {
  PointState s;
  s.h = galewsky_table().gh(p.theta) / gravity;
  if (perturbed) {
    const double alpha = 1.0 / 3, beta = 1.0 / 15, t2 = pi / 4;
    s.h += 120 * std::cos(p.theta) * std::exp(-(p.lambda / alpha) * (p.lambda / alpha)) *
           std::exp(-((t2 - p.theta) / beta) * ((t2 - p.theta) / beta));
  }
  s.u = cartesian(galewsky_u(p.theta), 0, p);
  s.f = 2 * omega * std::sin(p.theta);
  return s;
}

constexpr std::array<std::string_view, 5> names = {"w2", "w5", "w6", "galewsky", "galewsky-steady"};

}  // namespace

Case case_from_name(std::string_view name) {
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (name == names[i]) return static_cast<Case>(i);
  }
  throw std::invalid_argument("swe: unknown case '" + std::string(name) +
                              "'; the cases are w2, w5, w6, galewsky, galewsky-steady");
}

std::string_view case_name(Case c) { return names.at(static_cast<std::size_t>(c)); }

PointState initial(Case c, const std::array<double, 3>& x, double alpha) {
  switch (c) {
    case Case::williamson2:
      return williamson2(x, alpha);
    case Case::williamson5:
      return williamson5(spherical(x));
    case Case::williamson6:
      return williamson6(spherical(x));
    case Case::galewsky:
      return galewsky(spherical(x), true);
    case Case::galewsky_steady:
      return galewsky(spherical(x), false);
  }
  throw std::invalid_argument("swe: unknown case");
}

PointState exact(Case c, const std::array<double, 3>& x, double alpha, double /*t*/) {
  if (c != Case::williamson2) {
    throw std::invalid_argument("swe: case " + std::string(case_name(c)) + " has no exact solution");
  }
  return williamson2(x, alpha);
}

}  // namespace swe
