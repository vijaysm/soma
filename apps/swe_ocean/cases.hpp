// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_CASES_HPP_
#define SWE_OCEAN_CASES_HPP_

// The analytic cases of the barotropic ocean, in the plane, with a flat bottom, in the linear mode without
// hyperviscosity (nu = 0), quadratic drag or tracers, on nx by ny quads of soma::testing::box_mesh:
//
// Stommel (1948): a square basin [0, L] x [0, L] on the beta plane f = f0 + beta (y - L / 2), under the wind
// stress tau_x = -tau0 cos(pi y / L), tau_y = 0, with linear drag r. Started from rest and run until the relative
// l2 change of u over an interval of 1 / r is below `tolerance`; then compared with the rigid-lid steady state
//   psi = gamma (L / pi)^2 sin(pi y / L) X(x),  u = -psi_y,  v = psi_x,  gamma = pi tau0 / (rho0 H r L),
//   X = 1 + P e^(m+ x) + Q e^(m- x),  m+- = -alpha / 2 +- sqrt(alpha^2 / 4 + pi^2 / L^2),  alpha = beta / r,
// with P + Q = -1 and P e^(m+ L) + Q e^(m- L) = -1 (X = 0 at both walls). The free surface's steady state has
// div(H u) = 0 with H flat, so the rigid-lid velocity is its exact velocity. The model's equations are du/dt + f
// k x u = -g grad eta + tau / (rho0 H) - r u with k = +z (the plane normal a_1 x a_2), those of the solution.
//
// Kelvin: a channel [0, lx] x [0, ly] on the f-plane, the wall y = 0 on the right of the propagation (f > 0):
//   eta = A exp(-y / R) exp(-((x - x0 - c t) / sigma)^2),  u = (g / c) eta,  v = 0,  c = sqrt(g H),  R = c / f,
// exact in the closed channel while the packet is far from the ends (v = 0 at both side walls). Run from t = 0 to
// distance / c and compared in eta.
//
// Errors are relative: error_l2 the l2 norm (the space's integral) of the difference over that of the analytic
// field (the velocity vector for Stommel, eta for Kelvin), error_max the largest pointwise difference (vector
// magnitude for Stommel) over the largest analytic value; both at the GLL points, the same bits on any number of
// ranks. digest is the model's state digest at the end (OceanModel::digest); `finish`, when given, sees the model
// at the end (the driver writes its output there).

#include <cstdint>
#include <functional>

#include "soma/parallel/comm.hpp"

namespace ocean {

class OceanModel;

struct CaseResult {
  double error_l2 = 0, error_max = 0;
  long steps = 0;
  std::uint64_t digest = 0;
};

using CaseFinish = std::function<void(OceanModel&)>;

// L (m), beta (1/(m s)), f0 (1/s), H (m), r (1/s), tau0 (N/m^2); tolerance: the steady-state test above;
// max_steps: the steps after which a run not yet steady throws (about 4 times what nx = 12, order 4 takes);
// cfl: the model's. The steady state does not depend on the step, so the cfl is set for speed: 0.6, below the
// limit measured at order 4 (stable at 0.8 at nx = 6 and 12, blown up at 0.9 within 400 steps).
struct StommelSpec {
  double L = 1e6, beta = 2e-11, f0 = 1e-4, H = 100, r = 2e-6, tau0 = 0.1;
  double tolerance = 1e-7;
  long max_steps = 200000;
  double cfl = 0.6;
};

// lx, ly (m): the channel; f (1/s), H (m); amplitude (m), x0, sigma (m): the packet at t = 0; distance (m): how
// far it travels; cfl: the model's, 0.1 so that the time error (third order, about 3e-7 of the packet at order 6
// on 40 x 10 elements, 2.5e-6 at cfl 0.2) stays below the spatial error of order 4 and near that of order 6.
struct KelvinSpec {
  double lx = 4e6, ly = 1e6, f = 1e-4, H = 100;
  double amplitude = 1, x0 = 1e6, sigma = 2e5, distance = 2e6;
  double cfl = 0.1;
};

// Collective: the Stommel case on nx by nx elements of the order. Throws std::invalid_argument on every rank for
// nx < 1, order < 1 or a spec value not positive and finite (tolerance, max_steps and every length, rate and
// depth; tau0 may be any finite value); std::runtime_error on every rank when the run is not steady after
// max_steps steps; what OceanModel throws.
CaseResult run_stommel(const soma::Comm& comm, int nx, int order, const StommelSpec& spec,
                       const CaseFinish& finish = {});

// Collective: the Kelvin case on nx by max(1, round(nx ly / lx)) elements of the order. Throws
// std::invalid_argument on every rank for nx < 1, order < 1, a spec value not finite, or a length, f, H, sigma or
// cfl not positive (amplitude, x0 and distance may be any finite value); what OceanModel throws.
CaseResult run_kelvin(const soma::Comm& comm, int nx, int order, const KelvinSpec& spec,
                      const CaseFinish& finish = {});

}  // namespace ocean

#endif  // SWE_OCEAN_CASES_HPP_
