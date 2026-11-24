// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "cases.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

#include "constants.hpp"
#include "model.hpp"
#include "ocean_mesh.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/topology.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/testing/box_mesh.hpp"

namespace ocean {

namespace {

using Point = std::array<double, 3>;

// A plane box [0, lx] x [0, ly] of nx by ny quads with a space of the order and depth H everywhere: a ghost layer
// and distributed edges on more than one rank (as the SEM tests' boxes).
OceanMesh flat_box(const soma::Comm& comm, int nx, int ny, double lx, double ly, int order, double H) {
  soma::testing::BoxSpec b;
  b.nx = nx;
  b.ny = ny;
  b.element = soma::Topology::quad;
  b.lx = lx;
  b.ly = ly;
  auto mesh = std::make_unique<soma::HostMesh>(soma::testing::box_mesh(comm, b));
  if (comm.size() > 1) {
    mesh->add_ghosts(comm, soma::GhostOptions{1, 0});
    soma::build_distributed_adjacencies(comm, *mesh, soma::AdjacencyRequest{}.add(2, 1));
  }
  OceanMesh o;
  o.mesh = std::move(mesh);
  o.space = std::make_unique<Space>(comm, *o.mesh, order, soma::sem::SemGeometry::plane());
  const auto n = static_cast<std::size_t>(o.space->num_elements() * o.space->points_per_element());
  o.bathymetry.H = Values("H", n, 1);
  for (std::size_t r = 0; r < n; ++r) o.bathymetry.H(r, 0) = H;
  return o;
}

// The linear mode without hyperviscosity, drag or tracers.
ModelOptions linear_options(double cfl) {
  ModelOptions o;
  o.linear = true;
  o.nu = 0;
  o.forcing.drag_quadratic = 0;
  o.cfl = cfl;
  o.tracers = 0;
  return o;
}

void check_size(int nx, int order, const char* name) {
  if (nx < 1 || order < 1) {
    throw std::invalid_argument(std::string(name) + ": nx and order must be at least 1 (got " + std::to_string(nx) +
                                ", " + std::to_string(order) + ")");
  }
}

void check_positive(double v, const char* what, const char* name) {
  if (!(std::isfinite(v) && v > 0)) {
    throw std::invalid_argument(std::string(name) + ": " + what + " must be positive and finite (got " +
                                std::to_string(v) + ")");
  }
}

void check_finite(double v, const char* what, const char* name) {
  if (!std::isfinite(v)) throw std::invalid_argument(std::string(name) + ": " + what + " must be finite");
}

// The owned rows of a space.
std::size_t owned_rows(const Space& s) {
  return static_cast<std::size_t>(s.owned_elements()) * static_cast<std::size_t>(s.points_per_element());
}

// Relative l2 and max errors of `components` components of `field` against `exact` (each a row per point of the
// space), as cases.hpp defines them.
CaseResult relative_errors(const soma::Comm& comm, const Space& s, ConstValues field, ConstValues exact,
                           int components) {
  const std::size_t n = field.extent(0), owned = owned_rows(s);
  Values diff2("diff2", n, 1), exact2("exact2", n, 1);
  double diff_max = 0, exact_max = 0;
  for (std::size_t r = 0; r < n; ++r) {
    double d = 0, e = 0;
    for (int k = 0; k < components; ++k) {
      const auto c = static_cast<std::size_t>(k);
      d += (field(r, c) - exact(r, c)) * (field(r, c) - exact(r, c));
      e += exact(r, c) * exact(r, c);
    }
    diff2(r, 0) = d;
    exact2(r, 0) = e;
    if (r < owned) {
      diff_max = std::max(diff_max, std::sqrt(d));
      exact_max = std::max(exact_max, std::sqrt(e));
    }
  }
  CaseResult result;
  result.error_l2 = std::sqrt(s.integral(comm, diff2) / s.integral(comm, exact2));
  result.error_max = comm.allreduce(diff_max, soma::Op::max) / comm.allreduce(exact_max, soma::Op::max);
  return result;
}

}  // namespace

CaseResult run_stommel(const soma::Comm& comm, int nx, int order, const StommelSpec& spec) {
  constexpr const char* name = "run_stommel";
  check_size(nx, order, name);
  check_positive(spec.L, "L", name);
  check_positive(spec.beta, "beta", name);
  check_positive(spec.f0, "f0", name);
  check_positive(spec.H, "H", name);
  check_positive(spec.r, "r", name);
  check_finite(spec.tau0, "tau0", name);
  check_positive(spec.tolerance, "tolerance", name);
  check_positive(static_cast<double>(spec.max_steps), "max_steps", name);
  check_positive(spec.cfl, "cfl", name);

  const double L = spec.L, pi = std::numbers::pi;
  ModelOptions o = linear_options(spec.cfl);
  const double f0 = spec.f0, beta = spec.beta, tau0 = spec.tau0;
  o.coriolis = [f0, beta, L](const Point& x) { return f0 + beta * (x[1] - L / 2); };
  o.forcing.wind = [tau0, L](const Space& s, Values tau) {
    const auto x = s.x();
    for (std::size_t r = 0; r < tau.extent(0); ++r) {
      tau(r, 0) = -tau0 * std::cos(std::numbers::pi * x(r, 1) / L);
      tau(r, 1) = tau(r, 2) = 0;
    }
  };
  o.forcing.drag_linear = spec.r;
  OceanModel model(comm, flat_box(comm, nx, nx, L, L, order, spec.H), o);

  // To steady state: the relative l2 change of u over each interval of 1 / r (0 / 0, the basin at rest under no
  // wind, counts as steady).
  const Space& s = model.space();
  const std::size_t n = model.u().extent(0);
  Values previous("u_previous", n, 3);
  const double interval = 1 / spec.r;
  for (;;) {
    Kokkos::deep_copy(previous, model.u());
    model.advance(comm, model.time() + interval);
    const double change = relative_errors(comm, s, previous, model.u(), 3).error_l2;
    if (!(change >= spec.tolerance)) break;
    if (model.steps() >= spec.max_steps) {
      throw std::runtime_error("run_stommel: not steady after " + std::to_string(model.steps()) + " steps, t = " +
                               std::to_string(model.time()) + " s (relative change " + std::to_string(change) +
                               " over 1 / r)");
    }
  }

  // The analytic velocity at the GLL points.
  const double alpha = beta / spec.r, root = std::sqrt(alpha * alpha / 4 + pi * pi / (L * L));
  const double mp = -alpha / 2 + root, mm = -alpha / 2 - root;
  const double P = (std::exp(mm * L) - 1) / (std::exp(mp * L) - std::exp(mm * L)), Q = -1 - P;
  const double gamma = pi * tau0 / (rho0 * spec.H * spec.r * L);
  const auto x = s.x();
  Values exact("u_exact", n, 3);
  for (std::size_t r = 0; r < n; ++r) {
    const double X = 1 + P * std::exp(mp * x(r, 0)) + Q * std::exp(mm * x(r, 0));
    const double dX = P * mp * std::exp(mp * x(r, 0)) + Q * mm * std::exp(mm * x(r, 0));
    exact(r, 0) = -gamma * (L / pi) * std::cos(pi * x(r, 1) / L) * X;
    exact(r, 1) = gamma * (L / pi) * (L / pi) * std::sin(pi * x(r, 1) / L) * dX;
    exact(r, 2) = 0;
  }
  CaseResult result = relative_errors(comm, s, model.u(), exact, 3);
  result.steps = model.steps();
  return result;
}

CaseResult run_kelvin(const soma::Comm& comm, int nx, int order, const KelvinSpec& spec) {
  constexpr const char* name = "run_kelvin";
  check_size(nx, order, name);
  check_positive(spec.lx, "lx", name);
  check_positive(spec.ly, "ly", name);
  check_positive(spec.f, "f", name);
  check_positive(spec.H, "H", name);
  check_positive(spec.sigma, "sigma", name);
  check_positive(spec.cfl, "cfl", name);
  check_finite(spec.amplitude, "amplitude", name);
  check_finite(spec.x0, "x0", name);
  check_finite(spec.distance, "distance", name);

  const double c = std::sqrt(g * spec.H), R = c / spec.f;
  const KelvinSpec k = spec;
  // The packet at time t.
  const auto packet = [k, c, R](const Point& x, double t) {
    const double s = (x[0] - k.x0 - c * t) / k.sigma;
    return k.amplitude * std::exp(-x[1] / R) * std::exp(-s * s);
  };
  const int ny = std::max(1, static_cast<int>(std::lround(nx * spec.ly / spec.lx)));
  ModelOptions o = linear_options(spec.cfl);
  const double f = spec.f;
  o.coriolis = [f](const Point&) { return f; };
  OceanModel model(comm, flat_box(comm, nx, ny, spec.lx, spec.ly, order, spec.H), o);
  model.initialize(
      comm, [&packet](const Point& x) { return packet(x, 0); },
      [&packet, c](const Point& x) { return Point{g / c * packet(x, 0), 0, 0}; });
  model.advance(comm, spec.distance / c);

  const double t = model.time();
  const auto x = model.space().x();
  Values exact("eta_exact", x.extent(0), 1);
  for (std::size_t r = 0; r < x.extent(0); ++r) exact(r, 0) = packet({x(r, 0), x(r, 1), x(r, 2)}, t);
  CaseResult result = relative_errors(comm, model.space(), model.eta(), exact, 1);
  result.steps = model.steps();
  return result;
}

}  // namespace ocean
