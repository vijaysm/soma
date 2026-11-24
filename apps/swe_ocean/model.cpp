// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/sem/operators.hpp"

namespace ocean {

namespace {

using soma::LO;
using Policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::IndexType<LO>>;

// out = wx x + wy (y + dt z), every row and column.
void combine(double wx, const Values& x, double wy, const Values& y, double dt, const Values& z, const Values& out) {
  const auto nc = static_cast<int>(x.extent(1));
  Kokkos::parallel_for(
      "ocean::combine", Policy(0, static_cast<LO>(x.extent(0))), KOKKOS_LAMBDA(const LO r) {
        for (int c = 0; c < nc; ++c) out(r, c) = wx * x(r, c) + wy * (y(r, c) + dt * z(r, c));
      });
  Kokkos::fence();
}

// out -= nu v.
void subtract(double nu, const Values& v, const Values& out) {
  const auto nc = static_cast<int>(v.extent(1));
  Kokkos::parallel_for(
      "ocean::subtract", Policy(0, static_cast<LO>(v.extent(0))), KOKKOS_LAMBDA(const LO r) {
        for (int c = 0; c < nc; ++c) out(r, c) -= nu * v(r, c);
      });
  Kokkos::fence();
}

// The spectral radius of the discrete del^4 (scalar and vector, the larger): 40 power iterations from a field
// set by element gid and point, made continuous; Rayleigh quotients in the mass inner product (exact sums), so
// the same on any number of ranks (swe_atm's).
double hyperviscous_radius(const soma::Comm& comm, const Space& s, const soma::HostMesh& mesh) {
  const auto ppe = s.points_per_element();
  const auto n = static_cast<std::size_t>(s.num_elements() * ppe);
  const auto gid = mesh.gid(2);
  const auto x = s.x();
  // swe_atm's small smooth term (z over the radius; 0 in the plane, where z = 0) beside the gid-seeded noise.
  const double scale = s.geometry().kind == soma::sem::SemGeometry::Kind::sphere ? s.geometry().radius : 1.0;
  double radius = 0;
  for (const int nc : {1, 3}) {
    Values v("ocean::power", n, static_cast<std::size_t>(nc)), w("ocean::power_image", n, static_cast<std::size_t>(nc));
    Values product("ocean::power_product", n, 2);
    for (std::size_t r = 0; r < n; ++r) {
      const double seed = static_cast<double>(gid(static_cast<LO>(r) / ppe)) * 0.618033988749 +
                          static_cast<double>(static_cast<LO>(r) % ppe) * 0.414213562373;
      for (int c = 0; c < nc; ++c) v(r, c) = std::sin(97.0 * seed + 13.0 * c) + 1e-3 * x(r, 2) / scale;
    }
    s.dss(comm, v);
    if (nc == 3) soma::sem::project_tangent(s, v);
    double lambda = 0;
    for (int it = 0; it < 40; ++it) {
      if (nc == 1) {
        soma::sem::hyperviscosity(comm, s, v, w);
      } else {
        soma::sem::vector_hyperviscosity(comm, s, v, w);
      }
      for (std::size_t r = 0; r < n; ++r) {
        product(r, 0) = 0;
        product(r, 1) = 0;
        for (int c = 0; c < nc; ++c) {
          product(r, 0) += v(r, c) * w(r, c);
          product(r, 1) += w(r, c) * w(r, c);
        }
      }
      const double vw = s.integral(comm, product, 0), ww = s.integral(comm, product, 1);
      if (!(ww > 0)) break;
      lambda = std::abs(vw);
      const double scale = 1 / std::sqrt(ww);
      for (std::size_t r = 0; r < n; ++r) {
        for (int c = 0; c < nc; ++c) v(r, c) = w(r, c) * scale;
      }
    }
    radius = std::max(radius, lambda);
  }
  return radius;
}

// The shortest element edge over all ranks: between consecutive GLL corner points of each owned element,
// great-circle on the sphere, straight in the plane. Exact (a min reduction).
double shortest_edge(const soma::Comm& comm, const Space& s) {
  const int p = s.order(), np = p + 1;
  const LO ppe = s.points_per_element();
  const LO corner[4] = {0, p, ppe - 1, p * np};
  const auto x = s.x();
  const bool sphere = s.geometry().kind == soma::sem::SemGeometry::Kind::sphere;
  double least = std::numeric_limits<double>::infinity();
  for (LO e = 0; e < s.owned_elements(); ++e) {
    for (int q = 0; q < 4; ++q) {
      const LO r0 = e * ppe + corner[q], r1 = e * ppe + corner[(q + 1) % 4];
      double length = std::hypot(x(r1, 0) - x(r0, 0), x(r1, 1) - x(r0, 1), x(r1, 2) - x(r0, 2));
      if (sphere) {
        const double c0 = x(r0, 1) * x(r1, 2) - x(r0, 2) * x(r1, 1), c1 = x(r0, 2) * x(r1, 0) - x(r0, 0) * x(r1, 2),
                     c2 = x(r0, 0) * x(r1, 1) - x(r0, 1) * x(r1, 0);
        const double dot = x(r0, 0) * x(r1, 0) + x(r0, 1) * x(r1, 1) + x(r0, 2) * x(r1, 2);
        length = s.geometry().radius * std::atan2(std::hypot(c0, c1, c2), dot);
      }
      least = std::min(least, length);
    }
  }
  return comm.allreduce(least, soma::Op::min);
}

void check(const ModelOptions& o) {
  const auto text = [](double v) {
    std::ostringstream s;
    s << v;
    return s.str();
  };
  std::string bad;
  if (!(o.cfl > 0 && std::isfinite(o.cfl))) bad = "cfl " + text(o.cfl) + " not positive";
  if (o.dt && !(*o.dt > 0 && std::isfinite(*o.dt))) bad = "dt " + text(*o.dt) + " not positive";
  if (o.nu && !(*o.nu >= 0 && std::isfinite(*o.nu))) bad = "nu " + text(*o.nu) + " negative";
  if (!(o.forcing.drag_quadratic >= 0 && std::isfinite(o.forcing.drag_quadratic))) {
    bad = "quadratic drag " + text(o.forcing.drag_quadratic) + " negative";
  }
  if (o.forcing.drag_linear && !(*o.forcing.drag_linear >= 0 && std::isfinite(*o.forcing.drag_linear))) {
    bad = "linear drag " + text(*o.forcing.drag_linear) + " negative";
  }
  if (o.nu_tracer && !(*o.nu_tracer >= 0 && std::isfinite(*o.nu_tracer))) {
    bad = "tracer viscosity " + text(*o.nu_tracer) + " negative";
  }
  if (o.tracers < 0) bad = std::to_string(o.tracers) + " tracers";
  if (o.vertical.levels < 1) bad = std::to_string(o.vertical.levels) + " levels";
  if (!bad.empty()) throw std::invalid_argument("swe_ocean: " + bad);
}

void check(const OceanMesh& m) {
  if (!m.mesh || !m.space) throw std::invalid_argument("swe_ocean: the ocean mesh has no mesh or no space");
  const auto rows = static_cast<std::size_t>(m.space->num_elements() * m.space->points_per_element());
  if (m.bathymetry.H.extent(0) != rows || m.bathymetry.H.extent(1) != 1) {
    throw std::invalid_argument("swe_ocean: H is " + std::to_string(m.bathymetry.H.extent(0)) + " x " +
                                std::to_string(m.bathymetry.H.extent(1)) + ", not " + std::to_string(rows) + " x 1");
  }
}

}  // namespace

struct OceanModel::State {
  ModelOptions options;
  OceanMesh ocean;
  soma::sem::SemBoundary<Host> boundary;
  Values H, eta, u, f, tau;
  // The step's start, a stage, the tendencies and scratch.
  Values eta0, u0, eta1, u1, deta, du, zeta, phi, lap, grad, nxu, flux, div, hyper_u;
  double time = 0, dt = 0, nu = 0, rho = 0;
  long steps = 0;
  // The tracers, and their state (Tracers::state) at the step's start, at a stage and its explicit tendency.
  std::unique_ptr<Tracers> tracers;
  Values hT0, hT1, dhT;

  State(const soma::Comm& comm, OceanMesh&& mesh, const ModelOptions& o)
      : options(o), ocean(std::move(mesh)), boundary(soma::sem::make_boundary(comm, *ocean.mesh, *ocean.space)) {
    const Space& s = *ocean.space;
    const auto n = static_cast<std::size_t>(s.num_elements() * s.points_per_element());
    const auto make = [n](const char* label, std::size_t c) { return Values(label, n, c); };
    H = ocean.bathymetry.H;
    eta = make("ocean::eta", 1), u = make("ocean::u", 3), f = make("ocean::f", 1), tau = make("ocean::tau", 3);
    eta0 = make("ocean::eta0", 1), u0 = make("ocean::u0", 3), eta1 = make("ocean::eta1", 1);
    u1 = make("ocean::u1", 3), deta = make("ocean::deta", 1), du = make("ocean::du", 3);
    zeta = make("ocean::zeta", 1), phi = make("ocean::phi", 1), lap = make("ocean::lap", 1);
    grad = make("ocean::grad", 3), nxu = make("ocean::nxu", 3), flux = make("ocean::flux", 3);
    div = make("ocean::div", 1), hyper_u = make("ocean::hyper_u", 3);

    // f and tau, continuous (shared points' positions differ by rounding between elements); tau tangent.
    const auto x = s.x();
    for (std::size_t r = 0; r < n; ++r) {
      const std::array<double, 3> p{x(r, 0), x(r, 1), x(r, 2)};
      if (o.coriolis) {
        f(r, 0) = o.coriolis(p);
      } else {
        const double len = std::hypot(p[0], p[1], p[2]);
        f(r, 0) = len > 0 ? 2 * Omega * p[2] / len : 0.0;
      }
    }
    s.dss(comm, f);
    if (o.forcing.wind) {
      o.forcing.wind(s, tau);
      s.dss(comm, tau);
      soma::sem::project_tangent(s, tau);
    }

    if (o.nu) {
      nu = *o.nu;
    } else if (!o.linear) {
      const double dx_30 = std::numbers::pi * a / 60;
      nu = 1.0e15 * std::pow(shortest_edge(comm, s) / dx_30, 3.2);
    }
    if (nu > 0 && !o.dt) rho = hyperviscous_radius(comm, s, *ocean.mesh);
    pick_dt(comm);

    tracers = std::make_unique<Tracers>(s, H, o.vertical, o.tracers, o.nu_tracer.value_or(nu));
    const auto nt = static_cast<std::size_t>(o.vertical.levels * o.tracers);
    hT0 = make("ocean::hT0", nt), hT1 = make("ocean::hT1", nt), dhT = make("ocean::dhT", nt);
  }

  // The step as the header says, from the state now.
  void pick_dt(const soma::Comm& comm) {
    if (options.dt) {
      dt = *options.dt;
      return;
    }
    const Space& s = *ocean.space;
    const int np = s.order() + 1;
    const LO ppe = s.points_per_element();
    const auto x = s.x();
    const auto distance = [&](LO r, LO q) {
      return std::hypot(x(q, 0) - x(r, 0), x(q, 1) - x(r, 1), x(q, 2) - x(r, 2));
    };
    double least = std::numeric_limits<double>::infinity();
    for (LO e = 0; e < s.owned_elements(); ++e) {
      for (LO k = 0; k < ppe; ++k) {
        const LO r = e * ppe + k, i = k % np, j = k / np;
        double gap = std::numeric_limits<double>::infinity();
        if (i > 0) gap = std::min(gap, distance(r, r - 1));
        if (i + 1 < np) gap = std::min(gap, distance(r, r + 1));
        if (j > 0) gap = std::min(gap, distance(r, r - np));
        if (j + 1 < np) gap = std::min(gap, distance(r, r + np));
        const double depth = std::max(H(r, 0) + eta(r, 0), 0.0);
        const double speed = std::hypot(u(r, 0), u(r, 1), u(r, 2)) + std::sqrt(g * depth);
        if (speed > 0) least = std::min(least, gap / speed);
      }
    }
    const double advective = options.cfl * comm.allreduce(least, soma::Op::min);
    const double hyperviscous = nu > 0 ? 2.5 * options.cfl / (nu * rho) : std::numeric_limits<double>::infinity();
    dt = std::min(advective, hyperviscous);
  }

  // deta, du and the tracers' explicit tendency dhT from (ee, uu) and their state hT.
  void tendencies(const soma::Comm& comm, const Values& ee, const Values& uu, const Values& hT) {
    const Space& s = *ocean.space;
    const bool linear = options.linear;
    // The flux D u + nu grad(L eta).
    {
      const auto h = H;
      const auto fl = flux;
      Kokkos::parallel_for(
          "ocean::flux", Policy(0, static_cast<LO>(ee.extent(0))), KOKKOS_LAMBDA(const LO r) {
            const double depth = linear ? h(r, 0) : h(r, 0) + ee(r, 0);
            for (int c = 0; c < 3; ++c) fl(r, c) = depth * uu(r, c);
          });
      Kokkos::fence();
    }
    if (nu > 0) {
      soma::sem::laplacian(comm, s, ee, lap);
      soma::sem::gradient(s, lap, grad);
      subtract(-nu, grad, flux);
    }
    soma::sem::divergence_weak(comm, s, flux, div);

    if (!linear) {
      soma::sem::vorticity(s, uu, zeta);
      s.dss(comm, zeta);
    }
    {
      const auto ph = phi;
      Kokkos::parallel_for(
          "ocean::potential", Policy(0, static_cast<LO>(ee.extent(0))), KOKKOS_LAMBDA(const LO r) {
            const double speed2 = uu(r, 0) * uu(r, 0) + uu(r, 1) * uu(r, 1) + uu(r, 2) * uu(r, 2);
            ph(r, 0) = g * ee(r, 0) + (linear ? 0.0 : 0.5 * speed2);
          });
      Kokkos::fence();
    }
    soma::sem::gradient(s, phi, grad);
    soma::sem::cross_normal(s, uu, nxu);
    {
      const auto h = H, ff = f, ze = zeta, nx = nxu, gr = grad, ta = tau, dv = div, de = deta, dd = du;
      const bool has_linear_drag = options.forcing.drag_linear.has_value();
      const double r_drag = has_linear_drag ? *options.forcing.drag_linear : 0.0;
      const double c_drag = options.forcing.drag_quadratic;
      Kokkos::parallel_for(
          "ocean::tendencies", Policy(0, static_cast<LO>(ee.extent(0))), KOKKOS_LAMBDA(const LO r) {
            const double depth = linear ? h(r, 0) : h(r, 0) + ee(r, 0);
            const double q = (linear ? 0.0 : ze(r, 0)) + ff(r, 0);
            const double speed = Kokkos::sqrt(uu(r, 0) * uu(r, 0) + uu(r, 1) * uu(r, 1) + uu(r, 2) * uu(r, 2));
            const double drag = has_linear_drag ? r_drag : c_drag * speed / depth;
            for (int c = 0; c < 3; ++c) {
              dd(r, c) = -q * nx(r, c) - gr(r, c) + ta(r, c) / (rho0 * depth) - drag * uu(r, c);
            }
            de(r, 0) = -dv(r, 0);
          });
      Kokkos::fence();
    }
    s.dss(comm, du);
    if (nu > 0) {
      soma::sem::vector_hyperviscosity(comm, s, uu, hyper_u);
      subtract(nu, hyper_u, du);
    }
    soma::sem::project_tangent(s, du);
    // The tracers, with the flux and d eta/dt of this stage.
    if (tracers->count() > 0) tracers->tendency(comm, ee, flux, deta, hT, dhT);
  }

  // No normal flow: remove_normal twice. One pass leaves a normal component of rounding size relative to the
  // velocity before it, which is much larger than the velocity kept where the flow runs into the coast; the
  // second leaves one of rounding size relative to the velocity kept.
  void slip(const Values& v) const {
    soma::sem::remove_normal(boundary, v);
    soma::sem::remove_normal(boundary, v);
  }

  // SSP-RK3 (Shu and Osher): y1 = y + dt F(y); y2 = 3/4 y + 1/4 (y1 + dt F(y1)); y = 1/3 y + 2/3 (y2 + dt
  // F(y2)); u loses its normal component at the coast after each (slip). The tracers' state is combined as eta
  // is and then takes the stage's vertical fluxes implicitly, weighted as the stage weights its tendency
  // (Tracers::vertical); h and hT follow eta at the end (Tracers::thicken).
  void step(const soma::Comm& comm, double k) {
    const Values hT = tracers->state();
    Kokkos::deep_copy(eta0, eta);
    Kokkos::deep_copy(u0, u);
    Kokkos::deep_copy(hT0, hT);
    tendencies(comm, eta0, u0, hT0);
    combine(0, eta0, 1, eta0, k, deta, eta1);
    combine(0, u0, 1, u0, k, du, u1);
    combine(0, hT0, 1, hT0, k, dhT, hT1);
    tracers->vertical(eta1, k, hT1);
    slip(u1);
    tendencies(comm, eta1, u1, hT1);
    combine(0.75, eta0, 0.25, eta1, k, deta, eta1);
    combine(0.75, u0, 0.25, u1, k, du, u1);
    combine(0.75, hT0, 0.25, hT1, k, dhT, hT1);
    tracers->vertical(eta1, 0.25 * k, hT1);
    slip(u1);
    tendencies(comm, eta1, u1, hT1);
    combine(1.0 / 3, eta0, 2.0 / 3, eta1, k, deta, eta);
    combine(1.0 / 3, u0, 2.0 / 3, u1, k, du, u);
    combine(1.0 / 3, hT0, 2.0 / 3, hT1, k, dhT, hT);
    tracers->vertical(eta, 2.0 / 3 * k, hT);
    slip(u);
    tracers->thicken(eta);
  }

  bool healthy() const {
    const auto owned = static_cast<std::size_t>(ocean.space->owned_elements() * ocean.space->points_per_element());
    for (std::size_t r = 0; r < owned; ++r) {
      if (!(H(r, 0) + eta(r, 0) > 0) || !std::isfinite(eta(r, 0))) return false;
      for (std::size_t c = 0; c < 3; ++c) {
        if (!std::isfinite(u(r, c))) return false;
      }
      const auto hT = tracers->hT();
      for (std::size_t c = 0; c < hT.extent(1); ++c) {
        if (!std::isfinite(hT(r, c))) return false;
      }
    }
    return true;
  }
};

OceanModel::OceanModel(const soma::Comm& comm, OceanMesh&& mesh, const ModelOptions& options) {
  check(options);
  check(mesh);
  s_ = std::make_unique<State>(comm, std::move(mesh), options);
}

OceanModel::~OceanModel() = default;

void OceanModel::initialize(const soma::Comm& comm,
                            const std::function<double(const std::array<double, 3>&)>& eta0) {
  State& s = *s_;
  if (s.steps > 0) throw std::logic_error("swe_ocean: initialize after " + std::to_string(s.steps) + " steps");
  const auto x = s.ocean.space->x();
  for (std::size_t r = 0; r < s.eta.extent(0); ++r) s.eta(r, 0) = eta0({x(r, 0), x(r, 1), x(r, 2)});
  s.ocean.space->dss(comm, s.eta);
  s.tracers->thicken(s.eta, true);
  s.pick_dt(comm);
}

namespace {

// The step from `time` that lands on t_end: shortened, or a hair longer rather than leaving a sliver of a step.
bool lands(double time, double t_end, double dt) { return t_end - time <= dt * (1 + 1e-9); }

}  // namespace

void OceanModel::take_step(const soma::Comm& comm, double length, double time_after) {
  State& s = *s_;
  s.step(comm, length);
  s.time = time_after;
  ++s.steps;
  const int sick = s.healthy() ? 0 : 1;
  if (comm.allreduce(sick, soma::Op::max) != 0) {
    std::ostringstream message;
    message << "swe_ocean: blew up at step " << s.steps << ", t = " << s.time << " s";
    throw std::runtime_error(message.str());
  }
}

void OceanModel::advance(const soma::Comm& comm, double t_end) {
  State& s = *s_;
  while (s.time < t_end) {
    if (lands(s.time, t_end, s.dt)) {
      take_step(comm, t_end - s.time, t_end);
    } else {
      take_step(comm, s.dt, s.time + s.dt);
    }
  }
}

void OceanModel::advance_whole_steps(const soma::Comm& comm, double t, double t_end) {
  State& s = *s_;
  while (t - s.time >= s.dt && !lands(s.time, t_end, s.dt)) take_step(comm, s.dt, s.time + s.dt);
}

double OceanModel::time() const noexcept { return s_->time; }
double OceanModel::dt() const noexcept { return s_->dt; }
double OceanModel::nu() const noexcept { return s_->nu; }
long OceanModel::steps() const noexcept { return s_->steps; }
const ModelOptions& OceanModel::options() const noexcept { return s_->options; }
const soma::HostMesh& OceanModel::mesh() const noexcept { return *s_->ocean.mesh; }
const Space& OceanModel::space() const noexcept { return *s_->ocean.space; }
const soma::sem::SemBoundary<Host>& OceanModel::boundary() const noexcept { return s_->boundary; }
ConstValues OceanModel::eta() const { return s_->eta; }
ConstValues OceanModel::u() const { return s_->u; }
ConstValues OceanModel::H() const { return s_->H; }
ConstValues OceanModel::f() const { return s_->f; }
Tracers& OceanModel::tracers() noexcept { return *s_->tracers; }
const Tracers& OceanModel::tracers() const noexcept { return *s_->tracers; }

double OceanModel::mass(const soma::Comm& comm) const { return s_->ocean.space->integral(comm, s_->eta, 0); }

std::uint64_t OceanModel::digest(const soma::Comm& comm) const {
  const State& s = *s_;
  const auto ppe = static_cast<std::size_t>(s.ocean.space->points_per_element());
  const auto hT = s.tracers->hT();
  const std::size_t width = 1 + (4 + hT.extent(1)) * ppe;
  const auto gid = s.ocean.mesh->gid(2);
  std::vector<std::uint64_t> mine;
  for (LO e = 0; e < s.ocean.space->owned_elements(); ++e) {
    mine.push_back(static_cast<std::uint64_t>(gid(e)));
    for (std::size_t k = 0; k < ppe; ++k) {
      const std::size_t r = static_cast<std::size_t>(e) * ppe + k;
      const auto push = [&mine](double v) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof bits);
        mine.push_back(bits);
      };
      for (const double v : {s.eta(r, 0), s.u(r, 0), s.u(r, 1), s.u(r, 2)}) push(v);
      for (std::size_t c = 0; c < hT.extent(1); ++c) push(hT(r, c));
    }
  }
  const std::vector<std::uint64_t> all = comm.allgatherv(std::span<const std::uint64_t>(mine));
  std::vector<std::size_t> order(all.size() / width);
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) { return all[i * width] < all[j * width]; });
  std::uint64_t hash = 1469598103934665603ull;
  for (const std::size_t i : order) {
    for (std::size_t j = 0; j < width; ++j) hash = (hash ^ all[i * width + j]) * 1099511628211ull;
  }
  return hash;
}

}  // namespace ocean
