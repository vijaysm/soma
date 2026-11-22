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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "constants.hpp"
#include "soma/fields/node_tables.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/sem/cubed_sphere.hpp"
#include "soma/sem/operators.hpp"

namespace swe {

namespace detail {

using soma::LO;
using Policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::IndexType<LO>>;

// out = a x + b (y + dt z), every row and column.
void combine(double a, const Values& x, double b, const Values& y, double dt, const Values& z, const Values& out) {
  const auto nc = static_cast<int>(x.extent(1));
  Kokkos::parallel_for(
      "swe::combine", Policy(0, static_cast<LO>(x.extent(0))), KOKKOS_LAMBDA(const LO r) {
        for (int c = 0; c < nc; ++c) out(r, c) = a * x(r, c) + b * (y(r, c) + dt * z(r, c));
      });
  Kokkos::fence();
}

// phi = g (h + hs) + |u|^2 / 2, and hu = h u.
void energy_and_flux(const Values& h, const Values& hs, const Values& u, const Values& phi, const Values& hu) {
  Kokkos::parallel_for(
      "swe::energy_and_flux", Policy(0, static_cast<LO>(h.extent(0))), KOKKOS_LAMBDA(const LO r) {
        phi(r, 0) = gravity * (h(r, 0) + hs(r, 0)) + 0.5 * (u(r, 0) * u(r, 0) + u(r, 1) * u(r, 1) + u(r, 2) * u(r, 2));
        for (int c = 0; c < 3; ++c) hu(r, c) = h(r, 0) * u(r, c);
      });
  Kokkos::fence();
}

// du = -(zeta + f) (n x u) - grad phi; dh = -div.
void momentum_and_continuity(const Values& zeta, const Values& f, const Values& nxu, const Values& grad,
                             const Values& div, const Values& du, const Values& dh) {
  Kokkos::parallel_for(
      "swe::tendencies", Policy(0, static_cast<LO>(du.extent(0))), KOKKOS_LAMBDA(const LO r) {
        const double q = zeta(r, 0) + f(r, 0);
        for (int c = 0; c < 3; ++c) du(r, c) = -q * nxu(r, c) - grad(r, c);
        dh(r, 0) = -div(r, 0);
      });
  Kokkos::fence();
}

// out -= nu v.
void subtract(double nu, const Values& v, const Values& out) {
  const auto nc = static_cast<int>(v.extent(1));
  Kokkos::parallel_for(
      "swe::subtract", Policy(0, static_cast<LO>(v.extent(0))), KOKKOS_LAMBDA(const LO r) {
        for (int c = 0; c < nc; ++c) out(r, c) -= nu * v(r, c);
      });
  Kokkos::fence();
}

soma::HostMesh cubed_sphere(const soma::Comm& comm, int ne) {
  soma::sem::CubedSphereSpec spec;
  spec.ne = ne;
  spec.radius = radius;
  soma::HostMesh mesh = soma::sem::cubed_sphere_mesh(comm, spec);
  if (comm.size() > 1) {
    mesh.add_ghosts(comm, soma::GhostOptions{1, 0});
    soma::build_distributed_adjacencies(comm, mesh, soma::AdjacencyRequest{}.add(2, 1));
  }
  return mesh;
}

// The spectral radius of the discrete del^4 (scalar and vector, the larger):
// 40 power iterations from a field set by element gid and point, made
// continuous; Rayleigh quotients in the mass inner product (exact sums), so
// the same on any number of ranks.
double hyperviscous_radius(const soma::Comm& comm, const Space& s, const soma::HostMesh& mesh) {
  const auto ppe = s.points_per_element();
  const auto n = static_cast<std::size_t>(s.num_elements() * ppe);
  const auto gid = mesh.gid(2);
  const auto x = s.x();
  double radius = 0;
  for (const int nc : {1, 3}) {
    Values v("swe::power", n, static_cast<std::size_t>(nc)), w("swe::power_image", n, static_cast<std::size_t>(nc));
    Values product("swe::power_product", n, 2);
    for (std::size_t r = 0; r < n; ++r) {
      const double seed = static_cast<double>(gid(static_cast<LO>(r) / ppe)) * 0.618033988749 +
                          static_cast<double>(static_cast<LO>(r) % ppe) * 0.414213562373;
      for (int c = 0; c < nc; ++c) v(r, c) = std::sin(97.0 * seed + 13.0 * c) + 1e-3 * x(r, 2) / swe::radius;
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
      // v . L v / v . v with v normalised: the Rayleigh quotient.
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

void check(const ModelOptions& o) {
  std::string bad;
  if (o.ne < 1) bad = "ne " + std::to_string(o.ne) + " < 1";
  if (o.order < 1) bad = "order " + std::to_string(o.order) + " < 1";
  if (!(o.cfl > 0)) bad = "cfl " + std::to_string(o.cfl) + " not positive";
  if (o.dt && !(*o.dt > 0 && std::isfinite(*o.dt))) bad = "dt " + std::to_string(*o.dt) + " not positive";
  if (o.nu && !(*o.nu >= 0 && std::isfinite(*o.nu))) bad = "nu " + std::to_string(*o.nu) + " negative";
  if (!bad.empty()) throw std::invalid_argument("swe: " + bad);
}

}  // namespace detail

struct SweModel::State {
  ModelOptions options;
  soma::HostMesh mesh;
  std::optional<Space> space;
  Values h, hs, u, f;
  // The step's start, a stage, the tendencies and scratch.
  Values h0, u0, h1, u1, dh, du, zeta, phi, grad, nxu, hu, div, hyper_h, hyper_u;
  double time = 0, dt = 0, nu = 0, advective_dt = 0, hyperviscous_dt = 0;
  long steps = 0;

  State(const soma::Comm& comm, const ModelOptions& o) : options(o), mesh(detail::cubed_sphere(comm, o.ne)) {
    space.emplace(comm, mesh, o.order, soma::sem::SemGeometry::sphere(radius));
    const auto n = static_cast<std::size_t>(space->num_elements() * space->points_per_element());
    const auto make = [n](const char* label, std::size_t c) { return Values(label, n, c); };
    h = make("swe::h", 1), hs = make("swe::hs", 1), u = make("swe::u", 3), f = make("swe::f", 1);
    h0 = make("swe::h0", 1), u0 = make("swe::u0", 3), h1 = make("swe::h1", 1), u1 = make("swe::u1", 3);
    dh = make("swe::dh", 1), du = make("swe::du", 3), zeta = make("swe::zeta", 1), phi = make("swe::phi", 1);
    grad = make("swe::grad", 3), nxu = make("swe::nxu", 3), hu = make("swe::hu", 3), div = make("swe::div", 1);
    hyper_h = make("swe::hyper_h", 1), hyper_u = make("swe::hyper_u", 3);
    const auto x = space->x();
    for (std::size_t r = 0; r < n; ++r) {
      const PointState p = initial(o.test, {x(r, 0), x(r, 1), x(r, 2)}, o.alpha);
      h(r, 0) = p.h;
      hs(r, 0) = p.hs;
      f(r, 0) = p.f;
      for (std::size_t c = 0; c < 3; ++c) u(r, c) = p.u[c];
    }
    // The initial state continuous (shared points' positions differ by rounding
    // between elements) and tangent.
    space->dss(comm, h);
    space->dss(comm, hs);
    soma::sem::project_tangent(*space, u);
    space->dss(comm, u);
    soma::sem::project_tangent(*space, u);

    const std::vector<soma::Real> xi = soma::gll_points_1d(o.order);
    const double gap = std::numbers::pi * radius / (2 * o.ne) * (xi[1] - xi[0]) / 2;
    double speed = 0;
    const std::size_t owned = static_cast<std::size_t>(space->owned_elements() * space->points_per_element());
    for (std::size_t r = 0; r < owned; ++r) {
      speed = std::max(speed, std::hypot(u(r, 0), u(r, 1), u(r, 2)) + std::sqrt(gravity * std::max(h(r, 0), 0.0)));
    }
    speed = comm.allreduce(speed, soma::Op::max);
    nu = o.nu ? *o.nu : (o.test == Case::williamson2 ? 0.0 : 1.0e15 * std::pow(30.0 / o.ne, 3.2));
    advective_dt = o.cfl * gap / speed;
    hyperviscous_dt = nu > 0 ? 2.5 * o.cfl / (nu * detail::hyperviscous_radius(comm, *space, mesh))
                             : std::numeric_limits<double>::infinity();
    dt = o.dt ? *o.dt : std::min(advective_dt, hyperviscous_dt);
  }

  // dh, du from (hh, uu).
  void tendencies(const soma::Comm& comm, const Values& hh, const Values& uu) {
    const Space& s = *space;
    soma::sem::vorticity(s, uu, zeta);
    s.dss(comm, zeta);
    detail::energy_and_flux(hh, hs, uu, phi, hu);
    soma::sem::gradient(s, phi, grad);
    soma::sem::cross_normal(s, uu, nxu);
    soma::sem::divergence(s, hu, div);
    detail::momentum_and_continuity(zeta, f, nxu, grad, div, du, dh);
    s.dss(comm, du);
    s.dss(comm, dh);
    if (nu > 0) {
      soma::sem::hyperviscosity(comm, s, hh, hyper_h);
      detail::subtract(nu, hyper_h, dh);
      soma::sem::vector_hyperviscosity(comm, s, uu, hyper_u);
      detail::subtract(nu, hyper_u, du);
    }
    soma::sem::project_tangent(s, du);
  }

  // SSP-RK3 (Shu and Osher): y1 = y + dt F(y); y2 = 3/4 y + 1/4 (y1 + dt F(y1));
  // y = 1/3 y + 2/3 (y2 + dt F(y2)).
  void step(const soma::Comm& comm, double k) {
    Kokkos::deep_copy(h0, h);
    Kokkos::deep_copy(u0, u);
    tendencies(comm, h0, u0);
    detail::combine(0, h0, 1, h0, k, dh, h1);
    detail::combine(0, u0, 1, u0, k, du, u1);
    tendencies(comm, h1, u1);
    detail::combine(0.75, h0, 0.25, h1, k, dh, h1);
    detail::combine(0.75, u0, 0.25, u1, k, du, u1);
    tendencies(comm, h1, u1);
    detail::combine(1.0 / 3, h0, 2.0 / 3, h1, k, dh, h);
    detail::combine(1.0 / 3, u0, 2.0 / 3, u1, k, du, u);
  }

  bool healthy() const {
    const auto owned = static_cast<std::size_t>(space->owned_elements() * space->points_per_element());
    for (std::size_t r = 0; r < owned; ++r) {
      if (!(h(r, 0) > 0) || !std::isfinite(h(r, 0))) return false;
      for (std::size_t c = 0; c < 3; ++c) {
        if (!std::isfinite(u(r, c))) return false;
      }
    }
    return true;
  }
};

SweModel::SweModel(const soma::Comm& comm, const ModelOptions& options) {
  detail::check(options);
  s_ = std::make_unique<State>(comm, options);
}

SweModel::~SweModel() = default;

namespace {

// The step from `time` that lands on t_end: shortened, or a hair longer rather
// than leaving a sliver of a step.
bool lands(double time, double t_end, double dt) { return t_end - time <= dt * (1 + 1e-9); }

}  // namespace

void SweModel::take_step(const soma::Comm& comm, double length, double time_after) {
  State& s = *s_;
  s.step(comm, length);
  s.time = time_after;
  ++s.steps;
  const int sick = s.healthy() ? 0 : 1;
  if (comm.allreduce(sick, soma::Op::max) != 0) {
    std::ostringstream message;
    message << "swe: blew up at step " << s.steps << ", t = " << s.time << " s";
    throw std::runtime_error(message.str());
  }
}

void SweModel::advance(const soma::Comm& comm, double t_end) {
  State& s = *s_;
  while (s.time < t_end) {
    if (lands(s.time, t_end, s.dt)) {
      take_step(comm, t_end - s.time, t_end);
    } else {
      take_step(comm, s.dt, s.time + s.dt);
    }
  }
}

void SweModel::advance_whole_steps(const soma::Comm& comm, double t, double t_end) {
  State& s = *s_;
  while (t - s.time >= s.dt && !lands(s.time, t_end, s.dt)) take_step(comm, s.dt, s.time + s.dt);
}

double SweModel::time() const noexcept { return s_->time; }
double SweModel::dt() const noexcept { return s_->dt; }
double SweModel::nu() const noexcept { return s_->nu; }
long SweModel::steps() const noexcept { return s_->steps; }
const ModelOptions& SweModel::options() const noexcept { return s_->options; }
soma::HostMesh& SweModel::mesh() noexcept { return s_->mesh; }
const soma::HostMesh& SweModel::mesh() const noexcept { return s_->mesh; }
const Space& SweModel::space() const noexcept { return *s_->space; }
ConstValues SweModel::h() const { return s_->h; }
ConstValues SweModel::hs() const { return s_->hs; }
ConstValues SweModel::u() const { return s_->u; }

std::uint64_t SweModel::digest(const soma::Comm& comm) const {
  const State& s = *s_;
  const auto ppe = static_cast<std::size_t>(s.space->points_per_element());
  const std::size_t width = 1 + 4 * ppe;
  const auto gid = s.mesh.gid(2);
  std::vector<std::uint64_t> mine;
  for (soma::LO e = 0; e < s.space->owned_elements(); ++e) {
    mine.push_back(static_cast<std::uint64_t>(gid(e)));
    for (std::size_t k = 0; k < ppe; ++k) {
      const std::size_t r = static_cast<std::size_t>(e) * ppe + k;
      for (const double v : {s.h(r, 0), s.u(r, 0), s.u(r, 1), s.u(r, 2)}) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof bits);
        mine.push_back(bits);
      }
    }
  }
  const std::vector<std::uint64_t> all = comm.allgatherv(std::span<const std::uint64_t>(mine));
  std::vector<std::size_t> order(all.size() / width);
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return all[a * width] < all[b * width]; });
  std::uint64_t hash = 1469598103934665603ull;
  for (const std::size_t i : order) {
    for (std::size_t j = 0; j < width; ++j) hash = (hash ^ all[i * width + j]) * 1099511628211ull;
  }
  return hash;
}

}  // namespace swe
