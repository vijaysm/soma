// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Tracers on the K levels of the s and z-star coordinates, carried by the ocean model on the synthetic Earth of
// test_ocean_mesh.cpp (synthetic_earth.hpp, a grid file written with HDF5) and in a plane box (no file needed):
// a uniform tracer stays uniform, a varying one is conserved to rounding, the vertical velocity diagnosed from
// the bottom up vanishes at the surface, the levels keep their thickness floor (z-star) or their reference
// proportions (s), and a run has the same bits on any number of ranks. Each run carries two tracers: tracer 0
// is 7 everywhere, tracer 1 is sin(lon) exp(z / 500).

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "grid_files.hpp"
#include "model.hpp"
#include "ocean_mesh.hpp"
#include "sem_meshes.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/operators.hpp"
#include "synthetic_earth.hpp"
#include "tracers.hpp"
#include "vertical.hpp"

namespace {

namespace fs = std::filesystem;
using ocean::ModelOptions;
using ocean::OceanModel;
using ocean::Vertical;
using soma::Comm;
using soma::LO;

// The bounds of the brief. A uniform tracer, a conserved total and the surface velocity are all exact but for
// rounding: each step rounds every update a few times, so 1e-13 leaves room for a few hundred steps' drift of
// a few ulps each.
constexpr double uniform_tolerance = 1e-13;
constexpr double conservation_tolerance = 1e-13;
constexpr double surface_tolerance = 1e-13;
// h_k = h0_k (1 + eta / H): a quotient, a sum and a product, a few ulps of h_k.
constexpr double thickness_tolerance = 1e-14;
// A tracer of 7 carried wholly as the anomaly (Tbar = 0) is uniform only as far as the tracer transport matches
// the levels' continuity. The plane box's levels are all thick enough that its rounding stays at a few ulps of 7
// over 30 steps; on the synthetic Earth, vanished z-star levels (1e-3 m) pass their thick neighbours' fluxes on
// and magnify rounding (4.2e-13 measured for a uniform tracer carried this way); 1e-11 leaves room for that and
// is still eleven orders below the 0.76 of a transport that weights a stage's vertical fluxes wrongly.
constexpr double offset_tolerance_plane = 1e-13;
constexpr double offset_tolerance_sphere = 1e-11;
constexpr long offset_steps = 30;
// No limiter: the overshoot of the varying tracer is measured, bounded only loosely to catch a blow-up.
constexpr double overshoot_factor = 10;
constexpr double uniform_value = 7;

ocean::VerticalSpec coordinate(Vertical kind) {
  ocean::VerticalSpec v;
  v.kind = kind;
  return v;
}

const char* name(Vertical kind) { return kind == Vertical::s ? "s" : "zstar"; }

// Tracer 0: uniform; tracer 1: sin(lon) exp(z / 500), lon the angle of the position about the z axis.
double initial_tracer(const std::array<double, 3>& x, double z, int tracer) {
  if (tracer == 0) return uniform_value;
  return std::sin(std::atan2(x[1], x[0])) * std::exp(z / 500);
}

// The rows of the owned elements.
std::size_t owned_rows(const OceanModel& m) {
  return static_cast<std::size_t>(m.space().owned_elements() * m.space().points_per_element());
}

// max over owned rows of |div_wk(D u)| (1/s . m), the scale of the surface velocity's rounding.
double max_divergence(const Comm& c, const OceanModel& m) {
  const auto eta = m.eta();
  const auto u = m.u();
  const auto H = m.H();
  ocean::Values flux("flux", eta.extent(0), 3), div("div", eta.extent(0), 1);
  for (std::size_t r = 0; r < eta.extent(0); ++r) {
    for (std::size_t k = 0; k < 3; ++k) flux(r, k) = (H(r, 0) + eta(r, 0)) * u(r, k);
  }
  soma::sem::divergence_weak(c, m.space(), flux, div);
  double most = 0;
  for (std::size_t r = 0; r < owned_rows(m); ++r) most = std::max(most, std::abs(div(r, 0)));
  return c.allreduce(most, soma::Op::max);
}

// max over owned rows and levels of |T - 7| of tracer 0.
double uniform_error(const Comm& c, const OceanModel& m) {
  const auto& t = m.tracers();
  const int K = t.levels();
  const auto h = t.h();
  const auto hT = t.hT();
  double worst = 0;
  for (std::size_t r = 0; r < owned_rows(m); ++r) {
    for (int k = 0; k < K; ++k) worst = std::max(worst, std::abs(hT(r, k) / h(r, k) - uniform_value));
  }
  return c.allreduce(worst, soma::Op::max);
}

// max over owned rows and levels of |T| of a tracer.
double extreme(const Comm& c, const OceanModel& m, int tracer) {
  const auto& t = m.tracers();
  const int K = t.levels();
  const auto h = t.h();
  const auto hT = t.hT();
  double most = 0;
  for (std::size_t r = 0; r < owned_rows(m); ++r) {
    for (int k = 0; k < K; ++k) most = std::max(most, std::abs(hT(r, tracer * K + k) / h(r, k)));
  }
  return c.allreduce(most, soma::Op::max);
}

// The integral of sum_k |T_k| h_k of a tracer (m^3 times its unit), the scale of its conservation bound.
double absolute_total(const Comm& c, const OceanModel& m, int tracer) {
  const auto& t = m.tracers();
  const int K = t.levels();
  const auto hT = t.hT();
  ocean::Values a("abs", hT.extent(0), 1);
  for (std::size_t r = 0; r < hT.extent(0); ++r) {
    for (int k = 0; k < K; ++k) a(r, 0) += std::abs(hT(r, tracer * K + k));
  }
  return m.space().integral(c, a, 0);
}

// The thickness check of the brief, worst over owned rows and levels: z-star, max(0, floor - h_k) / h_min with
// floor = h_min (1 - max |eta| / H_min); s, |h_k - h0_k (1 + eta / H)| / h_k with h0_k from
// reference_thicknesses.
double thickness_error(const Comm& c, const OceanModel& m) {
  const auto& t = m.tracers();
  const auto& spec = m.options().vertical;
  const int K = t.levels();
  const auto h = t.h();
  const auto eta = m.eta();
  const auto H = m.H();
  double eta_max = 0;
  for (std::size_t r = 0; r < owned_rows(m); ++r) eta_max = std::max(eta_max, std::abs(eta(r, 0)));
  eta_max = c.allreduce(eta_max, soma::Op::max);
  const double floor = spec.h_min * (1 - eta_max / ocean::H_min);
  std::vector<double> h0(static_cast<std::size_t>(K));
  double worst = 0;
  for (std::size_t r = 0; r < owned_rows(m); ++r) {
    ocean::reference_thicknesses(spec, H(r, 0), h0);
    for (int k = 0; k < K; ++k) {
      if (spec.kind == Vertical::zstar) {
        worst = std::max(worst, std::max(0.0, floor - h(r, k)) / spec.h_min);
      } else {
        const double expected = h0[static_cast<std::size_t>(k)] * (1 + eta(r, 0) / H(r, 0));
        worst = std::max(worst, std::abs(h(r, k) - expected) / h(r, k));
      }
    }
  }
  return c.allreduce(worst, soma::Op::max);
}

// What a run measures, each over all its steps where it says "worst".
struct Measures {
  double uniform = 0;     // worst max |T - 7|
  double total0 = 0;      // tracer 1's total at t = 0
  double total = 0;       // and at the end
  double scale = 0;       // the integral of sum_k |T_k| h_k of tracer 1 at t = 0
  double surface = 0;     // worst w_surface_max / max |div(D u)|
  double surface_w = 0;   // worst w_surface_max (m/s)
  double thickness = 0;   // worst thickness_error
  double extreme0 = 0;    // max |T| of tracer 1 at t = 0 (no limiter: its overshoot is measured, not bounded)
  double extreme = 0;     // and over the run
  std::uint64_t digest = 0;
  long steps = 0;
};

// Sets the two tracers on m and takes `steps` steps, one at a time, measuring after each.
Measures measure(const Comm& c, OceanModel& m, long steps) {
  Measures out;
  m.tracers().set(c, initial_tracer);
  out.total0 = m.tracers().total(c, 1);
  out.scale = absolute_total(c, m, 1);
  out.uniform = uniform_error(c, m);
  out.extreme0 = extreme(c, m, 1);
  for (long n = 0; n < steps; ++n) {
    m.advance(c, m.time() + m.dt());
    const double w = m.tracers().w_surface_max(c);
    out.surface_w = std::max(out.surface_w, w);
    out.surface = std::max(out.surface, w / max_divergence(c, m));
    out.uniform = std::max(out.uniform, uniform_error(c, m));
    out.thickness = std::max(out.thickness, thickness_error(c, m));
    out.extreme = std::max(out.extreme, extreme(c, m, 1));
  }
  out.total = m.tracers().total(c, 1);
  out.digest = m.digest(c);
  out.steps = m.steps();
  return out;
}

// Tracer 0 set to 0 (Tbar = 0) and its state then written as 7 h_k, so that T = 7 rides wholly on the anomaly
// and its transport, not on Tbar; the worst max |T - 7| over offset_steps steps.
double offset_error(const Comm& c, OceanModel& m) {
  m.tracers().set(c, [](const std::array<double, 3>&, double, int) { return 0.0; });
  const auto a = m.tracers().state();
  const auto h = m.tracers().h();
  const int K = m.tracers().levels();
  for (std::size_t r = 0; r < a.extent(0); ++r) {
    for (int k = 0; k < K; ++k) a(r, k) = uniform_value * h(r, k);
  }
  m.tracers().thicken(m.eta());
  double worst = uniform_error(c, m);
  for (long n = 0; n < offset_steps; ++n) {
    m.advance(c, m.time() + m.dt());
    worst = std::max(worst, uniform_error(c, m));
  }
  return worst;
}

void report(const Comm& c, const char* where, Vertical kind, const Measures& r) {
  if (c.rank() != 0) return;
  std::printf("%s %s: %ld steps, max |T - 7| %.3g, total %.17g -> %.17g (drift %.3g of %.17g), w_0 %.3g m/s "
              "(%.3g of max |div(D u)|), thickness %.3g, max |T_1| %.3g -> %.3g\n",
              where, name(kind), r.steps, r.uniform, r.total0, r.total, std::abs(r.total - r.total0) / r.scale,
              r.scale, r.surface_w, r.surface, r.thickness, r.extreme0, r.extreme);
}

void expect_measures(const Measures& r, Vertical kind) {
  EXPECT_LT(r.extreme, overshoot_factor * r.extreme0) << name(kind);
  EXPECT_LE(r.uniform, uniform_tolerance) << name(kind);
  EXPECT_GT(r.scale, 0) << name(kind);
  EXPECT_LE(std::abs(r.total - r.total0), conservation_tolerance * r.scale) << name(kind);
  EXPECT_GT(r.surface_w, 0) << name(kind) << ": no rounding at the surface? the check proves nothing";
  EXPECT_LE(r.surface, surface_tolerance) << name(kind);
}

void expect_thickness(const Measures& r, Vertical kind) {
  if (kind == Vertical::zstar) {
    EXPECT_EQ(r.thickness, 0.0) << "a z-star level below h_min (1 - max |eta| / H_min)";
  } else {
    EXPECT_LE(r.thickness, thickness_tolerance) << "an s level off h0_k (1 + eta / H)";
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The synthetic Earth at ne = 4, order 3, refined toward the coast at 400 km, with the bump of
// test_ocean_model.cpp under the westerlies, quadratic drag and the default viscosity, 50 steps.

#ifdef SWE_OCEAN_HAVE_HDF5

double bump(const std::array<double, 3>& x) {
  const double r = std::hypot(x[0], x[1], x[2]);
  const double d = ocean::testing::angle_deg({x[0] / r, x[1] / r, x[2] / r}, ocean::testing::at(0, -50));
  return std::exp(-(d / 10) * (d / 10));
}

void westerlies(const ocean::Space& s, ocean::Values tau) {
  const auto x = s.x();
  for (std::size_t r = 0; r < tau.extent(0); ++r) {
    const double rho = std::hypot(x(r, 0), x(r, 1)), len = std::hypot(x(r, 0), x(r, 1), x(r, 2));
    const double scale = rho > 0 ? 0.1 * (rho / len) / rho : 0.0;
    tau(r, 0) = -x(r, 1) * scale;
    tau(r, 1) = x(r, 0) * scale;
    tau(r, 2) = 0;
  }
}

constexpr long bump_steps = 50;

class TracersTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const Comm c = Comm::world();
    std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
    c.bcast(std::span<std::int64_t>(&id, 1), 0);
    dir_ = fs::temp_directory_path() / ("ocean_tracers_" + std::to_string(c.size()) + "_" + std::to_string(id));
    if (c.rank() == 0) {
      fs::create_directories(dir_);
      ocean::testing::write_grid_file(dir_ / "earth.h5", 361, 721, ocean::testing::Registration::grid, true,
                                      ocean::testing::Storage::int32, ocean::testing::heights);
    }
    c.barrier();
  }

  static void TearDownTestSuite() {
    runs_.clear();
    const Comm c = Comm::world();
    c.barrier();
    if (c.rank() == 0) {
      std::error_code ec;
      fs::remove_all(dir_, ec);
    }
  }

  static std::unique_ptr<OceanModel> bump_model(const Comm& c, Vertical kind) {
    ocean::MeshOptions mo;
    mo.ne = 4;
    mo.order = 3;
    mo.coast_km = 400;
    mo.bathymetry.grid.file = dir_ / "earth.h5";
    ModelOptions o;
    o.forcing.wind = westerlies;
    o.vertical = coordinate(kind);
    auto m = std::make_unique<OceanModel>(c, ocean::build_ocean_mesh(c, mo), o);
    m->initialize(c, bump);
    return m;
  }

  static Measures bump_run(const Comm& c, Vertical kind) { return measure(c, *bump_model(c, kind), bump_steps); }

  // The world run of a coordinate, made once.
  static const Measures& run(Vertical kind) {
    auto it = runs_.find(kind);
    if (it == runs_.end()) {
      it = runs_.emplace(kind, bump_run(Comm::world(), kind)).first;
      report(Comm::world(), "sphere", kind, it->second);
    }
    return it->second;
  }

  static fs::path dir_;
  static std::map<Vertical, Measures> runs_;
};

fs::path TracersTest::dir_;
std::map<Vertical, Measures> TracersTest::runs_;

TEST_F(TracersTest, uniform_stays_uniform) {
  for (const Vertical kind : {Vertical::s, Vertical::zstar}) {
    EXPECT_LE(run(kind).uniform, uniform_tolerance) << name(kind);
  }
}

TEST_F(TracersTest, conserved) {
  for (const Vertical kind : {Vertical::s, Vertical::zstar}) {
    const Measures& r = run(kind);
    EXPECT_GT(r.scale, 0) << name(kind);
    EXPECT_LE(std::abs(r.total - r.total0), conservation_tolerance * r.scale) << name(kind);
    EXPECT_LT(r.extreme, overshoot_factor * r.extreme0) << name(kind);
  }
}

TEST_F(TracersTest, transport_matches_level_continuity) {
  const Comm c = Comm::world();
  for (const Vertical kind : {Vertical::s, Vertical::zstar}) {
    const double error = offset_error(c, *bump_model(c, kind));
    if (c.rank() == 0) std::printf("sphere %s: offset tracer max |T - 7| %.3g\n", name(kind), error);
    EXPECT_LE(error, offset_tolerance_sphere) << name(kind);
  }
}

TEST_F(TracersTest, surface_w_zero) {
  for (const Vertical kind : {Vertical::s, Vertical::zstar}) {
    const Measures& r = run(kind);
    EXPECT_GT(r.surface_w, 0) << name(kind) << ": no rounding at the surface? the check proves nothing";
    EXPECT_LE(r.surface, surface_tolerance) << name(kind);
  }
}

TEST_F(TracersTest, thickness_floor) {
  for (const Vertical kind : {Vertical::s, Vertical::zstar}) expect_thickness(run(kind), kind);
}

TEST_F(TracersTest, rank_independent) {
  const Comm c = Comm::world();
  if (c.size() == 1) GTEST_SKIP() << "one rank: nothing to compare";
  for (const Vertical kind : {Vertical::s, Vertical::zstar}) {
    EXPECT_EQ(run(kind).digest, bump_run(Comm::self(), kind).digest) << name(kind);
  }
}

#else

TEST(TracersTest, needs_hdf5) { GTEST_SKIP() << "the synthetic Earth is a grid file written with HDF5"; }

#endif

// ---------------------------------------------------------------------------------------------------------------
// A plane box, 600 km by 400 km, 100 to 150 m deep, on a beta plane (test_ocean_model.cpp's), 30 steps: no H5
// file needed. Its z-star levels are one wet level, a partial one where H > 100.8 m and vanished ones below.

constexpr double lx = 6e5, ly = 4e5;
constexpr long plane_steps = 30;

ocean::OceanMesh box(const Comm& c) {
  ocean::OceanMesh o;
  o.mesh = std::make_unique<soma::HostMesh>(sem_test::box(c, 8, 6, lx, ly));
  o.space = std::make_unique<ocean::Space>(c, *o.mesh, 3, soma::sem::SemGeometry::plane());
  const auto n = static_cast<std::size_t>(o.space->num_elements() * o.space->points_per_element());
  o.bathymetry.H = ocean::Values("H", n, 1);
  const auto x = o.space->x();
  for (std::size_t r = 0; r < n; ++r) o.bathymetry.H(r, 0) = 100 + 50 * x(r, 0) / lx;
  return o;
}

ModelOptions beta_plane(Vertical kind) {
  ModelOptions o;
  o.coriolis = [](const std::array<double, 3>& x) { return 1e-4 + 2e-11 * (x[1] - ly / 2); };
  o.forcing.wind = [](const ocean::Space& s, ocean::Values tau) {
    const auto x = s.x();
    for (std::size_t r = 0; r < tau.extent(0); ++r) {
      tau(r, 0) = -0.1 * std::cos(std::numbers::pi * x(r, 1) / ly);
      tau(r, 1) = tau(r, 2) = 0;
    }
  };
  o.vertical = coordinate(kind);
  return o;
}

double plane_bump(const std::array<double, 3>& x) {
  const double dx = (x[0] - lx / 3) / 5e4, dy = (x[1] - ly / 2) / 5e4;
  return 0.1 * std::exp(-dx * dx - dy * dy);
}

Measures plane_run(const Comm& c, Vertical kind) {
  OceanModel m(c, box(c), beta_plane(kind));
  m.initialize(c, plane_bump);
  return measure(c, m, plane_steps);
}

TEST(Tracers, plane_box) {
  const Comm c = Comm::world();
  for (const Vertical kind : {Vertical::s, Vertical::zstar}) {
    const Measures r = plane_run(c, kind);
    report(c, "plane", kind, r);
    expect_measures(r, kind);
    expect_thickness(r, kind);
    EXPECT_EQ(r.digest, plane_run(Comm::self(), kind).digest) << name(kind);
  }
}

TEST(Tracers, plane_transport_matches_level_continuity) {
  const Comm c = Comm::world();
  for (const Vertical kind : {Vertical::s, Vertical::zstar}) {
    OceanModel m(c, box(c), beta_plane(kind));
    m.initialize(c, plane_bump);
    const double error = offset_error(c, m);
    if (c.rank() == 0) std::printf("plane %s: offset tracer max |T - 7| %.3g\n", name(kind), error);
    EXPECT_LE(error, offset_tolerance_plane) << name(kind);
  }
}

// The tracer z passed to set is the level's middle below the surface eta: in a column at rest (eta = 0) the
// levels' middles, from the reference thicknesses.
TEST(Tracers, set_passes_level_middles) {
  const Comm c = Comm::world();
  OceanModel m(c, box(c), beta_plane(Vertical::s));
  const int K = m.tracers().levels();
  ASSERT_EQ(K, 10);
  ASSERT_EQ(m.tracers().count(), 2);
  m.tracers().set(c, [](const std::array<double, 3>&, double z, int tracer) { return tracer == 0 ? z : 1.0; });
  const auto h = m.tracers().h();
  const auto hT = m.tracers().hT();
  std::vector<double> h0(static_cast<std::size_t>(K));
  for (std::size_t r = 0; r < owned_rows(m); ++r) {
    ocean::reference_thicknesses(m.options().vertical, m.H()(r, 0), h0);
    double top = 0;
    for (int k = 0; k < K; ++k) {
      const double middle = top - h0[static_cast<std::size_t>(k)] / 2;
      EXPECT_NEAR(hT(r, k) / h(r, k), middle, 1e-12 * m.H()(r, 0)) << "row " << r << " level " << k;
      EXPECT_NEAR(hT(r, K + k) / h(r, k), 1.0, 1e-15) << "row " << r << " level " << k;
      top -= h0[static_cast<std::size_t>(k)];
    }
  }
}

TEST(Tracers, rejects_bad_options) {
  const Comm c = Comm::world();
  for (int k = 0; k < 3; ++k) {
    ModelOptions o;
    if (k == 0) o.tracers = -1;
    if (k == 1) o.nu_tracer = -1;
    if (k == 2) o.vertical.levels = 0;
    EXPECT_THROW(OceanModel(c, box(c), o), std::invalid_argument) << k;
  }
}

}  // namespace
