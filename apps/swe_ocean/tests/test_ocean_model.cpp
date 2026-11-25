// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The barotropic ocean model on the synthetic Earth of test_ocean_mesh.cpp (synthetic_earth.hpp) and on a plane
// box: the ocean at rest stays exactly at rest; a free-surface bump under a wind, with quadratic drag and the
// default viscosity, conserves mass to rounding and has no flow through the coast; and a run has the same bits
// on any number of ranks (and threads, through the .threads and .plane_threads tests, which run
// rank_and_thread_independent and plane_box_thread_independent at several thread counts and compare the digests
// they print).

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
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
#include "synthetic_earth.hpp"

namespace {

namespace fs = std::filesystem;
using ocean::ModelOptions;
using ocean::OceanModel;
using ocean::testing::P3;
using soma::Comm;
using soma::LO;

// The integral of |eta| (m^3), the scale of the mass conservation bound.
double absolute_mass(const Comm& c, const OceanModel& m) {
  const auto eta = m.eta();
  ocean::Values a("abs_eta", eta.extent(0), 1);
  for (std::size_t r = 0; r < eta.extent(0); ++r) a(r, 0) = std::abs(eta(r, 0));
  return m.space().integral(c, a, 0);
}

// Every owned point: |u . n| <= 1e-15 |u| at coast points and u = 0 at no-slip points. Returns the coast points
// with |u| > 0 over all ranks, so that a test can see the check bite.
std::int64_t expect_no_normal_flow(const Comm& c, const OceanModel& m) {
  const auto u = m.u();
  const auto& b = m.boundary();
  const auto rows = static_cast<std::size_t>(m.space().owned_elements() * m.space().points_per_element());
  std::int64_t moving = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    const double speed = std::hypot(u(r, 0), u(r, 1), u(r, 2));
    if (b.kind(r) == 1) {
      const double un = u(r, 0) * b.normal(r, 0) + u(r, 1) * b.normal(r, 1) + u(r, 2) * b.normal(r, 2);
      EXPECT_LE(std::abs(un), 1e-15 * speed) << "row " << r;
      moving += speed > 0 ? 1 : 0;
    } else if (b.kind(r) == 2) {
      EXPECT_EQ(speed, 0.0) << "row " << r;
    }
  }
  return c.allreduce(moving, soma::Op::sum);
}

// ---------------------------------------------------------------------------------------------------------------
// The synthetic Earth at ne = 4, order 3, refined toward the coast at 400 km (as test_ocean_mesh.cpp).

#ifdef SWE_OCEAN_HAVE_HDF5

// A free-surface bump 1 m high, Gaussian of 10 degrees about (0, -50), 10 degrees west of continent A.
double bump(const std::array<double, 3>& x) {
  const double r = std::hypot(x[0], x[1], x[2]);
  const double d = ocean::testing::angle_deg({x[0] / r, x[1] / r, x[2] / r}, ocean::testing::at(0, -50));
  return std::exp(-(d / 10) * (d / 10));
}

// An eastward stress of 0.1 cos(latitude) N/m^2 at every row.
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

class OceanModelTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const Comm c = Comm::world();
    std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
    c.bcast(std::span<std::int64_t>(&id, 1), 0);
    dir_ = fs::temp_directory_path() / ("ocean_model_" + std::to_string(c.size()) + "_" + std::to_string(id));
    if (c.rank() == 0) {
      fs::create_directories(dir_);
      ocean::testing::write_grid_file(dir_ / "earth.h5", 361, 721, ocean::testing::Registration::grid, true,
                                      ocean::testing::Storage::int32, ocean::testing::heights);
    }
    c.barrier();
  }

  static void TearDownTestSuite() {
    run_.reset();
    const Comm c = Comm::world();
    c.barrier();
    if (c.rank() == 0) {
      std::error_code ec;
      fs::remove_all(dir_, ec);
    }
  }

  static ocean::OceanMesh earth(const Comm& c) {
    ocean::MeshOptions o;
    o.ne = 4;
    o.order = 3;
    o.coast_km = 400;
    o.bathymetry.grid.file = dir_ / "earth.h5";
    return ocean::build_ocean_mesh(c, o);
  }

  // The bump under the westerlies, quadratic drag, default viscosity, bump_steps steps on c.
  static std::unique_ptr<OceanModel> bump_run(const Comm& c, double* mass0 = nullptr, double* scale = nullptr) {
    ModelOptions o;
    o.forcing.wind = westerlies;
    auto m = std::make_unique<OceanModel>(c, earth(c), o);
    m->initialize(c, bump);
    if (mass0 != nullptr) *mass0 = m->mass(c);
    if (scale != nullptr) *scale = absolute_mass(c, *m);
    m->advance(c, static_cast<double>(bump_steps) * m->dt());
    return m;
  }

  // The world run, made once.
  static const OceanModel& run() {
    if (!run_) run_ = bump_run(Comm::world(), &mass0_, &scale_);
    return *run_;
  }

  static fs::path dir_;
  static std::unique_ptr<OceanModel> run_;
  static double mass0_, scale_;
};

fs::path OceanModelTest::dir_;
std::unique_ptr<OceanModel> OceanModelTest::run_;
double OceanModelTest::mass0_ = 0, OceanModelTest::scale_ = 0;

TEST_F(OceanModelTest, rest_stays_at_rest) {
  const Comm c = Comm::world();
  OceanModel m(c, earth(c), ModelOptions{});
  m.advance(c, 20 * m.dt());
  EXPECT_EQ(m.steps(), 20);
  std::int64_t moved = 0;
  const auto eta = m.eta();
  const auto u = m.u();
  for (std::size_t r = 0; r < eta.extent(0); ++r) {
    moved += eta(r, 0) != 0 || u(r, 0) != 0 || u(r, 1) != 0 || u(r, 2) != 0 ? 1 : 0;
  }
  EXPECT_EQ(c.allreduce(moved, soma::Op::sum), 0);
}

TEST_F(OceanModelTest, mass_conserved) {
  const Comm c = Comm::world();
  const OceanModel& m = run();
  EXPECT_EQ(m.steps(), bump_steps);
  EXPECT_GT(m.nu(), 0);
  const double mass = m.mass(c), drift = std::abs(mass - mass0_);
  if (c.rank() == 0) {
    std::printf("dt %.17g s, nu %.17g m^4/s, mass %.17g -> %.17g m^3, drift %.3g of the integral of |eta|\n",
                m.dt(), m.nu(), mass0_, mass, drift / scale_);
  }
  EXPECT_GT(scale_, 0);
  EXPECT_LE(drift, 1e-13 * scale_);
}

TEST_F(OceanModelTest, no_normal_flow) {
  const Comm c = Comm::world();
  const OceanModel& m = run();
  const std::int64_t coast = c.allreduce(static_cast<std::int64_t>(m.boundary().coast_points), soma::Op::sum);
  const std::int64_t no_slip = c.allreduce(static_cast<std::int64_t>(m.boundary().no_slip_points), soma::Op::sum);
  EXPECT_GT(coast, 0);
  EXPECT_GT(no_slip, 0);
  EXPECT_GT(expect_no_normal_flow(c, m), 0) << "no coast point moves: the check proves nothing";
}

// Land points of kept elements are 10 m of water to the model: under the westerlies from rest they must stay
// nearly still while the ocean moves, rather than being the fastest water of the run.
TEST_F(OceanModelTest, land_stays_still) {
  const Comm c = Comm::world();
  ModelOptions o;
  o.forcing.wind = westerlies;
  ocean::OceanMesh mesh = earth(c);
  const std::vector<std::uint8_t> land = mesh.bathymetry.land;
  OceanModel m(c, std::move(mesh), o);
  m.advance(c, 200 * m.dt());
  const auto u = m.u();
  const auto rows = static_cast<std::size_t>(m.space().owned_elements() * m.space().points_per_element());
  ASSERT_GE(land.size(), rows);
  double land_max = 0, ocean_max = 0;
  std::int64_t land_points = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    const double speed = std::hypot(u(r, 0), u(r, 1), u(r, 2));
    if (land[r] != 0) {
      land_max = std::max(land_max, speed);
      ++land_points;
    } else {
      ocean_max = std::max(ocean_max, speed);
    }
  }
  land_max = c.allreduce(land_max, soma::Op::max);
  ocean_max = c.allreduce(ocean_max, soma::Op::max);
  land_points = c.allreduce(land_points, soma::Op::sum);
  if (c.rank() == 0) {
    std::printf("%lld land points: max |u| %.3e m/s, ocean %.3e m/s\n", static_cast<long long>(land_points),
                land_max, ocean_max);
  }
  EXPECT_GT(land_points, 0);
  EXPECT_GT(ocean_max, 0);
  EXPECT_LE(land_max, 0.01 * ocean_max);
}

TEST_F(OceanModelTest, rank_and_thread_independent) {
  const Comm c = Comm::world();
  const OceanModel& m = run();
  const std::uint64_t digest = m.digest(c);
  if (c.size() > 1) {
    const std::unique_ptr<OceanModel> serial = bump_run(Comm::self());
    EXPECT_EQ(m.dt(), serial->dt());
    EXPECT_EQ(m.nu(), serial->nu());
    EXPECT_EQ(m.steps(), serial->steps());
    EXPECT_EQ(digest, serial->digest(Comm::self()));
  }
  if (c.rank() == 0) {
    std::printf("digest reproducible %016llx\n", static_cast<unsigned long long>(digest));
    std::printf("concurrency %d\n", Kokkos::DefaultHostExecutionSpace().concurrency());
  }
}

#else

TEST(OceanModelTest, needs_hdf5) { GTEST_SKIP() << "the synthetic Earth is a grid file written with HDF5"; }

#endif

// ---------------------------------------------------------------------------------------------------------------
// A plane box, 600 km by 400 km, 100 to 150 m deep, on a beta plane: no H5 file needed.

constexpr double lx = 6e5, ly = 4e5;

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

ModelOptions beta_plane(bool linear) {
  ModelOptions o;
  o.linear = linear;
  o.coriolis = [](const std::array<double, 3>& x) { return 1e-4 + 2e-11 * (x[1] - ly / 2); };
  o.forcing.wind = [](const ocean::Space& s, ocean::Values tau) {
    const auto x = s.x();
    for (std::size_t r = 0; r < tau.extent(0); ++r) {
      tau(r, 0) = -0.1 * std::cos(std::numbers::pi * x(r, 1) / ly);
      tau(r, 1) = tau(r, 2) = 0;
    }
  };
  if (linear) o.forcing.drag_linear = 1e-6;
  return o;
}

double plane_bump(const std::array<double, 3>& x) {
  const double dx = (x[0] - lx / 3) / 5e4, dy = (x[1] - ly / 2) / 5e4;
  return 0.1 * std::exp(-dx * dx - dy * dy);
}

TEST(OceanModel, plane_box_conserves_and_slips) {
  const Comm c = Comm::world();
  for (const bool linear : {false, true}) {
    OceanModel m(c, box(c), beta_plane(linear));
    EXPECT_EQ(m.nu() > 0, !linear);
    m.initialize(c, plane_bump);
    const double mass0 = m.mass(c), scale = absolute_mass(c, m);
    m.advance(c, 30 * m.dt());
    EXPECT_LE(std::abs(m.mass(c) - mass0), 1e-13 * scale) << "linear " << linear;
    EXPECT_GT(expect_no_normal_flow(c, m), 0) << "linear " << linear;
    EXPECT_THROW(m.initialize(c, plane_bump), std::logic_error);
    OceanModel serial(Comm::self(), box(Comm::self()), beta_plane(linear));
    serial.initialize(Comm::self(), plane_bump);
    serial.advance(Comm::self(), 30 * serial.dt());
    EXPECT_EQ(m.digest(c), serial.digest(Comm::self())) << "linear " << linear;
  }
}

// The digest sees every value: one tracer value at one interior point of one element (on the first rank that
// owns any), changed, changes it on every rank; changed back, the digest is the old one.
TEST(OceanModel, digest_sees_one_value) {
  const Comm c = Comm::world();
  OceanModel m(c, box(c), beta_plane(false));
  m.initialize(c, plane_bump);
  m.advance(c, 3 * m.dt());
  const std::uint64_t before = m.digest(c);
  const auto state = m.tracers().state();
  const std::size_t row = static_cast<std::size_t>(1 + (m.space().order() + 1));  // point (1, 1) of element 0
  const std::int32_t first = c.allreduce(m.space().owned_elements() > 0 ? c.rank() : c.size(), soma::Op::min);
  const double old = c.rank() == first ? state(row, 0) : 0.0;
  const auto nudge = [&](double v) {
    if (c.rank() == first) state(row, 0) = v;
    m.tracers().thicken(m.eta());
  };
  nudge(old + 1);
  const std::uint64_t after = m.digest(c);
  EXPECT_NE(after, before);
  std::uint64_t lowest = after, highest = after;
  lowest = c.allreduce(lowest, soma::Op::min);
  highest = c.allreduce(highest, soma::Op::max);
  EXPECT_EQ(lowest, highest) << "the digest differs between ranks";
  nudge(old);
  EXPECT_EQ(m.digest(c), before);
}

// initialize's velocity: set at every point, kept in the interior, with no normal flow at the coast.
TEST(OceanModel, plane_box_initial_velocity) {
  const Comm c = Comm::world();
  OceanModel m(c, box(c), beta_plane(true));
  m.initialize(c, plane_bump, [](const std::array<double, 3>&) { return std::array<double, 3>{0.3, -0.2, 0}; });
  const auto u = m.u();
  const auto& b = m.boundary();
  std::int64_t interior = 0;
  for (std::size_t r = 0; r < u.extent(0); ++r) {
    if (b.kind(r) != 0) continue;
    // Direct stiffness summation averages a point's copies with mass weights: equal values to rounding.
    EXPECT_NEAR(u(r, 0), 0.3, 1e-15) << "row " << r;
    EXPECT_NEAR(u(r, 1), -0.2, 1e-15) << "row " << r;
    EXPECT_EQ(u(r, 2), 0.0) << "row " << r;
    ++interior;
  }
  EXPECT_GT(c.allreduce(interior, soma::Op::sum), 0);
  EXPECT_GT(expect_no_normal_flow(c, m), 0);
}

// For compare_threads.cmake (test_ocean_model.plane_threads), which runs this at several thread counts: the
// nonlinear mode, with hyperviscosity, wind and quadratic drag.
TEST(OceanModel, plane_box_thread_independent) {
  const Comm c = Comm::world();
  ModelOptions o = beta_plane(false);
  OceanModel m(c, box(c), o);
  m.initialize(c, plane_bump);
  m.advance(c, 30 * m.dt());
  const std::uint64_t digest = m.digest(c);
  if (c.rank() == 0) {
    std::printf("digest reproducible %016llx\n", static_cast<unsigned long long>(digest));
    std::printf("concurrency %d\n", Kokkos::DefaultHostExecutionSpace().concurrency());
  }
}

TEST(OceanModel, plane_box_rest_stays_at_rest) {
  const Comm c = Comm::world();
  ModelOptions o = beta_plane(false);
  o.forcing.wind = nullptr;
  OceanModel m(c, box(c), o);
  m.advance(c, 10 * m.dt());
  const auto eta = m.eta();
  const auto u = m.u();
  std::int64_t moved = 0;
  for (std::size_t r = 0; r < eta.extent(0); ++r) {
    moved += eta(r, 0) != 0 || u(r, 0) != 0 || u(r, 1) != 0 || u(r, 2) != 0 ? 1 : 0;
  }
  EXPECT_EQ(c.allreduce(moved, soma::Op::sum), 0);
}

TEST(OceanModel, rejects_bad_options) {
  const Comm c = Comm::world();
  for (int k = 0; k < 8; ++k) {
    ModelOptions o;
    if (k == 0) o.cfl = 0;
    if (k == 1) o.dt = -1;
    if (k == 2) o.nu = -1;
    if (k == 3) o.forcing.drag_quadratic = -1;
    if (k == 4) o.forcing.drag_linear = -1;
    if (k == 5) o.dt = std::nan("");
    if (k == 6) o.forcing.land_damping = -1;
    if (k == 7) o.forcing.land_damping = std::nan("");
    EXPECT_THROW(OceanModel(c, box(c), o), std::invalid_argument) << k;
  }
}

}  // namespace
