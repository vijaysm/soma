// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Bathymetry at the GLL points of a cubed sphere of the Earth's radius: footprint means of a smooth field and
// of a step, bilinear interpolation for fine footprints, smoothing to the slope factor r_max, bounded windows,
// footprints over a pole and across the antimeridian, and the same bits on any number of ranks.

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <map>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "bathymetry.hpp"
#include "constants.hpp"
#include "grid_file.hpp"
#include "grid_files.hpp"
#include "sem_meshes.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/sem_space.hpp"

namespace {

namespace fs = std::filesystem;
using ocean::testing::Registration;
using ocean::testing::Storage;
using soma::Comm;
using soma::HostMesh;
using Geometry = soma::sem::SemGeometry;

constexpr double deg = std::numbers::pi / 180;

// A directory for this run's files, the same on every rank; rank 0 removes it.
class Scratch {
 public:
  explicit Scratch(const Comm& c) : c_(c) {
    std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
    c.bcast(std::span<std::int64_t>(&id, 1), 0);
    dir_ = fs::temp_directory_path() / ("ocean_bathymetry_" + std::to_string(c.size()) + "_" + std::to_string(id));
    if (c.rank() == 0) fs::create_directories(dir_);
    c.barrier();
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    c_.barrier();
    if (c_.rank() == 0) {
      std::error_code ec;
      fs::remove_all(dir_, ec);
    }
  }
  [[nodiscard]] fs::path path(const std::string& name) const { return dir_ / name; }

 private:
  const Comm& c_;
  fs::path dir_;
};

// Latitude and longitude (degrees) of row r of the space.
std::pair<double, double> lat_lon(const ocean::Space& s, std::size_t r) {
  const auto x = s.x();
  const double norm = std::sqrt(x(r, 0) * x(r, 0) + x(r, 1) * x(r, 1) + x(r, 2) * x(r, 2));
  return {std::asin(x(r, 2) / norm) / deg, std::atan2(x(r, 1), x(r, 0)) / deg};
}

std::size_t owned_rows(const ocean::Space& s) {
  return static_cast<std::size_t>(s.owned_elements()) * static_cast<std::size_t>(s.points_per_element());
}

// The angular radius (radians) of the footprint of row r: sqrt(m / pi) over the radius.
double footprint(const ocean::Space& s, std::size_t r) {
  return std::sqrt(s.mass()(r, 0) / std::numbers::pi) / ocean::a;
}

#ifdef SWE_OCEAN_HAVE_HDF5

// Rank 0 writes heights z(lat, lon); every rank waits for the file.
void write_on_root(const Comm& c, const fs::path& path, std::size_t nlat, std::size_t nlon, Registration reg,
                   Storage storage, const std::function<double(double, double)>& z) {
  if (c.rank() == 0) ocean::testing::write_grid_file(path, nlat, nlon, reg, true, storage, z);
  c.barrier();
}

// A field with ocean, land, a pole and an antimeridian worth comparing: depth varies with longitude
// everywhere (so the footprint over the pole has roughness) and is land in patches.
double rough_heights(double lat, double lon) {
  return -(2000 + 1500 * std::sin(2 * lon * deg) * std::cos(lat * deg) + 1000 * std::sin(lat * deg)) + 1200;
}

// Values of an owned row keyed by (element gid, point).
using Key = std::pair<soma::GO, int>;
struct Point {
  double H, roughness;
  std::uint8_t land;
};

std::map<Key, Point> by_gid(const ocean::Space& s, const HostMesh& m, const ocean::Bathymetry& b) {
  std::map<Key, Point> out;
  const auto gid = m.gid(2);
  const int ppe = s.points_per_element();
  for (std::size_t r = 0; r < owned_rows(s); ++r) {
    const auto e = static_cast<int>(r) / ppe;
    out[{gid(e), static_cast<int>(r) % ppe}] = {b.H(r, 0), b.roughness(r, 0), b.land[r]};
  }
  return out;
}

// Shared points (the same place to a millimetre) hold the same H and roughness in every element on the rank,
// ghosts included: the result is continuous.
void expect_continuous(const ocean::Space& s, const ocean::Bathymetry& b) {
  const auto x = s.x();
  std::map<std::array<long long, 3>, std::pair<double, double>> seen;
  for (std::size_t r = 0; r < b.land.size(); ++r) {
    const std::array<long long, 3> key{std::llround(x(r, 0) * 1e3), std::llround(x(r, 1) * 1e3),
                                       std::llround(x(r, 2) * 1e3)};
    const auto [it, fresh] = seen.try_emplace(key, b.H(r, 0), b.roughness(r, 0));
    if (fresh) continue;
    EXPECT_EQ(it->second.first, b.H(r, 0)) << "row " << r;
    EXPECT_EQ(it->second.second, b.roughness(r, 0)) << "row " << r;
  }
}

// The field depth = 3000 + 500 z + 300 x + 300 y of the unit vector (cos lat cos lon, cos lat sin lon, sin lat):
// linear in space, smooth through the poles and across longitude 180 and not symmetric about it (y changes sign
// there), all ocean. Its mean over a cap of angular radius
// rho about p is exactly its value at p (1 + cos rho) / 2 (the mean unit vector of a cap), within 1.6 m of the
// value at the point for these discs. Sampling a disc of 2 to 5 degrees by 0.25-degree cell centres moves its
// centroid by a few hundredths of a degree (~ 0.2 m at 8.7 m per degree); a disc over the pole or across the
// seam that missed a part of its cells would be off by metres. 1e-3 of 3000 m (3 m) separates the two, for
// the cap mean and, as asked, for the value at the point.
constexpr double linear_tolerance = 1e-3;

TEST(Bathymetry, footprint_mean_of_linear_field) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("linear.h5");
  const auto depth = [](double ux, double uy, double uz, double scale) {
    return 3000 + (500 * uz + 300 * ux + 300 * uy) * scale;
  };
  write_on_root(c, path, 721, 1441, Registration::grid, Storage::float64, [&](double lat, double lon) {
    return -depth(std::cos(lat * deg) * std::cos(lon * deg), std::cos(lat * deg) * std::sin(lon * deg),
                  std::sin(lat * deg), 1);
  });
  HostMesh mesh = sem_test::sphere(c, 4, ocean::a);
  const ocean::Space space(c, mesh, 3, Geometry::sphere(ocean::a));
  ocean::BathymetryOptions o;
  o.grid.file = path;
  const ocean::Bathymetry b = ocean::average(c, space, mesh, o);
  const auto x = space.x();
  double worst = 0, worst_point = 0;
  std::int64_t seam = 0;
  for (std::size_t r = 0; r < owned_rows(space); ++r) {
    const double norm = std::sqrt(x(r, 0) * x(r, 0) + x(r, 1) * x(r, 1) + x(r, 2) * x(r, 2));
    const double ux = x(r, 0) / norm, uy = x(r, 1) / norm, uz = x(r, 2) / norm;
    const double rho = footprint(space, r);
    const double cap = depth(ux, uy, uz, (1 + std::cos(rho)) / 2), point = depth(ux, uy, uz, 1);
    // Discs clear of the poles that reach across longitude 180.
    const auto [lat, lon] = lat_lon(space, r);
    const bool clear = std::abs(lat) * deg + rho < std::numbers::pi / 2;
    if (clear && std::abs(lon) * deg + rho / std::cos(lat * deg) > std::numbers::pi) ++seam;
    worst = std::max(worst, std::abs(b.H(r, 0) - cap) / cap);
    worst_point = std::max(worst_point, std::abs(b.H(r, 0) - point) / point);
    EXPECT_EQ(b.land[r], 0);
  }
  worst = c.allreduce(worst, soma::Op::max);
  worst_point = c.allreduce(worst_point, soma::Op::max);
  seam = c.allreduce(seam, soma::Op::sum);
  if (c.rank() == 0) {
    std::printf("linear field: worst relative error %.3e (cap mean), %.3e (point); %lld discs across 180\n",
                worst, worst_point, static_cast<long long>(seam));
  }
  EXPECT_GT(seam, 0);
  EXPECT_LE(worst, linear_tolerance);
  EXPECT_LE(worst_point, linear_tolerance);
}

// The step's meridian: longitude 2, off the cubed sphere's lines of points, so its discs are cut off centre.
constexpr double step_lon = 2;
// The normal of the step meridian's plane, pointing east.
const std::array<double, 3> step_normal{-std::sin(step_lon * deg), std::cos(step_lon * deg), 0};

// The fraction of the disc of angular radius rho about the unit vector p east of the step meridian, by the
// midpoint rule on 200 x 400 polar cells: far finer than the grid's cells.
double east_fraction(const std::array<double, 3>& p, double rho) {
  // A tangent basis at p.
  std::array<double, 3> e1{-p[1], p[0], 0};
  double n1 = std::hypot(e1[0], e1[1]);
  if (n1 < 1e-12) {
    e1 = {1, 0, 0};
    n1 = 1;
  }
  for (double& v : e1) v /= n1;
  const std::array<double, 3> e2{p[1] * e1[2] - p[2] * e1[1], p[2] * e1[0] - p[0] * e1[2],
                                 p[0] * e1[1] - p[1] * e1[0]};
  constexpr int nr = 200, nt = 400;
  double east = 0, total = 0;
  for (int i = 0; i < nr; ++i) {
    const double t = rho * (i + 0.5) / nr, w = std::sin(t);
    for (int k = 0; k < nt; ++k) {
      const double phi = 2 * std::numbers::pi * (k + 0.5) / nt;
      double side = 0;
      for (int c = 0; c < 3; ++c) {
        const auto cc = static_cast<std::size_t>(c);
        side += (std::cos(t) * p[cc] + std::sin(t) * (std::cos(phi) * e1[cc] + std::sin(phi) * e2[cc])) *
                step_normal[cc];
      }
      total += w;
      if (side > 0) east += w;
    }
  }
  return east / total;
}

// Land (z = 500) west of longitude 2, 4000 m deep east of it: a point whose disc the meridian crosses takes
// 4000 times the disc's area east of it. 2% of 4000 m (80 m) covers the cell-centre sampling of the disc's
// rim (0.25-degree cells on discs of 2 to 5 degrees; 13 m measured) with room to spare.
constexpr double step_tolerance = 0.02 * 4000;

TEST(Bathymetry, footprint_mean_of_step) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("step.h5");
  write_on_root(c, path, 720, 1440, Registration::cell, Storage::int32,
                [](double, double lon) { return lon > step_lon ? -4000.0 : 500.0; });
  HostMesh mesh = sem_test::sphere(c, 4, ocean::a);
  const ocean::Space space(c, mesh, 3, Geometry::sphere(ocean::a));
  ocean::BathymetryOptions o;
  o.grid.file = path;
  const ocean::Bathymetry b = ocean::average(c, space, mesh, o);
  const auto x = space.x();
  double worst = 0;
  std::int64_t checked = 0, off_centre = 0;
  for (std::size_t r = 0; r < owned_rows(space); ++r) {
    const double norm = std::sqrt(x(r, 0) * x(r, 0) + x(r, 1) * x(r, 1) + x(r, 2) * x(r, 2));
    const std::array<double, 3> p{x(r, 0) / norm, x(r, 1) / norm, x(r, 2) / norm};
    const double rho = footprint(space, r);
    const double lat = lat_lon(space, r).first;
    // Near longitude 2 (not 182), the disc crossed by the meridian and clear of the poles.
    const double s = std::asin(p[0] * step_normal[0] + p[1] * step_normal[1]);
    if (p[0] <= 0 || std::abs(s) >= rho || std::abs(lat) * deg + rho > 80 * deg) continue;
    const double expected = 4000 * east_fraction(p, rho);
    worst = std::max(worst, std::abs(b.H(r, 0) - expected));
    ++checked;
    if (std::abs(expected / 4000 - 0.5) > 0.1) ++off_centre;
  }
  worst = c.allreduce(worst, soma::Op::max);
  checked = c.allreduce(checked, soma::Op::sum);
  off_centre = c.allreduce(off_centre, soma::Op::sum);
  if (c.rank() == 0) {
    std::printf("step: %lld points on the step (%lld cut off centre), worst error %.2f m\n",
                static_cast<long long>(checked), static_cast<long long>(off_centre), worst);
  }
  EXPECT_GT(off_centre, 10);
  EXPECT_LE(worst, step_tolerance);
}

// Bilinear interpolation reproduces a bilinear field to rounding: values below 7000 m, a few ulps each.
constexpr double bilinear_tolerance = 1e-9;

TEST(Bathymetry, bilinear_when_fine) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("bilinear.h5");
  const auto depth = [](double lat, double lon) { return 4000 + 10 * lat + 5 * lon + 0.05 * lat * lon; };
  write_on_root(c, path, 181, 361, Registration::grid, Storage::float64,
                [&](double lat, double lon) { return -depth(lat, lon); });
  HostMesh mesh = sem_test::sphere(c, 4, ocean::a);
  const ocean::Space space(c, mesh, 3, Geometry::sphere(ocean::a));
  ocean::BathymetryOptions o;
  o.grid.file = path;
  o.bilinear_below_m = 1e12;  // every footprint is "fine"
  const ocean::Bathymetry b = ocean::average(c, space, mesh, o);
  double worst = 0;
  for (std::size_t r = 0; r < owned_rows(space); ++r) {
    const auto [lat, lon] = lat_lon(space, r);
    EXPECT_EQ(b.roughness(r, 0), 0);
    if (std::abs(lon) > 178) continue;  // the stencil there wraps through the seam, where the field jumps
    worst = std::max(worst, std::abs(b.H(r, 0) - depth(lat, lon)) / depth(lat, lon));
  }
  worst = c.allreduce(worst, soma::Op::max);
  EXPECT_LE(worst, bilinear_tolerance);
}

// A Gaussian seamount rising from 5000 m to 200 m. Masked diffusion moves H between neighbours and changes
// its integral only where the mask cuts the Laplacian's flux (3.5e-5 measured); 5% is the bound asked for.
constexpr double mass_tolerance = 0.05;

TEST(Bathymetry, smoothing_reaches_rmax) {
  const Comm c = Comm::world();
  HostMesh mesh = sem_test::sphere(c, 8, ocean::a);
  const ocean::Space space(c, mesh, 3, Geometry::sphere(ocean::a));
  const auto n = static_cast<std::size_t>(space.num_elements() * space.points_per_element());
  ocean::Bathymetry b;
  b.H = ocean::Values("H", n, 1);
  b.roughness = ocean::Values("roughness", n, 1);
  b.land.assign(n, 0);
  const auto x = space.x();
  const std::array<double, 3> top{std::cos(20 * deg) * std::cos(30 * deg), std::cos(20 * deg) * std::sin(30 * deg),
                                  std::sin(20 * deg)};
  constexpr double L = 800e3;
  for (std::size_t r = 0; r < n; ++r) {
    const double dot = (x(r, 0) * top[0] + x(r, 1) * top[1] + x(r, 2) * top[2]) / ocean::a;
    const double d = ocean::a * std::acos(std::clamp(dot, -1.0, 1.0));
    b.H(r, 0) = 5000 - 4800 * std::exp(-(d / L) * (d / L));
  }
  const ocean::Values initial("initial", n, 1);
  Kokkos::deep_copy(initial, b.H);
  const double before = ocean::max_rx0(c, space, b.H);
  const double mass0 = space.integral(c, b.H);
  ASSERT_GT(before, 0.2);
  ocean::smooth(c, space, b, ocean::BathymetryOptions{});
  const double after = ocean::max_rx0(c, space, b.H);
  double low = 1e300;
  for (std::size_t r = 0; r < owned_rows(space); ++r) low = std::min(low, b.H(r, 0));
  low = c.allreduce(low, soma::Op::min);
  const double mass1 = space.integral(c, b.H);
  if (c.rank() == 0) {
    std::printf("seamount: rx0 %.4f -> %.4f in %d sweeps, mass change %.3e\n", before, after, b.sweeps,
                std::abs(mass1 - mass0) / mass0);
  }
  EXPECT_TRUE(b.smoothed);
  EXPECT_GT(b.sweeps, 0);
  EXPECT_LE(after, 0.2);
  EXPECT_EQ(b.rx0, after);
  EXPECT_GE(low, 10);
  EXPECT_LT(std::abs(mass1 - mass0) / mass0, mass_tolerance);
  // The sweeps are masked to the steep elements and their neighbours: far from the seamount (beyond 5 L,
  // where it is 4800 exp(-25), flat to 1e-7 m) no point moves at all.
  std::int64_t far = 0;
  for (std::size_t r = 0; r < n; ++r) {
    const double dot = (x(r, 0) * top[0] + x(r, 1) * top[1] + x(r, 2) * top[2]) / ocean::a;
    if (ocean::a * std::acos(std::clamp(dot, -1.0, 1.0)) < 5 * L) continue;
    EXPECT_EQ(b.H(r, 0), initial(r, 0)) << "row " << r;
    ++far;
  }
  EXPECT_GT(c.allreduce(far, soma::Op::sum), 0);
}

// A 10 m shelf against 5000 m along latitude 33 (a coast after the H_min raise). Explicit diffusion of a jump
// overshoots: the first sweep pulls points next to it to -83 m (order 4), where rx0 of a pair whose sum is
// near or below 0 means nothing. And at order 4 a mask of the steep elements alone settles to a steady state
// (rx0 0.209, every sweep after the 5000th a no-op) that the sweep limit ends unsmoothed. Smoothing must reach
// r_max at either order, with H at least H_min, and report smoothed only when it did.
TEST(Bathymetry, smoothing_survives_undershoot_at_a_shelf) {
  const Comm c = Comm::world();
  const ocean::BathymetryOptions o;
  HostMesh mesh = sem_test::sphere(c, 4, ocean::a);
  for (const int order : {3, 4}) {
    const ocean::Space space(c, mesh, order, Geometry::sphere(ocean::a));
    const auto n = static_cast<std::size_t>(space.num_elements() * space.points_per_element());
    ocean::Bathymetry b;
    b.H = ocean::Values("H", n, 1);
    b.roughness = ocean::Values("roughness", n, 1);
    b.land.assign(n, 0);
    const auto x = space.x();
    const double shelf = std::sin(33 * deg) * ocean::a;
    for (std::size_t r = 0; r < n; ++r) b.H(r, 0) = x(r, 2) > shelf + 1 ? 5000 : 10;
    ocean::smooth(c, space, b, o);
    const double after = ocean::max_rx0(c, space, b.H);
    double low = 1e300;
    for (std::size_t r = 0; r < n; ++r) low = std::min(low, b.H(r, 0));
    low = c.allreduce(low, soma::Op::min);
    if (c.rank() == 0) {
      std::printf("shelf, order %d: rx0 %.4f in %d sweeps, smoothed %d\n", order, after, b.sweeps, b.smoothed);
    }
    EXPECT_EQ(b.rx0, after) << "order " << order;
    EXPECT_TRUE(b.smoothed) << "order " << order;
    if (b.smoothed) EXPECT_LE(after, o.r_max) << "order " << order;
    EXPECT_GE(low, o.H_min) << "order " << order;
  }
}

TEST(Bathymetry, smoothing_raises_land_and_stops_at_the_limit) {
  const Comm c = Comm::world();
  HostMesh mesh = sem_test::sphere(c, 4, ocean::a);
  const ocean::Space space(c, mesh, 3, Geometry::sphere(ocean::a));
  const auto n = static_cast<std::size_t>(space.num_elements() * space.points_per_element());
  ocean::Bathymetry b;
  b.H = ocean::Values("H", n, 1);
  b.roughness = ocean::Values("roughness", n, 1);
  b.land.assign(n, 0);
  const auto x = space.x();
  for (std::size_t r = 0; r < n; ++r) b.H(r, 0) = x(r, 2) > 0 ? 4000 : 0;  // land in the south
  ocean::BathymetryOptions o;
  o.sweep_limit = 2;
  ocean::smooth(c, space, b, o);
  EXPECT_FALSE(b.smoothed);
  EXPECT_EQ(b.sweeps, 2);
  EXPECT_GT(b.rx0, 0.2);
  for (std::size_t r = 0; r < n; ++r) EXPECT_GE(b.H(r, 0), 10);
  o.r_max = 0;
  EXPECT_THROW(ocean::smooth(c, space, b, o), std::invalid_argument);
}

// At 4 ranks each rank reads well under the file: its strips' columns only (a quarter of the cubed sphere's
// elements reach at most three quarters of the rows, and those rows only in part). On one rank every value
// is read once: nothing twice, the dropped periodic column never.
TEST(Bathymetry, window_bounded) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("bounded.h5");
  write_on_root(c, path, 181, 361, Registration::grid, Storage::int32, rough_heights);
  HostMesh mesh = sem_test::sphere(c, 4, ocean::a);
  const ocean::Space space(c, mesh, 3, Geometry::sphere(ocean::a));
  ocean::BathymetryOptions o;
  o.grid.file = path;
  o.strip_rows = 9;  // nine degrees of rows, as 540 rows of ETOPO1's arc minutes
  const std::size_t before = ocean::values_read();
  const ocean::Bathymetry b = ocean::average(c, space, mesh, o);
  const std::size_t read = ocean::values_read() - before;
  const std::size_t file = 181 * 361, grid = 181 * 360;
  std::printf("rank %d of %d: read %zu of %zu values, peak window %zu\n", c.rank(), c.size(), read, file,
              b.peak_window);
  EXPECT_LE(read, grid);
  EXPECT_GT(b.peak_window, 0U);
  EXPECT_LE(b.peak_window, read);
  if (c.size() == 1) EXPECT_EQ(read, grid);
  if (c.size() == 4) EXPECT_LT(static_cast<double>(read), 0.6 * static_cast<double>(file));
}

// The distributed result and the one-rank result (each rank builds the whole sphere on its own) by (element
// gid, point), compared bit for bit.
struct Compared {
  std::int64_t points = 0, pole = 0, seam = 0;
};

Compared compare_with_one_rank(const Comm& c, const fs::path& path, bool smoothed) {
  ocean::BathymetryOptions o;
  o.grid.file = path;
  o.strip_rows = 9;
  HostMesh mesh = sem_test::sphere(c, 4, ocean::a);
  const ocean::Space space(c, mesh, 3, Geometry::sphere(ocean::a));
  ocean::Bathymetry b = ocean::average(c, space, mesh, o);
  expect_continuous(space, b);
  const Comm self = Comm::self();
  HostMesh whole = sem_test::sphere(self, 4, ocean::a);
  const ocean::Space serial(self, whole, 3, Geometry::sphere(ocean::a));
  ocean::Bathymetry s = ocean::average(self, serial, whole, o);
  if (smoothed) {
    ocean::smooth(c, space, b, o);
    ocean::smooth(self, serial, s, o);
    EXPECT_EQ(b.sweeps, s.sweeps);
    EXPECT_EQ(b.rx0, s.rx0);
    EXPECT_EQ(b.smoothed, s.smoothed);
  }
  const auto expected = by_gid(serial, whole, s);
  const auto got = by_gid(space, mesh, b);
  Compared n;
  for (const auto& [key, value] : got) {
    const auto it = expected.find(key);
    if (it == expected.end()) {
      ADD_FAILURE() << "gid " << key.first << " point " << key.second << " missing from the one-rank result";
      continue;
    }
    EXPECT_EQ(value.H, it->second.H) << key.first << " " << key.second;
    EXPECT_EQ(value.roughness, it->second.roughness) << key.first << " " << key.second;
    EXPECT_EQ(value.land, it->second.land) << key.first << " " << key.second;
    ++n.points;
  }
  for (std::size_t r = 0; r < owned_rows(space); ++r) {
    const auto [lat, lon] = lat_lon(space, r);
    const double rho = footprint(space, r) / deg;
    if (lat + rho >= 90) {
      ++n.pole;
      // The footprint over the pole is averaged (it varies in longitude), not interpolated.
      if (!smoothed) {
        EXPECT_GT(b.roughness(r, 0), 0) << "row " << r;
      }
    }
    // A disc clear of the poles reaching across longitude 180.
    const bool clear = lat + rho < 90 && lat - rho > -90;
    if (clear && std::abs(lon) + rho / std::cos(lat * deg) > 180) ++n.seam;
  }
  n.points = c.allreduce(n.points, soma::Op::sum);
  n.pole = c.allreduce(n.pole, soma::Op::sum);
  n.seam = c.allreduce(n.seam, soma::Op::sum);
  return n;
}

TEST(Bathymetry, pole_and_antimeridian) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("pole.h5");
  write_on_root(c, path, 181, 361, Registration::grid, Storage::int32, rough_heights);
  const Compared n = compare_with_one_rank(c, path, false);
  EXPECT_EQ(n.points, 6 * 4 * 4 * 16);
  EXPECT_GT(n.pole, 0);
  EXPECT_GT(n.seam, 0);
}

TEST(Bathymetry, rank_independent) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("ranks.h5");
  write_on_root(c, path, 181, 361, Registration::grid, Storage::int32, rough_heights);
  const Compared n = compare_with_one_rank(c, path, true);
  EXPECT_EQ(n.points, 6 * 4 * 4 * 16);
}

TEST(Bathymetry, rejects_a_plane) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("plane.h5");
  write_on_root(c, path, 19, 37, Registration::grid, Storage::int32, rough_heights);
  HostMesh mesh = sem_test::box(c, 2, 2, 1, 1);
  const ocean::Space space(c, mesh, 2, Geometry::plane());
  ocean::BathymetryOptions o;
  o.grid.file = path;
  EXPECT_THROW(ocean::average(c, space, mesh, o), std::invalid_argument);
}

#else

TEST(Bathymetry, WithoutHdf5Throws) {
  const Comm c = Comm::world();
  HostMesh mesh = sem_test::sphere(c, 2, ocean::a);
  const ocean::Space space(c, mesh, 2, Geometry::sphere(ocean::a));
  ocean::BathymetryOptions o;
  o.grid.file = "etopo.nc";
  EXPECT_THROW(ocean::average(c, space, mesh, o), std::runtime_error);
}

#endif

}  // namespace
