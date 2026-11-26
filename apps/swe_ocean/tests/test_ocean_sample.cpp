// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Sampling the ocean's space at any point and the transport through a meridian (sample.hpp): exact for fields the
// elements represent exactly, NaN outside the mesh, and the same bits on any number of ranks.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <vector>

#include <gtest/gtest.h>

#include "constants.hpp"
#include "sample.hpp"
#include "sem_meshes.hpp"
#include "soma/parallel/comm.hpp"

namespace {

using P3 = std::array<double, 3>;
using soma::Comm;

// A field bilinear in x and y: the elements of a box (bilinear maps) hold it exactly at any order.
double bilinear(double x, double y) { return 1 + 2e-6 * x + 3e-6 * y + 5e-13 * x * y; }

// Points over the box [0, 4e6] x [0, 3e6] of 4 x 3 elements: inside, on edges and corners, and outside.
std::vector<P3> box_points() {
  std::vector<P3> p;
  for (int i = 0; i <= 16; ++i) {
    for (int j = 0; j <= 12; ++j) p.push_back({0.25e6 * i + 1234.5 * (i % 3), 0.25e6 * j + 777.25 * (j % 2), 0});
  }
  for (const P3 q : {P3{-1e5, 1e6, 0}, P3{4.2e6, 1e6, 0}, P3{2e6, -1, 0}, P3{2e6, 3.5e6, 0}}) p.push_back(q);
  return p;
}

std::vector<double> sample_box(const Comm& c) {
  soma::HostMesh mesh = sem_test::box(c, 4, 3, 4e6, 3e6);
  const ocean::Space space(c, mesh, 3, soma::sem::SemGeometry::plane());
  const auto x = space.x();
  ocean::Values v("v", x.extent(0), 2);
  for (std::size_t r = 0; r < x.extent(0); ++r) {
    v(r, 0) = bilinear(x(r, 0), x(r, 1));
    v(r, 1) = -v(r, 0);
  }
  const ocean::Sampler sampler(space, mesh);
  const std::vector<P3> points = box_points();
  return sampler.sample(c, v, points);
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

TEST(Sampler, exact_in_the_plane_and_nan_outside) {
  const Comm c = Comm::world();
  const std::vector<double> v = sample_box(c);
  const std::vector<P3> points = box_points();
  ASSERT_EQ(v.size(), 2 * points.size());
  for (std::size_t q = 0; q < points.size(); ++q) {
    const P3& p = points[q];
    const bool inside = p[0] >= 0 && p[0] <= 4e6 && p[1] >= 0 && p[1] <= 3e6;
    if (inside) {
      const double f = bilinear(p[0], p[1]);
      EXPECT_NEAR(v[2 * q], f, 1e-13 * std::abs(f)) << "point " << q;
      EXPECT_EQ(v[2 * q + 1], -v[2 * q]) << "point " << q;
    } else {
      EXPECT_TRUE(std::isnan(v[2 * q])) << "point " << q;
      EXPECT_TRUE(std::isnan(v[2 * q + 1])) << "point " << q;
    }
  }
}

TEST(Sampler, rank_independent) {
  const Comm c = Comm::world();
  const std::vector<double> v = sample_box(c);
  if (c.size() > 1) {
    const std::vector<double> serial = sample_box(Comm::self());
    EXPECT_TRUE(same_bits(v, serial));
  }
}

// Uniform eastward flow U over depth D on the cubed sphere of the Earth: the transport through a meridian from
// 60 S to 45 N is D U a (105 degrees in radians), all of it wet. (The section keeps out of the elements touching the
// poles, which reach 65.8 degrees at ne = 4: eastward has no limit at a pole.)
ocean::Transport sphere_transport(const Comm& c, bool land_band = false) {
  constexpr double D = 1000, U = 0.1;
  soma::HostMesh mesh = sem_test::sphere(c, 4, ocean::a);
  const ocean::Space space(c, mesh, 4, soma::sem::SemGeometry::sphere(ocean::a));
  const auto x = space.x();
  const std::size_t n = x.extent(0);
  ocean::Values H("H", n, 1), eta("eta", n, 1), u("u", n, 3);
  for (std::size_t r = 0; r < n; ++r) {
    H(r, 0) = D - 1;
    eta(r, 0) = 1;
    const double rho = std::hypot(x(r, 0), x(r, 1));
    u(r, 0) = -U * x(r, 1) / rho;
    u(r, 1) = U * x(r, 0) / rho;
    u(r, 2) = 0;
  }
  const ocean::Sampler sampler(space, mesh);
  ocean::Section s;
  s.lat_south = -60;
  s.lat_north = 45;
  s.lat_start = 0;
  ocean::Values land("land", land_band ? n : 0, 1);
  if (land_band) {
    for (std::size_t r = 0; r < n; ++r) {
      const double lat = std::asin(x(r, 2) / ocean::a) * 180 / std::numbers::pi;
      land(r, 0) = lat >= -30 && lat <= -20 ? 1 : 0;
    }
  }
  return ocean::meridian_transport(c, sampler, space, H, eta, u, land, s);
}

TEST(MeridianTransport, uniform_flow) {
  const Comm c = Comm::world();
  const ocean::Transport t = sphere_transport(c);
  const double exact = 1000 * 0.1 * ocean::a * 105 * std::numbers::pi / 180 / 1e6;
  EXPECT_TRUE(t.wet);
  EXPECT_DOUBLE_EQ(t.lat_from, -60);
  EXPECT_NEAR(t.lat_to, 45, 1e-9);
  // e_east interpolated at order 4 on elements of 22.5 degrees: a few parts in 1e5.
  EXPECT_NEAR(t.sv, exact, 1e-4 * exact) << t.sv << " vs " << exact;
  if (c.size() > 1) {
    const ocean::Transport serial = sphere_transport(Comm::self());
    EXPECT_EQ(t.sv, serial.sv);
  }
  if (c.rank() == 0) std::printf("transport %.12g Sv, exact %.12g Sv\n", t.sv, exact);
}

// Land flags of 1 from 30 S to 20 S end the run between the last land and the first ocean GLL point (5.6 degrees
// apart at ne = 4, p = 4); the transport is that of the run.
TEST(MeridianTransport, land_flags_end_the_run) {
  const Comm c = Comm::world();
  const ocean::Transport t = sphere_transport(c, true);
  EXPECT_TRUE(t.wet);
  EXPECT_GT(t.lat_from, -20);
  EXPECT_LT(t.lat_from, -14);
  EXPECT_NEAR(t.lat_to, 45, 1e-9);
  const double exact = 1000 * 0.1 * ocean::a * (t.lat_to - t.lat_from) * std::numbers::pi / 180 / 1e6;
  EXPECT_NEAR(t.sv, exact, 1e-4 * exact) << t.sv << " vs " << exact;
  if (c.size() > 1) EXPECT_EQ(t.sv, sphere_transport(Comm::self(), true).sv);
  if (c.rank() == 0) std::printf("land from 30 S to 20 S: run %.2f..%.2f, %.12g Sv\n", t.lat_from, t.lat_to, t.sv);
}

TEST(MeridianTransport, rejects_bad_sections) {
  const Comm c = Comm::world();
  soma::HostMesh mesh = sem_test::sphere(c, 2, ocean::a);
  const ocean::Space space(c, mesh, 2, soma::sem::SemGeometry::sphere(ocean::a));
  const std::size_t n = space.x().extent(0);
  ocean::Values H("H", n, 1), eta("eta", n, 1), u("u", n, 3);
  const ocean::Sampler sampler(space, mesh);
  ocean::Section s;
  s.lat_start = -85;
  EXPECT_THROW(ocean::meridian_transport(c, sampler, space, H, eta, u, {}, s), std::invalid_argument);
  s = ocean::Section{};
  s.spacing = 0;
  EXPECT_THROW(ocean::meridian_transport(c, sampler, space, H, eta, u, {}, s), std::invalid_argument);
}

}  // namespace
