// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Point location on the cubed sphere, spectral sampling on a latitude-longitude
// grid with poles and cube edges on it, the same bits on any number of ranks,
// and the .npy writer.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numbers>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "constants.hpp"
#include "latlon.hpp"
#include "sem_meshes.hpp"
#include "soma/fields/node_tables.hpp"

namespace {

using soma::Comm;
using soma::GO;
using soma::LO;

constexpr double pi = std::numbers::pi;

swe::Space make_space(const Comm& c, soma::HostMesh& m, int p) {
  return swe::Space(c, m, p, soma::sem::SemGeometry::sphere(swe::radius));
}

TEST(SweLatLon, LocatesGridPointsExactly) {
  const Comm c = Comm::world();
  const int ne = 3, p = 4, np = p + 1;
  soma::HostMesh m = sem_test::sphere(c, ne, swe::radius);
  const swe::Space s = make_space(c, m, p);
  const std::vector<double> xi = soma::gll_points_1d(p);
  const auto x = s.x();
  const auto gid = m.gid(2);
  for (LO e = 0; e < s.owned_elements(); ++e) {
    for (int j = 0; j < np; ++j) {
      for (int i = 0; i < np; ++i) {
        const LO r = e * np * np + i + np * j;
        const std::array<double, 3> at = {x(r, 0) / swe::radius, x(r, 1) / swe::radius, x(r, 2) / swe::radius};
        const swe::Located l = swe::locate(ne, at);
        const bool interior = i > 0 && i < p && j > 0 && j < p;
        if (interior) EXPECT_EQ(l.gid, gid(e)) << "element " << gid(e) << " point " << i << " " << j;
        EXPECT_LE(l.gid, gid(e));
        if (l.gid == gid(e)) {
          EXPECT_NEAR(l.xi, xi[static_cast<std::size_t>(i)], 1e-12);
          EXPECT_NEAR(l.eta, xi[static_cast<std::size_t>(j)], 1e-12);
        }
        const auto back = swe::element_point(ne, l.gid, l.xi, l.eta);
        for (int k = 0; k < 3; ++k) EXPECT_NEAR(back[static_cast<std::size_t>(k)], at[static_cast<std::size_t>(k)], 1e-12);
      }
    }
  }
}

// lat -90..90 every 2 degrees and lon 0..359 every degree: both poles (face
// centres, element corners at even ne), the cube's edges (lon 45, 135, ... at
// the equator; lat 45 on lon 0, 90, ...) and element edges at every ne.
std::vector<double> lats() {
  std::vector<double> v;
  for (int i = 0; i <= 90; ++i) v.push_back(-90.0 + 2 * i);
  return v;
}

std::vector<double> lons() {
  std::vector<double> v;
  for (int i = 0; i < 360; ++i) v.push_back(1.0 * i);
  return v;
}

TEST(SweLatLon, SamplesSmoothFieldsSpectrally) {
  const Comm c = Comm::world();
  const int ne = 4;
  soma::HostMesh m = sem_test::sphere(c, ne, swe::radius);
  const swe::Space s = make_space(c, m, 8);
  const auto n = static_cast<std::size_t>(s.num_elements() * s.points_per_element());
  swe::Values v("v", n, 2);
  const auto x = s.x();
  const auto f = [](double a, double b, double z) { return a * b + z * z * z; };
  for (std::size_t r = 0; r < n; ++r) {
    v(r, 1) = f(x(r, 0) / swe::radius, x(r, 1) / swe::radius, x(r, 2) / swe::radius);
  }
  const std::vector<double> lat = lats(), lon = lons();
  const std::vector<double> got = swe::sample(c, s, m, ne, v, 1, lat, lon);
  ASSERT_EQ(got.size(), lat.size() * lon.size());
  double worst = 0;
  for (std::size_t i = 0; i < lat.size(); ++i) {
    for (std::size_t j = 0; j < lon.size(); ++j) {
      const double t = lat[i] * pi / 180, l = lon[j] * pi / 180;
      const double want = f(std::cos(t) * std::cos(l), std::cos(t) * std::sin(l), std::sin(t));
      worst = std::max(worst, std::abs(got[i * lon.size() + j] - want));
    }
  }
  EXPECT_LE(worst, 1e-8);
}

TEST(SweLatLon, SameOnAnyRankCount) {
  for (const int ne : {1, 3}) {
    const auto run = [ne](const Comm& c) {
      soma::HostMesh m = sem_test::sphere(c, ne, swe::radius);
      const swe::Space s = make_space(c, m, 3);
      const auto n = static_cast<std::size_t>(s.num_elements() * s.points_per_element());
      swe::Values v("v", n, 1);
      const auto x = s.x();
      const auto gid = m.gid(2);
      for (std::size_t r = 0; r < n; ++r) {
        v(r, 0) = std::sin(x(r, 0) / swe::radius) + static_cast<double>(gid(static_cast<LO>(r) / s.points_per_element()));
      }
      return swe::sample(c, s, m, ne, v, 0, lats(), lons());
    };
    const std::vector<double> a = run(Comm::world()), b = run(Comm::self());
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t k = 0; k < a.size(); ++k) {
      ASSERT_EQ(std::bit_cast<std::uint64_t>(a[k]), std::bit_cast<std::uint64_t>(b[k])) << "ne " << ne << " point " << k;
    }
  }
}

TEST(SweLatLon, NpyHeader) {
  const Comm c = Comm::world();
  const auto path = std::filesystem::path(::testing::TempDir()) /
                    ("swe_latlon_" + std::to_string(c.rank()) + "_" + std::to_string(c.size()) + ".npy");
  const std::vector<double> data = {1, 2, 3, 4, 5, 6};
  swe::write_npy(path, data, 2, 3);
  std::ifstream in(path, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::filesystem::remove(path);
  ASSERT_GE(bytes.size(), 10U);
  EXPECT_EQ(bytes.substr(0, 8), std::string("\x93NUMPY\x01\x00", 8));
  const std::size_t header = static_cast<unsigned char>(bytes[8]) + 256U * static_cast<unsigned char>(bytes[9]);
  EXPECT_EQ((10 + header) % 64, 0U);
  const std::string dict = bytes.substr(10, header);
  EXPECT_NE(dict.find("'<f8'"), std::string::npos) << dict;
  EXPECT_NE(dict.find("False"), std::string::npos) << dict;
  EXPECT_NE(dict.find("(2, 3)"), std::string::npos) << dict;
  EXPECT_EQ(dict.back(), '\n');
  ASSERT_EQ(bytes.size(), 10 + header + 8 * data.size());
  double last = 0;
  std::memcpy(&last, bytes.data() + bytes.size() - 8, 8);
  EXPECT_EQ(last, 6.0);
  EXPECT_THROW(swe::write_npy(path, data, 4, 2), std::invalid_argument);
}

}  // namespace
