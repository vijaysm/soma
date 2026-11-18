// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Reference files in any orientation (latitudes either way, longitudes from 0
// or from -180, with or without a time dimension) read to the same grid, the
// model's field compared on it, and bad files reported on every rank.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

#include <gtest/gtest.h>

#include "constants.hpp"
#include "reference.hpp"
#include "sem_meshes.hpp"

namespace {

using soma::Comm;
constexpr double pi = std::numbers::pi;

// Smooth on the sphere: 1000 + 100 z + 50 x on the unit sphere.
double field(double lat, double lon) {
  const double t = lat * pi / 180, l = lon * pi / 180;
  return 1000 + 100 * std::sin(t) + 50 * std::cos(t) * std::cos(l);
}

void write_dataset(hid_t file, const char* name, const std::vector<hsize_t>& dims, const std::vector<double>& v) {
  const hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
  const hid_t set = H5Dcreate2(file, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(set, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
  H5Dclose(set);
  H5Sclose(space);
}

struct Grid {
  std::vector<double> lat, lon;
};

// The canonical grid: lat -80..80 every 20, lon 0..330 every 30.
Grid canonical() {
  Grid g;
  for (int i = 0; i <= 8; ++i) g.lat.push_back(-80.0 + 20 * i);
  for (int j = 0; j < 12; ++j) g.lon.push_back(30.0 * j);
  return g;
}

enum class Variant { plain, north_to_south, from_minus_180, timed };

// Rank 0 writes the canonical field in the variant's layout (values taken from
// the canonical grid, so every variant holds the same bits); times > 1 adds a
// leading time dimension whose other slices are 0.
std::filesystem::path write_file(const Comm& c, const std::string& tag, Variant variant, int times = 1,
                                 int slice = -1) {
  const auto path = std::filesystem::path(::testing::TempDir()) /
                    ("swe_reference_" + tag + "_" + std::to_string(c.size()) + ".nc");
  if (c.rank() == 0) {
    const Grid g = canonical();
    std::vector<std::size_t> rows(g.lat.size()), cols(g.lon.size());
    for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = variant == Variant::north_to_south ? rows.size() - 1 - i : i;
    for (std::size_t j = 0; j < cols.size(); ++j) cols[j] = variant == Variant::from_minus_180 ? (j + 6) % 12 : j;
    std::vector<double> lat, lon, values;
    for (const std::size_t i : rows) lat.push_back(g.lat[i]);
    for (const std::size_t j : cols) lon.push_back(variant == Variant::from_minus_180 ? g.lon[j] - (j >= 6 ? 360 : 0) : g.lon[j]);
    const int last = slice < 0 ? times - 1 : slice;
    for (int t = 0; t < times; ++t) {
      for (const std::size_t i : rows) {
        for (const std::size_t j : cols) values.push_back(t == last ? field(g.lat[i], g.lon[j]) : 0.0);
      }
    }
    const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    write_dataset(file, "lat", {lat.size()}, lat);
    write_dataset(file, "lon", {lon.size()}, lon);
    if (times > 1 || variant == Variant::timed) {
      write_dataset(file, "h", {static_cast<hsize_t>(times), lat.size(), lon.size()}, values);
    } else {
      write_dataset(file, "h", {lat.size(), lon.size()}, values);
    }
    write_dataset(file, "line", {lat.size()}, lat);
    H5Fclose(file);
  }
  c.barrier();
  return path;
}

// Every rank done with the files, rank 0 removes them.
void remove_files(const Comm& c, const std::vector<std::filesystem::path>& paths) {
  c.barrier();
  if (c.rank() == 0) {
    for (const auto& p : paths) std::filesystem::remove(p);
  }
}

void expect_same(const swe::Reference& a, const swe::Reference& b, const std::string& what) {
  const auto bits = [](const std::vector<double>& v) {
    std::vector<std::uint64_t> out;
    for (const double x : v) out.push_back(std::bit_cast<std::uint64_t>(x));
    return out;
  };
  EXPECT_EQ(bits(a.lat), bits(b.lat)) << what;
  EXPECT_EQ(bits(a.lon), bits(b.lon)) << what;
  EXPECT_EQ(bits(a.values), bits(b.values)) << what;
}

TEST(SweReference, ReadsAnyOrientation) {
  const Comm c = Comm::world();
  swe::ReferenceSpec spec;
  spec.file = write_file(c, "plain", Variant::plain);
  const swe::Reference base = swe::read_reference(c, spec);
  const Grid g = canonical();
  EXPECT_EQ(base.lat, g.lat);
  EXPECT_EQ(base.lon, g.lon);
  ASSERT_EQ(base.values.size(), g.lat.size() * g.lon.size());
  EXPECT_EQ(base.values[13], field(g.lat[1], g.lon[1]));

  spec.file = write_file(c, "north", Variant::north_to_south);
  expect_same(swe::read_reference(c, spec), base, "north to south");
  spec.file = write_file(c, "west", Variant::from_minus_180);
  expect_same(swe::read_reference(c, spec), base, "from -180");
  spec.file = write_file(c, "timed", Variant::timed, 2);
  expect_same(swe::read_reference(c, spec), base, "time, the last");
  spec.time_index = 1;
  expect_same(swe::read_reference(c, spec), base, "time 1");
  spec.file = write_file(c, "first", Variant::timed, 2, 0);
  spec.time_index = 0;
  expect_same(swe::read_reference(c, spec), base, "time 0");
  std::vector<std::filesystem::path> paths;
  for (const char* tag : {"plain", "north", "west", "timed", "first"}) {
    paths.push_back(std::filesystem::path(::testing::TempDir()) /
                    ("swe_reference_" + std::string(tag) + "_" + std::to_string(c.size()) + ".nc"));
  }
  remove_files(c, paths);
}

TEST(SweReference, ComparesAgainstTheModelField) {
  const Comm c = Comm::world();
  const int ne = 4;
  soma::HostMesh m = sem_test::sphere(c, ne, swe::radius);
  // p = 8: the projected map makes even a linear field an interpolation at
  // p = 6 good to 4e-9 relative.
  const swe::Space s(c, m, 8, soma::sem::SemGeometry::sphere(swe::radius));
  const auto n = static_cast<std::size_t>(s.num_elements() * s.points_per_element());
  swe::Values h("h", n, 1);
  const auto x = s.x();
  for (std::size_t r = 0; r < n; ++r) {
    h(r, 0) = 1000 + 100 * x(r, 2) / swe::radius + 50 * x(r, 0) / swe::radius;
  }
  swe::Reference ref;
  for (int i = 0; i <= 60; ++i) ref.lat.push_back(-90.0 + 3 * i);
  for (int j = 0; j < 120; ++j) ref.lon.push_back(3.0 * j);
  for (const double t : ref.lat) {
    for (const double l : ref.lon) ref.values.push_back(field(t, l));
  }
  const swe::Norms e = swe::compare(c, s, m, ne, h, ref);
  EXPECT_LE(e.l2, 1e-9);
  EXPECT_LE(e.l1, 1e-9);
  EXPECT_LE(e.linf, 1e-9);
  // A wrong field is seen: 1 m off everywhere is 1e-3 relative.
  for (std::size_t r = 0; r < n; ++r) h(r, 0) += 1;
  EXPECT_NEAR(swe::compare(c, s, m, ne, h, ref).l1, 1.0 / 1000, 1e-4);
}

TEST(SweReference, ReportsBadFiles) {
  const Comm c = Comm::world();
  const std::filesystem::path timed = write_file(c, "bad", Variant::timed, 2);
  for (int k = 0; k < 4; ++k) {
    swe::ReferenceSpec spec;
    spec.file = timed;
    if (k == 0) spec.variable = "height";
    if (k == 1) spec.variable = "line";
    if (k == 2) spec.time_index = 5;
    if (k == 3) spec.file = timed.string() + ".missing";
    try {
      (void)swe::read_reference(c, spec);
      ADD_FAILURE() << "case " << k << " read";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find(spec.file.string()), std::string::npos) << e.what();
    }
  }
  remove_files(c, {timed});
}

}  // namespace
