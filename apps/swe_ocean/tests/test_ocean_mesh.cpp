// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The ocean mesh from a synthetic Earth: two continents, an inland lake and a basin that meets the world ocean
// only at an element corner. The lake and the basin go, the coast is refined, the result is the same by gid on
// any number of ranks and survives a SOMA file written at one rank count and read at another; a seed on land
// is an error on every rank.
//
// The geography (degrees; ocean 1000 m deep, the lake and the basin 500 m, land 500 m high) is laid on the
// cubed sphere at ne = 4, whose face about +x has its centre V = (0, 0) at a vertex and the mesh lines lon = 0
// and lat = 0 (great circles) through it, coarse cells 22.5 degrees wide. Continent A is the box |lat|, |lon|
// <= 40 about V less
//   - the bay {lat > 4, lon > 4}, open to the ocean across the box's north and east sides, widened by
//     {lon > 18, lat > -18} and {lat > 18, lon > -18}: those coasts cross the coarse cells south-east and
//     north-west of V, so all four coarse cells at V hold coast and are refined (trisected; at 400 km every
//     cell is, by promotion) and V is the corner of four children 7.5 degrees wide, one per quadrant;
//   - the basin {-16 < lat < -4, -16 < lon < -4}.
// The children at V south-east and north-west of it lie 4 degrees or more from any water: their points'
// footprints (radii of 1.6 degrees at most, order 3 on 7.5-degree cells) see land only, so they are land
// elements and go. The child north-east of V reaches into the bay and the one south-west into the basin: both
// are ocean elements, and they touch at V alone. The basin is otherwise ringed by land elements, so it is cut
// off from the world ocean by a strait one corner wide and goes as disconnected. Continent B is the box
// |lat| <= 35, |lon - 90| <= 35 with the lake: the disc of radius 8 degrees about (0, 90), ringed by more than
// 20 degrees of land. The seed (0, -150) is open ocean; seeded in the basin instead, the mesh is the basin.

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "coastline.hpp"
#include "constants.hpp"
#include "grid_files.hpp"
#include "ocean_mesh.hpp"
#include "soma/algorithms/refine_graded.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/cubed_sphere.hpp"
#include "synthetic_earth.hpp"

#ifdef SWE_OCEAN_HAVE_HDF5

namespace {

namespace fs = std::filesystem;
using ocean::testing::angle_deg;
using ocean::testing::at;
using ocean::testing::deg;
using ocean::testing::heights;
using ocean::testing::in_basin;
using ocean::testing::in_lake;
using ocean::testing::lake_radius;
using ocean::testing::Registration;
using ocean::testing::Storage;
using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;

// Latitude and longitude (degrees) of a point at any radius.
std::pair<double, double> lat_lon(const double* x) {
  const double r = std::hypot(x[0], x[1], x[2]);
  return {std::asin(x[2] / r) / deg, std::atan2(x[1], x[0]) / deg};
}

// The centroid of owned element e's corners, as latitude and longitude.
std::pair<double, double> centre(const HostMesh& m, LO e) {
  const auto view = m.view();
  const auto x = m.coords();
  double c[3] = {0, 0, 0};
  for (LO q = 0; q < 4; ++q) {
    const LO v = view.vertex(2, e, q);
    for (int k = 0; k < 3; ++k) c[k] += x(v, k) / 4;
  }
  return lat_lon(c);
}

// A directory for this run's files, the same on every rank, removed by rank 0 at the end.
fs::path scratch(const Comm& c) {
  std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
  c.bcast(std::span<std::int64_t>(&id, 1), 0);
  return fs::temp_directory_path() / ("ocean_mesh_" + std::to_string(c.size()) + "_" + std::to_string(id));
}

// What a mesh holds by gid: each owned element's corner coordinates and its points' H, roughness and land.
struct Element {
  std::array<double, 12> corners{};
  std::vector<double> H, roughness;
  std::vector<std::uint8_t> land;
};

std::map<GO, Element> by_gid(const ocean::OceanMesh& o) {
  std::map<GO, Element> out;
  const HostMesh& m = *o.mesh;
  const auto view = m.view();
  const auto x = m.coords();
  const auto gid = m.gid(2);
  const LO ppe = o.space->points_per_element();
  for (LO e = 0; e < o.space->owned_elements(); ++e) {
    Element& el = out[gid(e)];
    for (LO q = 0; q < 4; ++q) {
      for (int k = 0; k < 3; ++k) el.corners[static_cast<std::size_t>(3 * q + k)] = x(view.vertex(2, e, q), k);
    }
    for (LO k = 0; k < ppe; ++k) {
      const auto r = static_cast<std::size_t>(e * ppe + k);
      el.H.push_back(o.bathymetry.H(r, 0));
      el.roughness.push_back(o.bathymetry.roughness(r, 0));
      el.land.push_back(o.bathymetry.land[r]);
    }
  }
  return out;
}

// Every owned element of `got` equals the element of that gid in `expected` (a whole mesh), and every element
// of `expected` is owned on some rank of c.
void expect_same(const Comm& c, const std::map<GO, Element>& got, const std::map<GO, Element>& expected) {
  for (const auto& [g, el] : got) {
    const auto it = expected.find(g);
    if (it == expected.end()) {
      ADD_FAILURE() << "element " << g << " is not in the expected mesh";
      continue;
    }
    EXPECT_EQ(el.corners, it->second.corners) << "element " << g;
    EXPECT_EQ(el.H, it->second.H) << "element " << g;
    EXPECT_EQ(el.roughness, it->second.roughness) << "element " << g;
    EXPECT_EQ(el.land, it->second.land) << "element " << g;
  }
  EXPECT_EQ(c.allreduce(static_cast<std::int64_t>(got.size()), soma::Op::sum),
            static_cast<std::int64_t>(expected.size()));
}

// The ghost elements of `o` hold, point for point, the values of their owners: those of `expected` (a whole
// mesh) by gid.
void expect_ghosts(const ocean::OceanMesh& o, const std::map<GO, Element>& expected) {
  const auto gid = o.mesh->gid(2);
  const LO ppe = o.space->points_per_element();
  for (LO e = o.space->owned_elements(); e < o.space->num_elements(); ++e) {
    const auto it = expected.find(gid(e));
    ASSERT_NE(it, expected.end()) << "ghost " << gid(e);
    for (LO k = 0; k < ppe; ++k) {
      const auto r = static_cast<std::size_t>(e * ppe + k);
      const auto j = static_cast<std::size_t>(k);
      EXPECT_EQ(o.bathymetry.H(r, 0), it->second.H[j]) << "ghost " << gid(e) << " point " << k;
      EXPECT_EQ(o.bathymetry.roughness(r, 0), it->second.roughness[j]) << "ghost " << gid(e) << " point " << k;
      EXPECT_EQ(o.bathymetry.land[r], it->second.land[j]) << "ghost " << gid(e) << " point " << k;
    }
  }
}

class OceanMesh : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const Comm c = Comm::world();
    dir_ = scratch(c);
    if (c.rank() == 0) {
      fs::create_directories(dir_);
      // Half a degree: footprints of 70 km and more hold several cells.
      ocean::testing::write_grid_file(dir_ / "earth.h5", 361, 721, Registration::grid, true, Storage::int32,
                                      heights);
    }
    c.barrier();
    built_ = std::make_unique<ocean::OceanMesh>(ocean::build_ocean_mesh(c, options()));
  }

  static void TearDownTestSuite() {
    built_.reset();
    const Comm c = Comm::world();
    c.barrier();
    if (c.rank() == 0) {
      std::error_code ec;
      fs::remove_all(dir_, ec);
    }
  }

  static ocean::MeshOptions options() {
    ocean::MeshOptions o;
    o.ne = 4;
    o.order = 3;
    o.coast_km = 400;
    o.bathymetry.grid.file = dir_ / "earth.h5";
    return o;
  }

  static fs::path dir_;
  static std::unique_ptr<ocean::OceanMesh> built_;
};

fs::path OceanMesh::dir_;
std::unique_ptr<ocean::OceanMesh> OceanMesh::built_;

TEST_F(OceanMesh, lake_and_corner_basin_removed) {
  const Comm c = Comm::world();
  const HostMesh& m = *built_->mesh;
  // Elements near V by quadrant: north-east (the bay), south-east, south-west (the basin), north-west.
  std::array<std::int64_t, 4> near_v{0, 0, 0, 0};
  std::int64_t basin = 0, lake = 0;
  for (LO e = 0; e < built_->space->owned_elements(); ++e) {
    const auto [lat, lon] = centre(m, e);
    basin += in_basin(lat, lon) ? 1 : 0;
    lake += in_lake(lat, lon) ? 1 : 0;
    if (std::abs(lat) < 7.5 && std::abs(lon) < 7.5) {
      const int quadrant = lat > 0 ? (lon > 0 ? 0 : 3) : (lon > 0 ? 1 : 2);
      ++near_v[static_cast<std::size_t>(quadrant)];
    }
  }
  EXPECT_EQ(c.allreduce(basin, soma::Op::sum), 0);
  EXPECT_EQ(c.allreduce(lake, soma::Op::sum), 0);
  c.allreduce(std::span<std::int64_t>(near_v), soma::Op::sum);
  EXPECT_EQ(near_v[0], 1) << "the bay's element at V";
  EXPECT_EQ(near_v[1] + near_v[2] + near_v[3], 0);
  // V is still a vertex of the mesh, the corner where the basin met the bay.
  double nearest = 180;
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) {
    const double p[3] = {x(v, 0), x(v, 1), x(v, 2)};
    const auto [lat, lon] = lat_lon(p);
    nearest = std::min(nearest, angle_deg(at(lat, lon), at(0, 0)));
  }
  EXPECT_LT(c.allreduce(nearest, soma::Op::min), 0.5);
  EXPECT_GT(built_->removed_land_km2, 0);
  // At least the basin's and the lake's own water, which lies in the elements removed as disconnected.
  const double km = ocean::a / 1000;
  const double basin_km2 = km * km * (12 * deg) * (std::sin(-4 * deg) - std::sin(-16 * deg));
  const double lake_km2 = 2 * std::numbers::pi * km * km * (1 - std::cos(lake_radius * deg));
  EXPECT_GT(built_->removed_disconnected_km2, basin_km2 + lake_km2);
  if (c.rank() == 0) {
    std::printf("removed: land %.6e km^2, disconnected %.6e km^2 (basin %.3e, lake %.3e)\n",
                built_->removed_land_km2, built_->removed_disconnected_km2, basin_km2, lake_km2);
  }
}

// The same Earth seeded in the basin keeps the basin alone: its elements are ocean, joined to one another and
// to nothing else through edges, and one of them has V for a corner. With the world-ocean seed they all go
// (above), though that element touches the bay's at V: a corner does not join.
TEST_F(OceanMesh, basin_meets_bay_at_a_corner) {
  const Comm c = Comm::world();
  ocean::MeshOptions o = options();
  o.seed_lat = -10;
  o.seed_lon = -10;
  const ocean::OceanMesh basin = ocean::build_ocean_mesh(c, o);
  const HostMesh& m = *basin.mesh;
  std::int64_t outside = 0;
  for (LO e = 0; e < basin.space->owned_elements(); ++e) {
    const auto [lat, lon] = centre(m, e);
    outside += lat < -24 || lat > 0 || lon < -24 || lon > 0 ? 1 : 0;
  }
  EXPECT_EQ(c.allreduce(outside, soma::Op::sum), 0);
  // The nine children of the coarse cell south-west of V: each reaches the basin's water.
  EXPECT_EQ(c.allreduce(static_cast<std::int64_t>(basin.space->owned_elements()), soma::Op::sum), 9);
  double nearest = 180;
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) {
    const double p[3] = {x(v, 0), x(v, 1), x(v, 2)};
    const auto [lat, lon] = lat_lon(p);
    nearest = std::min(nearest, angle_deg(at(lat, lon), at(0, 0)));
  }
  EXPECT_LT(c.allreduce(nearest, soma::Op::min), 0.5);
}

// The levels of a built mesh against its coast: every owned cell within coast_km of a coastline point (the
// target refine_graded used, asked with that growth) is at the finest level, which is at least `least`. Returns
// the cells near the coast and the cells coarser than the finest level, over every rank.
std::pair<std::int64_t, std::int64_t> check_levels(const Comm& c, const ocean::OceanMesh& built,
                                                   const ocean::MeshOptions& o, std::int32_t least) {
  const auto points = ocean::coastline(c, o.bathymetry.grid, o.coast_km);
  EXPECT_FALSE(points.empty());
  EXPECT_EQ(built.coastline_points, points.size()) << "the build reports the coastline it refined toward";
  // The mesh on the unit sphere, as the target takes it.
  HostMesh unit = built.mesh->clone();
  const auto x = unit.coords();
  for (LO v = 0; v < unit.count(0); ++v) {
    for (int k = 0; k < 3; ++k) x(v, k) /= ocean::a;
  }
  const LO owned = unit.segment_table(2).owned_end;
  const std::vector<soma::Real> grow(static_cast<std::size_t>(owned), o.coast_km);
  const std::vector<std::uint8_t> near = ocean::coast_target(points)(unit, grow);
  const std::vector<std::int32_t> level = soma::refine_levels(unit);
  std::int32_t finest = 0;
  for (const std::int32_t l : level) finest = std::max(finest, l);
  finest = c.allreduce(finest, soma::Op::max);
  EXPECT_GE(finest, least);
  std::array<std::int64_t, 2> n{0, 0};
  for (LO e = 0; e < owned; ++e) {
    const auto i = static_cast<std::size_t>(e);
    n[1] += level[i] < finest ? 1 : 0;
    if (near[i] == 0) continue;
    ++n[0];
    EXPECT_EQ(level[i], finest) << "element " << unit.gid(2)(e);
  }
  c.allreduce(std::span<std::int64_t>(n), soma::Op::sum);
  return {n[0], n[1]};
}

// At ne = 4 and 400 km one trisection reaches the spacing (2300 km cells, 770 km GLL spacing, over 1.5 x 400
// km), and the 32 cells holding the coast, their corners marked and promoted, refine every cell: the mesh is
// uniformly at level 1.
TEST_F(OceanMesh, refined_near_coast) {
  const Comm c = Comm::world();
  const auto [near, coarser] = check_levels(c, *built_, options(), 1);
  EXPECT_GT(near, 0);
  EXPECT_GT(built_->refined.passes.refined, 0);
}

// At 150 km two trisections are needed near the coast (to 85 km): the levels grade, level 2 at the coast and
// coarser cells away from it.
TEST_F(OceanMesh, graded_toward_coast) {
  const Comm c = Comm::world();
  ocean::MeshOptions o = options();
  o.coast_km = 150;
  const ocean::OceanMesh built = ocean::build_ocean_mesh(c, o);
  const auto [near, coarser] = check_levels(c, built, o, 2);
  EXPECT_GT(near, 0);
  EXPECT_GT(coarser, 0);
}

TEST_F(OceanMesh, same_by_gid) {
  const Comm c = Comm::world();
  const Comm self = Comm::self();
  const ocean::OceanMesh serial = ocean::build_ocean_mesh(self, options());
  const std::map<GO, Element> expected = by_gid(serial);
  expect_same(c, by_gid(*built_), expected);
  expect_ghosts(*built_, expected);
  EXPECT_EQ(built_->removed_land_km2, serial.removed_land_km2);
  EXPECT_EQ(built_->removed_disconnected_km2, serial.removed_disconnected_km2);
  EXPECT_EQ(built_->bathymetry.sweeps, serial.bathymetry.sweeps);
  EXPECT_EQ(built_->bathymetry.rx0, serial.bathymetry.rx0);
}

TEST_F(OceanMesh, round_trip) {
  const Comm c = Comm::world();
  const fs::path path = dir_ / "ocean.soma";
  ocean::write_ocean_mesh(c, *built_, path);
  // The whole mesh as built, by gid, on every rank.
  const ocean::OceanMesh whole = ocean::read_ocean_mesh(Comm::self(), path, 3);
  const std::map<GO, Element> expected = by_gid(whole);
  expect_same(c, by_gid(*built_), expected);
  EXPECT_EQ(whole.removed_land_km2, built_->removed_land_km2);
  EXPECT_EQ(whole.coastline_points, 0U) << "a read mesh has no coastline";
  EXPECT_EQ(whole.removed_disconnected_km2, built_->removed_disconnected_km2);
  // A space of another order finds no matching fields: an error on every rank.
  EXPECT_THROW((void)ocean::read_ocean_mesh(c, path, 4), std::invalid_argument);
  // Read at the writing rank count and at three.
  const ocean::OceanMesh again = ocean::read_ocean_mesh(c, path, 3);
  expect_same(c, by_gid(again), expected);
  expect_ghosts(again, expected);
  const soma::SplitComm three = c.split(c.rank() < 3 ? 0 : -1, c.rank());
  if (three.member() && c.size() >= 3) {
    const ocean::OceanMesh read = ocean::read_ocean_mesh(three.comm(), path, 3);
    expect_same(three.comm(), by_gid(read), expected);
    expect_ghosts(read, expected);
    EXPECT_EQ(read.removed_disconnected_km2, built_->removed_disconnected_km2);
  }
  c.barrier();
}

// The land flags survive the removal by (element gid, point): with no refinement and no vertex smoothing the
// mesh before the removal is the cubed sphere at ne scaled to the Earth, whose first averaging (on one rank:
// the result is the same on any number) gives each kept element's flags; the built mesh holds the same flags,
// and its land points stay at exactly H_min through the smoothing.
TEST_F(OceanMesh, land_carried_by_gid) {
  const Comm c = Comm::world();
  ocean::MeshOptions o = options();
  o.coast_km = 0;
  o.optimize = {0, 0, 0};
  const ocean::OceanMesh built = ocean::build_ocean_mesh(c, o);
  const Comm self = Comm::self();
  soma::sem::CubedSphereSpec spec;
  spec.ne = o.ne;
  HostMesh whole = soma::sem::cubed_sphere_mesh(self, spec);
  const auto x = whole.coords();
  for (LO v = 0; v < whole.count(0); ++v) {
    for (int k = 0; k < 3; ++k) x(v, k) *= ocean::a;
  }
  const ocean::Space space(self, whole, o.order, soma::sem::SemGeometry::sphere(ocean::a));
  const ocean::Bathymetry first = ocean::average(self, space, whole, o.bathymetry);
  const LO ppe = space.points_per_element();
  std::map<GO, std::vector<std::uint8_t>> expected;
  const auto wgid = whole.gid(2);
  for (LO e = 0; e < space.owned_elements(); ++e) {
    auto& flags = expected[wgid(e)];
    for (LO k = 0; k < ppe; ++k) flags.push_back(first.land[static_cast<std::size_t>(e * ppe + k)]);
  }
  const auto gid = built.mesh->gid(2);
  std::int64_t land = 0, flooded = 0;
  for (LO e = 0; e < built.space->num_elements(); ++e) {
    const auto it = expected.find(gid(e));
    ASSERT_NE(it, expected.end()) << "element " << gid(e);
    for (LO k = 0; k < ppe; ++k) {
      const auto r = static_cast<std::size_t>(e * ppe + k);
      EXPECT_EQ(built.bathymetry.land[r], it->second[static_cast<std::size_t>(k)]) << gid(e) << " " << k;
      if (built.bathymetry.land[r] == 0) continue;
      if (e < built.space->owned_elements()) ++land;
      if (built.bathymetry.H(r, 0) != o.bathymetry.H_min) ++flooded;
    }
  }
  EXPECT_GT(c.allreduce(land, soma::Op::sum), 0);
  EXPECT_EQ(c.allreduce(flooded, soma::Op::sum), 0) << "land points not at H_min";
}

// A seed on land is refused on every rank, and the same Earth seeded in its ocean is not: a planet of land
// with an ocean north of 60 degrees, on the coarse cubed sphere (no refinement, quick), whose element at
// (0, 0), 22.5 degrees wide, holds no point within its footprint (5 degrees at most) of water.
TEST_F(OceanMesh, seed_on_land_throws) {
  const Comm c = Comm::world();
  const fs::path path = dir_ / "polar.h5";
  if (c.rank() == 0) {
    ocean::testing::write_grid_file(path, 181, 361, Registration::grid, true, Storage::int32,
                                    [](double lat, double) { return lat > 60 ? -1000.0 : 500.0; });
  }
  c.barrier();
  ocean::MeshOptions o = options();
  o.coast_km = 0;
  o.bathymetry.grid.file = path;
  o.seed_lat = 0;
  o.seed_lon = 0;
  try {
    (void)ocean::build_ocean_mesh(c, o);
    ADD_FAILURE() << "a seed on land was accepted";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("land"), std::string::npos) << e.what();
  }
  o.seed_lat = 80;
  const ocean::OceanMesh polar = ocean::build_ocean_mesh(c, o);
  EXPECT_GT(polar.removed_land_km2, 0);
  EXPECT_EQ(polar.removed_disconnected_km2, 0);
}

}  // namespace

#else

TEST(OceanMesh, needs_hdf5) { GTEST_SKIP() << "the ocean mesh tests write grid files with HDF5"; }

#endif
