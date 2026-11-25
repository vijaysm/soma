// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Latitude-longitude grid files read by window: the normalised grid of grid- and cell-registered files in
// either latitude order, windows read in strips with exactly the analytic values and counted, windows across
// the antimeridian, and clear errors on every rank.

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "grid_file.hpp"
#include "grid_files.hpp"
#include "soma/parallel/comm.hpp"

namespace {

namespace fs = std::filesystem;
using ocean::testing::Registration;
using ocean::testing::Storage;

// The test function: integer at the integer-degree grids, so int32 storage is exact.
double f(double lat, double lon) { return lat + 1000.0 * lon; }
// Values are compared at coordinates the writer computes as start + (end - start) i / n and the reader hands
// back unchanged: they agree to rounding of numbers below 2e5, a few 1e-11 at most; 1e-9 is above that and
// far below the unit steps of f between neighbours.
constexpr double value_tolerance = 1e-9;

// A directory for this run's files, the same on every rank; rank 0 removes it.
class Scratch {
 public:
  explicit Scratch(const soma::Comm& c) : c_(c) {
    std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
    c.bcast(std::span<std::int64_t>(&id, 1), 0);
    dir_ = fs::temp_directory_path() / ("ocean_grid_file_" + std::to_string(c.size()) + "_" + std::to_string(id));
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
  const soma::Comm& c_;
  fs::path dir_;
};

#ifdef SWE_OCEAN_HAVE_HDF5

// Rank 0 writes; every rank waits for the file.
void write_on_root(const soma::Comm& c, const fs::path& path, std::size_t nlat, std::size_t nlon, Registration reg,
                   bool south_first, Storage storage, const ocean::testing::GridNames& names = {}) {
  if (c.rank() == 0) ocean::testing::write_grid_file(path, nlat, nlon, reg, south_first, storage, f, names);
  c.barrier();
}

// Reads a window and checks every value against f at the normalised coordinates; returns the rows seen.
std::vector<std::size_t> check_window(const ocean::GridSpec& spec, const ocean::GridInfo& info,
                                      const ocean::Window& w, std::size_t strip) {
  std::vector<std::size_t> rows;
  ocean::read_window(spec, info, w, strip, [&](std::size_t i, std::span<const double> row) {
    rows.push_back(i);
    ASSERT_EQ(row.size(), w.nlon);
    for (std::size_t j = 0; j < w.nlon; ++j) {
      EXPECT_NEAR(row[j], f(info.lat[i], info.lon[w.lon0 + j]), value_tolerance) << i << " " << j;
    }
  });
  return rows;
}

TEST(OceanGridFile, GridRegisteredNorthFirstIsNormalised) {
  const soma::Comm c = soma::Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("grid_north.h5");
  write_on_root(c, path, 181, 361, Registration::grid, false, Storage::int32);
  const ocean::GridSpec spec{path};
  const ocean::GridInfo info = ocean::open_grid(c, spec);
  ASSERT_EQ(info.lat.size(), 181U);
  EXPECT_EQ(info.lat.front(), -90.0);
  EXPECT_EQ(info.lat.back(), 90.0);
  EXPECT_FALSE(info.south_first);
  ASSERT_EQ(info.lon.size(), 360U);  // the duplicate +180 column is dropped
  EXPECT_EQ(info.lon.front(), -180.0);
  EXPECT_EQ(info.lon.back(), 179.0);
  EXPECT_TRUE(info.periodic);
  EXPECT_DOUBLE_EQ(info.dlat, 1.0);
  EXPECT_DOUBLE_EQ(info.dlon, 1.0);
}

TEST(OceanGridFile, CellRegisteredSouthFirstIsNormalised) {
  const soma::Comm c = soma::Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("cell_south.h5");
  write_on_root(c, path, 180, 360, Registration::cell, true, Storage::float64);
  const ocean::GridInfo info = ocean::open_grid(c, ocean::GridSpec{path});
  ASSERT_EQ(info.lat.size(), 180U);
  EXPECT_DOUBLE_EQ(info.lat.front(), -89.5);
  EXPECT_DOUBLE_EQ(info.lat.back(), 89.5);
  EXPECT_TRUE(info.south_first);
  ASSERT_EQ(info.lon.size(), 360U);
  EXPECT_DOUBLE_EQ(info.lon.front(), -179.5);
  EXPECT_TRUE(info.periodic);
}

TEST(OceanGridFile, RegionalGridIsNotPeriodicAndClampsWindows) {
  const soma::Comm c = soma::Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("region.h5");
  // Latitudes 0 .. 40 and longitudes 100 .. 160 every 2 degrees, grid-registered: no seam.
  const ocean::testing::Extent region{0, 40, 100, 160};
  if (c.rank() == 0) {
    ocean::testing::write_grid_file(path, 21, 31, Registration::grid, true, Storage::float64, f, {}, region);
  }
  c.barrier();
  const ocean::GridSpec spec{path};
  const ocean::GridInfo info = ocean::open_grid(c, spec);
  EXPECT_FALSE(info.periodic);
  EXPECT_EQ(info.lon.size(), 31U);  // nothing dropped
  EXPECT_DOUBLE_EQ(info.dlon, 2.0);
  EXPECT_DOUBLE_EQ(info.dlat, 2.0);
  // Longitudes 90 .. 105 clamp to the west edge: columns 100 .. 106 bracket up to 105.
  const std::vector<ocean::Window> ws = ocean::windows_for(info, 5, 7, 90, 105);
  ASSERT_EQ(ws.size(), 1U);
  EXPECT_EQ(ws[0].lon0, 0U);
  EXPECT_EQ(ws[0].nlon, 4U);
  EXPECT_EQ(check_window(spec, info, ws[0], 2).size(), ws[0].nlat);
  // A reversed range has no seam to cross.
  EXPECT_THROW(ocean::windows_for(info, 5, 7, 150, 110), std::invalid_argument);
}

TEST(OceanGridFile, WindowReadsAnalyticValuesInStrips) {
  const soma::Comm c = soma::Comm::world();
  const Scratch dir(c);
  for (const bool south_first : {false, true}) {
    const fs::path path = dir.path(south_first ? "w_south.h5" : "w_north.h5");
    write_on_root(c, path, 181, 361, Registration::grid, south_first, Storage::int32);
    const ocean::GridSpec spec{path};
    const ocean::GridInfo info = ocean::open_grid(c, spec);
    // Rows 20 .. 56 of the file, columns 100 .. 129.
    const ocean::Window w{20, 37, 100, 30};
    const std::size_t before = ocean::values_read();
    const std::vector<std::size_t> rows = check_window(spec, info, w, 8);
    EXPECT_EQ(ocean::values_read() - before, w.nlat * w.nlon);
    ASSERT_EQ(rows.size(), w.nlat);
    // Rows come in the file's order, each with its normalised (ascending) latitude index.
    for (std::size_t r = 0; r < rows.size(); ++r) {
      EXPECT_EQ(rows[r], south_first ? w.lat0 + r : info.lat.size() - 1 - (w.lat0 + r));
    }
  }
}

TEST(OceanGridFile, WindowsForAWindowInsideTheGridAreExact) {
  const soma::Comm c = soma::Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("inside.h5");
  write_on_root(c, path, 181, 361, Registration::grid, false, Storage::int32);
  const ocean::GridSpec spec{path};
  const ocean::GridInfo info = ocean::open_grid(c, spec);
  // Latitudes 10.5 .. 20.5 are bracketed by rows of 10 .. 21, longitudes -30.2 .. -25.5 by -31 .. -25.
  const std::vector<ocean::Window> ws = ocean::windows_for(info, 10.5, 20.5, -30.2, -25.5);
  ASSERT_EQ(ws.size(), 1U);
  EXPECT_EQ(ws[0].nlat, 12U);
  EXPECT_EQ(ws[0].nlon, 7U);
  EXPECT_EQ(ws[0].lon0, 149U);
  // North-first file: file row of latitude 21 is 90 - 21 = 69.
  EXPECT_EQ(ws[0].lat0, 69U);
  const std::vector<std::size_t> rows = check_window(spec, info, ws[0], 5);
  EXPECT_EQ(rows.size(), 12U);
}

TEST(OceanGridFile, WindowAcrossTheAntimeridianGivesTwoPieces) {
  const soma::Comm c = soma::Comm::world();
  const Scratch dir(c);
  for (const bool south_first : {false, true}) {
    const fs::path path = dir.path(south_first ? "a_south.h5" : "a_north.h5");
    write_on_root(c, path, 181, 361, Registration::grid, south_first, Storage::int32);
    const ocean::GridSpec spec{path};
    const ocean::GridInfo info = ocean::open_grid(c, spec);
    const std::vector<ocean::Window> ws = ocean::windows_for(info, -10, 10, 170, -170);
    ASSERT_EQ(ws.size(), 2U);
    EXPECT_EQ(ws[0].lon0, 350U);  // 170 .. 179
    EXPECT_EQ(ws[0].nlon, 10U);
    EXPECT_EQ(ws[1].lon0, 0U);  // -180 .. -170
    EXPECT_EQ(ws[1].nlon, 11U);
    EXPECT_EQ(ws[0].nlat, 21U);
    const std::size_t before = ocean::values_read();
    for (const ocean::Window& w : ws) check_window(spec, info, w, 6);
    // Nothing is read twice: the two pieces hold 21 distinct columns of 21 rows.
    EXPECT_EQ(ocean::values_read() - before, 21U * 21U);
    // The same range written as 170 .. 190 is the same two windows.
    const std::vector<ocean::Window> same = ocean::windows_for(info, -10, 10, 170, 190);
    ASSERT_EQ(same.size(), 2U);
    EXPECT_EQ(same[0].lon0, ws[0].lon0);
    EXPECT_EQ(same[1].nlon, ws[1].nlon);
  }
}

TEST(OceanGridFile, FullCircleAndClampedLatitudes) {
  const soma::Comm c = soma::Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("full.h5");
  write_on_root(c, path, 180, 360, Registration::cell, true, Storage::float64);
  const ocean::GridInfo info = ocean::open_grid(c, ocean::GridSpec{path});
  const std::vector<ocean::Window> all = ocean::windows_for(info, 80, 100, -180, 180);
  ASSERT_EQ(all.size(), 1U);
  EXPECT_EQ(all[0].lon0, 0U);
  EXPECT_EQ(all[0].nlon, 360U);
  // Latitude clamped at the pole: centres 79.5 and 89.5 bracket 80 .. 100 (to the last row).
  EXPECT_EQ(all[0].lat0, 169U);
  EXPECT_EQ(all[0].nlat, 11U);
  // A range wider than 360 degrees is the full circle too, once.
  const std::vector<ocean::Window> wide = ocean::windows_for(info, 0, 1, 100, 500);
  ASSERT_EQ(wide.size(), 1U);
  EXPECT_EQ(wide[0].nlon, 360U);
  // A cell-registered window straddling the seam wraps through column 359 to column 0.
  const std::vector<ocean::Window> seam = ocean::windows_for(info, 0, 1, 179.2, -179.2);
  ASSERT_EQ(seam.size(), 2U);
  EXPECT_EQ(seam[0].lon0 + seam[0].nlon, 360U);
  EXPECT_EQ(seam[1].lon0, 0U);
}

TEST(OceanGridFile, StripSizeDoesNotChangeTheValues) {
  const soma::Comm c = soma::Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("strips.h5");
  write_on_root(c, path, 181, 361, Registration::grid, true, Storage::int32);
  const ocean::GridSpec spec{path};
  const ocean::GridInfo info = ocean::open_grid(c, spec);
  const ocean::Window w{5, 50, 340, 20};
  for (const std::size_t strip : {std::size_t{1}, std::size_t{7}, std::size_t{50}, std::size_t{1000}}) {
    EXPECT_EQ(check_window(spec, info, w, strip).size(), 50U);
  }
  EXPECT_THROW(ocean::read_window(spec, info, w, 0, [](std::size_t, std::span<const double>) {}),
               std::invalid_argument);
  EXPECT_THROW(ocean::read_window(spec, info, {5, 50, 350, 20}, 4, [](std::size_t, std::span<const double>) {}),
               std::invalid_argument);
  EXPECT_THROW(ocean::read_window(spec, info, {160, 50, 0, 20}, 4, [](std::size_t, std::span<const double>) {}),
               std::invalid_argument);
}

TEST(OceanGridFile, BadFilesThrowOnEveryRank) {
  const soma::Comm c = soma::Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("bad.h5");
  write_on_root(c, path, 19, 37, Registration::grid, true, Storage::float64);
  ocean::GridSpec spec{path};
  spec.variable = "nothing";
  try {
    ocean::open_grid(c, spec);
    ADD_FAILURE() << "a missing variable did not throw";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("nothing"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("bad.h5"), std::string::npos) << e.what();
  }
  spec.variable = "z";
  spec.lat = "latitude";
  EXPECT_THROW(ocean::open_grid(c, spec), std::runtime_error);
  // A variable that is not (lat, lon): the longitudes themselves, one-dimensional.
  spec.lat = "y";
  spec.variable = "x";
  EXPECT_THROW(ocean::open_grid(c, spec), std::runtime_error);
  EXPECT_THROW(ocean::open_grid(c, ocean::GridSpec{dir.path("absent.h5")}), std::runtime_error);
  // Reading a window of a variable that is not there throws, naming it.
  const ocean::GridInfo info = ocean::open_grid(c, ocean::GridSpec{path});
  spec.variable = "nothing";
  EXPECT_THROW(ocean::read_window(spec, info, {0, 2, 0, 2}, 1, [](std::size_t, std::span<const double>) {}),
               std::runtime_error);
}

#else

TEST(OceanGridFile, WithoutHdf5Throws) {
  const soma::Comm c = soma::Comm::world();
  EXPECT_THROW(ocean::open_grid(c, ocean::GridSpec{"etopo.nc"}), std::runtime_error);
  const ocean::GridInfo info;
  EXPECT_THROW(ocean::read_window(ocean::GridSpec{"etopo.nc"}, info, {0, 1, 0, 1}, 1,
                                  [](std::size_t, std::span<const double>) {}),
               std::runtime_error);
}

#endif

}  // namespace
