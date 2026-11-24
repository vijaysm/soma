// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The coastline of round islands on latitude-longitude grids: every point on the circle, one per bin, across
// the antimeridian, the same bits on any number of ranks; and the coast-distance target on a cubed sphere,
// against the distance from every cell to every point.

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "coastline.hpp"
#include "constants.hpp"
#include "grid_file.hpp"
#include "grid_files.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/cubed_sphere.hpp"

namespace {

namespace fs = std::filesystem;
using ocean::testing::Registration;
using ocean::testing::Storage;
using soma::Comm;
using soma::HostMesh;
using soma::LO;
using soma::Real;
using P3 = std::array<double, 3>;

constexpr double deg = std::numbers::pi / 180;
constexpr double earth_km = ocean::a / 1000;

double dot(const P3& a, const P3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
P3 cross(const P3& a, const P3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
P3 unit(const P3& a) {
  const double n = std::sqrt(dot(a, a));
  return {a[0] / n, a[1] / n, a[2] / n};
}
double angle(const P3& a, const P3& b) { return std::atan2(std::sqrt(dot(cross(a, b), cross(a, b))), dot(a, b)); }
P3 at(double lat, double lon) {
  return {std::cos(lat * deg) * std::cos(lon * deg), std::cos(lat * deg) * std::sin(lon * deg), std::sin(lat * deg)};
}

// A directory for this run's files, the same on every rank; rank 0 removes it.
class Scratch {
 public:
  explicit Scratch(const Comm& c) : c_(c) {
    std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
    c.bcast(std::span<std::int64_t>(&id, 1), 0);
    dir_ = fs::temp_directory_path() / ("ocean_coastline_" + std::to_string(c.size()) + "_" + std::to_string(id));
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

// A round island: heights radius - distance (km) from the centre, land inside the circle.
struct Island {
  double lat, lon, radius_km;
  [[nodiscard]] double height(double la, double lo) const {
    return radius_km - angle(at(lat, lon), at(la, lo)) * earth_km;
  }
  [[nodiscard]] double off_circle_km(const P3& p) const {
    return std::abs(angle(at(lat, lon), p) * earth_km - radius_km);
  }
  [[nodiscard]] double circumference_km() const {
    return 2 * std::numbers::pi * earth_km * std::sin(radius_km / earth_km);
  }
};

// An island of radius 500 km at 30N, 60W, on a regional grid of 0.02 degrees (2.2 km) from 20N to 40N,
// 75W to 45W, rows north first.
const Island island{30, -60, 500};
const ocean::testing::Extent island_extent{20, 40, -75, -45};
constexpr std::size_t island_nlat = 1001, island_nlon = 1501;
constexpr double island_step_km = 0.02 * deg * earth_km;
constexpr double island_spacing_km = 20;

// An island across the antimeridian on a global 0.25-degree grid: its east end reaches past 180E, so its
// coast crosses the edges from the last column (179.75E) to the first (180W).
const Island seam_island{-20, 174.5, 600};
constexpr std::size_t seam_nlat = 721, seam_nlon = 1441;
constexpr double seam_step_km = 0.25 * deg * earth_km;
constexpr double seam_spacing_km = 40;

#ifdef SWE_OCEAN_HAVE_HDF5

fs::path write_island(const Comm& c, const Scratch& dir) {
  const fs::path path = dir.path("island.h5");
  if (c.rank() == 0) {
    ocean::testing::write_grid_file(path, island_nlat, island_nlon, Registration::grid, false, Storage::float64,
                                    [](double la, double lo) { return island.height(la, lo); }, {}, island_extent);
  }
  c.barrier();
  return path;
}

fs::path write_seam(const Comm& c, const Scratch& dir) {
  const fs::path path = dir.path("seam.h5");
  if (c.rank() == 0) {
    ocean::testing::write_grid_file(path, seam_nlat, seam_nlon, Registration::grid, true, Storage::float64,
                                    [](double la, double lo) { return seam_island.height(la, lo); });
  }
  c.barrier();
  return path;
}

// Every point on the circle to within half a grid step: the crossing interpolated along a 2 km edge of the
// distance to the centre is off it by about (edge)^2 / (8 radius), metres, far inside half a step (1.1 km).
TEST(Coastline, island_points_lie_on_the_circle) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const ocean::GridSpec spec{write_island(c, dir)};
  const auto points = ocean::coastline(c, spec, island_spacing_km, 16);
  ASSERT_FALSE(points.empty());
  for (const P3& p : points) {
    EXPECT_NEAR(dot(p, p), 1, 1e-14);
    EXPECT_LE(island.off_circle_km(p), island_step_km / 2);
  }
  // One point per bin of h = spacing / 2 the circle passes through, to 10%. The circle enters a new bin at
  // every bin side it crosses: about 4 R / h latitude sides and as many longitude sides (R its radius), so it
  // passes through 8 R / h = (4 / pi) circumference / h bins, not circumference / h: a curve through square
  // bins of h crosses (|cos t| + |sin t|) / h sides per length at heading t, 4 / (pi h) on average over t.
  const double expected = 4 / std::numbers::pi * island.circumference_km() / (island_spacing_km / 2);
  EXPECT_NEAR(static_cast<double>(points.size()), expected, 0.1 * expected) << points.size();
}

TEST(Coastline, wraps_across_the_antimeridian) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const ocean::GridSpec spec{write_seam(c, dir)};
  const auto points = ocean::coastline(c, spec, seam_spacing_km, 9);
  ASSERT_FALSE(points.empty());
  std::size_t wrap = 0, west = 0;
  for (const P3& p : points) {
    EXPECT_LE(seam_island.off_circle_km(p), seam_step_km / 2);
    const double lon = std::atan2(p[1], p[0]) / deg;
    // Strictly between the last column and 180: only the wrap edges cross there.
    wrap += lon > 179.76 && lon < 179.99;
    west += lon < -179.9;
  }
  EXPECT_GT(wrap, 0U);
  EXPECT_GT(west, 0U);
}

// The grid's top row has no row above it, but its row edges are still scanned: a grid ending at 30N, through
// the island's centre, has the two crossings of that row (lat 30 exactly; h = 1 km keeps every crossing).
TEST(Coastline, top_row_edges) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const fs::path path = dir.path("half.h5");
  if (c.rank() == 0) {
    ocean::testing::write_grid_file(path, 101, 301, Registration::grid, true, Storage::float64,
                                    [](double la, double lo) { return island.height(la, lo); }, {},
                                    {20, 30, -75, -45});
  }
  c.barrier();
  const auto points = ocean::coastline(c, ocean::GridSpec{path}, 2, 8);
  std::size_t top = 0;
  for (const P3& p : points) top += std::abs(std::asin(p[2]) / deg - 30) < 1e-9;
  EXPECT_EQ(top, 2U);
}

// The list every rank holds is the one a single rank finds, bit for bit, in the same order.
TEST(Coastline, same_on_any_rank_count) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const Comm self = Comm::self();
  for (const auto& [path, spacing] : {std::pair{write_island(c, dir), island_spacing_km},
                                      std::pair{write_seam(c, dir), seam_spacing_km}}) {
    const ocean::GridSpec spec{path};
    const auto all = ocean::coastline(c, spec, spacing, 7);
    const auto one = ocean::coastline(self, spec, spacing, 540);
    ASSERT_EQ(all.size(), one.size()) << path;
    for (std::size_t k = 0; k < all.size(); ++k) {
      ASSERT_EQ(all[k], one[k]) << path << " point " << k;
    }
  }
}

TEST(Coastline, rejects_bad_spacing) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const ocean::GridSpec spec{write_island(c, dir)};
  EXPECT_THROW((void)ocean::coastline(c, spec, 0, 16), std::invalid_argument);
  EXPECT_THROW((void)ocean::coastline(c, spec, std::nan(""), 16), std::invalid_argument);
  EXPECT_THROW((void)ocean::coastline(c, spec, 10, 0), std::invalid_argument);
}

// The distance (km on the Earth) from p to the cell with corners q: 0 inside it (on the inner side of every
// side, the sides oriented by the cell's area vector), else the least great-circle distance to a side.
double distance_km(const P3& p, const std::vector<P3>& q) {
  P3 area{0, 0, 0}, mean{0, 0, 0};
  for (std::size_t i = 0; i < q.size(); ++i) {
    const P3 t = cross(q[i], q[(i + 1) % q.size()]);
    for (std::size_t k = 0; k < 3; ++k) {
      area[k] += t[k];
      mean[k] += q[i][k];
    }
  }
  const double sense = dot(area, mean) >= 0 ? 1 : -1;
  bool inside = true;
  double least = std::numbers::pi;
  for (std::size_t i = 0; i < q.size(); ++i) {
    const P3& a = q[i];
    const P3& b = q[(i + 1) % q.size()];
    const P3 n = unit(cross(a, b));
    const double side = dot(p, n);
    inside = inside && sense * side >= 0;
    const P3 foot = unit({p[0] - side * n[0], p[1] - side * n[1], p[2] - side * n[2]});
    const bool between = dot(cross(a, foot), n) >= 0 && dot(cross(foot, b), n) >= 0;
    least = std::min(least, between ? std::asin(std::min(std::abs(side), 1.0)) : std::min(angle(p, a), angle(p, b)));
  }
  return inside ? 0 : least * earth_km;
}

// Every owned cell of an ne = 8 unit cubed sphere, with grow (km) from -1 (no answer) to 375 by its gid, marked
// as the brute force over every point marks it: within grow of a point (inside counting as 0). counts: cells
// marked, marked with a point inside, unmarked yet within 1000 km of a point (over every rank).
std::array<std::int64_t, 3> expect_brute_force(const Comm& c, const std::vector<P3>& points) {
  soma::sem::CubedSphereSpec s;
  s.ne = 8;
  s.radius = 1;
  const HostMesh mesh = soma::sem::cubed_sphere_mesh(c, s);
  const LO owned = mesh.segment_table(2).owned_end;
  const auto gid = mesh.gid(2);
  std::vector<Real> grow(static_cast<std::size_t>(owned));
  for (LO e = 0; e < owned; ++e) grow[static_cast<std::size_t>(e)] = gid(e) % 5 == 0 ? -1 : 75.0 * (gid(e) % 6);
  const soma::GradedTarget target = ocean::coast_target(points);
  const auto marked = target(mesh, grow);
  EXPECT_EQ(marked.size(), grow.size());
  const auto mv = mesh.view();
  const auto x = mesh.coords();
  std::array<std::int64_t, 3> count{0, 0, 0};
  for (LO e = 0; e < owned && marked.size() == grow.size(); ++e) {
    std::vector<P3> q;
    for (LO i = 0; i < mv.num_vertices(2, e); ++i) {
      const LO v = mv.vertex(2, e, i);
      q.push_back(unit({x(v, 0), x(v, 1), x(v, 2)}));
    }
    double least = 1e300;
    for (const P3& p : points) least = std::min(least, distance_km(p, q));
    const double g = grow[static_cast<std::size_t>(e)];
    const bool want = g >= 0 && least <= g;
    EXPECT_EQ(marked[static_cast<std::size_t>(e)] != 0, want)
        << "gid " << gid(e) << " grow " << g << " least " << least;
    count[0] += want;
    count[1] += want && least == 0;
    count[2] += !want && least < 1000;
  }
  c.allreduce(std::span<std::int64_t>(count), soma::Op::sum);
  // A wrong number of grow values is refused.
  grow.push_back(0);
  EXPECT_THROW((void)target(mesh, grow), std::invalid_argument);
  return count;
}

TEST(Coastline, target_marks_cells_near_the_coast) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const ocean::GridSpec spec{write_island(c, dir)};
  const auto count = expect_brute_force(c, ocean::coastline(c, spec, island_spacing_km, 16));
  EXPECT_GT(count[0], 1);
  EXPECT_GT(count[1], 0);
  EXPECT_GT(count[2], 0);
}

// Enough points (24000, on eight circles of 300 km about the globe, 0.6 km apart as a coast's) that the
// index's bins (about 0.06, 380 km wide) are narrower than a query's reach, so a query box that left out grow,
// or the circumradius, would miss points the brute force finds.
TEST(Coastline, target_with_many_points) {
  const Comm c = Comm::world();
  std::vector<P3> points;
  const double r = 300 / earth_km;
  for (int k = 0; k < 8; ++k) {
    const P3 centre = at(-70 + 20 * k, -170 + 45 * k);
    const P3 east = unit(cross({0, 0, 1}, centre));
    const P3 north = cross(centre, east);
    for (int i = 0; i < 3000; ++i) {
      const double t = 2 * std::numbers::pi * i / 3000;
      P3 p;
      for (std::size_t d = 0; d < 3; ++d) {
        p[d] = std::cos(r) * centre[d] + std::sin(r) * (std::cos(t) * east[d] + std::sin(t) * north[d]);
      }
      points.push_back(p);
    }
  }
  const auto count = expect_brute_force(c, points);
  EXPECT_GT(count[0], 1);
  EXPECT_GT(count[1], 0);
  EXPECT_GT(count[2], 0);
}

#else

TEST(Coastline, WithoutHdf5Throws) {
  const Comm c = Comm::world();
  ocean::GridSpec spec;
  spec.file = "etopo.nc";
  EXPECT_THROW((void)ocean::coastline(c, spec, 10), std::runtime_error);
}

#endif

}  // namespace
