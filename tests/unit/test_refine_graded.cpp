// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Graded refinement toward a point on the ne4 cubed sphere: the cells near
// the point reach the spacing asked, every band is at least `transition`
// cells, the cells and their levels are the serial ones on any number of
// ranks, a hook before marking again sees the levels made, and max_cells
// stops the pass that would pass it.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "soma/algorithms/refine_graded.hpp"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/cubed_sphere.hpp"

using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Real;

namespace {

using P3 = std::array<Real, 3>;

constexpr Real kDegree = std::numbers::pi / 180;
constexpr Real kEarthKm = 6.37122e3;  // the Earth's radius (km)

Real dot(const P3& a, const P3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
P3 cross(const P3& a, const P3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
P3 unit(const P3& a) {
  const Real n = std::sqrt(dot(a, a));
  return {a[0] / n, a[1] / n, a[2] / n};
}
Real angle(const P3& a, const P3& b) { return std::atan2(std::sqrt(dot(cross(a, b), cross(a, b))), dot(a, b)); }

P3 lat_lon(Real lat, Real lon) {
  return {std::cos(lat * kDegree) * std::cos(lon * kDegree), std::cos(lat * kDegree) * std::sin(lon * kDegree),
          std::sin(lat * kDegree)};
}

const P3 kPoint = lat_lon(40, -100);

std::vector<P3> corners(const HostMesh& m, LO e) {
  const auto mv = m.view();
  const auto x = m.coords();
  std::vector<P3> p;
  for (LO i = 0; i < mv.num_vertices(2, e); ++i) {
    const LO v = mv.vertex(2, e, i);
    p.push_back(unit({x(v, 0), x(v, 1), x(v, 2)}));
  }
  return p;
}

// The great-circle distance (km on the Earth) from the point to the cell
// whose corners (counter-clockwise seen from outside) are `p`: 0 inside it,
// else the least distance to a side (a great-circle arc).
Real distance_km(const std::vector<P3>& p) {
  bool inside = true;
  Real least = 1e300;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const P3& a = p[i];
    const P3& b = p[(i + 1) % p.size()];
    const P3 n = unit(cross(a, b));
    const Real side = dot(kPoint, n);
    inside = inside && side >= 0;
    const P3 c = unit({kPoint[0] - side * n[0], kPoint[1] - side * n[1], kPoint[2] - side * n[2]});
    const bool between = dot(cross(a, c), n) >= 0 && dot(cross(c, b), n) >= 0;
    least = std::min(least, between ? std::asin(std::min(std::abs(side), Real{1}))
                                    : std::min(angle(kPoint, a), angle(kPoint, b)));
  }
  return inside ? 0 : least * kEarthKm;
}

// The target: the owned cells within grow (km) of the point.
std::vector<std::uint8_t> near_point(const HostMesh& m, const std::vector<Real>& grow) {
  std::vector<std::uint8_t> in(grow.size(), 0);
  for (std::size_t e = 0; e < grow.size(); ++e) {
    in[e] = grow[e] >= 0 && distance_km(corners(m, static_cast<LO>(e))) <= grow[e] ? 1 : 0;
  }
  return in;
}

soma::GradedOptions options() {
  soma::GradedOptions o;
  o.spacing = 50;
  o.length = [](Real s) { return s * kEarthKm; };
  return o;
}

HostMesh ne4(const Comm& comm) {
  soma::sem::CubedSphereSpec spec;
  spec.ne = 4;
  spec.radius = 1;
  return soma::sem::cubed_sphere_mesh(comm, spec);
}

// Every owned cell's gid and level, gathered and ordered by gid.
std::vector<std::pair<GO, std::int32_t>> cell_levels(const Comm& comm, const HostMesh& m) {
  const auto gid = m.gid(2);
  const auto level = soma::refine_levels(m);
  std::vector<GO> gids;
  for (std::size_t e = 0; e < level.size(); ++e) gids.push_back(gid(e));
  const auto all_gids = comm.allgatherv(std::span<const GO>(gids));
  const auto all_levels = comm.allgatherv(std::span<const std::int32_t>(level));
  std::vector<std::pair<GO, std::int32_t>> out;
  for (std::size_t i = 0; i < all_gids.size(); ++i) out.emplace_back(all_gids[i], all_levels[i]);
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

TEST(RefineGraded, point_target) {
  const Comm comm = Comm::world();
  const soma::GradedOptions o = options();
  soma::GradedResult r = soma::refine_graded(comm, ne4(comm), o, near_point);
  EXPECT_GT(r.passes.passes, 1);
  EXPECT_EQ(r.passes.marked, 0);

  // The cells within `spacing` of the point: at the spacing asked, no trisection left. The ne4 cells' spacing
  // (about 770 km) asks three trisections (770, 257, 86 > 1.5 * 50 >= 29), so the finest level is 3; a transition
  // template's cells near the point may carry a level under it at the finest size.
  const auto level = soma::refine_levels(r.mesh);
  std::int32_t finest = 0;
  for (const auto l : level) finest = std::max(finest, l);
  finest = static_cast<std::int32_t>(comm.allreduce(static_cast<Real>(finest), soma::Op::max));
  EXPECT_EQ(finest, 3);
  Real near = 0, wrong = 0;
  for (std::size_t e = 0; e < level.size(); ++e) {
    const auto p = corners(r.mesh, static_cast<LO>(e));
    if (distance_km(p) > o.spacing) continue;
    // The cell's size from its corners' polygon (as refine_graded measures it).
    P3 a{0, 0, 0};
    for (std::size_t i = 1; i + 1 < p.size(); ++i) {
      const P3 t = cross({p[i][0] - p[0][0], p[i][1] - p[0][1], p[i][2] - p[0][2]},
                         {p[i + 1][0] - p[0][0], p[i + 1][1] - p[0][1], p[i + 1][2] - p[0][2]});
      for (std::size_t q = 0; q < 3; ++q) a[q] += t[q];
    }
    const Real spacing = o.length(std::sqrt(std::sqrt(dot(a, a)) / 2)) / (o.np - 1);
    near += 1;
    wrong += spacing <= 1.5 * o.spacing ? 0 : 1;
  }
  EXPECT_GT(comm.allreduce(near, soma::Op::sum), 0);
  EXPECT_EQ(comm.allreduce(wrong, soma::Op::sum), 0);

  // Every band at least `transition` cells, or none to measure.
  EXPECT_FALSE(r.widths.empty());
  for (const Real w : r.widths) EXPECT_TRUE(w == -1 || w >= o.transition) << w;
  EXPECT_TRUE(r.settled);

  // The serial result, gid for gid and level for level.
  const Comm self = Comm::self();
  const soma::GradedResult serial = soma::refine_graded(self, ne4(self), o, near_point);
  EXPECT_EQ(cell_levels(comm, r.mesh), cell_levels(self, serial.mesh));
  EXPECT_EQ(r.passes.cells, serial.passes.cells);
  EXPECT_EQ(r.widths, serial.widths);
  EXPECT_EQ(r.reruns, serial.reruns);
}

// A hook that leaves the vertices alone changes nothing; it is called before
// every pass after the first, of every run, with the run and the passes made
// (the last run's last call before its last pass).
TEST(RefineGraded, hook_before_marking_again) {
  const Comm comm = Comm::world();
  const soma::GradedOptions o = options();
  const soma::GradedResult plain = soma::refine_graded(comm, ne4(comm), o, near_point);
  std::vector<std::pair<int, int>> calls;
  const soma::GradedResult hooked =
      soma::refine_graded(comm, ne4(comm), o, near_point, [&](HostMesh&, int run, int pass) {
        calls.emplace_back(run, pass);
      });
  EXPECT_EQ(cell_levels(comm, hooked.mesh), cell_levels(comm, plain.mesh));
  ASSERT_FALSE(calls.empty());
  EXPECT_EQ(calls.front(), std::make_pair(0, 1));
  EXPECT_EQ(calls.back().first, plain.reruns);
  EXPECT_EQ(calls.back().second, plain.passes.passes - 1);
}

TEST(RefineGraded, max_cells_names_the_pass) {
  const Comm comm = Comm::world();
  soma::GradedOptions o = options();
  o.max_cells = 100;
  try {
    (void)soma::refine_graded(comm, ne4(comm), o, near_point);
    FAIL() << "no throw";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("pass 1 "), std::string::npos) << e.what();
  }
}

// A spacing that is not positive and finite is refused on every rank (0 or less would never stop dividing the
// cells' size by 3 to reach it).
TEST(RefineGraded, spacing_must_be_positive) {
  const Comm comm = Comm::world();
  for (const Real spacing : {0.0, -1.0, std::numeric_limits<Real>::quiet_NaN(),
                             std::numeric_limits<Real>::infinity()}) {
    soma::GradedOptions o = options();
    o.spacing = spacing;
    EXPECT_THROW((void)soma::refine_graded(comm, ne4(comm), o, near_point), std::invalid_argument) << spacing;
  }
}

TEST(RefineGraded, a_target_must_answer_every_cell) {
  const Comm comm = Comm::world();
  EXPECT_THROW((void)soma::refine_graded(comm, ne4(comm), options(),
                                         [](const HostMesh&, const std::vector<Real>&) {
                                           return std::vector<std::uint8_t>{};
                                         }),
               std::invalid_argument);
}
