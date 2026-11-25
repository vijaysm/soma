// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The boundary of a spectral-element space: coast points and their outward
// normals, no-slip corners and pinch points, the normal flow removed, and the
// same bits on any number of ranks.

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "sem_meshes.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_edit.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/boundary.hpp"
#include "soma/sem/operators.hpp"
#include "soma/sem/sem_space.hpp"

using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Op;
using soma::Real;
using soma::sem::SemGeometry;
using Space = soma::sem::SemSpace<Kokkos::HostSpace>;
using Boundary = soma::sem::SemBoundary<Kokkos::HostSpace>;
using Values = soma::View2<Real, Kokkos::HostSpace>;
using Point = std::array<Real, 3>;

namespace {

LO rows(const Space& s) { return s.num_elements() * s.points_per_element(); }
LO owned_rows(const Space& s) { return s.owned_elements() * s.points_per_element(); }

Point centroid(const HostMesh& m, LO e) {
  const auto view = m.view();
  Point c{};
  for (LO q = 0; q < 4; ++q) {
    const LO v = view.vertex(2, e, q);
    for (int k = 0; k < 3; ++k) c[static_cast<std::size_t>(k)] += m.coords()(v, k) / 4;
  }
  return c;
}

Real distance(const Point& a, const Point& b) {
  return std::hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// `whole` (built on one rank) without the elements `drop` picks, removed with
// MeshEdit and compressed: on one rank that mesh itself, on more its live
// elements distributed again, with ghosts and distributed edges.
template <class Drop>
HostMesh without(const Comm& c, HostMesh whole, const Drop& drop) {
  std::vector<LO> gone;
  for (LO e = 0; e < whole.count(2); ++e) {
    if (drop(centroid(whole, e))) gone.push_back(e);
  }
  soma::MeshEdit edit(whole);
  edit.remove(2, gone);
  edit.apply();
  whole.compress();
  if (c.size() == 1) return whole;
  const auto view = whole.view();
  const auto egid = whole.gid(2);
  const auto vgid = whole.gid(0);
  std::vector<GO> elements, nodes;
  std::map<GO, std::array<Real, 3>> xyz;
  for (LO e = 0; e < whole.count(2); ++e) {
    elements.push_back(egid(e));
    for (LO q = 0; q < 4; ++q) nodes.push_back(vgid(view.vertex(2, e, q)));
  }
  for (LO v = 0; v < whole.count(0); ++v) {
    xyz[vgid(v)] = {whole.coords()(v, 0), whole.coords()(v, 1), whole.coords()(v, 2)};
  }
  return sem_test::distribute(c, elements, 4, nodes, xyz);
}

HostMesh whole_box(int nx, int ny, Real lx, Real ly) {
  soma::testing::BoxSpec s;
  s.nx = nx;
  s.ny = ny;
  s.element = soma::Topology::quad;
  s.lx = lx;
  s.ly = ly;
  return soma::testing::box_mesh(Comm::self(), s);
}

// The cubed sphere ne = 6 on the unit sphere without the 2x2 block of elements
// around the vertex at equiangular coordinates (15, 15) degrees on the face
// about +x: the hole spans [0, 30] x [0, 30] degrees there.
const Point hole_centre = [] {
  const Real t = std::tan(std::numbers::pi / 12);
  const Real n = std::sqrt(1 + 2 * t * t);
  return Point{1 / n, t / n, t / n};
}();

HostMesh holed_sphere(const Comm& c) {
  soma::sem::CubedSphereSpec spec;
  spec.ne = 6;
  return without(c, soma::sem::cubed_sphere_mesh(Comm::self(), spec),
                 [](const Point& x) { return distance(x, hole_centre) < 0.2; });
}

// A 4 x 4 unit box without the elements south-east and north-west of the
// vertex (0.5, 0.5): the two left there touch at that vertex only.
HostMesh pinched_box(const Comm& c) {
  return without(c, whole_box(4, 4, 1, 1), [](const Point& x) {
    return (std::abs(x[0] - 0.625) < 0.01 && std::abs(x[1] - 0.375) < 0.01) ||
           (std::abs(x[0] - 0.375) < 0.01 && std::abs(x[1] - 0.625) < 0.01);
  });
}

std::array<LO, 2> global_counts(const Comm& c, const Boundary& b) {
  return {c.allreduce(b.coast_points, Op::sum), c.allreduce(b.no_slip_points, Op::sum)};
}

Real norm(const Boundary& b, LO r) {
  return std::sqrt(b.normal(r, 0) * b.normal(r, 0) + b.normal(r, 1) * b.normal(r, 1) + b.normal(r, 2) * b.normal(r, 2));
}

TEST(Boundary, box_normals) {
  const Comm c = Comm::world();
  const Real lx = 2, ly = 1.5;
  HostMesh m = sem_test::box(c, 4, 3, lx, ly);
  const Space s(c, m, 3, SemGeometry::plane());
  const Boundary b = soma::sem::make_boundary(c, m, s);
  ASSERT_EQ(b.kind.extent(0), static_cast<std::size_t>(rows(s)));
  const auto x = s.x();
  const Real eps = 1e-12;
  for (LO r = 0; r < rows(s); ++r) {
    const bool left = x(r, 0) < eps, right = x(r, 0) > lx - eps, bottom = x(r, 1) < eps, top = x(r, 1) > ly - eps;
    const int walls = int(left) + int(right) + int(bottom) + int(top);
    if (walls == 0) {
      EXPECT_EQ(b.kind(r), 0) << "row " << r;
      EXPECT_EQ(norm(b, r), 0.0);
      continue;
    }
    if (walls == 2) {
      EXPECT_EQ(b.kind(r), 2) << "corner (" << x(r, 0) << ", " << x(r, 1) << ")";
      continue;
    }
    ASSERT_EQ(b.kind(r), 1) << "(" << x(r, 0) << ", " << x(r, 1) << ")";
    const Point want = left ? Point{-1, 0, 0} : right ? Point{1, 0, 0} : bottom ? Point{0, -1, 0} : Point{0, 1, 0};
    for (int k = 0; k < 3; ++k) {
      EXPECT_NEAR(b.normal(r, k), want[static_cast<std::size_t>(k)], 1e-14) << "(" << x(r, 0) << ", " << x(r, 1) << ")";
    }
  }
  // 13 x 10 points: 2 * 13 + 2 * 10 - 4 = 42 on the perimeter, 4 of them corners.
  const auto [coast, no_slip] = global_counts(c, b);
  EXPECT_EQ(coast, 38);
  EXPECT_EQ(no_slip, 4);
}

// The hole is a geodesic quadrilateral: its angles sum to more than 360
// degrees (Gauss-Bonnet), so not all four of its corners can turn the coast
// by 90 degrees or more. Its corners on the face's axes, (0, 0), (30, 0) and
// (0, 30), have right angles exactly (no-slip); at (30, 30) the hole's angle
// is acos(-1/4), about 104.5 degrees, so the coast turns by 75.5 there and the
// point keeps a slip normal, the two edges' normals' bisector.
TEST(Boundary, removed_patch_on_sphere) {
  const Comm c = Comm::world();
  HostMesh m = holed_sphere(c);
  EXPECT_EQ(c.allreduce(m.segment_table(2).owned_end, Op::sum), 6 * 36 - 4);
  const int p = 3;
  const Space s(c, m, p, SemGeometry::sphere(1));
  const Boundary b = soma::sem::make_boundary(c, m, s);
  const auto x = s.x();
  const Real t = std::tan(std::numbers::pi / 6);
  const auto on_face = [](Real a, Real b2) {
    const Real n = std::sqrt(1 + a * a + b2 * b2);
    return Point{1 / n, a / n, b2 / n};
  };
  const std::array<Point, 3> right_angles = {on_face(0, 0), on_face(t, 0), on_face(0, t)};
  int corners_seen = 0;
  for (LO r = 0; r < owned_rows(s); ++r) {
    const Point at{x(r, 0), x(r, 1), x(r, 2)};
    bool right = false;
    for (const Point& q : right_angles) right = right || distance(at, q) < 1e-12;
    if (right) {
      EXPECT_EQ(b.kind(r), 2) << "hole corner (" << at[0] << ", " << at[1] << ", " << at[2] << ")";
      ++corners_seen;
      continue;
    }
    if (b.kind(r) == 0) {
      EXPECT_EQ(norm(b, r), 0.0);
      continue;
    }
    ASSERT_EQ(b.kind(r), 1) << "(" << at[0] << ", " << at[1] << ", " << at[2] << ")";
    EXPECT_NEAR(norm(b, r), 1.0, 1e-14);
    const Real radial = b.normal(r, 0) * at[0] + b.normal(r, 1) * at[1] + b.normal(r, 2) * at[2];
    EXPECT_LT(std::abs(radial), 1e-14);
    // Out of the kept region: toward the hole.
    Real toward = 0;
    for (int k = 0; k < 3; ++k) {
      toward += b.normal(r, k) * (hole_centre[static_cast<std::size_t>(k)] - at[static_cast<std::size_t>(k)]);
    }
    EXPECT_GT(toward, 0.0) << "(" << at[0] << ", " << at[1] << ", " << at[2] << ")";
  }
  EXPECT_GT(c.allreduce(corners_seen, Op::sum), 0);
  // The hole's 8 edges carry 8 p distinct points.
  const auto [coast, no_slip] = global_counts(c, b);
  EXPECT_EQ(no_slip, 3);
  EXPECT_EQ(coast, 8 * p - 3);
}

TEST(Boundary, pinch_point_is_no_slip) {
  const Comm c = Comm::world();
  HostMesh m = pinched_box(c);
  const Space s(c, m, 4, SemGeometry::plane());
  const Boundary b = soma::sem::make_boundary(c, m, s);
  Values u("u", static_cast<std::size_t>(rows(s)), 3);
  std::mt19937_64 rng(17 + static_cast<unsigned>(c.rank()));
  std::uniform_real_distribution<Real> uniform(-1, 1);
  for (LO r = 0; r < rows(s); ++r) {
    for (int k = 0; k < 2; ++k) u(r, k) = uniform(rng);
  }
  s.dss(c, u);
  soma::sem::remove_normal(b, u);
  const auto x = s.x();
  int pinch = 0;
  for (LO r = 0; r < rows(s); ++r) {
    for (int k = 0; k < 3; ++k) EXPECT_TRUE(std::isfinite(u(r, k)));
    if (std::abs(x(r, 0) - 0.5) < 1e-12 && std::abs(x(r, 1) - 0.5) < 1e-12) {
      ++pinch;
      EXPECT_EQ(b.kind(r), 2);
      for (int k = 0; k < 3; ++k) {
        EXPECT_EQ(u(r, k), 0.0);
        EXPECT_EQ(b.normal(r, k), 0.0);
      }
    }
  }
  EXPECT_GT(c.allreduce(pinch, Op::sum), 0);
}

TEST(Boundary, remove_normal_kills_flux) {
  const Comm c = Comm::world();
  for (int shape = 0; shape < 2; ++shape) {
    SCOPED_TRACE(shape == 0 ? "pinched box" : "holed sphere");
    HostMesh m = shape == 0 ? pinched_box(c) : holed_sphere(c);
    const Space s(c, m, 4, shape == 0 ? SemGeometry::plane() : SemGeometry::sphere(1));
    const Boundary b = soma::sem::make_boundary(c, m, s);
    const auto x = s.x();
    Values u("u", static_cast<std::size_t>(rows(s)), 3);
    std::mt19937_64 rng(5 + static_cast<unsigned>(c.rank()));
    std::uniform_real_distribution<Real> uniform(-1, 1);
    for (LO r = 0; r < rows(s); ++r) {
      Real v[3] = {uniform(rng), uniform(rng), shape == 0 ? 0 : uniform(rng)};
      if (shape == 1) {  // tangent to the sphere
        const Real d = v[0] * x(r, 0) + v[1] * x(r, 1) + v[2] * x(r, 2);
        for (int k = 0; k < 3; ++k) v[k] -= d * x(r, k);
      }
      for (int k = 0; k < 3; ++k) u(r, k) = v[k];
    }
    s.dss(c, u);
    const auto weak_integral = [&](const Values& v) {
      Values div("div", static_cast<std::size_t>(rows(s)), 1);
      soma::sem::divergence_weak(c, s, v, div);
      return s.integral(c, div);
    };
    const Real before = weak_integral(u);
    soma::sem::remove_normal(b, u);
    const Real after = weak_integral(u);
    EXPECT_LE(std::abs(before), 1e-12);
    EXPECT_LE(std::abs(after), 1e-12);
    EXPECT_LE(std::abs(after - before), 1e-12);
    int coast = 0;
    for (LO r = 0; r < rows(s); ++r) {
      if (b.kind(r) == 1) {
        ++coast;
        EXPECT_LE(std::abs(u(r, 0) * b.normal(r, 0) + u(r, 1) * b.normal(r, 1) + u(r, 2) * b.normal(r, 2)), 1e-15);
      } else if (b.kind(r) == 2) {
        for (int k = 0; k < 3; ++k) EXPECT_EQ(u(r, k), 0.0);
      }
    }
    EXPECT_GT(c.allreduce(coast, Op::sum), 0);
    // Still continuous: dss leaves it as it is, to rounding.
    Values again("again", static_cast<std::size_t>(rows(s)), 3);
    Kokkos::deep_copy(again, u);
    s.dss(c, again);
    for (LO r = 0; r < rows(s); ++r) {
      for (int k = 0; k < 3; ++k) EXPECT_NEAR(again(r, k), u(r, k), 1e-15);
    }
  }
}

using ByPoint = std::map<std::tuple<GO, LO>, std::array<std::uint64_t, 4>>;

ByPoint by_point(const Space& s, const HostMesh& m, const Boundary& b) {
  ByPoint out;
  const auto gid = m.gid(2);
  const LO ppe = s.points_per_element();
  for (LO r = 0; r < rows(s); ++r) {
    out[{gid(r / ppe), r % ppe}] = {b.kind(r), std::bit_cast<std::uint64_t>(b.normal(r, 0)),
                                    std::bit_cast<std::uint64_t>(b.normal(r, 1)),
                                    std::bit_cast<std::uint64_t>(b.normal(r, 2))};
  }
  return out;
}

TEST(Boundary, rank_independent) {
  const Comm c = Comm::world();
  for (int shape = 0; shape < 3; ++shape) {
    SCOPED_TRACE(shape == 0 ? "box" : shape == 1 ? "holed sphere" : "pinched box");
    const auto build = [&](const Comm& k) {
      return shape == 0 ? sem_test::box(k, 4, 3, 2, 1.5) : shape == 1 ? holed_sphere(k) : pinched_box(k);
    };
    const SemGeometry g = shape == 1 ? SemGeometry::sphere(1) : SemGeometry::plane();
    HostMesh whole = build(Comm::self());
    const Space ws(Comm::self(), whole, 4, g);
    const Boundary wb = soma::sem::make_boundary(Comm::self(), whole, ws);
    const ByPoint expected = by_point(ws, whole, wb);
    HostMesh part = build(c);
    const Space s(c, part, 4, g);
    const Boundary b = soma::sem::make_boundary(c, part, s);
    for (const auto& [key, value] : by_point(s, part, b)) {
      EXPECT_EQ(value, expected.at(key)) << "element " << std::get<0>(key) << " point " << std::get<1>(key);
    }
    const auto [coast, no_slip] = global_counts(c, b);
    EXPECT_EQ(coast, wb.coast_points);
    EXPECT_EQ(no_slip, wb.no_slip_points);
  }
}

}  // namespace
