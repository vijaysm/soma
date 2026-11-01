// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "box_mesh_check.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_view.hpp"
#include "soma/testing/box_mesh.hpp"

using soma::GO;
using soma::LO;
using soma::Ordering;
using soma::Topology;
using soma::testing::BoxCounts;
using soma::testing::BoxSpec;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace soma_test {

BoxSpec make_spec(int nx, int ny, int nz, Topology element, double lx = 1, double ly = 1,
                  double lz = 1) {
  BoxSpec s;
  s.nx = nx;
  s.ny = ny;
  s.nz = nz;
  s.element = element;
  s.lx = lx;
  s.ly = ly;
  s.lz = lz;
  return s;
}

// Lengths such as 0.1 and 0.3 make lx*i/nx differ from i*(lx/nx) in the last
// bit, and lx*n/n differ from lx at the far face.
std::vector<BoxSpec> sample_specs() {
  return {make_spec(1, 1, 1, Topology::hex),
          make_spec(2, 3, 4, Topology::hex),
          make_spec(5, 6, 7, Topology::hex, 0.1, 0.3, 0.7),
          make_spec(3, 3, 9, Topology::hex, 0.1, 0.7, 0.9),
          make_spec(8, 16, 16, Topology::hex),
          make_spec(1, 1, 1, Topology::quad),
          make_spec(4, 3, 1, Topology::quad),
          make_spec(5, 6, 1, Topology::quad, 0.1, 0.3),
          make_spec(7, 1, 1, Topology::quad, 2.5, 0.7),
          make_spec(1, 1, 1, Topology::tri),
          make_spec(2, 2, 1, Topology::tri),
          make_spec(5, 6, 1, Topology::tri, 0.1, 0.3),
          make_spec(3, 9, 1, Topology::tri, 0.7, 0.9)};
}

std::string describe(const BoxSpec& s) {
  return std::string(soma::name(s.element)) + " " + std::to_string(s.nx) + "x" +
         std::to_string(s.ny) + "x" + std::to_string(s.nz);
}

// Host-side reference measures, independent of the kernel below: a hex is cut
// into five tetrahedra (the kernel uses six), a polygon is measured with the
// shoelace formula (the kernel uses cross products of diagonals or edges).
using Point = std::array<double, 3>;

Point corner(const soma::HostMesh& m, int dim, LO e, int k) {
  const LO v = m.conn(dim)(m.conn_offsets(dim)(e) + k);
  return {m.coords()(v, 0), m.coords()(v, 1), m.coords()(v, 2)};
}

double tet_volume(const Point& a, const Point& b, const Point& c, const Point& d) {
  const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double w[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
  return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
          u[2] * (v[0] * w[1] - v[1] * w[0])) /
         6.0;
}

double host_measure(const soma::HostMesh& m, int dim, LO e) {
  const int n = static_cast<int>(m.conn_offsets(dim)(e + 1) - m.conn_offsets(dim)(e));
  std::array<Point, 8> p{};
  for (int k = 0; k < n; ++k) p[static_cast<std::size_t>(k)] = corner(m, dim, e, k);
  if (n == 8) {
    return tet_volume(p[0], p[1], p[3], p[4]) + tet_volume(p[1], p[2], p[3], p[6]) +
           tet_volume(p[1], p[4], p[5], p[6]) + tet_volume(p[3], p[4], p[6], p[7]) +
           tet_volume(p[1], p[3], p[4], p[6]);
  }
  double twice = 0;
  for (int k = 0; k < n; ++k) {
    const Point& a = p[static_cast<std::size_t>(k)];
    const Point& b = p[static_cast<std::size_t>((k + 1) % n)];
    twice += a[0] * b[1] - b[0] * a[1];
  }
  return 0.5 * twice;
}

double expected_measure(const BoxSpec& s) {
  const double cell = s.lx * s.ly * (s.element == Topology::hex ? s.lz : 1.0) /
                      (static_cast<double>(s.nx) * s.ny * (s.element == Topology::hex ? s.nz : 1));
  return s.element == Topology::tri ? 0.5 * cell : cell;
}

double box_measure(const BoxSpec& s) {
  return s.lx * s.ly * (s.element == Topology::hex ? s.lz : 1.0);
}

// Device-callable kernels, dispatching on the vertex count (8 hex, 4 quad, 3
// triangle). The measure of a hex is the sum of six tetrahedra around the v0-v6
// diagonal, of a quad half the cross product of its diagonals, of a triangle
// half the cross product of two edges.
KOKKOS_INLINE_FUNCTION double tet_det(const double* a, const double* b, const double* c,
                                      const double* d) {
  double u[3];
  double v[3];
  double w[3];
  for (int i = 0; i < 3; ++i) {
    u[i] = b[i] - a[i];
    v[i] = c[i] - a[i];
    w[i] = d[i] - a[i];
  }
  return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
          u[2] * (v[0] * w[1] - v[1] * w[0])) /
         6.0;
}

template <class M>
KOKKOS_INLINE_FUNCTION double signed_measure(const soma::MeshView<M>& v, int dim, LO e) {
  double p[8][3] = {};
  const LO n = v.num_vertices(dim, e);
  for (LO k = 0; k < n && k < 8; ++k) {
    for (int c = 0; c < 3; ++c) p[k][c] = v.x(v.vertex(dim, e, k), c);
  }
  if (n == 8) {
    return tet_det(p[0], p[1], p[2], p[6]) + tet_det(p[0], p[2], p[3], p[6]) +
           tet_det(p[0], p[3], p[7], p[6]) + tet_det(p[0], p[7], p[4], p[6]) +
           tet_det(p[0], p[4], p[5], p[6]) + tet_det(p[0], p[5], p[1], p[6]);
  }
  if (n == 4) {
    const double ax = p[2][0] - p[0][0];
    const double ay = p[2][1] - p[0][1];
    const double bx = p[3][0] - p[1][0];
    const double by = p[3][1] - p[1][1];
    return 0.5 * (ax * by - ay * bx);
  }
  const double ax = p[1][0] - p[0][0];
  const double ay = p[1][1] - p[0][1];
  const double bx = p[2][0] - p[0][0];
  const double by = p[2][1] - p[0][1];
  return 0.5 * (ax * by - ay * bx);
}

template <class M>
double total_measure(soma::MeshView<M> v) {
  double total = 0;
  const int dim = v.dim;
  Kokkos::parallel_reduce(
      "soma::test::box_measure_sum",
      Kokkos::RangePolicy<typename M::execution_space>(0, v.count(dim)),
      KOKKOS_LAMBDA(const LO e, double& acc) { acc += signed_measure<M>(v, dim, e); }, total);
  return total;
}

template <class M>
double least_measure(soma::MeshView<M> v) {
  double lowest = std::numeric_limits<double>::max();
  const int dim = v.dim;
  Kokkos::parallel_reduce(
      "soma::test::box_measure_min",
      Kokkos::RangePolicy<typename M::execution_space>(0, v.count(dim)),
      KOKKOS_LAMBDA(const LO e, double& acc) { acc = Kokkos::min(acc, signed_measure<M>(v, dim, e)); },
      Kokkos::Min<double>(lowest));
  return lowest;
}

template <class M>
std::int64_t gid_total(soma::MeshView<M> v, int dim) {
  std::int64_t total = 0;
  Kokkos::parallel_reduce(
      "soma::test::box_gid_sum", Kokkos::RangePolicy<typename M::execution_space>(0, v.count(dim)),
      KOKKOS_LAMBDA(const LO e, std::int64_t& acc) { acc += v.gid(dim, e); }, total);
  return total;
}

void expect_overflow_naming(const BoxSpec& s, const std::string& what, const std::string& type) {
  try {
    (void)soma::testing::box_counts(s);
    ADD_FAILURE() << "expected an overflow of " << type;
  } catch (const std::overflow_error& e) {
    EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find(type + " limit"), std::string::npos) << e.what();
  }
}

template <class E>
void expect_throws_naming(const BoxSpec& s, const std::string& what) {
  try {
    (void)soma::testing::box_mesh(s);
    ADD_FAILURE() << "expected an exception naming '" << what << "'";
  } catch (const E& e) {
    EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
  }
}

}  // namespace soma_test

TEST(BoxMesh, TheDefaultSpecIsOneUnitHexInFileOrder) {
  const BoxSpec spec;
  EXPECT_EQ(spec.nx, 1);
  EXPECT_EQ(spec.ny, 1);
  EXPECT_EQ(spec.nz, 1);
  EXPECT_EQ(spec.element, Topology::hex);
  EXPECT_EQ(spec.lx, 1.0);
  EXPECT_EQ(spec.ly, 1.0);
  EXPECT_EQ(spec.lz, 1.0);
  EXPECT_EQ(spec.ordering, Ordering::file);
  const auto mesh = soma::testing::box_mesh(spec);
  ASSERT_EQ(mesh.count(0), 8);
  ASSERT_EQ(mesh.count(3), 1);
  // Vertex ids run i fastest, so the canonical corner order is 0 1 3 2 4 5 7 6.
  const double by_gid[8][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                               {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
  const LO canonical[8] = {0, 1, 3, 2, 4, 5, 7, 6};
  for (LO v = 0; v < 8; ++v) {
    EXPECT_EQ(mesh.gid(0)(v), v + 1);
    for (int c = 0; c < 3; ++c) EXPECT_EQ(mesh.coords()(v, c), by_gid[v][c]) << "vertex " << v;
    EXPECT_EQ(mesh.conn(3)(v), canonical[v]) << "corner " << v;
  }
  EXPECT_EQ(mesh.gid(3)(0), 1);
}

TEST(BoxMesh, HexCountsMatchTheClosedForms) {
  struct Shape {
    int nx, ny, nz;
  };
  for (const Shape s : {Shape{1, 1, 1}, Shape{2, 3, 4}, Shape{5, 1, 7}, Shape{1, 9, 1},
                        Shape{8, 16, 16}}) {
    const auto mesh = soma::testing::box_mesh(soma_test::make_spec(s.nx, s.ny, s.nz, Topology::hex));
    EXPECT_EQ(mesh.dim(), 3);
    EXPECT_EQ(mesh.count(0), (s.nx + 1) * (s.ny + 1) * (s.nz + 1));
    EXPECT_EQ(mesh.count(3), s.nx * s.ny * s.nz);
    EXPECT_EQ(mesh.count(1), 0);
    EXPECT_EQ(mesh.count(2), 0);
  }
}

// data/hex_2048.h5m holds 16 x 16 x 8 hexes on 2 x 2 x 1; the counts alone
// (2601 vertices, 2048 hexes) do not tell the axes apart.
TEST(BoxMesh, The2048HexBoxesHaveTheCountsOfTheDataFile) {
  for (const BoxSpec& spec : {BoxSpec{8, 16, 16, Topology::hex}, BoxSpec{16, 16, 8, Topology::hex, 2, 2, 1}}) {
    const auto mesh = soma::testing::box_mesh(spec);
    EXPECT_EQ(mesh.count(0), 2601);
    EXPECT_EQ(mesh.count(3), 2048);
  }
}

TEST(BoxMesh, QuadBoxHasTwentyVerticesAndTheLastIsTheFarCorner) {
  const auto mesh = soma::testing::box_mesh(soma_test::make_spec(4, 3, 1, Topology::quad));
  EXPECT_EQ(mesh.dim(), 2);
  ASSERT_EQ(mesh.count(0), 20);
  ASSERT_EQ(mesh.count(2), 12);
  EXPECT_EQ(mesh.count(3), 0);
  EXPECT_EQ(mesh.gid(0)(19), 20);
  EXPECT_EQ(mesh.coords()(19, 0), 1.0);
  EXPECT_EQ(mesh.coords()(19, 1), 1.0);
  EXPECT_EQ(mesh.coords()(19, 2), 0.0);
  EXPECT_EQ(mesh.coords()(0, 0), 0.0);
  EXPECT_EQ(mesh.coords()(0, 1), 0.0);
}

// Quad q = i + nx*j splits into (v0,v1,v2) and (v0,v2,v3).
TEST(BoxMesh, TriangleBoxSplitsEveryQuadAlongItsFirstDiagonal) {
  const auto mesh = soma::testing::box_mesh(soma_test::make_spec(2, 2, 1, Topology::tri));
  ASSERT_EQ(mesh.count(2), 8);
  ASSERT_EQ(mesh.count(0), 9);
  const std::vector<std::vector<GO>> table{{1, 2, 5}, {1, 5, 4}, {2, 3, 6}, {2, 6, 5},
                                           {4, 5, 8}, {4, 8, 7}, {5, 6, 9}, {5, 9, 8}};
  for (LO e = 0; e < 8; ++e) {
    EXPECT_EQ(mesh.gid(2)(e), e + 1);
    for (LO k = 0; k < 3; ++k) {
      const LO lid = mesh.conn(2)(mesh.conn_offsets(2)(e) + k);
      EXPECT_EQ(mesh.gid(0)(lid), table[static_cast<std::size_t>(e)][static_cast<std::size_t>(k)])
          << "triangle " << e + 1 << " corner " << k;
    }
  }
}

TEST(BoxMesh, EverySampleBoxSatisfiesTheInvariantsInFileOrder) {
  for (BoxSpec spec : soma_test::sample_specs()) {
    SCOPED_TRACE(soma_test::describe(spec));
    spec.ordering = Ordering::file;
    const auto mesh = soma::testing::box_mesh(spec);
    ASSERT_NO_FATAL_FAILURE(soma_test::expect_box_invariants(mesh, spec));
  }
}

TEST(BoxMesh, EverySampleBoxSatisfiesTheInvariantsInTheDefaultOrder) {
  for (BoxSpec spec : soma_test::sample_specs()) {
    SCOPED_TRACE(soma_test::describe(spec));
    spec.ordering = soma::default_ordering;
    const auto mesh = soma::testing::box_mesh(spec);
    ASSERT_NO_FATAL_FAILURE(soma_test::expect_box_invariants(mesh, spec));
  }
}

TEST(BoxMesh, FileOrderNumbersEntitiesByGid) {
  for (const BoxSpec& spec : soma_test::sample_specs()) {
    SCOPED_TRACE(soma_test::describe(spec));
    const auto mesh = soma::testing::box_mesh(spec);
    const int dim = mesh.dim();
    for (LO v = 0; v < mesh.count(0); ++v) ASSERT_EQ(mesh.gid(0)(v), v + 1);
    for (LO e = 0; e < mesh.count(dim); ++e) ASSERT_EQ(mesh.gid(dim)(e), e + 1);
  }
}

TEST(BoxMesh, ElementGidsRunIFastestThenJThenK) {
  const auto mesh = soma::testing::box_mesh(soma_test::make_spec(3, 2, 2, Topology::hex));
  for (LO e = 0; e < 12; ++e) {
    const LO i = e % 3;
    const LO j = (e / 3) % 2;
    const LO k = e / 6;
    const LO first = mesh.conn(3)(mesh.conn_offsets(3)(e));
    EXPECT_EQ(mesh.gid(0)(first), 1 + i + 4 * (j + 3 * k)) << "element " << e;
  }
  const auto quads = soma::testing::box_mesh(soma_test::make_spec(3, 2, 1, Topology::quad));
  for (LO e = 0; e < 6; ++e) {
    const LO first = quads.conn(2)(quads.conn_offsets(2)(e));
    EXPECT_EQ(quads.gid(0)(first), 1 + e % 3 + 4 * (e / 3)) << "quad " << e;
  }
}

TEST(BoxMesh, CoordinatesAreBitwiseLengthTimesIndexOverCount) {
  const auto mesh = soma::testing::box_mesh(soma_test::make_spec(5, 6, 7, Topology::hex, 0.1, 0.3, 0.7));
  const int px = 6;
  const int py = 7;
  for (LO v = 0; v < mesh.count(0); ++v) {
    const int i = static_cast<int>(v) % px;
    const int j = (static_cast<int>(v) / px) % py;
    const int k = static_cast<int>(v) / (px * py);
    if (i < 5) EXPECT_EQ(mesh.coords()(v, 0), 0.1 * i / 5) << "vertex " << v;
    if (j < 6) EXPECT_EQ(mesh.coords()(v, 1), 0.3 * j / 6) << "vertex " << v;
    if (k < 7) EXPECT_EQ(mesh.coords()(v, 2), 0.7 * k / 7) << "vertex " << v;
  }
  // Both rules are within rounding of each other; they differ in the last bit.
  EXPECT_NE(0.1 * 3 / 5, 3 * (0.1 / 5));
}

TEST(BoxMesh, TheFarFaceSitsExactlyAtTheLength) {
  const auto hex = soma::testing::box_mesh(soma_test::make_spec(3, 3, 9, Topology::hex, 0.1, 0.7, 0.9));
  EXPECT_NE(0.1 * 3 / 3, 0.1);
  double max_x = 0;
  double max_y = 0;
  double max_z = 0;
  for (LO v = 0; v < hex.count(0); ++v) {
    max_x = std::max(max_x, hex.coords()(v, 0));
    max_y = std::max(max_y, hex.coords()(v, 1));
    max_z = std::max(max_z, hex.coords()(v, 2));
  }
  EXPECT_EQ(max_x, 0.1);
  EXPECT_EQ(max_y, 0.7);
  EXPECT_EQ(max_z, 0.9);
  const auto quad = soma::testing::box_mesh(soma_test::make_spec(3, 9, 1, Topology::quad, 0.1, 0.9));
  EXPECT_EQ(quad.coords()(quad.count(0) - 1, 0), 0.1);
  EXPECT_EQ(quad.coords()(quad.count(0) - 1, 1), 0.9);
}

TEST(BoxMesh, PlanarBoxesLieInTheZeroPlane) {
  for (const Topology t : {Topology::quad, Topology::tri}) {
    const auto mesh = soma::testing::box_mesh(soma_test::make_spec(4, 3, 1, t, 2.0, 3.0, 99.0));
    EXPECT_EQ(mesh.dim(), 2);
    for (LO v = 0; v < mesh.count(0); ++v) EXPECT_EQ(mesh.coords()(v, 2), 0.0) << "vertex " << v;
  }
}

TEST(BoxMesh, HexesHavePositiveVolumeEqualToTheCell) {
  for (const BoxSpec& spec : soma_test::sample_specs()) {
    if (spec.element != Topology::hex) continue;
    SCOPED_TRACE(soma_test::describe(spec));
    const auto mesh = soma::testing::box_mesh(spec);
    const double cell = soma_test::expected_measure(spec);
    for (LO e = 0; e < mesh.count(3); ++e) {
      const double volume = soma_test::host_measure(mesh, 3, e);
      ASSERT_GT(volume, 0.0) << "hex " << e;
      ASSERT_NEAR(volume, cell, 1e-12 * cell) << "hex " << e;
    }
  }
}

TEST(BoxMesh, QuadsHavePositiveAreaEqualToTheCell) {
  for (const BoxSpec& spec : soma_test::sample_specs()) {
    if (spec.element != Topology::quad) continue;
    SCOPED_TRACE(soma_test::describe(spec));
    const auto mesh = soma::testing::box_mesh(spec);
    const double cell = soma_test::expected_measure(spec);
    for (LO e = 0; e < mesh.count(2); ++e) {
      const double area = soma_test::host_measure(mesh, 2, e);
      ASSERT_GT(area, 0.0) << "quad " << e;
      ASSERT_NEAR(area, cell, 1e-12 * cell) << "quad " << e;
    }
  }
}

TEST(BoxMesh, TrianglesHavePositiveAreaEqualToHalfTheCell) {
  for (const BoxSpec& spec : soma_test::sample_specs()) {
    if (spec.element != Topology::tri) continue;
    SCOPED_TRACE(soma_test::describe(spec));
    const auto mesh = soma::testing::box_mesh(spec);
    const double half = soma_test::expected_measure(spec);
    for (LO e = 0; e < mesh.count(2); ++e) {
      const double area = soma_test::host_measure(mesh, 2, e);
      ASSERT_GT(area, 0.0) << "triangle " << e + 1;
      ASSERT_NEAR(area, half, 1e-12 * half) << "triangle " << e + 1;
    }
  }
}

TEST(BoxMesh, BoundaryVerticesAreTheTotalMinusTheInteriorClosedForm) {
  for (const BoxSpec& spec : soma_test::sample_specs()) {
    SCOPED_TRACE(soma_test::describe(spec));
    const auto mesh = soma::testing::box_mesh(spec);
    const bool solid = spec.element == Topology::hex;
    LO boundary = 0;
    for (LO v = 0; v < mesh.count(0); ++v) {
      const bool on_x = mesh.coords()(v, 0) == 0.0 || mesh.coords()(v, 0) == spec.lx;
      const bool on_y = mesh.coords()(v, 1) == 0.0 || mesh.coords()(v, 1) == spec.ly;
      const bool on_z = solid && (mesh.coords()(v, 2) == 0.0 || mesh.coords()(v, 2) == spec.lz);
      if (on_x || on_y || on_z) ++boundary;
    }
    const LO interior = (spec.nx - 1) * (spec.ny - 1) * (solid ? spec.nz - 1 : 1);
    EXPECT_EQ(boundary, mesh.count(0) - interior);
  }
}

TEST(BoxMesh, EveryTopologyOtherThanHexQuadTriIsRejectedByName) {
  for (int t = 0; t < soma::topology_count; ++t) {
    const auto topology = static_cast<Topology>(t);
    if (topology == Topology::hex || topology == Topology::quad || topology == Topology::tri) continue;
    SCOPED_TRACE(soma::name(topology));
    soma_test::expect_throws_naming<std::invalid_argument>(
        soma_test::make_spec(2, 2, 2, topology), soma::name(topology));
  }
  soma_test::expect_throws_naming<std::invalid_argument>(soma_test::make_spec(2, 2, 2, Topology::tet),
                                                         "supported: hex, quad, tri");
}

TEST(BoxMesh, CellCountsBelowOneAreRejectedByAxis) {
  for (const int bad : {0, -1, -100}) {
    soma_test::expect_throws_naming<std::invalid_argument>(soma_test::make_spec(bad, 2, 2, Topology::hex), "nx");
    soma_test::expect_throws_naming<std::invalid_argument>(soma_test::make_spec(2, bad, 2, Topology::hex), "ny");
    soma_test::expect_throws_naming<std::invalid_argument>(soma_test::make_spec(2, 2, bad, Topology::hex), "nz");
    soma_test::expect_throws_naming<std::invalid_argument>(soma_test::make_spec(bad, 2, 1, Topology::quad), "nx");
    soma_test::expect_throws_naming<std::invalid_argument>(soma_test::make_spec(2, bad, 1, Topology::tri), "ny");
  }
  soma_test::expect_throws_naming<std::invalid_argument>(soma_test::make_spec(0, 2, 2, Topology::hex), "at least 1");
}

TEST(BoxMesh, LengthsMustBePositiveAndFinite) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {0.0, -1.0, nan, inf, -inf}) {
    soma_test::expect_throws_naming<std::invalid_argument>(
        soma_test::make_spec(2, 2, 2, Topology::hex, bad, 1, 1), "lx");
    soma_test::expect_throws_naming<std::invalid_argument>(
        soma_test::make_spec(2, 2, 2, Topology::hex, 1, bad, 1), "ly");
    soma_test::expect_throws_naming<std::invalid_argument>(
        soma_test::make_spec(2, 2, 2, Topology::hex, 1, 1, bad), "lz");
    soma_test::expect_throws_naming<std::invalid_argument>(
        soma_test::make_spec(2, 2, 1, Topology::quad, bad, 1), "lx");
    soma_test::expect_throws_naming<std::invalid_argument>(
        soma_test::make_spec(2, 2, 1, Topology::tri, 1, bad), "ly");
  }
}

TEST(BoxMesh, PlanarElementsRequireOneCellInZAndIgnoreLz) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const Topology t : {Topology::quad, Topology::tri}) {
    soma_test::expect_throws_naming<std::invalid_argument>(soma_test::make_spec(2, 2, 3, t), "nz");
    for (const double lz : {-5.0, 0.0, nan}) {
      SCOPED_TRACE(std::string(soma::name(t)) + " lz " + std::to_string(lz));
      soma::HostMesh mesh;
      ASSERT_NO_THROW(mesh = soma::testing::box_mesh(soma_test::make_spec(2, 2, 1, t, 1, 1, lz)));
      ASSERT_EQ(mesh.count(0), 9);
      for (LO v = 0; v < mesh.count(0); ++v) EXPECT_EQ(mesh.coords()(v, 2), 0.0) << "vertex " << v;
    }
  }
}

TEST(BoxMesh, AnOrderingThatTheBuilderRejectsIsReported) {
  BoxSpec spec = soma_test::make_spec(2, 2, 2, Topology::hex);
  spec.ordering = Ordering::tiled;
  EXPECT_THROW((void)soma::testing::box_mesh(spec), std::invalid_argument);
}

TEST(BoxCounts, MatchTheClosedForms) {
  const BoxCounts hex = soma::testing::box_counts(soma_test::make_spec(8, 16, 16, Topology::hex));
  EXPECT_EQ(hex.vertices, 2601);
  EXPECT_EQ(hex.elements, 2048);
  EXPECT_EQ(hex.connectivity, 16384);
  const BoxCounts quad = soma::testing::box_counts(soma_test::make_spec(4, 3, 1, Topology::quad));
  EXPECT_EQ(quad.vertices, 20);
  EXPECT_EQ(quad.elements, 12);
  EXPECT_EQ(quad.connectivity, 48);
  const BoxCounts tri = soma::testing::box_counts(soma_test::make_spec(2, 2, 1, Topology::tri));
  EXPECT_EQ(tri.vertices, 9);
  EXPECT_EQ(tri.elements, 8);
  EXPECT_EQ(tri.connectivity, 24);
  const BoxCounts one = soma::testing::box_counts(soma_test::make_spec(1, 1, 1, Topology::hex));
  EXPECT_EQ(one.vertices, 8);
  EXPECT_EQ(one.elements, 1);
  EXPECT_EQ(one.connectivity, 8);
}

TEST(BoxCounts, AgreeWithTheMeshesBuilt) {
  for (const BoxSpec& spec : soma_test::sample_specs()) {
    SCOPED_TRACE(soma_test::describe(spec));
    const BoxCounts c = soma::testing::box_counts(spec);
    const auto mesh = soma::testing::box_mesh(spec);
    EXPECT_EQ(c.vertices, mesh.count(0));
    EXPECT_EQ(c.elements, mesh.count(mesh.dim()));
    EXPECT_EQ(c.connectivity, static_cast<std::int64_t>(mesh.conn(mesh.dim()).extent(0)));
  }
}

// The first count that does not fit its type is reported: vertices and
// elements must fit LO, connectivity entries Offset. The shapes sit on both
// sides of 2^31 (65535 x 65536 vertices is 2^32 + 65536, which 32-bit
// arithmetic wraps to a small positive number) and of 2^63 (2^21 hexes per
// axis). The expectation is decided by division, so it cannot overflow itself.
bool fits(std::int64_t a, std::int64_t b, std::int64_t limit) { return a <= limit / b; }

TEST(BoxCounts, EachCountIsCheckedAgainstTheWidthOfItsType) {
  struct Case {
    int nx, ny, nz;
    Topology element;
  };
  const std::vector<Case> cases{
      {46339, 46339, 1, Topology::quad},       {46340, 46340, 1, Topology::quad},
      {65535, 65535, 1, Topology::quad},       {65535, 65536, 1, Topology::quad},
      {30000, 30000, 1, Topology::tri},        {32768, 32768, 1, Topology::tri},
      {40000, 40000, 1, Topology::tri},
      {1000, 1000, 1000, Topology::hex},       {1290, 1290, 1290, Topology::hex},
      {1, 1, 1, Topology::hex},                {2147483647, 1, 1, Topology::quad},
      {1, 1, 2147483647, Topology::hex},       {2097150, 2097151, 2097151, Topology::hex},
      {2097151, 2097151, 2097151, Topology::hex}, {2147483647, 2147483647, 1, Topology::tri}};
  const auto lo_max = static_cast<std::int64_t>(std::numeric_limits<LO>::max());
  const auto offset_max = static_cast<std::int64_t>(std::numeric_limits<soma::Offset>::max());
  for (const Case& c : cases) {
    const BoxSpec spec = soma_test::make_spec(c.nx, c.ny, c.nz, c.element);
    SCOPED_TRACE(soma_test::describe(spec));
    const bool solid = c.element == Topology::hex;
    const std::int64_t px = c.nx + std::int64_t{1};
    const std::int64_t py = c.ny + std::int64_t{1};
    const std::int64_t pz = solid ? c.nz + std::int64_t{1} : 1;
    const std::int64_t per_cell = c.element == Topology::tri ? 2 : 1;
    const int arity = soma::num_vertices(c.element);
    const bool vertices_fit = fits(px, py, lo_max) && fits(px * py, pz, lo_max);
    const bool elements_fit = fits(c.nx, c.ny, lo_max) && fits(std::int64_t{c.nx} * c.ny, c.nz, lo_max) &&
                              fits(std::int64_t{c.nx} * c.ny * c.nz, per_cell, lo_max);
    const bool connectivity_fit = elements_fit && fits(std::int64_t{c.nx} * c.ny * c.nz * per_cell, arity, offset_max);
    if (!vertices_fit) {
      soma_test::expect_overflow_naming(spec, "vertices", "LO");
    } else if (!elements_fit) {
      soma_test::expect_overflow_naming(spec, "elements", "LO");
    } else if (!connectivity_fit) {
      soma_test::expect_overflow_naming(spec, "connectivity entries", "Offset");
    } else {
      const std::int64_t cells = std::int64_t{c.nx} * c.ny * c.nz;
      const BoxCounts got = soma::testing::box_counts(spec);
      EXPECT_EQ(got.vertices, px * py * pz);
      EXPECT_EQ(got.elements, cells * per_cell);
      EXPECT_EQ(got.connectivity, cells * per_cell * arity);
    }
  }
}

// The ASSERT on box_counts comes first: a missing check then stops the test
// before box_mesh is asked to allocate an impossible amount of memory.
TEST(BoxCounts, SizesFarBeyondAnyTypeOverflowBeforeAnythingIsAllocated) {
  constexpr int big = std::numeric_limits<int>::max();
  for (const Topology t : {Topology::hex, Topology::tri, Topology::quad}) {
    const BoxSpec spec = soma_test::make_spec(big, big, t == Topology::hex ? big : 1, t);
    ASSERT_THROW((void)soma::testing::box_counts(spec), std::overflow_error) << soma::name(t);
    EXPECT_THROW((void)soma::testing::box_mesh(spec), std::overflow_error) << soma::name(t);
  }
}

template <class M>
class BoxMeshKernels : public ::testing::Test {};
using KernelSpaces = ::testing::Types<HostMem, DevMem>;
TYPED_TEST_SUITE(BoxMeshKernels, KernelSpaces);

TYPED_TEST(BoxMeshKernels, ElementMeasuresAreAllPositiveAndSumToTheBoxMeasure) {
  for (const BoxSpec& spec : soma_test::sample_specs()) {
    SCOPED_TRACE(soma_test::describe(spec));
    const auto host = soma::testing::box_mesh(spec);
    const auto mesh = host.template mirror<TypeParam>();
    const double total = soma_test::total_measure<TypeParam>(mesh.view());
    const double box = soma_test::box_measure(spec);
    EXPECT_NEAR(total, box, 1e-12 * box);
    const double least = soma_test::least_measure<TypeParam>(mesh.view());
    EXPECT_GT(least, 0.0);
    const double cell = soma_test::expected_measure(spec);
    EXPECT_NEAR(least, cell, 1e-12 * cell);
  }
}

TYPED_TEST(BoxMeshKernels, GidsSumToTheClosedForms) {
  for (const BoxSpec& spec : soma_test::sample_specs()) {
    SCOPED_TRACE(soma_test::describe(spec));
    const auto host = soma::testing::box_mesh(spec);
    const auto mesh = host.template mirror<TypeParam>();
    const std::int64_t nv = mesh.count(0);
    const std::int64_t ne = mesh.count(mesh.dim());
    EXPECT_EQ((soma_test::gid_total<TypeParam>(mesh.view(), 0)), nv * (nv + 1) / 2);
    EXPECT_EQ((soma_test::gid_total<TypeParam>(mesh.view(), mesh.dim())), ne * (ne + 1) / 2);
  }
}

TEST(BoxMeshDevice, HexVolumesOnTheDefaultSpaceSumToTheBoxVolume) {
  const BoxSpec spec = soma_test::make_spec(5, 4, 3, Topology::hex, 2.5, 0.7, 1.3);
  const auto host = soma::testing::box_mesh(spec);
  const auto device = host.mirror<DevMem>();
  const double expected = 2.5 * 0.7 * 1.3;
  EXPECT_NEAR(soma_test::total_measure<DevMem>(device.view()), expected, 1e-12 * expected);
  double host_total = 0;
  for (LO e = 0; e < host.count(3); ++e) host_total += soma_test::host_measure(host, 3, e);
  EXPECT_NEAR(host_total, expected, 1e-12 * expected);
}

TEST(BoxMeshDevice, TrianglesHavePositiveAreaOnTheDefaultSpace) {
  const auto host = soma::testing::box_mesh(soma_test::make_spec(2, 2, 1, Topology::tri));
  const auto device = host.mirror<DevMem>();
  ASSERT_EQ(device.count(2), 8);
  EXPECT_GT(soma_test::least_measure<DevMem>(device.view()), 0.0);
  EXPECT_NEAR(soma_test::total_measure<DevMem>(device.view()), 1.0, 1e-14);
}
