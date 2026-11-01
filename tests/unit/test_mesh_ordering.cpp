// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "hilbert_reference.hpp"
#include "mesh_fixture.hpp"
#include "soma/core/locality.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/sfc.hpp"

using soma::GO;
using soma::LO;
using soma::Ordering;
using soma::Real;
using soma::Topology;

namespace {

constexpr int n = 16;  // cells per side

// An n x n grid of quads whose vertex and element gids are a fixed scramble of
// the row-major numbering, so file order (gid order) is far from spatial order.
soma::HostMesh scrambled_grid(Ordering ordering) {
  const int nv = (n + 1) * (n + 1);
  const auto vgid = [&](int i, int j) {
    return static_cast<GO>(1 + (7919 * (i + (n + 1) * j)) % nv);
  };
  const auto egid = [&](int i, int j) { return static_cast<GO>(1 + (97 * (i + n * j)) % (n * n)); };
  soma::MeshBuilder b(2);
  std::vector<GO> vg;
  std::vector<Real> xyz;
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      vg.push_back(vgid(i, j));
      xyz.insert(xyz.end(), {static_cast<Real>(i), static_cast<Real>(j), 0.0});
    }
  }
  b.add_vertices(vg, xyz);
  std::vector<GO> eg;
  std::vector<GO> conn;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      eg.push_back(egid(i, j));
      conn.insert(conn.end(), {vgid(i, j), vgid(i + 1, j), vgid(i + 1, j + 1), vgid(i, j + 1)});
    }
  }
  b.add_elements(Topology::quad, eg, conn);
  return std::move(b).build(ordering);
}

// A 64 x 64 grid of quads in the plane z = 0.3, with a third of the vertices
// lifted to z = `lifted`.
soma::HostMesh plane_at_z(Real lifted) {
  constexpr int m = 64;
  const auto vgid = [](int i, int j) { return static_cast<GO>(1 + i + (m + 1) * j); };
  soma::MeshBuilder b(2);
  std::vector<GO> vg;
  std::vector<Real> xyz;
  for (int j = 0; j <= m; ++j) {
    for (int i = 0; i <= m; ++i) {
      vg.push_back(vgid(i, j));
      const Real z = (i * 7 + j * 13) % 3 == 0 ? lifted : 0.3;
      xyz.insert(xyz.end(), {static_cast<Real>(i), static_cast<Real>(j), z});
    }
  }
  b.add_vertices(vg, xyz);
  std::vector<GO> eg;
  std::vector<GO> conn;
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < m; ++i) {
      eg.push_back(1 + i + m * j);
      conn.insert(conn.end(), {vgid(i, j), vgid(i + 1, j), vgid(i + 1, j + 1), vgid(i, j + 1)});
    }
  }
  b.add_elements(Topology::quad, eg, conn);
  return std::move(b).build(Ordering::type_major_sfc);
}

// The same quantisation as the library (floor(t * (2^32 - 1)) on each axis of
// a vertex box that starts at the origin, t = v / extent with the extent of the
// widest axis), fed to the independent 2D reference.
std::uint64_t reference_key(double x, double y, double extent) {
  constexpr std::uint32_t last = 0xFFFFFFFFu;
  const auto cell = [](double v, double extent) {
    const double t = v / extent;
    if (!(t > 0)) return std::uint32_t{0};
    if (t >= 1) return last;
    return static_cast<std::uint32_t>(t * static_cast<double>(last));
  };
  return soma_test::hilbert_reference_2d(std::uint64_t{1} << 32, cell(x, extent),
                                         cell(y, extent));
}

// (reference key of the centroid, gid) of the elements of one segment. The
// centroid is the vertex sum times 1 / arity, summed in connectivity order, as
// the builder computes it.
std::vector<std::tuple<std::uint64_t, GO>> centroid_keys(const soma::HostMesh& mesh,
                                                         const soma::Segment& seg, double extent) {
  std::vector<std::tuple<std::uint64_t, GO>> keys;
  const auto off = mesh.conn_offsets(2);
  for (LO e = seg.begin; e < seg.end; ++e) {
    double c[2] = {0, 0};
    for (soma::Offset k = off(e); k < off(e + 1); ++k) {
      for (int q = 0; q < 2; ++q) c[q] += mesh.coords()(mesh.conn(2)(k), q);
    }
    const double inv = 1.0 / static_cast<double>(off(e + 1) - off(e));
    keys.emplace_back(reference_key(c[0] * inv, c[1] * inv, extent), mesh.gid(2)(e));
  }
  return keys;
}

}  // namespace

TEST(MeshOrdering, VerticesFollowTheHilbertKeyThenGid) {
  const auto mesh = scrambled_grid(Ordering::type_major_sfc);
  std::vector<std::tuple<std::uint64_t, GO>> expected;
  for (LO v = 0; v < mesh.count(0); ++v) {
    expected.emplace_back(reference_key(mesh.coords()(v, 0), mesh.coords()(v, 1), n),
                          mesh.gid(0)(v));
  }
  EXPECT_TRUE(std::is_sorted(expected.begin(), expected.end()));
  EXPECT_FALSE(std::is_sorted(mesh.gid(0).data(), mesh.gid(0).data() + mesh.count(0)));
}

TEST(MeshOrdering, ElementsFollowTheCentroidKeyThenGid) {
  const auto mesh = scrambled_grid(Ordering::type_major_sfc);
  std::vector<std::tuple<std::uint64_t, GO>> keys;
  for (LO e = 0; e < mesh.count(2); ++e) {
    double cx = 0;
    double cy = 0;
    for (soma::Offset k = mesh.conn_offsets(2)(e); k < mesh.conn_offsets(2)(e + 1); ++k) {
      cx += mesh.coords()(mesh.conn(2)(k), 0) / 4;
      cy += mesh.coords()(mesh.conn(2)(k), 1) / 4;
    }
    keys.emplace_back(reference_key(cx, cy, n), mesh.gid(2)(e));
  }
  EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));
  EXPECT_FALSE(std::is_sorted(mesh.gid(2).data(), mesh.gid(2).data() + mesh.count(2)));
}

// 200 collinear vertices 1e-7 of the extent apart, gids decreasing upwards.
// Each falls in its own cell, so they follow the curve without falling back to
// the gid, which would reverse them.
TEST(MeshOrdering, AGradedColumnFollowsTheCurve) {
  constexpr int count = 200;
  soma::MeshBuilder b(2);
  std::vector<GO> vg;
  std::vector<Real> xyz;
  for (int i = 0; i < count; ++i) {
    vg.push_back(count - i);
    xyz.insert(xyz.end(), {1.0, 1e-7 * i, 0.0});
  }
  vg.insert(vg.end(), {1000, 1001});
  xyz.insert(xyz.end(), {0.0, 0.0, 0.0, 2.0, 1.0, 0.0});
  b.add_vertices(vg, xyz);
  const auto mesh = std::move(b).build(Ordering::type_major_sfc);
  std::vector<std::uint64_t> column;
  for (LO v = 0; v < mesh.count(0); ++v) {
    if (mesh.gid(0)(v) > count) continue;
    column.push_back(reference_key(mesh.coords()(v, 0), mesh.coords()(v, 1), 2.0));
  }
  ASSERT_EQ(column.size(), static_cast<std::size_t>(count));
  for (std::size_t i = 1; i < column.size(); ++i) {
    EXPECT_LT(column[i - 1], column[i]) << "column vertex " << i;
  }
}

// The plane z = 0.3 gets the 2D curve of x and y.
TEST(MeshOrdering, AnExactlyPlanarMeshOffTheOriginFollowsTheTwoAxisCurve) {
  const auto mesh = plane_at_z(0.3);
  std::vector<std::tuple<std::uint64_t, GO>> keys;
  for (LO v = 0; v < mesh.count(0); ++v) {
    keys.emplace_back(reference_key(mesh.coords()(v, 0), mesh.coords()(v, 1), 64),
                      mesh.gid(0)(v));
  }
  EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));
}

// One ulp of noise in z is far below one cell of x or y and changes nothing.
TEST(MeshOrdering, OneUlpOfNoiseOffThePlaneLeavesTheOrderUnchanged) {
  const auto exact = plane_at_z(0.3);
  const auto noisy = plane_at_z(std::nextafter(0.3, 1.0));
  for (const int d : {0, 2}) {
    ASSERT_EQ(exact.count(d), noisy.count(d));
    for (LO i = 0; i < exact.count(d); ++i) {
      ASSERT_EQ(exact.gid(d)(i), noisy.gid(d)(i)) << "dimension " << d << ", entity " << i;
    }
  }
}

// Noise of 1e-4 in z is a few cells of the 3D curve, so z takes part, but on
// the scale of x and y: the mean span stays close to that of the exact plane
// (observed 88.7 against 78.8) instead of following the noise (585 when z is
// stretched over the whole grid).
TEST(MeshOrdering, NoiseJustAboveOneCellKeepsTheSpanOfThePlane) {
  const double exact = soma::vertex_span_report(plane_at_z(0.3), 2).mean;
  const double noisy = soma::vertex_span_report(plane_at_z(0.3 + 1e-4), 2).mean;
  EXPECT_LE(noisy, 1.15 * exact);
}

// The box spans more than the largest Real; the vertices still sort along x.
TEST(MeshOrdering, CoordinatesNearTheLimitOfRealStillOrderAlongTheAxis) {
  soma::MeshBuilder b(1);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5},
                 std::vector<Real>{1e308, 0, 0, -1e308, 0, 0, 0, 0, 0, 5e307, 0, 0, -5e307, 0, 0});
  const auto mesh = std::move(b).build(Ordering::type_major_sfc);
  const std::vector<GO> by_x{2, 5, 3, 4, 1};
  for (LO v = 0; v < mesh.count(0); ++v) {
    EXPECT_EQ(mesh.gid(0)(v), by_x[static_cast<std::size_t>(v)]) << "vertex " << v;
  }
}

TEST(MeshOrdering, FileOrderStaysByGid) {
  const auto mesh = scrambled_grid(Ordering::file);
  EXPECT_TRUE(std::is_sorted(mesh.gid(0).data(), mesh.gid(0).data() + mesh.count(0)));
  EXPECT_TRUE(std::is_sorted(mesh.gid(2).data(), mesh.gid(2).data() + mesh.count(2)));
}

TEST(MeshOrdering, OrderingsHoldTheSameEntities) {
  const auto a = scrambled_grid(Ordering::file);
  const auto b = scrambled_grid(Ordering::type_major_sfc);
  std::vector<GO> ga(a.gid(2).data(), a.gid(2).data() + a.count(2));
  std::vector<GO> gb(b.gid(2).data(), b.gid(2).data() + b.count(2));
  std::sort(gb.begin(), gb.end());
  EXPECT_EQ(ga, gb);
}

TEST(MeshOrdering, ConnectivityStaysConsistentWithTheCoordinates) {
  const auto mesh = scrambled_grid(Ordering::type_major_sfc);
  for (LO e = 0; e < mesh.count(2); ++e) {
    const LO first = mesh.conn(2)(mesh.conn_offsets(2)(e));
    double xmin = 1e9;
    double ymin = 1e9;
    for (soma::Offset k = mesh.conn_offsets(2)(e); k < mesh.conn_offsets(2)(e + 1); ++k) {
      xmin = std::min(xmin, mesh.coords()(mesh.conn(2)(k), 0));
      ymin = std::min(ymin, mesh.coords()(mesh.conn(2)(k), 1));
    }
    EXPECT_DOUBLE_EQ(mesh.coords()(first, 0), xmin) << "element " << e;
    EXPECT_DOUBLE_EQ(mesh.coords()(first, 1), ymin) << "element " << e;
  }
}

TEST(MeshOrdering, SpaceFillingOrderShrinksTheVertexSpan) {
  const auto by_gid = soma::vertex_span_report(scrambled_grid(Ordering::file), 2);
  const auto by_curve = soma::vertex_span_report(scrambled_grid(Ordering::type_major_sfc), 2);
  EXPECT_EQ(by_gid.elements, n * n);
  EXPECT_GT(by_gid.mean, 100.0);
  EXPECT_LT(by_curve.mean, 0.25 * by_gid.mean);
  EXPECT_LT(by_curve.max, by_gid.max);
}

TEST(MeshOrdering, VertexSpanOfASingleQuad) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  const auto r = soma::vertex_span_report(mesh, 2);
  EXPECT_EQ(r.elements, 3);
  EXPECT_EQ(r.max, 4);
  EXPECT_NEAR(r.mean, (4.0 + 4.0 + 4.0) / 3.0, 1e-12);
  EXPECT_EQ(soma::vertex_span_report(mesh, 3).elements, 0);
}

TEST(MeshOrdering, VertexSpanRejectsADimensionOutsideOneToThree) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  EXPECT_THROW(soma::vertex_span_report(mesh, 0), std::out_of_range);
  EXPECT_THROW(soma::vertex_span_report(mesh, 4), std::out_of_range);
  EXPECT_THROW(soma::vertex_span_report(mesh, -1), std::out_of_range);
}

// An element without vertices has span 0 and its connectivity is not read: the
// empty View has no storage, so a read would fault.
TEST(MeshOrdering, VertexSpanOfAnElementWithoutVerticesIsZero) {
  using Offsets = Kokkos::View<soma::Offset*, Kokkos::HostSpace>;
  using Conn = Kokkos::View<LO*, Kokkos::HostSpace>;
  const Offsets empty_offsets("offsets", 2);
  const auto r = soma::detail::vertex_span(1, empty_offsets, Conn{});
  EXPECT_EQ(r.elements, 1);
  EXPECT_EQ(r.max, 0);
  EXPECT_EQ(r.mean, 0.0);
  const Offsets offsets("offsets", 3);
  offsets(1) = 3;
  offsets(2) = 3;
  const Conn conn("conn", 3);
  conn(0) = 4;
  conn(1) = 9;
  conn(2) = 6;
  const auto s = soma::detail::vertex_span(2, offsets, conn);
  EXPECT_EQ(s.elements, 2);
  EXPECT_EQ(s.max, 5);
  EXPECT_DOUBLE_EQ(s.mean, 2.5);
}

TEST(MeshOrdering, TypeBlocksStayTogetherUnderTheCurve) {
  soma::MeshBuilder b(2);
  std::vector<GO> vg;
  std::vector<Real> xyz;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      vg.push_back(1 + i + 3 * j);
      xyz.insert(xyz.end(), {static_cast<Real>(i), static_cast<Real>(j), 0.0});
    }
  }
  b.add_vertices(vg, xyz);
  b.add_elements(Topology::quad, std::vector<GO>{1, 2}, std::vector<GO>{1, 2, 5, 4, 2, 3, 6, 5});
  b.add_elements(Topology::tri, std::vector<GO>{3, 4, 5},
                 std::vector<GO>{4, 5, 8, 5, 6, 9, 4, 8, 7});
  const auto mesh = std::move(b).build(Ordering::type_major_sfc);
  ASSERT_EQ(mesh.segments(2).size(), 2u);
  EXPECT_EQ(mesh.segments(2)[0].topology, Topology::tri);
  EXPECT_EQ(mesh.segments(2)[0].size(), 3);
  EXPECT_EQ(mesh.segments(2)[1].topology, Topology::quad);
  for (const soma::Segment& seg : mesh.segments(2)) {
    const auto keys = centroid_keys(mesh, seg, 2.0);
    EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end())) << soma::name(seg.topology);
  }
}

// Pentagons and hexagons on a 4 x 4 grid with scrambled gids: the pentagons
// come first, and each block follows the centroid key, then the gid.
TEST(MeshOrdering, PolygonsFollowTheCentroidKeyWithinEachArity) {
  const std::vector<std::array<double, 2>> pentagon{
      {-1, -1}, {1, -1}, {1, 0.5}, {0, 1}, {-1, 0.5}};
  const std::vector<std::array<double, 2>> hexagon{
      {-0.5, -1}, {0.5, -1}, {1, 0}, {0.5, 1}, {-0.5, 1}, {-1, 0}};
  soma::MeshBuilder b(2);
  std::vector<GO> vg;
  std::vector<Real> xyz;
  std::vector<GO> pg;
  std::vector<soma::Offset> off{0};
  std::vector<GO> conn;
  for (int p = 0; p < 16; ++p) {
    const double cx = 1 + 2 * (p % 4);
    const double cy = 1 + 2 * (p / 4);
    const auto& ring = (p + p / 4) % 2 == 0 ? pentagon : hexagon;
    for (const auto& r : ring) {
      vg.push_back(static_cast<GO>(vg.size() + 1));
      xyz.insert(xyz.end(), {cx + r[0], cy + r[1], 0.0});
      conn.push_back(vg.back());
    }
    pg.push_back(1 + (7 * p) % 16);
    off.push_back(static_cast<soma::Offset>(conn.size()));
  }
  b.add_vertices(vg, xyz);
  b.add_polygons(pg, off, conn);
  const auto mesh = std::move(b).build(Ordering::type_major_sfc);
  const auto segs = mesh.segments(2);
  ASSERT_EQ(segs.size(), 2u);
  EXPECT_EQ(segs[0].arity, 5);
  EXPECT_EQ(segs[0].size(), 8);
  EXPECT_EQ(segs[1].arity, 6);
  EXPECT_EQ(segs[1].size(), 8);
  for (const soma::Segment& seg : segs) {
    const auto keys = centroid_keys(mesh, seg, 8.0);
    EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end())) << "arity " << seg.arity;
    const GO* g = mesh.gid(2).data();
    EXPECT_FALSE(std::is_sorted(g + seg.begin, g + seg.end)) << "arity " << seg.arity;
  }
}
