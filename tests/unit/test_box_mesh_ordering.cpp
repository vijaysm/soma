// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "box_mesh_check.hpp"
#include "soma/core/locality.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/testing/box_mesh.hpp"

using soma::GO;
using soma::LO;
using soma::Ordering;
using soma::Topology;
using soma::testing::BoxSpec;

namespace {

BoxSpec spec_of(int nx, int ny, int nz, Topology element, Ordering ordering) {
  BoxSpec s;
  s.nx = nx;
  s.ny = ny;
  s.nz = nz;
  s.element = element;
  s.lx = 0.1;
  s.ly = 0.3;
  s.lz = 0.7;
  s.ordering = ordering;
  return s;
}

std::string describe(const BoxSpec& s) {
  return std::string(soma::name(s.element)) + " " + std::to_string(s.nx) + "x" +
         std::to_string(s.ny) + "x" + std::to_string(s.nz);
}

std::vector<GO> gids_of(const soma::HostMesh& m, int dim) {
  return std::vector<GO>(m.gid(dim).data(), m.gid(dim).data() + m.count(dim));
}

std::vector<BoxSpec> shapes(Ordering ordering) {
  return {spec_of(8, 8, 1, Topology::quad, ordering),   spec_of(16, 16, 1, Topology::quad, ordering),
          spec_of(32, 32, 1, Topology::quad, ordering), spec_of(16, 16, 1, Topology::tri, ordering),
          spec_of(8, 8, 8, Topology::hex, ordering),    spec_of(16, 16, 16, Topology::hex, ordering),
          spec_of(8, 16, 16, Topology::hex, ordering),  spec_of(4, 4, 32, Topology::hex, ordering),
          spec_of(64, 16, 1, Topology::quad, ordering), spec_of(3, 5, 1, Topology::tri, ordering)};
}

}  // namespace

TEST(BoxMeshOrdering, TheCurveHoldsTheSameEntitiesAndTheSameCoordinatesPerGid) {
  for (const BoxSpec& file_spec : shapes(Ordering::file)) {
    SCOPED_TRACE(describe(file_spec));
    BoxSpec curve_spec = file_spec;
    curve_spec.ordering = Ordering::type_major_sfc;
    const auto by_file = soma::testing::box_mesh(file_spec);
    const auto by_curve = soma::testing::box_mesh(curve_spec);
    ASSERT_NO_FATAL_FAILURE(soma_test::expect_box_invariants(by_curve, curve_spec));

    const int dim = by_file.dim();
    std::vector<GO> vertices = gids_of(by_curve, 0);
    std::vector<GO> elements = gids_of(by_curve, dim);
    std::sort(vertices.begin(), vertices.end());
    std::sort(elements.begin(), elements.end());
    EXPECT_EQ(vertices, gids_of(by_file, 0));
    EXPECT_EQ(elements, gids_of(by_file, dim));

    std::map<GO, std::vector<double>> where;
    for (LO v = 0; v < by_file.count(0); ++v) {
      where[by_file.gid(0)(v)] = {by_file.coords()(v, 0), by_file.coords()(v, 1),
                                  by_file.coords()(v, 2)};
    }
    for (LO v = 0; v < by_curve.count(0); ++v) {
      const std::vector<double>& p = where.at(by_curve.gid(0)(v));
      for (int c = 0; c < 3; ++c) {
        EXPECT_EQ(by_curve.coords()(v, c), p[static_cast<std::size_t>(c)]) << "vertex gid " << by_curve.gid(0)(v);
      }
    }
  }
}

TEST(BoxMeshOrdering, TheCurveReordersWhatFileOrderLeavesByGid) {
  for (const Topology t : {Topology::quad, Topology::tri, Topology::hex}) {
    const int nz = t == Topology::hex ? 8 : 1;
    const auto by_file = soma::testing::box_mesh(spec_of(8, 8, nz, t, Ordering::file));
    const auto by_curve = soma::testing::box_mesh(spec_of(8, 8, nz, t, Ordering::type_major_sfc));
    const int dim = by_file.dim();
    EXPECT_TRUE(std::is_sorted(by_file.gid(0).data(), by_file.gid(0).data() + by_file.count(0)));
    EXPECT_FALSE(std::is_sorted(by_curve.gid(0).data(), by_curve.gid(0).data() + by_curve.count(0)))
        << soma::name(t);
    EXPECT_FALSE(std::is_sorted(by_curve.gid(dim).data(), by_curve.gid(dim).data() + by_curve.count(dim)))
        << soma::name(t);
    const auto again = soma::testing::box_mesh(spec_of(8, 8, nz, t, Ordering::type_major_sfc));
    EXPECT_EQ(gids_of(again, 0), gids_of(by_curve, 0));
    EXPECT_EQ(gids_of(again, dim), gids_of(by_curve, dim));
  }
}

TEST(BoxMeshOrdering, FileOrderVertexSpanIsTheClosedForm) {
  for (const BoxSpec& spec : shapes(Ordering::file)) {
    SCOPED_TRACE(describe(spec));
    const auto mesh = soma::testing::box_mesh(spec);
    const auto report = soma::vertex_span_report(mesh, mesh.dim());
    const int px = spec.nx + 1;
    const int py = spec.ny + 1;
    const int span = spec.element == Topology::hex ? 1 + px + px * py : px + 1;
    EXPECT_EQ(report.elements, mesh.count(mesh.dim()));
    EXPECT_EQ(report.max, span);
    EXPECT_DOUBLE_EQ(report.mean, static_cast<double>(span));
  }
}

// The file-order span is the closed form checked above: it grows with the
// vertices per row, so file order is good when i is the short axis and poor
// when it is the long one. The curve does not depend on the axis lengths. On
// compact boxes the two agree to within 17 per cent (observed) and the bound
// below is 50 per cent; on elongated boxes the curve wins when i is long
// (64 x 16 quads: 0.66 of the file span) and loses when it is short
// (4 x 4 x 32 hexes: 6 times), so nothing is asserted there. The largest span
// under the curve was several times the file-order one in every shape probed.
TEST(BoxMeshOrdering, TheCurveKeepsTheMeanVertexSpanNearFileOrderOnCompactBoxes) {
  const std::vector<BoxSpec> compact{
      spec_of(8, 8, 1, Topology::quad, Ordering::file),   spec_of(16, 16, 1, Topology::quad, Ordering::file),
      spec_of(32, 32, 1, Topology::quad, Ordering::file), spec_of(16, 16, 1, Topology::tri, Ordering::file),
      spec_of(8, 8, 8, Topology::hex, Ordering::file),    spec_of(16, 16, 16, Topology::hex, Ordering::file)};
  for (BoxSpec spec : compact) {
    SCOPED_TRACE(describe(spec));
    const auto by_file = soma::testing::box_mesh(spec);
    spec.ordering = Ordering::type_major_sfc;
    const auto by_curve = soma::testing::box_mesh(spec);
    const double file_mean = soma::vertex_span_report(by_file, by_file.dim()).mean;
    const double curve_mean = soma::vertex_span_report(by_curve, by_curve.dim()).mean;
    EXPECT_LE(curve_mean, 1.5 * file_mean);
  }
}
