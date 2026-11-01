// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mesh_fixture.hpp"
#include "soma/core/mesh_builder.hpp"

using soma::GO;
using soma::LO;
using soma::Ownership;
using soma::Segment;
using soma::Topology;

TEST(MeshBuilder, CountsAndDimension) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  EXPECT_EQ(mesh.dim(), 2);
  EXPECT_EQ(mesh.count(0), 7);
  EXPECT_EQ(mesh.count(2), 3);
  EXPECT_EQ(mesh.count(1), 0);
}

TEST(MeshBuilder, VerticesAreSortedByGid) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  const std::vector<GO> expected{10, 20, 30, 40, 50, 60, 70};
  for (LO i = 0; i < 7; ++i) EXPECT_EQ(mesh.gid(0)(i), expected[static_cast<std::size_t>(i)]);
  EXPECT_DOUBLE_EQ(mesh.coords()(6, 0), 3.0);
  EXPECT_DOUBLE_EQ(mesh.coords()(6, 1), 0.5);
  EXPECT_DOUBLE_EQ(mesh.coords()(4, 1), 1.0);
}

TEST(MeshBuilder, ElementsAreSortedByTopologyThenGid) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  EXPECT_EQ(mesh.gid(2)(0), 5);
  EXPECT_EQ(mesh.gid(2)(1), 3);
  EXPECT_EQ(mesh.gid(2)(2), 7);
  EXPECT_EQ(mesh.topology(2)(0), static_cast<std::uint8_t>(Topology::tri));
  EXPECT_EQ(mesh.topology(2)(1), static_cast<std::uint8_t>(Topology::quad));
}

// The same fixture under the curve. Vertex keys (box 3 x 1, so a 2D curve):
// 10 and 20 sit on the first cells of the curve, 70 at the far end of x.
TEST(MeshBuilder, SpaceFillingOrderOnTheFixture) {
  const auto mesh = soma_test::two_quads_and_a_triangle(soma::Ordering::type_major_sfc);
  const std::vector<GO> vertices{10, 20, 40, 50, 60, 70, 30};
  for (LO i = 0; i < 7; ++i) {
    EXPECT_EQ(mesh.gid(0)(i), vertices[static_cast<std::size_t>(i)]) << "vertex " << i;
  }
  const std::vector<GO> elements{5, 3, 7};
  for (LO e = 0; e < 3; ++e) {
    EXPECT_EQ(mesh.gid(2)(e), elements[static_cast<std::size_t>(e)]) << "element " << e;
  }
  const auto off = mesh.conn_offsets(2);
  const auto conn = mesh.conn(2);
  const std::vector<LO> tri{6, 5, 4};
  for (LO k = 0; k < 3; ++k) EXPECT_EQ(conn(off(0) + k), tri[static_cast<std::size_t>(k)]);
  const std::vector<LO> quad3{1, 6, 4, 3};
  for (LO k = 0; k < 4; ++k) EXPECT_EQ(conn(off(1) + k), quad3[static_cast<std::size_t>(k)]);
  for (LO e = 0; e < 3; ++e) {
    for (soma::Offset k = off(e); k < off(e + 1); ++k) {
      EXPECT_LT(conn(k), mesh.count(0));
    }
  }
  EXPECT_EQ(mesh.segments(2)[0].topology, Topology::tri);
}

TEST(MeshBuilder, ConnectivityUsesLocalIds) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  const auto off = mesh.conn_offsets(2);
  const auto conn = mesh.conn(2);
  EXPECT_EQ(off(0), 0);
  EXPECT_EQ(off(1), 3);
  EXPECT_EQ(off(3), 11);
  const std::vector<LO> tri{2, 6, 5};
  for (LO k = 0; k < 3; ++k) EXPECT_EQ(conn(off(0) + k), tri[static_cast<std::size_t>(k)]);
  const std::vector<LO> quad3{1, 2, 5, 4};
  for (LO k = 0; k < 4; ++k) EXPECT_EQ(conn(off(1) + k), quad3[static_cast<std::size_t>(k)]);
}

TEST(MeshBuilder, SegmentsAndOwnershipBoundaries) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  const auto seg = mesh.segments(2);
  ASSERT_EQ(seg.size(), 2u);
  EXPECT_EQ(seg[0], (Segment{0, 1, Topology::tri, 3, Ownership::interior}));
  EXPECT_EQ(seg[1], (Segment{1, 3, Topology::quad, 4, Ownership::interior}));
  EXPECT_EQ(mesh.segment_table(2).owned_end, 3);
  EXPECT_EQ(mesh.segment_table(2).interior_end, 3);
  ASSERT_EQ(mesh.segments(0).size(), 1u);
  EXPECT_EQ(mesh.segments(0)[0].end, 7);
}

TEST(MeshBuilder, DuplicateVertexGidThrows) {
  soma::MeshBuilder b(2);
  const std::vector<GO> vg{1, 2, 1};
  const std::vector<soma::Real> xyz(9, 0.0);
  b.add_vertices(vg, xyz);
  try {
    std::move(b).build();
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("duplicate vertex gid 1"), std::string::npos) << e.what();
  }
}

TEST(MeshBuilder, UnknownVertexGidNamesTheGid) {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3}, std::vector<soma::Real>(9, 0.0));
  b.add_elements(Topology::tri, std::vector<GO>{9}, std::vector<GO>{1, 2, 77});
  try {
    std::move(b).build();
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("77"), std::string::npos) << e.what();
  }
}

TEST(MeshBuilder, ElementGidsAreUniquePerDimensionAcrossTopologies) {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4}, std::vector<soma::Real>(12, 0.0));
  b.add_elements(Topology::tri, std::vector<GO>{5}, std::vector<GO>{1, 2, 3});
  b.add_elements(Topology::quad, std::vector<GO>{5}, std::vector<GO>{1, 2, 3, 4});
  try {
    std::move(b).build();
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("gid 5"), std::string::npos) << msg;
    EXPECT_NE(msg.find("tri"), std::string::npos) << msg;
    EXPECT_NE(msg.find("quad"), std::string::npos) << msg;
  }
}

TEST(MeshBuilder, ElementGidsAreUniqueAcrossPolygonArities) {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6}, std::vector<soma::Real>(18, 0.0));
  b.add_polygons(std::vector<GO>{9}, std::vector<soma::Offset>{0, 5}, std::vector<GO>{1, 2, 3, 4, 5});
  b.add_polygons(std::vector<GO>{9}, std::vector<soma::Offset>{0, 6}, std::vector<GO>{1, 2, 3, 4, 5, 6});
  EXPECT_THROW(std::move(b).build(), std::invalid_argument);
}

TEST(MeshBuilder, ADuplicateGidWithinOneTopologyStillThrows) {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3}, std::vector<soma::Real>(9, 0.0));
  b.add_elements(Topology::tri, std::vector<GO>{4, 4}, std::vector<GO>{1, 2, 3, 1, 2, 3});
  try {
    std::move(b).build();
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("duplicate element gid 4"), std::string::npos) << e.what();
  }
}

TEST(MeshBuilder, TheSameGidInDifferentDimensionsIsFine) {
  soma::MeshBuilder b(3);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8}, std::vector<soma::Real>(24, 0.0));
  b.add_elements(Topology::hex, std::vector<GO>{5}, std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8});
  b.add_elements(Topology::quad, std::vector<GO>{5}, std::vector<GO>{1, 2, 3, 4});
  const auto mesh = std::move(b).build();
  EXPECT_EQ(mesh.count(3), 1);
  EXPECT_EQ(mesh.count(2), 1);
}

TEST(MeshBuilder, TheDefaultOrderingIsNamedAndImplemented) {
  EXPECT_EQ(soma::default_ordering, soma::Ordering::file);
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3}, std::vector<soma::Real>(9, 0.0));
  b.add_elements(Topology::tri, std::vector<GO>{1}, std::vector<GO>{1, 2, 3});
  EXPECT_EQ(std::move(b).build().count(2), 1);
}

TEST(MeshBuilder, PolygonsOfDifferentArityFormSeparateSegments) {
  soma::MeshBuilder b(2);
  std::vector<GO> vg(9);
  for (GO i = 0; i < 9; ++i) vg[static_cast<std::size_t>(i)] = i + 1;
  b.add_vertices(vg, std::vector<soma::Real>(27, 0.0));
  const std::vector<soma::Offset> off{0, 5, 11};
  b.add_polygons(std::vector<GO>{1, 2}, off,
                 std::vector<GO>{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6});
  const auto mesh = std::move(b).build();
  ASSERT_EQ(mesh.segments(2).size(), 2u);
  EXPECT_EQ(mesh.segments(2)[0].arity, 5);
  EXPECT_EQ(mesh.segments(2)[1].arity, 6);
}

TEST(MeshBuilder, UnsupportedInputsThrow) {
  soma::MeshBuilder b(3);
  EXPECT_THROW(b.add_elements(Topology::polyhedron, std::vector<GO>{1}, std::vector<GO>{}),
               std::invalid_argument);
  EXPECT_THROW(b.add_elements(Topology::polygon, std::vector<GO>{1}, std::vector<GO>{1, 2, 3}),
               std::invalid_argument);
  EXPECT_THROW(b.add_elements(Topology::hex, std::vector<GO>{1}, std::vector<GO>{1, 2, 3}),
               std::invalid_argument);
  EXPECT_THROW(soma::MeshBuilder(4), std::invalid_argument);
  EXPECT_THROW(std::move(b).build(soma::Ordering::tiled), std::invalid_argument);
}
