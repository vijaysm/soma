// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mesh_fixture.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/fields/domain.hpp"

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

// The same fixture under the curve. The box is 3 x 1, so a 2D curve over a
// 3 x 3 square; the order is that of hilbert_reference_2d on the cells
// floor(v / 3 * (2^32 - 1)) of the vertices and of the element centroids.
TEST(MeshBuilder, SpaceFillingOrderOnTheFixture) {
  const auto mesh = soma_test::two_quads_and_a_triangle(soma::Ordering::type_major_sfc);
  const std::vector<GO> vertices{10, 20, 50, 40, 60, 30, 70};
  for (LO i = 0; i < 7; ++i) {
    EXPECT_EQ(mesh.gid(0)(i), vertices[static_cast<std::size_t>(i)]) << "vertex " << i;
  }
  const std::vector<GO> elements{5, 7, 3};
  for (LO e = 0; e < 3; ++e) {
    EXPECT_EQ(mesh.gid(2)(e), elements[static_cast<std::size_t>(e)]) << "element " << e;
  }
  const auto off = mesh.conn_offsets(2);
  const auto conn = mesh.conn(2);
  const std::vector<LO> tri{5, 6, 4};
  for (LO k = 0; k < 3; ++k) EXPECT_EQ(conn(off(0) + k), tri[static_cast<std::size_t>(k)]);
  const std::vector<LO> quad3{1, 5, 4, 2};
  for (LO k = 0; k < 4; ++k) EXPECT_EQ(conn(off(2) + k), quad3[static_cast<std::size_t>(k)]);
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

TEST(MeshBuilder, VertexCoordinatesMustHoldThreeValuesPerVertex) {
  soma::MeshBuilder b(2);
  EXPECT_THROW(b.add_vertices(std::vector<GO>{1, 2}, std::vector<soma::Real>(5, 0.0)),
               std::invalid_argument);
  EXPECT_THROW(b.add_vertices(std::vector<GO>{1}, std::vector<soma::Real>(4, 0.0)),
               std::invalid_argument);
  b.add_vertices(std::vector<GO>{1}, std::vector<soma::Real>{1, 2, 3});
  const auto mesh = std::move(b).build();
  ASSERT_EQ(mesh.count(0), 1);
  EXPECT_DOUBLE_EQ(mesh.coords()(0, 2), 3.0);
}

namespace {

// The message of the exception add_vertices throws for vertices 11, 12, 13
// whose coordinate `slot` is `bad`, or "" when it accepts them.
std::string non_finite_error(std::size_t slot, soma::Real bad) {
  soma::MeshBuilder b(2);
  std::vector<soma::Real> xyz{0, 0, 0, 1, 0, 0, 0, 1, 0};
  xyz[slot] = bad;
  try {
    b.add_vertices(std::vector<GO>{11, 12, 13}, xyz);
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST(MeshBuilder, NonFiniteCoordinatesAreRejectedByVertexGid) {
  const soma::Real nan = std::numeric_limits<soma::Real>::quiet_NaN();
  const soma::Real inf = std::numeric_limits<soma::Real>::infinity();
  const struct {
    std::size_t slot;
    soma::Real value;
    const char* gid;
  } cases[] = {{0, nan, "11"}, {7, nan, "13"}, {2, inf, "11"}, {4, -inf, "12"}, {8, inf, "13"}};
  for (const auto& c : cases) {
    const std::string what = non_finite_error(c.slot, c.value);
    EXPECT_NE(what.find(std::string("vertex gid ") + c.gid), std::string::npos)
        << "slot " << c.slot << ": '" << what << "'";
  }
  EXPECT_EQ(non_finite_error(0, 1e308), "");
}

// The check runs for every ordering, and a rejected block leaves nothing behind.
TEST(MeshBuilder, ARejectedVertexBlockLeavesTheBuilderUsable) {
  soma::MeshBuilder b(2);
  const soma::Real nan = std::numeric_limits<soma::Real>::quiet_NaN();
  EXPECT_THROW(b.add_vertices(std::vector<GO>{1, 2}, std::vector<soma::Real>{0, 0, 0, 1, nan, 0}),
               std::invalid_argument);
  b.add_vertices(std::vector<GO>{1}, std::vector<soma::Real>{4, 5, 6});
  const auto mesh = std::move(b).build(soma::Ordering::file);
  ASSERT_EQ(mesh.count(0), 1);
  EXPECT_EQ(mesh.coords()(0, 0), 4.0);
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
  b.add_polygons(std::vector<GO>{9}, std::vector<soma::Offset>{0, 5},
                 std::vector<GO>{1, 2, 3, 4, 5});
  b.add_polygons(std::vector<GO>{9}, std::vector<soma::Offset>{0, 6},
                 std::vector<GO>{1, 2, 3, 4, 5, 6});
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

TEST(MeshBuilder, TheDefaultOrderingIsTheSpaceFillingCurve) {
  EXPECT_EQ(soma::default_ordering, soma::Ordering::type_major_sfc);
  const auto build = [](bool explicit_ordering) {
    soma::MeshBuilder b(2);
    b.add_vertices(std::vector<GO>{50, 10, 70, 30, 20, 60, 40},
                   std::vector<soma::Real>{1, 1, 0, 0, 0, 0, 3, 0.5, 0, 2, 0, 0, 1, 0, 0, 2, 1, 0,
                                           0, 1, 0});
    b.add_elements(Topology::quad, std::vector<GO>{7, 3},
                   std::vector<GO>{10, 20, 50, 40, 20, 30, 60, 50});
    b.add_elements(Topology::tri, std::vector<GO>{5}, std::vector<GO>{30, 70, 60});
    return explicit_ordering ? std::move(b).build(soma::Ordering::type_major_sfc)
                             : std::move(b).build();
  };
  const auto a = build(true);
  const auto d = build(false);
  for (LO v = 0; v < 7; ++v) EXPECT_EQ(a.gid(0)(v), d.gid(0)(v));
  for (LO e = 0; e < 3; ++e) EXPECT_EQ(a.gid(2)(e), d.gid(2)(e));
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

namespace {

using Offsets = std::vector<soma::Offset>;

soma::MeshBuilder six_vertices() {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6}, std::vector<soma::Real>(18, 0.0));
  return b;
}

std::string polygon_error(const Offsets& offsets, const std::vector<GO>& vertex_gids) {
  auto b = six_vertices();
  std::vector<GO> gids(offsets.empty() ? 0 : offsets.size() - 1);
  for (std::size_t i = 0; i < gids.size(); ++i) gids[i] = static_cast<GO>(i + 1);
  try {
    b.add_polygons(gids, offsets, vertex_gids);
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST(MeshBuilder, PolygonOffsetsMustStartAtZero) {
  EXPECT_NE(polygon_error(Offsets{1, 4}, std::vector<GO>{1, 2, 3, 4}).find("start at 0"),
            std::string::npos);
  EXPECT_NE(polygon_error(Offsets{2, 5}, std::vector<GO>{1, 2, 3, 4, 5}).find("start at 0"),
            std::string::npos);
}

TEST(MeshBuilder, PolygonOffsetsMustNotDecrease) {
  EXPECT_NE(polygon_error(Offsets{0, 3, 0}, std::vector<GO>{1, 2, 3}).find("decrease"),
            std::string::npos);
}

TEST(MeshBuilder, PolygonOffsetsMustEndAtTheConnectivitySize) {
  EXPECT_NE(polygon_error(Offsets{0, 10}, std::vector<GO>{1, 2, 3}).find("end at"),
            std::string::npos);
  EXPECT_NE(polygon_error(Offsets{0, 3}, std::vector<GO>{1, 2, 3, 4}).find("end at"),
            std::string::npos);
}

TEST(MeshBuilder, PolygonArityMustFitTheSegmentArity) {
  EXPECT_NE(polygon_error(Offsets{0, 0, 3}, std::vector<GO>{1, 2, 3}).find("arity 0"),
            std::string::npos);
  EXPECT_NE(polygon_error(Offsets{0, 2}, std::vector<GO>{1, 2}).find("arity 2"),
            std::string::npos);
  std::vector<GO> big(65536);
  for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<GO>(i % 6 + 1);
  EXPECT_NE(polygon_error(Offsets{0, 65536}, big).find("arity 65536"), std::string::npos);
  big.pop_back();
  EXPECT_EQ(polygon_error(Offsets{0, 65535}, big), "");
}

TEST(MeshBuilder, ARejectedPolygonBlockLeavesTheBuilderUsable) {
  auto b = six_vertices();
  EXPECT_THROW(b.add_polygons(std::vector<GO>{1}, Offsets{0, 10}, std::vector<GO>{1, 2, 3}),
               std::invalid_argument);
  b.add_polygons(std::vector<GO>{2}, Offsets{0, 3}, std::vector<GO>{1, 2, 3});
  const auto mesh = std::move(b).build();
  EXPECT_EQ(mesh.count(2), 1);
  EXPECT_EQ(mesh.gid(2)(0), 2);
}

TEST(MeshBuilder, CheckedConversionToOffsetThrowsPastTheLimit) {
  constexpr auto off_max = static_cast<std::size_t>(std::numeric_limits<soma::Offset>::max());
  EXPECT_EQ(soma::detail::to_offset(0, "x"), 0);
  EXPECT_EQ(soma::detail::to_offset(off_max, "x"), std::numeric_limits<soma::Offset>::max());
  try {
    (void)soma::detail::to_offset(off_max + 1, "connectivity size");
    FAIL() << "expected std::length_error";
  } catch (const std::length_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("connectivity size"), std::string::npos) << msg;
    EXPECT_NE(msg.find(std::to_string(off_max)), std::string::npos) << msg;
  }
  EXPECT_THROW((void)soma::detail::to_offset(std::numeric_limits<std::size_t>::max(), "x"),
               std::length_error);
}

TEST(MeshBuilder, ElementDimensionMustBeInOneToTheBuilderDimension) {
  soma::MeshBuilder b2(2);
  b2.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8}, std::vector<soma::Real>(24, 0.0));
  EXPECT_THROW(b2.add_elements(Topology::vertex, std::vector<GO>{1}, std::vector<GO>{1}),
               std::invalid_argument);
  EXPECT_THROW(b2.add_elements(Topology::hex, std::vector<GO>{1},
                               std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8}),
               std::invalid_argument);
  b2.add_elements(Topology::quad, std::vector<GO>{1}, std::vector<GO>{1, 2, 3, 4});
  b2.add_elements(Topology::edge, std::vector<GO>{1}, std::vector<GO>{1, 2});
  const auto mesh = std::move(b2).build();
  EXPECT_EQ(mesh.count(3), 0);
  EXPECT_EQ(mesh.count(2), 1);
  EXPECT_EQ(mesh.count(1), 1);

  soma::MeshBuilder b1(1);
  EXPECT_THROW(b1.add_elements(Topology::tri, std::vector<GO>{1}, std::vector<GO>{1, 2, 3}),
               std::invalid_argument);
  EXPECT_THROW(b1.add_polygons(std::vector<GO>{1}, std::vector<soma::Offset>{0, 3},
                               std::vector<GO>{1, 2, 3}),
               std::invalid_argument);
}
