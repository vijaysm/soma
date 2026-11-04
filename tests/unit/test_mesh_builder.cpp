// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mesh_fixture.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/sfc.hpp"
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
  EXPECT_EQ(mesh.segments(0)[0], (Segment{0, 7, Topology::vertex, 1, Ownership::interior}));
  EXPECT_EQ(mesh.segment_table(0).count, 7);
  EXPECT_EQ(mesh.segment_table(0).owned_end, 7);
  EXPECT_EQ(mesh.segment_table(0).interior_end, 7);
  EXPECT_EQ(mesh.segment_table(2).count, 3);
}

// Without entities there are no segments and both ends are 0.
TEST(MeshBuilder, AnEmptyMeshHasNoSegments) {
  const auto mesh = soma::MeshBuilder(2).build();
  for (int d = 0; d <= 2; ++d) {
    EXPECT_TRUE(mesh.segments(d).empty()) << d;
    EXPECT_EQ(mesh.segment_table(d).count, 0) << d;
    EXPECT_EQ(mesh.segment_table(d).owned_end, 0) << d;
    EXPECT_EQ(mesh.segment_table(d).interior_end, 0) << d;
  }
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

TEST(MeshBuilder, AGidBelowTheSmallestVertexGidIsUnknown) {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3}, std::vector<soma::Real>(9, 0.0));
  b.add_elements(Topology::tri, std::vector<GO>{9}, std::vector<GO>{1, 0, 3});
  try {
    std::move(b).build();
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("unknown vertex gid 0"), std::string::npos) << e.what();
  }
}

TEST(MeshBuilder, GidsSpanningTheWholeRangeAreFound) {
  constexpr GO lo = std::numeric_limits<GO>::min();
  constexpr GO hi = std::numeric_limits<GO>::max();
  soma::MeshBuilder b(1);
  b.add_vertices(std::vector<GO>{hi, lo}, std::vector<soma::Real>{1, 0, 0, 0, 0, 0});
  b.add_elements(Topology::edge, std::vector<GO>{1}, std::vector<GO>{lo, hi});
  const auto m = std::move(b).build(soma::Ordering::file);
  EXPECT_EQ(m.gid(0)(0), lo);
  EXPECT_EQ(m.conn(1)(0), 0);
  EXPECT_EQ(m.conn(1)(1), 1);
}

// Contiguous vertex gids are looked up by index, others by search; both must
// give the same mesh.
TEST(MeshBuilder, ContiguousAndScatteredGidsGiveTheSameConnectivity) {
  const auto build = [](GO scale) {
    soma::MeshBuilder b(2);
    std::vector<GO> vg;
    for (GO g = 1; g <= 6; ++g) vg.push_back(g * scale);
    b.add_vertices(vg, std::vector<soma::Real>{0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0});
    const std::vector<GO> quads{1, 2, 5, 4, 2, 3, 6, 5};
    std::vector<GO> conn;
    for (const GO g : quads) conn.push_back(g * scale);
    b.add_elements(Topology::quad, std::vector<GO>{8, 7}, conn);
    return std::move(b).build(soma::Ordering::file);
  };
  const auto a = build(1);
  const auto s = build(10);
  ASSERT_EQ(a.count(2), 2);
  ASSERT_EQ(s.count(2), 2);
  const std::vector<LO> want{1, 2, 5, 4, 0, 1, 4, 3};
  for (std::size_t k = 0; k < want.size(); ++k) {
    EXPECT_EQ(a.conn(2)(k), want[k]) << k;
    EXPECT_EQ(s.conn(2)(k), want[k]) << k;
  }
}

// Sorted input skips the sorts; the mesh must not depend on it.
TEST(MeshBuilder, InputInAnyOrderGivesTheSameMesh) {
  const std::vector<soma::Real> xyz{0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0};
  const auto build = [&](bool reversed, soma::Ordering ordering) {
    soma::MeshBuilder b(2);
    std::vector<GO> vg{1, 2, 3, 4, 5, 6};
    std::vector<soma::Real> v = xyz;
    if (reversed) {
      std::reverse(vg.begin(), vg.end());
      for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t c = 0; c < 3; ++c) std::swap(v[3 * i + c], v[3 * (5 - i) + c]);
      }
    }
    b.add_vertices(vg, v);
    if (reversed) {
      b.add_elements(Topology::tri, std::vector<GO>{12, 11}, std::vector<GO>{3, 6, 5, 2, 3, 5});
      b.add_elements(Topology::quad, std::vector<GO>{10}, std::vector<GO>{1, 2, 5, 4});
    } else {
      b.add_elements(Topology::quad, std::vector<GO>{10}, std::vector<GO>{1, 2, 5, 4});
      b.add_elements(Topology::tri, std::vector<GO>{11, 12}, std::vector<GO>{2, 3, 5, 3, 6, 5});
    }
    return std::move(b).build(ordering);
  };
  for (const auto ordering : {soma::Ordering::file, soma::Ordering::type_major_sfc}) {
    const auto a = build(false, ordering);
    const auto r = build(true, ordering);
    for (int d : {0, 2}) {
      ASSERT_EQ(a.count(d), r.count(d));
      for (LO i = 0; i < a.count(d); ++i) EXPECT_EQ(a.gid(d)(i), r.gid(d)(i)) << "dim " << d << " entity " << i;
    }
    for (std::size_t k = 0; k < a.conn(2).extent(0); ++k) EXPECT_EQ(a.conn(2)(k), r.conn(2)(k)) << k;
    for (LO i = 0; i < a.count(0); ++i) EXPECT_EQ(a.coords()(i, 0), r.coords()(i, 0)) << i;
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

namespace {

// Vertices 1..6 on a 3 x 2 grid; quads 100 and 101 and a triangle 102.
soma::MeshBuilder grid_builder() {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6},
                 std::vector<soma::Real>{0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0});
  b.add_elements(Topology::quad, std::vector<GO>{100, 101}, std::vector<GO>{1, 2, 5, 4, 2, 3, 6, 5});
  b.add_elements(Topology::tri, std::vector<GO>{102}, std::vector<GO>{1, 2, 5});
  return b;
}

std::vector<GO> gids(const soma::HostMesh& m, int d) {
  std::vector<GO> out;
  for (LO e = 0; e < m.count(d); ++e) out.push_back(m.gid(d)(e));
  return out;
}

}  // namespace

// Ownership first, then (for vertices) group, then the usual order.
TEST(MeshBuilder, OwnershipOrdersVerticesByKindThenGroup) {
  soma::MeshBuilder b = grid_builder();
  b.set_ownership(0, std::vector<GO>{2, 5, 3, 6},
                  std::vector<Ownership>{Ownership::shared, Ownership::shared, Ownership::shared_copy, Ownership::ghost},
                  std::vector<std::uint32_t>{1, 0, 0, 0});
  const soma::HostMesh m = std::move(b).build(soma::Ordering::file);
  EXPECT_EQ(gids(m, 0), (std::vector<GO>{1, 4, 5, 2, 3, 6}));
  const auto& t = m.segment_table(0);
  EXPECT_EQ(t.interior_end, 2);
  EXPECT_EQ(t.owned_end, 4);
  EXPECT_EQ(t.shared_end, 5);
  ASSERT_EQ(t.segments.size(), 4U);
  EXPECT_EQ(t.segments[2], (Segment{4, 5, Topology::vertex, 1, Ownership::shared_copy}));
  // Connectivity still names the same vertices.
  const auto conn = m.conn(2);
  const auto off = m.conn_offsets(2);
  std::vector<std::vector<GO>> elems;
  for (LO e = 0; e < m.count(2); ++e) {
    std::vector<GO> vs;
    for (soma::Offset k = off(e); k < off(e + 1); ++k) vs.push_back(m.gid(0)(conn(k)));
    elems.push_back(vs);
  }
  EXPECT_EQ(gids(m, 2), (std::vector<GO>{102, 100, 101}));
  EXPECT_EQ(elems[0], (std::vector<GO>{1, 2, 5}));
  EXPECT_EQ(elems[2], (std::vector<GO>{2, 3, 6, 5}));
}

TEST(MeshBuilder, OwnershipComesBeforeTopologyForElements) {
  soma::MeshBuilder b = grid_builder();
  b.set_ownership(2, std::vector<GO>{102, 100}, std::vector<Ownership>{Ownership::ghost, Ownership::shared},
                  std::vector<std::uint32_t>{0, 0});
  const soma::HostMesh m = std::move(b).build(soma::Ordering::file);
  EXPECT_EQ(gids(m, 2), (std::vector<GO>{101, 100, 102}));
  const auto& t = m.segment_table(2);
  EXPECT_EQ(t.interior_end, 1);
  EXPECT_EQ(t.owned_end, 2);
  EXPECT_EQ(t.shared_end, 2);
  EXPECT_EQ(t.segments.back(), (Segment{2, 3, Topology::tri, 3, Ownership::ghost}));
  EXPECT_EQ(m.segment_table(0).interior_end, 6);
}

TEST(MeshBuilder, OwnershipRejectsBadInput) {
  soma::MeshBuilder b = grid_builder();
  const std::vector<GO> one{1};
  const std::vector<Ownership> shared{Ownership::shared};
  const std::vector<std::uint32_t> zero{0};
  EXPECT_THROW(b.set_ownership(3, one, shared, zero), std::invalid_argument);
  EXPECT_THROW(b.set_ownership(-1, one, shared, zero), std::invalid_argument);
  EXPECT_THROW(b.set_ownership(0, one, std::vector<Ownership>{}, zero), std::invalid_argument);
  const auto fails = [](std::vector<GO> g, int d) {
    soma::MeshBuilder x = grid_builder();
    x.set_ownership(d, g, std::vector<Ownership>(g.size(), Ownership::shared), std::vector<std::uint32_t>(g.size(), 0));
    EXPECT_THROW((void)std::move(x).build(), std::invalid_argument);
  };
  fails({2, 2}, 0);   // named twice
  fails({7}, 0);      // no such vertex
  fails({100, 5}, 2); // no such element
  // Corners first and ownership first cannot both hold.
  soma::MeshBuilder h(2);
  h.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6},
                 std::vector<soma::Real>{0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0});
  h.add_elements(Topology::tri, std::vector<GO>{9}, std::vector<GO>{1, 2, 3, 4, 5, 6}, 6);
  h.set_ownership(0, one, shared, zero);
  EXPECT_THROW((void)std::move(h).build(), std::invalid_argument);
}

// With one key box, ranks holding different vertices order them alike. The
// subset lies on a line, so its own box would order it by x alone; the
// shared box gives the 2D curve's order, which is not.
TEST(MeshBuilder, AKeyBoxMakesCurveOrderAgreeAcrossSubsets) {
  const soma::Real lo[3] = {0, 0, 0}, hi[3] = {9, 9, 0};
  const auto box = soma::sfc::KeyBox::make(lo, hi);
  std::vector<GO> g;
  std::vector<soma::Real> xyz;
  std::vector<std::pair<std::uint64_t, GO>> keyed;
  for (GO i = 0; i < 10; ++i) {
    const soma::Real x = static_cast<soma::Real>(i), y = 4.5;
    g.push_back(i + 1);
    xyz.insert(xyz.end(), {x, y, 0});
    keyed.emplace_back(soma::sfc::hilbert_key(box, x, y, 0), i + 1);
  }
  std::sort(keyed.begin(), keyed.end());
  std::vector<GO> want;
  for (const auto& [k, id] : keyed) want.push_back(id);
  ASSERT_NE(want, g) << "the curve must not follow x along this line, or the test proves nothing";
  soma::MeshBuilder b(2);
  b.add_vertices(g, xyz);
  b.set_key_box({0, 0, 0}, {9, 9, 0});
  EXPECT_EQ(gids(std::move(b).build(soma::Ordering::type_major_sfc), 0), want);
  soma::MeshBuilder bad(2);
  EXPECT_THROW(bad.set_key_box({0, 0, 0}, {-1, 1, 1}), std::invalid_argument);
  EXPECT_THROW(bad.set_key_box({0, std::numeric_limits<soma::Real>::infinity(), 0}, {1, 1, 1}), std::invalid_argument);
}
