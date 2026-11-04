// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"

using soma::GO;
using soma::LO;
using soma::Real;
using soma::Topology;

namespace {

// A quad9 beside a quad4, sharing the edge 20-50, whose mid-node 35 the
// quad4 does not have (allowed: nothing makes neighbours agree).
//   corners  10(0,0) 20(1,0) 30(2,0) 40(0,1) 50(1,1) 60(2,1); 65(3,3) on no element
//   nodes    15(0.5,0) 35(1,0.5) 45(0.5,1) 25(0,0.5) 55(0.5,0.5)
//   quad9 gid 1: 10 20 50 40 | 15 35 45 25 | 55;  quad4 gid 2: 20 30 60 50
// Node gids interleave the corner gids, so ordering by gid alone would mix them.
soma::HostMesh quad9_and_quad4(soma::Ordering ordering) {
  soma::MeshBuilder b(2);
  const std::vector<GO> gids{10, 20, 30, 40, 50, 60, 65, 15, 35, 45, 25, 55};
  const std::vector<Real> xyz{0, 0, 0,  1, 0, 0,  2, 0, 0,  0, 1, 0,  1, 1, 0,  2, 1, 0,  3, 3, 0,
                              0.5, 0, 0,  1, 0.5, 0,  0.5, 1, 0,  0, 0.5, 0,  0.5, 0.5, 0};
  b.add_vertices(gids, xyz);
  b.add_elements(Topology::quad, std::vector<GO>{1},
                 std::vector<GO>{10, 20, 50, 40, 15, 35, 45, 25, 55}, 9);
  b.add_elements(Topology::quad, std::vector<GO>{2}, std::vector<GO>{20, 30, 60, 50}, 4);
  return std::move(b).build(ordering);
}

std::vector<GO> node_gids(const soma::HostMesh& mesh, LO e) {
  const auto v = mesh.view();
  std::vector<GO> out;
  for (LO k = 0; k < v.num_nodes(2, e); ++k) out.push_back(v.gid(0, v.node(2, e, k)));
  return out;
}

}  // namespace

TEST(HighOrder, NodeOnlyVerticesComeAfterTheCorners) {
  for (const auto ordering : {soma::Ordering::file, soma::Ordering::type_major_sfc}) {
    const auto mesh = quad9_and_quad4(ordering);
    EXPECT_EQ(mesh.count(0), 12);
    EXPECT_EQ(mesh.corners(), 7);  // the vertex on no element is a corner
    std::vector<GO> corners;
    std::vector<GO> nodes;
    for (LO v = 0; v < mesh.count(0); ++v) (v < 7 ? corners : nodes).push_back(mesh.gid(0)(v));
    std::sort(corners.begin(), corners.end());
    std::sort(nodes.begin(), nodes.end());
    EXPECT_EQ(corners, (std::vector<GO>{10, 20, 30, 40, 50, 60, 65}));
    EXPECT_EQ(nodes, (std::vector<GO>{15, 25, 35, 45, 55}));
    for (LO v = 0; v < mesh.count(0); ++v) {  // coordinates moved with their vertex
      const GO g = mesh.gid(0)(v);
      if (g == 35) {
        EXPECT_EQ(mesh.coords()(v, 0), 1.0);
        EXPECT_EQ(mesh.coords()(v, 1), 0.5);
      }
      if (g == 65) EXPECT_EQ(mesh.coords()(v, 0), 3.0);
    }
  }
  // In file order each part keeps gid order.
  const auto mesh = quad9_and_quad4(soma::Ordering::file);
  const std::vector<GO> expected{10, 20, 30, 40, 50, 60, 65, 15, 25, 35, 45, 55};
  for (LO v = 0; v < 12; ++v) EXPECT_EQ(mesh.gid(0)(v), expected[static_cast<std::size_t>(v)]);
}

TEST(HighOrder, ElementsKeepTheirNodesInMoabOrder) {
  for (const auto ordering : {soma::Ordering::file, soma::Ordering::type_major_sfc}) {
    const auto mesh = quad9_and_quad4(ordering);
    ASSERT_TRUE(mesh.has_nodes(2));
    EXPECT_FALSE(mesh.has_nodes(1));
    // By node count before gid: the quad4 first.
    EXPECT_EQ(mesh.gid(2)(0), 2);
    EXPECT_EQ(mesh.gid(2)(1), 1);
    EXPECT_EQ(node_gids(mesh, 0), (std::vector<GO>{20, 30, 60, 50}));
    EXPECT_EQ(node_gids(mesh, 1), (std::vector<GO>{10, 20, 50, 40, 15, 35, 45, 25, 55}));
    EXPECT_EQ(mesh.node_offsets(2)(0), 0);
    EXPECT_EQ(mesh.node_offsets(2)(1), 0);
    EXPECT_EQ(mesh.node_offsets(2)(2), 5);
    // The connectivity is the corners alone, all below corners().
    EXPECT_EQ(mesh.view().num_vertices(2, 1), 4);
    for (std::size_t k = 0; k < mesh.conn(2).extent(0); ++k) EXPECT_LT(mesh.conn(2)(k), 7);
    // Mixed counts in one segment: segments are by corner count.
    ASSERT_EQ(mesh.segments(2).size(), 1u);
    EXPECT_EQ(mesh.segments(2)[0].end, 2);
  }
}

TEST(HighOrder, ALinearMeshHasNoNodes) {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5}, std::vector<Real>(15, 0.0));
  b.add_elements(Topology::quad, std::vector<GO>{1}, std::vector<GO>{1, 2, 3, 4}, 4);
  const auto mesh = std::move(b).build();
  EXPECT_FALSE(mesh.has_nodes(2));
  EXPECT_EQ(mesh.corners(), 5);
  EXPECT_EQ(mesh.view().num_nodes(2, 0), 4);
  EXPECT_EQ(mesh.view().num_nodes(0, 3), 1);
}

TEST(HighOrder, CountsThatAreNoConfigurationAndUnknownNodesThrow) {
  soma::MeshBuilder b(3);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8, 9}, std::vector<Real>(27, 0.0));
  EXPECT_THROW(b.add_elements(Topology::quad, std::vector<GO>{1}, std::vector<GO>(7, 1), 7),
               std::invalid_argument);
  EXPECT_THROW(b.add_elements(Topology::hex, std::vector<GO>{1}, std::vector<GO>(10, 1), 10),
               std::invalid_argument);
  EXPECT_THROW(b.add_elements(Topology::quad, std::vector<GO>{1}, std::vector<GO>(8, 1), 9),
               std::invalid_argument);
  EXPECT_THROW(b.add_elements(Topology::quad, std::vector<GO>{1}, std::vector<GO>(5, 1), 4),
               std::invalid_argument);
  b.add_elements(Topology::quad, std::vector<GO>{1}, std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 99}, 8);
  try {
    std::move(b).build();
    FAIL() << "an unknown node gid was accepted";
  } catch (const std::invalid_argument& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("99"), std::string::npos) << what;
    EXPECT_NE(what.find("quad8"), std::string::npos) << what;
  }
}

TEST(HighOrder, MirrorsAndViewsCarryTheNodes) {
  const auto mesh = quad9_and_quad4(soma::Ordering::file);
  const auto clone = mesh.clone();
  EXPECT_EQ(clone.corners(), 7);
  EXPECT_EQ(node_gids(clone, 1), node_gids(mesh, 1));
  using Device = Kokkos::DefaultExecutionSpace::memory_space;
  const auto device = mesh.mirror<Device>();
  EXPECT_EQ(device.corners(), 7);
  ASSERT_TRUE(device.has_nodes(2));
  const auto v = device.view();
  LO sum = 0;
  Kokkos::parallel_reduce(
      "test::node_sum", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, v.count(2)),
      KOKKOS_LAMBDA(const LO e, LO& s) {
        for (LO k = 0; k < v.num_nodes(2, e); ++k) s += v.node(2, e, k) * (k + 1);
      },
      sum);
  // quad4: 1, 2, 5, 4 at k = 0..3; quad9: 0, 1, 4, 3 then 7, 9, 10, 8, 11.
  EXPECT_EQ(sum, (1 + 4 + 15 + 16) + (0 + 2 + 12 + 12 + 35 + 54 + 70 + 64 + 99));
  EXPECT_EQ(v.corners, 7);
  soma::TransferOptions no_faces;
  no_faces.dims = {true, true, false, true};
  const auto partial = mesh.mirror<Kokkos::HostSpace>(no_faces);
  EXPECT_FALSE(partial.has_nodes(2));
}

// A hanging node: 4, the quad8's mid-node on the edge 2-6, is a corner of the
// two quads beside it, so it stays among the corners. Gids are contiguous.
//   1(0,0) 2(1,0) 3(2,0) 5(0,1) 6(1,1) 7(2,1) 8(2,0.5) 4(1,0.5); 9(0.5,0) 10(0.5,1) 11(0,0.5)
//   quad8 gid 1: 1 2 6 5 | 9 4 10 11;  quads gid 2: 2 3 8 4, gid 3: 4 8 7 6
TEST(HighOrder, AHangingNodeStaysACorner) {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                 std::vector<Real>{0, 0, 0,  1, 0, 0,  2, 0, 0,  1, 0.5, 0,  0, 1, 0,  1, 1, 0,
                                   2, 1, 0,  2, 0.5, 0,  0.5, 0, 0,  0.5, 1, 0,  0, 0.5, 0});
  b.add_elements(Topology::quad, std::vector<GO>{1}, std::vector<GO>{1, 2, 6, 5, 9, 4, 10, 11}, 8);
  b.add_elements(Topology::quad, std::vector<GO>{2, 3}, std::vector<GO>{2, 3, 8, 4, 4, 8, 7, 6});
  const auto mesh = std::move(b).build(soma::Ordering::file);
  EXPECT_EQ(mesh.corners(), 8);
  const std::vector<GO> expected{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  for (LO v = 0; v < 11; ++v) EXPECT_EQ(mesh.gid(0)(v), expected[static_cast<std::size_t>(v)]);
  EXPECT_EQ(mesh.gid(2)(2), 1);
  EXPECT_EQ(node_gids(mesh, 2), (std::vector<GO>{1, 2, 6, 5, 9, 4, 10, 11}));
  EXPECT_EQ(node_gids(mesh, 0), (std::vector<GO>{2, 3, 8, 4}));
}

// Node-only vertices with gids below the corners' and contiguous gids: the
// dense lookup follows the move too.
TEST(HighOrder, NodeOnlyVerticesMoveUnderContiguousGids) {
  soma::MeshBuilder b(1);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5}, std::vector<Real>{0.5, 0, 0,  1.5, 0, 0,  0, 0, 0,
                                                                   1, 0, 0,  2, 0, 0});
  b.add_elements(Topology::edge, std::vector<GO>{1, 2}, std::vector<GO>{3, 4, 1, 4, 5, 2}, 3);
  auto built = std::move(b).build(soma::Ordering::file);
  const auto mesh = std::move(built);  // moves swap, and keep corners()
  EXPECT_EQ(mesh.corners(), 3);
  const std::vector<GO> expected{3, 4, 5, 1, 2};
  for (LO v = 0; v < 5; ++v) EXPECT_EQ(mesh.gid(0)(v), expected[static_cast<std::size_t>(v)]);
  EXPECT_EQ(mesh.coords()(3, 0), 0.5);
  const auto v = mesh.view();
  EXPECT_EQ(v.node(1, 0, 2), 3);
  EXPECT_EQ(v.node(1, 1, 2), 4);
  EXPECT_EQ(v.vertex(1, 1, 0), 1);
}
