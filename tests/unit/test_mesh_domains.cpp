// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/errors.hpp"

using soma::DomainKind;
using soma::Layout;
using soma::LO;
using soma::NodeFamily;
using soma::SegmentFilter;
using soma::Topology;

using HostMem = Kokkos::HostSpace;
using Dom = soma::Domain<HostMem>;

TEST(MeshDomains, DomainAllIsFullWithTheMeshEpoch) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  const auto d = Dom::all(mesh, 2);
  EXPECT_EQ(d.info(0).kind, DomainKind::full);
  EXPECT_EQ(d.size(), 3);
  EXPECT_EQ(d.epoch(), mesh.epoch());
}

TEST(MeshDomains, GlobalIdsFeedTheFingerprint) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  const auto with_gids = Dom::all(mesh, 2);
  const auto lids_only = Dom::all(2, 3, mesh.epoch());
  EXPECT_FALSE(with_gids.fingerprint(0) == lids_only.fingerprint(0));
  EXPECT_EQ(with_gids.fingerprint(0).size, 3);
}

TEST(MeshDomains, SegmentsFilterByTopologyAndYieldIntervals) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  SegmentFilter quads;
  quads.topology = Topology::quad;
  const auto dq = Dom::segments(mesh, 2, quads);
  EXPECT_EQ(dq.size(), 2);
  EXPECT_EQ(dq.row(2, 0), soma::invalid_lo);
  EXPECT_EQ(dq.row(2, 1), 0);
  EXPECT_EQ(dq.row(2, 2), 1);

  SegmentFilter tris;
  tris.topology = Topology::tri;
  const auto dt = Dom::segments(mesh, 2, tris);
  EXPECT_EQ(dt.size(), 1);
  EXPECT_EQ(dt.row(2, 0), 0);
  EXPECT_FALSE(dq == dt);

  EXPECT_EQ(Dom::segments(mesh, 2, SegmentFilter{}), Dom::all(mesh, 2));
  SegmentFilter pentagons;
  pentagons.topology = Topology::polygon;
  EXPECT_EQ(Dom::segments(mesh, 2, pentagons).size(), 0);
}

TEST(MeshDomains, SegmentFilterOnOwnership) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  SegmentFilter owned;
  owned.ownership = SegmentFilter::Own::owned;
  EXPECT_EQ(Dom::segments(mesh, 2, owned), Dom::all(mesh, 2));
  SegmentFilter ghost;
  ghost.ownership = SegmentFilter::Own::ghost;
  EXPECT_EQ(Dom::segments(mesh, 2, ghost).size(), 0);
}

// Elements 0 (tri), 1 (quad) and 2 (quad) are interior, shared and ghost.
TEST(MeshDomains, OwnedSegmentsIncludeSharedAndExcludeGhosts) {
  const auto mesh = soma_test::mesh_with_shared_and_ghost_elements();
  SegmentFilter owned;
  owned.ownership = SegmentFilter::Own::owned;
  const auto d = Dom::segments(mesh, 2, owned);
  EXPECT_EQ(d.size(), 2);
  EXPECT_EQ(d.row(2, 0), 0);
  EXPECT_EQ(d.row(2, 1), 1);
  EXPECT_EQ(d.row(2, 2), soma::invalid_lo);
  EXPECT_EQ(d, Dom::from_list(mesh, 2, std::vector<LO>{0, 1}));
}

TEST(MeshDomains, InteriorSegmentsExcludeSharedAndGhosts) {
  const auto mesh = soma_test::mesh_with_shared_and_ghost_elements();
  SegmentFilter interior;
  interior.ownership = SegmentFilter::Own::interior;
  const auto d = Dom::segments(mesh, 2, interior);
  EXPECT_EQ(d.size(), 1);
  EXPECT_EQ(d.row(2, 0), 0);
  EXPECT_EQ(d.row(2, 1), soma::invalid_lo);
  EXPECT_EQ(d.row(2, 2), soma::invalid_lo);
}

TEST(MeshDomains, GhostSegmentsHoldOnlyGhosts) {
  const auto mesh = soma_test::mesh_with_shared_and_ghost_elements();
  SegmentFilter ghost;
  ghost.ownership = SegmentFilter::Own::ghost;
  const auto d = Dom::segments(mesh, 2, ghost);
  EXPECT_EQ(d.size(), 1);
  EXPECT_EQ(d.row(2, 2), 0);
  EXPECT_EQ(d.row(2, 0), soma::invalid_lo);
  EXPECT_EQ(d.row(2, 1), soma::invalid_lo);
}

TEST(MeshDomains, ArityFilterCombinesWithOwnership) {
  const auto mesh = soma_test::mesh_with_shared_and_ghost_elements();
  SegmentFilter four;
  four.arity = 4;
  EXPECT_EQ(Dom::segments(mesh, 2, four), Dom::from_list(mesh, 2, std::vector<LO>{1, 2}));
  four.ownership = SegmentFilter::Own::owned;
  EXPECT_EQ(Dom::segments(mesh, 2, four), Dom::from_list(mesh, 2, std::vector<LO>{1}));
  SegmentFilter three;
  three.arity = 3;
  EXPECT_EQ(Dom::segments(mesh, 2, three), Dom::from_list(mesh, 2, std::vector<LO>{0}));
  SegmentFilter five;
  five.arity = 5;
  EXPECT_EQ(Dom::segments(mesh, 2, five).size(), 0);
}

TEST(MeshDomains, FromListValidatesAgainstTheMesh) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  const auto d = Dom::from_list(mesh, 2, std::vector<LO>{0, 2});
  EXPECT_EQ(d.size(), 2);
  EXPECT_EQ(d.row(2, 2), 1);
  EXPECT_EQ(d.fingerprint(0), Dom::from_list(2, 3, mesh.epoch(), std::vector<LO>{0, 2},
                                             mesh.gid(2)).fingerprint(0));
  EXPECT_THROW(Dom::from_list(mesh, 2, std::vector<LO>{0, 3}), std::invalid_argument);
  EXPECT_THROW(Dom::all(mesh, 5), std::invalid_argument);
}

TEST(MeshDomains, ElementNodesSpaceUsesTheMeshTopology) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  const auto sp = soma::make_space(mesh, Dom::all(mesh, 2),
                                   Layout::element_nodes(NodeFamily::gll, 2));
  EXPECT_EQ(sp.rows(), 9 + 9 + 6);
  EXPECT_EQ(sp.entity_rows(2, 0).end, 6);
  EXPECT_EQ(sp.entity_rows(2, 2).begin, 15);
}

TEST(MeshDomains, DisjointSegmentFieldsShareAFamilyAndOverlapIsRejected) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  SegmentFilter quads;
  quads.topology = Topology::quad;
  SegmentFilter tris;
  tris.topology = Topology::tri;
  auto& reg = mesh.fields();
  reg.add<double>("T", soma::make_space(mesh, Dom::segments(mesh, 2, quads), Layout::per_entity()),
                  1);
  reg.add<double>("T", soma::make_space(mesh, Dom::segments(mesh, 2, tris), Layout::per_entity()),
                  1);
  const auto fam = reg.family<double>("T", 1);
  EXPECT_EQ(reg.index_kind(fam, 2), soma::DimIndexKind::multi_owner);
  try {
    reg.add_tag<double>("T", 2, 1);
    FAIL() << "expected FieldOverlap";
  } catch (const soma::FieldOverlap& e) {
    EXPECT_NE(std::string(e.what()).find("in dim 2 at entity gid 3"), std::string::npos)
        << e.what();
  }
}

TEST(MeshDomains, BumpedEpochReachesNewDomains) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  mesh.bump_epoch();
  const auto d = Dom::all(mesh, 0);
  EXPECT_EQ(d.epoch(), 2u);
  EXPECT_NO_THROW(
      mesh.fields().add<double>("t", soma::make_space(mesh, d, Layout::per_entity()), 1));
}
