// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "adjacency_check.hpp"
#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/topology/build.hpp"
#include "soma/topology/orientation.hpp"
#include "topology_meshes.hpp"

using soma::AdjacencyRequest;
using soma::GO;
using soma::LO;
using soma::Real;
using soma::Topology;

using DevExec = Kokkos::DefaultExecutionSpace;
using DevMem = DevExec::memory_space;
using HostMem = Kokkos::HostSpace;
using HostExec = Kokkos::DefaultHostExecutionSpace;

namespace soma_test {

soma::HostMesh hex_box(int n = 3) { return box(n, n, n, Topology::hex); }

// Every graph a 3D mesh can hold.
AdjacencyRequest everything_3d() {
  AdjacencyRequest r;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (a == b) continue;
      if (b != 0) r.add(a, b);
      r.add_bridge(a, b);
    }
  }
  return r;
}

template <class M>
std::vector<HostGraph> all_graphs(const soma::Mesh<M>& mesh) {
  std::vector<HostGraph> out;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (a == b) continue;
      if (mesh.has_adjacency(a, b)) out.push_back(to_host(mesh.adjacency(a, b)));
      if (mesh.has_bridge(a, b)) out.push_back(to_host(mesh.bridge(a, b)));
    }
  }
  return out;
}

// The quads of the 2 x 2 x 2 box at z = 0, outward, with gids 100..103; the
// box gid of vertex (i, j, 0) is 1 + i + 3 j.
soma::HostMesh box_with_bottom_quads() {
  const auto g = [](int i, int j) { return GO{1 + i + 3 * j}; };
  std::vector<GO> quads;
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      quads.insert(quads.end(), {g(i, j), g(i, j + 1), g(i + 1, j + 1), g(i + 1, j)});
    }
  }
  return box_with_explicit(2, {100, 101, 102, 103}, quads);
}

}  // namespace soma_test

TEST(BuildAdjacencies, FaceToRegionAloneDerivesTheFaces) {
  auto mesh = soma_test::hex_box();
  EXPECT_EQ(mesh.count(2), 0);
  soma::build_adjacencies(mesh, AdjacencyRequest{}.add(2, 3));
  EXPECT_EQ(mesh.count(2), 108);  // 3 * 3 * 3 * 4
  EXPECT_EQ(mesh.count(1), 0);    // edges were not asked for
  EXPECT_EQ(mesh.adjacencies(), AdjacencyRequest{}.add(2, 3));
  const auto f2r = soma_test::to_host(mesh.adjacency(2, 3));
  int boundary = 0;
  for (LO f = 0; f < f2r.rows(); ++f) boundary += f2r.row(f).size() == 1;
  EXPECT_EQ(boundary, 54);
  ASSERT_EQ(mesh.segments(2).size(), 1u);
  EXPECT_EQ(mesh.segments(2)[0].topology, Topology::quad);
  EXPECT_EQ(mesh.fields().has_entities(2), true);
}

TEST(BuildAdjacencies, ASecondCallChangesNothing) {
  auto mesh = soma_test::hex_box();
  const auto req = soma_test::everything_3d();
  soma::build_adjacencies(mesh, req);
  EXPECT_EQ(mesh.adjacencies(), req);
  const auto before = soma_test::all_graphs(mesh);
  const auto faces = mesh.conn(2).data();
  const auto epoch = mesh.epoch();
  soma::build_adjacencies(mesh, req);
  EXPECT_EQ(soma_test::all_graphs(mesh), before);
  EXPECT_EQ(mesh.conn(2).data(), faces);
  EXPECT_EQ(mesh.epoch(), epoch);
  soma::build_adjacencies(mesh, AdjacencyRequest{}.add(3, 2));
  EXPECT_EQ(mesh.adjacencies(), req);
}

TEST(BuildAdjacencies, LaterRequestsReuseDerivedEntities) {
  auto mesh = soma_test::hex_box();
  soma::build_adjacencies(mesh, AdjacencyRequest{}.add(2, 3));
  const auto f2r = soma_test::to_host(mesh.adjacency(2, 3));
  const auto faces = mesh.conn(2).data();
  soma::build_adjacencies(mesh, AdjacencyRequest{}.add(0, 2));
  EXPECT_EQ(mesh.conn(2).data(), faces);
  EXPECT_EQ(mesh.adjacencies(), AdjacencyRequest{}.add(2, 3).add(0, 2));
  EXPECT_EQ(soma_test::to_host(mesh.adjacency(2, 3)), f2r);
}

TEST(BuildAdjacencies, UpwardCodesAreTheSideIndex) {
  auto mesh = soma_test::hex_box();
  soma::build_adjacencies(mesh, AdjacencyRequest{}.add(1, 3).add(3, 1).add(0, 3));
  const auto down = soma_test::to_host(mesh.adjacency(3, 1));
  const auto up = soma_test::to_host(mesh.adjacency(1, 3));
  for (LO e = 0; e < up.rows(); ++e) {
    const auto row = up.row(e);
    const auto codes = up.row_codes(e);
    for (std::size_t q = 0; q < row.size(); ++q) {
      EXPECT_EQ(down.row(row[q])[codes[q]], e);
    }
  }
  const auto v2r = soma_test::to_host(mesh.adjacency(0, 3));
  const auto conn = soma_test::to_vector(mesh.conn(3));
  for (LO v = 0; v < v2r.rows(); ++v) {
    const auto row = v2r.row(v);
    const auto codes = v2r.row_codes(v);
    for (std::size_t q = 0; q < row.size(); ++q) {
      EXPECT_EQ(conn[static_cast<std::size_t>(8 * row[q] + codes[q])], v);
    }
  }
}

TEST(BuildAdjacencies, FaceToEdgeIsConsistentWithRegionToEdge) {
  auto mesh = soma_test::hex_box();
  soma::build_adjacencies(mesh, AdjacencyRequest{}.add(2, 1).add(3, 2).add(3, 1));
  const auto f2e = soma_test::to_host(mesh.adjacency(2, 1));
  const auto r2f = soma_test::to_host(mesh.adjacency(3, 2));
  const auto r2e = soma_test::to_host(mesh.adjacency(3, 1));
  ASSERT_EQ(f2e.rows(), mesh.count(2));
  for (LO r = 0; r < r2f.rows(); ++r) {
    std::vector<LO> from_faces;
    for (const LO f : r2f.row(r)) {
      for (const LO e : f2e.row(f)) from_faces.push_back(e);
    }
    std::vector<LO> edges = r2e.row(r);
    std::sort(from_faces.begin(), from_faces.end());
    from_faces.erase(std::unique(from_faces.begin(), from_faces.end()), from_faces.end());
    std::sort(edges.begin(), edges.end());
    EXPECT_EQ(from_faces, edges) << "region " << r;
  }
}

TEST(BuildAdjacencies, IsBitwiseEqualAcrossExecutionSpaces) {
  const auto host = soma_test::hex_box(4);
  const auto req = soma_test::everything_3d();
  auto a = host.clone();
  soma::build_adjacencies(a, req, HostExec{});
  auto dev = host.mirror<DevMem>();
  soma::build_adjacencies(dev, req, DevExec{});
  EXPECT_EQ(soma_test::all_graphs(dev), soma_test::all_graphs(a));
#ifdef KOKKOS_ENABLE_SERIAL
  auto s = host.clone();
  soma::build_adjacencies(s, req, Kokkos::Serial{});
  EXPECT_EQ(soma_test::all_graphs(s), soma_test::all_graphs(a));
#endif
}

TEST(BuildAdjacencies, RejectsDimensionsAboveTheMesh) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  EXPECT_THROW(soma::build_adjacencies(mesh, AdjacencyRequest{}.add(3, 2)), std::invalid_argument);
  EXPECT_THROW(soma::build_adjacencies(mesh, AdjacencyRequest{}.add_bridge(0, 3)),
               std::invalid_argument);
  EXPECT_TRUE(mesh.adjacencies().empty());
}

TEST(BuildAdjacencies, FillingAnEmptyDimensionKeepsFieldsCurrent) {
  auto mesh = soma_test::hex_box();
  auto t = mesh.fields().add_tag<double>("t", 0, 1);
  const auto epoch = mesh.epoch();
  soma::build_adjacencies(mesh, AdjacencyRequest{}.add(1, 0).add(0, 1));
  EXPECT_EQ(mesh.epoch(), epoch);
  EXPECT_NO_THROW((void)mesh.fields().get<double>("t", 0));
  // a tag on the new dimension sees its entities
  auto e = mesh.fields().add_tag<double>("on_edges", 1, 1);
  EXPECT_EQ(e.view().extent(0), static_cast<std::size_t>(mesh.count(1)));
}

TEST(BuildAdjacencies, AdoptingExplicitFacesBumpsTheEpoch) {
  auto mesh = soma_test::box_with_bottom_quads();
  (void)mesh.fields().add_tag<double>("t", 0, 1);
  const auto epoch = mesh.epoch();
  soma::build_adjacencies(mesh, AdjacencyRequest{}.add(3, 2));
  EXPECT_EQ(mesh.epoch(), epoch + 1);
  EXPECT_EQ(mesh.count(2), 36);
  EXPECT_THROW((void)mesh.fields().get<double>("t", 0), soma::StaleField);
}

namespace soma_test {

std::vector<GO> gids_of(const soma::HostMesh& m, int dim, std::vector<LO> lids) {
  std::vector<GO> out;
  for (const LO l : lids) out.push_back(m.gid(dim)(static_cast<std::size_t>(l)));
  return out;
}

// Set gid 1 holds quads 0 and 2, vertex 0 and region 0; set gid 2, ordered,
// holds quads 3, 1, 3.
void add_quad_sets(soma::HostMesh& mesh) {
  soma::SetBuilder b;
  b.add(soma::SetSpec{1, soma::set_unique, {{2, 2}, {0, 0}, {2, 0}, {3, 0}}, {}, {}});
  b.add(soma::SetSpec{2, soma::set_ordered, {{2, 3}, {2, 1}, {2, 3}}, {}, {}});
  std::move(b).install(mesh);
}

}  // namespace soma_test

// Adopting explicit quads renumbers them; the sets keep the same quads. The
// gids run against the derived order, so renumbered rows must be sorted again.
TEST(BuildAdjacencies, SetsFollowRenumberedEntities) {
  const auto g = [](int i, int j) { return GO{1 + i + 3 * j}; };
  std::vector<GO> quads;
  for (int j = 1; j >= 0; --j) {
    for (int i = 1; i >= 0; --i) {
      quads.insert(quads.end(), {g(i, j), g(i, j + 1), g(i + 1, j + 1), g(i + 1, j)});
    }
  }
  auto mesh = soma_test::box_with_explicit(2, {100, 101, 102, 103}, quads);
  soma_test::add_quad_sets(mesh);
  (void)mesh.fields().add_tag<int>("on_sets", soma::set_dim, 1);
  const auto before = soma_test::gids_of(mesh, 2, {0, 1, 2, 3});
  ASSERT_EQ(before, (std::vector<GO>{100, 101, 102, 103}));
  soma::build_adjacencies(mesh, AdjacencyRequest{}.add(3, 2));
  ASSERT_EQ(mesh.count(2), 36);
  const auto& s = mesh.sets();
  const auto faces = s.members_of(0, 2);
  const std::vector<LO> face_lids(faces.begin(), faces.end());
  EXPECT_NE(face_lids, (std::vector<LO>{0, 2}));  // the quads did move
  EXPECT_TRUE(std::is_sorted(face_lids.begin(), face_lids.end()));
  auto got = soma_test::gids_of(mesh, 2, face_lids);
  std::sort(got.begin(), got.end());
  EXPECT_EQ(got, (std::vector<GO>{100, 102}));
  std::vector<GO> seq;
  for (const soma::SetEntry& e : s.sequence(1)) {
    EXPECT_EQ(e.slot, 2);
    seq.push_back(mesh.gid(2)(static_cast<std::size_t>(e.lid)));
  }
  EXPECT_EQ(seq, (std::vector<GO>{103, 101, 103}));
  EXPECT_EQ(std::vector<LO>(s.members_of(0, 0).begin(), s.members_of(0, 0).end()), std::vector<LO>{0});
  EXPECT_EQ(std::vector<LO>(s.members_of(0, 3).begin(), s.members_of(0, 3).end()), std::vector<LO>{0});
  EXPECT_THROW((void)mesh.fields().get<int>("on_sets", soma::set_dim), soma::StaleField);
}

TEST(BuildAdjacencies, SetsOfAFailedBuildAreUnchanged) {
  auto bad = soma_test::box_with_explicit(2, {100, 101}, {1, 4, 5, 2, 2, 5, 4, 1});
  soma::SetBuilder b;
  b.add(soma::SetSpec{1, soma::set_unique, {{2, 1}}, {}, {}});
  std::move(b).install(bad);
  EXPECT_THROW(soma::build_adjacencies(bad, AdjacencyRequest{}.add(3, 2)), std::invalid_argument);
  EXPECT_EQ(std::vector<LO>(bad.sets().members_of(0, 2).begin(), bad.sets().members_of(0, 2).end()),
            std::vector<LO>{1});
}

TEST(BuildAdjacencies, AFailedBuildLeavesTheMeshUnchanged) {
  // (2, 5, 4, 1) repeats the vertices of (1, 4, 5, 2)
  auto bad = soma_test::box_with_explicit(2, {100, 101}, {1, 4, 5, 2, 2, 5, 4, 1});
  const auto conn2 = soma_test::to_vector(bad.conn(2));
  EXPECT_THROW(soma::build_adjacencies(bad, AdjacencyRequest{}.add(3, 2).add(0, 3)),
               std::invalid_argument);
  EXPECT_EQ(bad.count(2), 2);
  EXPECT_EQ(soma_test::to_vector(bad.conn(2)), conn2);
  EXPECT_TRUE(bad.adjacencies().empty());
  EXPECT_EQ(bad.epoch(), 1u);
}

TEST(BuildAdjacencies, MirrorRebuildsOrCopiesTheGraphs) {
  auto host = soma_test::hex_box();
  const auto req = AdjacencyRequest{}.add(2, 3).add_bridge(0, 1);
  soma::build_adjacencies(host, req);

  soma::TransferOptions rebuild;
  rebuild.adjacencies = req;
  const auto a = host.mirror<DevMem>(rebuild);
  EXPECT_EQ(soma_test::all_graphs(a), soma_test::all_graphs(host));

  // without the derived dimensions the destination derives them again
  soma::TransferOptions lean = rebuild;
  lean.dims = {true, false, false, true};
  lean.adjacencies = AdjacencyRequest{}.add(0, 3);
  const auto b = host.mirror<DevMem>(lean);
  EXPECT_EQ(b.count(2), 0);
  auto expected = host.clone();
  soma::build_adjacencies(expected, AdjacencyRequest{}.add(0, 3));
  EXPECT_EQ(soma_test::to_host(b.adjacency(0, 3)), soma_test::to_host(expected.adjacency(0, 3)));
  soma::TransferOptions both = rebuild;
  both.dims = {true, false, false, true};
  both.adjacencies = AdjacencyRequest{};
  auto c = host.mirror<DevMem>(both);
  soma::build_adjacencies(c, req);
  EXPECT_EQ(soma_test::all_graphs(c), soma_test::all_graphs(host));
  EXPECT_EQ(soma_test::to_vector(c.conn(2)), soma_test::to_vector(host.conn(2)));

  soma::TransferOptions copy = rebuild;
  copy.adj_source = soma::TransferOptions::AdjSource::copy;
  const auto d = host.mirror<DevMem>(copy);
  EXPECT_EQ(soma_test::all_graphs(d), soma_test::all_graphs(host));
}

TEST(BuildAdjacencies, DerivingWithoutElementsIsRejected) {
  auto host = soma_test::hex_box();
  soma::build_adjacencies(host, AdjacencyRequest{}.add(2, 3));
  soma::TransferOptions o;
  o.dims = {true, true, true, false};
  o.adjacencies = AdjacencyRequest{}.add(0, 1);
  EXPECT_THROW((void)host.mirror<DevMem>(o), std::invalid_argument);
  o.adjacencies = AdjacencyRequest{}.add(0, 2);  // faces are there: nothing to derive
  const auto m = host.mirror<DevMem>(o);
  EXPECT_EQ(m.count(2), host.count(2));
  EXPECT_TRUE(m.has_adjacency(0, 2));
}

namespace soma_test {
struct NoExecutionSpace {};
}  // namespace soma_test

// Mesh::mirror<Dst> instantiates the rebuild path for every Dst, so a space
// without a default execution space must still compile (as CudaUVMSpace).
static_assert(soma::topology::detail::DefaultExec<HostMem>::available);
static_assert(soma::topology::detail::DefaultExec<DevMem>::available);
static_assert(!soma::topology::detail::DefaultExec<soma_test::NoExecutionSpace>::available);
