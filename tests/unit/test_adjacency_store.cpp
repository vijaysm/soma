// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "adjacency_check.hpp"
#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_view.hpp"
#include "soma/topology/adjacency.hpp"

using soma::AdjacencyRequest;
using soma::LO;
using soma::Offset;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace soma_test {

// A made-up graph on the fixture mesh (3 elements): element e lists vertices
// e and e + 1, with codes 10 + e.
soma::Adjacency<HostMem> made_up_graph() {
  soma::Adjacency<HostMem> g;
  g.offsets = soma::View1<Offset, HostMem>("offsets", 4);
  g.targets = soma::View1<LO, HostMem>("targets", 6);
  g.codes = soma::View1<std::uint8_t, HostMem>("codes", 6);
  for (LO e = 0; e < 3; ++e) {
    g.offsets(e + 1) = static_cast<Offset>(2 * (e + 1));
    g.targets(2 * e) = e;
    g.targets(2 * e + 1) = e + 1;
    g.codes(2 * e) = static_cast<std::uint8_t>(10 + e);
    g.codes(2 * e + 1) = static_cast<std::uint8_t>(10 + e);
  }
  return g;
}

soma::HostMesh mesh_with_two_graphs() {
  soma::HostMesh mesh = two_quads_and_a_triangle();
  soma::MeshTestAccess::put_adjacency(mesh, soma::pair_slot(2, 1), made_up_graph());
  soma::MeshTestAccess::put_adjacency(mesh, soma::bridge_slot(2, 0), made_up_graph());
  return mesh;
}

// Sum over the rows of every element of target * 100 + code, read in a kernel.
template <class M>
long long weighted_row_sum(soma::MeshView<M> v) {
  long long total = 0;
  Kokkos::parallel_reduce(
      "soma::test::weighted_row_sum",
      Kokkos::RangePolicy<typename M::execution_space>(0, v.count(2)),
      KOKKOS_LAMBDA(const LO e, long long& acc) {
        const soma::Row<const LO> r = v.adj(2, 1, e);
        const soma::Row<const std::uint8_t> c = v.adj_codes(2, 1, e);
        for (LO i = 0; i < r.size; ++i) acc += 100LL * r[i] + c[i];
        const soma::Row<const LO> b = v.bridge(2, 0, e);
        for (LO i = 0; i < b.size; ++i) acc += 7LL * b[i];
        const soma::Row<const LO> verts = v.adj(2, 0, e);
        for (LO i = 0; i < verts.size; ++i) acc += 1000000LL * verts[i];
      },
      total);
  return total;
}

}  // namespace soma_test

TEST(AdjacencyRequest, AddAndHasRoundTripThroughBits) {
  AdjacencyRequest r;
  EXPECT_TRUE(r.empty());
  r.add(3, 2).add(0, 3).add_bridge(0, 1).add_bridge(3, 2);
  EXPECT_TRUE(r.has(3, 2));
  EXPECT_TRUE(r.has(0, 3));
  EXPECT_FALSE(r.has(2, 3));
  EXPECT_TRUE(r.has_bridge(0, 1));
  EXPECT_TRUE(r.has_bridge(3, 2));
  EXPECT_FALSE(r.has_bridge(1, 0));
  EXPECT_FALSE(r.has(3, 3));
  EXPECT_EQ(AdjacencyRequest::from_bits(r.bits()), r);
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (a == b) continue;
      const AdjacencyRequest one = AdjacencyRequest{}.add(a, b);
      const AdjacencyRequest via = AdjacencyRequest{}.add_bridge(a, b);
      EXPECT_EQ(one.bits(), std::uint32_t{1} << soma::pair_slot(a, b));
      EXPECT_EQ(via.bits(), std::uint32_t{1} << soma::bridge_slot(a, b));
      EXPECT_EQ(AdjacencyRequest::from_bits(one.bits() | via.bits()), one | via);
    }
  }
}

TEST(AdjacencyRequest, RejectsDiagonalsAndDimensionsOutOfRange) {
  AdjacencyRequest r;
  EXPECT_THROW(r.add(2, 2), std::invalid_argument);
  EXPECT_THROW(r.add(-1, 0), std::invalid_argument);
  EXPECT_THROW(r.add(0, 4), std::invalid_argument);
  EXPECT_THROW(r.add_bridge(1, 1), std::invalid_argument);
  EXPECT_THROW(r.add_bridge(4, 0), std::invalid_argument);
  EXPECT_TRUE(r.empty());
  for (int d = 0; d < 4; ++d) {
    EXPECT_THROW(AdjacencyRequest::from_bits(std::uint32_t{1} << soma::pair_slot(d, d)),
                 std::invalid_argument);
    EXPECT_THROW(AdjacencyRequest::from_bits(std::uint32_t{1} << soma::bridge_slot(d, d)),
                 std::invalid_argument);
  }
}

TEST(MeshAdjacency, UnheldGraphsThrowAndConnectivityIsAlwaysThere) {
  const soma::HostMesh mesh = soma_test::two_quads_and_a_triangle();
  EXPECT_TRUE(mesh.adjacencies().empty());
  EXPECT_TRUE(mesh.has_adjacency(2, 0));
  EXPECT_TRUE(mesh.has_adjacency(1, 0));
  EXPECT_FALSE(mesh.has_adjacency(3, 0));
  EXPECT_FALSE(mesh.has_adjacency(2, 1));
  EXPECT_FALSE(mesh.has_bridge(2, 0));
  EXPECT_THROW((void)mesh.adjacency(2, 1), soma::MissingAdjacency);
  EXPECT_THROW((void)mesh.adjacency(0, 2), soma::MissingAdjacency);
  EXPECT_THROW((void)mesh.adjacency(3, 0), soma::MissingAdjacency);
  EXPECT_THROW((void)mesh.bridge(2, 0), soma::MissingAdjacency);
  EXPECT_THROW((void)mesh.has_adjacency(2, 2), std::invalid_argument);
  EXPECT_THROW((void)mesh.has_bridge(0, 4), std::invalid_argument);

  const auto conn = soma_test::to_host(mesh.adjacency(2, 0));
  EXPECT_EQ(conn.targets, soma_test::to_vector(mesh.conn(2)));
  EXPECT_EQ(conn.offsets, soma_test::to_vector(mesh.conn_offsets(2)));
  EXPECT_TRUE(conn.codes.empty());
}

TEST(MeshAdjacency, HeldGraphsAreVisibleOnHostAndInKernels) {
  const soma::HostMesh mesh = soma_test::mesh_with_two_graphs();
  EXPECT_EQ(mesh.adjacencies(),
            AdjacencyRequest{}.add(2, 1).add_bridge(2, 0));
  EXPECT_EQ(soma_test::to_host(mesh.adjacency(2, 1)), soma_test::to_host(soma_test::made_up_graph()));
  // rows: targets {0,1},{1,2},{2,3}; codes 10, 11, 12
  const long long expected_rows = 100LL * (0 + 1 + 1 + 2 + 2 + 3) + 2 * (10 + 11 + 12);
  const long long expected_bridge = 7LL * (0 + 1 + 1 + 2 + 2 + 3);
  long long expected_conn = 0;
  for (const LO v : soma_test::to_vector(mesh.conn(2))) expected_conn += 1000000LL * v;
  EXPECT_EQ(soma_test::weighted_row_sum(mesh.view()),
            expected_rows + expected_bridge + expected_conn);
}

TEST(MeshAdjacency, MirrorCopiesOnlyTheRequestedGraphs) {
  const soma::HostMesh mesh = soma_test::mesh_with_two_graphs();
  soma::TransferOptions o;
  o.adj_source = soma::TransferOptions::AdjSource::copy;
  o.adjacencies = AdjacencyRequest{}.add(2, 1);
  const auto dev = mesh.mirror<DevMem>(o);
  EXPECT_EQ(dev.adjacencies(), AdjacencyRequest{}.add(2, 1));
  EXPECT_EQ(soma_test::to_host(dev.adjacency(2, 1)), soma_test::to_host(mesh.adjacency(2, 1)));
  EXPECT_THROW((void)dev.bridge(2, 0), soma::MissingAdjacency);

  o.adjacencies.add(2, 0);  // the connectivity: always there, never stored
  EXPECT_EQ(mesh.mirror<DevMem>(o).adjacencies(), AdjacencyRequest{}.add(2, 1));

  const auto lean = mesh.mirror<DevMem>();
  EXPECT_TRUE(lean.adjacencies().empty());

  const auto copy = mesh.clone();
  EXPECT_EQ(copy.adjacencies(), mesh.adjacencies());
  EXPECT_NE(copy.adjacency(2, 1).targets.data(), mesh.adjacency(2, 1).targets.data());
  EXPECT_EQ(soma_test::to_host(copy.bridge(2, 0)), soma_test::to_host(mesh.bridge(2, 0)));

  const auto alias = mesh.mirror_view<HostMem>(o);
  EXPECT_EQ(alias.adjacency(2, 1).targets.data(), mesh.adjacency(2, 1).targets.data());
}

TEST(MeshAdjacency, MirrorRejectsMissingGraphsAndDroppedDimensions) {
  const soma::HostMesh mesh = soma_test::mesh_with_two_graphs();
  soma::TransferOptions o;
  o.adj_source = soma::TransferOptions::AdjSource::copy;
  o.adjacencies = AdjacencyRequest{}.add(0, 2);
  EXPECT_THROW((void)mesh.mirror<DevMem>(o), soma::MissingAdjacency);
  o.adjacencies = AdjacencyRequest{}.add(2, 1);
  o.dims = {true, false, true, true};
  EXPECT_THROW((void)mesh.mirror<DevMem>(o), std::invalid_argument);
  o.adjacencies = AdjacencyRequest{}.add_bridge(2, 0);
  o.dims = {true, true, false, true};
  EXPECT_THROW((void)mesh.mirror<DevMem>(o), std::invalid_argument);
}

TEST(MeshAdjacency, MovesAndSwapsCarryTheGraphs) {
  soma::HostMesh a = soma_test::mesh_with_two_graphs();
  const auto before = soma_test::to_host(a.adjacency(2, 1));
  soma::HostMesh b = std::move(a);
  EXPECT_TRUE(a.adjacencies().empty());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(soma_test::to_host(b.adjacency(2, 1)), before);
  soma::HostMesh c;
  c.swap(b);
  EXPECT_TRUE(b.adjacencies().empty());
  EXPECT_EQ(soma_test::to_host(c.adjacency(2, 1)), before);
}
