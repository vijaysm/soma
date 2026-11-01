// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <stdexcept>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_view.hpp"

using soma::LO;
using soma::Topology;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

static_assert(!std::is_copy_constructible_v<soma::HostMesh>);
static_assert(!std::is_copy_assignable_v<soma::HostMesh>);
static_assert(std::is_nothrow_move_constructible_v<soma::HostMesh>);
static_assert(std::is_nothrow_move_assignable_v<soma::HostMesh>);

namespace soma_test {

template <class M>
double coordinate_sum_over_elements(soma::MeshView<M> v) {
  double total = 0;
  Kokkos::parallel_reduce(
      "soma::test::coordinate_sum", Kokkos::RangePolicy<typename M::execution_space>(0, v.count(2)),
      KOKKOS_LAMBDA(const LO e, double& acc) {
        for (LO k = 0; k < v.num_vertices(2, e); ++k) {
          for (int c = 0; c < 3; ++c) acc += v.x(v.vertex(2, e, k), c);
        }
      },
      total);
  return total;
}

// A view of a temporary mesh would dangle as soon as the statement ends.
template <class M>
bool view_needs_a_live_mesh() {
  using Mesh = soma::Mesh<M>;
  static_assert(requires(const Mesh& m) { m.view(); });
  static_assert(requires(Mesh& m) { m.view(); });
  static_assert(!requires(Mesh&& m) { std::move(m).view(); });
  static_assert(!requires(const Mesh&& m) { std::move(m).view(); });
  static_assert(!requires { Mesh().view(); });
  return true;
}

}  // namespace soma_test

TEST(MeshSemantics, TemporaryMeshesCannotHandOutAView) {
  EXPECT_TRUE(soma_test::view_needs_a_live_mesh<HostMem>());
  EXPECT_TRUE(soma_test::view_needs_a_live_mesh<DevMem>());
}

TEST(MeshSemantics, MoveEmptiesTheSource) {
  auto a = soma_test::two_quads_and_a_triangle();
  soma::HostMesh b = std::move(a);
  EXPECT_EQ(a.count(2), 0);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(a.coords().extent(0), 0u);
  EXPECT_EQ(b.count(2), 3);
  soma::HostMesh c;
  c = std::move(b);
  EXPECT_EQ(b.count(2), 0);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(c.count(2), 3);
}

TEST(MeshSemantics, AMovedFromMeshIsEmptyAndUsable) {
  auto a = soma_test::two_quads_and_a_triangle();
  a.fields().add_tag<double>("t", 0, 1);
  const soma::HostMesh b = std::move(a);
  EXPECT_EQ(a.view().count(0), 0);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(a.segments(2).size(), 0u);
  const auto copy = a.clone();
  EXPECT_EQ(copy.count(0), 0);
  EXPECT_EQ(copy.coords().extent(0), 0u);
  EXPECT_TRUE(copy.fields().list().empty());
  a = soma_test::two_quads_and_a_triangle();
  EXPECT_EQ(a.count(2), 3);
  EXPECT_TRUE(a.fields().list().empty());
  EXPECT_EQ(b.count(2), 3);
}

TEST(MeshSemantics, CloneIsDeep) {
  const auto b = soma_test::two_quads_and_a_triangle();
  auto c = b.clone();
  EXPECT_NE(c.coords().data(), b.coords().data());
  EXPECT_NE(c.conn(2).data(), b.conn(2).data());
  for (LO v = 0; v < b.count(0); ++v) {
    for (int k = 0; k < 3; ++k) EXPECT_DOUBLE_EQ(c.coords()(v, k), b.coords()(v, k));
  }
  c.coords()(0, 0) = 99.0;
  EXPECT_DOUBLE_EQ(b.coords()(0, 0), 0.0);
}

TEST(MeshSemantics, MirrorViewAliasesWhenTheSpaceIsTheSame) {
  const auto b = soma_test::two_quads_and_a_triangle();
  const auto alias = b.mirror_view<HostMem>();
  EXPECT_EQ(alias.coords().data(), b.coords().data());
  const auto deep = b.mirror<HostMem>();
  EXPECT_NE(deep.coords().data(), b.coords().data());
}

TEST(MeshSemantics, DroppedPartsAreNotCopied) {
  const auto b = soma_test::two_quads_and_a_triangle();
  soma::TransferOptions opt;
  opt.coordinates = false;
  opt.dims = {true, false, false, false};
  const auto m = b.mirror<HostMem>(opt);
  EXPECT_EQ(m.coords().extent(0), 0u);
  EXPECT_EQ(m.count(2), 0);
  EXPECT_EQ(m.count(0), 7);
}

TEST(MeshSemantics, ViewReflectsTheMesh) {
  const auto h = soma_test::two_quads_and_a_triangle();
  double expected = 0;
  for (LO e = 0; e < h.count(2); ++e) {
    for (soma::Offset k = h.conn_offsets(2)(e); k < h.conn_offsets(2)(e + 1); ++k) {
      for (int c = 0; c < 3; ++c) expected += h.coords()(h.conn(2)(k), c);
    }
  }
  const auto d = h.mirror<DevMem>();
  EXPECT_DOUBLE_EQ(soma_test::coordinate_sum_over_elements<DevMem>(d.view()), expected);
  const auto v = h.view();
  EXPECT_EQ(v.num_vertices(2, 2), 4);
  EXPECT_EQ(v.num_vertices(2, 0), 3);
  EXPECT_EQ(v.topology(2, 0), Topology::tri);
  EXPECT_EQ(v.topology(2, 1), Topology::quad);
  EXPECT_EQ(v.topology(0, 3), Topology::vertex);
  EXPECT_EQ(v.gid(2, 1), 3);
  EXPECT_TRUE(v.owned(2, 2));
  EXPECT_FALSE(v.owned(2, 3));
  EXPECT_TRUE(v.interior(2, 0));
}

TEST(MeshSemantics, ViewSeparatesInteriorOwnedAndGhostEntities) {
  const auto mesh = soma_test::mesh_with_shared_and_ghost_elements();
  EXPECT_EQ(mesh.segment_table(2).interior_end, 1);
  EXPECT_EQ(mesh.segment_table(2).owned_end, 2);
  const auto v = mesh.view();
  EXPECT_TRUE(v.interior(2, 0));
  EXPECT_FALSE(v.interior(2, 1));
  EXPECT_TRUE(v.owned(2, 0));
  EXPECT_TRUE(v.owned(2, 1));
  EXPECT_FALSE(v.owned(2, 2));
  EXPECT_FALSE(v.interior(2, 2));
  const auto dev = mesh.mirror<DevMem>();
  const auto dv = dev.view();
  EXPECT_EQ(dv.owned_ends[2], 2);
  EXPECT_EQ(dv.interior_ends[2], 1);
}

TEST(MeshSemantics, ViewDoesNotOwn) {
  const auto h = soma_test::two_quads_and_a_triangle();
  const auto before = h.coords().use_count();
  const auto v = h.view();
  EXPECT_EQ(h.coords().use_count(), before);
  EXPECT_EQ(v.count(2), 3);
}

TEST(MeshSemantics, ATransferThatDropsVerticesKeepsNothingThatNeedsThem) {
  const auto b = soma_test::two_quads_and_a_triangle();
  soma::TransferOptions elements;
  elements.dims = {false, false, true, false};
  elements.coordinates = false;
  EXPECT_THROW((void)b.mirror<HostMem>(elements), std::invalid_argument);
  EXPECT_THROW((void)b.mirror_view<HostMem>(elements), std::invalid_argument);
  EXPECT_THROW((void)b.mirror<DevMem>(elements), std::invalid_argument);
  soma::TransferOptions coords;
  coords.dims = {false, false, false, false};
  EXPECT_THROW((void)b.mirror<HostMem>(coords), std::invalid_argument);
  EXPECT_THROW((void)b.mirror_view<DevMem>(coords), std::invalid_argument);
  soma::TransferOptions nothing;
  nothing.dims = {false, false, false, false};
  nothing.coordinates = false;
  const auto m = b.mirror<HostMem>(nothing);
  EXPECT_EQ(m.count(0), 0);
  EXPECT_EQ(m.count(2), 0);
  EXPECT_NO_THROW((void)b.clone());
}

TEST(MeshSemantics, AccessorsRejectADimensionOutsideZeroToThree) {
  const auto m = soma_test::two_quads_and_a_triangle();
  for (const int d : {-1, 4}) {
    EXPECT_THROW((void)m.count(d), std::out_of_range) << d;
    EXPECT_THROW((void)m.segment_table(d), std::out_of_range) << d;
    EXPECT_THROW((void)m.segments(d), std::out_of_range) << d;
    EXPECT_THROW((void)m.conn_offsets(d), std::out_of_range) << d;
    EXPECT_THROW((void)m.conn(d), std::out_of_range) << d;
    EXPECT_THROW((void)m.topology(d), std::out_of_range) << d;
    EXPECT_THROW((void)m.gid(d), std::out_of_range) << d;
  }
  EXPECT_EQ(m.count(3), 0);
  EXPECT_EQ(m.gid(0).extent(0), 7u);
}
