// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/space.hpp"

using soma::Layout;
using soma::LO;
using soma::NodeFamily;
using soma::Topology;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace soma_test {

template <class M>
soma::View1<const std::uint8_t, M> topology_view(const std::vector<Topology>& t) {
  soma::View1<std::uint8_t, Kokkos::HostSpace> h("soma::test::topo", t.size());
  for (std::size_t i = 0; i < t.size(); ++i) h(i) = static_cast<std::uint8_t>(t[i]);
  return Kokkos::create_mirror_view_and_copy(M{}, h);
}

template <class M>
soma::View1<const soma::Offset, M> offsets_view(const std::vector<soma::Offset>& o) {
  soma::View1<soma::Offset, Kokkos::HostSpace> h("soma::test::offsets", o.size());
  for (std::size_t i = 0; i < o.size(); ++i) h(i) = o[i];
  return Kokkos::create_mirror_view_and_copy(M{}, h);
}

template <class M>
soma::ElementInfo<M> cells(int dim, const std::vector<Topology>& t) {
  soma::ElementInfo<M> info;
  info.topology[static_cast<std::size_t>(dim)] = topology_view<M>(t);
  return info;
}

// Row range of each entity and the entity of each row, evaluated on the device.
template <class M>
Kokkos::View<LO*, Kokkos::HostSpace> rows_of_entities(soma::SpaceView<M> sv, int dim, LO count) {
  Kokkos::View<LO*, M> out("soma::test::first_rows", 2 * count);
  Kokkos::parallel_for(
      "soma::test::first_rows", Kokkos::RangePolicy<typename M::execution_space>(0, count),
      KOKKOS_LAMBDA(const LO l) {
        const soma::RowRange r = sv.entity_rows(dim, l);
        out(2 * l) = r.begin;
        out(2 * l + 1) = r.end;
      });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
}

template <class M>
Kokkos::View<LO*, Kokkos::HostSpace> entities_of_rows(soma::SpaceView<M> sv, LO rows) {
  Kokkos::View<LO*, M> out("soma::test::row_entities", 2 * rows);
  Kokkos::parallel_for(
      "soma::test::row_entities", Kokkos::RangePolicy<typename M::execution_space>(0, rows),
      KOKKOS_LAMBDA(const LO r) {
        const auto loc = sv.locate(r);
        out(2 * r) = loc.lid;
        out(2 * r + 1) = loc.node;
      });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
}

}  // namespace soma_test

template <class M>
class SpaceTyped : public ::testing::Test {};
using Spaces = ::testing::Types<HostMem, DevMem>;
TYPED_TEST_SUITE(SpaceTyped, Spaces);

TYPED_TEST(SpaceTyped, PerEntityRowsEqualDomainSize) {
  using D = soma::Domain<TypeParam>;
  const soma::Space<TypeParam> all(D::all(3, 10, 1), Layout::per_entity());
  EXPECT_EQ(all.rows(), 10);
  EXPECT_EQ(all.entity_rows(3, 4).begin, 4);
  EXPECT_EQ(all.entity_rows(3, 4).end, 5);

  const soma::Space<TypeParam> some(D::from_list(2, 40, 1, std::vector<LO>{3, 9, 20}),
                                    Layout::per_entity());
  EXPECT_EQ(some.rows(), 3);
  EXPECT_EQ(some.entity_rows(2, 9).begin, 1);
  EXPECT_FALSE(some.entity_rows(2, 10).valid());
  EXPECT_FALSE(some.entity_rows(3, 9).valid());
}

// Value points need element topology: vertices and entity sets have none.
TYPED_TEST(SpaceTyped, ElementNodesNeedElements) {
  using D = soma::Domain<TypeParam>;
  const auto info = soma_test::cells<TypeParam>(3, std::vector<Topology>(4, Topology::hex));
  const auto nodes = Layout::element_nodes(NodeFamily::gll, 2);
  for (const int slot : {0, soma::set_dim}) {
    try {
      soma::Space<TypeParam>(D::all(slot, 4, 1), nodes, info);
      ADD_FAILURE() << "slot " << slot << " accepted";
    } catch (const std::invalid_argument& e) {
      const std::string what = e.what();
      EXPECT_NE(what.find(slot == 0 ? "dimension 0 has no topology" : "entity sets have no topology"),
                std::string::npos) << what;
    }
  }
  const soma::Space<TypeParam> per(D::all(soma::set_dim, 4, 1), Layout::per_entity(), info);
  EXPECT_EQ(per.rows(), 4);
  EXPECT_EQ(per.entity_rows(soma::set_dim, 3).begin, 3);
}

TYPED_TEST(SpaceTyped, UniformBlocksUseAConstantStride) {
  using D = soma::Domain<TypeParam>;
  const std::vector<Topology> topo(6, Topology::hex);
  const auto info = soma_test::cells<TypeParam>(3, topo);
  const soma::Space<TypeParam> sp(D::all(3, 6, 1),
                                  Layout::element_nodes(NodeFamily::gll, 2), info);
  EXPECT_EQ(sp.rows(), 6 * 27);
  EXPECT_EQ(sp.block(0).stride, 27);
  EXPECT_EQ(sp.entity_rows(3, 2).begin, 54);
  EXPECT_EQ(sp.entity_rows(3, 2).end, 81);
}

TYPED_TEST(SpaceTyped, MixedBlocksUseAPrefixArray) {
  using D = soma::Domain<TypeParam>;
  const std::vector<Topology> topo{Topology::quad, Topology::quad, Topology::tri, Topology::tri,
                                   Topology::quad};
  const auto info = soma_test::cells<TypeParam>(2, topo);
  const soma::Space<TypeParam> sp(D::all(2, 5, 1),
                                  Layout::element_nodes(NodeFamily::gll, 2), info);
  EXPECT_EQ(sp.rows(), 9 + 9 + 6 + 6 + 9);
  EXPECT_EQ(sp.block(0).stride, 0);
  const std::vector<LO> begin{0, 9, 18, 24, 30};
  for (LO l = 0; l < 5; ++l) {
    EXPECT_EQ(sp.entity_rows(2, l).begin, begin[static_cast<std::size_t>(l)]) << l;
  }
  EXPECT_EQ(sp.entity_rows(2, 4).end, 39);
}

TYPED_TEST(SpaceTyped, SubsetOfAMixedBlockCountsOnlyItsMembers) {
  using D = soma::Domain<TypeParam>;
  const std::vector<Topology> topo{Topology::quad, Topology::tri, Topology::tri, Topology::quad};
  const auto info = soma_test::cells<TypeParam>(2, topo);
  const soma::Space<TypeParam> sp(D::from_list(2, 4, 1, std::vector<LO>{1, 2, 3}),
                                  Layout::element_nodes(NodeFamily::gll, 2), info);
  EXPECT_EQ(sp.rows(), 6 + 6 + 9);
  EXPECT_EQ(sp.entity_rows(2, 3).begin, 12);
  EXPECT_FALSE(sp.entity_rows(2, 0).valid());
}

TYPED_TEST(SpaceTyped, SubsetWithOneTopologyIsUniform) {
  using D = soma::Domain<TypeParam>;
  const std::vector<Topology> topo{Topology::quad, Topology::tri, Topology::tri, Topology::quad};
  const auto info = soma_test::cells<TypeParam>(2, topo);
  const soma::Space<TypeParam> sp(D::from_list(2, 4, 1, std::vector<LO>{0, 3}),
                                  Layout::element_nodes(NodeFamily::gll, 3), info);
  EXPECT_EQ(sp.block(0).stride, 16);
  EXPECT_EQ(sp.rows(), 32);
}

TYPED_TEST(SpaceTyped, PolygonsTakeTheirArityFromTheOffsets) {
  using D = soma::Domain<TypeParam>;
  soma::ElementInfo<TypeParam> info;
  info.topology[2] = soma_test::topology_view<TypeParam>(
      {Topology::polygon, Topology::polygon, Topology::polygon});
  info.conn_offsets[2] = soma_test::offsets_view<TypeParam>({0, 5, 11, 16});
  const soma::Space<TypeParam> sp(
      D::all(2, 3, 1), Layout::element_nodes(NodeFamily::lagrange_equispaced, 1), info);
  EXPECT_EQ(sp.rows(), 5 + 6 + 5);
  EXPECT_EQ(sp.entity_rows(2, 1).begin, 5);
  EXPECT_EQ(sp.entity_rows(2, 1).end, 11);
}

TYPED_TEST(SpaceTyped, MultiBlockDomainStacksRowRanges) {
  using D = soma::Domain<TypeParam>;
  soma::ElementInfo<TypeParam> info;
  info.topology[2] = soma_test::topology_view<TypeParam>(std::vector<Topology>(4, Topology::quad));
  info.topology[3] = soma_test::topology_view<TypeParam>(std::vector<Topology>(2, Topology::hex));
  const auto dom = D::combine(D::all(2, 4, 1), D::all(3, 2, 1));
  const soma::Space<TypeParam> sp(dom, Layout::element_nodes(NodeFamily::gll, 1), info);
  EXPECT_EQ(sp.rows(), 4 * 4 + 2 * 8);
  EXPECT_EQ(sp.block_row_begin(1), 16);
  EXPECT_EQ(sp.entity_rows(3, 1).begin, 16 + 8);
  EXPECT_EQ(sp.entity_rows(2, 3).begin, 12);
}

TYPED_TEST(SpaceTyped, DeviceViewMatchesTheHostInBothDirections) {
  using D = soma::Domain<TypeParam>;
  const std::vector<Topology> topo{Topology::quad, Topology::tri, Topology::tri, Topology::quad,
                                   Topology::quad, Topology::tri};
  const auto info = soma_test::cells<TypeParam>(2, topo);
  const soma::Space<TypeParam> sp(D::from_list(2, 6, 1, std::vector<LO>{1, 2, 3, 5}),
                                  Layout::element_nodes(NodeFamily::gll, 2), info);
  const auto ranges = soma_test::rows_of_entities<TypeParam>(sp.view(), 2, 6);
  for (LO l = 0; l < 6; ++l) {
    const soma::RowRange h = sp.entity_rows(2, l);
    EXPECT_EQ(ranges(2 * l), h.begin) << l;
    EXPECT_EQ(ranges(2 * l + 1), h.end) << l;
  }
  const auto owners = soma_test::entities_of_rows<TypeParam>(sp.view(), sp.rows());
  for (LO r = 0; r < sp.rows(); ++r) {
    const LO lid = owners(2 * r);
    const LO node = owners(2 * r + 1);
    ASSERT_NE(lid, soma::invalid_lo) << "row " << r;
    EXPECT_EQ(sp.entity_rows(2, lid).begin + node, r) << "row " << r;
  }
}

TYPED_TEST(SpaceTyped, UnsupportedCombinationsThrowWithTheirName) {
  using D = soma::Domain<TypeParam>;
  const auto info = soma_test::cells<TypeParam>(3, {Topology::pyramid, Topology::pyramid});
  try {
    const soma::Space<TypeParam> sp(D::all(3, 2, 1), Layout::element_nodes(NodeFamily::gll, 3),
                                    info);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("pyramid"), std::string::npos) << e.what();
  }
  EXPECT_THROW(soma::Space<TypeParam>(D::all(3, 2, 1),
                                      Layout::element_nodes(NodeFamily::gll, 2)),
               std::invalid_argument);
  EXPECT_THROW(soma::Space<TypeParam>(D::all(0, 2, 1),
                                      Layout::element_nodes(NodeFamily::gll, 2)),
               std::invalid_argument);
}

TYPED_TEST(SpaceTyped, DeferredLayoutKindsAreRejected) {
  using D = soma::Domain<TypeParam>;
  try {
    const soma::Space<TypeParam> sp(D::all(3, 2, 1), Layout::shared_nodes(NodeFamily::gll, 2));
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("is not supported yet"), std::string::npos)
        << e.what();
  }
  EXPECT_THROW(soma::Space<TypeParam>(D::all(3, 2, 1), Layout::quadrature(2)),
               std::invalid_argument);
}

TYPED_TEST(SpaceTyped, EqualityNeedsDomainAndLayout) {
  using D = soma::Domain<TypeParam>;
  const auto info = soma_test::cells<TypeParam>(3, std::vector<Topology>(4, Topology::hex));
  const soma::Space<TypeParam> a(D::all(3, 4, 1), Layout::per_entity());
  const soma::Space<TypeParam> b(D::all(3, 4, 1), Layout::per_entity());
  const soma::Space<TypeParam> c(D::all(3, 4, 1), Layout::element_nodes(NodeFamily::gll, 2), info);
  const soma::Space<TypeParam> d(D::from_list(3, 4, 1, std::vector<LO>{0, 2}), Layout::per_entity());
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_FALSE(a == d);
}

TYPED_TEST(SpaceTyped, MirrorKeepsRowNumbering) {
  const std::vector<Topology> topo{Topology::quad, Topology::tri, Topology::tri, Topology::quad};
  const auto info = soma_test::cells<HostMem>(2, topo);
  const auto dom = soma::Domain<HostMem>::from_list(2, 4, 1, std::vector<LO>{1, 2, 3});
  const soma::Space<HostMem> h(dom, Layout::element_nodes(NodeFamily::gll, 2), info);
  const auto m = h.template mirror<TypeParam>(dom.template mirror<TypeParam>());
  EXPECT_EQ(m.rows(), h.rows());
  const auto ranges = soma_test::rows_of_entities<TypeParam>(m.view(), 2, 4);
  for (LO l = 0; l < 4; ++l) EXPECT_EQ(ranges(2 * l), h.entity_rows(2, l).begin) << l;
}

namespace soma_test {

constexpr std::int64_t lo_max = std::numeric_limits<LO>::max();
constexpr int hex_order = 31;  // 32^3 = 32768 nodes per hex
constexpr std::int64_t hex_nodes = 32768;
constexpr std::int64_t max_cells = 1 << 20;

soma::Layout hex_layout() { return Layout::element_nodes(NodeFamily::gll, hex_order); }

template <class M>
soma::ElementInfo<M> hexes_then_prisms(std::int64_t hexes, std::int64_t prisms) {
  std::vector<Topology> t(static_cast<std::size_t>(hexes), Topology::hex);
  t.insert(t.end(), static_cast<std::size_t>(prisms), Topology::prism);
  return cells<M>(3, t);
}

}  // namespace soma_test

TYPED_TEST(SpaceTyped, RowsAtTheLimitOfLOAreAccepted) {
  using D = soma::Domain<TypeParam>;
  const std::int64_t n = std::min(soma_test::lo_max / soma_test::hex_nodes, soma_test::max_cells);
  const auto info = soma_test::cells<TypeParam>(
      3, std::vector<Topology>(static_cast<std::size_t>(n), Topology::hex));
  const soma::Space<TypeParam> sp(D::all(3, static_cast<LO>(n), 1), soma_test::hex_layout(), info);
  EXPECT_EQ(static_cast<std::int64_t>(sp.rows()), n * soma_test::hex_nodes);
  EXPECT_GT(sp.rows(), 0);
}

TYPED_TEST(SpaceTyped, UniformBlockRowsPastTheLimitOfLOThrowLengthError) {
  using D = soma::Domain<TypeParam>;
  const std::int64_t n = soma_test::lo_max / soma_test::hex_nodes + 1;
  if (n > soma_test::max_cells) GTEST_SKIP() << "needs " << n << " entities with this LO";
  const auto info = soma_test::cells<TypeParam>(
      3, std::vector<Topology>(static_cast<std::size_t>(n), Topology::hex));
  try {
    const soma::Space<TypeParam> sp(D::all(3, static_cast<LO>(n), 1), soma_test::hex_layout(),
                                    info);
    FAIL() << "expected std::length_error, rows = " << sp.rows();
  } catch (const std::length_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("order 31"), std::string::npos) << msg;
    EXPECT_NE(msg.find(std::to_string(soma_test::lo_max)), std::string::npos) << msg;
  }
}

TYPED_TEST(SpaceTyped, MixedBlockRowsPastTheLimitOfLOThrowLengthError) {
  using D = soma::Domain<TypeParam>;
  const std::int64_t hexes = soma_test::lo_max / soma_test::hex_nodes;
  if (hexes > soma_test::max_cells) GTEST_SKIP() << "needs " << hexes << " entities with this LO";
  const std::int64_t prisms = 40;
  const auto info = soma_test::hexes_then_prisms<TypeParam>(hexes, prisms);
  EXPECT_THROW(soma::Space<TypeParam>(D::all(3, static_cast<LO>(hexes + prisms), 1),
                                      soma_test::hex_layout(), info),
               std::length_error);
}

TYPED_TEST(SpaceTyped, RowsSummedOverBlocksPastTheLimitOfLOThrowLengthError) {
  using D = soma::Domain<TypeParam>;
  const std::int64_t hexes = soma_test::lo_max / soma_test::hex_nodes;
  if (hexes > soma_test::max_cells) GTEST_SKIP() << "needs " << hexes << " entities with this LO";
  soma::ElementInfo<TypeParam> info;
  info.topology[3] = soma_test::topology_view<TypeParam>(
      std::vector<Topology>(static_cast<std::size_t>(hexes), Topology::hex));
  info.topology[2] =
      soma_test::topology_view<TypeParam>(std::vector<Topology>(1000, Topology::quad));
  const auto dom = D::combine(D::all(2, 1000, 1), D::all(3, static_cast<LO>(hexes), 1));
  EXPECT_THROW(soma::Space<TypeParam>(dom, soma_test::hex_layout(), info), std::length_error);
}

TYPED_TEST(SpaceTyped, PolygonArityBeyondIntIsRejected) {
  using D = soma::Domain<TypeParam>;
  if constexpr (sizeof(soma::Offset) < 8) {
    GTEST_SKIP() << "needs 64-bit Offset";
  } else {
    soma::ElementInfo<TypeParam> info;
    info.topology[2] = soma_test::topology_view<TypeParam>({Topology::polygon});
    const std::int64_t wide = (std::int64_t{1} << 32) + 5;
    info.conn_offsets[2] = soma_test::offsets_view<TypeParam>({0, static_cast<soma::Offset>(wide)});
    EXPECT_THROW(soma::Space<TypeParam>(D::all(2, 1, 1),
                                        Layout::element_nodes(NodeFamily::lagrange_equispaced, 1),
                                        info),
                 std::invalid_argument);
  }
}

TYPED_TEST(SpaceTyped, MovedFromSpaceIsEmptyAndUsable) {
  using D = soma::Domain<TypeParam>;
  const auto info = soma_test::cells<TypeParam>(3, std::vector<Topology>(4, Topology::hex));
  soma::Space<TypeParam> a(D::all(3, 4, 1), Layout::element_nodes(NodeFamily::gll, 2), info);
  const soma::Space<TypeParam> copy = a;
  soma::Space<TypeParam> b(std::move(a));
  EXPECT_EQ(a.rows(), 0);
  EXPECT_EQ(a.num_blocks(), 0);
  EXPECT_TRUE(a.domain().empty());
  EXPECT_TRUE(a.layout() == Layout::per_entity());
  EXPECT_FALSE(a.entity_rows(3, 1).valid());
  EXPECT_EQ(a.view().rows, 0);
  EXPECT_EQ(a.view().num_blocks, 0);
  EXPECT_THROW((void)a.block(0), std::out_of_range);
  EXPECT_TRUE(a == soma::Space<TypeParam>());
  EXPECT_EQ(a.template mirror<HostMem>(a.domain().template mirror<HostMem>()).rows(), 0);
  EXPECT_EQ(b.rows(), 4 * 27);
  EXPECT_TRUE(b == copy);

  soma::Space<TypeParam> c(D::all(3, 2, 1), Layout::per_entity());
  c = std::move(b);
  EXPECT_EQ(b.rows(), 0);
  EXPECT_TRUE(b.domain().empty());
  EXPECT_EQ(c.rows(), 4 * 27);

  b = copy;
  EXPECT_EQ(b.rows(), 4 * 27);
}

TYPED_TEST(SpaceTyped, SelfMoveLeavesTheSpaceUnchanged) {
  using D = soma::Domain<TypeParam>;
  soma::Space<TypeParam> a(D::all(3, 4, 1), Layout::per_entity());
  soma::Space<TypeParam>& alias = a;
  a = std::move(alias);
  EXPECT_EQ(a.rows(), 4);
}

TYPED_TEST(SpaceTyped, TheEmptySnapshotCarriesNoReferenceCount) {
  using D = soma::Domain<TypeParam>;
  const soma::Space<TypeParam> first;
  EXPECT_EQ(first.snapshot_use_count(), 0L);
  soma::Space<TypeParam> source(D::all(2, 5, 1), Layout::per_entity());
  std::vector<soma::Space<TypeParam>> many(64);
  for (auto& s : many) {
    soma::Space<TypeParam> taken(std::move(source));
    s = std::move(taken);
    source = soma::Space<TypeParam>(D::all(2, 5, 1), Layout::per_entity());
  }
  EXPECT_EQ(first.snapshot_use_count(), 0L);
  soma::Space<TypeParam> moved_from(D::all(2, 5, 1), Layout::per_entity());
  const soma::Space<TypeParam> kept(std::move(moved_from));
  EXPECT_EQ(moved_from.snapshot_use_count(), 0L);
  EXPECT_EQ(kept.snapshot_use_count(), 1L) << "real snapshots are still counted";
  EXPECT_EQ(first.storage_id(), moved_from.storage_id());
}
