// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/domain.hpp"

using soma::DomainKind;
using soma::GO;
using soma::Interval;
using soma::LO;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace soma_test {

template <class M>
soma::View1<const GO, M> gid_view(const std::vector<GO>& g) {
  soma::View1<GO, Kokkos::HostSpace> h("soma::test::gid", g.size());
  for (std::size_t i = 0; i < g.size(); ++i) h(i) = g[i];
  return Kokkos::create_mirror_view_and_copy(M{}, h);
}

template <class M>
Kokkos::View<LO*, Kokkos::HostSpace> rows_on_space(soma::DomainView<M> dv, int dim, LO count) {
  Kokkos::View<LO*, M> out("soma::test::rows", count);
  Kokkos::parallel_for(
      "soma::test::rows", Kokkos::RangePolicy<typename M::execution_space>(0, count),
      KOKKOS_LAMBDA(const LO l) { out(l) = dv.row(dim, l); });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
}

template <class M>
Kokkos::View<LO*, Kokkos::HostSpace> entities_on_space(soma::DomainView<M> dv, LO size) {
  Kokkos::View<LO*, M> out("soma::test::entities", size);
  Kokkos::parallel_for(
      "soma::test::entities", Kokkos::RangePolicy<typename M::execution_space>(0, size),
      KOKKOS_LAMBDA(const LO r) { out(r) = dv.locate(r).lid; });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
}

}  // namespace soma_test

template <class M>
class DomainTyped : public ::testing::Test {};
using Spaces = ::testing::Types<HostMem, DevMem>;
TYPED_TEST_SUITE(DomainTyped, Spaces);

TYPED_TEST(DomainTyped, AllIsFullAndIdentity) {
  const auto d = soma::Domain<TypeParam>::all(3, 12, 1);
  ASSERT_EQ(d.num_blocks(), 1);
  EXPECT_EQ(d.info(0).kind, DomainKind::full);
  EXPECT_EQ(d.size(), 12);
  EXPECT_EQ(d.epoch(), 1u);
  for (LO l = 0; l < 12; ++l) EXPECT_EQ(d.row(3, l), l);
  EXPECT_EQ(d.row(3, 12), soma::invalid_lo);
  EXPECT_EQ(d.row(2, 0), soma::invalid_lo);
}

TYPED_TEST(DomainTyped, FewRunsBecomeIntervals) {
  const std::vector<Interval> runs{{10, 40}, {60, 90}};
  const auto d = soma::Domain<TypeParam>::from_intervals(2, 100, 1, runs);
  EXPECT_EQ(d.info(0).kind, DomainKind::intervals);
  EXPECT_EQ(d.info(0).runs, 2);
  EXPECT_EQ(d.size(), 60);
  EXPECT_EQ(d.row(2, 10), 0);
  EXPECT_EQ(d.row(2, 39), 29);
  EXPECT_EQ(d.row(2, 40), soma::invalid_lo);
  EXPECT_EQ(d.row(2, 59), soma::invalid_lo);
  EXPECT_EQ(d.row(2, 60), 30);
  EXPECT_EQ(d.row(2, 89), 59);
  EXPECT_EQ(d.row(2, 0), soma::invalid_lo);
  EXPECT_EQ(d.row(2, 99), soma::invalid_lo);
  const auto hv = d.host_block(0);
  for (LO r = 0; r < hv.size; ++r) EXPECT_EQ(hv.row(hv.entity(r)), r);
  EXPECT_EQ(hv.entity(30), 60);
  EXPECT_EQ(hv.entity(60), soma::invalid_lo);
  EXPECT_EQ(hv.entity(-1), soma::invalid_lo);
}

TYPED_TEST(DomainTyped, ScatteredIdsBecomeAList) {
  const std::vector<LO> lids{1, 5, 9, 20, 21, 50};
  const auto d = soma::Domain<TypeParam>::from_list(1, 64, 1, lids);
  EXPECT_EQ(d.info(0).kind, DomainKind::list);
  EXPECT_EQ(d.size(), 6);
  for (std::size_t i = 0; i < lids.size(); ++i) EXPECT_EQ(d.row(1, lids[i]), static_cast<LO>(i));
  EXPECT_EQ(d.row(1, 0), soma::invalid_lo);
  EXPECT_EQ(d.row(1, 22), soma::invalid_lo);
  EXPECT_EQ(d.row(1, 63), soma::invalid_lo);
  EXPECT_EQ(d.members(0), lids);
}

TYPED_TEST(DomainTyped, ListThatCoversEverythingIsFull) {
  std::vector<LO> lids(8);
  std::iota(lids.begin(), lids.end(), LO{0});
  EXPECT_EQ(soma::Domain<TypeParam>::from_list(0, 8, 1, lids).info(0).kind, DomainKind::full);
}

TYPED_TEST(DomainTyped, ContiguousListIsStoredAsIntervals) {
  std::vector<LO> lids(40);
  std::iota(lids.begin(), lids.end(), LO{100});
  const auto d = soma::Domain<TypeParam>::from_list(3, 1000, 1, lids);
  EXPECT_EQ(d.info(0).kind, DomainKind::intervals);
  EXPECT_EQ(d.info(0).runs, 1);
  EXPECT_EQ(d.row(3, 100), 0);
  EXPECT_EQ(d.row(3, 139), 39);
  EXPECT_EQ(d.row(3, 140), soma::invalid_lo);
}

TYPED_TEST(DomainTyped, EmptyDomainHasNoRows) {
  const auto d = soma::Domain<TypeParam>::from_list(2, 10, 1, {});
  EXPECT_EQ(d.size(), 0);
  EXPECT_EQ(d.row(2, 3), soma::invalid_lo);
  const auto e = soma::Domain<TypeParam>::from_intervals(2, 10, 1, {});
  EXPECT_EQ(d, e);
}

TYPED_TEST(DomainTyped, AdjacentAndEmptyRunsAreCanonicalised) {
  const std::vector<Interval> runs{{0, 0}, {5, 20}, {20, 30}, {50, 50}};
  const auto d = soma::Domain<TypeParam>::from_intervals(1, 100, 1, runs);
  EXPECT_EQ(d.info(0).runs, 1);
  EXPECT_EQ(d.size(), 25);
}

TYPED_TEST(DomainTyped, ConstructionRejectsBadInput) {
  using D = soma::Domain<TypeParam>;
  const std::vector<LO> unsorted{3, 2};
  const std::vector<LO> repeated{3, 3};
  const std::vector<LO> outside{3, 10};
  const std::vector<LO> negative{-1};
  EXPECT_THROW(D::from_list(1, 10, 1, unsorted), std::invalid_argument);
  EXPECT_THROW(D::from_list(1, 10, 1, repeated), std::invalid_argument);
  EXPECT_THROW(D::from_list(1, 10, 1, outside), std::invalid_argument);
  EXPECT_THROW(D::from_list(1, 10, 1, negative), std::invalid_argument);
  EXPECT_THROW(D::from_intervals(1, 10, 1, std::vector<Interval>{{5, 12}}), std::invalid_argument);
  EXPECT_THROW(D::from_intervals(1, 10, 1, std::vector<Interval>{{6, 8}, {2, 4}}),
               std::invalid_argument);
  EXPECT_THROW(D::from_intervals(1, 10, 1, std::vector<Interval>{{2, 6}, {4, 8}}),
               std::invalid_argument);
  EXPECT_THROW(D::all(4, 10, 1), std::invalid_argument);
  EXPECT_THROW(D::all(1, -1, 1), std::invalid_argument);
  EXPECT_THROW(D::all(1, 10, 1, soma_test::gid_view<TypeParam>({1, 2, 3})), std::invalid_argument);
}

TYPED_TEST(DomainTyped, DeviceLookupsAgreeWithTheHost) {
  const std::vector<LO> lids{2, 3, 4, 17, 40, 41, 90};
  const auto lst = soma::Domain<TypeParam>::from_list(1, 100, 1, lids);
  const auto itv = soma::Domain<TypeParam>::from_intervals(
      1, 100, 1, std::vector<Interval>{{10, 40}, {60, 90}});
  const auto all = soma::Domain<TypeParam>::all(1, 100, 1);
  for (const auto& d : {lst, itv, all}) {
    const auto rows = soma_test::rows_on_space<TypeParam>(d.view(), 1, 100);
    for (LO l = 0; l < 100; ++l) EXPECT_EQ(rows(l), d.row(1, l)) << "lid " << l;
    const auto ents = soma_test::entities_on_space<TypeParam>(d.view(), d.size());
    for (LO r = 0; r < d.size(); ++r) EXPECT_EQ(d.row(1, ents(r)), r) << "row " << r;
  }
}

TYPED_TEST(DomainTyped, CombineStacksBlocksByDimension) {
  using D = soma::Domain<TypeParam>;
  const auto faces = D::from_list(2, 50, 7, std::vector<LO>{3, 9, 12});
  const auto cells = D::all(3, 20, 7);
  const auto both = D::combine(faces, cells);
  ASSERT_EQ(both.num_blocks(), 2);
  EXPECT_EQ(both.size(), 23);
  EXPECT_EQ(both.info(0).dim, 2);
  EXPECT_EQ(both.info(1).dim, 3);
  EXPECT_EQ(both.info(1).row_begin, 3);
  EXPECT_EQ(both.row(2, 9), 1);
  EXPECT_EQ(both.row(3, 0), 3);
  EXPECT_EQ(both.row(3, 19), 22);
  EXPECT_EQ(both.row(0, 0), soma::invalid_lo);
  EXPECT_EQ(D::combine(cells, faces), both);

  const auto ents = soma_test::entities_on_space<TypeParam>(both.view(), both.size());
  EXPECT_EQ(ents(0), 3);
  EXPECT_EQ(ents(2), 12);
  EXPECT_EQ(ents(3), 0);
  EXPECT_EQ(ents(22), 19);
}

TYPED_TEST(DomainTyped, CombineRejectsSharedDimensionAndMixedEpochs) {
  using D = soma::Domain<TypeParam>;
  EXPECT_THROW(D::combine(D::all(2, 5, 1), D::from_list(2, 9, 1, std::vector<LO>{1})),
               std::invalid_argument);
  EXPECT_THROW(D::combine(D::all(2, 5, 1), D::all(3, 5, 2)), std::invalid_argument);
  EXPECT_EQ(D::combine(D{}, D::all(2, 5, 1)), D::all(2, 5, 1));
}

TYPED_TEST(DomainTyped, FingerprintIgnoresMemberOrderAndRepresentationPath) {
  using D = soma::Domain<TypeParam>;
  std::vector<GO> g(30);
  std::iota(g.begin(), g.end(), GO{1000});
  const auto gids = soma_test::gid_view<TypeParam>(g);
  const auto by_list = D::from_list(2, 30, 4, std::vector<LO>{4, 5, 6, 7, 8, 9, 10, 11}, gids);
  const auto by_runs = D::from_intervals(2, 30, 4, std::vector<Interval>{{4, 8}, {8, 12}}, gids);
  EXPECT_EQ(by_list.fingerprint(0), by_runs.fingerprint(0));
  EXPECT_EQ(by_list, by_runs);
  EXPECT_FALSE(by_list.shares_storage_with(by_runs));
  EXPECT_NE(by_list.fingerprint(0).sum, 0u);
  EXPECT_NE(by_list.fingerprint(0).xor_hash, 0u);
  EXPECT_NE(by_list.fingerprint(0).sum, by_list.fingerprint(0).xor_hash);
}

TYPED_TEST(DomainTyped, FingerprintSeparatesMembershipEpochAndDimension) {
  using D = soma::Domain<TypeParam>;
  const auto base = D::from_list(2, 30, 4, std::vector<LO>{1, 5, 9});
  EXPECT_FALSE(base == D::from_list(2, 30, 4, std::vector<LO>{1, 5, 10}));
  EXPECT_FALSE(base == D::from_list(2, 30, 5, std::vector<LO>{1, 5, 9}));
  EXPECT_FALSE(base == D::from_list(3, 30, 4, std::vector<LO>{1, 5, 9}));
  EXPECT_FALSE(base == D::from_list(2, 30, 4, std::vector<LO>{1, 5}));
  EXPECT_EQ(base, D::from_list(2, 30, 4, std::vector<LO>{1, 5, 9}));
}

TYPED_TEST(DomainTyped, GlobalIdsMakeTheHashIndependentOfLocalNumbering) {
  using D = soma::Domain<TypeParam>;
  std::vector<GO> g1(20);
  std::iota(g1.begin(), g1.end(), GO{100});
  std::vector<GO> g2(g1.rbegin(), g1.rend());
  const auto a = D::from_list(1, 20, 1, std::vector<LO>{0, 1, 2}, soma_test::gid_view<TypeParam>(g1));
  const auto b = D::from_list(1, 20, 1, std::vector<LO>{17, 18, 19},
                              soma_test::gid_view<TypeParam>(g2));
  EXPECT_EQ(a.fingerprint(0), b.fingerprint(0));
}

// Two different member sets with identical global ids produce the same
// fingerprint, so the exact comparison must be what tells them apart.
TYPED_TEST(DomainTyped, EqualFingerprintsDoNotMakeDifferentDomainsEqual) {
  using D = soma::Domain<TypeParam>;
  std::vector<GO> g(20);
  for (std::size_t i = 0; i < g.size(); ++i) g[i] = static_cast<GO>(i % 10);
  const auto gids = soma_test::gid_view<TypeParam>(g);
  const auto a = D::from_list(1, 20, 1, std::vector<LO>{1, 3, 5}, gids);
  const auto b = D::from_list(1, 20, 1, std::vector<LO>{11, 13, 15}, gids);
  ASSERT_TRUE(a.same_fingerprints(b));
  EXPECT_FALSE(a.same_members(b));
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a == D::from_list(1, 20, 1, std::vector<LO>{1, 3, 5}, gids));
}

TYPED_TEST(DomainTyped, MirrorPreservesMembersAndFingerprints) {
  const auto h = soma::Domain<HostMem>::from_list(2, 200, 3, std::vector<LO>{4, 40, 41, 150});
  const auto m = h.template mirror<TypeParam>();
  EXPECT_EQ(m.members(0), h.members(0));
  EXPECT_EQ(m.fingerprint(0), h.fingerprint(0));
  const auto rows = soma_test::rows_on_space<TypeParam>(m.view(), 2, 200);
  for (LO l = 0; l < 200; ++l) EXPECT_EQ(rows(l), h.row(2, l));
}

TEST(Domain, MirrorInTheSameSpaceSharesTheSnapshot) {
  const auto h = soma::Domain<HostMem>::all(1, 9, 1);
  EXPECT_TRUE(h.mirror<HostMem>().shares_storage_with(h));
}

TEST(Domain, DefaultDomainIsEmpty) {
  const soma::Domain<HostMem> d;
  EXPECT_TRUE(d.empty());
  EXPECT_EQ(d.size(), 0);
  EXPECT_EQ(d.num_blocks(), 0);
}

namespace soma_test {

template <class M>
void expect_empty_domain(const soma::Domain<M>& d) {
  EXPECT_TRUE(d.empty());
  EXPECT_EQ(d.size(), 0);
  EXPECT_EQ(d.num_blocks(), 0);
  EXPECT_EQ(d.epoch(), 0u);
  EXPECT_EQ(d.find_block(2), -1);
  EXPECT_FALSE(d.contains_dim(2));
  EXPECT_EQ(d.row(2, 3), soma::invalid_lo);
  EXPECT_EQ(d.view().num_blocks, 0);
  EXPECT_EQ(d.view().size, 0);
  EXPECT_EQ(d.view().locate(0).lid, soma::invalid_lo);
  EXPECT_TRUE(d == soma::Domain<M>());
  EXPECT_THROW((void)d.info(0), std::out_of_range);
  EXPECT_TRUE(d.template mirror<HostMem>().empty());
  EXPECT_TRUE(d.template mirror<DevMem>().empty());
}

}  // namespace soma_test

TYPED_TEST(DomainTyped, MovedFromDomainIsEmptyAndUsable) {
  using D = soma::Domain<TypeParam>;
  D a = D::from_list(2, 40, 7, std::vector<LO>{3, 9, 20});
  const D copy = a;
  D b(std::move(a));
  soma_test::expect_empty_domain(a);
  EXPECT_EQ(b.size(), 3);
  EXPECT_TRUE(b == copy);
  EXPECT_TRUE(b.shares_storage_with(copy));

  D c = D::all(3, 5, 1);
  c = std::move(b);
  soma_test::expect_empty_domain(b);
  EXPECT_EQ(c.size(), 3);
  EXPECT_EQ(c.row(2, 9), 1);

  b = D::all(1, 4, 2);  // a moved-from object can be assigned to and used again
  EXPECT_EQ(b.size(), 4);
  EXPECT_EQ(b.epoch(), 2u);
}

TYPED_TEST(DomainTyped, SelfMoveLeavesTheDomainUnchanged) {
  using D = soma::Domain<TypeParam>;
  D a = D::from_list(2, 40, 7, std::vector<LO>{3, 9, 20});
  const D copy = a;
  D& alias = a;
  a = std::move(alias);
  EXPECT_TRUE(a == copy);
  EXPECT_EQ(a.size(), 3);
}

TYPED_TEST(DomainTyped, CopiesStayCheapAndShareTheSnapshot) {
  using D = soma::Domain<TypeParam>;
  const D a = D::all(2, 9, 1);
  D b = a;
  EXPECT_TRUE(b.shares_storage_with(a));
  D c;
  c = a;
  EXPECT_TRUE(c.shares_storage_with(a));
  EXPECT_EQ(a.size(), 9);
}

TEST(Domain, CheckedConversionToLOThrowsPastTheLimit) {
  constexpr auto lo_max = static_cast<std::size_t>(std::numeric_limits<LO>::max());
  EXPECT_EQ(soma::detail::to_lo(0, "x"), 0);
  EXPECT_EQ(soma::detail::to_lo(lo_max, "x"), std::numeric_limits<LO>::max());
  try {
    (void)soma::detail::to_lo(lo_max + 1, "pool offset");
    FAIL() << "expected std::length_error";
  } catch (const std::length_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("pool offset"), std::string::npos) << msg;
    EXPECT_NE(msg.find(std::to_string(std::numeric_limits<LO>::max())), std::string::npos) << msg;
  }
  EXPECT_THROW((void)soma::detail::to_lo(std::numeric_limits<std::size_t>::max(), "x"),
               std::length_error);
}
