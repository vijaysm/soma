// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <memory>
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
#include "soma/fields/errors.hpp"
#include "soma/fields/field.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/registry.hpp"
#include "soma/fields/resolver.hpp"
#include "soma/fields/space.hpp"

using soma::DimIndexKind;
using soma::GO;
using soma::Interval;
using soma::Layout;
using soma::LO;
using soma::NodeFamily;
using soma::Topology;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace soma_test {

template <class M>
Kokkos::View<LO*, Kokkos::HostSpace> lookup_on_space(soma::FamilyView<M> fv, int dim, LO count) {
  Kokkos::View<LO*, M> out("soma::test::family_hits", 2 * count);
  Kokkos::parallel_for(
      "soma::test::family_hits", Kokkos::RangePolicy<typename M::execution_space>(0, count),
      KOKKOS_LAMBDA(const LO l) {
        const soma::FamilyHit h = fv.lookup(dim, l);
        out(2 * l) = h.slot;
        out(2 * l + 1) = h.row;
      });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
}

template <class M>
soma::View1<const GO, M> gids(LO n, GO first) {
  soma::View1<GO, Kokkos::HostSpace> h("soma::test::gids", static_cast<std::size_t>(n));
  for (LO i = 0; i < n; ++i) h(i) = first + i;
  return Kokkos::create_mirror_view_and_copy(M{}, h);
}

}  // namespace soma_test

template <class M>
class FamilyTyped : public ::testing::Test {
 protected:
  using D = soma::Domain<M>;
  using S = soma::Space<M>;
  soma::FieldRegistry<M> reg;

  void SetUp() override {
    reg.set_entities(2, 100, soma_test::gids<M>(100, 5000));
    reg.set_entities(3, 40);
  }

  S space(const D& d) const { return S(d, Layout::per_entity()); }
  D list(int dim, LO count, std::vector<LO> lids) const {
    return D::from_list(dim, count, reg.epoch(), lids);
  }
  D runs(int dim, LO count, std::vector<Interval> r) const {
    return D::from_intervals(dim, count, reg.epoch(), r);
  }
};
using Spaces = ::testing::Types<HostMem, DevMem>;
TYPED_TEST_SUITE(FamilyTyped, Spaces);

TYPED_TEST(FamilyTyped, FullThenSubsetOverlapNamesBothFieldsDimensionAndGid) {
  this->reg.template add_tag<double>("T", 2, 1);
  try {
    this->reg.template add<double>("T", this->space(this->runs(2, 100, {{10, 60}})), 1);
    FAIL() << "expected FieldOverlap";
  } catch (const soma::FieldOverlap& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("field 'T' (f64x1) on {intervals of dimension 2, 50 of 100 entities"),
              std::string::npos) << msg;
    EXPECT_NE(msg.find("intersects existing 'T' (f64x1) on {full of dimension 2, 100 of 100"),
              std::string::npos) << msg;
    EXPECT_NE(msg.find("in dim 2 at entity gid 5010"), std::string::npos) << msg;
  }
}

TYPED_TEST(FamilyTyped, OverlapIsAnInvalidArgument) {
  this->reg.template add<double>("T", this->space(this->list(2, 100, {4, 8})), 1);
  EXPECT_THROW(this->reg.template add<double>("T", this->space(this->list(2, 100, {8, 9})), 1),
               std::invalid_argument);
}

TYPED_TEST(FamilyTyped, WithoutGlobalIdsTheMessageNamesTheLocalId) {
  this->reg.template add_tag<double>("T", 3, 1);
  try {
    this->reg.template add<double>("T", this->space(this->list(3, 40, {7, 9})), 1);
    FAIL() << "expected FieldOverlap";
  } catch (const soma::FieldOverlap& e) {
    EXPECT_NE(std::string(e.what()).find("in dim 3 at entity lid 7"), std::string::npos) << e.what();
  }
}

TYPED_TEST(FamilyTyped, EveryRepresentationPairDetectsAnOverlap) {
  struct Pair {
    typename TestFixture::D first;
    typename TestFixture::D second;
    LO shared;
  };
  const std::vector<Pair> pairs{
      {this->runs(2, 100, {{0, 30}}), this->runs(2, 100, {{29, 50}}), 29},
      {this->runs(2, 100, {{0, 30}, {60, 90}}), this->runs(2, 100, {{40, 50}, {85, 99}}), 85},
      {this->runs(2, 100, {{0, 30}, {60, 90}}), this->list(2, 100, {5, 31, 70}), 5},
      {this->list(2, 100, {5, 31, 70}), this->runs(2, 100, {{0, 30}, {60, 90}}), 5},
      {this->list(2, 100, {1, 3, 5, 99}), this->list(2, 100, {2, 4, 5}), 5},
      {this->list(2, 100, {0, 99}), this->list(2, 100, {99}), 99},
      {this->runs(2, 100, {{10, 20}}), this->list(2, 100, {10, 50}), 10},
      {this->list(2, 100, {10, 50}), this->runs(2, 100, {{10, 20}}), 10},
      {this->list(2, 100, {3, 4}), this->runs(2, 100, {{0, 100}}), 3},
      {this->runs(2, 100, {{0, 100}}), this->list(2, 100, {3, 4}), 3},
  };
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    soma::FieldRegistry<TypeParam> r;
    r.set_entities(2, 100);
    r.template add<double>("T", this->space(pairs[i].first), 1);
    try {
      r.template add<double>("T", this->space(pairs[i].second), 1);
      ADD_FAILURE() << "pair " << i << " did not throw";
    } catch (const soma::FieldOverlap& e) {
      EXPECT_NE(std::string(e.what()).find("lid " + std::to_string(pairs[i].shared)),
                std::string::npos)
          << "pair " << i << ": " << e.what();
    }
  }
}

TYPED_TEST(FamilyTyped, DisjointNeighboursAreAccepted) {
  this->reg.template add<double>("T", this->space(this->runs(2, 100, {{0, 30}})), 1);
  this->reg.template add<double>("T", this->space(this->runs(2, 100, {{30, 50}})), 1);
  this->reg.template add<double>("T", this->space(this->list(2, 100, {55, 60, 99})), 1);
  this->reg.template add<double>("T", this->space(this->list(2, 100, {56, 61})), 1);
  EXPECT_EQ(this->reg.list().size(), 4u);
}

TYPED_TEST(FamilyTyped, DifferentDimensionsNeverOverlap) {
  this->reg.template add_tag<double>("GLOBAL_ID", 2, 1);
  EXPECT_NO_THROW(this->reg.template add_tag<double>("GLOBAL_ID", 3, 1));
  EXPECT_EQ(this->reg.list()[0].family, this->reg.list()[1].family);
}

TYPED_TEST(FamilyTyped, OtherFamiliesDoNotInteract) {
  this->reg.template add_tag<double>("T", 2, 1);
  EXPECT_NO_THROW(this->reg.template add_tag<float>("T", 2, 1));
  EXPECT_NO_THROW(this->reg.template add_tag<double>("U", 2, 1));
  EXPECT_NO_THROW(this->reg.template add_tag<double>("T", 2, 3));
}

TYPED_TEST(FamilyTyped, AFailedAddChangesNothing) {
  this->reg.template add<double>("T", this->space(this->list(2, 100, {4, 8})), 1);
  const auto before = this->reg.list();
  EXPECT_THROW(this->reg.template add<double>("T", this->space(this->list(2, 100, {8, 9})), 1),
               soma::FieldOverlap);
  EXPECT_EQ(this->reg.list().size(), before.size());
  const auto fam = this->reg.template family<double>("T", 1);
  EXPECT_EQ(this->reg.family_members(fam, 2).size(), 1u);
  const auto ok = this->reg.template add<double>("T", this->space(this->list(2, 100, {9, 10})), 1);
  EXPECT_EQ(ok.id(), before.size());
}

TYPED_TEST(FamilyTyped, RejectedOverlapLeavesTheResolverDeterministic) {
  const auto a = this->reg.template add_tag<double>("T", 2, 1);
  const auto fam = this->reg.template family<double>("T", 1);
  EXPECT_THROW(this->reg.template add<double>("T", this->space(this->runs(2, 100, {{10, 60}})), 1),
               soma::FieldOverlap);
  EXPECT_EQ(this->reg.index_kind(fam, 2), DimIndexKind::single_full);
  EXPECT_EQ(this->reg.family_members(fam, 2), std::vector<soma::FieldId>{a.id()});
  for (const LO l : {LO{0}, LO{10}, LO{35}, LO{59}, LO{99}}) {
    const auto h = this->reg.lookup(fam, 2, l);
    EXPECT_EQ(h.slot, 0) << l;
    EXPECT_EQ(h.row, l) << l;
  }
}

TYPED_TEST(FamilyTyped, LayoutIsPartOfTheFamilyKey) {
  using D = typename TestFixture::D;
  soma::View1<std::uint8_t, Kokkos::HostSpace> h("soma::test::topo", 40);
  for (int i = 0; i < 40; ++i) h(i) = static_cast<std::uint8_t>(Topology::hex);
  soma::ElementInfo<TypeParam> info;
  info.topology[3] = Kokkos::create_mirror_view_and_copy(TypeParam{}, h);
  this->reg.template add_tag<double>("u", 3, 1);
  const soma::Space<TypeParam> sem(D::all(3, 40, this->reg.epoch()),
                                   Layout::element_nodes(NodeFamily::gll, 2), info);
  EXPECT_NO_THROW(this->reg.template add<double>("u", sem, 1));
  const auto per = this->reg.template family<double>("u", 1);
  const auto nodes = this->reg.template family<double>("u", 1, Layout::element_nodes(NodeFamily::gll, 2));
  EXPECT_NE(per, nodes);
  const auto hit = this->reg.lookup(nodes, 3, 2);
  EXPECT_EQ(hit.slot, 0);
  EXPECT_EQ(hit.row, 54);
}

TYPED_TEST(FamilyTyped, HandleKeepsServingTheOldIndexesAfterTheFamilyChanges) {
  const auto a = this->reg.template add<double>("T", this->space(this->list(2, 100, {4, 8})), 1);
  const auto fam = this->reg.template family<double>("T", 1);
  const auto handle = this->reg.family_handle(fam);
  ASSERT_EQ(handle.members(2), std::vector<soma::FieldId>{a.id()});
  const auto before = soma_test::lookup_on_space<TypeParam>(handle.view(), 2, 100);

  const auto b = this->reg.template add<double>("T", this->space(this->list(2, 100, {9, 10})), 1);
  this->reg.template add<double>("T", this->space(this->runs(2, 100, {{50, 60}})), 1);
  this->reg.remove(b.id());
  EXPECT_FALSE(this->reg.lookup(fam, 2, 9).found());
  EXPECT_TRUE(this->reg.lookup(fam, 2, 55).found());
  this->reg.set_epoch(this->reg.epoch() + 1);
  this->reg.set_epoch(this->reg.epoch() - 1);
  EXPECT_TRUE(this->reg.lookup(fam, 2, 55).found());

  const auto after = soma_test::lookup_on_space<TypeParam>(handle.view(), 2, 100);
  for (LO l = 0; l < 100; ++l) {
    EXPECT_EQ(after(2 * l), before(2 * l)) << "lid " << l;
    EXPECT_EQ(after(2 * l + 1), before(2 * l + 1)) << "lid " << l;
  }
  EXPECT_EQ(after(2 * 4), 0);
  EXPECT_EQ(after(2 * 8 + 1), 1);
  EXPECT_EQ(after(2 * 9), -1);
  EXPECT_EQ(after(2 * 55), -1);
  EXPECT_EQ(handle.members(2), std::vector<soma::FieldId>{a.id()});
}

TYPED_TEST(FamilyTyped, HandleCopiesShareTheIndexes) {
  this->reg.template add<double>("T", this->space(this->list(2, 100, {4, 8})), 1);
  const auto fam = this->reg.template family<double>("T", 1);
  auto first = std::make_unique<soma::FamilyHandle<TypeParam>>(this->reg.family_handle(fam));
  const soma::FamilyHandle<TypeParam> copy = *first;
  first.reset();
  this->reg.template add<double>("T", this->space(this->list(2, 100, {9})), 1);
  const auto got = soma_test::lookup_on_space<TypeParam>(copy.view(), 2, 100);
  EXPECT_EQ(got(2 * 8 + 1), 1);
  EXPECT_EQ(got(2 * 9), -1);
}

TYPED_TEST(FamilyTyped, AFamilyHoldsAtMost255MembersOnADimension) {
  using D = typename TestFixture::D;
  soma::FieldRegistry<TypeParam> reg;
  reg.set_entities(2, 300);
  for (LO i = 0; i < soma::max_family_members; ++i) {
    reg.template add<double>(
        "T", soma::Space<TypeParam>(D::from_list(2, 300, reg.epoch(), std::vector<LO>{i}),
                                    Layout::per_entity()),
        1);
  }
  const auto fam = reg.template family<double>("T", 1);
  EXPECT_EQ(reg.family_members(fam, 2).size(), 255u);
  EXPECT_EQ(reg.lookup(fam, 2, 254).slot, 254);
  EXPECT_FALSE(reg.lookup(fam, 2, 255).found());

  const auto before = reg.list().size();
  try {
    reg.template add<double>(
        "T", soma::Space<TypeParam>(D::from_list(2, 300, reg.epoch(), std::vector<LO>{255}),
                                    Layout::per_entity()),
        1);
    FAIL() << "expected std::length_error";
  } catch (const std::length_error& e) {
    EXPECT_NE(std::string(e.what()).find("at most 255 members on dimension 2"), std::string::npos)
        << e.what();
  }
  EXPECT_EQ(reg.list().size(), before);
  EXPECT_EQ(reg.family_members(fam, 2).size(), 255u);
  EXPECT_FALSE(reg.lookup(fam, 2, 255).found());
  EXPECT_NO_THROW(reg.template add<double>(
      "T", soma::Space<TypeParam>(D::from_list(3, 10, reg.epoch(), std::vector<LO>{0}),
                                  Layout::per_entity()),
      1));
}

TYPED_TEST(FamilyTyped, IndexKindFollowsTheMembers) {
  this->reg.template add_tag<double>("a", 2, 1);
  const auto fam_a = this->reg.template family<double>("a", 1);
  EXPECT_EQ(this->reg.index_kind(fam_a, 2), DimIndexKind::single_full);
  EXPECT_EQ(this->reg.index_kind(fam_a, 3), DimIndexKind::none);
  EXPECT_EQ(this->reg.index_kind(fam_a, 0), DimIndexKind::none);

  this->reg.template add<double>("b", this->space(this->list(2, 100, {4, 8})), 1);
  const auto fam_b = this->reg.template family<double>("b", 1);
  EXPECT_EQ(this->reg.index_kind(fam_b, 2), DimIndexKind::single_subset);

  this->reg.template add<double>("b", this->space(this->runs(2, 100, {{10, 40}})), 1);
  EXPECT_EQ(this->reg.index_kind(fam_b, 2), DimIndexKind::multi_owner);

  this->reg.template add<double>("c", this->space(this->runs(2, 100, {{10, 40}, {60, 80}})), 1);
  this->reg.template add<double>("c", this->space(this->runs(2, 100, {{40, 60}, {90, 100}})), 1);
  const auto fam_c = this->reg.template family<double>("c", 1);
  EXPECT_EQ(this->reg.index_kind(fam_c, 2), DimIndexKind::multi_intervals);
}

TYPED_TEST(FamilyTyped, SingleFullAndSubsetLookupsGiveRows) {
  this->reg.template add_tag<double>("a", 2, 1);
  const auto fa = this->reg.template family<double>("a", 1);
  EXPECT_EQ(this->reg.lookup(fa, 2, 57).row, 57);
  EXPECT_EQ(this->reg.lookup(fa, 2, 57).slot, 0);
  EXPECT_FALSE(this->reg.lookup(fa, 3, 1).found());
  EXPECT_FALSE(this->reg.lookup(fa, 2, 100).found());

  this->reg.template add<double>("b", this->space(this->list(2, 100, {4, 8, 50})), 1);
  const auto fb = this->reg.template family<double>("b", 1);
  EXPECT_EQ(this->reg.lookup(fb, 2, 8).row, 1);
  EXPECT_FALSE(this->reg.lookup(fb, 2, 9).found());
}

TYPED_TEST(FamilyTyped, DisjointMembersResolveToTheirOwnMember) {
  const auto m0 = this->reg.template add<double>("T", this->space(this->runs(2, 100, {{10, 40}, {60, 80}})), 1);
  const auto m1 = this->reg.template add<double>("T", this->space(this->runs(2, 100, {{40, 60}})), 1);
  const auto m2 = this->reg.template add<double>("T", this->space(this->list(2, 100, {0, 99})), 1);
  const auto fam = this->reg.template family<double>("T", 1);
  ASSERT_EQ(this->reg.index_kind(fam, 2), DimIndexKind::multi_owner);
  const auto members = this->reg.family_members(fam, 2);
  ASSERT_EQ(members.size(), 3u);

  const auto h0 = this->reg.lookup(fam, 2, 70);
  EXPECT_EQ(members[static_cast<std::size_t>(h0.slot)], m0.id());
  EXPECT_EQ(h0.row, 30 + 10);
  const auto h1 = this->reg.lookup(fam, 2, 41);
  EXPECT_EQ(members[static_cast<std::size_t>(h1.slot)], m1.id());
  EXPECT_EQ(h1.row, 1);
  const auto h2 = this->reg.lookup(fam, 2, 99);
  EXPECT_EQ(members[static_cast<std::size_t>(h2.slot)], m2.id());
  EXPECT_EQ(h2.row, 1);
  EXPECT_FALSE(this->reg.lookup(fam, 2, 5).found());
  EXPECT_FALSE(this->reg.lookup(fam, 2, 85).found());
}

TYPED_TEST(FamilyTyped, MergedIntervalTableResolvesEveryEntity) {
  this->reg.template add<double>("T", this->space(this->runs(2, 100, {{10, 40}, {60, 80}})), 1);
  this->reg.template add<double>("T", this->space(this->runs(2, 100, {{40, 60}, {90, 100}})), 1);
  const auto fam = this->reg.template family<double>("T", 1);
  ASSERT_EQ(this->reg.index_kind(fam, 2), DimIndexKind::multi_intervals);
  EXPECT_EQ(this->reg.lookup(fam, 2, 10).slot, 0);
  EXPECT_EQ(this->reg.lookup(fam, 2, 10).row, 0);
  EXPECT_EQ(this->reg.lookup(fam, 2, 60).slot, 0);
  EXPECT_EQ(this->reg.lookup(fam, 2, 60).row, 30);
  EXPECT_EQ(this->reg.lookup(fam, 2, 59).slot, 1);
  EXPECT_EQ(this->reg.lookup(fam, 2, 59).row, 19);
  EXPECT_EQ(this->reg.lookup(fam, 2, 95).slot, 1);
  EXPECT_EQ(this->reg.lookup(fam, 2, 95).row, 25);
  EXPECT_FALSE(this->reg.lookup(fam, 2, 5).found());
  EXPECT_FALSE(this->reg.lookup(fam, 2, 85).found());
  EXPECT_FALSE(this->reg.lookup(fam, 2, 100).found());
}

TYPED_TEST(FamilyTyped, DeviceFamilyViewMatchesTheHostLookup) {
  this->reg.template add<double>("T", this->space(this->runs(2, 100, {{10, 40}, {60, 80}})), 1);
  this->reg.template add<double>("T", this->space(this->runs(2, 100, {{40, 60}})), 1);
  this->reg.template add<double>("T", this->space(this->list(2, 100, {0, 99})), 1);
  this->reg.template add<double>("S", this->space(this->runs(2, 100, {{5, 50}})), 1);
  this->reg.template add<double>("S", this->space(this->runs(2, 100, {{70, 90}})), 1);
  for (const char* name : {"T", "S"}) {
    const auto fam = this->reg.template family<double>(name, 1);
    const auto handle = this->reg.family_handle(fam);
    const auto got = soma_test::lookup_on_space<TypeParam>(handle.view(), 2, 100);
    for (LO l = 0; l < 100; ++l) {
      const auto h = this->reg.lookup(fam, 2, l);
      EXPECT_EQ(got(2 * l), h.slot) << name << " lid " << l;
      EXPECT_EQ(got(2 * l + 1), h.row) << name << " lid " << l;
    }
  }
}

TYPED_TEST(FamilyTyped, IndexIsBuiltOnceAndRebuiltWhenMembershipChanges) {
  this->reg.template add<double>("T", this->space(this->list(2, 100, {1, 2})), 1);
  const auto fam = this->reg.template family<double>("T", 1);
  EXPECT_EQ(this->reg.index_builds(), 0u);
  this->reg.lookup(fam, 2, 1);
  EXPECT_EQ(this->reg.index_builds(), 1u);
  this->reg.lookup(fam, 2, 2);
  this->reg.lookup(fam, 2, 50);
  EXPECT_EQ(this->reg.index_builds(), 1u);
  this->reg.lookup(fam, 3, 0);
  EXPECT_EQ(this->reg.index_builds(), 2u);

  const auto added = this->reg.template add<double>("T", this->space(this->list(2, 100, {7})), 1);
  EXPECT_TRUE(this->reg.lookup(fam, 2, 7).found());
  EXPECT_EQ(this->reg.index_builds(), 3u);
  this->reg.remove(added.id());
  EXPECT_FALSE(this->reg.lookup(fam, 2, 7).found());
  EXPECT_EQ(this->reg.index_builds(), 4u);
}

TYPED_TEST(FamilyTyped, EpochChangeInvalidatesTheIndex) {
  this->reg.template add<double>("T", this->space(this->list(2, 100, {1, 2})), 1);
  const auto fam = this->reg.template family<double>("T", 1);
  this->reg.lookup(fam, 2, 1);
  this->reg.set_epoch(this->reg.epoch() + 1);
  EXPECT_THROW(this->reg.lookup(fam, 2, 1), soma::StaleField);
}

TEST(FamilyHost, ValueFallsBackToTheDefaultWhenNoMemberHasTheEntity) {
  using D = soma::Domain<HostMem>;
  soma::FieldRegistry<HostMem> reg;
  reg.set_entities(2, 20);
  soma::FieldOptions<double> d0;
  d0.default_value = std::vector<double>{-7.0, -8.0};
  auto a = reg.add<double>("T", soma::Space<HostMem>(D::from_list(2, 20, 1, std::vector<LO>{2, 3}), Layout::per_entity()), 2, d0);
  auto b = reg.add<double>("T", soma::Space<HostMem>(D::from_list(2, 20, 1, std::vector<LO>{10, 11}), Layout::per_entity()), 2);
  a.view()(1, 1) = 3.5;
  b.view()(0, 0) = 6.5;
  const auto fam = reg.family<double>("T", 2);
  EXPECT_DOUBLE_EQ(reg.value<double>(fam, 2, 3, 1), 3.5);
  EXPECT_DOUBLE_EQ(reg.value<double>(fam, 2, 10, 0), 6.5);
  EXPECT_DOUBLE_EQ(reg.value<double>(fam, 2, 15, 0), -7.0);
  EXPECT_DOUBLE_EQ(reg.value<double>(fam, 2, 15, 1), -8.0);
  EXPECT_DOUBLE_EQ(reg.value<double>(fam, 3, 0, 0), -7.0);
}

TEST(FamilyHost, GatherReadsAcrossDimensionsAndMembers) {
  using D = soma::Domain<HostMem>;
  soma::FieldRegistry<HostMem> reg;
  reg.set_entities(0, 6);
  reg.set_entities(2, 10);
  auto v = reg.add_tag<std::int32_t>("GLOBAL_ID", 0, 1);
  auto f = reg.add<std::int32_t>("GLOBAL_ID", soma::Space<HostMem>(D::from_list(2, 10, 1, std::vector<LO>{3, 7}), Layout::per_entity()), 1);
  for (LO i = 0; i < 6; ++i) v.view()(i, 0) = 100 + static_cast<std::int32_t>(i);
  f.view()(0, 0) = 33;
  f.view()(1, 0) = 77;
  const auto fam = reg.family<std::int32_t>("GLOBAL_ID", 1);
  const std::vector<std::pair<int, LO>> q{{0, 4}, {2, 7}, {2, 5}, {0, 0}, {2, 3}};
  std::vector<std::int32_t> out(q.size(), -1);
  reg.gather<std::int32_t>(fam, q, 0, out);
  EXPECT_EQ(out, (std::vector<std::int32_t>{104, 77, 0, 100, 33}));
  std::vector<std::int32_t> small(2);
  EXPECT_THROW(reg.gather<std::int32_t>(fam, q, 0, small), std::invalid_argument);
}

TEST(FamilyHost, MirroredRegistryResolvesLikeTheSource) {
  using D = soma::Domain<HostMem>;
  soma::FieldRegistry<HostMem> reg;
  reg.set_entities(2, 50);
  reg.add<double>("T", soma::Space<HostMem>(D::from_intervals(2, 50, 1, std::vector<Interval>{{0, 20}}), Layout::per_entity()), 1);
  reg.add<double>("T", soma::Space<HostMem>(D::from_list(2, 50, 1, std::vector<LO>{30, 40}), Layout::per_entity()), 1);
  soma::FieldRegistry<DevMem> dev;
  reg.mirror_into<DevMem>(dev, {"*"}, {true, true, true, true}, false);
  const auto fam = reg.family<double>("T", 1);
  const auto dfam = dev.family<double>("T", 1);
  EXPECT_EQ(fam, dfam);
  const auto handle = dev.family_handle(dfam);
  const auto got = soma_test::lookup_on_space<DevMem>(handle.view(), 2, 50);
  for (LO l = 0; l < 50; ++l) {
    const auto h = reg.lookup(fam, 2, l);
    EXPECT_EQ(got(2 * l), h.slot) << l;
    EXPECT_EQ(got(2 * l + 1), h.row) << l;
  }
  EXPECT_THROW(dev.add<double>("T", soma::Space<DevMem>(D::all(2, 50, 1).mirror<DevMem>(), Layout::per_entity()), 1),
               soma::FieldOverlap);
}
