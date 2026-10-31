// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
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
#include "soma/fields/space.hpp"

using soma::Layout;
using soma::LO;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace soma_test {

inline int fences_seen = 0;

inline void count_fence(const char*, std::uint32_t, std::uint64_t*) { ++fences_seen; }

// Number of Kokkos fences issued while `op` runs.
template <class Op>
int fences_during(Op&& op) {
  Kokkos::Tools::Experimental::set_begin_fence_callback(&count_fence);
  fences_seen = 0;
  op();
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  return fences_seen;
}

}  // namespace soma_test

namespace soma_test {

// A memory space the host can access that is not HostSpace, standing in for
// managed or unified memory (not available in this build).
struct ManagedLikeSpace {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = ManagedLikeSpace;
  using device_type = Kokkos::Device<execution_space, memory_space>;
  static constexpr const char* name() { return "ManagedLike"; }
};

}  // namespace soma_test

namespace Kokkos {
template <>
struct SpaceAccessibility<Kokkos::HostSpace, soma_test::ManagedLikeSpace> {
  enum : bool { assignable = false, accessible = true, deepcopy = true };
};
}  // namespace Kokkos

namespace soma_test {

using soma::detail::RowWritePath;
static_assert(soma::detail::write_row_path<Kokkos::HostSpace>() == RowWritePath::direct);
static_assert(soma::detail::write_row_path<ManagedLikeSpace>() == RowWritePath::fenced_direct);
static_assert(soma::detail::write_row_path<Kokkos::DefaultExecutionSpace::memory_space>() !=
              RowWritePath::staged);

}  // namespace soma_test

namespace {

using Reg = soma::FieldRegistry<HostMem>;
using Dom = soma::Domain<HostMem>;
using Spc = soma::Space<HostMem>;

Spc subset(const Reg& reg, int dim, LO count, std::vector<LO> lids) {
  return Spc(Dom::from_list(dim, count, reg.epoch(), lids), Layout::per_entity());
}

template <class T>
void put(Reg& reg, soma::FieldId id, LO lid, std::vector<T> v) {
  reg.set_pending<T>(id, lid, v);
}

}  // namespace

TEST(FieldCommit, WriteToAnExistingMemberGoesStraightToTheData) {
  Reg reg;
  reg.set_entities(2, 30);
  auto f = reg.add<double>("q", subset(reg, 2, 30, {3, 8, 20}), 1);
  put<double>(reg, f.id(), 8, {4.5});
  EXPECT_EQ(reg.pending_count(f.id()), 0u);
  EXPECT_DOUBLE_EQ(f.view()(1, 0), 4.5);
}

TEST(FieldCommit, WriteToANonMemberIsBufferedUntilCommit) {
  Reg reg;
  reg.set_entities(2, 30);
  auto f = reg.add<double>("q", subset(reg, 2, 30, {3, 8, 20}), 2);
  put<double>(reg, f.id(), 12, {1.0, 2.0});
  put<double>(reg, f.id(), 5, {3.0, 4.0});
  put<double>(reg, f.id(), 12, {5.0, 6.0});
  EXPECT_EQ(reg.pending_count(f.id()), 2u);
  EXPECT_EQ(f.rows(), 3);
  EXPECT_EQ(reg.get<double>(f.id()).rows(), 3);
}

TEST(FieldCommit, CommitMergesIntoANewDomainSortedByEntity) {
  Reg reg;
  reg.set_entities(2, 30);
  auto f = reg.add<double>("q", subset(reg, 2, 30, {3, 8, 20}), 2);
  f.view()(0, 0) = 30.0;
  f.view()(1, 1) = 81.0;
  f.view()(2, 0) = 200.0;
  put<double>(reg, f.id(), 12, {12.0, 12.5});
  put<double>(reg, f.id(), 5, {5.0, 5.5});
  put<double>(reg, f.id(), 25, {25.0, 25.5});
  reg.commit(f.id());
  EXPECT_EQ(reg.pending_count(f.id()), 0u);

  const auto g = reg.get<double>(f.id());
  EXPECT_EQ(g.id(), f.id());
  EXPECT_EQ(g.rows(), 6);
  EXPECT_EQ(g.domain().members(0), (std::vector<LO>{3, 5, 8, 12, 20, 25}));
  EXPECT_DOUBLE_EQ(g.view()(0, 0), 30.0);
  EXPECT_DOUBLE_EQ(g.view()(1, 0), 5.0);
  EXPECT_DOUBLE_EQ(g.view()(1, 1), 5.5);
  EXPECT_DOUBLE_EQ(g.view()(2, 1), 81.0);
  EXPECT_DOUBLE_EQ(g.view()(3, 1), 12.5);
  EXPECT_DOUBLE_EQ(g.view()(4, 0), 200.0);
  EXPECT_DOUBLE_EQ(g.view()(5, 0), 25.0);
  EXPECT_EQ(g.domain().row(2, 12), 3);
  EXPECT_GT(g.revision(), f.revision());
  EXPECT_EQ(reg.info(f.id()).rows, 6);
}

TEST(FieldCommit, CommitKeepsTheMeshEpochAndTheFieldId) {
  Reg reg;
  reg.set_entities(2, 30);
  reg.set_epoch(4);
  auto f = reg.add<double>("q", subset(reg, 2, 30, {1}), 1);
  const auto other = reg.add_tag<double>("other", 2, 1);
  put<double>(reg, f.id(), 2, {1.0});
  reg.commit(f.id());
  EXPECT_EQ(reg.epoch(), 4u);
  const auto g = reg.get<double>("q");
  EXPECT_EQ(g.id(), f.id());
  EXPECT_EQ(g.domain().epoch(), 4u);
  EXPECT_EQ(reg.get<double>(other.id()).id(), other.id());
}

TEST(FieldCommit, CommitIsIdempotentAndCheapWhenNothingIsPending) {
  Reg reg;
  reg.set_entities(2, 30);
  auto f = reg.add<double>("q", subset(reg, 2, 30, {3, 8}), 1);
  reg.commit(f.id());
  reg.commit_all();
  const auto same = reg.get<double>(f.id());
  EXPECT_EQ(same.view().data(), f.view().data());
  EXPECT_EQ(same.revision(), 0u);
  put<double>(reg, f.id(), 9, {1.0});
  reg.commit(f.id());
  const auto once = reg.get<double>(f.id());
  reg.commit(f.id());
  EXPECT_EQ(reg.get<double>(f.id()).view().data(), once.view().data());
  EXPECT_EQ(reg.get<double>(f.id()).revision(), 1u);
}

TEST(FieldCommit, CommitAllMergesEveryField) {
  Reg reg;
  reg.set_entities(2, 30);
  reg.set_entities(3, 10);
  const auto a = reg.add<double>("a", subset(reg, 2, 30, {1}), 1);
  const auto b = reg.add<std::int32_t>("b", subset(reg, 3, 10, {2}), 1);
  put<double>(reg, a.id(), 4, {1.0});
  put<std::int32_t>(reg, b.id(), 5, {7});
  reg.commit_all();
  EXPECT_EQ(reg.pending_count(a.id()), 0u);
  EXPECT_EQ(reg.pending_count(b.id()), 0u);
  EXPECT_EQ(reg.get<double>(a.id()).rows(), 2);
  EXPECT_EQ(reg.get<std::int32_t>(b.id()).view()(1, 0), 7);
}

TEST(FieldCommit, DefaultsApplyToNonMembersAndMembersCreatedByCommit) {
  Reg reg;
  reg.set_entities(2, 30);
  soma::FieldOptions<double> opt;
  opt.default_value = std::vector<double>{-1.0};
  auto f = reg.add<double>("q", subset(reg, 2, 30, {3}), 1, opt);
  put<double>(reg, f.id(), 7, {2.0});
  reg.commit(f.id());
  const auto g = reg.get<double>(f.id());
  EXPECT_TRUE(g.has_default());
  EXPECT_DOUBLE_EQ(g.value_or_default(2, 7), 2.0);
  EXPECT_DOUBLE_EQ(g.value_or_default(2, 3), -1.0);
  EXPECT_DOUBLE_EQ(g.value_or_default(2, 9), -1.0);
}

// The second field cannot buffer entity 3 while the first waits to commit it, so
// no commit ever finds an overlap; what the second field buffered earlier stays.
TEST(FieldCommit, ARejectedBufferWriteKeepsTheEarlierOnesAndEveryCommitSucceeds) {
  Reg reg;
  reg.set_entities(2, 30);
  auto a = reg.add<double>("T", subset(reg, 2, 30, {1, 2}), 1);
  auto b = reg.add<double>("T", subset(reg, 2, 30, {10, 11}), 1);
  b.view()(0, 0) = 10.5;
  put<double>(reg, a.id(), 3, {3.25});
  put<double>(reg, b.id(), 20, {20.5});
  EXPECT_THROW(put<double>(reg, b.id(), 3, {3.5}), soma::FieldOverlap);
  EXPECT_EQ(reg.pending_count(b.id()), 1u);
  reg.commit(a.id());
  EXPECT_NO_THROW(reg.commit(b.id()));
  const auto kept = reg.get<double>(b.id());
  EXPECT_EQ(kept.domain().members(0), (std::vector<LO>{10, 11, 20}));
  EXPECT_DOUBLE_EQ(kept.view()(0, 0), 10.5);
  EXPECT_DOUBLE_EQ(kept.view()(2, 0), 20.5);
  const auto fam = reg.family<double>("T", 1);
  EXPECT_EQ(reg.lookup(fam, 2, 3).slot, 0);
  EXPECT_EQ(reg.lookup(fam, 2, 20).slot, 1);
}

TEST(FieldCommit, OverlapIsReRunForTheAffectedMemberOnly) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto a = reg.add<double>("T", subset(reg, 2, 30, {1, 2, 3}), 1);
  const auto b = reg.add<double>("T", subset(reg, 2, 30, {10, 11}), 1);
  put<double>(reg, b.id(), 20, {1.0});
  EXPECT_NO_THROW(reg.commit(b.id()));
  put<double>(reg, a.id(), 21, {1.0});
  EXPECT_NO_THROW(reg.commit(a.id()));
  const auto fam = reg.family<double>("T", 1);
  EXPECT_TRUE(reg.lookup(fam, 2, 20).found());
  EXPECT_TRUE(reg.lookup(fam, 2, 21).found());
  EXPECT_EQ(reg.lookup(fam, 2, 20).slot, 1);
  EXPECT_EQ(reg.lookup(fam, 2, 21).slot, 0);
}

TEST(FieldCommit, FamilyLookupSeesCommittedMembers) {
  Reg reg;
  reg.set_entities(2, 30);
  auto f = reg.add<double>("q", subset(reg, 2, 30, {3, 8}), 1);
  const auto fam = reg.family<double>("q", 1);
  EXPECT_FALSE(reg.lookup(fam, 2, 5).found());
  put<double>(reg, f.id(), 5, {1.0});
  EXPECT_FALSE(reg.lookup(fam, 2, 5).found());
  reg.commit(f.id());
  EXPECT_EQ(reg.lookup(fam, 2, 5).row, 1);
  EXPECT_EQ(reg.lookup(fam, 2, 8).row, 2);
}

TEST(FieldCommit, WritesToFullFieldsAreDataWrites) {
  Reg reg;
  reg.set_entities(2, 5);
  auto f = reg.add_tag<double>("full", 2, 1);
  put<double>(reg, f.id(), 3, {9.0});
  EXPECT_EQ(reg.pending_count(f.id()), 0u);
  EXPECT_DOUBLE_EQ(f.view()(3, 0), 9.0);
  reg.commit(f.id());
  EXPECT_EQ(reg.get<double>(f.id()).view().data(), f.view().data());
}

TEST(FieldCommit, BadWritesAreRejected) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto f = reg.add<double>("q", subset(reg, 2, 30, {3}), 2);
  EXPECT_THROW(put<float>(reg, f.id(), 4, {1.0F, 2.0F}), std::invalid_argument);
  EXPECT_THROW(put<double>(reg, f.id(), 4, {1.0}), std::invalid_argument);
  EXPECT_THROW(put<double>(reg, f.id(), 30, {1.0, 2.0}), std::out_of_range);
  EXPECT_THROW(put<double>(reg, f.id(), -1, {1.0, 2.0}), std::out_of_range);
  EXPECT_THROW(put<double>(reg, 99, 4, {1.0, 2.0}), std::out_of_range);
  EXPECT_EQ(reg.pending_count(f.id()), 0u);
}

TEST(FieldCommit, PendingWritesNeedAPerEntityFieldOnOneDimension) {
  Reg reg;
  reg.set_entities(2, 4);
  reg.set_entities(3, 4);
  const auto both = reg.add<double>(
      "both", Spc(Dom::combine(Dom::all(2, 4, 1), Dom::all(3, 4, 1)), Layout::per_entity()), 1);
  EXPECT_THROW(put<double>(reg, both.id(), 1, {1.0}), std::invalid_argument);
}

TEST(FieldCommit, StaleFieldsRejectWritesAndCommit) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto f = reg.add<double>("q", subset(reg, 2, 30, {3}), 1);
  put<double>(reg, f.id(), 5, {1.0});
  reg.set_epoch(reg.epoch() + 1);
  EXPECT_THROW(put<double>(reg, f.id(), 6, {1.0}), soma::StaleField);
  EXPECT_THROW(reg.commit(f.id()), soma::StaleField);
  EXPECT_EQ(reg.pending_count(f.id()), 1u);
  soma::FieldRegistry<HostMem> dst;
  EXPECT_THROW(reg.mirror_into<HostMem>(dst, {"*"}, {true, true, true, true}, false),
               soma::StaleField);
  reg.set_epoch(reg.epoch() - 1);
  EXPECT_NO_THROW(reg.commit(f.id()));
  EXPECT_EQ(reg.get<double>(f.id()).rows(), 2);
}

// The write to a stale field must be refused by the currency check itself, not
// only by the family lookup that follows it, so the buffered entity here is
// already pending: no lookup is needed for it.
TEST(FieldCommit, AStaleFieldRefusesAWriteToAnEntityThatIsAlreadyPending) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto f = reg.add<double>("q", subset(reg, 2, 30, {3}), 1);
  put<double>(reg, f.id(), 5, {1.0});
  reg.set_epoch(reg.epoch() + 1);
  EXPECT_THROW(put<double>(reg, f.id(), 5, {2.0}), soma::StaleField);
  EXPECT_THROW(put<double>(reg, f.id(), 3, {2.0}), soma::StaleField);
  reg.set_epoch(reg.epoch() - 1);
  EXPECT_NO_THROW(put<double>(reg, f.id(), 5, {2.0}));
  EXPECT_EQ(reg.pending_count(f.id()), 1u);
}

TEST(FieldCommit, StaleFieldsAreRejectedEvenWithNothingPending) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto f = reg.add<double>("q", subset(reg, 2, 30, {3}), 1);
  const auto g = reg.add_tag<double>("full", 2, 1);
  reg.set_epoch(reg.epoch() + 1);
  EXPECT_THROW(reg.commit(f.id()), soma::StaleField);
  EXPECT_THROW(reg.commit(g.id()), soma::StaleField);
  EXPECT_THROW(reg.commit_all(), soma::StaleField);
  soma::FieldRegistry<HostMem> everything;
  EXPECT_THROW(reg.mirror_into<HostMem>(everything, {"*"}, {true, true, true, true}, false),
               soma::StaleField);
  soma::FieldRegistry<HostMem> named;
  EXPECT_THROW(reg.mirror_into<HostMem>(named, {"q"}, {true, true, true, true}, true),
               soma::StaleField);
  reg.set_epoch(reg.epoch() - 1);
  soma::FieldRegistry<HostMem> ok;
  EXPECT_NO_THROW(reg.mirror_into<HostMem>(ok, {"*"}, {true, true, true, true}, false));
  EXPECT_TRUE(ok.contains("q"));
}

TEST(FieldCommit, AStaleFieldThatIsNotSelectedDoesNotBlockAMirror) {
  Reg reg;
  reg.set_entities(2, 30);
  reg.set_entities(3, 10);
  reg.add_tag<double>("cells", 3, 1);
  reg.set_epoch(reg.epoch() + 1);
  const auto fresh = reg.add_tag<double>("fresh", 3, 1);
  soma::FieldRegistry<HostMem> dst;
  EXPECT_NO_THROW(reg.mirror_into<HostMem>(dst, {"fresh"}, {true, true, true, true}, false));
  EXPECT_TRUE(dst.contains("fresh"));
  EXPECT_FALSE(dst.contains("cells"));
  EXPECT_EQ(dst.get<double>(fresh.id()).id(), fresh.id());
}

TEST(FieldCommit, WriteToAnEntityOfAnotherMemberIsRejectedAtWriteTime) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto a = reg.add<double>("T", subset(reg, 2, 30, {1, 2, 3}), 1);
  const auto b = reg.add<double>("T", subset(reg, 2, 30, {10, 11}), 1);
  try {
    put<double>(reg, b.id(), 3, {3.5});
    FAIL() << "expected FieldOverlap";
  } catch (const soma::FieldOverlap& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("cannot take entity lid 3 in dim 2"), std::string::npos) << msg;
    EXPECT_NE(msg.find("belongs to existing 'T'"), std::string::npos) << msg;
  }
  EXPECT_EQ(reg.pending_count(b.id()), 0u);
  EXPECT_THROW(put<double>(reg, a.id(), 10, {1.0}), soma::FieldOverlap);
  put<double>(reg, b.id(), 20, {2.0});
  EXPECT_EQ(reg.pending_count(b.id()), 1u);
  put<double>(reg, b.id(), 20, {2.5});
  EXPECT_EQ(reg.pending_count(b.id()), 1u);
  EXPECT_NO_THROW(reg.commit(b.id()));
  EXPECT_EQ(reg.get<double>(b.id()).domain().members(0), (std::vector<LO>{10, 11, 20}));
}

// Both members buffer the same non-member entity: whichever commits second
// would overlap, and its buffer would block every later mirror.
TEST(FieldCommit, TwoMembersCannotBufferTheSameNonMemberEntity) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto a = reg.add<double>("T", subset(reg, 2, 30, {1, 2}), 1);
  const auto b = reg.add<double>("T", subset(reg, 2, 30, {20, 21}), 1);
  put<double>(reg, a.id(), 10, {1.0});
  try {
    put<double>(reg, b.id(), 10, {2.0});
    FAIL() << "expected FieldOverlap";
  } catch (const soma::FieldOverlap& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("pending write"), std::string::npos) << msg;
    EXPECT_NE(msg.find("lid 10"), std::string::npos) << msg;
    EXPECT_NE(msg.find("'T'"), std::string::npos) << msg;
  }
  EXPECT_EQ(reg.pending_count(b.id()), 0u);
  EXPECT_EQ(reg.pending_count(a.id()), 1u);
  EXPECT_NO_THROW(reg.commit_all());
  soma::FieldRegistry<HostMem> dst;
  EXPECT_NO_THROW(reg.mirror_into<HostMem>(dst, {"*"}, {true, true, true, true}, false));
  EXPECT_EQ(reg.get<double>(a.id()).domain().members(0), (std::vector<LO>{1, 2, 10}));
  EXPECT_EQ(reg.get<double>(b.id()).domain().members(0), (std::vector<LO>{20, 21}));
}

TEST(FieldCommit, RewritingAnEntityPendingInTheSameFieldIsFine) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto a = reg.add<double>("T", subset(reg, 2, 30, {1, 2}), 1);
  reg.add<double>("T", subset(reg, 2, 30, {20, 21}), 1);
  put<double>(reg, a.id(), 10, {1.0});
  EXPECT_NO_THROW(put<double>(reg, a.id(), 10, {2.0}));
  EXPECT_EQ(reg.pending_count(a.id()), 1u);
  reg.commit(a.id());
  EXPECT_DOUBLE_EQ(reg.get<double>(a.id()).view()(2, 0), 2.0);
}

TEST(FieldCommit, ABufferedEntityIsFreeAgainOnceTheOtherFieldDiscardsIt) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto a = reg.add<double>("T", subset(reg, 2, 30, {1, 2}), 1);
  const auto b = reg.add<double>("T", subset(reg, 2, 30, {20, 21}), 1);
  put<double>(reg, a.id(), 10, {1.0});
  EXPECT_THROW(put<double>(reg, b.id(), 10, {2.0}), soma::FieldOverlap);
  reg.discard_pending(a.id());
  EXPECT_NO_THROW(put<double>(reg, b.id(), 10, {2.0}));
  EXPECT_THROW(put<double>(reg, a.id(), 10, {1.0}), soma::FieldOverlap);
  reg.commit(b.id());
  EXPECT_THROW(put<double>(reg, a.id(), 10, {1.0}), soma::FieldOverlap) << "now a member of b";
}

TEST(FieldCommit, PendingEntitiesOfAnotherDimensionOrFamilyDoNotConflict) {
  Reg reg;
  reg.set_entities(2, 30);
  reg.set_entities(3, 30);
  const auto a = reg.add<double>("T", subset(reg, 2, 30, {1, 2}), 1);
  const auto other_dim = reg.add<double>("T", subset(reg, 3, 30, {20}), 1);
  const auto other_name = reg.add<double>("U", subset(reg, 2, 30, {20}), 1);
  const auto other_type = reg.add<float>("T", subset(reg, 2, 30, {20}), 1);
  put<double>(reg, a.id(), 10, {1.0});
  EXPECT_NO_THROW(put<double>(reg, other_dim.id(), 10, {1.0}));
  EXPECT_NO_THROW(put<double>(reg, other_name.id(), 10, {1.0}));
  EXPECT_NO_THROW(reg.set_pending<float>(other_type.id(), 10, std::vector<float>{1.0F}));
  EXPECT_NO_THROW(reg.commit_all());
}

TEST(FieldCommit, ThreeMembersOfOneFamilyAreAllCheckedAgainstEachOther) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto a = reg.add<double>("T", subset(reg, 2, 30, {1}), 1);
  const auto b = reg.add<double>("T", subset(reg, 2, 30, {5}), 1);
  const auto c = reg.add<double>("T", subset(reg, 2, 30, {9}), 1);
  put<double>(reg, b.id(), 10, {1.0});
  EXPECT_THROW(put<double>(reg, a.id(), 10, {1.0}), soma::FieldOverlap);
  EXPECT_THROW(put<double>(reg, c.id(), 10, {1.0}), soma::FieldOverlap);
  EXPECT_NO_THROW(put<double>(reg, c.id(), 11, {1.0}));
  EXPECT_NO_THROW(reg.commit_all());
}

TEST(FieldCommit, ARejectedWriteDoesNotPoisonMirrorOrClone) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto a = reg.add<double>("T", subset(reg, 2, 30, {1, 2, 3}), 1);
  const auto b = reg.add<double>("T", subset(reg, 2, 30, {10, 11}), 1);
  EXPECT_THROW(put<double>(reg, b.id(), 3, {3.5}), soma::FieldOverlap);
  soma::FieldRegistry<HostMem> dst;
  EXPECT_NO_THROW(reg.mirror_into<HostMem>(dst, {"*"}, {true, true, true, true}, false));
  EXPECT_EQ(dst.get<double>(b.id()).rows(), 2);
  EXPECT_NO_THROW(reg.commit_all());
  (void)a;
}

TEST(FieldCommit, WritesToAnEntityOfAFamilyMemberOnAnotherDimensionAreFine) {
  Reg reg;
  reg.set_entities(2, 30);
  reg.set_entities(3, 30);
  reg.add<double>("T", subset(reg, 3, 30, {3}), 1);
  const auto b = reg.add<double>("T", subset(reg, 2, 30, {10}), 1);
  EXPECT_NO_THROW(put<double>(reg, b.id(), 3, {1.0}));
  EXPECT_EQ(reg.pending_count(b.id()), 1u);
}

TEST(FieldCommit, DiscardPendingDropsBufferedWritesOnly) {
  Reg reg;
  reg.set_entities(2, 30);
  auto f = reg.add<double>("q", subset(reg, 2, 30, {3, 8}), 1);
  put<double>(reg, f.id(), 8, {8.5});
  put<double>(reg, f.id(), 12, {1.0});
  put<double>(reg, f.id(), 13, {2.0});
  ASSERT_EQ(reg.pending_count(f.id()), 2u);
  reg.discard_pending(f.id());
  EXPECT_EQ(reg.pending_count(f.id()), 0u);
  reg.commit(f.id());
  const auto g = reg.get<double>(f.id());
  EXPECT_EQ(g.rows(), 2);
  EXPECT_EQ(g.view().data(), f.view().data());
  EXPECT_DOUBLE_EQ(g.view()(1, 0), 8.5);
  put<double>(reg, f.id(), 12, {3.0});
  reg.commit(f.id());
  EXPECT_EQ(reg.get<double>(f.id()).domain().members(0), (std::vector<LO>{3, 8, 12}));
  EXPECT_THROW(reg.discard_pending(99), std::out_of_range);
}

TEST(FieldCommit, AHandleHeldAcrossACommitIsNoLongerCurrent) {
  Reg reg;
  reg.set_entities(2, 30);
  auto stale = reg.add<double>("q", subset(reg, 2, 30, {3, 8}), 1);
  EXPECT_TRUE(reg.is_current(stale));
  put<double>(reg, stale.id(), 5, {1.0});
  EXPECT_TRUE(reg.is_current(stale)) << "buffering does not replace the storage";
  reg.commit(stale.id());
  EXPECT_FALSE(reg.is_current(stale));
  const auto fresh = reg.get<double>(stale.id());
  EXPECT_TRUE(reg.is_current(fresh));
  EXPECT_FALSE(reg.is_current(stale));
}

TEST(FieldCommit, AHandleOnAFieldWithoutRowsGoesStaleAtCommitToo) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto empty = reg.add<double>("q", subset(reg, 2, 30, {}), 1);
  ASSERT_EQ(empty.rows(), 0);
  EXPECT_TRUE(reg.is_current(empty));
  put<double>(reg, empty.id(), 5, {1.0});
  reg.commit(empty.id());
  EXPECT_FALSE(reg.is_current(empty));
  EXPECT_TRUE(reg.is_current(reg.get<double>(empty.id())));
}

TEST(FieldCommit, ACommitWithNothingPendingKeepsHandlesCurrent) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto f = reg.add<double>("q", subset(reg, 2, 30, {3, 8}), 1);
  reg.commit(f.id());
  reg.commit_all();
  EXPECT_TRUE(reg.is_current(f));
}

TEST(FieldCommit, WritesThroughAStaleHandleAreLostAndIsCurrentSaysSo) {
  Reg reg;
  reg.set_entities(2, 30);
  auto stale = reg.add<double>("q", subset(reg, 2, 30, {3, 8}), 1);
  put<double>(reg, stale.id(), 5, {1.0});
  reg.commit(stale.id());
  ASSERT_FALSE(reg.is_current(stale));
  stale.view()(0, 0) = 99.0;
  EXPECT_DOUBLE_EQ(reg.get<double>(stale.id()).view()(0, 0), 0.0);
}

TEST(FieldCommit, IsCurrentIsFalseAfterRemoveAndForUnknownFields) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto f = reg.add<double>("q", subset(reg, 2, 30, {3}), 1);
  const auto g = reg.add<double>("g", subset(reg, 2, 30, {4}), 1);
  reg.remove(f.id());
  EXPECT_FALSE(reg.is_current(f));
  EXPECT_TRUE(reg.is_current(g));
  EXPECT_FALSE(reg.is_current(soma::Field<double, HostMem>()));
  EXPECT_FALSE(reg.current_revision(f.id()).has_value());
  EXPECT_FALSE(reg.current_revision(99).has_value());
}

TEST(FieldCommit, ACurrentHandleOfAnotherRegistryIsNotCurrentHere) {
  Reg a;
  Reg b;
  a.set_entities(2, 30);
  b.set_entities(2, 30);
  const auto fa = a.add<double>("q", subset(a, 2, 30, {3}), 1);
  const auto fb = b.add<double>("q", subset(b, 2, 30, {3}), 1);
  ASSERT_EQ(fa.id(), fb.id());
  EXPECT_TRUE(a.is_current(fa));
  EXPECT_FALSE(a.is_current(fb));
}

TEST(FieldCommit, CurrentRevisionFollowsCommits) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto f = reg.add<double>("q", subset(reg, 2, 30, {3}), 1);
  EXPECT_EQ(reg.current_revision(f.id()), std::optional<std::uint64_t>(0));
  put<double>(reg, f.id(), 5, {1.0});
  reg.commit(f.id());
  EXPECT_EQ(reg.current_revision(f.id()), std::optional<std::uint64_t>(1));
  EXPECT_EQ(reg.current_revision(f.id()), reg.get<double>(f.id()).revision());
}

TEST(FieldCommit, IsCurrentDoesNotJudgeTheMeshEpoch) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto f = reg.add<double>("q", subset(reg, 2, 30, {3}), 1);
  reg.set_epoch(reg.epoch() + 1);
  EXPECT_TRUE(reg.is_current(f));
  EXPECT_THROW(reg.get<double>(f.id()), soma::StaleField);
}

template <class M>
class MemberWriteTyped : public ::testing::Test {};
using Spaces = ::testing::Types<HostMem, DevMem>;
TYPED_TEST_SUITE(MemberWriteTyped, Spaces);

TYPED_TEST(MemberWriteTyped, EveryComponentOfAMemberRowIsWrittenAndNothingElse) {
  using R = soma::FieldRegistry<TypeParam>;
  using D = soma::Domain<TypeParam>;
  for (int ncomp = 1; ncomp <= 3; ++ncomp) {
    R reg;
    reg.set_entities(2, 30);
    const std::vector<LO> lids{3, 8, 20};
    soma::FieldOptions<double> opt;
    opt.default_value = std::vector<double>(static_cast<std::size_t>(ncomp), -1.0);
    const auto f = reg.template add<double>(
        "q", soma::Space<TypeParam>(D::from_list(2, 30, reg.epoch(), lids), Layout::per_entity()),
        ncomp, opt);
    std::vector<double> v(static_cast<std::size_t>(ncomp));
    for (int c = 0; c < ncomp; ++c) v[static_cast<std::size_t>(c)] = 10.0 * (c + 1) + 0.5;
    reg.template set_pending<double>(f.id(), 8, v);
    EXPECT_EQ(reg.pending_count(f.id()), 0u);
    const auto h = Kokkos::create_mirror_view_and_copy(HostMem{}, f.view());
    for (int c = 0; c < ncomp; ++c) {
      const auto cc = static_cast<std::size_t>(c);
      EXPECT_DOUBLE_EQ(h(0, cc), -1.0) << ncomp << " comp " << c;
      EXPECT_DOUBLE_EQ(h(1, cc), v[cc]) << ncomp << " comp " << c;
      EXPECT_DOUBLE_EQ(h(2, cc), -1.0) << ncomp << " comp " << c;
    }
  }
}

TYPED_TEST(MemberWriteTyped, IntegerAndFloatFieldsKeepTheirValues) {
  using R = soma::FieldRegistry<TypeParam>;
  using D = soma::Domain<TypeParam>;
  R reg;
  reg.set_entities(2, 30);
  const auto make = [&](const std::vector<LO>& l) {
    return soma::Space<TypeParam>(D::from_list(2, 30, reg.epoch(), l), Layout::per_entity());
  };
  const auto i = reg.template add<std::int32_t>("i", make({1, 2}), 2);
  const auto fl = reg.template add<float>("f", make({1, 2}), 3);
  reg.template set_pending<std::int32_t>(i.id(), 2, std::vector<std::int32_t>{-7, 9});
  reg.template set_pending<float>(fl.id(), 1, std::vector<float>{0.25F, 0.5F, 0.75F});
  const auto hi = Kokkos::create_mirror_view_and_copy(HostMem{}, i.view());
  const auto hf = Kokkos::create_mirror_view_and_copy(HostMem{}, fl.view());
  EXPECT_EQ(hi(1, 0), -7);
  EXPECT_EQ(hi(1, 1), 9);
  EXPECT_EQ(hi(0, 0), 0);
  EXPECT_FLOAT_EQ(hf(0, 2), 0.75F);
  EXPECT_FLOAT_EQ(hf(1, 2), 0.0F);
}

// A write to a host-accessible field is plain assignment: it must not fence the
// execution space (a fence per component is what made writes cost tens of
// microseconds with threads).
TEST(FieldCommit, AHostWriteToAMemberRowIssuesNoFence) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto f = reg.add<double>("q", subset(reg, 2, 30, {3, 8}), 3);
  const std::vector<double> v{1.0, 2.0, 3.0};
  const int fences = soma_test::fences_during([&] { reg.set_pending<double>(f.id(), 8, v); });
  EXPECT_EQ(fences, 0);
  EXPECT_DOUBLE_EQ(f.view()(1, 2), 3.0);
}

// Each path run on HostSpace memory (the only memory these tests can write
// directly), so the fences each one issues can be counted.
TEST(FieldCommit, TheFencedDirectPathFencesOnceNotPerComponent) {
  soma::View2<double, HostMem> data("soma::test::rows", 4, 3);
  Kokkos::deep_copy(data, -1.0);
  const std::vector<double> v{1.0, 2.0, 3.0};
  using soma::detail::RowWritePath;
  const int fenced = soma_test::fences_during([&] {
    soma::detail::write_row_via<RowWritePath::fenced_direct, double, HostMem>(data, 2, v);
  });
  EXPECT_EQ(fenced, 1);
  EXPECT_DOUBLE_EQ(data(2, 0), 1.0);
  EXPECT_DOUBLE_EQ(data(2, 2), 3.0);
  EXPECT_DOUBLE_EQ(data(1, 0), -1.0);
  const int direct = soma_test::fences_during([&] {
    soma::detail::write_row_via<RowWritePath::direct, double, HostMem>(data, 3, v);
  });
  EXPECT_EQ(direct, 0);
  EXPECT_DOUBLE_EQ(data(3, 1), 2.0);
}

TYPED_TEST(MemberWriteTyped, TheStagedPathWritesTheSameRow) {
  for (std::size_t ncomp = 1; ncomp <= 3; ++ncomp) {
    soma::View2<double, TypeParam> data("soma::test::rows", 4, ncomp);
    Kokkos::deep_copy(data, -1.0);
    std::vector<double> v(ncomp);
    for (std::size_t c = 0; c < ncomp; ++c) v[c] = 5.0 + static_cast<double>(c);
    soma::detail::write_row_staged<double, TypeParam>(data, 2, v);
    const auto h = Kokkos::create_mirror_view_and_copy(HostMem{}, data);
    for (std::size_t r = 0; r < 4; ++r) {
      for (std::size_t c = 0; c < ncomp; ++c) {
        EXPECT_DOUBLE_EQ(h(r, c), r == 2 ? v[c] : -1.0) << ncomp << " " << r << " " << c;
      }
    }
  }
}

template <class M>
class CommitMirrorTyped : public ::testing::Test {};
using Spaces = ::testing::Types<HostMem, DevMem>;
TYPED_TEST_SUITE(CommitMirrorTyped, Spaces);

TYPED_TEST(CommitMirrorTyped, MirrorCommitsPendingWritesFirst) {
  Reg reg;
  reg.set_entities(2, 30);
  auto f = reg.add<double>("q", subset(reg, 2, 30, {3, 8}), 1);
  f.view()(0, 0) = 3.0;
  f.view()(1, 0) = 8.0;
  put<double>(reg, f.id(), 5, {5.0});
  ASSERT_EQ(reg.pending_count(f.id()), 1u);

  soma::FieldRegistry<TypeParam> dst;
  reg.mirror_into<TypeParam>(dst, {"*"}, {true, true, true, true}, false);
  EXPECT_EQ(reg.pending_count(f.id()), 0u);
  const auto m = dst.template get<double>("q");
  EXPECT_EQ(m.rows(), 3);
  EXPECT_EQ(m.domain().members(0), (std::vector<LO>{3, 5, 8}));
  const auto h = Kokkos::create_mirror_view_and_copy(HostMem{}, m.view());
  EXPECT_DOUBLE_EQ(h(0, 0), 3.0);
  EXPECT_DOUBLE_EQ(h(1, 0), 5.0);
  EXPECT_DOUBLE_EQ(h(2, 0), 8.0);
}

TYPED_TEST(CommitMirrorTyped, MirrorCommitsOnlyTheSelectedFields) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto a = reg.add<double>("a", subset(reg, 2, 30, {3}), 1);
  const auto b = reg.add<double>("b", subset(reg, 2, 30, {4}), 1);
  put<double>(reg, a.id(), 6, {1.0});
  put<double>(reg, b.id(), 7, {1.0});
  soma::FieldRegistry<TypeParam> dst;
  reg.mirror_into<TypeParam>(dst, {"a"}, {true, true, true, true}, false);
  EXPECT_EQ(dst.template get<double>("a").rows(), 2);
  EXPECT_FALSE(dst.contains("b"));
  EXPECT_EQ(reg.pending_count(a.id()), 0u);
  EXPECT_EQ(reg.pending_count(b.id()), 1u) << "an unselected field keeps its buffer";
  EXPECT_EQ(reg.get<double>(b.id()).rows(), 1);
}

// A field that went stale with a write pending cannot be committed. A mirror
// that does not select it must not be affected, and one that does must throw.
TYPED_TEST(CommitMirrorTyped, AFailingCommitOfAnUnselectedFieldDoesNotBreakAMirror) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto old = reg.add<double>("old", subset(reg, 2, 30, {1}), 1);
  put<double>(reg, old.id(), 3, {1.0});
  reg.set_epoch(reg.epoch() + 1);
  const auto fresh = reg.add<double>("fresh", subset(reg, 2, 30, {5}), 1);
  put<double>(reg, fresh.id(), 6, {3.0});
  soma::FieldRegistry<TypeParam> dst;
  EXPECT_NO_THROW(reg.mirror_into<TypeParam>(dst, {"fresh"}, {true, true, true, true}, false));
  EXPECT_EQ(dst.template get<double>("fresh").rows(), 2);
  EXPECT_FALSE(dst.contains("old"));
  EXPECT_EQ(reg.pending_count(old.id()), 1u);

  soma::FieldRegistry<TypeParam> all;
  EXPECT_THROW(reg.mirror_into<TypeParam>(all, {"*"}, {true, true, true, true}, false),
               soma::StaleField);
}

TYPED_TEST(CommitMirrorTyped, AFieldSwitchedOffByDimensionIsNotCommitted) {
  Reg reg;
  reg.set_entities(2, 30);
  reg.set_entities(3, 30);
  const auto f2 = reg.add<double>("f2", subset(reg, 2, 30, {3}), 1);
  const auto f3 = reg.add<double>("f3", subset(reg, 3, 30, {3}), 1);
  put<double>(reg, f2.id(), 6, {1.0});
  put<double>(reg, f3.id(), 6, {1.0});
  soma::FieldRegistry<TypeParam> dst;
  reg.mirror_into<TypeParam>(dst, {"*"}, {true, true, false, true}, false);
  EXPECT_FALSE(dst.contains("f2"));
  EXPECT_EQ(reg.pending_count(f2.id()), 1u);
  EXPECT_EQ(reg.pending_count(f3.id()), 0u);
}

// A new member may not take an entity that a sibling is waiting to commit:
// whichever commit came first would make the other one impossible.
TEST(FieldCommit, AddRejectsAMemberThatTakesAnEntityAPendingWriteWaitsFor) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto a = reg.add<double>("T", subset(reg, 2, 30, {1, 2}), 1);
  put<double>(reg, a.id(), 20, {1.0});
  try {
    reg.add<double>("T", subset(reg, 2, 30, {20, 21}), 1);
    FAIL() << "expected FieldOverlap";
  } catch (const soma::FieldOverlap& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("pending write"), std::string::npos) << msg;
    EXPECT_NE(msg.find("'T'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("lid 20"), std::string::npos) << msg;
  }
  EXPECT_EQ(reg.list().size(), 1u);
  EXPECT_EQ(reg.family_members(reg.family<double>("T", 1), 2).size(), 1u);
  EXPECT_EQ(reg.pending_count(a.id()), 1u);
  EXPECT_NO_THROW(reg.commit_all());
  EXPECT_EQ(reg.get<double>(a.id()).domain().members(0), (std::vector<LO>{1, 2, 20}));
  soma::FieldRegistry<HostMem> dst;
  EXPECT_NO_THROW(reg.mirror_into<HostMem>(dst, {"*"}, {true, true, true, true}, false));
}

TEST(FieldCommit, AddAcceptsMembersThatAvoidEveryPendingEntity) {
  Reg reg;
  reg.set_entities(2, 30);
  reg.set_entities(3, 30);
  const auto a = reg.add<double>("T", subset(reg, 2, 30, {1, 2}), 1);
  put<double>(reg, a.id(), 20, {1.0});
  put<double>(reg, a.id(), 22, {1.0});
  EXPECT_NO_THROW(reg.add<double>("T", subset(reg, 2, 30, {21, 23}), 1));
  EXPECT_NO_THROW(reg.add<double>("T", subset(reg, 3, 30, {20, 22}), 1));
  EXPECT_NO_THROW(reg.add<double>("U", subset(reg, 2, 30, {20, 22}), 1));
  EXPECT_NO_THROW(reg.add<float>("T", subset(reg, 2, 30, {20, 22}), 1));
  EXPECT_EQ(reg.pending_count(a.id()), 2u);
}

TEST(FieldCommit, AddChecksEveryBlockOfTheNewSpaceAgainstPendingWrites) {
  Reg reg;
  reg.set_entities(2, 30);
  reg.set_entities(3, 30);
  const auto a = reg.add<double>("T", subset(reg, 3, 30, {1}), 1);
  put<double>(reg, a.id(), 20, {1.0});
  const Spc both(Dom::combine(Dom::from_list(2, 30, 1, std::vector<LO>{5}),
                              Dom::from_list(3, 30, 1, std::vector<LO>{20})),
                 Layout::per_entity());
  EXPECT_THROW(reg.add<double>("T", both, 1), soma::FieldOverlap);
}

TEST(FieldCommit, AddIsAcceptedOnceThePendingWriteIsDiscarded) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto a = reg.add<double>("T", subset(reg, 2, 30, {1}), 1);
  put<double>(reg, a.id(), 20, {1.0});
  EXPECT_THROW(reg.add<double>("T", subset(reg, 2, 30, {20}), 1), soma::FieldOverlap);
  reg.discard_pending(a.id());
  EXPECT_NO_THROW(reg.add<double>("T", subset(reg, 2, 30, {20}), 1));
}
