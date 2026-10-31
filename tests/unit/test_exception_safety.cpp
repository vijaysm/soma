// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Fails the n-th heap allocation made while one registry call runs, for every n
// until the call succeeds, and checks that a failed call changed nothing.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/errors.hpp"
#include "soma/fields/field.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/registry.hpp"
#include "soma/fields/space.hpp"

namespace soma_test {

thread_local long allocations_until_failure = -1;  // negative: not armed

struct FailNth {
  explicit FailNth(long n) { allocations_until_failure = n; }
  ~FailNth() { allocations_until_failure = -1; }
  FailNth(const FailNth&) = delete;
  FailNth& operator=(const FailNth&) = delete;
};

constexpr long sweep_limit = 20000;

}  // namespace soma_test

namespace soma_test {

// Kokkos allocates while it releases a View, inside a noexcept destructor, where
// no caller could recover from a failure. Such an allocation is not failed; the
// next one is.
bool inside_kokkos_deallocate() {
  void* frames[32];
  const int depth = backtrace(frames, 32);
  char** symbols = backtrace_symbols(frames, depth);
  bool found = false;
  for (int i = 0; symbols != nullptr && i < depth && !found; ++i) {
    found = std::strstr(symbols[i], "deallocate") != nullptr;
  }
  std::free(symbols);
  return found;
}

}  // namespace soma_test

void* operator new(std::size_t n) {
  if (soma_test::allocations_until_failure == 0 && soma_test::inside_kokkos_deallocate()) {
    // leave the counter at zero: the next allocation outside it fails
  } else if (soma_test::allocations_until_failure >= 0 &&
             soma_test::allocations_until_failure-- == 0) {
    throw std::bad_alloc();
  }
  if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using soma::Layout;
using soma::LO;

using HostMem = Kokkos::HostSpace;
using Reg = soma::FieldRegistry<HostMem>;
using Dom = soma::Domain<HostMem>;
using Spc = soma::Space<HostMem>;

namespace {

Spc subset(const Reg& reg, LO count, std::vector<LO> lids) {
  return Spc(Dom::from_list(2, count, reg.epoch(), lids), Layout::per_entity());
}

// Runs `op` with the n-th allocation failing, for n = 0, 1, ... until it
// succeeds. `after_failure(n)` runs (disarmed) after every failed attempt.
// Returns the number of failed attempts.
template <class Op, class Check>
long sweep(Op&& op, Check&& after_failure) {
  for (long n = 0; n < soma_test::sweep_limit; ++n) {
    bool failed = false;
    {
      const soma_test::FailNth arm(n);
      try {
        op();
      } catch (const std::bad_alloc&) {
        failed = true;
      }
    }
    if (!failed) return n;
    after_failure(n);
    if (::testing::Test::HasFatalFailure()) return n;
  }
  ADD_FAILURE() << "the call still failed after " << soma_test::sweep_limit << " allocations";
  return soma_test::sweep_limit;
}

}  // namespace

TEST(ExceptionSafety, AFailedAddToAnExistingFamilyChangesNothing) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto a = reg.add<double>("T", subset(reg, 30, {1, 2}), 1);
  const auto fam = reg.family<double>("T", 1);
  const Spc space = subset(reg, 30, {10, 11});
  const long failures = sweep(
      [&] { reg.add<double>("T", space, 1); },
      [&](long n) {
        EXPECT_EQ(reg.list().size(), 1u) << "failed allocation " << n;
        EXPECT_EQ(reg.family_members(fam, 2), std::vector<soma::FieldId>{a.id()}) << n;
        EXPECT_FALSE(reg.lookup(fam, 2, 10).found()) << n;
        EXPECT_TRUE(reg.lookup(fam, 2, 2).found()) << n;
      });
  EXPECT_GT(failures, 0);
  EXPECT_EQ(reg.list().size(), 2u);
  EXPECT_EQ(reg.family_members(fam, 2).size(), 2u);
  EXPECT_TRUE(reg.lookup(fam, 2, 10).found());
}

TEST(ExceptionSafety, AFailedAddOfANewFamilyLeavesNoFamilyBehind) {
  Reg reg;
  reg.set_entities(2, 30);
  reg.add<double>("T", subset(reg, 30, {1, 2}), 1);
  const Spc space = subset(reg, 30, {10, 11});
  const long failures = sweep(
      [&] { reg.add<double>("U", space, 1); },
      [&](long n) {
        EXPECT_EQ(reg.list().size(), 1u) << "failed allocation " << n;
        EXPECT_FALSE(reg.contains("U")) << n;
        EXPECT_THROW(reg.family<double>("U", 1), std::out_of_range) << n;
      });
  EXPECT_GT(failures, 0);
  EXPECT_TRUE(reg.contains("U"));
  EXPECT_NO_THROW(reg.family<double>("U", 1));
}

TEST(ExceptionSafety, AFailedMirrorLeavesTheDestinationEmptyAndRetryable) {
  for (const bool alias : {true, false}) {
    Reg reg;
    reg.set_entities(2, 30);
    soma::FieldOptions<double> opt;
    opt.default_value = std::vector<double>{2.5};
    const auto a = reg.add<double>("a", subset(reg, 30, {3, 8}), 1, opt);
    const auto b = reg.add<float>("b", subset(reg, 30, {4}), 2);
    reg.add_tag<std::int32_t>("c", 2, 1);
    Reg dst;
    const long failures = sweep(
        [&] { reg.mirror_into<HostMem>(dst, {"*"}, {true, true, true, true}, alias); },
        [&](long n) {
          EXPECT_TRUE(dst.list().empty()) << "alias " << alias << ", failed allocation " << n;
          EXPECT_FALSE(dst.contains("a")) << n;
        });
    EXPECT_GT(failures, 0);
    EXPECT_EQ(dst.list().size(), 3u);
    EXPECT_EQ(dst.get<double>(a.id()).rows(), 2);
    EXPECT_EQ(dst.get<float>(b.id()).rows(), 1);
    EXPECT_DOUBLE_EQ(dst.get<double>("a").view()(0, 0), 2.5);
  }
}

// "good" is mirrored first; "late" was defined at another epoch and is stale.
TEST(ExceptionSafety, AStaleSelectedFieldAfterAGoodOneLeavesTheDestinationEmpty) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto good = reg.add_tag<double>("good", 2, 1);
  reg.set_epoch(reg.epoch() + 1);
  reg.add_tag<double>("late", 2, 1);
  reg.set_epoch(reg.epoch() - 1);
  Reg dst;
  EXPECT_THROW(reg.mirror_into<HostMem>(dst, {"good", "late"}, {true, true, true, true}, false),
               soma::StaleField);
  EXPECT_TRUE(dst.list().empty());
  EXPECT_FALSE(dst.contains("good"));
  EXPECT_NO_THROW(reg.mirror_into<HostMem>(dst, {"good"}, {true, true, true, true}, false));
  EXPECT_EQ(dst.get<double>(good.id()).rows(), 30);
  EXPECT_FALSE(dst.contains("late"));
}

TEST(ExceptionSafety, AFailedMirrorKeepsTheEntityCountsOfTheDestination) {
  Reg reg;
  reg.set_entities(2, 30);
  reg.add<double>("a", subset(reg, 30, {3, 8}), 1);
  Reg dst;
  dst.set_entities(2, 30);
  sweep([&] { reg.mirror_into<HostMem>(dst, {"*"}, {true, true, true, true}, false); },
        [&](long n) { EXPECT_TRUE(dst.has_entities(2)) << n; });
  EXPECT_TRUE(dst.has_entities(2));
  EXPECT_EQ(dst.get<double>("a").rows(), 2);
}

namespace {

struct Fixture {
  Reg reg;
  soma::FieldId id = soma::invalid_field;
};

// Like sweep, but every attempt starts from a fresh fixture, so the checks after
// a failure may change it (commit, mirror).
template <class Make, class Op, class Check>
long sweep_fresh(Make&& make, Op&& op, Check&& after_failure) {
  for (long n = 0; n < soma_test::sweep_limit; ++n) {
    auto fx = make();
    bool failed = false;
    {
      const soma_test::FailNth arm(n);
      try {
        op(*fx);
      } catch (const std::bad_alloc&) {
        failed = true;
      }
    }
    if (!failed) return n;
    after_failure(*fx, n);
    if (::testing::Test::HasFatalFailure()) return n;
  }
  ADD_FAILURE() << "the call still failed after " << soma_test::sweep_limit << " allocations";
  return soma_test::sweep_limit;
}

std::unique_ptr<Fixture> fixture(int ncomp, std::vector<LO> lids,
                                 const std::vector<std::pair<LO, double>>& pending) {
  auto fx = std::make_unique<Fixture>();
  fx->reg.set_entities(2, 30);
  fx->id = fx->reg.add<double>("q", subset(fx->reg, 30, std::move(lids)), ncomp).id();
  for (const auto& [lid, x] : pending) {
    fx->reg.set_pending<double>(fx->id, lid,
                                std::vector<double>(static_cast<std::size_t>(ncomp), x));
  }
  return fx;
}

}  // namespace

TEST(ExceptionSafety, AFailedWriteOfANewPendingEntityBuffersNothing) {
  const std::vector<double> v{4.0, 5.0, 6.0};
  const long failures = sweep_fresh(
      [] { return fixture(3, {1, 2}, {{10, 1.0}}); },
      [&](Fixture& fx) { fx.reg.set_pending<double>(fx.id, 20, v); },
      [&](Fixture& fx, long n) {
        EXPECT_EQ(fx.reg.pending_count(fx.id), 1u) << "failed allocation " << n;
        EXPECT_NO_THROW(fx.reg.commit(fx.id)) << n;
        const auto g = fx.reg.get<double>(fx.id);
        EXPECT_EQ(g.domain().members(0), (std::vector<LO>{1, 2, 10})) << n;
        EXPECT_DOUBLE_EQ(g.view()(2, 2), 1.0) << n;
      });
  EXPECT_GT(failures, 0);
  auto fx = fixture(3, {1, 2}, {{10, 1.0}});
  fx->reg.set_pending<double>(fx->id, 20, v);
  EXPECT_EQ(fx->reg.pending_count(fx->id), 2u);
  fx->reg.commit(fx->id);
  const auto g = fx->reg.get<double>(fx->id);
  EXPECT_EQ(g.domain().members(0), (std::vector<LO>{1, 2, 10, 20}));
  EXPECT_DOUBLE_EQ(g.view()(3, 2), 6.0);
}

TEST(ExceptionSafety, AFailedFirstWriteLeavesACommittableField) {
  const std::vector<double> v{4.0, 5.0};
  const long failures = sweep_fresh(
      [] { return fixture(2, {1, 2}, {}); },
      [&](Fixture& fx) { fx.reg.set_pending<double>(fx.id, 20, v); },
      [&](Fixture& fx, long n) {
        EXPECT_EQ(fx.reg.pending_count(fx.id), 0u) << "failed allocation " << n;
        EXPECT_NO_THROW(fx.reg.commit(fx.id)) << n;
        EXPECT_EQ(fx.reg.get<double>(fx.id).domain().members(0), (std::vector<LO>{1, 2})) << n;
        fx.reg.set_pending<double>(fx.id, 11, std::vector<double>{7.0, 8.0});
        EXPECT_EQ(fx.reg.pending_count(fx.id), 1u) << n;
        Reg dst;
        EXPECT_NO_THROW(fx.reg.mirror_into<HostMem>(dst, {"*"}, {true, true, true, true}, false))
            << n;
        const auto g = dst.get<double>(fx.id);
        EXPECT_EQ(g.domain().members(0), (std::vector<LO>{1, 2, 11})) << n;
        EXPECT_DOUBLE_EQ(g.view()(2, 1), 8.0) << n;
      });
  EXPECT_GT(failures, 0);
}

TEST(ExceptionSafety, AFailedReplacementOfAPendingValueKeepsTheOldValue) {
  const std::vector<double> v{7.0, 8.0, 9.0};
  const long failures = sweep_fresh(
      [] { return fixture(3, {1, 2}, {{20, 1.0}}); },
      [&](Fixture& fx) { fx.reg.set_pending<double>(fx.id, 20, v); },
      [&](Fixture& fx, long n) {
        EXPECT_EQ(fx.reg.pending_count(fx.id), 1u) << "failed allocation " << n;
        fx.reg.commit(fx.id);
        const auto g = fx.reg.get<double>(fx.id);
        ASSERT_EQ(g.rows(), 3) << n;
        EXPECT_DOUBLE_EQ(g.view()(2, 0), 1.0) << "the old value survives, n=" << n;
        EXPECT_DOUBLE_EQ(g.view()(2, 2), 1.0) << n;
      });
  EXPECT_GT(failures, 0);
  auto fx = fixture(3, {1, 2}, {{20, 1.0}});
  fx->reg.set_pending<double>(fx->id, 20, v);
  fx->reg.commit(fx->id);
  EXPECT_DOUBLE_EQ(fx->reg.get<double>(fx->id).view()(2, 2), 9.0);
}

// The registry's vectors double when full. Whatever the growth policy, some
// count of existing fields puts the next add on a capacity boundary of the
// entry list, the family list or a member list; sweep every count in a range.
TEST(ExceptionSafety, AFailedAddChangesNothingAtEveryCapacityBoundary) {
  for (int existing = 0; existing <= 9; ++existing) {
    for (const bool same_family : {false, true}) {
      Reg reg;
      reg.set_entities(2, 100);
      for (int i = 0; i < existing; ++i) {
        const std::string name = same_family ? "T" : "T" + std::to_string(i);
        reg.add<double>(name, subset(reg, 100, {static_cast<LO>(i)}), 1);
      }
      const Spc space = subset(reg, 100, {50});
      const std::string name = same_family ? "T" : "new";
      const long failures = sweep(
          [&] { reg.add<double>(name, space, 1); },
          [&](long n) {
            EXPECT_EQ(reg.list().size(), static_cast<std::size_t>(existing))
                << existing << " fields, failed allocation " << n;
            EXPECT_FALSE(reg.contains("new")) << existing << " " << n;
            if (!(same_family && existing > 0)) {
              EXPECT_THROW(reg.family<double>(name, 1), std::out_of_range) << existing << " " << n;
            }
            if (same_family && existing > 0) {
              EXPECT_EQ(reg.family_members(reg.family<double>("T", 1), 2).size(),
                        static_cast<std::size_t>(existing))
                  << n;
            }
          });
      EXPECT_GT(failures, 0) << existing;
      EXPECT_EQ(reg.list().size(), static_cast<std::size_t>(existing) + 1);
      EXPECT_TRUE(reg.lookup(reg.family<double>(name, 1), 2, 50).found());
    }
  }
}

// The same sweep for a mirror into the default memory space and into the host.
// Both are HostSpace in a Serial or OpenMP build, so this only differs from the
// HostSpace sweep above on a GPU build, where it runs the device-copy path.
template <class Dst>
void expect_failed_mirrors_leave_the_destination_empty(bool alias) {
  Reg reg;
  reg.set_entities(2, 30);
  soma::FieldOptions<double> opt;
  opt.default_value = std::vector<double>{2.5};
  const auto a = reg.add<double>("a", subset(reg, 30, {3, 8}), 1, opt);
  reg.add<float>("b", subset(reg, 30, {4}), 2);
  reg.add_tag<std::int32_t>("c", 2, 1);
  soma::FieldRegistry<Dst> dst;
  const long failures = sweep(
      [&] { reg.mirror_into<Dst>(dst, {"*"}, {true, true, true, true}, alias); },
      [&](long n) {
        EXPECT_TRUE(dst.list().empty()) << "alias " << alias << ", failed allocation " << n;
      });
  EXPECT_GT(failures, 0);
  EXPECT_EQ(dst.list().size(), 3u);
  EXPECT_EQ(dst.template get<double>(a.id()).rows(), 2);
}

TEST(ExceptionSafety, AFailedMirrorIntoTheDefaultSpaceLeavesTheDestinationEmpty) {
  for (const bool alias : {true, false}) {
    expect_failed_mirrors_leave_the_destination_empty<Kokkos::DefaultExecutionSpace::memory_space>(alias);
  }
}

TEST(ExceptionSafety, AFailedMirrorIntoTheHostSpaceLeavesTheDestinationEmpty) {
  for (const bool alias : {true, false}) {
    expect_failed_mirrors_leave_the_destination_empty<Kokkos::HostSpace>(alias);
  }
}

// A stale selection is rejected before any commit runs, so nothing is committed
// on the source and nothing is copied.
TEST(ExceptionSafety, AStaleSelectionIsRejectedBeforeAnyCommit) {
  Reg reg;
  reg.set_entities(2, 30);
  const auto first = reg.add<double>("A", subset(reg, 30, {1}), 1);
  const auto second = reg.add<double>("B", subset(reg, 30, {2}), 1);
  reg.set_pending<double>(first.id(), 5, std::vector<double>{1.0});
  reg.set_pending<double>(second.id(), 6, std::vector<double>{2.0});
  reg.set_epoch(reg.epoch() + 1);
  reg.add<double>("C", subset(reg, 30, {3}), 1);
  reg.set_epoch(reg.epoch() - 1);
  Reg dst;
  EXPECT_THROW(reg.mirror_into<HostMem>(dst, {"A", "B", "C"}, {true, true, true, true}, false),
               soma::StaleField);
  EXPECT_EQ(reg.pending_count(first.id()), 1u);
  EXPECT_EQ(reg.pending_count(second.id()), 1u);
  EXPECT_TRUE(dst.list().empty());
  EXPECT_NO_THROW(reg.mirror_into<HostMem>(dst, {"A", "B"}, {true, true, true, true}, false));
  EXPECT_EQ(reg.pending_count(first.id()), 0u);
}

// A commit builds the grown domain, space and data first and publishes them at
// the end. Every allocation failure, wherever it lands, must leave the field as
// it was: same rows, same members, same buffer, a handle that is still current,
// no family lookup for a pending entity, and a retry that gives the result of a
// commit that was never interrupted.
TEST(ExceptionSafety, AFailedCommitLeavesTheFieldAsItWas) {
  const auto make = [] { return fixture(2, {1, 2}, {{10, 5.0}, {7, 3.0}}); };
  auto reference = make();
  reference->reg.commit(reference->id);
  const auto expected = reference->reg.get<double>(reference->id);
  const long failures = sweep_fresh(
      make, [](Fixture& fx) { fx.reg.commit(fx.id); },
      [&](Fixture& fx, long n) {
        const auto before = fx.reg.get<double>(fx.id);
        EXPECT_EQ(fx.reg.pending_count(fx.id), 2u) << "failed allocation " << n;
        EXPECT_EQ(before.rows(), 2) << n;
        EXPECT_EQ(fx.reg.info(fx.id).rows, 2) << n;
        EXPECT_EQ(before.revision(), 0u) << n;
        EXPECT_EQ(before.domain().members(0), (std::vector<LO>{1, 2})) << n;
        EXPECT_TRUE(fx.reg.is_current(before)) << n;
        const auto fam = fx.reg.family<double>("q", 2);
        EXPECT_FALSE(fx.reg.lookup(fam, 2, 10).found()) << n;
        EXPECT_FALSE(fx.reg.lookup(fam, 2, 7).found()) << n;
        EXPECT_TRUE(fx.reg.lookup(fam, 2, 2).found()) << n;
        Reg copy;
        EXPECT_NO_THROW(fx.reg.mirror_into<HostMem>(copy, {"*"}, {true, true, true, true}, false))
            << n;
        EXPECT_EQ(fx.reg.pending_count(fx.id), 0u) << "the mirror commits first, n=" << n;
        const auto after = fx.reg.get<double>(fx.id);
        EXPECT_EQ(after.domain().members(0), expected.domain().members(0)) << n;
        EXPECT_EQ(after.rows(), expected.rows()) << n;
        for (LO r = 0; r < after.rows(); ++r) {
          EXPECT_DOUBLE_EQ(after.view()(r, 1), expected.view()(r, 1)) << n << " row " << r;
        }
      });
  EXPECT_GT(failures, 0);
}

TEST(ExceptionSafety, ARetryAfterAFailedCommitMatchesAnUninterruptedCommit) {
  const auto make = [] { return fixture(2, {1, 2}, {{10, 5.0}, {7, 3.0}}); };
  auto reference = make();
  reference->reg.commit(reference->id);
  const auto expected = reference->reg.get<double>(reference->id);
  const long failures = sweep_fresh(
      make, [](Fixture& fx) { fx.reg.commit(fx.id); },
      [&](Fixture& fx, long n) {
        ASSERT_NO_THROW(fx.reg.commit(fx.id)) << "retry after failed allocation " << n;
        const auto got = fx.reg.get<double>(fx.id);
        EXPECT_EQ(got.domain().members(0), expected.domain().members(0)) << n;
        EXPECT_EQ(got.revision(), expected.revision()) << n;
        EXPECT_EQ(fx.reg.info(fx.id).rows, expected.rows()) << n;
        EXPECT_EQ(fx.reg.pending_count(fx.id), 0u) << n;
        const auto fam = fx.reg.family<double>("q", 2);
        EXPECT_TRUE(fx.reg.lookup(fam, 2, 10).found()) << n;
        EXPECT_EQ(fx.reg.lookup(fam, 2, 10).row, fx.reg.lookup(fam, 2, 10).row) << n;
        for (LO r = 0; r < got.rows(); ++r) {
          for (int c = 0; c < 2; ++c) {
            EXPECT_DOUBLE_EQ(got.view()(r, static_cast<std::size_t>(c)),
                             expected.view()(r, static_cast<std::size_t>(c)))
                << n << " row " << r;
          }
        }
        EXPECT_TRUE(fx.reg.is_current(got)) << n;
      });
  EXPECT_GT(failures, 0);
}
