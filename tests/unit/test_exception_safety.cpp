// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Fails the n-th heap allocation made while one registry call runs, for every n
// until the call succeeds, and checks that a failed call changed nothing.

#include <cstdint>
#include <cstdlib>
#include <new>
#include <string>
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

void* operator new(std::size_t n) {
  if (soma_test::allocations_until_failure >= 0 && soma_test::allocations_until_failure-- == 0) {
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

