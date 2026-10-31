// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// A Domain or Space with static storage is moved from while the program shuts
// down. The empty snapshot that a move leaves behind must still exist then, even
// though it was first needed after the holder was constructed (statics are
// destroyed in reverse order of construction). Run under AddressSanitizer: the
// defect is a use of a destroyed shared_ptr that is silent otherwise.

#include <utility>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/space.hpp"

using HostMem = Kokkos::HostSpace;

namespace soma_test {

// The holders own no Views (those would have to be freed before
// Kokkos::finalize). Their destructors run at program exit and create and move
// from an empty Domain or Space, which needs the shared empty snapshot.
struct DomainHolder {
  int unused = 0;
  ~DomainHolder() {
    soma::Domain<HostMem> a;
    const soma::Domain<HostMem> b(std::move(a));
    unused = b.size();
  }
};

struct SpaceHolder {
  int unused = 0;
  ~SpaceHolder() {
    soma::Space<HostMem> a;
    const soma::Space<HostMem> b(std::move(a));
    unused = static_cast<int>(b.rows());
  }
};

}  // namespace soma_test

// The holders are constructed first; the empty snapshots are first built after
// them, so a function-local static snapshot would be destroyed before the
// holders' destructors run.
TEST(StaticLifetime, EmptyDomainAndSpaceAreUsableDuringShutdown) {
  static soma_test::DomainHolder domain_holder;
  static soma_test::SpaceHolder space_holder;
  const soma::Domain<HostMem> late_domain;
  const soma::Space<HostMem> late_space;
  EXPECT_TRUE(late_domain.empty());
  EXPECT_EQ(late_space.rows(), 0);
  EXPECT_EQ(domain_holder.unused, 0);
  EXPECT_EQ(space_holder.unused, 0);
}
