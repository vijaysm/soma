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
  soma::LO unused = 0;
  ~DomainHolder() {
    soma::Domain<HostMem> a;
    const soma::Domain<HostMem> b(std::move(a));
    unused = b.size();
  }
};

struct SpaceHolder {
  soma::LO unused = 0;
  ~SpaceHolder() {
    soma::Space<HostMem> a;
    const soma::Space<HostMem> b(std::move(a));
    unused = b.rows();
  }
};

}  // namespace soma_test

// The holders are constructed first; the empty snapshots are first built after
// them, so a function-local static snapshot would be destroyed before the
// holders' destructors run. The two types get one test each (ctest runs every
// test in its own process): an empty Space holds a copy of the empty Domain
// snapshot, which would keep that one alive in a process that used both.
TEST(StaticLifetime, EmptyDomainIsUsableDuringShutdown) {
  static soma_test::DomainHolder holder;
  const soma::Domain<HostMem> late;
  EXPECT_TRUE(late.empty());
  EXPECT_EQ(holder.unused, 0);
}

TEST(StaticLifetime, EmptySpaceIsUsableDuringShutdown) {
  static soma_test::SpaceHolder holder;
  const soma::Space<HostMem> late;
  EXPECT_EQ(late.rows(), 0);
  EXPECT_EQ(holder.unused, 0);
}
