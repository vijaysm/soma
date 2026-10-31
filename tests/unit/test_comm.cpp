// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "soma/parallel/comm.hpp"

using soma::Comm;
using soma::Op;

TEST(Comm, WorldRankIsInRange) {
  const Comm c = Comm::world();
  EXPECT_GE(c.rank(), 0);
  EXPECT_LT(c.rank(), c.size());
}

TEST(Comm, SelfIsSingleRank) {
  const Comm s = Comm::self();
  EXPECT_EQ(s.rank(), 0);
  EXPECT_EQ(s.size(), 1);
}

TEST(Comm, AllreduceSumCountsRanks) {
  const Comm c = Comm::world();
  EXPECT_EQ(c.allreduce(std::int32_t{1}, Op::sum), c.size());
}

TEST(Comm, AllreduceMinMax) {
  const Comm c = Comm::world();
  const std::int64_t r = c.rank();
  EXPECT_EQ(c.allreduce(r, Op::min), 0);
  EXPECT_EQ(c.allreduce(r, Op::max), c.size() - 1);
}

TEST(Comm, AllreduceSpanIsElementwise) {
  const Comm c = Comm::world();
  std::vector<double> v{1.0, static_cast<double>(c.rank())};
  c.allreduce(std::span<double>(v), Op::sum);
  EXPECT_DOUBLE_EQ(v[0], c.size());
  EXPECT_DOUBLE_EQ(v[1], c.size() * (c.size() - 1) / 2.0);
}

TEST(Comm, ExscanIsExclusive) {
  const Comm c = Comm::world();
  EXPECT_EQ(c.exscan(1), c.rank());
  const auto r = static_cast<std::int64_t>(c.rank());
  EXPECT_EQ(c.exscan<std::int64_t>(r + 1), r * (r + 1) / 2);
}

TEST(Comm, BcastDeliversRootBuffer) {
  const Comm c = Comm::world();
  std::array<std::byte, 4> b{};
  if (c.rank() == 0) {
    b.fill(std::byte{0x5A});
  }
  c.bcast(std::span<std::byte>(b), 0);
  for (const std::byte x : b) {
    EXPECT_EQ(x, std::byte{0x5A});
  }
}
