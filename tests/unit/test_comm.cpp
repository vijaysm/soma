// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
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

TEST(Comm, AllreduceUint64) {
  const Comm c = Comm::world();
  const std::uint64_t high = std::uint64_t{1} << 63;
  const auto contribution = [&](int r) {
    const auto u = static_cast<std::uint64_t>(r);
    return r % 2 == 0 ? high + u : u;
  };
  std::uint64_t expected_sum = 0;
  std::uint64_t expected_min = contribution(0);
  std::uint64_t expected_max = contribution(0);
  for (int r = 0; r < c.size(); ++r) {
    expected_sum += contribution(r);
    expected_min = std::min(expected_min, contribution(r));
    expected_max = std::max(expected_max, contribution(r));
  }
  const std::uint64_t mine = contribution(c.rank());
  EXPECT_EQ(c.allreduce(mine, Op::sum), expected_sum);
  EXPECT_EQ(c.allreduce(mine, Op::min), expected_min);
  EXPECT_EQ(c.allreduce(mine, Op::max), expected_max);
}

TEST(Comm, AllreduceInt32MinMax) {
  const Comm c = Comm::world();
  const std::int32_t r = c.rank();
  EXPECT_EQ(c.allreduce(-r, Op::min), -(c.size() - 1));
  EXPECT_EQ(c.allreduce(-r, Op::max), 0);
}

TEST(Comm, AllreduceProduct) {
  const Comm c = Comm::world();
  EXPECT_EQ(c.allreduce(std::int64_t{2}, Op::prod), std::int64_t{1} << c.size());
  double factorial = 1.0;
  for (int k = 2; k <= c.size(); ++k) {
    factorial *= k;
  }
  EXPECT_DOUBLE_EQ(c.allreduce(static_cast<double>(c.rank() + 1), Op::prod), factorial);
}

TEST(Comm, BcastFromNonZeroRoot) {
  const Comm c = Comm::world();
  const int root = c.size() - 1;
  std::array<std::int32_t, 3> b{-1, -1, -1};
  if (c.rank() == root) {
    b = {root + 10, root + 20, root + 30};
  }
  c.bcast(std::span<std::int32_t>(b), root);
  EXPECT_EQ(b[0], root + 10);
  EXPECT_EQ(b[1], root + 20);
  EXPECT_EQ(b[2], root + 30);
}

TEST(Comm, SelfReductionsIgnoreOtherRanks) {
  const Comm c = Comm::world();
  const Comm s = Comm::self();
  const auto mine = static_cast<std::int64_t>(c.rank()) + 5;
  EXPECT_EQ(s.allreduce(mine, Op::sum), mine);
  EXPECT_EQ(s.allreduce(mine, Op::max), mine);
  EXPECT_EQ(s.allreduce(mine, Op::prod), mine);
  EXPECT_EQ(s.exscan<std::int64_t>(mine), 0);
}

TEST(Comm, AlltoallSendsBlockRToRankR) {
  const Comm c = Comm::world();
  const auto p = static_cast<std::size_t>(c.size());
  std::vector<std::int32_t> send(2 * p), recv(2 * p, -1);
  for (std::size_t d = 0; d < p; ++d) {
    send[2 * d] = 100 * c.rank() + static_cast<std::int32_t>(d);
    send[2 * d + 1] = -send[2 * d];
  }
  c.alltoall(std::span<const std::int32_t>(send), std::span<std::int32_t>(recv));
  for (std::size_t s = 0; s < p; ++s) {
    EXPECT_EQ(recv[2 * s], 100 * static_cast<std::int32_t>(s) + c.rank());
    EXPECT_EQ(recv[2 * s + 1], -recv[2 * s]);
  }
}

// Rank r sends (r + d) % 3 values to rank d, each naming r, d and its place.
TEST(Comm, AlltoallvKeepsRankOrderAndCounts) {
  const Comm c = Comm::world();
  const int p = c.size(), me = c.rank();
  const auto count = [](int from, int to) { return static_cast<std::size_t>((from + to) % 3); };
  std::vector<std::size_t> sc, rc;
  std::vector<std::int64_t> send, expected;
  for (int r = 0; r < p; ++r) {
    sc.push_back(count(me, r));
    rc.push_back(count(r, me));
    for (std::size_t k = 0; k < count(me, r); ++k) send.push_back(10000 * me + 100 * r + static_cast<std::int64_t>(k));
    for (std::size_t k = 0; k < count(r, me); ++k) expected.push_back(10000 * r + 100 * me + static_cast<std::int64_t>(k));
  }
  std::vector<std::int64_t> recv(expected.size(), -1);
  c.alltoallv(std::span<const std::int64_t>(send), std::span<const std::size_t>(sc), std::span<std::int64_t>(recv),
              std::span<const std::size_t>(rc));
  EXPECT_EQ(recv, expected);
}

TEST(Comm, AlltoallRejectsMismatchedBuffers) {
  const Comm c = Comm::world();
  const auto p = static_cast<std::size_t>(c.size());
  std::vector<std::int32_t> send(p + 1), recv(p + 1), short_recv(p);
  if (p > 1) {
    EXPECT_THROW(c.alltoall(std::span<const std::int32_t>(send), std::span<std::int32_t>(recv)), std::invalid_argument);
  }
  std::vector<std::int32_t> whole(p);
  EXPECT_THROW(c.alltoall(std::span<const std::int32_t>(send), std::span<std::int32_t>(short_recv)),
               std::invalid_argument);
  std::vector<std::size_t> counts(p, 1), few(p > 1 ? p - 1 : 0, 1), many(p, 2);
  EXPECT_THROW(c.alltoallv(std::span<const std::int32_t>(whole), std::span<const std::size_t>(few),
                           std::span<std::int32_t>(whole), std::span<const std::size_t>(counts)),
               std::invalid_argument);
  EXPECT_THROW(c.alltoallv(std::span<const std::int32_t>(whole), std::span<const std::size_t>(counts),
                           std::span<std::int32_t>(whole), std::span<const std::size_t>(many)),
               std::invalid_argument);
}
