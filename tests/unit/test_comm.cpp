// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
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

TEST(Comm, ExscanSpanIsExclusivePerEntry) {
  const Comm c = Comm::world();
  const auto r = static_cast<std::uint64_t>(c.rank());
  std::array<std::uint64_t, 3> v{1, r + 1, 0};
  c.exscan(std::span<std::uint64_t>(v));
  EXPECT_EQ(v[0], r);
  EXPECT_EQ(v[1], r * (r + 1) / 2);
  EXPECT_EQ(v[2], 0u);
}

TEST(Comm, CountsEveryCollectiveCall) {
  const Comm c = Comm::world();
  const auto before = soma::detail::comm_counts();
  c.barrier();
  (void)c.allreduce(std::int32_t{1}, Op::sum);
  std::array<std::int64_t, 2> two{1, 2};
  c.allreduce(std::span<std::int64_t>(two), Op::max);
  (void)c.exscan(std::int64_t{1});
  std::array<std::uint64_t, 2> scan{1, 2};
  c.exscan(std::span<std::uint64_t>(scan));
  std::array<int, 1> b{3};
  c.bcast(std::span<int>(b), 0);
  std::vector<int> send(static_cast<std::size_t>(c.size()), 1), recv(send.size());
  c.alltoall(std::span<const int>(send), std::span<int>(recv));
  std::vector<std::size_t> counts(send.size(), 1);
  c.alltoallv(std::span<const int>(send), std::span<const std::size_t>(counts), std::span<int>(recv),
              std::span<const std::size_t>(counts));
  c.ibarrier().wait();
  const auto after = soma::detail::comm_counts();
  EXPECT_EQ(after.collectives - before.collectives, 9u);
  EXPECT_EQ(after.routes, before.routes);
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

using soma::Request;

TEST(Comm, RingMessagesArriveWhicheverSideStartsFirst) {
  const Comm c = Comm::world();
  const int next = (c.rank() + 1) % c.size(), prev = (c.rank() + c.size() - 1) % c.size();
  for (const bool receive_first : {true, false}) {
    const std::array<std::int32_t, 3> out{c.rank(), 2 * c.rank(), receive_first ? 1 : 0};
    std::array<std::int32_t, 3> in{-1, -1, -1};
    std::array<Request, 2> r;
    if (receive_first) r[0] = c.irecv(std::span<std::int32_t>(in), prev, 5);
    r[1] = c.isend(std::span<const std::int32_t>(out), next, 5);
    if (!receive_first) r[0] = c.irecv(std::span<std::int32_t>(in), prev, 5);
    Request::wait_all(r);
    EXPECT_EQ(in, (std::array<std::int32_t, 3>{prev, 2 * prev, receive_first ? 1 : 0}));
  }
}

TEST(Comm, IprobeFindsTheSenderAndSize) {
  const Comm c = Comm::world();
  const int next = (c.rank() + 1) % c.size(), prev = (c.rank() + c.size() - 1) % c.size();
  const std::vector<std::int32_t> out(static_cast<std::size_t>(c.rank() + 1), c.rank());
  Request s = c.issend(std::span<const std::int32_t>(out), next, 7);
  // Bounded: a probe that never finds the message fails instead of spinning.
  std::optional<soma::Envelope> e;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while (!(e = c.iprobe(Comm::any_source, 7)) && std::chrono::steady_clock::now() < deadline) {
  }
  ASSERT_TRUE(e.has_value()) << "no message arrived within 20 s";
  EXPECT_EQ(e->source, prev);
  EXPECT_EQ(e->tag, 7);
  EXPECT_EQ(e->bytes, sizeof(std::int32_t) * static_cast<std::size_t>(prev + 1));
  std::vector<std::int32_t> in(e->bytes / sizeof(std::int32_t), -1);
  c.irecv(std::span<std::int32_t>(in), e->source, 7).wait();
  s.wait();
  EXPECT_EQ(in, std::vector<std::int32_t>(static_cast<std::size_t>(prev + 1), prev));
  c.barrier();
  EXPECT_FALSE(c.iprobe(Comm::any_source, 7).has_value());
}

// A synchronous send to oneself cannot complete before its receive is posted.
TEST(Comm, IssendCompletesOnlyOnceReceived) {
  const Comm c = Comm::world();
  const std::int64_t out = 42;
  std::int64_t in = 0;
  Request s = c.issend(std::span<const std::int64_t>(&out, 1), c.rank(), 11);
  EXPECT_FALSE(s.test());
  Request r = c.irecv(std::span<std::int64_t>(&in, 1), c.rank(), 11);
  r.wait();
  s.wait();
  EXPECT_TRUE(s.test());
  EXPECT_EQ(in, 42);
  Request plain = c.isend(std::span<const std::int64_t>(&out, 1), c.rank(), 12);
  c.irecv(std::span<std::int64_t>(&in, 1), c.rank(), 12).wait();
  plain.wait();
  EXPECT_TRUE(Request{}.test());
}

TEST(Comm, MessagesWithTheSameTagArriveInOrder) {
  const Comm c = Comm::world();
  const std::array<std::int32_t, 2> out{1, 2};
  std::array<std::int32_t, 2> in{};
  std::array<Request, 4> r;
  r[0] = c.isend(std::span<const std::int32_t>(&out[0], 1), c.rank(), 13);
  r[1] = c.isend(std::span<const std::int32_t>(&out[1], 1), c.rank(), 13);
  r[2] = c.irecv(std::span<std::int32_t>(&in[0], 1), c.rank(), 13);
  r[3] = c.irecv(std::span<std::int32_t>(&in[1], 1), c.rank(), 13);
  Request::wait_all(r);
  EXPECT_EQ(in, out);
}

TEST(Comm, IbarrierCompletes) {
  const Comm c = Comm::world();
  Request b = c.ibarrier();
  b.wait();
  EXPECT_TRUE(b.test());
}

TEST(Comm, SequencesArePerCommunicatorAndAgree) {
  const std::uint64_t a = Comm::world().next_sequence();
  const std::uint64_t b = Comm::world().next_sequence();
  EXPECT_EQ(b, a + 1);
  const Comm c = Comm::world();
  EXPECT_EQ(c.allreduce(a, Op::min), c.allreduce(a, Op::max));
}

TEST(Comm, PointToPointRejectsBadRanksAndTags) {
  const Comm c = Comm::world();
  std::array<std::byte, 1> b{};
  EXPECT_THROW((void)c.isend_bytes(b, c.size(), 0), std::invalid_argument);
  EXPECT_THROW((void)c.issend_bytes(b, -1, 0), std::invalid_argument);
  EXPECT_THROW((void)c.isend_bytes(b, 0, -1), std::invalid_argument);
  EXPECT_THROW((void)c.irecv_bytes(b, 0, 32768), std::invalid_argument);
  EXPECT_THROW((void)c.irecv_bytes(b, -2, 0), std::invalid_argument);
  EXPECT_THROW((void)c.iprobe(c.size(), 0), std::invalid_argument);
}

#if !SOMA_ENABLE_MPI
// Without MPI nothing else can complete a request, so waiting is an error,
// and a receive dropped unmatched leaves later messages queued.
TEST(Comm, SerialWaitsThatCannotCompleteThrow) {
  const Comm c = Comm::world();
  std::int32_t in = 0;
  const std::int32_t out = 3;
  Request r = c.irecv(std::span<std::int32_t>(&in, 1), 0, 20);
  EXPECT_FALSE(r.test());
  EXPECT_THROW(r.wait(), std::logic_error);
  r = Request{};
  Request s = c.issend(std::span<const std::int32_t>(&out, 1), 0, 20);
  EXPECT_THROW(s.wait(), std::logic_error);
  EXPECT_EQ(in, 0);
  c.irecv(std::span<std::int32_t>(&in, 1), 0, 20).wait();
  EXPECT_EQ(in, 3);
  EXPECT_TRUE(s.test());
  std::array<std::byte, 8> big{};
  std::array<std::byte, 4> small{};
  c.isend_bytes(big, 0, 21).wait();
  EXPECT_THROW(c.irecv_bytes(small, 0, 21).wait(), std::runtime_error);
}

// Rank 0 is its only peer: what it sends itself must be what it receives,
// even when each side's counts add up to its own buffer.
TEST(Comm, SerialAlltoallvSelfCountsMustAgree) {
  const Comm c = Comm::world();
  const std::vector<std::int32_t> one{7};
  std::vector<std::int32_t> two(2, 0);
  const std::vector<std::size_t> sc{1}, rc{2};
  EXPECT_THROW(c.alltoallv(std::span<const std::int32_t>(one), std::span<const std::size_t>(sc),
                           std::span<std::int32_t>(two), std::span<const std::size_t>(rc)),
               std::invalid_argument);
}
#endif
