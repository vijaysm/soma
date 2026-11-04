// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "soma/parallel/comm.hpp"
#include "soma/parallel/transport/comm_plan.hpp"
#include "soma/parallel/transport/route.hpp"

using soma::Comm;
using soma::transport::CommPlan;
using soma::transport::RecordBatch;
using soma::transport::Strategy;

namespace {

using Space = Kokkos::DefaultExecutionSpace;
using Mem = Space::memory_space;

enum class Pattern { empty, all_to_one, all_to_all, ring, random };

// A record as sent: where it goes and its payload, three ints naming the
// sender, its place and its destination.
struct Record {
  int dest;
  std::array<std::int32_t, 3> payload;
};

// The records rank `from` sends, which every rank can regenerate.
std::vector<Record> records(Pattern pattern, int from, int size) {
  std::vector<int> dests;
  switch (pattern) {
    case Pattern::empty: break;
    case Pattern::all_to_one: dests.assign(3, size - 1); break;
    case Pattern::all_to_all:
      for (int d = size - 1; d >= 0; --d) dests.insert(dests.end(), static_cast<std::size_t>(d % 3 + 1), d);
      break;
    case Pattern::ring: dests = {(from + 1) % size, (from + size - 1) % size}; break;
    case Pattern::random: {
      std::mt19937 gen(static_cast<std::mt19937::result_type>(1000 * size + from));
      std::uniform_int_distribution<int> count(0, 3 * size), dest(0, size - 1);
      dests.resize(static_cast<std::size_t>(count(gen)));
      for (int& d : dests) d = dest(gen);
      break;
    }
  }
  std::vector<Record> out;
  for (std::size_t i = 0; i < dests.size(); ++i) out.push_back({dests[i], {from, static_cast<std::int32_t>(i), dests[i]}});
  return out;
}

RecordBatch<Mem> batch(const std::vector<Record>& rs) {
  RecordBatch<Mem> b(rs.size(), sizeof(Record::payload));
  const auto d = Kokkos::create_mirror_view(b.dest);
  const auto p = Kokkos::create_mirror_view(b.payload);
  for (std::size_t i = 0; i < rs.size(); ++i) {
    d(i) = rs[i].dest;
    std::memcpy(&p(i, 0), rs[i].payload.data(), sizeof(Record::payload));
  }
  Kokkos::deep_copy(b.dest, d);
  Kokkos::deep_copy(b.payload, p);
  return b;
}

// (source, payload) of every record of a batch, in order.
std::vector<std::pair<int, std::array<std::int32_t, 3>>> contents(const RecordBatch<Mem>& b) {
  const auto s = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.source);
  const auto p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.payload);
  std::vector<std::pair<int, std::array<std::int32_t, 3>>> out(b.size());
  for (std::size_t i = 0; i < b.size(); ++i) {
    out[i].first = s(i);
    std::memcpy(out[i].second.data(), &p(i, 0), sizeof(Record::payload));
  }
  return out;
}

}  // namespace

// Every strategy delivers exactly the records sent here, grouped by source
// and in each source's order.
TEST(Transport, EveryStrategyDeliversTheSameRecordsInOrder) {
  const Comm c = Comm::world();
  for (const Pattern pattern : {Pattern::empty, Pattern::all_to_one, Pattern::all_to_all, Pattern::ring, Pattern::random}) {
    std::vector<std::pair<int, std::array<std::int32_t, 3>>> expected;
    for (int s = 0; s < c.size(); ++s) {
      for (const Record& r : records(pattern, s, c.size())) {
        if (r.dest == c.rank()) expected.emplace_back(s, r.payload);
      }
    }
    const RecordBatch<Mem> in = batch(records(pattern, c.rank(), c.size()));
    for (const Strategy strategy : {Strategy::automatic, Strategy::crystal, Strategy::alltoallv, Strategy::nbx}) {
      const RecordBatch<Mem> out = soma::transport::route(Space{}, c, in, strategy);
      EXPECT_EQ(contents(out), expected) << "pattern " << static_cast<int>(pattern) << ", strategy "
                                         << static_cast<int>(strategy);
      const auto d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out.dest);
      for (std::size_t i = 0; i < out.size(); ++i) EXPECT_EQ(d(i), c.rank());
      EXPECT_EQ(out.width(), in.width());
    }
  }
}

// Exchanges in quick succession must not take each other's messages.
TEST(Transport, BackToBackExchangesStaySeparate) {
  const Comm c = Comm::world();
  for (int round = 0; round < 20; ++round) {
    const Strategy strategy = round % 3 == 0 ? Strategy::crystal : Strategy::nbx;
    RecordBatch<Mem> in(static_cast<std::size_t>(c.size()), sizeof(std::int32_t));
    const auto d = Kokkos::create_mirror_view(in.dest);
    const auto p = Kokkos::create_mirror_view(in.payload);
    for (int r = 0; r < c.size(); ++r) {
      d(static_cast<std::size_t>(r)) = r;
      const std::int32_t v = 1000 * round + c.rank();
      std::memcpy(&p(static_cast<std::size_t>(r), 0), &v, sizeof v);
    }
    Kokkos::deep_copy(in.dest, d);
    Kokkos::deep_copy(in.payload, p);
    const RecordBatch<Mem> out = soma::transport::route(Space{}, c, in, strategy);
    const auto got = contents(out);
    ASSERT_EQ(out.size(), static_cast<std::size_t>(c.size()));
    const auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out.payload);
    for (int s = 0; s < c.size(); ++s) {
      std::int32_t v = 0;
      std::memcpy(&v, &hp(static_cast<std::size_t>(s), 0), sizeof v);
      EXPECT_EQ(v, 1000 * round + s);
      EXPECT_EQ(got[static_cast<std::size_t>(s)].first, s);
    }
  }
}

TEST(Transport, RecordsWithoutPayloadCarryTheirSource) {
  const Comm c = Comm::world();
  RecordBatch<Mem> in(2, 0);
  Kokkos::deep_copy(in.dest, 0);
  for (const Strategy strategy : {Strategy::crystal, Strategy::alltoallv, Strategy::nbx}) {
    const RecordBatch<Mem> out = soma::transport::route(Space{}, c, in, strategy);
    const auto s = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out.source);
    ASSERT_EQ(out.size(), c.rank() == 0 ? 2 * static_cast<std::size_t>(c.size()) : 0);
    for (std::size_t i = 0; i < out.size(); ++i) EXPECT_EQ(s(i), static_cast<int>(i / 2));
  }
}

// A bad argument on one rank throws on every rank instead of hanging.
TEST(Transport, BadArgumentsThrowOnEveryRank) {
  const Comm c = Comm::world();
  const int last = c.size() - 1;
  RecordBatch<Mem> in(1, 4);
  Kokkos::deep_copy(in.dest, c.rank() == last ? c.size() : 0);
  EXPECT_THROW((void)soma::transport::route(Space{}, c, in, Strategy::crystal), std::invalid_argument);
  Kokkos::deep_copy(in.dest, c.rank() == last ? -1 : 0);
  EXPECT_THROW((void)soma::transport::route(Space{}, c, in, Strategy::nbx), std::invalid_argument);
  if (c.size() > 1) {
    RecordBatch<Mem> wide(1, c.rank() == last ? 8 : 4);
    EXPECT_THROW((void)soma::transport::route(Space{}, c, wide, Strategy::alltoallv), std::invalid_argument);
    RecordBatch<Mem> ok(1, 4);
    EXPECT_THROW((void)soma::transport::route(Space{}, c, ok, c.rank() == last ? Strategy::nbx : Strategy::crystal),
                 std::invalid_argument);
  }
  RecordBatch<Mem> ragged(1, 4);
  if (c.rank() == last) ragged.payload = decltype(ragged.payload)("ragged", 2, 4);
  EXPECT_THROW((void)soma::transport::route(Space{}, c, ragged), std::invalid_argument);
}

TEST(Transport, AutomaticPicksByRankCount) {
  using soma::transport::detail::resolve;
  EXPECT_EQ(resolve(Strategy::automatic, 1), Strategy::alltoallv);
  EXPECT_EQ(resolve(Strategy::automatic, 64), Strategy::alltoallv);
  EXPECT_EQ(resolve(Strategy::automatic, 65), Strategy::crystal);
  EXPECT_EQ(resolve(Strategy::nbx, 1000), Strategy::nbx);
}

// A plan learnt from one exchange replays it: forward delivers what route()
// did, in its order, and reverse undoes forward.
TEST(Transport, PlansReplayTheRoutedExchange) {
  const Comm c = Comm::world();
  for (const Pattern pattern : {Pattern::empty, Pattern::all_to_one, Pattern::all_to_all, Pattern::ring, Pattern::random}) {
    const std::vector<Record> mine = records(pattern, c.rank(), c.size());
    const RecordBatch<Mem> sent = batch(mine);
    const RecordBatch<Mem> arrived = soma::transport::route(Space{}, c, sent, Strategy::crystal);
    const CommPlan plan = soma::transport::learn_plan(c, sent, arrived);
    const auto routed = contents(arrived);
    ASSERT_EQ(plan.num_sent(), mine.size());
    ASSERT_EQ(plan.num_arrived(), routed.size());

    std::set<int> to, from;
    for (const Record& r : mine) {
      if (r.dest != c.rank()) to.insert(r.dest);
    }
    for (const auto& r : routed) {
      if (r.first != c.rank()) from.insert(r.first);
    }
    EXPECT_EQ(plan.send_ranks(), std::vector<int>(to.begin(), to.end()));
    EXPECT_EQ(plan.recv_ranks(), std::vector<int>(from.begin(), from.end()));

    std::vector<std::array<std::int32_t, 3>> values(mine.size()), got(routed.size());
    for (std::size_t i = 0; i < mine.size(); ++i) values[i] = mine[i].payload;
    plan.forward(c, std::span<const std::array<std::int32_t, 3>>(values), std::span<std::array<std::int32_t, 3>>(got));
    for (std::size_t i = 0; i < routed.size(); ++i) EXPECT_EQ(got[i], routed[i].second);

    // Answers computed where the records arrived go back to their places.
    std::vector<double> answers(routed.size()), back(mine.size(), -1.0);
    for (std::size_t i = 0; i < routed.size(); ++i) answers[i] = 0.5 * routed[i].second[1] + routed[i].second[0];
    plan.reverse(c, std::span<const double>(answers), std::span<double>(back));
    for (std::size_t i = 0; i < mine.size(); ++i) EXPECT_EQ(back[i], 0.5 * static_cast<double>(i) + c.rank());
  }
}

TEST(Transport, PlansRejectInconsistentHalves) {
  const Comm c = Comm::world();
  const int me = c.rank();
  EXPECT_THROW(CommPlan(c, std::vector<int>{c.size()}, std::vector<int>{}), std::invalid_argument);
  EXPECT_THROW(CommPlan(c, std::vector<int>{}, std::vector<int>{-1}), std::invalid_argument);
  if (c.size() > 1) {
    EXPECT_THROW(CommPlan(c, std::vector<int>{}, std::vector<int>{1, 0}), std::invalid_argument);
  }
  EXPECT_THROW(CommPlan(c, std::vector<int>{me, me}, std::vector<int>{me}), std::invalid_argument);
  const CommPlan plan(c, std::vector<int>{me, me}, std::vector<int>{me, me});
  EXPECT_TRUE(plan.send_ranks().empty());
  EXPECT_TRUE(plan.recv_ranks().empty());
  std::vector<std::int32_t> two{1, 2}, three(3), out(2);
  EXPECT_THROW(plan.forward(c, std::span<const std::int32_t>(two), std::span<std::int32_t>(three)), std::invalid_argument);
  EXPECT_THROW(plan.reverse(c, std::span<const std::int32_t>(three), std::span<std::int32_t>(out)), std::invalid_argument);
  plan.forward(c, std::span<const std::int32_t>(two), std::span<std::int32_t>(out));
  EXPECT_EQ(out, two);
  EXPECT_EQ(CommPlan().num_arrived(), 0U);
}
