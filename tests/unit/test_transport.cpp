// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "soma/parallel/comm.hpp"
#include "soma/parallel/transport/comm_plan.hpp"
#include "soma/parallel/transport/rendezvous.hpp"
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

// The keys rank `from` holds, some repeated, some shared with other ranks.
std::vector<std::uint64_t> held_keys(int from, int size) {
  std::mt19937 gen(static_cast<std::mt19937::result_type>(77 * size + from));
  std::uniform_int_distribution<std::uint64_t> key(0, static_cast<std::uint64_t>(4 * size));
  std::vector<std::uint64_t> out(static_cast<std::size_t>(2 * size + from));
  for (std::uint64_t& k : out) k = key(gen);
  return out;
}

// For each arrival: the lowest rank holding its key and how many records
// were sent for it.
Kokkos::View<std::byte**, Kokkos::LayoutRight, Mem> first_holders(const Kokkos::View<std::uint64_t*, Mem>& keys,
                                                                 const RecordBatch<Mem>& arrived) {
  const auto k = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, keys);
  const auto s = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, arrived.source);
  std::map<std::uint64_t, std::array<std::int32_t, 2>> seen;
  for (std::size_t i = 0; i < k.extent(0); ++i) {
    auto [it, fresh] = seen.try_emplace(k(i), std::array<std::int32_t, 2>{s(i), 0});
    it->second[0] = std::min(it->second[0], s(i));
    ++it->second[1];
  }
  Kokkos::View<std::byte**, Kokkos::LayoutRight, Mem> rows("rows", k.extent(0), 8);
  const auto h = Kokkos::create_mirror_view(rows);
  for (std::size_t i = 0; i < k.extent(0); ++i) std::memcpy(&h(i, 0), seen.at(k(i)).data(), 8);
  Kokkos::deep_copy(rows, h);
  return rows;
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

TEST(Transport, EachRouteIsCountedOnce) {
  const Comm c = Comm::world();
  RecordBatch<Mem> in(1, 0);
  Kokkos::deep_copy(in.dest, 0);
  for (const Strategy strategy : {Strategy::crystal, Strategy::alltoallv, Strategy::nbx}) {
    const auto before = soma::detail::comm_counts().routes;
    (void)soma::transport::route(Space{}, c, in, strategy);
    EXPECT_EQ(soma::detail::comm_counts().routes - before, 1u);
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

TEST(Transport, OwnersDependOnlyOnTheKeyAndRankCount) {
  using soma::transport::mix;
  using soma::transport::owner;
  static_assert(mix(1) == 0x5692161d100b05e5ULL && mix(2) == 0xdbd238973a2b148aULL);
  EXPECT_EQ(mix(12345), 0xf36cf1164265dd51ULL);
  const std::array<int, 10> expected{0, 6, 1, 4, 6, 2, 4, 2, 6, 4};
  for (std::uint64_t k = 0; k < 10; ++k) EXPECT_EQ(owner(k, 7), expected[k]);
  EXPECT_EQ(owner(~std::uint64_t{0}, 1), 0);
}

// Every rank learns, for each key it holds, the lowest rank holding it and
// how many records name it.
TEST(Transport, RendezvousAnswersEachKeyInOrder) {
  const Comm c = Comm::world();
  const std::vector<std::uint64_t> mine = held_keys(c.rank(), c.size());
  std::map<std::uint64_t, std::array<std::int32_t, 2>> truth;
  for (int r = 0; r < c.size(); ++r) {
    for (const std::uint64_t k : held_keys(r, c.size())) {
      auto [it, fresh] = truth.try_emplace(k, std::array<std::int32_t, 2>{r, 0});
      it->second[0] = std::min(it->second[0], r);
      ++it->second[1];
    }
  }
  Kokkos::View<std::uint64_t*, Mem> keys("keys", mine.size());
  Kokkos::View<std::byte**, Kokkos::LayoutRight, Mem> payload("payload", mine.size(), 0);
  const auto hk = Kokkos::create_mirror_view(keys);
  for (std::size_t i = 0; i < mine.size(); ++i) hk(i) = mine[i];
  Kokkos::deep_copy(keys, hk);
  for (const Strategy strategy : {Strategy::crystal, Strategy::alltoallv, Strategy::nbx}) {
    const auto answers = soma::transport::rendezvous(Space{}, c, keys, payload, 8, first_holders, strategy);
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, answers);
    ASSERT_EQ(h.extent(0), mine.size());
    for (std::size_t i = 0; i < mine.size(); ++i) {
      std::array<std::int32_t, 2> got;
      std::memcpy(got.data(), &h(i, 0), 8);
      EXPECT_EQ(got, truth.at(mine[i])) << "key " << mine[i];
    }
  }
}

// rendezvous_to sends each key where the caller says: here key % P, which
// answers with its rank and the sender's.
TEST(Transport, RendezvousToAChosenRank) {
  const Comm c = Comm::world();
  const std::vector<std::uint64_t> mine = held_keys(c.rank(), c.size());
  Kokkos::View<std::uint64_t*, Mem> keys("keys", mine.size());
  Kokkos::View<std::byte**, Kokkos::LayoutRight, Mem> payload("payload", mine.size(), 0);
  const auto hk = Kokkos::create_mirror_view(keys);
  for (std::size_t i = 0; i < mine.size(); ++i) hk(i) = mine[i];
  Kokkos::deep_copy(keys, hk);
  const int me = c.rank(), size = c.size();
  const auto who = [me](const Kokkos::View<std::uint64_t*, Mem>& arrived, const RecordBatch<Mem>& from) {
    Kokkos::View<std::byte**, Kokkos::LayoutRight, Mem> rows("rows", arrived.extent(0), 8);
    const auto r = Kokkos::create_mirror_view(rows);
    const auto s = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, from.source);
    for (std::size_t i = 0; i < r.extent(0); ++i) {
      const std::array<std::int32_t, 2> v{me, s(i)};
      std::memcpy(&r(i, 0), v.data(), 8);
    }
    Kokkos::deep_copy(rows, r);
    return rows;
  };
  const auto answers = soma::transport::rendezvous_to(
      Space{}, c, keys, payload, 8, who, [size](std::uint64_t k) { return static_cast<int>(k % static_cast<std::uint64_t>(size)); });
  const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, answers);
  ASSERT_EQ(h.extent(0), mine.size());
  for (std::size_t i = 0; i < mine.size(); ++i) {
    std::array<std::int32_t, 2> got;
    std::memcpy(got.data(), &h(i, 0), 8);
    EXPECT_EQ(got, (std::array<std::int32_t, 2>{static_cast<std::int32_t>(mine[i] % static_cast<std::uint64_t>(size)), me}));
  }
  EXPECT_THROW((void)soma::transport::rendezvous_to(Space{}, c, keys, payload, 8, who, [size](std::uint64_t) { return size; }),
               std::invalid_argument);
}

namespace {

Kokkos::View<std::byte**, Kokkos::LayoutRight, Mem> short_rows(const Kokkos::View<std::uint64_t*, Mem>& keys,
                                                              const RecordBatch<Mem>&) {
  return Kokkos::View<std::byte**, Kokkos::LayoutRight, Mem>("short", keys.extent(0) + 1, 8);
}

Kokkos::View<std::byte**, Kokkos::LayoutRight, Mem> throw_on_rank_zero(const Kokkos::View<std::uint64_t*, Mem>& keys,
                                                                      const RecordBatch<Mem>&) {
  if (Comm::world().rank() == 0) throw std::domain_error("no answer");
  return Kokkos::View<std::byte**, Kokkos::LayoutRight, Mem>("rows", keys.extent(0), 8);
}

}  // namespace

TEST(Transport, RendezvousFailuresReachEveryRank) {
  const Comm c = Comm::world();
  Kokkos::View<std::uint64_t*, Mem> keys("keys", 2);
  Kokkos::View<std::byte**, Kokkos::LayoutRight, Mem> payload("payload", c.rank() == 0 ? 3 : 2, 4);
  EXPECT_THROW((void)soma::transport::rendezvous(Space{}, c, keys, payload, 8, first_holders), std::invalid_argument);
  payload = Kokkos::View<std::byte**, Kokkos::LayoutRight, Mem>("payload", 2, 4);
  EXPECT_THROW((void)soma::transport::rendezvous(Space{}, c, keys, payload, 8, short_rows), std::invalid_argument);
  if (c.rank() == 0) {
    EXPECT_THROW((void)soma::transport::rendezvous(Space{}, c, keys, payload, 8, throw_on_rank_zero), std::domain_error);
  } else {
    EXPECT_THROW((void)soma::transport::rendezvous(Space{}, c, keys, payload, 8, throw_on_rank_zero), std::runtime_error);
  }
}

// A received count is checked before any buffer is sized from it, so a
// corrupt or mismatched message throws instead of allocating without bound.
TEST(Transport, IncomingMessagesAreBoundedByWhatWasSent) {
  using soma::transport::detail::check_incoming;
  const std::size_t huge = std::numeric_limits<std::size_t>::max();
  EXPECT_NO_THROW(check_incoming(0, 40, 20, 2));
  EXPECT_NO_THROW(check_incoming(1, 20, 20, 2));
  EXPECT_NO_THROW(check_incoming(5, 0, 20, 5));
  EXPECT_THROW(check_incoming(0, 41, 20, 10), std::runtime_error);
  EXPECT_THROW(check_incoming(0, 60, 20, 2), std::runtime_error);
  EXPECT_THROW(check_incoming(2, 20, 20, 2), std::runtime_error);
  EXPECT_THROW(check_incoming(0, std::numeric_limits<std::uint64_t>::max() - 15, 8, 1000), std::runtime_error);
  EXPECT_THROW(check_incoming(huge, 8, 8, huge), std::runtime_error);
}
