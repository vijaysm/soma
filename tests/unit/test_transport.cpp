// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "soma/parallel/comm.hpp"
#include "soma/parallel/transport/route.hpp"

using soma::Comm;
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
