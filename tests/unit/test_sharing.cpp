// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "soma/parallel/comm.hpp"
#include "soma/parallel/sharing.hpp"

using soma::Comm;
using soma::GO;
using soma::Ownership;

namespace {

// Whether rank r holds gid g: a mix of gids held once, by neighbours, and by
// many ranks; gids are negative too.
bool holds(int r, GO g, int size) {
  const auto u = static_cast<std::uint64_t>(g + 1000);
  if (u % 11 == 0) return true;                                     // everywhere
  if (u % 3 == 0) return static_cast<int>(u % static_cast<std::uint64_t>(size)) == r;  // once
  return (u * 7 + static_cast<std::uint64_t>(r) * 5) % 4 < 2;       // a few ranks
}

std::vector<GO> held_by(int r, int size) {
  std::vector<GO> out;
  for (GO g = -40; g < 200; ++g) {
    if (holds(r, g, size)) out.push_back(g);
  }
  // Not sorted: the order given is the order of the rows.
  std::reverse(out.begin(), out.end());
  return out;
}

}  // namespace

TEST(Sharing, EveryHolderLearnsTheOthersAndTheOwner) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  const std::vector<GO> mine = held_by(me, p);
  const soma::Sharing s = soma::resolve_sharing(c, mine);
  ASSERT_EQ(s.offsets.size(), mine.size() + 1);
  for (std::size_t i = 0; i < mine.size(); ++i) {
    std::vector<int> want;
    for (int r = 0; r < p; ++r) {
      if (r != me && holds(r, mine[i], p)) want.push_back(r);
    }
    const std::vector<int> got(s.ranks.begin() + s.offsets[i], s.ranks.begin() + s.offsets[i + 1]);
    EXPECT_EQ(got, want) << "gid " << mine[i];
    const int owner = want.empty() ? me : std::min(me, want.front());
    EXPECT_EQ(s.owner(i, me), owner);
    EXPECT_EQ(s.kind(i, me), want.empty() ? Ownership::interior : owner == me ? Ownership::shared : Ownership::shared_copy);
  }
}

// Groups are equal exactly for equal rank sets and follow their order.
TEST(Sharing, GroupsNumberTheRankSetsInOrder) {
  const Comm c = Comm::world();
  const std::vector<GO> mine = held_by(c.rank(), c.size());
  const soma::Sharing s = soma::resolve_sharing(c, mine);
  const std::vector<std::uint32_t> g = s.groups();
  const auto row = [&](std::size_t i) {
    return std::vector<int>(s.ranks.begin() + s.offsets[i], s.ranks.begin() + s.offsets[i + 1]);
  };
  for (std::size_t i = 0; i < mine.size(); ++i) {
    for (std::size_t j = 0; j < mine.size(); ++j) {
      EXPECT_EQ(g[i] == g[j], row(i) == row(j));
      EXPECT_EQ(g[i] < g[j], row(i) < row(j));
    }
  }
}

TEST(Sharing, NothingHeldAndRepeatedGids) {
  const Comm c = Comm::world();
  const soma::Sharing none = soma::resolve_sharing(c, {});
  EXPECT_EQ(none.offsets.size(), 1U);
  EXPECT_TRUE(none.groups().empty());
  const std::vector<GO> twice{5, 5}, once{5};
  EXPECT_THROW((void)soma::resolve_sharing(c, c.rank() == c.size() - 1 ? twice : once), std::invalid_argument);
}
