// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "soma/parallel/comm.hpp"
#include "soma/parallel/transport/sort.hpp"

using soma::Comm;
using soma::transport::rank_keys;
using soma::transport::Strategy;

namespace {

using Key = std::vector<std::uint64_t>;

// The keys rank `from` holds: some ranks hold none, values repeat within and
// across ranks.
std::vector<Key> keys_of(int from, int size, std::size_t words, std::uint64_t range) {
  std::mt19937_64 gen(static_cast<std::uint64_t>(31 * size + from) * 7919 + words);
  const std::size_t n = from % 3 == 2 ? 0 : static_cast<std::size_t>(gen() % 60);
  std::vector<Key> out(n, Key(words));
  for (Key& k : out) {
    for (std::uint64_t& x : k) x = gen() % range;
  }
  return out;
}

std::vector<std::uint64_t> flat(const std::vector<Key>& ks) {
  std::vector<std::uint64_t> out;
  for (const Key& k : ks) out.insert(out.end(), k.begin(), k.end());
  return out;
}

}  // namespace

// Distinct ranks number the distinct keys of all ranks in ascending order.
TEST(KeyRanks, DistinctIndicesNumberTheDistinctKeys) {
  const Comm c = Comm::world();
  for (const std::size_t words : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
    for (const std::uint64_t range : {std::uint64_t{3}, std::uint64_t{1000000}}) {
      std::vector<Key> all;
      for (int r = 0; r < c.size(); ++r) {
        const auto ks = keys_of(r, c.size(), words, range);
        all.insert(all.end(), ks.begin(), ks.end());
      }
      std::sort(all.begin(), all.end());
      all.erase(std::unique(all.begin(), all.end()), all.end());
      const auto mine = keys_of(c.rank(), c.size(), words, range);
      for (const Strategy s : {Strategy::crystal, Strategy::alltoallv, Strategy::nbx}) {
        const auto got = rank_keys(c, flat(mine), words, true, s);
        EXPECT_EQ(got.total, all.size());
        ASSERT_EQ(got.index.size(), mine.size());
        for (std::size_t i = 0; i < mine.size(); ++i) {
          const auto at = static_cast<std::uint64_t>(std::lower_bound(all.begin(), all.end(), mine[i]) - all.begin());
          EXPECT_EQ(got.index[i], at) << "words " << words << " range " << range;
        }
      }
    }
  }
}

// Without `distinct` every key has its own index: by key, then sender rank,
// then the sender's order.
TEST(KeyRanks, PlainIndicesOrderTiesByRankThenPlace) {
  const Comm c = Comm::world();
  for (const std::uint64_t range : {std::uint64_t{4}, std::uint64_t{1000000}}) {
    std::vector<std::tuple<Key, int, std::size_t>> all;
    for (int r = 0; r < c.size(); ++r) {
      const auto ks = keys_of(r, c.size(), 2, range);
      for (std::size_t i = 0; i < ks.size(); ++i) all.emplace_back(ks[i], r, i);
    }
    std::sort(all.begin(), all.end());
    const auto mine = keys_of(c.rank(), c.size(), 2, range);
    const auto got = rank_keys(c, flat(mine), 2, false);
    EXPECT_EQ(got.total, all.size());
    for (std::size_t i = 0; i < mine.size(); ++i) {
      const auto at = std::lower_bound(all.begin(), all.end(), std::make_tuple(mine[i], c.rank(), i));
      EXPECT_EQ(got.index[i], static_cast<std::uint64_t>(at - all.begin()));
    }
  }
}

TEST(KeyRanks, NoKeysAnywhereIsEmpty) {
  const Comm c = Comm::world();
  const auto got = rank_keys(c, {}, 3, true);
  EXPECT_EQ(got.total, 0U);
  EXPECT_TRUE(got.index.empty());
}

TEST(KeyRanks, BadArgumentsThrowOnEveryRank) {
  const Comm c = Comm::world();
  const int last = c.size() - 1;
  const std::vector<std::uint64_t> three{1, 2, 3};
  EXPECT_THROW((void)rank_keys(c, three, c.rank() == last ? 2 : 1, true), std::invalid_argument);
  EXPECT_THROW((void)rank_keys(c, {}, c.rank() == last ? 0 : 1, true), std::invalid_argument);
  if (c.size() > 1) {
    EXPECT_THROW((void)rank_keys(c, three, c.rank() == last ? 3 : 1, true), std::invalid_argument);
    EXPECT_THROW((void)rank_keys(c, three, 1, c.rank() == last), std::invalid_argument);
  }
}
