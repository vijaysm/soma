// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "soma/core/lid_runs.hpp"

using soma::DomainKind;
using soma::Interval;
using soma::LO;
using soma::invalid_lo;

namespace {

// Indexes a vector from offset `at` on, as the run search expects.
struct At {
  const std::vector<LO>* v;
  std::size_t at;
  LO operator()(LO i) const { return (*v)[at + static_cast<std::size_t>(i)]; }
};

struct Packed {
  DomainKind kind;
  LO count;
  LO size;
  LO runs;
  std::vector<LO> data;
  LO entity(LO row) const {
    return soma::detail::run_entity(kind, size, runs, At{&data, 0}, At{&data, static_cast<std::size_t>(runs)}, row);
  }
  LO row(LO lid) const {
    return soma::detail::run_row(kind, count, size, runs, At{&data, 0}, At{&data, static_cast<std::size_t>(runs)},
                                 lid);
  }
};

Packed pack(const std::vector<LO>& lids, LO count) {
  const auto runs = soma::detail::runs_of(lids);
  const auto n = static_cast<LO>(lids.size());
  const auto r = static_cast<LO>(runs.size());
  const DomainKind k = soma::detail::form_of(n, count, r);
  return {k, count, n, k == DomainKind::intervals ? r : 0, soma::detail::pack_runs(runs, k)};
}

}  // namespace

TEST(LidRuns, RunsOfASortedList) {
  EXPECT_TRUE(soma::detail::runs_of({}).empty());
  const std::vector<LO> l{0, 1, 2, 5, 7, 8};
  EXPECT_EQ(soma::detail::runs_of(l), (std::vector<Interval>{{0, 3}, {5, 6}, {7, 9}}));
}

// full when every id is a member; intervals only when 2 runs + 1 < size.
TEST(LidRuns, FormIsTheSmallest) {
  using soma::detail::form_of;
  EXPECT_EQ(form_of(10, 10, 1), DomainKind::full);
  EXPECT_EQ(form_of(0, 0, 0), DomainKind::full);
  EXPECT_EQ(form_of(0, 10, 0), DomainKind::list);
  EXPECT_EQ(form_of(4, 10, 2), DomainKind::list);       // 5 ids either way: the list
  EXPECT_EQ(form_of(6, 10, 2), DomainKind::intervals);  // 5 < 6
  EXPECT_EQ(form_of(5, 10, 2), DomainKind::list);
}

TEST(LidRuns, PackedFormsHoldTheirIds) {
  const std::vector<Interval> runs{{2, 5}, {7, 10}};
  EXPECT_EQ(soma::detail::pack_runs(runs, DomainKind::intervals), (std::vector<LO>{2, 7, 0, 3, 6}));
  EXPECT_EQ(soma::detail::pack_runs(runs, DomainKind::list), (std::vector<LO>{2, 3, 4, 7, 8, 9}));
  EXPECT_TRUE(soma::detail::pack_runs(runs, DomainKind::full).empty());
}

// Every form maps rows to ids and ids to rows, and refuses what is not there.
TEST(LidRuns, SearchInEveryForm) {
  const std::vector<std::pair<std::vector<LO>, LO>> cases{
      {{0, 1, 2, 3, 4, 5}, 6},                  // full
      {{1, 2, 3, 4, 10, 11, 12, 13, 14}, 20},   // intervals
      {{0, 3, 6, 9}, 12},                       // list
      {{}, 4}};                                 // empty list
  std::vector<DomainKind> kinds;
  for (const auto& [lids, count] : cases) {
    const Packed p = pack(lids, count);
    kinds.push_back(p.kind);
    for (LO k = 0; k < p.size; ++k) {
      EXPECT_EQ(p.entity(k), lids[static_cast<std::size_t>(k)]) << k;
      EXPECT_EQ(p.row(lids[static_cast<std::size_t>(k)]), k) << k;
    }
    EXPECT_EQ(p.entity(-1), invalid_lo);
    EXPECT_EQ(p.entity(p.size), invalid_lo);
    EXPECT_EQ(p.row(-1), invalid_lo);
    EXPECT_EQ(p.row(count), invalid_lo);
    for (LO l = 0; l < count; ++l) {
      const bool in = std::find(lids.begin(), lids.end(), l) != lids.end();
      EXPECT_EQ(p.row(l) != invalid_lo, in) << l;
    }
  }
  EXPECT_EQ(kinds, (std::vector<DomainKind>{DomainKind::full, DomainKind::intervals, DomainKind::list,
                                            DomainKind::list}));
}
