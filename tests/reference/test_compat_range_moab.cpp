// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// soma::compat::Range and CN side by side with MOAB's: the same operations
// give the same handles, runs and text.

#include <algorithm>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <moab/CN.hpp>
#include <moab/Range.hpp>

#include "soma/compat/cn.hpp"
#include "soma/compat/range.hpp"

namespace sc = soma::compat;

namespace {

std::vector<std::uint64_t> handles(const sc::Range& r) { return {r.begin(), r.end()}; }
std::vector<std::uint64_t> handles(const moab::Range& r) { return {r.begin(), r.end()}; }

std::vector<std::pair<std::uint64_t, std::uint64_t>> runs(const sc::Range& r) { return {r.pair_begin(), r.pair_end()}; }
std::vector<std::pair<std::uint64_t, std::uint64_t>> runs(const moab::Range& r) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
  for (auto p = r.const_pair_begin(); p != r.const_pair_end(); ++p) out.emplace_back(p->first, p->second);
  return out;
}

void expect_same(const sc::Range& a, const moab::Range& b, const std::string& where) {
  SCOPED_TRACE(where);
  ASSERT_EQ(a.size(), b.size());
  EXPECT_EQ(a.psize(), b.psize());
  EXPECT_EQ(a.empty(), b.empty());
  EXPECT_EQ(runs(a), runs(b));
  if (a.size() < 100000) EXPECT_EQ(handles(a), handles(b));
  EXPECT_EQ(a.str_rep("  "), b.str_rep("  "));
  std::ostringstream sa;
  std::ostringstream sb;
  sa << a;
  sb << b;
  EXPECT_EQ(sa.str(), sb.str());
  if (!a.empty()) {
    EXPECT_EQ(a.front(), b.front());
    EXPECT_EQ(a.back(), b.back());
  }
}

std::uint64_t handle(int type, std::uint64_t id) { return (static_cast<std::uint64_t>(type) << 60) | id; }

}  // namespace

// Random inserts, erases and pops of single handles and runs over a few
// types, compared after every step.
TEST(CompatRange, RandomEditsMatchMoab) {
  std::mt19937_64 rng(42);
  sc::Range a;
  moab::Range b;
  for (int step = 0; step < 3000; ++step) {
    const int type = static_cast<int>(rng() % 4) * 3;  // vertex, quad, prism, polyhedron
    const std::uint64_t id = 1 + rng() % 60;
    const std::uint64_t len = rng() % 6;
    switch (rng() % 7) {
      case 0:
      case 1:
        a.insert(handle(type, id));
        b.insert(handle(type, id));
        break;
      case 2:
      case 3:
        a.insert(handle(type, id), handle(type, id + len));
        b.insert(handle(type, id), handle(type, id + len));
        break;
      case 4:
        a.erase(handle(type, id));
        b.erase(handle(type, id));
        break;
      case 5:
        if (!a.empty() && rng() % 2 == 0) {
          EXPECT_EQ(a.pop_front(), b.pop_front());
        } else if (!a.empty()) {
          EXPECT_EQ(a.pop_back(), b.pop_back());
        }
        break;
      default: {
        // Erase between two iterators.
        if (a.size() < 2) break;
        const auto i = static_cast<std::int64_t>(rng() % a.size());
        const auto j = i + static_cast<std::int64_t>(rng() % (a.size() - static_cast<std::size_t>(i)));
        a.erase(a.begin() + i, a.begin() + j);
        auto bi = b.begin();
        bi += i;
        auto bj = b.begin();
        bj += j;
        b.erase(bi, bj);
      }
    }
    a.sanity_check();
    expect_same(a, b, "step " + std::to_string(step));
    if (HasFatalFailure() || HasFailure()) return;
  }
}

TEST(CompatRange, QueriesMatchMoab) {
  std::mt19937_64 rng(7);
  for (int trial = 0; trial < 50; ++trial) {
    sc::Range a;
    moab::Range b;
    sc::Range c;
    moab::Range d;
    for (int k = 0; k < 40; ++k) {
      const int type = static_cast<int>(rng() % 12);
      const std::uint64_t id = 1 + rng() % 30;
      const std::uint64_t len = rng() % 4;
      a.insert(handle(type, id), handle(type, id + len));
      b.insert(handle(type, id), handle(type, id + len));
      const int t2 = static_cast<int>(rng() % 12);
      c.insert(handle(t2, id + 2));
      d.insert(handle(t2, id + 2));
    }
    expect_same(sc::intersect(a, c), moab::intersect(b, d), "intersect");
    expect_same(sc::subtract(a, c), moab::subtract(b, d), "subtract");
    expect_same(sc::unite(a, c), moab::unite(b, d), "unite");
    sc::Range m = a;
    moab::Range n = b;
    m.merge(c);
    n.merge(d);
    expect_same(m, n, "merge");
    m -= c;
    n -= d;
    expect_same(m, n, "-=");
    EXPECT_EQ(a.contains(c), b.contains(d));
    EXPECT_TRUE(a.contains(sc::intersect(a, c)));
    for (int t = 0; t <= 12; ++t) {
      const auto type = static_cast<sc::EntityType>(t);
      const auto mtype = static_cast<moab::EntityType>(t);
      if (t < 12) {
        expect_same(a.subset_by_type(type), b.subset_by_type(mtype), "subset_by_type");
        EXPECT_EQ(a.num_of_type(type), b.num_of_type(mtype));
        EXPECT_EQ(a.all_of_type(type), b.all_of_type(mtype));
        EXPECT_EQ(a.lower_bound(type) == a.end(), b.lower_bound(mtype) == b.end());
        if (a.lower_bound(type) != a.end()) EXPECT_EQ(*a.lower_bound(type), *b.lower_bound(mtype));
        if (a.upper_bound(type) != a.end()) EXPECT_EQ(*a.upper_bound(type), *b.upper_bound(mtype));
      }
    }
    for (int dim = 0; dim <= 4; ++dim) {
      expect_same(a.subset_by_dimension(dim), b.subset_by_dimension(dim), "subset_by_dimension");
      EXPECT_EQ(a.num_of_dimension(dim), b.num_of_dimension(dim));
      EXPECT_EQ(a.all_of_dimension(dim), b.all_of_dimension(dim));
    }
    for (std::size_t i = 0; i < a.size(); i += 3) {
      EXPECT_EQ(a[static_cast<sc::EntityID>(i)], b[static_cast<moab::EntityID>(i)]);
      EXPECT_EQ(a.index(a[static_cast<sc::EntityID>(i)]), static_cast<int>(i));
    }
    for (int k = 0; k < 30; ++k) {
      const std::uint64_t h = handle(static_cast<int>(rng() % 12), rng() % 36);
      EXPECT_EQ(a.index(h), b.index(h));
      EXPECT_EQ(a.find(h) == a.end(), b.find(h) == b.end());
      const auto lo = a.lower_bound(h);
      const auto mlo = b.lower_bound(h);
      ASSERT_EQ(lo == a.end(), mlo == b.end());
      if (lo != a.end()) EXPECT_EQ(*lo, *mlo);
      const auto up = a.upper_bound(h);
      const auto mup = b.upper_bound(h);
      ASSERT_EQ(up == a.end(), mup == b.end());
      if (up != a.end()) EXPECT_EQ(*up, *mup);
    }
    // Iterators: distance, stepping both ways, reverse order.
    EXPECT_EQ(a.end() - a.begin(), static_cast<sc::EntityID>(a.size()));
    std::vector<std::uint64_t> back(a.rbegin(), a.rend());
    std::vector<std::uint64_t> mback(b.rbegin(), b.rend());
    EXPECT_EQ(back, mback);
    if (a.size() > 5) {
      auto it = a.begin() + 5;
      it -= 3;
      EXPECT_EQ(*it, b[2]);
      EXPECT_EQ(*(a.end() - 1), b.back());
    }
    // Ranges inserted from another range's iterators, partial runs included.
    if (a.size() > 4) {
      sc::Range part;
      part.insert(a.begin() + 1, a.end() - 2);
      moab::Range mpart;
      auto mb = b.begin();
      ++mb;
      auto me = b.end();
      --me;
      --me;
      mpart.insert(mb, me);
      expect_same(part, mpart, "insert(first, last)");
    }
  }
}

TEST(CompatRange, PrintsLikeMoab) {
  sc::Range a;
  moab::Range b;
  expect_same(a, b, "empty");
  EXPECT_EQ(a.str_rep(), "\tempty\n");
  a.insert(handle(0, 1), handle(3, 2));  // a run across types
  b.insert(handle(0, 1), handle(3, 2));
  expect_same(a, b, "across types");
  EXPECT_EQ(a.str_rep(), "\tVertex 1 - Quad 2\n");
  sc::Range s(handle(11, 1), handle(11, 1));
  EXPECT_EQ(s.str_rep(">"), ">\tEntitySet 1\n");
}

TEST(CompatCN, TablesMatchMoab) {
  for (int t = 0; t < 12; ++t) {
    const auto type = static_cast<sc::EntityType>(t);
    const auto mtype = static_cast<moab::EntityType>(t);
    SCOPED_TRACE(moab::CN::EntityTypeName(mtype));
    EXPECT_STREQ(sc::CN::EntityTypeName(type), moab::CN::EntityTypeName(mtype));
    EXPECT_EQ(sc::CN::EntityTypeFromName(sc::CN::EntityTypeName(type)), type);
    EXPECT_EQ(sc::CN::Dimension(type), moab::CN::Dimension(mtype));
    if (t == sc::MBPOLYGON || t == sc::MBPOLYHEDRON || t == sc::MBENTITYSET) continue;
    EXPECT_EQ(sc::CN::VerticesPerEntity(type), moab::CN::VerticesPerEntity(mtype));
    for (int d = 0; d <= sc::CN::Dimension(type); ++d) {
      EXPECT_EQ(sc::CN::NumSubEntities(type, d), moab::CN::NumSubEntities(mtype, d)) << "dim " << d;
      for (int i = 0; i < sc::CN::NumSubEntities(type, d); ++i) {
        EXPECT_EQ(static_cast<int>(sc::CN::SubEntityType(type, d, i)), static_cast<int>(moab::CN::SubEntityType(mtype, d, i)));
        int x[8] = {};
        int y[8] = {};
        sc::CN::SubEntityVertexIndices(type, d, i, x);
        moab::CN::SubEntityVertexIndices(mtype, d, i, y);
        EXPECT_EQ(std::vector<int>(x, x + 8), std::vector<int>(y, y + 8)) << "dim " << d << " side " << i;
      }
    }
    for (int n = 1; n <= 27; ++n) EXPECT_EQ(sc::CN::HasMidNodes(type, n), moab::CN::HasMidNodes(mtype, n)) << n << " nodes";
  }
  for (int d = 0; d <= 4; ++d) {
    EXPECT_EQ(static_cast<int>(sc::CN::TypeDimensionMap[d].first), static_cast<int>(moab::CN::TypeDimensionMap[d].first));
    EXPECT_EQ(static_cast<int>(sc::CN::TypeDimensionMap[d].second), static_cast<int>(moab::CN::TypeDimensionMap[d].second));
  }
  EXPECT_EQ(sc::CN::EntityTypeFromName("Nonesuch"), sc::MBMAXTYPE);
}

// Every side of every fixed type, in every rotation and both directions,
// given by connectivity values: the same side number, sense and offset.
TEST(CompatCN, SideNumberMatchesMoab) {
  for (int t = sc::MBEDGE; t <= sc::MBHEX; ++t) {
    if (t == sc::MBPOLYGON) continue;
    const auto type = static_cast<sc::EntityType>(t);
    const auto mtype = static_cast<moab::EntityType>(t);
    std::vector<std::uint64_t> conn(static_cast<std::size_t>(sc::CN::VerticesPerEntity(type)));
    for (std::size_t k = 0; k < conn.size(); ++k) conn[k] = 100 + 7 * k;
    for (int d = 1; d <= sc::CN::Dimension(type); ++d) {
      const int count = d == sc::CN::Dimension(type) ? 1 : sc::CN::NumSubEntities(type, d);
      for (int s = 0; s < count; ++s) {
        int idx[8] = {};
        sc::CN::SubEntityVertexIndices(type, d, s, idx);
        const int n = d == sc::CN::Dimension(type) ? static_cast<int>(conn.size())
                                                   : sc::CN::VerticesPerEntity(sc::CN::SubEntityType(type, d, s));
        for (int rot = 0; rot < n; ++rot) {
          for (int dir = 0; dir < 2; ++dir) {
            std::vector<std::uint64_t> child(static_cast<std::size_t>(n));
            for (int k = 0; k < n; ++k) {
              const int at = dir == 0 ? (rot + k) % n : (rot + n - k) % n;
              child[static_cast<std::size_t>(k)] = conn[static_cast<std::size_t>(idx[at])];
            }
            int side = 0, sense = 0, offset = 0;
            int mside = 0, msense = 0, moffset = 0;
            const int r = sc::CN::SideNumber(type, conn.data(), child.data(), n, d, side, sense, offset);
            const int mr = moab::CN::SideNumber(mtype, conn.data(), child.data(), n, d, mside, msense, moffset);
            SCOPED_TRACE(std::string(moab::CN::EntityTypeName(mtype)) + " dim " + std::to_string(d) + " side " +
                         std::to_string(s) + " rot " + std::to_string(rot) + " dir " + std::to_string(dir));
            EXPECT_EQ(r, mr);
            EXPECT_EQ(side, mside);
            EXPECT_EQ(sense, msense);
            EXPECT_EQ(offset, moffset);
          }
        }
      }
    }
    // A child that is no side.
    std::vector<std::uint64_t> stranger{conn[0], 1};
    int side = 0, sense = 0, offset = 0;
    int mside = 0, msense = 0, moffset = 0;
    EXPECT_EQ(sc::CN::SideNumber(type, conn.data(), stranger.data(), 2, 1, side, sense, offset),
              moab::CN::SideNumber(mtype, conn.data(), stranger.data(), 2, 1, mside, msense, moffset));
  }
}
