// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "soma/core/sfc.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/partition.hpp"

using soma::Comm;
using soma::GO;
using soma::Real;

namespace {

// Interleaves cells bit by bit, the first axis most significant: an
// independent statement of what morton_key computes.
std::uint64_t interleave(const std::vector<std::uint32_t>& c, int bits) {
  std::uint64_t key = 0;
  for (int b = bits - 1; b >= 0; --b) {
    for (const std::uint32_t x : c) key = key << 1 | ((x >> b) & 1U);
  }
  return key;
}

}  // namespace

TEST(MortonKey, InterleavesTheQuantizedKeptAxes) {
  using soma::sfc::morton_key;
  using soma::sfc::quantize;
  const Real lo[3] = {-1, 0, 2}, hi[3] = {3, 2, 3};
  const auto box = soma::sfc::KeyBox::make(lo, hi);
  ASSERT_EQ(box.n, 3);
  const std::uint32_t last = soma::sfc::max_cell(3);
  std::mt19937_64 gen(5);
  std::uniform_real_distribution<Real> u(-1.5, 3.5);
  for (int t = 0; t < 200; ++t) {
    const Real p[3] = {u(gen), u(gen), u(gen)};
    std::vector<std::uint32_t> c;
    for (int i = 0; i < 3; ++i) c.push_back(quantize(p[i], box.lo[i], box.extent, last));
    EXPECT_EQ(morton_key(box, p[0], p[1], p[2]), interleave(c, 21));
  }
  EXPECT_EQ(morton_key(box, -1, 0, 2), 0U);
  // The far corner of the widest axis sets every bit of x's cell.
  EXPECT_EQ(morton_key(box, 3, 0, 2), interleave({last, 0, 0}, 21));
  // A planar box keeps two axes of 32 bits; a box without extent gives 0.
  const Real flat_lo[3] = {0, 0, 1}, flat_hi[3] = {1, 1, 1};
  const auto flat = soma::sfc::KeyBox::make(flat_lo, flat_hi);
  ASSERT_EQ(flat.n, 2);
  const std::uint32_t last2 = soma::sfc::max_cell(2);
  EXPECT_EQ(morton_key(flat, 1, 0, 1), interleave({last2, 0}, 32));
  EXPECT_EQ(morton_key(flat, 0.25, 0.75, 9),
            interleave({quantize(0.25, 0, 1, last2), quantize(0.75, 0, 1, last2)}, 32));
  EXPECT_EQ(morton_key(soma::sfc::KeyBox{}, 3, 4, 5), 0U);
}

namespace {

// An 8 x 6 x 5 grid of element centroids with scrambled gids, dealt to ranks
// unevenly (`deal` picks the rank holding element i).
struct Element {
  GO gid;
  Real x, y, z;
};

std::vector<Element> grid() {
  std::vector<Element> out;
  for (int k = 0; k < 5; ++k) {
    for (int j = 0; j < 6; ++j) {
      for (int i = 0; i < 8; ++i) {
        const auto n = static_cast<GO>(out.size());
        out.push_back({(n * 37) % 240 - 100, 0.5 + i, 2.0 * j - 3.0, 0.25 * k});
      }
    }
  }
  return out;
}

int deal(std::size_t i, int size, int how) {
  if (how == 0) return static_cast<int>(i % static_cast<std::size_t>(size));
  if (how == 1) return 0;
  return static_cast<int>((i * i + 3) % static_cast<std::size_t>(size));
}

// The expected destination of every gid, computed from all elements at once.
std::map<GO, int> expected(const std::vector<Element>& all, int size) {
  Real lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
  for (const Element& e : all) {
    const Real p[3] = {e.x, e.y, e.z};
    for (int d = 0; d < 3; ++d) {
      lo[d] = std::min(lo[d], p[d]);
      hi[d] = std::max(hi[d], p[d]);
    }
  }
  const auto box = soma::sfc::KeyBox::make(lo, hi);
  std::vector<std::tuple<std::uint64_t, GO>> keys;
  for (const Element& e : all) keys.emplace_back(soma::sfc::morton_key(box, e.x, e.y, e.z), e.gid);
  std::sort(keys.begin(), keys.end());
  std::map<GO, int> out;
  const std::size_t n = keys.size(), p = static_cast<std::size_t>(size);
  std::size_t at = 0;
  for (std::size_t r = 0; r < p; ++r) {
    const std::size_t len = n / p + (r < n % p ? 1 : 0);
    for (std::size_t k = 0; k < len; ++k) out[std::get<1>(keys[at++])] = static_cast<int>(r);
  }
  return out;
}

}  // namespace

// Runs along the curve, sizes within one, whatever the ranks held first.
TEST(PartitionByCurve, CutsTheCurveIntoEqualRunsWhateverTheInput) {
  const Comm c = Comm::world();
  const std::vector<Element> all = grid();
  const std::map<GO, int> want = expected(all, c.size());
  for (int how = 0; how < 3; ++how) {
    std::vector<Real> xyz;
    std::vector<GO> gids;
    for (std::size_t i = 0; i < all.size(); ++i) {
      if (deal(i, c.size(), how) != c.rank()) continue;
      xyz.insert(xyz.end(), {all[i].x, all[i].y, all[i].z});
      gids.push_back(all[i].gid);
    }
    const std::vector<int> dest = soma::partition_by_curve(c, xyz, gids);
    ASSERT_EQ(dest.size(), gids.size());
    std::vector<std::int64_t> counts(static_cast<std::size_t>(c.size()), 0);
    for (std::size_t i = 0; i < gids.size(); ++i) {
      EXPECT_EQ(dest[i], want.at(gids[i])) << "gid " << gids[i] << ", deal " << how;
      ++counts[static_cast<std::size_t>(dest[i])];
    }
    c.allreduce(std::span<std::int64_t>(counts), soma::Op::sum);
    const auto [mn, mx] = std::minmax_element(counts.begin(), counts.end());
    EXPECT_LE(*mx - *mn, 1);
  }
}

TEST(PartitionByCurve, CoincidentCentroidsAreOrderedByGid) {
  const Comm c = Comm::world();
  std::vector<GO> gids;
  for (GO g = c.rank(); g < 3 * c.size(); g += c.size()) gids.push_back(-g);
  const std::vector<Real> xyz(3 * gids.size(), 2.5);
  const std::vector<int> dest = soma::partition_by_curve(c, xyz, gids);
  // Gids -3P+1 .. 0 in order: three per rank.
  for (std::size_t i = 0; i < gids.size(); ++i) EXPECT_EQ(dest[i], static_cast<int>((gids[i] + 3 * c.size() - 1) / 3));
}

TEST(PartitionByCurve, FewerElementsThanRanksAndNoneAtAll) {
  const Comm c = Comm::world();
  EXPECT_TRUE(soma::partition_by_curve(c, {}, {}).empty());
  std::vector<GO> one;
  std::vector<Real> xyz;
  if (c.rank() == 0) {
    one = {7};
    xyz = {1, 2, 3};
  }
  const auto dest = soma::partition_by_curve(c, xyz, one);
  if (c.rank() == 0) {
    ASSERT_EQ(dest.size(), 1U);
    EXPECT_EQ(dest[0], 0);
  }
}

TEST(PartitionByCurve, BadInputThrowsOnEveryRank) {
  const Comm c = Comm::world();
  const bool last = c.rank() == c.size() - 1;
  const std::vector<GO> gids{1};
  const std::vector<Real> short_xyz{1, 2}, good{1, 2, 3},
      nan{1, std::numeric_limits<Real>::quiet_NaN(), 3};
  EXPECT_THROW((void)soma::partition_by_curve(c, last ? short_xyz : good, gids), std::invalid_argument);
  EXPECT_THROW((void)soma::partition_by_curve(c, last ? nan : good, gids), std::invalid_argument);
}
