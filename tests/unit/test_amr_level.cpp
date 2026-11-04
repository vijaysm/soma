// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Box arrays on a curve, chopping, distribution, and the storage of level
// fields: every stored value belongs to exactly one point of one box.

#include <algorithm>
#include <functional>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/amr/box_array.hpp"
#include "soma/amr/level_field.hpp"
#include "soma/core/sfc.hpp"

namespace {

using soma::amr::Box;
using soma::amr::BoxArray;
using soma::amr::BoxList;
using soma::amr::Curve;
using soma::amr::IndexType;
using soma::amr::IntVect;

Box box2(int i0, int j0, int i1, int j1) { return Box(IntVect{{i0, j0, 0}}, IntVect{{i1, j1, 0}}, 2); }
Box box3(IntVect lo, IntVect hi) { return Box(lo, hi, 3); }

std::string what_of(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST(AmrBoxArray, OrderedAlongTheCurve) {
  const Box domain = box2(0, 0, 15, 15);
  const BoxList quads{box2(8, 8, 15, 15), box2(0, 8, 7, 15), box2(8, 0, 15, 7), box2(0, 0, 7, 7)};
  const BoxArray m(quads, domain);
  // Morton interleaves i in the lowest bit: (0,0), (8,0), (0,8), (8,8).
  EXPECT_EQ(m[0], box2(0, 0, 7, 7));
  EXPECT_EQ(m[1], box2(8, 0, 15, 7));
  EXPECT_EQ(m[2], box2(0, 8, 7, 15));
  EXPECT_EQ(m[3], box2(8, 8, 15, 15));
  EXPECT_EQ(m.num_pts(), 256);
  const BoxArray h(quads, domain, Curve::hilbert);
  for (std::size_t i = 0; i + 1 < h.size(); ++i) {
    std::uint32_t a[2] = {static_cast<std::uint32_t>(h[i].lo(0)), static_cast<std::uint32_t>(h[i].lo(1))};
    std::uint32_t b[2] = {static_cast<std::uint32_t>(h[i + 1].lo(0)), static_cast<std::uint32_t>(h[i + 1].lo(1))};
    EXPECT_LT(soma::sfc::hilbert_index(a, 2, 32), soma::sfc::hilbert_index(b, 2, 32));
  }
  // Consecutive Hilbert boxes share a face; Morton jumps from (8,0) to (0,8).
  const auto share_face = [](const Box& a, const Box& b) { return a.grow(0, 1).intersects(b) || a.grow(1, 1).intersects(b); };
  for (std::size_t i = 0; i + 1 < h.size(); ++i) EXPECT_TRUE(share_face(h[i], h[i + 1]));
  EXPECT_FALSE(share_face(m[1], m[2]));
  // Keys are taken from the domain's lower corner, also below zero.
  const Box shifted(IntVect{{-8, -8, 0}}, IntVect{{7, 7, 0}}, 2);
  BoxList moved;
  for (const Box& b : quads) moved.push_back(b.shift(IntVect{{-8, -8, 0}}));
  const BoxArray s(moved, shifted);
  for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(s[i], m[i].shift(IntVect{{-8, -8, 0}}));
}

TEST(AmrBoxArray, RejectsWhatIsNoLevel) {
  const Box domain = box2(0, 0, 15, 15);
  EXPECT_NE(what_of([&] { BoxArray({box2(0, 0, 7, 7), box2(7, 7, 9, 9)}, domain); }).find("[0,0]-[7,7] and [7,7]-[9,9] overlap"),
            std::string::npos);
  EXPECT_NE(what_of([&] { BoxArray({box2(10, 10, 16, 12)}, domain); }).find("[10,10]-[16,12]"), std::string::npos);
  EXPECT_FALSE(what_of([&] { BoxArray({box3(IntVect{}, IntVect{{1, 1, 1}})}, domain); }).empty());
  EXPECT_FALSE(what_of([&] { BoxArray({box2(0, 0, 1, 1).convert(IndexType::node(2))}, domain); }).empty());
  EXPECT_FALSE(what_of([&] { BoxArray({Box()}, domain); }).empty());
}

TEST(AmrBoxArray, ChopKeepsEveryPointOnce) {
  std::mt19937 g(3);
  std::uniform_int_distribution<int> len(1, 70), at(-40, 40);
  for (int trial = 0; trial < 60; ++trial) {
    const int dim = 2 + trial % 2;
    const int max_box = 4 + trial % 29;
    IntVect lo, hi;
    for (int d = 0; d < dim; ++d) {
      lo[d] = at(g);
      hi[d] = lo[d] + len(g) - 1;
    }
    const Box b(lo, hi, dim);
    const BoxList pieces = soma::amr::chop({b}, max_box);
    EXPECT_TRUE(soma::amr::disjoint(pieces));
    EXPECT_EQ(soma::amr::num_pts(pieces), b.num_pts());
    EXPECT_TRUE(soma::amr::covers(pieces, b));
    for (int d = 0; d < dim; ++d) {
      int shortest = max_box, longest = 0;
      for (const Box& p : pieces) {
        EXPECT_TRUE(b.contains(p));
        EXPECT_LE(p.length(d), max_box);
        shortest = std::min(shortest, p.length(d));
        longest = std::max(longest, p.length(d));
      }
      EXPECT_LE(longest - shortest, 1) << "pieces of near-equal length";
    }
    // Cut into the fewest pieces of near-equal length per direction.
    std::int64_t expect = 1;
    for (int d = 0; d < dim; ++d) expect *= (b.length(d) + max_box - 1) / max_box;
    EXPECT_EQ(static_cast<std::int64_t>(pieces.size()), expect);
  }
  // A box aligned to the blocking factor is cut at its multiples.
  const Box aligned(IntVect{{8, 0, 16}}, IntVect{{8 + 80 - 1, 23, 16 + 40 - 1}});
  for (const Box& p : soma::amr::chop({aligned}, 32, 8)) {
    for (int d = 0; d < 3; ++d) {
      EXPECT_EQ(p.lo(d) % 8, 0) << d;
      EXPECT_EQ(p.length(d) % 8, 0) << d;
      EXPECT_LE(p.length(d), 32);
    }
  }
  // A max_box that is no multiple of the blocking factor still bounds the pieces.
  for (const Box& p : soma::amr::chop({aligned}, 20, 8)) {
    for (int d = 0; d < 3; ++d) EXPECT_LE(p.length(d), 20) << d;
  }
  EXPECT_THROW(soma::amr::chop({aligned}, 0), std::invalid_argument);
}

TEST(AmrDistribution, ContiguousRunsOfEqualWork) {
  BoxList six;
  for (int i = 0; i < 6; ++i) six.push_back(box2(4 * i, 0, 4 * i + 3, 3));
  const BoxArray ba(six, box2(0, 0, 23, 3));
  EXPECT_EQ(soma::amr::Distribution::along_curve(ba, 1).owner, (std::vector<int>(6, 0)));
  EXPECT_EQ(soma::amr::Distribution::along_curve(ba, 3).owner, (std::vector<int>{0, 0, 1, 1, 2, 2}));
  // A big box weighs more: it takes a rank of its own.
  const BoxArray uneven({box2(0, 0, 11, 3), box2(12, 0, 13, 3), box2(14, 0, 15, 3), box2(16, 0, 17, 3)},
                        box2(0, 0, 17, 3));
  EXPECT_EQ(soma::amr::Distribution::along_curve(uneven, 2).owner, (std::vector<int>{0, 1, 1, 1}));
  const auto many = soma::amr::Distribution::along_curve(ba, 10).owner;
  EXPECT_TRUE(std::is_sorted(many.begin(), many.end()));
  EXPECT_EQ(many.back(), 9);
  EXPECT_THROW(soma::amr::Distribution::along_curve(ba, 0), std::invalid_argument);
}

namespace {

// Kernels live in named functions: CUDA refuses device lambdas in a test
// body (a private member function) or inside another lambda.
// Adds 1 to every component of every point of `region`.
void add_one(const soma::amr::LevelField<int>& f, soma::amr::Region region) {
  const auto v = f.view();
  const int ncomp = f.ncomp();
  soma::amr::for_each_point(f, region, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    for (int c = 0; c < ncomp; ++c) Kokkos::atomic_add(&v.patch(b)(i, j, k, c), 1);
  });
}

// Adds 1 to component 1 of every valid point; returns how many points it met.
std::int64_t count_valid(const soma::amr::LevelField<int>& f) {
  const auto v = f.view();
  Kokkos::View<std::int64_t, Kokkos::DefaultExecutionSpace::memory_space> n("n");
  soma::amr::for_each_point(f, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    Kokkos::atomic_add(&v.patch(b)(i, j, k, 1), 1);
    Kokkos::atomic_add(&n(), std::int64_t{1});
  });
  auto nh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, n);
  return nh();
}

}  // namespace

TEST(AmrLevelField, EveryValueBelongsToOnePoint) {
  for (const int dim : {2, 3}) {
    const Box domain = dim == 2 ? box2(0, 0, 31, 31) : box3(IntVect{}, IntVect{{15, 15, 15}});
    const BoxList boxes = soma::amr::chop({domain.grow(-2)}, 7);
    const BoxArray ba(boxes, domain);
    for (const IndexType t : {IndexType::cell(), IndexType::node(dim), IndexType::face(0), IndexType::face(1)}) {
      for (const int ghosts : {0, 2}) {
        const int ncomp = 2;
        soma::amr::LevelField<int> f(ba, t, ncomp, ghosts);
        ASSERT_EQ(f.num_boxes(), ba.size());
        std::int64_t total = 0;
        for (std::size_t b = 0; b < f.num_boxes(); ++b) {
          EXPECT_EQ(f.valid(b), ba[b].convert(t));
          EXPECT_EQ(f.grown(b), ba[b].convert(t).grow(ghosts));
          EXPECT_EQ(f.offset(b), total);
          total += f.grown(b).num_pts() * ncomp;
        }
        ASSERT_EQ(static_cast<std::int64_t>(f.view().data.extent(0)), total);
        // Each point of each grown box adds 1 to each of its components:
        // every stored value ends at 1.
        add_one(f, soma::amr::Region::grown);
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f.view().data);
        for (std::size_t q = 0; q < h.extent(0); ++q) ASSERT_EQ(h(q), 1) << q;
        // The valid points: as many as the boxes hold, each once.
        f.fill(0);
        const std::int64_t met = count_valid(f);
        std::int64_t valid = 0;
        for (std::size_t b = 0; b < f.num_boxes(); ++b) valid += f.valid(b).num_pts();
        EXPECT_EQ(met, valid);
        Kokkos::deep_copy(h, f.view().data);
        std::int64_t ones = 0;
        for (std::size_t q = 0; q < h.extent(0); ++q) {
          ASSERT_LE(h(q), 1);
          ones += h(q);
        }
        EXPECT_EQ(ones, valid);
      }
    }
  }
}

TEST(AmrLevelField, PatchIndexesTheGrownBox) {
  const BoxArray ba({box3(IntVect{{4, 0, 2}}, IntVect{{7, 2, 3}})}, box3(IntVect{}, IntVect{{9, 9, 9}}));
  soma::amr::LevelField<double, Kokkos::HostSpace> f(ba, IndexType::cell(), 3, 1);
  const auto p = f.view().patch(0);
  // Column-major from the grown box's lower corner (3, -1, 1), sizes 6 x 5 x 4.
  EXPECT_EQ(&p(3, -1, 1, 0), f.view().data.data());
  EXPECT_EQ(&p(4, -1, 1, 0) - &p(3, -1, 1, 0), 1);
  EXPECT_EQ(&p(3, 0, 1, 0) - &p(3, -1, 1, 0), 6);
  EXPECT_EQ(&p(3, -1, 2, 0) - &p(3, -1, 1, 0), 30);
  EXPECT_EQ(&p(3, -1, 1, 1) - &p(3, -1, 1, 0), 120);
  EXPECT_EQ(&p(8, 3, 4, 2) - f.view().data.data() + 1, 360);
}
