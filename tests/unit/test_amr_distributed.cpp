// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// AMR levels over many ranks: every operation gives each rank's boxes the
// bits the same operation gives them with every box on one process, which
// each rank computes for itself, so nothing is gathered to compare.

#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/amr/box_array.hpp"
#include "soma/amr/copy.hpp"
#include "soma/amr/halo.hpp"
#include "soma/amr/level_field.hpp"
#include "soma/parallel/comm.hpp"

namespace {

using soma::Comm;
using soma::amr::Box;
using soma::amr::BoxArray;
using soma::amr::BoxList;
using soma::amr::Distribution;
using soma::amr::IndexType;
using soma::amr::IntVect;
using soma::amr::LevelField;
using soma::amr::Periodicity;

constexpr double sentinel = -1.0e300;

Box box(int dim, int n) { return dim == 2 ? Box(IntVect{}, IntVect{{n - 1, n - 1, 0}}, 2) : Box(IntVect{}, IntVect{{n - 1, n - 1, n - 1}}, 3); }

// Kernels live in named functions: CUDA refuses device lambdas in a test
// body or inside another lambda.
// Valid points take a value of their position and component; the rest the
// sentinel.
void fill_pattern(const LevelField<double>& f, double offset) {
  f.fill(sentinel);
  const auto v = f.view();
  const int ncomp = f.ncomp();
  soma::amr::for_each_point(f, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    for (int c = 0; c < ncomp; ++c) v.patch(b)(i, j, k, c) = offset + 0.25 * i + 1009.5 * j + 30011.75 * k + 0.125 * c;
  });
}

// Each local box's stored values (ghosts included) equal those of the same
// box in the field holding every box.
void expect_same_boxes(const LevelField<double>& part, const LevelField<double>& whole) {
  const auto p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, part.view().data);
  const auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, whole.view().data);
  for (std::size_t b = 0; b < part.num_boxes(); ++b) {
    const auto g = static_cast<std::size_t>(part.global(b));
    ASSERT_EQ(part.grown(b), whole.grown(g));
    const auto n = static_cast<std::size_t>(part.grown(b).num_pts() * part.ncomp());
    ASSERT_EQ(std::memcmp(p.data() + part.offset(b), w.data() + whole.offset(g), n * sizeof(double)), 0)
        << "box " << g << " on rank " << part.rank();
  }
}

std::vector<IndexType> types(int dim) { return {IndexType::cell(), IndexType::node(dim), IndexType::face(0), IndexType::face(1)}; }

}  // namespace

// Halo fills over many ranks equal the fill with every box on one process,
// for every index type, ghost width and periodicity; the physical boundary
// regions of all ranks together are the serial ones.
TEST(AmrDistributed, HaloFillsEqualTheSerialFill) {
  const Comm c = Comm::world();
  for (const int dim : {2, 3}) {
    const Box domain = box(dim, dim == 2 ? 32 : 16);
    const BoxArray ba(soma::amr::chop({domain}, dim == 2 ? 8 : 4), domain);
    const Distribution dist = Distribution::along_curve(ba, c.size());
    for (const bool periodic : {false, true}) {
      const Periodicity per{{periodic, periodic, periodic}};
      for (const IndexType t : types(dim)) {
        for (const int ghosts : {1, 2}) {
          const soma::amr::HaloPlan whole_plan(ba, t, ghosts, per);
          const soma::amr::HaloPlan plan(c, ba, dist, t, ghosts, per);
          LevelField<double> whole(ba, t, 2, ghosts), part(ba, dist, c.rank(), t, 2, ghosts);
          fill_pattern(whole, 0.0);
          fill_pattern(part, 0.0);
          soma::amr::fill_halo(whole, whole_plan);
          soma::amr::fill_halo(part, plan);
          expect_same_boxes(part, whole);
          const auto regions = static_cast<std::int64_t>(plan.boundary().size());
          EXPECT_EQ(c.allreduce(regions, soma::Op::sum), static_cast<std::int64_t>(whole_plan.boundary().size()));
          for (const auto& r : plan.boundary()) ASSERT_LT(static_cast<std::size_t>(r.box), part.num_boxes());
        }
      }
    }
  }
}

// A rank sends one message to each rank owning a box its boxes fill, and
// receives one from each owner of a box filling its own.
TEST(AmrDistributed, OneMessagePerNeighbour) {
  const Comm c = Comm::world();
  const Box domain = box(2, 32);
  const BoxArray ba(soma::amr::chop({domain}, 8), domain);
  const Distribution dist = Distribution::along_curve(ba, c.size());
  const soma::amr::HaloPlan plan(c, ba, dist, IndexType::cell(), 1, Periodicity{{true, true, false}});
  std::set<int> to, from;
  std::int64_t sent = 0, received = 0, local = 0;
  for (const auto& op : plan.copies()) {
    const int s = dist.owner[static_cast<std::size_t>(op.src)], d = dist.owner[static_cast<std::size_t>(op.dst)];
    if (s == c.rank() && d != c.rank()) {
      to.insert(d);
      sent += op.region.num_pts();
    }
    if (d == c.rank() && s != c.rank()) {
      from.insert(s);
      received += op.region.num_pts();
    }
    if (s == c.rank() && d == c.rank()) local += op.region.num_pts();
  }
  EXPECT_EQ(plan.copy_plan().comm_plan().send_ranks(), std::vector<int>(to.begin(), to.end()));
  EXPECT_EQ(plan.copy_plan().comm_plan().recv_ranks(), std::vector<int>(from.begin(), from.end()));
  EXPECT_EQ(plan.copy_plan().num_sent(), sent);
  EXPECT_EQ(plan.copy_plan().num_received(), received);
  EXPECT_EQ(plan.copy_plan().num_copied(), local);
}

// A split fill sends what the valid points held when it began: changing them
// before end() changes no ghost.
TEST(AmrDistributed, SplitFillsSendWhatTheyBeganWith) {
  const Comm c = Comm::world();
  const Box domain = box(3, 16);
  const BoxArray ba(soma::amr::chop({domain}, 4), domain);
  const Distribution dist = Distribution::along_curve(ba, c.size());
  const Periodicity per{{true, false, true}};
  const soma::amr::HaloPlan whole_plan(ba, IndexType::cell(), 2, per);
  const soma::amr::HaloPlan plan(c, ba, dist, IndexType::cell(), 2, per);
  LevelField<double> whole(ba, IndexType::cell(), 1, 2), part(ba, dist, c.rank(), IndexType::cell(), 1, 2);
  fill_pattern(whole, 0.0);
  soma::amr::fill_halo(whole, whole_plan);
  fill_pattern(part, 0.0);
  auto pending = soma::amr::begin_fill_halo(part, plan);
  // Interior work: the valid points change while the messages fly.
  LevelField<double> scratch(ba, dist, c.rank(), IndexType::cell(), 1, 2);
  fill_pattern(scratch, 5.0);
  {
    const auto from = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scratch.view().data);
    const auto to = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, part.view().data);
    // Valid points only: ghosts keep whatever the local copies wrote.
    for (std::size_t b = 0; b < part.num_boxes(); ++b) {
      const Box g = part.grown(b), v = part.valid(b);
      for (std::int64_t q = 0; q < g.num_pts(); ++q) {
        const IntVect x = soma::amr::detail::point_of(g, q);
        if (v.contains(Box(x, x, 3))) to(static_cast<std::size_t>(part.offset(b) + q)) = from(static_cast<std::size_t>(part.offset(b) + q));
      }
    }
    Kokkos::deep_copy(part.view().data, to);
  }
  pending.end();
  pending.end();
  // Ghosts are the old pattern; valid points the new one.
  fill_pattern(whole, 5.0);
  const auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, whole.view().data);
  LevelField<double> old(ba, IndexType::cell(), 1, 2);
  fill_pattern(old, 0.0);
  soma::amr::fill_halo(old, whole_plan);
  const auto o = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, old.view().data);
  const auto p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, part.view().data);
  for (std::size_t b = 0; b < part.num_boxes(); ++b) {
    const auto gb = static_cast<std::size_t>(part.global(b));
    const Box g = part.grown(b), v = part.valid(b);
    for (std::int64_t q = 0; q < g.num_pts(); ++q) {
      const IntVect x = soma::amr::detail::point_of(g, q);
      const double expected = v.contains(Box(x, x, 3)) ? w(static_cast<std::size_t>(whole.offset(gb) + q))
                                                        : o(static_cast<std::size_t>(old.offset(gb) + q));
      ASSERT_EQ(p(static_cast<std::size_t>(part.offset(b) + q)), expected) << gb << " " << q;
    }
  }
}

// A copy between two distributions of the same boxes moves every valid
// point to its new owner; a field distributed otherwise is refused.
TEST(AmrDistributed, CopiesMoveBoxesBetweenDistributions) {
  const Comm c = Comm::world();
  const Box domain = box(2, 32);
  const BoxArray ba(soma::amr::chop({domain}, 8), domain);
  const Distribution along = Distribution::along_curve(ba, c.size());
  Distribution reversed = along;
  for (std::size_t i = 0; i < ba.size(); ++i) reversed.owner[i] = static_cast<int>((i * 7 + 3) % static_cast<std::size_t>(c.size()));
  std::vector<BoxList> regions;
  for (const Box& b : ba.boxes()) regions.push_back({b});
  const soma::amr::CopyPlan plan(c, ba, along, regions, reversed, IndexType::cell(), Periodicity{});
  LevelField<double> src(ba, along, c.rank(), IndexType::cell(), 3, 1), dst(ba, reversed, c.rank(), IndexType::cell(), 3, 1);
  LevelField<double> whole(ba, IndexType::cell(), 3, 1);
  fill_pattern(src, 0.0);
  dst.fill(sentinel);
  fill_pattern(whole, 0.0);
  soma::amr::copy(src, dst, plan);
  expect_same_boxes(dst, whole);
  if (along.owner != reversed.owner) EXPECT_THROW(soma::amr::copy(dst, src, plan), std::invalid_argument);
  const Distribution short_one{std::vector<int>(ba.size() - 1, 0)};
  EXPECT_THROW(soma::amr::CopyPlan(c, ba, short_one, regions, reversed, IndexType::cell(), Periodicity{}),
               std::invalid_argument);
  Distribution outside = along;
  outside.owner[0] = c.size();
  EXPECT_THROW(soma::amr::CopyPlan(c, ba, along, regions, outside, IndexType::cell(), Periodicity{}), std::invalid_argument);
}
