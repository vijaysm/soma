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
#include "soma/amr/flux_register.hpp"
#include "soma/amr/halo.hpp"
#include "soma/amr/interlevel.hpp"
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

namespace {

// Two levels: a coarse level over the whole domain and a fine level over its
// middle half, properly nested, each cut along the curve over the ranks.
struct TwoLevels {
  BoxArray coarse, fine;
  Distribution coarse_owner, fine_owner;
};

TwoLevels two_levels(int dim, int nranks) {
  const int n = dim == 2 ? 32 : 16;
  const Box domain = box(dim, n);
  const Box middle = dim == 2 ? Box(IntVect{{n / 4, n / 4, 0}}, IntVect{{3 * n / 4 - 1, 3 * n / 4 - 1, 0}}, 2)
                              : Box(IntVect{{n / 4, n / 4, n / 4}}, IntVect{{3 * n / 4 - 1, 3 * n / 4 - 1, 3 * n / 4 - 1}}, 3);
  TwoLevels t;
  t.coarse = BoxArray(soma::amr::chop({domain}, dim == 2 ? 8 : 4), domain);
  t.fine = BoxArray(soma::amr::chop({middle.refine(2)}, 8), domain.refine(2));
  t.coarse_owner = Distribution::along_curve(t.coarse, nranks);
  t.fine_owner = Distribution::along_curve(t.fine, nranks);
  return t;
}

}  // namespace

// The coarse-fine fill over many ranks equals the serial one for every
// interpolation and index type, after halo fills on both levels.
TEST(AmrDistributed, CoarseFineFillsEqualTheSerialFill) {
  const Comm c = Comm::world();
  using soma::amr::Interp;
  for (const int dim : {2, 3}) {
    const TwoLevels t = two_levels(dim, c.size());
    for (const bool periodic : {false, true}) {
      const Periodicity per{{periodic, periodic, periodic}};
      for (const IndexType type : types(dim)) {
        for (const Interp interp : {Interp::piecewise_constant, Interp::linear, Interp::quadratic}) {
          LevelField<double> cw(t.coarse, type, 2, 1), cp(t.coarse, t.coarse_owner, c.rank(), type, 2, 1);
          LevelField<double> fw(t.fine, type, 2, 2), fp(t.fine, t.fine_owner, c.rank(), type, 2, 2);
          fill_pattern(cw, 0.0);
          fill_pattern(cp, 0.0);
          fill_pattern(fw, 3.0);
          fill_pattern(fp, 3.0);
          soma::amr::fill_halo(cw, soma::amr::HaloPlan(t.coarse, type, 1, per));
          soma::amr::fill_halo(cp, soma::amr::HaloPlan(c, t.coarse, t.coarse_owner, type, 1, per));
          soma::amr::fill_halo(fw, soma::amr::HaloPlan(t.fine, type, 2, per));
          soma::amr::fill_halo(fp, soma::amr::HaloPlan(c, t.fine, t.fine_owner, type, 2, per));
          const soma::amr::CoarseFinePlan whole(t.fine, t.coarse, type, 2, 2, per, interp);
          const soma::amr::CoarseFinePlan part(c, t.fine, t.fine_owner, t.coarse, t.coarse_owner, type, 2, 2, per, interp);
          soma::amr::fill_coarse_fine(fw, cw, whole);
          soma::amr::fill_coarse_fine(fp, cp, part);
          expect_same_boxes(fp, fw);
          if (t.fine_owner.owner != std::vector<int>(t.fine.size(), 0)) {
            EXPECT_THROW(soma::amr::fill_coarse_fine(fw, cw, part), std::invalid_argument);
          }
        }
      }
    }
  }
}

// Restriction over many ranks equals the serial restriction.
TEST(AmrDistributed, AverageDownEqualsTheSerialRestriction) {
  const Comm c = Comm::world();
  for (const int dim : {2, 3}) {
    const TwoLevels t = two_levels(dim, c.size());
    for (const IndexType type : types(dim)) {
      LevelField<double> cw(t.coarse, type, 2, 1), cp(t.coarse, t.coarse_owner, c.rank(), type, 2, 1);
      LevelField<double> fw(t.fine, type, 2, 1), fp(t.fine, t.fine_owner, c.rank(), type, 2, 1);
      fill_pattern(cw, 0.0);
      fill_pattern(cp, 0.0);
      fill_pattern(fw, 7.0);
      fill_pattern(fp, 7.0);
      soma::amr::average_down(fw, cw, 2);
      soma::amr::average_down(fp, cp, soma::amr::AverageDownPlan(c, t.fine, t.fine_owner, t.coarse, t.coarse_owner, type, 2));
      expect_same_boxes(cp, cw);
    }
  }
}

namespace {

// A face field normal to dir with a pattern scaled by s.
LevelField<double> flux(const BoxArray& boxes, const Distribution* owner, int rank, int dir, double s) {
  LevelField<double> f = owner ? LevelField<double>(boxes, *owner, rank, IndexType::face(dir), 2, 0)
                               : LevelField<double>(boxes, IndexType::face(dir), 2, 0);
  fill_pattern(f, s);
  return f;
}

}  // namespace

// Refluxing over many ranks equals serial refluxing: fine sums move to the
// coarse owners and correct the same cells by the same amounts.
TEST(AmrDistributed, RefluxEqualsTheSerialReflux) {
  const Comm c = Comm::world();
  for (const int dim : {2, 3}) {
    const TwoLevels t = two_levels(dim, c.size());
    for (const bool periodic : {false, true}) {
      const Periodicity per{{periodic, periodic, periodic}};
      const soma::amr::FluxRegister whole(t.fine, t.coarse, 2, per, 2);
      const soma::amr::FluxRegister part(c, t.fine, t.fine_owner, t.coarse, t.coarse_owner, 2, per, 2);
      EXPECT_EQ(part.num_faces(), whole.num_faces());
      EXPECT_EQ(c.allreduce(part.num_coarse_rows(), soma::Op::sum), whole.num_faces());
      EXPECT_EQ(c.allreduce(part.num_fine_rows(), soma::Op::sum), whole.num_faces());
      LevelField<double> sw(t.coarse, IndexType::cell(), 2, 0), sp(t.coarse, t.coarse_owner, c.rank(), IndexType::cell(), 2, 0);
      fill_pattern(sw, 1.0);
      fill_pattern(sp, 1.0);
      whole.reset();
      part.reset();
      for (int d = 0; d < dim; ++d) {
        whole.set_coarse(d, flux(t.coarse, nullptr, 0, d, 0.5), 0.1);
        part.set_coarse(d, flux(t.coarse, &t.coarse_owner, c.rank(), d, 0.5), 0.1);
        for (const double step : {0.25, 0.75}) {
          whole.add_fine(d, flux(t.fine, nullptr, 0, d, step), 0.05);
          part.add_fine(d, flux(t.fine, &t.fine_owner, c.rank(), d, step), 0.05);
        }
      }
      whole.reflux(sw);
      part.reflux(sp);
      expect_same_boxes(sp, sw);
      if (c.size() > 1) EXPECT_THROW(part.reflux(sw), std::invalid_argument);
    }
  }
}
