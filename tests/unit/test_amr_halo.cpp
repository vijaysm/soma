// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Same-level halo fill against brute force: on a level with a hole, every
// ghost point gets the value of the point it shows (through periodic images
// too) when some box covers that point, the physical-boundary value outside
// the domain, and keeps its old value otherwise; valid points are untouched.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/amr/box_list.hpp"
#include "soma/amr/halo.hpp"

namespace {

using soma::amr::Box;
using soma::amr::BoxArray;
using soma::amr::BoxList;
using soma::amr::IndexType;
using soma::amr::IntVect;
using soma::amr::Periodicity;

constexpr double sentinel = -1.0e30;

// p moved into the domain through periodic directions (nodes modulo the
// length too, so node L and node 0 are one point).
IntVect wrap(IntVect p, const Box& domain, Periodicity per) {
  for (int d = 0; d < domain.dim(); ++d) {
    if (!per[d]) continue;
    const int len = domain.length(d);
    p[d] = domain.lo(d) + ((p[d] - domain.lo(d)) % len + len) % len;
  }
  return p;
}

KOKKOS_INLINE_FUNCTION double value(const IntVect& w, int c) { return w[0] + 100.0 * w[1] + 10000.0 * w[2] + 1.0e6 * c; }
KOKKOS_INLINE_FUNCTION double boundary_value(int i, int j, int k, int dir, int side) {
  return -1.0 - i - 10.0 * j - 100.0 * k - 1000.0 * (2 * dir + side);
}

// Kernels live in named functions: CUDA refuses device lambdas in a test
// body (a private member function) or inside another lambda.
// Fills ghost region r of box b with boundary values.
void set_boundary(const soma::amr::LevelField<double>& f, int b, const Box& r, int dir, int side) {
  const auto v = f.view();
  const int ncomp = f.ncomp();
  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::IndexType<std::int64_t>>(0, r.num_pts()), KOKKOS_LAMBDA(std::int64_t q) {
    const IntVect p = soma::amr::detail::point_of(r, q);
    for (int c = 0; c < ncomp; ++c) v.patch(b)(p[0], p[1], p[2], c) = boundary_value(p[0], p[1], p[2], dir, side) - c;
  });
  Kokkos::fence();
}

// Valid points +value, ghost points -value.
void set_signed(const soma::amr::LevelField<double>& a) {
  const auto av = a.view();
  soma::amr::for_each_point(a, soma::amr::Region::grown, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    av.patch(b)(i, j, k) = av.valid(b).contains(IntVect{{i, j, k}}) ? value(IntVect{{i, j, k}}, 0) : -value(IntVect{{i, j, k}}, 0);
  });
}

struct Case {
  int dim;
  IndexType type;
  int ghosts;
  Periodicity per;
};

void check(const Case& cs) {
  const int dim = cs.dim;
  const Box domain = dim == 2 ? Box(IntVect{}, IntVect{{23, 17, 0}}, 2) : Box(IntVect{{-4, 0, 0}}, IntVect{{7, 9, 11}});
  const Box hole = Box(IntVect{{1, 3, 4}}, IntVect{{4, 6, 6}}, dim);
  const BoxArray ba(soma::amr::chop(soma::amr::complement(domain, {hole}), 5), domain);
  const int ncomp = 2;
  soma::amr::LevelField<double> f(ba, cs.type, ncomp, cs.ghosts);
  f.fill(sentinel);
  const auto v = f.view();
  const Box dom = domain;
  const Periodicity per = cs.per;
  // Valid points hold the value of their wrapped point, so shared nodes agree.
  soma::amr::for_each_point(f, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    IntVect w{{i, j, k}};
    for (int d = 0; d < dom.dim(); ++d) {
      if (!per[d]) continue;
      const int len = dom.length(d);
      w[d] = dom.lo(d) + ((w[d] - dom.lo(d)) % len + len) % len;
    }
    for (int c = 0; c < ncomp; ++c) v.patch(b)(i, j, k, c) = value(w, c);
  });
  const soma::amr::HaloPlan plan(ba, cs.type, cs.ghosts, cs.per);
  int calls = 0;
  soma::amr::fill_halo(f, plan, [&](int b, const Box& r, int dir, int side) {
    ++calls;
    set_boundary(f, b, r, dir, side);
  });
  EXPECT_EQ(calls, static_cast<int>(plan.boundary().size()));

  // Brute force on the host.
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v.data);
  const Box domain_t = domain.convert(cs.type);
  std::vector<IntVect> shifts{IntVect{}};
  for (int d = 0; d < dim; ++d) {
    if (!cs.per[d]) continue;
    const auto n = shifts.size();
    for (const int s : {-1, 1}) {
      for (std::size_t i = 0; i < n; ++i) {
        IntVect x = shifts[i];
        x[d] = s * domain.length(d);
        shifts.push_back(x);
      }
    }
  }
  std::int64_t filled = 0, kept = 0, outside = 0;
  for (std::size_t b = 0; b < f.num_boxes(); ++b) {
    const Box g = f.grown(b), valid = f.valid(b);
    for (std::int64_t q = 0; q < g.num_pts(); ++q) {
      const IntVect p = soma::amr::detail::point_of(g, q);
      int out_dir = -1, out_side = 0;
      for (int d = 0; d < dim; ++d) {
        if (cs.per[d]) continue;
        if (p[d] < domain_t.lo(d) && out_dir < 0) out_dir = d, out_side = 0;
        if (p[d] > domain_t.hi(d) && out_dir < 0) out_dir = d, out_side = 1;
      }
      bool covered = false;
      for (std::size_t s = 0; s < ba.size() && !covered; ++s) {
        for (const IntVect& sh : shifts) covered = covered || ba[s].convert(cs.type).contains(p - sh);
      }
      for (int c = 0; c < ncomp; ++c) {
        const double got = h(static_cast<std::size_t>(f.offset(b) + q + g.num_pts() * c));
        if (valid.contains(p) || (out_dir < 0 && covered)) {
          ASSERT_EQ(got, value(wrap(p, domain, cs.per), c)) << "box " << b << " point " << p[0] << "," << p[1] << "," << p[2];
          filled += valid.contains(p) ? 0 : 1;
        } else if (out_dir >= 0) {
          // Some boundary callback wrote it (corners: any of their directions).
          bool any = false;
          for (int d = 0; d < dim; ++d) {
            for (int side = 0; side < 2; ++side) any = any || got == boundary_value(p[0], p[1], p[2], d, side) - c;
          }
          ASSERT_TRUE(any) << "box " << b << " point " << p[0] << "," << p[1] << "," << p[2];
          ++outside;
        } else {
          ASSERT_EQ(got, sentinel);
          ++kept;
        }
      }
    }
  }
  EXPECT_GT(filled, 0);
  EXPECT_GT(kept, 0);  // ghosts over the hole
  // The copies of a box are disjoint, outside its valid points, and as many
  // points as were filled.
  std::int64_t copied = 0;
  for (std::size_t i = 0; i < plan.copies().size(); ++i) {
    const auto& a = plan.copies()[i];
    EXPECT_FALSE(f.valid(static_cast<std::size_t>(a.dst)).intersects(a.region));
    EXPECT_TRUE(f.grown(static_cast<std::size_t>(a.dst)).contains(a.region));
    EXPECT_TRUE(f.valid(static_cast<std::size_t>(a.src)).contains(a.region.shift(IntVect{} - a.shift)));
    for (std::size_t j = i + 1; j < plan.copies().size(); ++j) {
      const auto& o = plan.copies()[j];
      if (o.dst == a.dst) EXPECT_FALSE(a.region.intersects(o.region));
    }
    copied += a.region.num_pts();
  }
  EXPECT_EQ(copied * ncomp, filled);
  // What stayed as it was is what the plan lists as uncovered.
  std::int64_t uncovered = 0;
  for (std::size_t b = 0; b < f.num_boxes(); ++b) {
    EXPECT_TRUE(soma::amr::disjoint(plan.uncovered()[b]));
    uncovered += soma::amr::num_pts(plan.uncovered()[b]);
  }
  EXPECT_EQ(uncovered * ncomp, kept);
  EXPECT_EQ(plan.num_copied(), copied);
  if (outside == 0) EXPECT_TRUE(plan.boundary().empty());
}

}  // namespace

TEST(AmrHalo, SameLevelFillMatchesBruteForce) {
  for (const int dim : {2, 3}) {
    std::vector<IndexType> types{IndexType::cell(), IndexType::node(dim), IndexType::face(0), IndexType::face(1)};
    if (dim == 3) types.push_back(IndexType::face(2));
    for (const IndexType t : types) {
      for (const int g : {1, 2}) {
        for (const Periodicity per : {Periodicity{}, Periodicity{{true, false, false}}, Periodicity{{true, true, true}}}) {
          SCOPED_TRACE(testing::Message() << "dim " << dim << " type " << int(t.nodal) << " ghosts " << g << " periodic "
                                          << per[0] << per[1] << per[2]);
          check({dim, t, g, per});
        }
      }
    }
  }
}

TEST(AmrHalo, RejectsAMismatchedPlan) {
  const Box domain(IntVect{}, IntVect{{7, 7, 0}}, 2);
  const BoxArray ba({domain}, domain);
  soma::amr::LevelField<double> f(ba, IndexType::cell(), 1, 1);
  EXPECT_THROW(soma::amr::fill_halo(f, soma::amr::HaloPlan(ba, IndexType::node(2), 1, {})), std::invalid_argument);
  EXPECT_THROW(soma::amr::fill_halo(f, soma::amr::HaloPlan(ba, IndexType::cell(), 2, {})), std::invalid_argument);
  EXPECT_THROW(soma::amr::HaloPlan(ba, IndexType::cell(), -1, {}), std::invalid_argument);
  // One periodic box fills its own ghosts from its far side.
  const soma::amr::HaloPlan self(ba, IndexType::cell(), 1, Periodicity{{true, true, false}});
  EXPECT_EQ(self.num_copied(), 10 * 10 - 64);
  EXPECT_TRUE(self.boundary().empty());
}

// A copy between two levels takes valid points first and the source's ghost
// points (filled by the caller) after them; what neither covers is left.
TEST(AmrCopy, ValidPointsThenGhostPoints) {
  const Box domain(IntVect{}, IntVect{{15, 15, 0}}, 2);
  const BoxArray src({Box(IntVect{{0, 0, 0}}, IntVect{{3, 7, 0}}, 2), Box(IntVect{{4, 0, 0}}, IntVect{{7, 7, 0}}, 2)}, domain);
  soma::amr::LevelField<double> a(src, IndexType::cell(), 1, 2);
  set_signed(a);
  const BoxArray dst({Box(IntVect{{2, 6, 0}}, IntVect{{11, 11, 0}}, 2)}, domain);
  soma::amr::LevelField<double> d(dst, IndexType::cell(), 1, 0);
  d.fill(sentinel);
  for (const int ghosts : {0, 2}) {
    d.fill(sentinel);
    const soma::amr::CopyPlan plan(src, {{dst[0]}}, IndexType::cell(), {}, ghosts);
    soma::amr::copy(a, d, plan);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d.view().data);
    std::int64_t left = 0;
    for (std::int64_t q = 0; q < dst[0].num_pts(); ++q) {
      const IntVect p = soma::amr::detail::point_of(dst[0], q);
      const bool valid = p[0] <= 7 && p[1] <= 7;
      const bool ghost = ghosts > 0 && p[0] <= 9 && p[1] <= 9;
      const double want = valid ? value(p, 0) : ghost ? -value(p, 0) : sentinel;
      ASSERT_EQ(h(static_cast<std::size_t>(q)), want) << p[0] << "," << p[1] << " ghosts " << ghosts;
      left += valid || ghost ? 0 : 1;
    }
    EXPECT_EQ(soma::amr::num_pts(plan.uncovered()[0]), left);
  }
  EXPECT_THROW(soma::amr::copy(a, d, soma::amr::CopyPlan(src, {{dst[0]}}, IndexType::cell(), {}, 3)), std::invalid_argument);
  // Ghost points reach into bins the valid points do not: a 4 x 4 box binned
  // by 4, read two cells to its right.
  const BoxArray small({Box(IntVect{}, IntVect{{3, 3, 0}}, 2)}, domain);
  soma::amr::LevelField<double> s4(small, IndexType::cell(), 1, 2);
  s4.fill(5.0);
  const BoxArray strip({Box(IntVect{{5, 0, 0}}, IntVect{{5, 3, 0}}, 2)}, domain);
  soma::amr::LevelField<double> t(strip, IndexType::cell(), 1, 0);
  t.fill(sentinel);
  const soma::amr::CopyPlan reach(small, {{strip[0]}}, IndexType::cell(), {}, 2);
  soma::amr::copy(s4, t, reach);
  EXPECT_TRUE(reach.uncovered()[0].empty());
  auto th = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, t.view().data);
  for (std::size_t q = 0; q < th.extent(0); ++q) EXPECT_EQ(th(q), 5.0);
  EXPECT_THROW(soma::amr::CopyPlan(src, {{dst[0]}}, IndexType::cell(), {}, -1), std::invalid_argument);
}
