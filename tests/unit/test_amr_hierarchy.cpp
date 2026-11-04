// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The hierarchy: levels checked for alignment and proper nesting, halos
// filled level by level exact for linear data, restriction across every
// level.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/amr/hierarchy.hpp"

namespace {

using soma::amr::Box;
using soma::amr::BoxList;
using soma::amr::Geometry;
using soma::amr::Hierarchy;
using soma::amr::HierarchyOptions;
using soma::amr::IndexType;
using soma::amr::Interp;
using soma::amr::IntVect;

constexpr double sentinel = -1.0e30;

KOKKOS_INLINE_FUNCTION double pos(int p, bool nodal, double h) { return (p + (nodal ? 0.0 : 0.5)) * h; }
KOKKOS_INLINE_FUNCTION double linear(double x, double y, double z, int c) { return 0.3 + 1.7 * x - 0.9 * y + 0.4 * z + 10.0 * c; }
KOKKOS_INLINE_FUNCTION double at(const IntVect& p, IndexType t, double h, int c, int dim) {
  return linear(pos(p[0], t.is_nodal(0), h), pos(p[1], t.is_nodal(1), h), dim == 3 ? pos(p[2], t.is_nodal(2), h) : 0.0, c);
}

// Three levels; level 1 and 2 against the low domain corner and inside.
struct Config {
  Geometry geom;
  HierarchyOptions opts;
  std::vector<BoxList> fine;
};

Config config(int dim) {
  Config s;
  s.opts.max_level = 2;
  s.opts.blocking_factor = 4;
  s.opts.max_box = 8;
  if (dim == 2) {
    s.geom.domain = Box(IntVect{}, IntVect{{31, 31, 0}}, 2);
    s.fine = {{Box(IntVect{{8, 8, 0}}, IntVect{{23, 15, 0}}, 2), Box(IntVect{{0, 0, 0}}, IntVect{{7, 7, 0}}, 2)},
              {Box(IntVect{{24, 20, 0}}, IntVect{{39, 27, 0}}, 2), Box(IntVect{{0, 0, 0}}, IntVect{{7, 7, 0}}, 2)}};
  } else {
    s.geom.domain = Box(IntVect{}, IntVect{{15, 15, 15}});
    s.fine = {{Box(IntVect{}, IntVect{{15, 15, 15}})}, {Box(IntVect{{4, 8, 8}}, IntVect{{11, 15, 15}}), Box(IntVect{}, IntVect{{7, 7, 7}})}};
  }
  return s;
}

std::vector<IndexType> types(int dim) { return {IndexType::cell(), IndexType::node(dim), IndexType::face(0)}; }

// Sets the points of `region` of box b of u to the linear data.
void set_linear(const soma::amr::LevelField<double>& u, int b, const Box& region, double h, int dim) {
  const auto v = u.view();
  const IndexType t = u.type();
  const int ncomp = u.ncomp();
  Kokkos::parallel_for(
      "set_linear", Kokkos::RangePolicy<Kokkos::IndexType<std::int64_t>>(0, region.num_pts()), KOKKOS_LAMBDA(std::int64_t q) {
        const IntVect p = soma::amr::detail::point_of(region, q);
        for (int c = 0; c < ncomp; ++c) v.patch(b)(p[0], p[1], p[2], c) = at(p, t, h, c, dim);
      });
  Kokkos::fence();
}

}  // namespace

TEST(AmrHierarchy, LevelsAreChecked) {
  for (const int dim : {2, 3}) {
    const Config s = config(dim);
    const Hierarchy h(s.geom, s.opts, s.fine);
    EXPECT_EQ(h.num_levels(), 3);
    EXPECT_EQ(h.finest_level(), 2);
    EXPECT_EQ(h.boxes(0).size(), soma::amr::chop({s.geom.domain}, 8, 4).size());
    EXPECT_TRUE(soma::amr::disjoint(h.boxes(0).boxes()));
    EXPECT_EQ(h.boxes(0).num_pts(), s.geom.domain.num_pts());
    EXPECT_EQ(h.domain(2), s.geom.domain.refine(4));
    EXPECT_DOUBLE_EQ(h.dx(1, 0), 1.0 / (2 * s.geom.domain.length(0)));
    EXPECT_EQ(soma::amr::num_pts(h.boxes(1).boxes()), soma::amr::num_pts(s.fine[0]));
    EXPECT_EQ(Hierarchy(s.geom, s.opts).num_levels(), 1);
    EXPECT_EQ(Hierarchy(s.geom, s.opts, {s.fine[0], {}}).num_levels(), 2);
    const auto bad_opts = [&](auto change) {
      HierarchyOptions o = s.opts;
      change(o);
      EXPECT_THROW(Hierarchy(s.geom, o), std::invalid_argument);
    };
    bad_opts([](HierarchyOptions& o) { o.ratio = 3; });
    bad_opts([](HierarchyOptions& o) { o.ratio = 8; });
    bad_opts([](HierarchyOptions& o) { o.ratio = 8, o.blocking_factor = 8; });
    bad_opts([](HierarchyOptions& o) { o.blocking_factor = 6; });
    bad_opts([](HierarchyOptions& o) { o.max_box = 10; });
    bad_opts([](HierarchyOptions& o) { o.max_box = 2; });
    bad_opts([](HierarchyOptions& o) { o.max_level = -1; });
    bad_opts([](HierarchyOptions& o) { o.nesting = -1; });
    bad_opts([](HierarchyOptions& o) { o.efficiency = 0; });
    // Ratio 4 and blocks of 6 (a domain of whole blocks, max_box 12): a
    // block is not whole coarse cells.
    Geometry six = s.geom;
    six.domain = Box(IntVect{}, IntVect{{23, 23, 23}}, dim);
    HierarchyOptions r4 = s.opts;
    r4.ratio = 4;
    r4.max_box = 12;
    r4.blocking_factor = 4;
    EXPECT_EQ(Hierarchy(Geometry{Box(IntVect{}, IntVect{{23, 23, 23}}, dim)}, r4).num_levels(), 1);
    r4.blocking_factor = 6;
    EXPECT_THROW(Hierarchy(six, r4), std::invalid_argument);
    Geometry g = s.geom;
    g.domain = g.domain.grow(0, 0, -1);
    EXPECT_THROW(Hierarchy(g, s.opts), std::invalid_argument);
    g.domain = s.geom.domain.shift(IntVect{{2, 0, 0}});
    EXPECT_THROW(Hierarchy(g, s.opts), std::invalid_argument);
    HierarchyOptions one = s.opts;
    one.max_level = 1;
    EXPECT_THROW(Hierarchy(s.geom, one, s.fine), std::invalid_argument);
    // Level 1 empty under level 2.
    EXPECT_THROW(Hierarchy(s.geom, s.opts, {{}, s.fine[1]}), std::invalid_argument);
    // Not aligned to the blocking factor.
    EXPECT_THROW(Hierarchy(s.geom, s.opts, {{s.fine[0][0].shift(IntVect{{2, 0, 0}})}}), std::invalid_argument);
    // Overlapping.
    EXPECT_THROW(Hierarchy(s.geom, s.opts, {{s.fine[0][0], s.fine[0][0]}}), std::invalid_argument);
    // Level 2 moved by a block: its grown coarsening leaves level 1.
    std::vector<BoxList> low = s.fine;
    low[1][0] = low[1][0].shift(dim == 2 ? IntVect{{0, -4, 0}} : IntVect{{0, 0, 16}});
    EXPECT_THROW(Hierarchy(s.geom, s.opts, low), std::invalid_argument);
    // Nested with no gap: refused at nesting 1, accepted at nesting 0.
    std::vector<BoxList> tight = s.fine;
    tight[1][0] = dim == 2 ? Box(IntVect{{16, 16, 0}}, IntVect{{31, 31, 0}}, 2) : Box(IntVect{{0, 8, 8}}, IntVect{{31, 23, 23}});
    EXPECT_THROW(Hierarchy(s.geom, s.opts, tight), std::invalid_argument);
    HierarchyOptions loose = s.opts;
    loose.nesting = 0;
    EXPECT_EQ(Hierarchy(s.geom, loose, tight).num_levels(), 3);
  }
}

TEST(AmrHierarchy, NestingWrapsPeriodicSides) {
  Geometry g{Box(IntVect{}, IntVect{{31, 31, 0}}, 2)};
  HierarchyOptions o;
  o.max_level = 2;
  o.blocking_factor = 4;
  o.max_box = 16;
  // Level 2 against the low x side: nested only if level 1 continues across
  // the periodic side.
  const BoxList l2{Box(IntVect{{0, 16, 0}}, IntVect{{7, 23, 0}}, 2)};
  const BoxList l1{Box(IntVect{{0, 4, 0}}, IntVect{{15, 15, 0}}, 2)};
  EXPECT_EQ(Hierarchy(g, o, {l1, l2}).num_levels(), 3);
  g.periodic.periodic[0] = true;
  EXPECT_THROW(Hierarchy(g, o, {l1, l2}), std::invalid_argument);
  BoxList across = l1;
  across.push_back(Box(IntVect{{60, 4, 0}}, IntVect{{63, 15, 0}}, 2));
  EXPECT_EQ(Hierarchy(g, o, {across, l2}).num_levels(), 3);
  // wrap: a box across the high side comes back on the low side.
  const BoxList w = soma::amr::detail::wrap({Box(IntVect{{30, 2, 0}}, IntVect{{33, 3, 0}}, 2)}, g.domain, g.periodic);
  EXPECT_EQ(soma::amr::num_pts(w), 8);
  EXPECT_TRUE(soma::amr::covers(w, Box(IntVect{{0, 2, 0}}, IntVect{{1, 3, 0}}, 2)));
  EXPECT_TRUE(soma::amr::covers(w, Box(IntVect{{30, 2, 0}}, IntVect{{31, 3, 0}}, 2)));
}

TEST(AmrHierarchy, FillHaloIsExactForLinearData) {
  for (const int dim : {2, 3}) {
    const Config s = config(dim);
    const Hierarchy h(s.geom, s.opts, s.fine);
    for (const IndexType t : types(dim)) {
      for (const int g : {1, 2}) {
        SCOPED_TRACE(testing::Message() << "dim " << dim << " type " << int(t.nodal) << " ghosts " << g);
        const auto u = h.make_field<double>(t, 2, g);
        for (int l = 0; l < h.num_levels(); ++l) {
          const auto& f = u[static_cast<std::size_t>(l)];
          f.fill(sentinel);
          for (std::size_t b = 0; b < f.num_boxes(); ++b) set_linear(f, static_cast<int>(b), f.valid(b), h.dx(l, 0), dim);
        }
        int calls = 0;
        h.fill_halo(u, Interp::linear, [&](int level, int box, const Box& region, int, int) {
          ++calls;
          set_linear(u[static_cast<std::size_t>(level)], box, region, h.dx(level, 0), dim);
        });
        EXPECT_GT(calls, 0);
        for (int l = 0; l < h.num_levels(); ++l) {
          const auto& f = u[static_cast<std::size_t>(l)];
          auto data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f.view().data);
          for (std::size_t b = 0; b < f.num_boxes(); ++b) {
            const Box gr = f.grown(b);
            for (std::int64_t q = 0; q < gr.num_pts(); ++q) {
              const IntVect p = soma::amr::detail::point_of(gr, q);
              for (int c = 0; c < 2; ++c) {
                ASSERT_NEAR(data(static_cast<std::size_t>(f.offset(b) + q + gr.num_pts() * c)), at(p, t, h.dx(l, 0), c, dim), 1e-11)
                    << "level " << l << " box " << b << " point " << p[0] << "," << p[1] << "," << p[2];
              }
            }
          }
        }
      }
    }
    // A field from another hierarchy is refused.
    const auto other = Hierarchy(s.geom, s.opts).make_field<double>(IndexType::cell(), 1, 1);
    EXPECT_THROW(h.fill_halo(other), std::invalid_argument);
    auto mixed = h.make_field<double>(IndexType::cell(), 1, 1);
    mixed[1] = h.make_level_field<double>(1, IndexType::cell(), 1, 2);
    EXPECT_THROW(h.average_down(mixed), std::invalid_argument);
  }
}

TEST(AmrHierarchy, AverageDownReachesLevelZero) {
  for (const int dim : {2, 3}) {
    const Config s = config(dim);
    const Hierarchy h(s.geom, s.opts, s.fine);
    const auto u = h.make_field<double>(IndexType::cell(), 1, 0);
    for (int l = 0; l < h.num_levels(); ++l) {
      const auto& f = u[static_cast<std::size_t>(l)];
      f.fill(sentinel);
      if (l == h.finest_level()) {
        for (std::size_t b = 0; b < f.num_boxes(); ++b) set_linear(f, static_cast<int>(b), f.valid(b), h.dx(l, 0), dim);
      }
    }
    h.average_down(u);
    // Level 0 under level 2 (through level 1) is the linear data again; the
    // rest of level 0 is untouched.
    const BoxList under2 = soma::amr::coarsen(h.boxes(2).boxes(), 4);
    const auto& f = u[0];
    auto data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f.view().data);
    int checked = 0;
    for (std::size_t b = 0; b < f.num_boxes(); ++b) {
      const Box v = f.valid(b);
      for (std::int64_t q = 0; q < v.num_pts(); ++q) {
        const IntVect p = soma::amr::detail::point_of(v, q);
        bool under = false;
        for (const Box& x : under2) under = under || x.contains(p);
        const double got = data(static_cast<std::size_t>(f.offset(b) + q));
        if (under) {
          ++checked;
          ASSERT_NEAR(got, at(p, IndexType::cell(), h.dx(0, 0), 0, dim), 1e-12);
        } else if (!soma::amr::covers(soma::amr::coarsen(h.boxes(1).boxes(), 2), Box(p, p, dim))) {
          ASSERT_EQ(got, sentinel);
        }
      }
    }
    EXPECT_EQ(checked, soma::amr::num_pts(under2));
  }
}
