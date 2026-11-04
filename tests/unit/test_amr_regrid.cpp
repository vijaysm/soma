// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The adapt cycle: new levels cover every tag, are aligned, chopped and
// properly nested (across periodic sides too); data move to them exactly
// for the data the prolongation reproduces, and unchanged when the boxes do
// not change; a failed regrid changes nothing.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
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
using soma::amr::TagField;

KOKKOS_INLINE_FUNCTION double pos(int p, bool nodal, double h) { return (p + (nodal ? 0.0 : 0.5)) * h; }
KOKKOS_INLINE_FUNCTION double linear(double x, double y, double z, int c) { return 0.3 + 1.7 * x - 0.9 * y + 0.4 * z + 10.0 * c; }
KOKKOS_INLINE_FUNCTION double at(const IntVect& p, IndexType t, double h, int c, int dim) {
  return linear(pos(p[0], t.is_nodal(0), h), pos(p[1], t.is_nodal(1), h), dim == 3 ? pos(p[2], t.is_nodal(2), h) : 0.0, c);
}
// Quadratic data, of degree 2 in each direction, averaged over what point p
// of index type t stands for (the point along node directions, the cell
// along cell directions).
KOKKOS_INLINE_FUNCTION double quad_at(const IntVect& p, IndexType t, double h, int c, int dim) {
  double m1[3] = {0, 0, 0}, m2[3] = {0, 0, 0};  // averages of x and x^2
  for (int d = 0; d < dim; ++d) {
    if (t.is_nodal(d)) {
      m1[d] = p[d] * h;
      m2[d] = m1[d] * m1[d];
    } else {
      const double a = p[d] * h, b = (p[d] + 1) * h;
      m1[d] = 0.5 * (a + b);
      m2[d] = (a * a + a * b + b * b) / 3.0;
    }
  }
  return 0.3 + 1.7 * m1[0] - 0.9 * m1[1] + 0.4 * m1[2] + 0.8 * m2[0] - 0.5 * m2[1] + 0.3 * m2[2] + 0.6 * m1[0] * m1[1] -
         0.7 * m1[1] * m1[2] + 0.2 * m2[0] * m1[1] - 0.15 * m2[0] * m2[1] * (dim == 3 ? m2[2] : 1.0) + 10.0 * c;
}
KOKKOS_INLINE_FUNCTION double noise(const IntVect& p, int c) {
  const unsigned u = static_cast<unsigned>(p[0] * 73856093) ^ static_cast<unsigned>(p[1] * 19349663) ^
                     static_cast<unsigned>(p[2] * 83492791) ^ (static_cast<unsigned>(c) * 2654435761u);
  return static_cast<double>(u % 1000u) / 100.0;
}

enum class Data { noise, linear, quadratic };

KOKKOS_INLINE_FUNCTION double data_at(Data kind, const IntVect& p, IndexType t, double h, int c, int dim) {
  return kind == Data::linear ? at(p, t, h, c, dim) : kind == Data::quadratic ? quad_at(p, t, h, c, dim) : noise(p, c);
}

// Cells whose centre lies within w of the circle (sphere) of radius 0.25
// around (cx, 0.5, 0.5) on the unit square (cube), periodic in x if asked.
KOKKOS_INLINE_FUNCTION bool on_ring(const IntVect& p, double h, int dim, double cx, bool periodic_x) {
  double dx = (p[0] + 0.5) * h - cx;
  if (periodic_x) dx -= Kokkos::floor(dx + 0.5);
  const double dy = (p[1] + 0.5) * h - 0.5, dz = dim == 3 ? (p[2] + 0.5) * h - 0.5 : 0.0;
  const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
  return Kokkos::fabs(r - 0.25) < 0.06;
}

struct Config {
  Geometry geom;
  HierarchyOptions opts;
};

Config config(int dim) {
  Config s;
  s.opts.max_level = 2;
  if (dim == 2) {
    s.geom.domain = Box(IntVect{}, IntVect{{63, 63, 0}}, 2);
    s.opts.blocking_factor = 8;
    s.opts.max_box = 16;
  } else {
    s.geom.domain = Box(IntVect{}, IntVect{{31, 31, 31}});
    s.opts.blocking_factor = 4;
    s.opts.max_box = 8;
  }
  return s;
}

// Kernels live in named functions: CUDA refuses device lambdas in a test
// body (a private member function) or inside another lambda.
void tag_ring(const TagField& tags, double dx, int dim, double cx, bool px) {
  const auto v = tags.view();
  soma::amr::for_each_point(tags, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    if (on_ring(IntVect{{i, j, k}}, dx, dim, cx, px)) v.patch(b)(i, j, k) = 1;
  });
}

Hierarchy::Tagger ring(const Hierarchy& h, double cx) {
  return [&h, cx](int level, const TagField& tags) {
    tag_ring(tags, h.dx(level, 0), h.geometry().domain.dim(), cx, h.geometry().periodic[0]);
  };
}

void set_points(const soma::amr::LevelField<double>& f, double h, int dim, Data kind) {
  const auto v = f.view();
  const IndexType t = f.type();
  const int ncomp = f.ncomp();
  soma::amr::for_each_point(f, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    for (int c = 0; c < ncomp; ++c) v.patch(b)(i, j, k, c) = data_at(kind, IntVect{{i, j, k}}, t, h, c, dim);
  });
}

// Exact values in the ghost regions outside the domain.
struct ExactBoundary {
  double dx0;
  int dim;
  Data kind;
  void operator()(const soma::amr::LevelField<double>& f, int level, int box, const Box& region, int, int) const {
    const auto v = f.view();
    const IndexType t = f.type();
    const int ncomp = f.ncomp(), d = dim;
    const double h = dx0 / (1 << level);
    const Data k = kind;
    Kokkos::parallel_for(
        "exact_boundary", Kokkos::RangePolicy<Kokkos::IndexType<std::int64_t>>(0, region.num_pts()), KOKKOS_LAMBDA(std::int64_t q) {
          const IntVect p = soma::amr::detail::point_of(region, q);
          for (int c = 0; c < ncomp; ++c) v.patch(box)(p[0], p[1], p[2], c) = data_at(k, p, t, h, c, d);
        });
    Kokkos::fence();
  }
};

bool inside(const BoxList& boxes, const IntVect& p) {
  for (const Box& b : boxes) {
    if (b.contains(p)) return true;
  }
  return false;
}

std::vector<double> host_valid(const soma::amr::LevelField<double>& f) {
  auto d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f.view().data);
  std::vector<double> out;
  for (std::size_t b = 0; b < f.num_boxes(); ++b) {
    const Box g = f.grown(b), v = f.valid(b);
    for (std::int64_t q = 0; q < g.num_pts(); ++q) {
      if (!v.contains(soma::amr::detail::point_of(g, q))) continue;
      for (int c = 0; c < f.ncomp(); ++c) out.push_back(d(static_cast<std::size_t>(f.offset(b) + q + g.num_pts() * c)));
    }
  }
  return out;
}

}  // namespace

TEST(AmrRegrid, GridsCoverTagsAndNest) {
  for (const int dim : {2, 3}) {
    for (const bool periodic : {false, true}) {
      Config s = config(dim);
      s.geom.periodic.periodic[0] = periodic;
      // The ring across the periodic side when there is one.
      const double cx = periodic ? 0.1 : 0.5;
      Hierarchy h(s.geom, s.opts);
      for (int cycle = 0; cycle < 3; ++cycle) {
        SCOPED_TRACE(testing::Message() << "dim " << dim << " periodic " << periodic << " cycle " << cycle);
        const std::vector<BoxList> grids = h.make_grids(ring(h, cx));
        ASSERT_EQ(static_cast<int>(grids.size()), std::min(cycle + 1, s.opts.max_level));
        for (std::size_t i = 0; i < grids.size(); ++i) {
          const int l = static_cast<int>(i) + 1;
          const Box domain = h.domain(l);
          EXPECT_TRUE(soma::amr::disjoint(grids[i]));
          for (const Box& b : grids[i]) {
            EXPECT_TRUE(domain.contains(b));
            for (int d = 0; d < dim; ++d) {
              EXPECT_EQ(soma::amr::floor_div(b.lo(d), s.opts.blocking_factor) * s.opts.blocking_factor, b.lo(d));
              EXPECT_EQ(b.length(d) % s.opts.blocking_factor, 0);
              EXPECT_LE(b.length(d), s.opts.max_box);
            }
          }
          // Every tagged cell of the level below (on its current boxes) is
          // refined.
          if (l - 1 <= h.finest_level()) {
            const BoxList under = soma::amr::coarsen(grids[i], s.opts.ratio);
            const double hx = h.dx(l - 1, 0);
            int tagged = 0;
            for (const Box& b : h.boxes(l - 1).boxes()) {
              for (std::int64_t q = 0; q < b.num_pts(); ++q) {
                const IntVect p = soma::amr::detail::point_of(b, q);
                if (!on_ring(p, hx, dim, cx, periodic)) continue;
                ++tagged;
                ASSERT_TRUE(inside(under, p)) << "level " << l - 1 << " cell " << p[0] << "," << p[1] << "," << p[2];
              }
            }
            EXPECT_GT(tagged, 0);
          }
          // Proper nesting, point by point: the coarse cells within one of
          // a fine box (wrapped across periodic sides) are on the level
          // below.
          if (i == 0) continue;
          const Box below = h.domain(l - 1);
          for (const Box& b : grids[i]) {
            const Box need = b.coarsen(s.opts.ratio).grow(s.opts.nesting);
            for (std::int64_t q = 0; q < need.num_pts(); ++q) {
              IntVect p = soma::amr::detail::point_of(need, q);
              bool outside = false;
              for (int d = 0; d < dim; ++d) {
                if (periodic && d == 0) p[d] = below.lo(d) + ((p[d] - below.lo(d)) % below.length(d) + below.length(d)) % below.length(d);
                outside = outside || p[d] < below.lo(d) || p[d] > below.hi(d);
              }
              if (outside) continue;
              ASSERT_TRUE(inside(grids[i - 1], p)) << "level " << l << " box not nested at " << p[0] << "," << p[1] << "," << p[2];
            }
          }
        }
        if (periodic) {
          // Boxes on both x sides of the finest level: the ring crosses.
          bool low = false, high = false;
          for (const Box& b : grids.back()) {
            low = low || b.lo(0) == h.domain(static_cast<int>(grids.size())).lo(0);
            high = high || b.hi(0) == h.domain(static_cast<int>(grids.size())).hi(0);
          }
          EXPECT_TRUE(low && high);
        }
        h.regrid(ring(h, cx), Interp::linear, soma::amr::NoBoundary{});
        EXPECT_EQ(h.num_levels(), static_cast<int>(grids.size()) + 1);
        for (std::size_t i = 0; i < grids.size(); ++i) EXPECT_EQ(h.boxes(static_cast<int>(i) + 1).num_pts(), soma::amr::num_pts(grids[i]));
      }
      // No tags: back to level 0.
      h.regrid([](int, const TagField&) {}, Interp::linear, soma::amr::NoBoundary{});
      EXPECT_EQ(h.num_levels(), 1);
    }
  }
}

// Linear data under linear prolongation, quadratic data under quadratic.
TEST(AmrRegrid, DataTheRuleReproducesMoveExactly) {
  for (const Interp interp : {Interp::linear, Interp::quadratic}) {
  const Data kind = interp == Interp::linear ? Data::linear : Data::quadratic;
  for (const int dim : {2, 3}) {
    const Config s = config(dim);
    const ExactBoundary exact{1.0 / s.geom.domain.length(0), dim, kind};
    for (const IndexType t : {IndexType::cell(), IndexType::node(dim), IndexType::face(0)}) {
      SCOPED_TRACE(testing::Message() << "quadratic " << (interp == Interp::quadratic) << " dim " << dim << " type " << int(t.nodal));
      Hierarchy h(s.geom, s.opts);
      auto u = h.make_field<double>(t, 2, 1);
      set_points(u[0], h.dx(0, 0), dim, kind);
      // The ring moves each cycle: new boxes take prolonged data, the
      // others keep theirs.
      for (int cycle = 0; cycle < 4; ++cycle) {
        // Against the low x side first, the slopes there reading ghosts.
        h.regrid(ring(h, 0.2 + 0.1 * cycle), interp, exact, u);
        ASSERT_EQ(u.size(), static_cast<std::size_t>(h.num_levels()));
        for (int l = 0; l < h.num_levels(); ++l) {
          const auto& f = u[static_cast<std::size_t>(l)];
          ASSERT_EQ(f.boxes().boxes(), h.boxes(l).boxes());
          auto d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f.view().data);
          for (std::size_t b = 0; b < f.num_boxes(); ++b) {
            const Box g = f.grown(b), v = f.valid(b);
            for (std::int64_t q = 0; q < g.num_pts(); ++q) {
              const IntVect p = soma::amr::detail::point_of(g, q);
              if (!v.contains(p)) continue;
              for (int c = 0; c < 2; ++c) {
                ASSERT_NEAR(d(static_cast<std::size_t>(f.offset(b) + q + g.num_pts() * c)), data_at(kind, p, t, h.dx(l, 0), c, dim), 1e-11)
                    << "cycle " << cycle << " level " << l << " point " << p[0] << "," << p[1] << "," << p[2];
              }
            }
          }
        }
      }
      EXPECT_EQ(h.num_levels(), 3);
    }
  }
  }
}

TEST(AmrRegrid, SameBoxesKeepTheData) {
  for (const int dim : {2, 3}) {
    const Config s = config(dim);
    Hierarchy h(s.geom, s.opts);
    auto u = h.make_field<double>(IndexType::cell(), 1, 1);
    auto w = h.make_field<float>(IndexType::node(dim), 1, 1);
    // Regrid until the boxes stop changing.
    int cycles = 0;
    for (;; ++cycles) {
      ASSERT_LT(cycles, 8);
      const Hierarchy next(s.geom, s.opts, h.make_grids(ring(h, 0.5)));
      bool same = next.num_levels() == h.num_levels();
      for (int l = 0; same && l < h.num_levels(); ++l) same = next.boxes(l).boxes() == h.boxes(l).boxes();
      if (same) break;
      h.regrid(ring(h, 0.5), Interp::piecewise_constant, soma::amr::NoBoundary{}, u, w);
    }
    EXPECT_GE(cycles, 2);
    ASSERT_EQ(h.num_levels(), 3);
    for (const auto& f : u) set_points(f, 0.0, dim, Data::noise);
    h.average_down(u);
    std::vector<std::vector<double>> before;
    for (const auto& f : u) before.push_back(host_valid(f));
    std::vector<BoxList> boxes;
    for (int l = 0; l < h.num_levels(); ++l) boxes.push_back(h.boxes(l).boxes());
    // v is not restricted: after the regrid its finest level is unchanged
    // and the levels below are its averages.
    auto v = h.make_field<double>(IndexType::cell(), 1, 1);
    for (const auto& f : v) set_points(f, 0.0, dim, Data::noise);
    const std::vector<double> v_finest = host_valid(v.back());
    h.regrid(ring(h, 0.5), Interp::linear, soma::amr::NoBoundary{}, u, w, v);
    EXPECT_EQ(host_valid(v.back()), v_finest);
    std::vector<std::vector<double>> v_after;
    for (const auto& f : v) v_after.push_back(host_valid(f));
    h.average_down(v);
    for (std::size_t l = 0; l < v.size(); ++l) EXPECT_EQ(host_valid(v[l]), v_after[l]) << "level " << l;
    ASSERT_EQ(h.num_levels(), 3);
    for (int l = 0; l < h.num_levels(); ++l) {
      EXPECT_EQ(h.boxes(l).boxes(), boxes[static_cast<std::size_t>(l)]);
      EXPECT_EQ(host_valid(u[static_cast<std::size_t>(l)]), before[static_cast<std::size_t>(l)]) << "level " << l;
      EXPECT_EQ(w[static_cast<std::size_t>(l)].boxes().boxes(), boxes[static_cast<std::size_t>(l)]);
    }
  }
}

TEST(AmrRegrid, FailedRegridChangesNothing) {
  const Config s = config(2);
  Hierarchy h(s.geom, s.opts);
  h.regrid(ring(h, 0.5), Interp::linear, soma::amr::NoBoundary{});
  auto u = h.make_field<double>(IndexType::cell(), 1, 1);
  auto bare = h.make_field<double>(IndexType::cell(), 1, 0);
  for (const auto& f : u) set_points(f, 0.0, 2, Data::noise);
  const auto before = host_valid(u[1]);
  const BoxList level1 = h.boxes(1).boxes();
  // Linear prolongation needs ghosts: the second field throws after the
  // first moved.
  try {
    h.regrid(ring(h, 0.5), Interp::linear, soma::amr::NoBoundary{}, u, bare);
    ADD_FAILURE() << "no exception";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("need a field with ghosts"), std::string::npos) << e.what();
  }
  EXPECT_EQ(h.num_levels(), 2);
  EXPECT_EQ(h.boxes(1).boxes(), level1);
  EXPECT_EQ(u[1].boxes().boxes(), level1);
  EXPECT_EQ(host_valid(u[1]), before);
  EXPECT_THROW(h.transfer(h, bare, Interp::quadratic), std::invalid_argument);
  EXPECT_NO_THROW(h.transfer(h, bare, Interp::piecewise_constant));
  // A field of another hierarchy, and hierarchies over other domains.
  EXPECT_THROW(h.transfer(h, Hierarchy(s.geom, s.opts).make_field<double>(IndexType::cell(), 1, 1)), std::invalid_argument);
  Hierarchy moved(s.geom, s.opts);
  moved.regrid(ring(moved, 0.6), Interp::linear, soma::amr::NoBoundary{});
  ASSERT_EQ(moved.num_levels(), 2);
  EXPECT_THROW(h.transfer(h, moved.make_field<double>(IndexType::cell(), 1, 1)), std::invalid_argument);
  Geometry other = s.geom;
  other.domain = Box(IntVect{}, IntVect{{31, 63, 0}}, 2);
  const Hierarchy o(other, s.opts);
  EXPECT_THROW(o.transfer(Hierarchy(s.geom, s.opts), Hierarchy(s.geom, s.opts).make_field<double>(IndexType::cell(), 1, 1)),
               std::invalid_argument);
  HierarchyOptions r4 = s.opts;
  r4.ratio = 4;
  const Hierarchy h4(s.geom, r4);
  EXPECT_THROW(h4.transfer(Hierarchy(s.geom, s.opts), Hierarchy(s.geom, s.opts).make_field<double>(IndexType::cell(), 1, 1)),
               std::invalid_argument);
}
