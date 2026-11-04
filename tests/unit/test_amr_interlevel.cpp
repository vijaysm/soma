// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Prolongation and restriction: exact for the data each rule reproduces
// (linear data for the linear rules, the parent's value for piecewise
// constant cells), conservative for cell data, and refused for fine levels
// the coarse one does not nest.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/amr/box_list.hpp"
#include "soma/amr/interlevel.hpp"

namespace {

using soma::amr::Box;
using soma::amr::BoxArray;
using soma::amr::BoxList;
using soma::amr::IndexType;
using soma::amr::Interp;
using soma::amr::IntVect;

constexpr double sentinel = -1.0e30;

// Position of point p of an index type on a grid of spacing h.
KOKKOS_INLINE_FUNCTION double pos(int p, bool nodal, double h) { return (p + (nodal ? 0.0 : 0.5)) * h; }
KOKKOS_INLINE_FUNCTION double linear(double x, double y, double z, int c) { return 0.3 + 1.7 * x - 0.9 * y + 0.4 * z + 10.0 * c; }
// In 2D the k index is 0 on every level and has no position.
KOKKOS_INLINE_FUNCTION double at(const IntVect& p, IndexType t, double h, int c, int dim) {
  return linear(pos(p[0], t.is_nodal(0), h), pos(p[1], t.is_nodal(1), h), dim == 3 ? pos(p[2], t.is_nodal(2), h) : 0.0, c);
}
// Quadratic data, of degree 2 in each direction, averaged over what point p
// of index type t stands for: the point along node directions, the cell
// [p h, (p + 1) h] along cell directions.
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
         0.7 * m1[1] * m1[2] + 0.25 * m1[0] * m1[2] + 0.2 * m2[0] * m1[1] - 0.15 * m2[0] * m2[1] * (dim == 3 ? m2[2] : 1.0) +
         10.0 * c;
}

// Deterministic data with no structure, for conservation.
KOKKOS_INLINE_FUNCTION double noise(const IntVect& p, int c) {
  const unsigned u = static_cast<unsigned>(p[0] * 73856093) ^ static_cast<unsigned>(p[1] * 19349663) ^
                     static_cast<unsigned>(p[2] * 83492791) ^ (static_cast<unsigned>(c) * 2654435761u);
  return static_cast<double>(u % 1000u) / 100.0;
}

// Kernels live in named functions: CUDA refuses device lambdas in a test
// body (a private member function) or inside another lambda.
void set_linear(const soma::amr::LevelField<double>& f, soma::amr::Region region, double h) {
  const auto v = f.view();
  const IndexType t = f.type();
  const int ncomp = f.ncomp(), dim = f.boxes().domain().dim();
  soma::amr::for_each_point(f, region, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    for (int c = 0; c < ncomp; ++c) v.patch(b)(i, j, k, c) = at(IntVect{{i, j, k}}, t, h, c, dim);
  });
}

void set_quadratic(const soma::amr::LevelField<double>& f, soma::amr::Region region, double h) {
  const auto v = f.view();
  const IndexType t = f.type();
  const int ncomp = f.ncomp(), dim = f.boxes().domain().dim();
  soma::amr::for_each_point(f, region, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    for (int c = 0; c < ncomp; ++c) v.patch(b)(i, j, k, c) = quad_at(IntVect{{i, j, k}}, t, h, c, dim);
  });
}

void set_noise(const soma::amr::LevelField<double>& f, soma::amr::Region region) {
  const auto v = f.view();
  soma::amr::for_each_point(f, region, KOKKOS_LAMBDA(int b, int i, int j, int k) { v.patch(b)(i, j, k) = noise(IntVect{{i, j, k}}, 0); });
}

// Along x: 0 (i <= 1), 0.1 (i = 2), 1 (i = 3..5), 0.5 (i = 6), 1 (i >= 7).
void set_steps(const soma::amr::LevelField<double>& f) {
  const auto v = f.view();
  soma::amr::for_each_point(f, soma::amr::Region::grown, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    v.patch(b)(i, j, k) = i <= 1 ? 0.0 : i == 2 ? 0.1 : i == 6 ? 0.5 : 1.0;
  });
}

struct Levels {
  BoxArray coarse, fine;
};

// A coarse level over the whole domain; a fine level over `region` (coarse
// cells) refined by r, both chopped.
Levels levels(int dim, int r, const Box& region) {
  const Box domain = dim == 2 ? Box(IntVect{}, IntVect{{15, 15, 0}}, 2) : Box(IntVect{}, IntVect{{7, 7, 7}});
  return {BoxArray(soma::amr::chop({domain}, 6), domain), BoxArray(soma::amr::chop({region.refine(r)}, 8, r), domain.refine(r))};
}

std::vector<IndexType> types(int dim) {
  std::vector<IndexType> t{IndexType::cell(), IndexType::node(dim), IndexType::face(0), IndexType::face(1)};
  if (dim == 3) t.push_back(IndexType::face(2));
  return t;
}

// Every point of every fine box: valid points and ghosts another fine box
// covers against valid(p, c), interpolated ghosts against want(p, c),
// ghosts a physical boundary owns at sentinel.
template <class V, class F>
void expect_points(const soma::amr::LevelField<double>& fine, const soma::amr::HaloPlan& halo, const V& valid, const F& want) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fine.view().data);
  for (std::size_t b = 0; b < fine.num_boxes(); ++b) {
    const Box g = fine.grown(b);
    for (std::int64_t q = 0; q < g.num_pts(); ++q) {
      const IntVect p = soma::amr::detail::point_of(g, q);
      bool outside = false;
      for (const auto& br : halo.boundary()) outside = outside || (br.box == static_cast<int>(b) && br.region.contains(p));
      for (int c = 0; c < fine.ncomp(); ++c) {
        const double got = h(static_cast<std::size_t>(fine.offset(b) + q + g.num_pts() * c));
        bool interpolated = false;
        for (const Box& u : halo.uncovered()[b]) interpolated = interpolated || u.contains(p);
        if (fine.valid(b).contains(p) || (!outside && !interpolated)) {
          ASSERT_NEAR(got, valid(p, c), 1e-12) << "valid point " << p[0] << "," << p[1] << "," << p[2];
        } else if (outside) {
          ASSERT_EQ(got, sentinel);
        } else {
          ASSERT_NEAR(got, want(p, c), 1e-12) << "box " << b << " point " << p[0] << "," << p[1] << "," << p[2] << " c " << c;
        }
      }
    }
  }
}

}  // namespace

TEST(AmrInterlevel, LinearRulesAreExactForLinearData) {
  for (const int dim : {2, 3}) {
    // Inside the domain, and against its low corner (the slopes there use the
    // coarse ghosts).
    for (const Box region : {Box(IntVect{{2, 3, 2}}, IntVect{{5, 5, 4}}, dim), Box(IntVect{{0, 0, 0}}, IntVect{{3, 2, 2}}, dim)}) {
      for (const int r : {2, 4}) {
        const Levels lv = levels(dim, r, region);
        for (const IndexType t : types(dim)) {
          for (const int g : {1, 2}) {
            for (const Interp interp : {Interp::linear, Interp::piecewise_constant}) {
              SCOPED_TRACE(testing::Message() << "dim " << dim << " r " << r << " type " << int(t.nodal) << " ghosts " << g
                                              << " linear " << (interp == Interp::linear) << " at " << region.lo(0));
              const int ncomp = 2;
              soma::amr::LevelField<double> coarse(lv.coarse, t, ncomp, 1), fine(lv.fine, t, ncomp, g);
              set_linear(coarse, soma::amr::Region::grown, 1.0);
              fine.fill(sentinel);
              const double hf = 1.0 / r;
              set_linear(fine, soma::amr::Region::valid, hf);
              const soma::amr::HaloPlan halo(lv.fine, t, g, {});
              soma::amr::fill_halo(fine, halo);
              const soma::amr::CoarseFinePlan plan(lv.fine, lv.coarse, t, g, r, {}, interp, 1);
              EXPECT_GT(plan.num_points(), 0);
              soma::amr::fill_coarse_fine(fine, coarse, plan);
              const auto fine_data = [&](const IntVect& p, int c) { return at(p, t, hf, c, dim); };
              expect_points(fine, halo, fine_data, [&](const IntVect& p, int c) {
                if (interp == Interp::linear) return at(p, t, hf, c, dim);
                // Piecewise constant: the parent's position along cell
                // directions, the fine one along node directions.
                double x[3] = {0, 0, 0};
                for (int d = 0; d < dim; ++d) {
                  x[d] = t.is_nodal(d) ? pos(p[d], true, hf) : pos(soma::amr::floor_div(p[d], r), false, 1.0);
                }
                return linear(x[0], x[1], x[2], c);
              });
            }
          }
        }
      }
    }
  }
}

// The quadratic rules reproduce quadratic data (averages over cells,
// values at nodes); the linear ones do not.
TEST(AmrInterlevel, QuadraticRulesAreExactForQuadraticData) {
  for (const int dim : {2, 3}) {
    for (const Box region : {Box(IntVect{{2, 3, 2}}, IntVect{{5, 5, 4}}, dim), Box(IntVect{{0, 0, 0}}, IntVect{{3, 2, 2}}, dim)}) {
      for (const int r : {2, 4}) {
        const Levels lv = levels(dim, r, region);
        for (const IndexType t : types(dim)) {
          for (const int g : {1, 2}) {
            SCOPED_TRACE(testing::Message() << "dim " << dim << " r " << r << " type " << int(t.nodal) << " ghosts " << g
                                            << " at " << region.lo(0));
            soma::amr::LevelField<double> coarse(lv.coarse, t, 2, 1), fine(lv.fine, t, 2, g);
            // Spacing 1/8 on the coarse level keeps the values near 1.
            set_quadratic(coarse, soma::amr::Region::grown, 0.125);
            fine.fill(sentinel);
            const double hf = 0.125 / r;
            set_quadratic(fine, soma::amr::Region::valid, hf);
            const soma::amr::HaloPlan halo(lv.fine, t, g, {});
            soma::amr::fill_halo(fine, halo);
            const soma::amr::CoarseFinePlan plan(lv.fine, lv.coarse, t, g, r, {}, Interp::quadratic, 1);
            soma::amr::fill_coarse_fine(fine, coarse, plan);
            const auto exact = [&](const IntVect& p, int c) { return quad_at(p, t, hf, c, dim); };
            expect_points(fine, halo, exact, exact);
          }
        }
      }
    }
  }
  // The linear rule misses the curvature.
  const Levels lv = levels(2, 2, Box(IntVect{{2, 3, 0}}, IntVect{{5, 5, 0}}, 2));
  soma::amr::LevelField<double> coarse(lv.coarse, IndexType::cell(), 1, 1), fine(lv.fine, IndexType::cell(), 1, 1);
  set_quadratic(coarse, soma::amr::Region::grown, 0.125);
  fine.fill(sentinel);
  soma::amr::fill_coarse_fine(fine, coarse, soma::amr::CoarseFinePlan(lv.fine, lv.coarse, IndexType::cell(), 1, 2, {}, Interp::linear, 1));
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fine.view().data);
  const Box g = fine.grown(0);
  double worst = 0;
  for (std::int64_t q = 0; q < g.num_pts(); ++q) {
    const IntVect p = soma::amr::detail::point_of(g, q);
    if (fine.valid(0).contains(p) || h(static_cast<std::size_t>(q)) == sentinel) continue;
    worst = std::max(worst, std::fabs(h(static_cast<std::size_t>(q)) - quad_at(p, IndexType::cell(), 0.0625, 0, 2)));
  }
  EXPECT_GT(worst, 1e-4);
}

// Prolongation into given points, valid ones included, exact for linear
// data; points outside the targets are left alone, and targets outside the
// fine boxes are refused.
TEST(AmrInterlevel, ProlongsIntoTargets) {
  for (const int dim : {2, 3}) {
    const Levels lv = levels(dim, 2, Box(IntVect{{1, 1, 1}}, IntVect{{3, 4, 3}}, dim));
    for (const IndexType t : types(dim)) {
      SCOPED_TRACE(testing::Message() << "dim " << dim << " type " << int(t.nodal));
      soma::amr::LevelField<double> coarse(lv.coarse, t, 1, 1), fine(lv.fine, t, 1, 1);
      set_linear(coarse, soma::amr::Region::grown, 1.0);
      fine.fill(sentinel);
      // The lower half of each box's valid points, and its low ghost layer
      // in x.
      std::vector<BoxList> targets(lv.fine.size());
      for (std::size_t b = 0; b < lv.fine.size(); ++b) {
        const Box v = fine.valid(b);
        targets[b].push_back(Box(v.lo(), IntVect{{v.hi(0), v.lo(1) + (v.length(1) - 1) / 2, v.hi(2)}}, dim, t));
        targets[b].push_back(Box(IntVect{{v.lo(0) - 1, v.lo(1), v.lo(2)}}, IntVect{{v.lo(0) - 1, v.hi(1), v.hi(2)}}, dim, t));
        // Empty boxes ask for nothing, wherever their corners lie.
        targets[b].push_back(Box(IntVect{{-50, -50, -50}}, IntVect{{-60, -60, -60}}, dim, t));
      }
      const soma::amr::CoarseFinePlan plan(lv.fine, targets, lv.coarse, t, 1, 2, {}, Interp::linear, 1);
      soma::amr::fill_coarse_fine(fine, coarse, plan);
      auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fine.view().data);
      for (std::size_t b = 0; b < fine.num_boxes(); ++b) {
        const Box g = fine.grown(b);
        for (std::int64_t q = 0; q < g.num_pts(); ++q) {
          const IntVect p = soma::amr::detail::point_of(g, q);
          bool in = false;
          for (const Box& x : targets[b]) in = in || x.contains(p);
          const double got = h(static_cast<std::size_t>(fine.offset(b) + q));
          if (in) {
            ASSERT_NEAR(got, at(p, t, 0.5, 0, dim), 1e-12) << p[0] << "," << p[1] << "," << p[2];
          } else {
            ASSERT_EQ(got, sentinel) << p[0] << "," << p[1] << "," << p[2];
          }
        }
      }
      std::vector<BoxList> outside(lv.fine.size());
      outside[0].push_back(fine.valid(0).grow(2));
      EXPECT_THROW(soma::amr::CoarseFinePlan(lv.fine, outside, lv.coarse, t, 1, 2, {}, Interp::linear, 1), std::invalid_argument);
      EXPECT_THROW(soma::amr::CoarseFinePlan(lv.fine, std::vector<BoxList>(lv.fine.size() + 1), lv.coarse, t, 1, 2, {}, Interp::linear, 1),
                   std::invalid_argument);
    }
  }
}

// The fine cells of a coarse cell average to it, limiter or not.
TEST(AmrInterlevel, CellRulesConserve) {
  for (const Interp interp : {Interp::linear, Interp::quadratic}) {
  for (const int dim : {2, 3}) {
    for (const int r : {2, 4}) {
      SCOPED_TRACE(testing::Message() << "quadratic " << (interp == Interp::quadratic) << " dim " << dim << " r " << r);
      const Levels lv = levels(dim, r, Box(IntVect{{3, 3, 3}}, IntVect{{4, 5, 4}}, dim));
      soma::amr::LevelField<double> coarse(lv.coarse, IndexType::cell(), 1, 1), fine(lv.fine, IndexType::cell(), 1, r);
      set_noise(coarse, soma::amr::Region::grown);
      fine.fill(0.0);
      const soma::amr::CoarseFinePlan plan(lv.fine, lv.coarse, IndexType::cell(), r, r, {}, interp, 1);
      soma::amr::fill_coarse_fine(fine, coarse, plan);
      auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fine.view().data);
      const soma::amr::HaloPlan halo(lv.fine, IndexType::cell(), r, {});
      int checked = 0;
      for (std::size_t b = 0; b < fine.num_boxes(); ++b) {
        const Box g = fine.grown(b);
        for (const Box& u : halo.uncovered()[b]) {
          // Coarse cells whose children all lie in this uncovered region.
          const Box whole = Box(u.lo(), u.hi(), dim).coarsen(r);
          for (std::int64_t q = 0; q < whole.num_pts(); ++q) {
            const IntVect cc = soma::amr::detail::point_of(whole, q);
            const Box kids = Box(cc, cc, dim).refine(r);
            if (!u.contains(kids)) continue;
            double sum = 0;
            for (std::int64_t k = 0; k < kids.num_pts(); ++k) {
              const IntVect p = soma::amr::detail::point_of(kids, k);
              const IntVect o = p - g.lo();
              sum += h(static_cast<std::size_t>(fine.offset(b) + o[0] + g.length(0) * (o[1] + g.length(1) * o[2])));
            }
            EXPECT_NEAR(sum / static_cast<double>(kids.num_pts()), noise(cc, 0), 1e-12);
            ++checked;
          }
        }
      }
      EXPECT_GT(checked, 0);
    }
  }
  }
}

TEST(AmrInterlevel, AverageDownIsExactForLinearData) {
  for (const int dim : {2, 3}) {
    for (const int r : {2, 4}) {
      const Levels lv = levels(dim, r, Box(IntVect{{2, 3, 2}}, IntVect{{5, 5, 4}}, dim));
      for (const IndexType t : types(dim)) {
        SCOPED_TRACE(testing::Message() << "dim " << dim << " r " << r << " type " << int(t.nodal));
        soma::amr::LevelField<double> coarse(lv.coarse, t, 2, 1), fine(lv.fine, t, 2, 1);
        const double hf = 1.0 / r;
        set_linear(fine, soma::amr::Region::grown, hf);
        coarse.fill(sentinel);
        soma::amr::average_down(fine, coarse, r);
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, coarse.view().data);
        BoxList under;  // coarse points the fine level covers
        for (std::size_t b = 0; b < lv.fine.size(); ++b) under.push_back(lv.fine[b].coarsen(r).convert(t));
        std::int64_t set = 0;
        for (std::size_t b = 0; b < coarse.num_boxes(); ++b) {
          const Box g = coarse.grown(b);
          for (std::int64_t q = 0; q < g.num_pts(); ++q) {
            const IntVect p = soma::amr::detail::point_of(g, q);
            bool covered = coarse.valid(b).contains(p);
            bool hit = false;
            for (const Box& u : under) hit = hit || u.contains(p);
            covered = covered && hit;
            for (int c = 0; c < 2; ++c) {
              const double got = h(static_cast<std::size_t>(coarse.offset(b) + q + g.num_pts() * c));
              if (covered) {
                ASSERT_NEAR(got, at(p, t, 1.0, c, dim), 1e-12) << p[0] << "," << p[1] << "," << p[2];
                ++set;
              } else {
                ASSERT_EQ(got, sentinel) << p[0] << "," << p[1] << "," << p[2];
              }
            }
          }
        }
        EXPECT_GT(set, 0);
      }
    }
  }
}

TEST(AmrInterlevel, RefusesWhatDoesNotNest) {
  const Box domain(IntVect{}, IntVect{{15, 15, 0}}, 2);
  // A coarse level with a hole at [6,9]^2; a fine level next to it.
  const BoxArray coarse(soma::amr::chop(soma::amr::complement(domain, {Box(IntVect{{6, 6, 0}}, IntVect{{9, 9, 0}}, 2)}), 6), domain);
  const BoxArray fine({Box(IntVect{{4, 4, 0}}, IntVect{{5, 5, 0}}, 2).refine(2)}, domain.refine(2));
  try {
    soma::amr::CoarseFinePlan(fine, coarse, IndexType::cell(), 2, 2, {}, Interp::linear, 1);
    ADD_FAILURE() << "accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("[8,8]-[11,11]"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("properly nested"), std::string::npos) << e.what();
  }
  // One coarse cell further it nests (the slopes reach one coarse cell).
  const BoxArray nested({Box(IntVect{{2, 2, 0}}, IntVect{{4, 4, 0}}, 2).refine(2)}, domain.refine(2));
  EXPECT_NO_THROW(soma::amr::CoarseFinePlan(nested, coarse, IndexType::cell(), 2, 2, {}, Interp::linear, 1));
  EXPECT_THROW(soma::amr::CoarseFinePlan(fine, coarse, IndexType::cell(), 2, 4, {}, Interp::linear, 1), std::invalid_argument);
  EXPECT_THROW(soma::amr::CoarseFinePlan(fine, coarse, IndexType::cell(), 2, 1, {}, Interp::linear, 1), std::invalid_argument);
  // Restriction needs fine boxes aligned to the ratio.
  const BoxArray odd({Box(IntVect{{3, 4, 0}}, IntVect{{8, 9, 0}}, 2)}, domain.refine(2));
  soma::amr::LevelField<double> f(odd, IndexType::cell(), 1, 0), c(coarse, IndexType::cell(), 1, 0);
  EXPECT_THROW(soma::amr::average_down(f, c, 2), std::invalid_argument);
  soma::amr::LevelField<double> n(coarse, IndexType::node(2), 1, 0);
  EXPECT_THROW(soma::amr::average_down(f, n, 2), std::invalid_argument);
  // A plan applied to fields it was not made for.
  const soma::amr::CoarseFinePlan plan(nested, coarse, IndexType::cell(), 2, 2, {}, Interp::linear, 1);
  soma::amr::LevelField<double> thin(nested, IndexType::cell(), 1, 1), cf(coarse, IndexType::cell(), 1, 1);
  EXPECT_THROW(soma::amr::fill_coarse_fine(thin, cf, plan), std::invalid_argument);
  soma::amr::LevelField<double> wide(nested, IndexType::cell(), 1, 2), bare(coarse, IndexType::cell(), 1, 0);
  EXPECT_THROW(soma::amr::fill_coarse_fine(wide, bare, plan), std::invalid_argument);
  EXPECT_NO_THROW(soma::amr::fill_coarse_fine(wide, cf, plan));
}

// The monotonized-central limiter: coarse cells 0, 0.1, 1 along x give the
// middle one the slope 2 * 0.1 (not the central 0.5), so its children are
// 0.05 and 0.15; at an extremum the slope is 0.
TEST(AmrInterlevel, SlopesAreLimited) {
  const Box domain(IntVect{}, IntVect{{15, 15, 0}}, 2);
  const BoxArray coarse({domain}, domain);
  const BoxArray fine({Box(IntVect{{3, 3, 0}}, IntVect{{4, 5, 0}}, 2).refine(2)}, domain.refine(2));
  soma::amr::LevelField<double> c(coarse, IndexType::cell(), 1, 1), f(fine, IndexType::cell(), 1, 2);
  set_steps(c);
  f.fill(sentinel);
  soma::amr::fill_coarse_fine(f, c, soma::amr::CoarseFinePlan(fine, coarse, IndexType::cell(), 2, 2, {}, Interp::linear, 1));
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f.view().data);
  const Box g = f.grown(0);
  const auto get = [&](int i, int j) { return h(static_cast<std::size_t>((i - g.lo(0)) + g.length(0) * (j - g.lo(1)))); };
  for (int j = 6; j <= 11; ++j) {
    EXPECT_NEAR(get(4, j), 0.05, 1e-15);
    EXPECT_NEAR(get(5, j), 0.15, 1e-15);
    // Coarse cell 5 (values 1, 1, 0.5 around it): an extremum on one side,
    // slope 0.
    EXPECT_DOUBLE_EQ(get(10, j), 1.0);
    EXPECT_DOUBLE_EQ(get(11, j), 1.0);
  }
}
