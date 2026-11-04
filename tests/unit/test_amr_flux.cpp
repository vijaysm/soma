// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Refluxing: a conservative update on two levels with arbitrary fluxes keeps
// the total to round-off once the register corrects the coarse cells, and
// the register's faces are exactly the coarse-fine faces.

#include <cmath>
#include <cstdint>
#include <set>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/amr/box_list.hpp"
#include "soma/amr/flux_register.hpp"
#include "soma/amr/interlevel.hpp"

namespace {

using soma::amr::Box;
using soma::amr::BoxArray;
using soma::amr::BoxList;
using soma::amr::IndexType;
using soma::amr::IntVect;
using soma::amr::LevelField;
using soma::amr::Periodicity;

// Data with no structure, by wrapped position, so shared faces agree.
KOKKOS_INLINE_FUNCTION double noise(IntVect p, int salt, const Box& dom, int ratio_to_dom) {
  for (int d = 0; d < dom.dim(); ++d) {
    const int len = dom.length(d) * ratio_to_dom;
    p[d] = ((p[d] % len) + len) % len;
  }
  const unsigned u = static_cast<unsigned>(p[0] * 73856093) ^ static_cast<unsigned>(p[1] * 19349663) ^
                     static_cast<unsigned>(p[2] * 83492791) ^ (static_cast<unsigned>(salt) * 2654435761u);
  return static_cast<double>(u % 1000u) / 1000.0 - 0.5;
}

double total(const LevelField<double>& u) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u.view().data);
  double s = 0;
  for (std::size_t b = 0; b < u.num_boxes(); ++b) {
    const Box g = u.grown(b), v = u.valid(b);
    for (std::int64_t q = 0; q < g.num_pts(); ++q) {
      if (v.contains(soma::amr::detail::point_of(g, q))) s += h(static_cast<std::size_t>(u.offset(b) + q));
    }
  }
  return s;
}

// u -= dtdx * (F(hi face) - F(lo face)) in every direction.
void update(const LevelField<double>& u, const std::vector<LevelField<double>>& flux, double dtdx) {
  const auto uv = u.view();
  for (std::size_t d = 0; d < flux.size(); ++d) {
    const auto fv = flux[d].view();
    const int dd = static_cast<int>(d);
    soma::amr::for_each_point(u, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
      int hi[3] = {i, j, k};
      ++hi[dd];
      uv.patch(b)(i, j, k) -= dtdx * (fv.patch(b)(hi[0], hi[1], hi[2]) - fv.patch(b)(i, j, k));
    });
  }
}

struct Config {
  int dim, r;
  Periodicity per;
};

void conservation(const Config& s, bool reflux, double& m0, double& m1) {
  const int dim = s.dim, r = s.r;
  const Box domain = dim == 2 ? Box(IntVect{}, IntVect{{15, 11, 0}}, 2) : Box(IntVect{}, IntVect{{7, 7, 5}});
  const BoxArray coarse(soma::amr::chop({domain}, 5), domain);
  // One fine box against the domain's low corner, another one coarse cell
  // away from it (that cell is outside both).
  const BoxArray fine({Box(IntVect{{0, 0, 0}}, IntVect{{2, 3, 2}}, dim).refine(r), Box(IntVect{{4, 1, 1}}, IntVect{{5, 4, 3}}, dim).refine(r)},
                      domain.refine(r));
  LevelField<double> uc(coarse, IndexType::cell(), 1, 0), uf(fine, IndexType::cell(), 1, 0);
  std::vector<LevelField<double>> fc, ff;
  for (int d = 0; d < dim; ++d) {
    fc.emplace_back(coarse, IndexType::face(d), 1, 0);
    ff.emplace_back(fine, IndexType::face(d), 1, 0);
  }
  const Box dom = domain;
  const auto ucv = uc.view(), ufv = uf.view();
  soma::amr::for_each_point(uc, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    ucv.patch(b)(i, j, k) = 1 + noise(IntVect{{i, j, k}}, 1, dom, 1);
  });
  soma::amr::for_each_point(uf, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    ufv.patch(b)(i, j, k) = 1 + noise(IntVect{{i, j, k}}, 2, dom, r);
  });
  for (int d = 0; d < dim; ++d) {
    const auto a = fc[static_cast<std::size_t>(d)].view(), f = ff[static_cast<std::size_t>(d)].view();
    soma::amr::for_each_point(fc[static_cast<std::size_t>(d)], soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
      a.patch(b)(i, j, k) = noise(IntVect{{i, j, k}}, 10 + d, dom, 1);
    });
    soma::amr::for_each_point(ff[static_cast<std::size_t>(d)], soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
      f.patch(b)(i, j, k) = noise(IntVect{{i, j, k}}, 20 + d, dom, r);
    });
  }
  soma::amr::average_down(uf, uc, r);
  m0 = total(uc);
  const double dtdx = 0.1;
  update(uc, fc, dtdx);
  update(uf, ff, dtdx * r);
  const soma::amr::FluxRegister reg(fine, coarse, r, s.per, 1);
  // Whatever a register held goes at reset.
  for (int d = 0; d < dim; ++d) {
    reg.set_coarse(d, fc[static_cast<std::size_t>(d)], 5.0);
    reg.add_fine(d, ff[static_cast<std::size_t>(d)], 7.0);
  }
  reg.reset();
  for (int d = 0; d < dim; ++d) {
    reg.set_coarse(d, fc[static_cast<std::size_t>(d)], dtdx);
    // Two half steps add up to one.
    reg.add_fine(d, ff[static_cast<std::size_t>(d)], dtdx / 2);
    reg.add_fine(d, ff[static_cast<std::size_t>(d)], dtdx / 2);
  }
  soma::amr::average_down(uf, uc, r);
  // A copy, not a mirror: on a host backend the mirror is the data itself.
  Kokkos::View<double*, Kokkos::HostSpace> before("before", uc.view().data.extent(0));
  Kokkos::deep_copy(before, uc.view().data);
  if (reflux) reg.reflux(uc);
  m1 = total(uc);
  // Exactly the coarse cells outside the fine level with a face on it
  // change (with these fluxes every correction is nonzero).
  auto after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, uc.view().data);
  const auto under = [&](IntVect p) {
    for (int d = 0; d < dim; ++d) p[d] = ((p[d] % domain.length(d)) + domain.length(d)) % domain.length(d);
    for (std::size_t f = 0; f < fine.size(); ++f) {
      if (fine[f].coarsen(r).contains(p)) return true;
    }
    return false;
  };
  if (reflux) {
    for (std::size_t b = 0; b < uc.num_boxes(); ++b) {
      const Box g = uc.grown(b);
      for (std::int64_t q = 0; q < g.num_pts(); ++q) {
        const IntVect p = soma::amr::detail::point_of(g, q);
        bool next = false;
        for (int d = 0; d < dim; ++d) {
          for (const int step : {-1, 1}) {
            IntVect n = p;
            n[d] += step;
            next = next || under(n);
          }
        }
        const bool want = !under(p) && next;
        const auto i = static_cast<std::size_t>(uc.offset(b) + q);
        EXPECT_EQ(before(i) != after(i), want) << "cell " << p[0] << "," << p[1] << "," << p[2] << " dim " << dim << " r " << r;
      }
    }
  }
}

}  // namespace

TEST(AmrFluxRegister, RefluxingConserves) {
  for (const int dim : {2, 3}) {
    for (const int r : {2, 4}) {
      double m0 = 0, m1 = 0, n0 = 0, n1 = 0;
      const Config s{dim, r, Periodicity{{true, true, true}}};
      conservation(s, true, m0, m1);
      conservation(s, false, n0, n1);
      EXPECT_NEAR(m1, m0, 1e-11 * std::abs(m0)) << dim << " " << r;
      // Without the correction the fluxes do not match: the test can tell.
      EXPECT_GT(std::abs(n1 - n0), 1e-3) << dim << " " << r;
    }
  }
}

// The faces are the coarse faces with the fine level on exactly one side
// (periodic images included; none on a boundary that is not periodic).
TEST(AmrFluxRegister, FacesAreTheCoarseFineFaces) {
  for (const int dim : {2, 3}) {
    for (const Periodicity per : {Periodicity{}, Periodicity{{true, false, true}}}) {
      const int r = 2;
      const Box domain = dim == 2 ? Box(IntVect{}, IntVect{{15, 11, 0}}, 2) : Box(IntVect{}, IntVect{{7, 7, 5}});
      const BoxArray coarse(soma::amr::chop({domain}, 5), domain);
      // The last box meets the first one across the periodic x boundary.
      const BoxList under{Box(IntVect{{0, 0, 0}}, IntVect{{2, 3, 2}}, dim), Box(IntVect{{4, 1, 1}}, IntVect{{5, 4, 3}}, dim),
                          Box(IntVect{{6, 1, 1}}, IntVect{{6, 2, 2}}, dim),
                          Box(IntVect{{domain.hi(0), 0, 0}}, IntVect{{domain.hi(0), 1, 1}}, dim)};
      const BoxArray fine(soma::amr::refine(under, r), domain.refine(r));
      const soma::amr::FluxRegister reg(fine, coarse, r, per, 1);
      const auto is_under = [&](IntVect p) {
        for (int d = 0; d < dim; ++d) {
          if (p[d] < domain.lo(d) || p[d] > domain.hi(d)) {
            if (!per[d]) return -1;  // no cell there
            const int len = domain.length(d);
            p[d] = ((p[d] % len) + len) % len;
          }
        }
        for (const Box& b : under) {
          if (b.contains(p)) return 1;
        }
        return 0;
      };
      std::set<std::tuple<int, int, int, int>> want, got;
      for (int d = 0; d < dim; ++d) {
        // Across a periodic direction face L is face 0.
        const Box faces = domain.convert(IndexType::face(d)).grow(d, 0, per[d] ? -1 : 0);
        for (std::int64_t q = 0; q < faces.num_pts(); ++q) {
          const IntVect f = soma::amr::detail::point_of(faces, q);
          IntVect below = f;
          --below[d];
          const int a = is_under(below), b = is_under(f);
          if ((a == 1 && b == 0) || (a == 0 && b == 1)) want.emplace(d, f[0], f[1], f[2]);
        }
      }
      for (const auto& pc : reg.faces()) {
        EXPECT_TRUE(pc.faces.type() == IndexType::face(pc.dir));
        for (std::int64_t q = 0; q < pc.faces.num_pts(); ++q) {
          const IntVect f = soma::amr::detail::point_of(pc.faces, q);
          IntVect cell = f + pc.shift;
          if (pc.side == 0) --cell[pc.dir];
          EXPECT_TRUE(coarse[static_cast<std::size_t>(pc.coarse)].contains(cell));
          EXPECT_TRUE(fine[static_cast<std::size_t>(pc.fine)].coarsen(r).grow(1).contains(cell - pc.shift));
          EXPECT_TRUE(got.emplace(pc.dir, f[0], f[1], f[2]).second) << "face listed twice";
        }
      }
      EXPECT_EQ(got, want) << "dim " << dim << " periodic " << per[0];
      EXPECT_EQ(reg.num_faces(), static_cast<std::int64_t>(want.size()));
    }
  }
}

TEST(AmrFluxRegister, RejectsWhatItCannotHold) {
  const Box domain(IntVect{}, IntVect{{7, 7, 0}}, 2);
  const BoxArray coarse({domain}, domain);
  const BoxArray fine({Box(IntVect{{2, 2, 0}}, IntVect{{3, 3, 0}}, 2).refine(2)}, domain.refine(2));
  EXPECT_THROW(soma::amr::FluxRegister(fine, coarse, 4, {}, 1), std::invalid_argument);
  EXPECT_THROW(soma::amr::FluxRegister(fine, coarse, 2, {}, 0), std::invalid_argument);
  // Coarse cells next to the fine box missing from the coarse level.
  const BoxArray holed(soma::amr::complement(domain, {Box(IntVect{{1, 2, 0}}, IntVect{{1, 3, 0}}, 2)}), domain);
  EXPECT_THROW(soma::amr::FluxRegister(fine, holed, 2, {}, 1), std::invalid_argument);
  const soma::amr::FluxRegister reg(fine, coarse, 2, {}, 1);
  LevelField<double> cells(coarse, IndexType::cell(), 1, 0), xf(coarse, IndexType::face(0), 1, 0);
  EXPECT_THROW(reg.set_coarse(1, xf, 1.0), std::invalid_argument);
  EXPECT_THROW(reg.add_fine(0, cells, 1.0), std::invalid_argument);
  EXPECT_THROW(reg.reflux(xf), std::invalid_argument);
}
