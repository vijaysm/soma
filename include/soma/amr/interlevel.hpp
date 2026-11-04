// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_INTERLEVEL_HPP_
#define SOMA_AMR_INTERLEVEL_HPP_

// Operators between a level and the next coarser one (refinement ratio r).
//
// Prolongation fills the ghost points of a fine level that no fine box covers
// (HaloPlan::uncovered) from the coarse level. Per direction: where points
// are nodes, linear between the two coarse nodes around the fine one (fine
// nodes on coarse nodes are copied); where they are cells, the coarse value
// (piecewise constant) or the coarse value plus a slope limited by the
// monotonized-central limiter times the offset of the fine cell centre from
// the coarse one (linear: conservative, and exact for linear data). So cell
// data are piecewise constant or conservative linear, node data
// multilinear, and face data linear across the faces and constant or
// limited-linear along them. The coarse level must cover the coarse points
// needed, its valid points or its ghosts filled by the caller (one ghost for
// the linear slopes next to the domain boundary): a fine level properly
// nested in the coarse one does.
//
// Quadratic prolongation (for third-order schemes and Q2 data), per
// direction over the coarse points q - 1, q, q + 1 around the fine one: where
// points are nodes, the quadratic through the three nodes at the fine node;
// where they are cells, the average over the fine cell of the quadratic whose
// averages over the three coarse cells are theirs (conservative). Tensor
// products of these are exact for data of degree at most 2 in each
// direction; they are not limited, so they suit smooth data.
//
// Restriction (average_down) sets each coarse point under the fine valid
// points to the average of the fine points it covers: cell averages
// (conservative), node injection, face averages over the coincident fine
// faces. Fine boxes must be aligned to the ratio.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/amr/box.hpp"
#include "soma/amr/box_array.hpp"
#include "soma/amr/copy.hpp"
#include "soma/amr/halo.hpp"
#include "soma/amr/level_field.hpp"

namespace soma::amr {

enum class Interp { piecewise_constant, linear, quadratic };

// Fill region `region` of fine box `fine` from scratch box `coarse`.
struct InterpOp {
  int coarse = 0;
  int fine = 0;
  Box region;
};

class CoarseFinePlan {
 public:
  using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

  CoarseFinePlan() = default;
  // Throws std::invalid_argument naming a fine box whose ghosts the coarse
  // level (valid points, then coarse_ghosts ghost points) does not cover.
  CoarseFinePlan(const BoxArray& fine, const BoxArray& coarse, IndexType type, int fine_ghosts, int ratio,
                 Periodicity periodic, Interp interp, int coarse_ghosts = 1);
  // Fills points `targets[b]` of fine box b (in the index type, inside its
  // valid points grown by fine_ghosts) instead, valid points included: the
  // prolongation of a regrid.
  CoarseFinePlan(const BoxArray& fine, const std::vector<BoxList>& targets, const BoxArray& coarse, IndexType type,
                 int fine_ghosts, int ratio, Periodicity periodic, Interp interp, int coarse_ghosts = 1);

  IndexType type() const { return type_; }
  int ratio() const { return ratio_; }
  int fine_ghosts() const { return fine_ghosts_; }
  int coarse_ghosts() const { return copy_.src_ghosts(); }
  Interp interp() const { return interp_; }
  const BoxArray& scratch() const { return scratch_; }
  const CopyPlan& copy_plan() const { return copy_; }
  const std::vector<InterpOp>& ops() const { return ops_; }
  std::int64_t num_points() const { return start_.back(); }
  Kokkos::View<const InterpOp*, DeviceSpace> device_ops() const { return d_ops_; }
  Kokkos::View<const std::int64_t*, DeviceSpace> device_start() const { return d_start_; }

 private:
  void build(const BoxArray& fine, const std::vector<BoxList>& targets, const BoxArray& coarse, Periodicity periodic,
             int coarse_ghosts);

  IndexType type_{};
  int ratio_ = 2;
  int fine_ghosts_ = 0;
  Interp interp_ = Interp::linear;
  BoxArray scratch_;
  CopyPlan copy_;
  std::vector<InterpOp> ops_;
  std::vector<std::int64_t> start_{0};
  Kokkos::View<const InterpOp*, DeviceSpace> d_ops_;
  Kokkos::View<const std::int64_t*, DeviceSpace> d_start_;
};

namespace detail {

KOKKOS_INLINE_FUNCTION double mc_slope(double minus, double plus) {
  if (minus * plus <= 0) return 0;
  const double a = minus < 0 ? -minus : minus, b = plus < 0 ? -plus : plus;
  const double central = 0.5 * (a + b);
  double m = 2 * a < 2 * b ? 2 * a : 2 * b;
  m = central < m ? central : m;
  return plus > 0 ? m : -m;
}

// The value at fine point p from the coarse patch s, component c.
template <class T>
KOKKOS_INLINE_FUNCTION T interpolate(const Patch<T>& s, const IntVect& p, int c, int dim, IndexType type, int r,
                                     bool linear) {
  int base[3] = {0, 0, 0}, nodal_next[3] = {0, 0, 0};
  double w[3] = {0, 0, 0}, off[3] = {0, 0, 0};
  for (int d = 0; d < dim; ++d) {
    base[d] = floor_div(p[d], r);
    const int rem = p[d] - base[d] * r;
    if (type.is_nodal(d)) {
      w[d] = static_cast<double>(rem) / r;
      nodal_next[d] = rem != 0 ? 1 : 0;
    } else {
      off[d] = (rem + 0.5) / r - 0.5;
    }
  }
  double sum = 0;
  for (int corner = 0; corner < 8; ++corner) {
    int q[3] = {base[0], base[1], base[2]};
    double weight = 1;
    bool skip = false;
    for (int d = 0; d < 3; ++d) {
      const int bit = (corner >> d) & 1;
      if (d >= dim || !type.is_nodal(d)) {
        skip = skip || bit != 0;
        continue;
      }
      if (bit != 0 && nodal_next[d] == 0) skip = true;
      q[d] += bit;
      weight *= bit != 0 ? w[d] : 1 - w[d];
    }
    if (skip) continue;
    double v = static_cast<double>(s(q[0], q[1], q[2], c));
    if (linear) {
      for (int d = 0; d < dim; ++d) {
        if (type.is_nodal(d)) continue;
        int m[3] = {q[0], q[1], q[2]}, n[3] = {q[0], q[1], q[2]};
        --m[d];
        ++n[d];
        const double here = static_cast<double>(s(q[0], q[1], q[2], c));
        v += mc_slope(here - static_cast<double>(s(m[0], m[1], m[2], c)), static_cast<double>(s(n[0], n[1], n[2], c)) - here) * off[d];
      }
    }
    sum += weight * v;
  }
  return static_cast<T>(sum);
}

// The quadratic rule's weights over the coarse offsets -1, 0, 1 in one
// direction, for fine index p at ratio r.
KOKKOS_INLINE_FUNCTION void quadratic_weights(int p, int r, bool nodal, double w[3]) {
  const int rem = p - floor_div(p, r) * r;
  if (nodal) {
    // Lagrange through the nodes at -1, 0, 1, at x in [0, 1).
    const double x = static_cast<double>(rem) / r;
    w[0] = 0.5 * x * (x - 1);
    w[1] = 1 - x * x;
    w[2] = 0.5 * x * (x + 1);
  } else {
    // Averages over [-3/2, -1/2], [-1/2, 1/2], [1/2, 3/2] fix
    // u(xi) = a + b xi + c xi^2; the fine cell's average, centred at off and
    // 1/r wide, is u0 + b off + c (off^2 + 1/(12 r^2) - 1/12).
    const double off = (rem + 0.5) / r - 0.5;
    const double s = off * off + 1.0 / (12.0 * r * r) - 1.0 / 12.0;
    w[0] = 0.5 * (s - off);
    w[1] = 1 - s;
    w[2] = 0.5 * (s + off);
  }
}

// The quadratic rule's value at fine point p from coarse patch s.
template <class T>
KOKKOS_INLINE_FUNCTION T interpolate_quadratic(const Patch<T>& s, const IntVect& p, int c, int dim, IndexType type, int r) {
  double w[3][3] = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
  int base[3] = {0, 0, 0};
  for (int d = 0; d < dim; ++d) {
    base[d] = floor_div(p[d], r);
    quadratic_weights(p[d], r, type.is_nodal(d), w[d]);
  }
  const int lo2 = dim > 2 ? -1 : 0, hi2 = dim > 2 ? 1 : 0;
  double sum = 0;
  for (int k = lo2; k <= hi2; ++k) {
    for (int j = -1; j <= 1; ++j) {
      for (int i = -1; i <= 1; ++i) {
        const double wt = w[0][i + 1] * w[1][j + 1] * w[2][k + 1];
        if (wt == 0) continue;
        sum += wt * static_cast<double>(s(base[0] + i, base[1] + j, base[2] + k, c));
      }
    }
  }
  return static_cast<T>(sum);
}

}  // namespace detail

// Fills the fine ghost points no fine box covers from the coarse level along
// the plan (the coarse level's ghosts filled first by the caller).
template <class T>
void fill_coarse_fine(const LevelField<T>& fine, const LevelField<T>& coarse, const CoarseFinePlan& plan) {
  using Exec = Kokkos::DefaultExecutionSpace;
  if (!(fine.type() == plan.type()) || !(coarse.type() == plan.type()) || fine.ghosts() < plan.fine_ghosts() ||
      coarse.ghosts() < plan.coarse_ghosts() || fine.ncomp() != coarse.ncomp()) {
    throw std::invalid_argument("fill_coarse_fine: the fields do not match the plan");
  }
  const LevelField<T> scratch(plan.scratch(), plan.type(), coarse.ncomp(), 0, "amr::fill_coarse_fine.scratch");
  detail::run_copies(coarse.view(), scratch.view(), plan.copy_plan());
  const auto ops = plan.device_ops();
  const auto start = plan.device_start();
  const auto s = scratch.view();
  const auto f = fine.view();
  const int n = static_cast<int>(ops.extent(0));
  const int ncomp = fine.ncomp(), dim = plan.scratch().domain().dim(), r = plan.ratio();
  const IndexType type = plan.type();
  const bool linear = plan.interp() == Interp::linear, quadratic = plan.interp() == Interp::quadratic;
  Kokkos::parallel_for(
      "amr::fill_coarse_fine", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, plan.num_points()),
      KOKKOS_LAMBDA(std::int64_t p) {
        int lo = 0, hi = n - 1;
        while (lo < hi) {
          const int mid = (lo + hi + 1) / 2;
          if (start(mid) <= p) {
            lo = mid;
          } else {
            hi = mid - 1;
          }
        }
        const InterpOp& op = ops(lo);
        const IntVect q = detail::point_of(op.region, p - start(lo));
        const auto from = s.patch(op.coarse);
        const auto to = f.patch(op.fine);
        for (int c = 0; c < ncomp; ++c) {
          to(q[0], q[1], q[2], c) = quadratic ? detail::interpolate_quadratic(from, q, c, dim, type, r)
                                              : detail::interpolate(from, q, c, dim, type, r, linear);
        }
      });
  Kokkos::fence("amr::fill_coarse_fine");
}

// Sets the coarse points under the fine level's valid points to the average
// of the fine points they cover. Throws std::invalid_argument for fields of
// different index types or component counts, or a fine box not aligned to
// the ratio.
template <class T>
void average_down(const LevelField<T>& fine, const LevelField<T>& coarse, int ratio);

namespace detail {

// The restriction work: coarse points `region` of coarse box `coarse` from
// fine box `fine`.
std::vector<InterpOp> restriction_ops(const BoxArray& fine, const BoxArray& coarse, IndexType type, int ratio);

}  // namespace detail

template <class T>
void average_down(const LevelField<T>& fine, const LevelField<T>& coarse, int ratio) {
  using Exec = Kokkos::DefaultExecutionSpace;
  if (!(fine.type() == coarse.type()) || fine.ncomp() != coarse.ncomp()) {
    throw std::invalid_argument("average_down: the fields differ in index type or components");
  }
  const std::vector<InterpOp> ops_h = detail::restriction_ops(fine.boxes(), coarse.boxes(), fine.type(), ratio);
  std::vector<std::int64_t> start_h{0};
  for (const InterpOp& op : ops_h) start_h.push_back(start_h.back() + op.region.num_pts());
  Kokkos::View<InterpOp*, typename LevelField<T>::memory_space> ops("amr::average_down.ops", ops_h.size());
  Kokkos::View<std::int64_t*, typename LevelField<T>::memory_space> start("amr::average_down.start", start_h.size());
  auto oh = Kokkos::create_mirror_view(ops);
  auto sh = Kokkos::create_mirror_view(start);
  for (std::size_t i = 0; i < ops_h.size(); ++i) oh(i) = ops_h[i];
  for (std::size_t i = 0; i < start_h.size(); ++i) sh(i) = start_h[i];
  Kokkos::deep_copy(ops, oh);
  Kokkos::deep_copy(start, sh);
  const auto f = fine.view();
  const auto cv = coarse.view();
  const int n = static_cast<int>(ops_h.size());
  const int ncomp = fine.ncomp(), dim = fine.boxes().domain().dim(), r = ratio;
  const IndexType type = fine.type();
  Kokkos::parallel_for(
      "amr::average_down", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, start_h.back()),
      KOKKOS_LAMBDA(std::int64_t p) {
        int lo = 0, hi = n - 1;
        while (lo < hi) {
          const int mid = (lo + hi + 1) / 2;
          if (start(mid) <= p) {
            lo = mid;
          } else {
            hi = mid - 1;
          }
        }
        const InterpOp& op = ops(lo);
        const IntVect q = detail::point_of(op.region, p - start(lo));
        const auto from = f.patch(op.fine);
        const auto to = cv.patch(op.coarse);
        int len[3] = {1, 1, 1};
        for (int d = 0; d < dim; ++d) len[d] = type.is_nodal(d) ? 1 : r;
        for (int c = 0; c < ncomp; ++c) {
          double sum = 0;
          for (int k = 0; k < len[2]; ++k) {
            for (int j = 0; j < len[1]; ++j) {
              for (int i = 0; i < len[0]; ++i) {
                sum += static_cast<double>(from(q[0] * r + i, q[1] * r + j, dim > 2 ? q[2] * r + k : 0, c));
              }
            }
          }
          to(q[0], q[1], q[2], c) = static_cast<T>(sum / (len[0] * len[1] * len[2]));
        }
      });
  Kokkos::fence("amr::average_down");
}

}  // namespace soma::amr

#endif  // SOMA_AMR_INTERLEVEL_HPP_
