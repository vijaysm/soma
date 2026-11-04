// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/interlevel.hpp"

#include <string>

#include "bins.hpp"

namespace soma::amr {

namespace {

template <class U>
Kokkos::View<const U*, CoarseFinePlan::DeviceSpace> to_device(const std::vector<U>& v, const std::string& label) {
  Kokkos::View<U*, CoarseFinePlan::DeviceSpace> d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

std::string text(const Box& b) {
  std::string s = "[";
  for (int d = 0; d < b.dim(); ++d) s += (d ? "," : "") + std::to_string(b.lo(d));
  s += "]-[";
  for (int d = 0; d < b.dim(); ++d) s += (d ? "," : "") + std::to_string(b.hi(d));
  return s + "]";
}

// The smallest box holding every box of the list.
Box bounding(const BoxList& l) {
  Box out = l.front();
  for (const Box& b : l) {
    IntVect lo = out.lo(), hi = out.hi();
    for (int d = 0; d < 3; ++d) {
      lo[d] = b.lo(d) < lo[d] ? b.lo(d) : lo[d];
      hi[d] = b.hi(d) > hi[d] ? b.hi(d) : hi[d];
    }
    out = Box(lo, hi, b.dim(), b.type());
  }
  return out;
}

}  // namespace

CoarseFinePlan::CoarseFinePlan(const BoxArray& fine, const BoxArray& coarse, IndexType type, int fine_ghosts, int ratio,
                               Periodicity periodic, Interp interp, int coarse_ghosts)
    : type_(type), ratio_(ratio), fine_ghosts_(fine_ghosts), interp_(interp) {
  build(fine, HaloPlan(fine, type, fine_ghosts, periodic).uncovered(), coarse, periodic, coarse_ghosts);
}

CoarseFinePlan::CoarseFinePlan(const BoxArray& fine, const std::vector<BoxList>& targets, const BoxArray& coarse,
                               IndexType type, int fine_ghosts, int ratio, Periodicity periodic, Interp interp,
                               int coarse_ghosts)
    : type_(type), ratio_(ratio), fine_ghosts_(fine_ghosts), interp_(interp) {
  if (targets.size() != fine.size()) throw std::invalid_argument("CoarseFinePlan: one target list per fine box expected");
  for (std::size_t b = 0; b < fine.size(); ++b) {
    const Box grown = fine[b].convert(type).grow(fine_ghosts);
    for (const Box& t : targets[b]) {
      if (!grown.contains(t)) {
        throw std::invalid_argument("CoarseFinePlan: target " + text(t) + " lies outside fine box " + text(fine[b]));
      }
    }
  }
  build(fine, targets, coarse, periodic, coarse_ghosts);
}

void CoarseFinePlan::build(const BoxArray& fine, const std::vector<BoxList>& targets_of, const BoxArray& coarse,
                           Periodicity periodic, int coarse_ghosts) {
  const int ratio = ratio_;
  if (ratio < 2) throw std::invalid_argument("CoarseFinePlan: the ratio must be at least 2");
  if (!(fine.domain() == coarse.domain().refine(ratio))) {
    throw std::invalid_argument("CoarseFinePlan: the fine domain is not the coarse domain refined by the ratio");
  }
  // One scratch box per fine box with targets: the coarse points they need,
  // one more on each side for the linear slopes and the quadratic stencils.
  BoxList scratch;
  std::vector<BoxList> regions;
  std::vector<std::size_t> owner;  // the fine box of each scratch box
  for (std::size_t b = 0; b < fine.size(); ++b) {
    BoxList targets;
    for (const Box& t : targets_of[b]) {
      if (!t.empty()) targets.push_back(t);
    }
    if (targets.empty()) continue;
    const Box need = bounding(targets).coarsen(ratio).grow(interp_ == Interp::piecewise_constant ? 0 : 1);
    for (const Box& t : targets) {
      ops_.push_back(InterpOp{static_cast<int>(scratch.size()), static_cast<int>(b), t});
      start_.push_back(start_.back() + t.num_pts());
    }
    // A cell box over the same index range: in the index type it holds need.
    scratch.push_back(Box(need.lo(), need.hi(), need.dim()));
    regions.push_back({need});
    owner.push_back(b);
  }
  scratch_ = BoxArray::scratch(scratch, coarse.domain());
  copy_ = CopyPlan(coarse, regions, type_, periodic, coarse_ghosts);
  for (std::size_t s = 0; s < regions.size(); ++s) {
    if (!copy_.uncovered()[s].empty()) {
      throw std::invalid_argument("CoarseFinePlan: the coarse level does not cover the ghosts of fine box " +
                                  text(fine[owner[s]]) + " near coarse " +
                                  text(copy_.uncovered()[s].front()) + " (is the fine level properly nested?)");
    }
  }
  d_ops_ = to_device(ops_, "amr::CoarseFinePlan.ops");
  d_start_ = to_device(start_, "amr::CoarseFinePlan.start");
}

std::vector<InterpOp> detail::restriction_ops(const BoxArray& fine, const BoxArray& coarse, IndexType type, int ratio) {
  if (ratio < 2) throw std::invalid_argument("average_down: the ratio must be at least 2");
  std::vector<InterpOp> out;
  const detail::Bins bins(coarse.boxes(), coarse.domain());
  for (std::size_t f = 0; f < fine.size(); ++f) {
    const Box& cells = fine[f];
    for (int d = 0; d < cells.dim(); ++d) {
      if (floor_div(cells.lo(d), ratio) * ratio != cells.lo(d) || floor_div(cells.hi(d) + 1, ratio) * ratio != cells.hi(d) + 1) {
        throw std::invalid_argument("average_down: fine box " + text(cells) + " is not aligned to the ratio " + std::to_string(ratio));
      }
    }
    // Coarse points covered: cells under the fine cells, nodes on fine nodes.
    const Box covered = cells.coarsen(ratio).convert(type);
    for (const int c : bins.near(cells.coarsen(ratio).grow(1))) {
      const Box r = covered & coarse[static_cast<std::size_t>(c)].convert(type);
      if (!r.empty()) out.push_back(InterpOp{c, static_cast<int>(f), r});
    }
  }
  return out;
}

}  // namespace soma::amr
