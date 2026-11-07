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
  build(std::nullopt, fine, std::vector<int>(fine.size(), 0), HaloPlan(fine, type, fine_ghosts, periodic).uncovered(),
        coarse, Distribution{std::vector<int>(coarse.size(), 0)}, periodic, coarse_ghosts, default_tag);
}

CoarseFinePlan::CoarseFinePlan(const BoxArray& fine, const std::vector<BoxList>& targets, const BoxArray& coarse,
                               IndexType type, int fine_ghosts, int ratio, Periodicity periodic, Interp interp,
                               int coarse_ghosts)
    : type_(type), ratio_(ratio), fine_ghosts_(fine_ghosts), interp_(interp) {
  check_targets(fine, targets);
  build(std::nullopt, fine, std::vector<int>(fine.size(), 0), targets, coarse,
        Distribution{std::vector<int>(coarse.size(), 0)}, periodic, coarse_ghosts, default_tag);
}

CoarseFinePlan::CoarseFinePlan(const Comm& comm, const BoxArray& fine, const Distribution& fine_owner,
                               const BoxArray& coarse, const Distribution& coarse_owner, IndexType type, int fine_ghosts,
                               int ratio, Periodicity periodic, Interp interp, int coarse_ghosts, int tag)
    : type_(type), ratio_(ratio), fine_ghosts_(fine_ghosts), interp_(interp) {
  build(comm, fine, fine_owner.owner, HaloPlan(fine, type, fine_ghosts, periodic).uncovered(), coarse, coarse_owner,
        periodic, coarse_ghosts, tag);
}

CoarseFinePlan::CoarseFinePlan(const Comm& comm, const BoxArray& fine, const Distribution& fine_owner,
                               const std::vector<BoxList>& targets, const BoxArray& coarse,
                               const Distribution& coarse_owner, IndexType type, int fine_ghosts, int ratio,
                               Periodicity periodic, Interp interp, int coarse_ghosts, int tag)
    : type_(type), ratio_(ratio), fine_ghosts_(fine_ghosts), interp_(interp) {
  check_targets(fine, targets);
  build(comm, fine, fine_owner.owner, targets, coarse, coarse_owner, periodic, coarse_ghosts, tag);
}

void CoarseFinePlan::check_targets(const BoxArray& fine, const std::vector<BoxList>& targets) const {
  if (targets.size() != fine.size()) throw std::invalid_argument("CoarseFinePlan: one target list per fine box expected");
  for (std::size_t b = 0; b < fine.size(); ++b) {
    const Box grown = fine[b].convert(type_).grow(fine_ghosts_);
    for (const Box& t : targets[b]) {
      if (!grown.contains(t)) {
        throw std::invalid_argument("CoarseFinePlan: target " + text(t) + " lies outside fine box " + text(fine[b]));
      }
    }
  }
}

void CoarseFinePlan::build(const std::optional<Comm>& comm, const BoxArray& fine, const std::vector<int>& fine_owner,
                           const std::vector<BoxList>& targets_of, const BoxArray& coarse,
                           const Distribution& coarse_owner, Periodicity periodic, int coarse_ghosts, int tag) {
  const int ratio = ratio_;
  if (ratio < 2) throw std::invalid_argument("CoarseFinePlan: the ratio must be at least 2");
  if (!(fine.domain() == coarse.domain().refine(ratio))) {
    throw std::invalid_argument("CoarseFinePlan: the fine domain is not the coarse domain refined by the ratio");
  }
  if (fine_owner.size() != fine.size()) {
    throw std::invalid_argument("CoarseFinePlan: the fine distribution does not give every box an owner");
  }
  fine_owner_ = fine_owner;
  const int rank = comm ? comm->rank() : 0;
  // One scratch box per fine box with targets: the coarse points they need,
  // one more on each side for the linear slopes and the quadratic stencils.
  BoxList scratch;
  std::vector<BoxList> regions;
  std::vector<std::size_t> owner;  // the fine box of each scratch box
  std::vector<int> fine_local(fine.size(), -1);
  for (std::size_t b = 0, n = 0; b < fine.size(); ++b) {
    if (fine_owner[b] == rank) fine_local[b] = static_cast<int>(n++);
  }
  int scratch_local = 0;
  for (std::size_t b = 0; b < fine.size(); ++b) {
    BoxList targets;
    for (const Box& t : targets_of[b]) {
      if (!t.empty()) targets.push_back(t);
    }
    if (targets.empty()) continue;
    const Box need = bounding(targets).coarsen(ratio).grow(interp_ == Interp::piecewise_constant ? 0 : 1);
    const bool mine = fine_owner[b] == rank;
    for (const Box& t : targets) {
      ops_.push_back(InterpOp{static_cast<int>(scratch.size()), static_cast<int>(b), t});
      if (mine) {
        local_.push_back(InterpOp{scratch_local, fine_local[b], t});
        start_.push_back(start_.back() + t.num_pts());
      }
    }
    if (mine) ++scratch_local;
    // A cell box over the same index range: in the index type it holds need.
    scratch.push_back(Box(need.lo(), need.hi(), need.dim()));
    regions.push_back({need});
    owner.push_back(b);
    scratch_owner_.owner.push_back(fine_owner[b]);
  }
  scratch_ = BoxArray::scratch(scratch, coarse.domain());
  copy_ = comm ? CopyPlan(*comm, coarse, coarse_owner, regions, scratch_owner_, type_, periodic, coarse_ghosts, tag)
               : CopyPlan(coarse, regions, type_, periodic, coarse_ghosts);
  for (std::size_t s = 0; s < regions.size(); ++s) {
    if (!copy_.uncovered()[s].empty()) {
      throw std::invalid_argument("CoarseFinePlan: the coarse level does not cover the ghosts of fine box " +
                                  text(fine[owner[s]]) + " near coarse " +
                                  text(copy_.uncovered()[s].front()) + " (is the fine level properly nested?)");
    }
  }
  d_ops_ = to_device(local_, "amr::CoarseFinePlan.ops");
  d_start_ = to_device(start_, "amr::CoarseFinePlan.start");
}

AverageDownPlan::AverageDownPlan(const BoxArray& fine, const BoxArray& coarse, IndexType type, int ratio)
    : type_(type), ratio_(ratio) {
  build(std::nullopt, fine, std::vector<int>(fine.size(), 0), coarse, Distribution{std::vector<int>(coarse.size(), 0)},
        default_tag);
}

AverageDownPlan::AverageDownPlan(const Comm& comm, const BoxArray& fine, const Distribution& fine_owner,
                                 const BoxArray& coarse, const Distribution& coarse_owner, IndexType type, int ratio,
                                 int tag)
    : type_(type), ratio_(ratio) {
  build(comm, fine, fine_owner.owner, coarse, coarse_owner, tag);
}

void AverageDownPlan::build(const std::optional<Comm>& comm, const BoxArray& fine, const std::vector<int>& fine_owner,
                            const BoxArray& coarse, const Distribution& coarse_owner, int tag) {
  const int ratio = ratio_;
  if (ratio < 2) throw std::invalid_argument("average_down: the ratio must be at least 2");
  if (fine_owner.size() != fine.size()) {
    throw std::invalid_argument("average_down: the fine distribution does not give every box an owner");
  }
  fine_owner_ = fine_owner;
  scratch_owner_.owner = fine_owner;
  const int rank = comm ? comm->rank() : 0;
  // Scratch box f: the coarse cells under fine box f, which in the index type
  // hold the coarse points it covers (cells under cells, nodes on nodes).
  BoxList under;
  BoxList covered;  // the coarse points covered, disjoint
  int local = 0;
  for (std::size_t f = 0; f < fine.size(); ++f) {
    const Box& cells = fine[f];
    for (int d = 0; d < cells.dim(); ++d) {
      if (floor_div(cells.lo(d), ratio) * ratio != cells.lo(d) || floor_div(cells.hi(d) + 1, ratio) * ratio != cells.hi(d) + 1) {
        throw std::invalid_argument("average_down: fine box " + text(cells) + " is not aligned to the ratio " + std::to_string(ratio));
      }
    }
    const Box c = cells.coarsen(ratio);
    under.push_back(c);
    const Box points = c.convert(type_);
    for (const Box& piece : subtract(BoxList{points}, covered)) covered.push_back(piece);
    if (fine_owner[f] == rank) {
      ops_.push_back(InterpOp{local, local, points});
      start_.push_back(start_.back() + points.num_pts());
      ++local;
    }
  }
  scratch_ = BoxArray::scratch(under, coarse.domain());
  std::vector<BoxList> regions(coarse.size());
  for (std::size_t c = 0; c < coarse.size(); ++c) regions[c] = intersect(covered, coarse[c].convert(type_));
  copy_ = comm ? CopyPlan(*comm, scratch_, scratch_owner_, regions, coarse_owner, type_, Periodicity{}, 0, tag)
               : CopyPlan(scratch_, regions, type_, Periodicity{});
  d_ops_ = to_device(ops_, "amr::AverageDownPlan.ops");
  d_start_ = to_device(start_, "amr::AverageDownPlan.start");
}

}  // namespace soma::amr
