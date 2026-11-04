// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/halo.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "soma/amr/box_list.hpp"

namespace soma::amr {

namespace {

// Boxes binned on a uniform grid of the domain, bins as large as the largest
// box, so a box lies in at most 2^dim bins and a query visits few.
class Bins {
 public:
  Bins(const BoxArray& boxes) : domain_(boxes.domain()) {
    for (std::size_t b = 0; b < boxes.size(); ++b) {
      for (int d = 0; d < domain_.dim(); ++d) size_ = std::max(size_, boxes[b].length(d));
    }
    for (int d = 0; d < 3; ++d) n_[d] = d < domain_.dim() ? (domain_.length(d) + size_ - 1) / size_ : 1;
    bins_.resize(static_cast<std::size_t>(n_[0]) * static_cast<std::size_t>(n_[1]) * static_cast<std::size_t>(n_[2]));
    for (std::size_t b = 0; b < boxes.size(); ++b) {
      visit(boxes[b], [&](std::size_t bin) { bins_[bin].push_back(static_cast<int>(b)); });
    }
  }

  // The boxes in bins that the cell box q (clamped to the domain) meets,
  // each once, in increasing order.
  std::vector<int> near(const Box& q) const {
    std::vector<int> out;
    visit(q, [&](std::size_t bin) { out.insert(out.end(), bins_[bin].begin(), bins_[bin].end()); });
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }

 private:
  template <class F>
  void visit(const Box& q, const F& f) const {
    int lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    for (int d = 0; d < domain_.dim(); ++d) {
      lo[d] = std::clamp((q.lo(d) - domain_.lo(d)) / size_, 0, n_[d] - 1);
      hi[d] = std::clamp(floor_div(q.hi(d) - domain_.lo(d), size_), 0, n_[d] - 1);
      if (q.hi(d) < domain_.lo(d) || q.lo(d) > domain_.hi(d)) return;
    }
    for (int k = lo[2]; k <= hi[2]; ++k) {
      for (int j = lo[1]; j <= hi[1]; ++j) {
        for (int i = lo[0]; i <= hi[0]; ++i) f(static_cast<std::size_t>(i + n_[0] * (j + n_[1] * k)));
      }
    }
  }

  Box domain_;
  int size_ = 1;
  int n_[3] = {1, 1, 1};
  std::vector<std::vector<int>> bins_;
};

template <class U>
Kokkos::View<const U*, Kokkos::DefaultExecutionSpace::memory_space> to_device(const std::vector<U>& v, const std::string& label) {
  Kokkos::View<U*, Kokkos::DefaultExecutionSpace::memory_space> d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

}  // namespace

HaloPlan::HaloPlan(const BoxArray& boxes, IndexType type, int ghosts, Periodicity periodic) : type_(type), ghosts_(ghosts) {
  if (ghosts < 0) throw std::invalid_argument("HaloPlan: ghosts must not be negative");
  const Box& domain = boxes.domain();
  const int dim = domain.dim();
  const Box domain_t = domain.convert(type);
  // Periodic images: shifts by the domain length in cells, each periodic
  // direction -1, 0 or +1, the unshifted one first.
  std::vector<IntVect> shifts{IntVect{}};
  for (int d = 0; d < dim; ++d) {
    if (!periodic[d]) continue;
    const auto n = shifts.size();
    for (const int s : {-1, 1}) {
      for (std::size_t i = 0; i < n; ++i) {
        IntVect x = shifts[i];
        x[d] = s * domain.length(d);
        shifts.push_back(x);
      }
    }
  }
  const Bins bins(boxes);
  start_.push_back(0);
  for (std::size_t b = 0; b < boxes.size(); ++b) {
    const Box valid = boxes[b].convert(type);
    const Box grown = valid.grow(ghosts);
    BoxList left = subtract(grown, valid);
    // Candidates: every (source, image) whose cell box comes near; sources in
    // curve order, then images in the order above.
    std::vector<std::pair<int, std::size_t>> cand;
    for (std::size_t s = 0; s < shifts.size(); ++s) {
      const IntVect back = IntVect{} - shifts[s];
      for (const int src : bins.near(boxes[b].grow(ghosts + 1).shift(back))) cand.emplace_back(src, s);
    }
    std::sort(cand.begin(), cand.end());
    for (const auto& [src, s] : cand) {
      if (left.empty()) break;
      const Box image = boxes[static_cast<std::size_t>(src)].convert(type).shift(shifts[s]);
      for (const Box& r : intersect(left, image)) {
        copies_.push_back(CopyOp{src, static_cast<int>(b), r, shifts[s]});
        start_.push_back(start_.back() + r.num_pts());
      }
      left = subtract(left, BoxList{image});
    }
    // Outside the domain in a direction that is not periodic.
    for (int d = 0; d < dim; ++d) {
      if (periodic[d]) continue;
      if (grown.lo(d) < domain_t.lo(d)) boundary_.push_back({static_cast<int>(b), grown.grow(d, 0, domain_t.lo(d) - 1 - grown.hi(d)), d, 0});
      if (grown.hi(d) > domain_t.hi(d)) boundary_.push_back({static_cast<int>(b), grown.grow(d, grown.lo(d) - (domain_t.hi(d) + 1), 0), d, 1});
    }
  }
  d_copies_ = to_device(copies_, "amr::HaloPlan.copies");
  d_start_ = to_device(start_, "amr::HaloPlan.start");
}

}  // namespace soma::amr
