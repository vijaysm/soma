// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/copy.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include "bins.hpp"

namespace soma::amr {

namespace {

template <class U>
Kokkos::View<const U*, CopyPlan::DeviceSpace> to_device(const std::vector<U>& v, const std::string& label) {
  Kokkos::View<U*, CopyPlan::DeviceSpace> d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

// The cell box around a region of any index type, one cell wider.
Box around(const Box& r) {
  return Box(r.lo() - IntVect{{1, 1, 1}}, r.hi() + IntVect{{1, 1, 1}}, r.dim());
}

}  // namespace

CopyPlan::CopyPlan(const BoxArray& src, const std::vector<BoxList>& regions, IndexType type, Periodicity periodic,
                   int src_ghosts)
    : type_(type), src_ghosts_(src_ghosts) {
  if (src_ghosts < 0) throw std::invalid_argument("CopyPlan: source ghosts must not be negative");
  const Box& domain = src.domain();
  const int dim = domain.dim();
  // Periodic images: shifts by the domain length, each periodic direction
  // -1, 0 or +1, the unshifted one first.
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
  const detail::Bins bins(grow(src.boxes(), src_ghosts), domain);
  uncovered_.resize(regions.size());
  for (std::size_t b = 0; b < regions.size(); ++b) {
    BoxList left = regions[b];
    std::vector<std::pair<int, std::size_t>> cand;  // (source, image)
    for (std::size_t s = 0; s < shifts.size(); ++s) {
      const IntVect back = IntVect{} - shifts[s];
      for (const Box& r : regions[b]) {
        for (const int i : bins.near(around(r).shift(back))) cand.emplace_back(i, s);
      }
    }
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    // Valid points first, then (if asked) the ghost points.
    for (const int width : src_ghosts > 0 ? std::vector<int>{0, src_ghosts} : std::vector<int>{0}) {
      for (const auto& [i, s] : cand) {
        if (left.empty()) break;
        const Box from = src[static_cast<std::size_t>(i)].convert(type).grow(width).shift(shifts[s]);
        for (const Box& r : intersect(left, from)) {
          copies_.push_back(CopyOp{i, static_cast<int>(b), r, shifts[s]});
          start_.push_back(start_.back() + r.num_pts());
        }
        left = subtract(left, BoxList{from});
      }
    }
    uncovered_[b] = std::move(left);
  }
  d_copies_ = to_device(copies_, "amr::CopyPlan.copies");
  d_start_ = to_device(start_, "amr::CopyPlan.start");
}

}  // namespace soma::amr
