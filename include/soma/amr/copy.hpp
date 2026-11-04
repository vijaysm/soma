// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_COPY_HPP_
#define SOMA_AMR_COPY_HPP_

// Copies between two sets of boxes of one index type. A CopyPlan fills given
// regions of the destination boxes from a source level: from its valid
// points first (the first source along the curve wins where they share
// points), then, if asked, from the ghost points of its boxes (which the
// caller has filled), through periodic images too. What no source covers is
// left as it is and listed as uncovered. The plan is computed once and
// replayed by copy(); its copies of one destination box are disjoint, so the
// result does not depend on the thread count.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/amr/box.hpp"
#include "soma/amr/box_array.hpp"
#include "soma/amr/box_list.hpp"
#include "soma/amr/level_field.hpp"

namespace soma::amr {

struct Periodicity {
  bool periodic[3] = {false, false, false};
  KOKKOS_INLINE_FUNCTION constexpr bool operator[](int d) const { return periodic[d]; }
};

// Points `region` of destination box dst take the values of points
// region - shift of source box src.
struct CopyOp {
  int src = 0;
  int dst = 0;
  Box region;
  IntVect shift;
};

class CopyPlan {
 public:
  using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

  CopyPlan() = default;
  // regions[b]: the points of destination box b to fill, in `type`. With
  // src_ghosts > 0, source ghost points up to that width serve after the
  // valid ones.
  CopyPlan(const BoxArray& src, const std::vector<BoxList>& regions, IndexType type, Periodicity periodic,
           int src_ghosts = 0);

  IndexType type() const { return type_; }
  int src_ghosts() const { return src_ghosts_; }
  const std::vector<CopyOp>& copies() const { return copies_; }
  // Per destination box, the points of its regions no source covered.
  const std::vector<BoxList>& uncovered() const { return uncovered_; }
  std::int64_t num_copied() const { return start_.back(); }
  Kokkos::View<const CopyOp*, DeviceSpace> device_copies() const { return d_copies_; }
  Kokkos::View<const std::int64_t*, DeviceSpace> device_start() const { return d_start_; }

 private:
  IndexType type_{};
  int src_ghosts_ = 0;
  std::vector<CopyOp> copies_;
  std::vector<std::int64_t> start_{0};
  std::vector<BoxList> uncovered_;
  Kokkos::View<const CopyOp*, DeviceSpace> d_copies_;
  Kokkos::View<const std::int64_t*, DeviceSpace> d_start_;
};

namespace detail {

template <class T, class MemSpace>
void run_copies(const LevelView<T, MemSpace>& src, const LevelView<T, MemSpace>& dst, const CopyPlan& plan) {
  using Exec = typename MemSpace::execution_space;
  const auto copies = plan.device_copies();
  const auto start = plan.device_start();
  const int n = static_cast<int>(copies.extent(0));
  const int ncomp = dst.ncomp < src.ncomp ? dst.ncomp : src.ncomp;
  Kokkos::parallel_for(
      "amr::copy", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, plan.num_copied()),
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
        const CopyOp& op = copies(lo);
        const IntVect q = point_of(op.region, p - start(lo));
        const auto to = dst.patch(op.dst);
        const auto from = src.patch(op.src);
        for (int c = 0; c < ncomp; ++c) {
          to(q[0], q[1], q[2], c) = from(q[0] - op.shift[0], q[1] - op.shift[1], q[2] - op.shift[2], c);
        }
      });
  Kokkos::fence("amr::copy");
}

}  // namespace detail

// Copies along the plan from src into dst (components up to the smaller
// count).
template <class T>
void copy(const LevelField<T>& src, const LevelField<T>& dst, const CopyPlan& plan) {
  if (!(src.type() == plan.type()) || !(dst.type() == plan.type()) || src.ghosts() < plan.src_ghosts()) {
    throw std::invalid_argument("amr::copy: the fields do not match the plan's index type or source ghosts");
  }
  detail::run_copies(src.view(), dst.view(), plan);
}

}  // namespace soma::amr

#endif  // SOMA_AMR_COPY_HPP_
