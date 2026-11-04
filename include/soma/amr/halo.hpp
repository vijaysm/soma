// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_HALO_HPP_
#define SOMA_AMR_HALO_HPP_

// Same-level halo exchange. A HaloPlan, computed once for a level's boxes, an
// index type and a ghost width, lists the copies that fill the ghost points
// of every box from the valid points of the others (faces, edges and
// corners), across periodic directions too, and the ghost regions outside
// the domain, which a physical-boundary callback fills. The copies of one
// box are disjoint (the first source along the curve wins where valid
// regions share points), so a fill gives the same result at any thread
// count. Ghost points that no box covers keep their values; the
// coarse-fine fill takes those. The plan is local; in Phase 4 the copies
// from other ranks' boxes become messages.

#include <cstdint>
#include <functional>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/amr/box.hpp"
#include "soma/amr/box_array.hpp"
#include "soma/amr/level_field.hpp"

namespace soma::amr {

struct Periodicity {
  bool periodic[3] = {false, false, false};
  KOKKOS_INLINE_FUNCTION constexpr bool operator[](int d) const { return periodic[d]; }
};

// Points `region` of box dst take the values of points region - shift of
// box src.
struct CopyOp {
  int src = 0;
  int dst = 0;
  Box region;
  IntVect shift;
};

// A ghost region of box `box` outside the domain, on the low (side 0) or
// high (side 1) end of direction dir.
struct BoundaryRegion {
  int box = 0;
  Box region;
  int dir = 0;
  int side = 0;
};

class HaloPlan {
 public:
  HaloPlan() = default;
  HaloPlan(const BoxArray& boxes, IndexType type, int ghosts, Periodicity periodic);

  const std::vector<CopyOp>& copies() const { return copies_; }
  const std::vector<BoundaryRegion>& boundary() const { return boundary_; }
  IndexType type() const { return type_; }
  int ghosts() const { return ghosts_; }
  std::int64_t num_copied() const { return start_.empty() ? 0 : start_.back(); }

  // Device copies of the plan.
  Kokkos::View<const CopyOp*, Kokkos::DefaultExecutionSpace::memory_space> device_copies() const { return d_copies_; }
  Kokkos::View<const std::int64_t*, Kokkos::DefaultExecutionSpace::memory_space> device_start() const { return d_start_; }

 private:
  IndexType type_{};
  int ghosts_ = 0;
  std::vector<CopyOp> copies_;
  std::vector<std::int64_t> start_;  // prefix of copied points per copy
  std::vector<BoundaryRegion> boundary_;
  Kokkos::View<const CopyOp*, Kokkos::DefaultExecutionSpace::memory_space> d_copies_;
  Kokkos::View<const std::int64_t*, Kokkos::DefaultExecutionSpace::memory_space> d_start_;
};

// Fills the ghost points of the field's boxes along the plan, then calls
// physical(box, region, dir, side) for each ghost region outside the domain
// (after the copies, so corners next to a periodic direction are set).
template <class T>
void fill_halo(const LevelField<T>& field, const HaloPlan& plan,
               const std::function<void(int box, const Box& region, int dir, int side)>& physical = {}) {
  using Exec = Kokkos::DefaultExecutionSpace;
  if (!(field.type() == plan.type()) || field.ghosts() < plan.ghosts()) {
    throw std::invalid_argument("fill_halo: the plan is for another index type or more ghosts");
  }
  const auto copies = plan.device_copies();
  const auto start = plan.device_start();
  const auto v = field.view();
  const int n = static_cast<int>(copies.extent(0));
  const int ncomp = field.ncomp();
  Kokkos::parallel_for(
      "amr::fill_halo", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, plan.num_copied()),
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
        const IntVect q = detail::point_of(op.region, p - start(lo));
        const auto dst = v.patch(op.dst);
        const auto src = v.patch(op.src);
        for (int c = 0; c < ncomp; ++c) {
          dst(q[0], q[1], q[2], c) = src(q[0] - op.shift[0], q[1] - op.shift[1], q[2] - op.shift[2], c);
        }
      });
  Kokkos::fence("amr::fill_halo");
  if (physical) {
    for (const BoundaryRegion& r : plan.boundary()) physical(r.box, r.region, r.dir, r.side);
  }
}

}  // namespace soma::amr

#endif  // SOMA_AMR_HALO_HPP_
