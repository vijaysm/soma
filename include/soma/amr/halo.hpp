// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_HALO_HPP_
#define SOMA_AMR_HALO_HPP_

/// \file
/// Same-level halo exchange. A HaloPlan, computed once for a level's boxes, an
/// index type and a ghost width, lists the copies that fill the ghost points
/// of every box from the valid points of the others (faces, edges and
/// corners), across periodic directions too, and the ghost regions outside
/// the domain, which a physical-boundary callback fills. The copies of one
/// box are disjoint (the first source along the curve wins where valid
/// regions share points), so a fill gives the same result at any thread
/// count. Ghost points that no box covers keep their values; the
/// coarse-fine fill takes those. Over many ranks the copies from other
/// ranks' boxes are messages (see CopyPlan), and a fill can be split:
///
/// ```
///   auto pending = amr::begin_fill_halo(u, plan);   // interior work here
///   pending.end();                                  // then the boundary
/// ```


#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/amr/box.hpp"
#include "soma/amr/box_array.hpp"
#include "soma/amr/box_list.hpp"
#include "soma/amr/copy.hpp"
#include "soma/amr/level_field.hpp"

namespace soma::amr {

/// A ghost region of box `box` (the field's local index) outside the domain,
/// on the low (side 0) or high (side 1) end of direction dir.
struct BoundaryRegion {
  int box = 0;
  Box region;
  int dir = 0;
  int side = 0;
};

class HaloPlan {
 public:
  HaloPlan() = default;
  static constexpr int default_tag = 32621;

  HaloPlan(const BoxArray& boxes, IndexType type, int ghosts, Periodicity periodic);
  /// The part of rank comm.rank(), box i owned by distribution.owner[i].
  HaloPlan(const Comm& comm, const BoxArray& boxes, const Distribution& distribution, IndexType type, int ghosts,
           Periodicity periodic, int tag = default_tag);

  const CopyPlan& copy_plan() const { return copy_; }
  const std::vector<CopyOp>& copies() const { return copy_.copies(); }
  /// The ghost regions of this rank's boxes outside the domain.
  const std::vector<BoundaryRegion>& boundary() const { return boundary_; }
  /// Per box of the level, its ghost points inside the domain (or a periodic image of it)
  /// that no box of the level covers: the coarse-fine fill's part.
  const std::vector<BoxList>& uncovered() const { return uncovered_; }
  IndexType type() const { return copy_.type(); }
  int ghosts() const { return ghosts_; }
  std::int64_t num_copied() const { return copy_.num_copied(); }

 private:
  void build(const std::optional<Comm>& comm, const BoxArray& boxes, const Distribution& distribution, IndexType type,
             Periodicity periodic, int tag);

  int ghosts_ = 0;
  CopyPlan copy_;
  std::vector<BoundaryRegion> boundary_;
  std::vector<BoxList> uncovered_;
};

/// A halo fill in flight; end() finishes the copies (a second call does
/// nothing).
template <class T>
class PendingHalo {
 public:
  void end() { copies_.end(); }

 private:
  template <class U>
  friend PendingHalo<U> begin_fill_halo(const LevelField<U>& field, const HaloPlan& plan);
  PendingCopy<T> copies_;
};

/// Starts filling the ghost points of the field's boxes along the plan.
template <class T>
PendingHalo<T> begin_fill_halo(const LevelField<T>& field, const HaloPlan& plan) {
  if (!(field.type() == plan.type()) || field.ghosts() < plan.ghosts()) {
    throw std::invalid_argument("fill_halo: the plan is for another index type or more ghosts");
  }
  PendingHalo<T> out;
  out.copies_ = begin_copy(field, field, plan.copy_plan());
  return out;
}

/// Fills the ghost points of the field's boxes along the plan, then calls
/// physical(box, region, dir, side) for each ghost region outside the domain
/// (after the copies, so corners next to a periodic direction are set).
template <class T>
void fill_halo(const LevelField<T>& field, const HaloPlan& plan,
               const std::function<void(int box, const Box& region, int dir, int side)>& physical = {}) {
  begin_fill_halo(field, plan).end();
  if (physical) {
    for (const BoundaryRegion& r : plan.boundary()) physical(r.box, r.region, r.dir, r.side);
  }
}

}  // namespace soma::amr

#endif  // SOMA_AMR_HALO_HPP_
