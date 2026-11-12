// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_REMAP_HPP_
#define SOMA_ALGORITHMS_REMAP_HPP_

// First-order conservative remap between two plane meshes or two volume
// meshes: a cell value on a target mesh from the cell values of a source
// mesh,
//
//   u_T = sum_S w_TS u_S,   w_TS = |T n S| / |T|,
//
// where |T n S| is the area (in 3D the volume) the cells share. Where the source covers a
// target cell its weights sum to one (constants are kept); where each mesh
// covers the other, sum_T |T| u_T = sum_S |S| u_S (mass is conserved).
//
//   const soma::RemapOperator<Mem> remap(comm, source, target, {.merge_tolerance = 1e-12});
//   remap.apply(comm, u_source, u_target);  // owned source cells -> owned target cells
//
// Plane meshes (dimension 2): each cell's corners in one plane z = const
// (each cell is taken in the xy plane); cells are triangles, quads or
// polygons of at most 64 corners. A cell is turned counter-clockwise and
// started at its least-gid corner, so every rank sees the same polygon; a
// non-convex cell is split into triangles by ear clipping (polygon.hpp) and
// its overlaps are summed over them. A cell thinner than the tolerance
// (twice its area over its longest side), crossing itself, or off a plane
// throws on every rank.
//
// Volume meshes (dimension 3): tets, hexes, prisms and pyramids, each split
// into tets: a tet is itself; another cell has one tet per triangle of its
// faces, apex its centroid, a quad face cut along the diagonal from its
// least-gid corner, so that cells sharing a face cut it alike and their tets
// fill space without gaps (non-planar and non-convex faces included). Tets
// are clipped by one another's face planes (tet_clip.hpp). A cell with a tet
// thinner than the tolerance, or not star-shaped about its centroid (its
// tets turned unlike), throws on every rank.
//
// The merge tolerance (a length, the same on every rank) decides when a
// corner is on a clipping line or plane (one outside by at most it goes onto
// it), merges clipped corners closer than it, and drops overlaps thinner
// than it. Merging moves areas by up to about the tolerance times a cell's
// perimeter, so weights sum to one to that: keep it far below the cells'
// size (the default 1e-12 suits cells of any size above about 1e-6).
//
// The meshes may be partitioned differently and need no ghosts. Each rank
// shares the bounding box of its owned target cells (an allgather of P
// boxes) and sends each owned source cell to every rank whose box it meets
// (transport::route); there a GridIndex over the cells that arrived finds
// the candidates for each owned target cell, whose overlaps are clipped. A
// row's entries are ordered by source gid, so the operator, and apply(), are
// the same bit for bit on any partition and thread count. apply() replays
// the routing for the source values: one exchange with the ranks found.
//
// A cell's pieces are made by its owner and sent as they are. The overlaps
// are clipped by Kokkos on the host; the operator's arrays
// then live in MemSpace, where apply() runs (values cross ranks through host
// memory).

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/to_space.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/transport/comm_plan.hpp"

namespace soma {

struct RemapOptions {
  // Corners closer than this merge; pieces of area below its square are
  // none. A length in the meshes' units; the same on every rank.
  Real merge_tolerance = 1e-12;
};

namespace detail {

// The operator as fv_plan builds it, on the host: rows the owned target
// cells, columns the source cells that arrived here.
struct RemapPlan {
  LO bad_target = 0, bad_source = 0;  // cells refused
  LO overflow = 0;                    // overlaps too big to clip (never, by the capacities)
  bool supported = true;
  std::vector<Offset> offsets;    // per owned target cell
  std::vector<LO> columns;        // arrived source cell
  std::vector<Real> overlap;      // |T n S|
  std::vector<Real> target_area;  // |T|
  std::vector<GO> source_gid;     // per arrived source cell
  std::vector<LO> sent_cell;      // per sent record: the owned source cell
  transport::CommPlan plan;
};

// Collective (route, allgather); the flags are not agreed here.
RemapPlan remap_plan(const Comm& comm, const Mesh<Kokkos::HostSpace>& source, const Mesh<Kokkos::HostSpace>& target,
                     Real tolerance);

}  // namespace detail

template <class MemSpace>
class RemapOperator {
 public:
  using Exec = typename MemSpace::execution_space;

  // Collective. Throws std::invalid_argument on every rank if any rank's
  // meshes are not both of dimension 2 or both of 3, have edits or
  // high-order nodes, or hold a cell other than a triangle, quad or polygon
  // (of at most 64 corners) in 2D, a tet, hex, prism or pyramid in 3D; if a
  // cell is bad as above; or if the tolerance is not finite and positive or
  // differs between ranks.
  RemapOperator(const Comm& comm, const Mesh<MemSpace>& source, const Mesh<MemSpace>& target,
                const RemapOptions& options = {}) {
    // Agreed through a flag, as a NaN may drop out of a max on some ranks.
    const Real tol = options.merge_tolerance;
    const bool good = tol > 0 && Kokkos::isfinite(tol);
    Real agree[3] = {good ? Real{0} : Real{1}, good ? tol : 0, good ? -tol : 0};
    comm.allreduce(std::span<Real>(agree), Op::max);
    if (agree[0] != 0 || agree[1] != -agree[2]) {
      throw std::invalid_argument("RemapOperator: the merge tolerance is not finite and positive, or differs between "
                                  "ranks");
    }
    detail::RemapPlan p;
    {
      std::optional<Mesh<Kokkos::HostSpace>> ms, mt;
      const Mesh<Kokkos::HostSpace>* hs = nullptr;
      const Mesh<Kokkos::HostSpace>* ht = nullptr;
      if constexpr (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
        hs = &source;
        ht = &target;
      } else {
        ms.emplace(source.template mirror<Kokkos::HostSpace>());
        mt.emplace(target.template mirror<Kokkos::HostSpace>());
        hs = &*ms;
        ht = &*mt;
      }
      const bool usable = !source.edited() && !target.edited();
      Real flags[4] = {usable ? Real{0} : Real{1}, 0, 0, 0};
      comm.allreduce(std::span<Real>(flags, 1), Op::max);
      if (flags[0] != 0) throw std::invalid_argument("RemapOperator: a rank's source or target mesh has edits");
      p = detail::remap_plan(comm, *hs, *ht, tol);
      flags[0] = p.supported ? 0 : 1;
      flags[1] = static_cast<Real>(p.bad_source);
      flags[2] = static_cast<Real>(p.bad_target);
      flags[3] = static_cast<Real>(p.overflow);
      comm.allreduce(std::span<Real>(flags), Op::max);
      if (flags[3] != 0) throw std::invalid_argument("RemapOperator: an overlap overflowed the clipping capacity");
      if (flags[0] != 0) {
        throw std::invalid_argument("RemapOperator: a rank's meshes are not both of dimension 2 or both of 3, have "
                                    "high-order nodes, or hold a cell other than a triangle, quad or polygon of at "
                                    "most 64 corners (2D) or a tet, hex, prism or pyramid (3D)");
      }
      if (flags[1] != 0 || flags[2] != 0) {
        throw std::invalid_argument(std::string("RemapOperator: a rank's ") + (flags[1] != 0 ? "source" : "target") +
                                    " mesh holds a cell thinner than the tolerance, crossing itself, off a plane "
                                    "z = const (2D), or not star-shaped about its centroid (3D)");
      }
    }
    rows_ = static_cast<LO>(p.target_area.size());
    cols_ = static_cast<LO>(p.source_gid.size());
    std::vector<Real> weights(p.overlap.size());
    for (LO t = 0; t < rows_; ++t) {
      const auto ti = static_cast<std::size_t>(t);
      for (Offset k = p.offsets[ti]; k < p.offsets[ti + 1]; ++k) {
        const auto ki = static_cast<std::size_t>(k);
        weights[ki] = p.overlap[ki] / p.target_area[ti];
      }
    }
    offsets_ = detail::to_space<Offset, MemSpace>(p.offsets, "soma::RemapOperator::offsets");
    columns_ = detail::to_space<LO, MemSpace>(p.columns, "soma::RemapOperator::columns");
    weights_ = detail::to_space<Real, MemSpace>(weights, "soma::RemapOperator::weights");
    overlap_ = detail::to_space<Real, MemSpace>(p.overlap, "soma::RemapOperator::overlap");
    target_area_ = detail::to_space<Real, MemSpace>(p.target_area, "soma::RemapOperator::target_area");
    source_gid_ = detail::to_space<GO, MemSpace>(p.source_gid, "soma::RemapOperator::source_gid");
    sent_cell_ = detail::to_space<LO, MemSpace>(p.sent_cell, "soma::RemapOperator::sent_cell");
    plan_ = std::move(p.plan);
    source_rows_ = source.segment_table(source.dim()).owned_end;
  }

  // Owned target cells (the rows), and the source cells that arrived here
  // (the columns).
  [[nodiscard]] LO rows() const { return rows_; }
  [[nodiscard]] LO cols() const { return cols_; }
  // CSR: a row's entries ordered by source gid.
  [[nodiscard]] const View1<Offset, MemSpace>& offsets() const { return offsets_; }
  [[nodiscard]] const View1<LO, MemSpace>& columns() const { return columns_; }
  [[nodiscard]] const View1<Real, MemSpace>& weights() const { return weights_; }
  // |T n S| per entry, |T| per row, and each column's source gid.
  [[nodiscard]] const View1<Real, MemSpace>& overlap() const { return overlap_; }
  [[nodiscard]] const View1<Real, MemSpace>& target_area() const { return target_area_; }
  [[nodiscard]] const View1<GO, MemSpace>& source_gid() const { return source_gid_; }

  // Collective: u_target (an entry per owned target cell, at least) from
  // u_source (one per owned source cell, at least). One exchange. A rank
  // whose vectors are too short throws std::invalid_argument and leaves the
  // others in the exchange.
  void apply(const Comm& comm, const View1<const Real, MemSpace>& u_source, const View1<Real, MemSpace>& u_target) const {
    const bool fits = u_source.extent(0) >= static_cast<std::size_t>(source_rows_) &&
                      u_target.extent(0) >= static_cast<std::size_t>(rows_);
    const auto sent_cell = sent_cell_;
    const std::size_t ns = sent_cell.extent(0);
    View1<Real, MemSpace> sent("soma::RemapOperator::sent", ns);
    if (fits) {
      Kokkos::parallel_for(
          "soma::RemapOperator::gather", Kokkos::RangePolicy<Exec>(0, ns),
          KOKKOS_LAMBDA(const std::size_t i) { sent(i) = u_source(sent_cell(i)); });
      Exec{}.fence();
    }
    // Through host memory.
    const auto hsent = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sent);
    Kokkos::View<Real*, Kokkos::LayoutLeft, Kokkos::HostSpace> harrived("soma::RemapOperator::arrived",
                                                                        static_cast<std::size_t>(cols_));
    plan_.forward(comm, std::span<const Real>(hsent.data(), hsent.extent(0)),
                  std::span<Real>(harrived.data(), harrived.extent(0)));
    if (!fits) throw std::invalid_argument("RemapOperator::apply: a vector shorter than the owned cells");
    const auto arrived = Kokkos::create_mirror_view_and_copy(MemSpace{}, harrived);
    const auto offsets = offsets_;
    const auto columns = columns_;
    const auto weights = weights_;
    Kokkos::parallel_for(
        "soma::RemapOperator::apply", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, rows_),
        KOKKOS_LAMBDA(const LO t) {
          Real sum = 0;
          for (Offset k = offsets(t); k < offsets(t + 1); ++k) sum += weights(k) * arrived(columns(k));
          u_target(t) = sum;
        });
    Exec{}.fence();
  }

 private:
  LO rows_ = 0, cols_ = 0, source_rows_ = 0;
  View1<Offset, MemSpace> offsets_;
  View1<LO, MemSpace> columns_;
  View1<Real, MemSpace> weights_, overlap_, target_area_;
  View1<GO, MemSpace> source_gid_;
  View1<LO, MemSpace> sent_cell_;
  transport::CommPlan plan_;
};

}  // namespace soma

#endif  // SOMA_ALGORITHMS_REMAP_HPP_
