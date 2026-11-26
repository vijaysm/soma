// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_REFINE_GRADED_HPP_
#define SOMA_ALGORITHMS_REFINE_GRADED_HPP_

/// \file
/// Graded all-quad refinement on the unit sphere toward a target the caller
/// describes (a region, a coastline): passes of refine_quads, each level
/// banded by about `transition` cells of its own before the next coarser,
/// rerun with wider bands where a band came out thin.
///
/// ```
///   soma::GradedOptions o{.spacing = 12, .length = [](soma::Real s) { return s * 6.37122e3; }};  // km
///   const auto near = [&](const soma::HostMesh& m, const std::vector<soma::Real>& grow) {
///     std::vector<std::uint8_t> in(grow.size(), 0);
///     for (std::size_t e = 0; e < grow.size(); ++e) in[e] = grow[e] >= 0 && distance_km(m, e) <= grow[e];
///     return in;
///   };
///   soma::GradedResult r = soma::refine_graded(comm, mesh, o, near);   // collective
/// ```
///
/// Sizes are measured on the unit sphere and given in the caller's units by
/// `length` (identity when empty): a cell's size s is length(sqrt(area)),
/// the area that of its corners' polygon (twice the area vector about the
/// first corner, halved), and its GLL spacing s / (np - 1). A cell of level l
/// (its "refine:level") that needs k > 0 more trisections to reach a spacing
/// of at most 1.5 `spacing` asks the target whether it lies within
///   g = cells s (1/3 + ... + 1/3^(k-1)) + widen[l + 1] + widen[l + 2] + ...
/// of it, cells = `grow` (or `transition` when `grow` < 0), widen what the
/// reruns added (none in the first run); a cell that needs no trisection is
/// given g = -1 and must be answered 0. The cells answered nonzero are
/// refined (refine_quads, onto the unit sphere) and the ranks evened out
/// (rebalance); passes repeat until none is marked or after `passes`. Then
/// band_widths measures each level's band; for each level l under
/// `transition` cells wide, widen[l] grows by the missing cells times the
/// level's mean size (level_sizes), and the refinement reruns from `start`,
/// at most `reruns` times.
///
/// Every sum is in a fixed order and the marks depend on a cell's corners and
/// level only, so the result is the same, gid for gid and bit for bit, on any
/// number of ranks, given a target that answers alike for a cell wherever it
/// lies.
///
/// The mesh is a quad mesh on the unit sphere, without ghosts, as
/// refine_quads takes it; the result has none. Host only.

#include <cstdint>
#include <functional>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

struct GradedOptions {
  int np = 4;          ///< GLL points per element edge (>= 2)
  int passes = 8;      ///< the most passes of one run
  int transition = 3;  ///< cells of each level between the next finer and the next coarser (>= 1)
  int grow = -1;       ///< cells of growth per level before the first measure; < 0: transition
  int reruns = 4;      ///< the most reruns with wider bands
  Real spacing = 12;   ///< target GLL spacing, in the units of `length`
  Real max_cells = 0;  ///< > 0: throw std::runtime_error before a pass would exceed it
  std::function<Real(Real)> length;  ///< unit-sphere length -> caller's units; empty: identity
};

/// Per owned cell of the mesh given, whether it lies within grow[e] (the
/// caller's units) of the target: nonzero to refine it. grow[e] < 0: the cell
/// needs no refinement, answer 0. Called on every rank at once (it may be
/// collective) and must return one value per owned cell.
using GradedTarget = std::function<std::vector<std::uint8_t>(const HostMesh&, const std::vector<Real>& grow)>;

/// Called on every rank at once before marking again a level an earlier pass
/// made: `run` the rerun index (from 0), `pass` the passes made so far in
/// this run (>= 1). It may move vertices (a smoothing, say); the cells are
/// then marked from the moved corners. It must leave the mesh without ghosts.
using GradedHook = std::function<void(HostMesh&, int run, int pass)>;

/// The passes of the last run.
struct GradedPasses {
  int passes = 0;
  Real marked = 0;            ///< cells still marked when the passes stopped, over every rank
  Real refined = 0;           ///< cells marked over every pass
  std::vector<Real> cells;    ///< cells after each pass, over every rank
  std::vector<Real> seconds;  ///< each pass's time (refine_quads and rebalance)
};

struct GradedResult {
  HostMesh mesh;
  GradedPasses passes;
  std::vector<Real> widths;  ///< band_widths of the last run
  int reruns = 0;
  bool settled = true;  ///< false: a band was still under `transition` cells after the last rerun
};

/// Collective: `start` refined toward `target` as described above.
/// Throws std::invalid_argument for np < 2, transition < 1, passes < 0,
/// reruns < 0 or a spacing not positive and finite (on every rank, as every
/// rank holds the same options), or a target that does not answer one value
/// per owned cell;
/// std::runtime_error, naming the pass, before a pass that would make more
/// than max_cells cells (at least 8 more per marked cell); what refine_quads
/// throws.
[[nodiscard]] GradedResult refine_graded(const Comm& comm, const HostMesh& start, const GradedOptions& options,
                                         const GradedTarget& target, const GradedHook& before_marking = {});

/// Each owned cell's "refine:level" (all 0 without the field). Not collective.
[[nodiscard]] std::vector<std::int32_t> refine_levels(const HostMesh& mesh);

/// Collective: per level l from 1 to the finest but one, the fewest level-l
/// cells crossed (through edges) from a level-l cell next to a finer one to a
/// cell coarser than l, counted up to `cap`; -1 where no level-l cell meets a
/// finer one or no cell is coarser. Adds and removes a ghost layer (on more
/// than one rank), so `mesh` must come without one.
[[nodiscard]] std::vector<Real> band_widths(const Comm& comm, HostMesh& mesh, int cap);

/// Collective: the mean size (length(sqrt(area)), the caller's units; the
/// unit-sphere size when `length` is empty) of the owned cells of each level
/// 0 .. count - 1 over every rank, 0 for a level with none.
[[nodiscard]] std::vector<Real> level_sizes(const Comm& comm, const HostMesh& mesh, std::size_t count,
                                            const std::function<Real(Real)>& length = {});

}  // namespace soma

#endif  // SOMA_ALGORITHMS_REFINE_GRADED_HPP_
