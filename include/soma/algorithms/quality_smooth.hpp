// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_QUALITY_SMOOTH_HPP_
#define SOMA_ALGORITHMS_QUALITY_SMOOTH_HPP_

/// \file
/// Optimization-based smoothing of a surface mesh of triangles and quads.
///
/// ```
///   soma::QualitySmoothOptions o;
///   o.sphere_radius = 6.37122e6;           // stay on the sphere
///   auto r = soma::quality_smooth(comm, mesh, o);
/// ```
///
/// Each free vertex moves, in its tangent plane, to improve the cells around
/// it (its patch), in three phases of sweeps:
///
/// - shape: lower the sum over the patch's corners of the corner's condition
///   number, kappa = |A|_F^2 / (2 det A), A the corner's Jacobian (its two
///   edges, projected on the plane) against the ideal corner: a square's for
///   quads, an equilateral triangle's for triangles (Knupp). kappa is 1 for
///   the ideal corner, does not depend on size, and grows without bound as the
///   corner degenerates;
/// - angle: for the vertices of cells whose least corner angle is below the
///   threshold, raise the least angle of the corners the vertex moves
///   (Freitag and Plassmann: the steepest ascent over the active angles),
///   no corner of the patch left worse in condition number than its worst:
///   angles alone bound neither a quad's aspect nor its largest angle;
/// - polish: shape sweeps again, kept above the least angle the angle phase
///   reached. Raising the worst corners costs some good ones; these give
///   most of it back without losing the gain.
///
/// A step is a line search along the direction found from central
/// differences in the plane (shape: Newton's, where the 2 x 2 Hessian is
/// positive definite, else steepest descent, tried first at 1.7 times its
/// length: over-relaxed, as improvement spreads one cell a sweep); it is kept
/// only if the objective improves, no
/// corner of the patch has det A <= 0, and the patch's least angle stays at or
/// above the mesh's least angle when the phase began, so the least angle
/// never falls. Angles are those of the corners in space, as cells are drawn.
///
/// The tangent plane: given a sphere radius, normal to the vertex's position
/// (the vertex is projected back on the sphere after each step); else normal
/// to the sum of the patch's area vectors. Fixed vertices stay: those of the
/// mask, else the skin (none on a closed surface).
///
/// With an area ratio bound, every phase's steps also keep the area ratios
/// of the cells sharing an edge with the patch's cells at or below the larger
/// of the bound and the patch's largest such ratio before the vertex moved:
/// the condition number does not depend on size, so without it the phases
/// trade size for shape where cells of two levels meet.
///
/// Sweeps visit the vertex colours of detail::vertex_colours in turn (no two
/// vertices of a colour share a cell), owners moving their vertices and a
/// halo update after each colour; patches sum their cells in cell-gid order.
/// The result is the same bit for bit on any number of ranks and threads.

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "soma/algorithms/laplacian.hpp"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

namespace detail {

/// Per vertex lid (ghosts included), a colour from 0: no two corners of one
/// cell share a colour. Jones and Plassmann's rounds: an uncoloured vertex
/// whose priority, a hash of its gid, beats that of every uncoloured vertex
/// sharing a cell with it takes the least colour none of those has. Colours
/// depend on gids and connectivity only, so they are the same on any number
/// of ranks. `cells` is vertex_cells(mesh). Collective; on more than one rank
/// the mesh needs a ghost layer through vertices. Throws std::logic_error on
/// every rank if a round colours no vertex (ranks disagreeing on the mesh).
std::vector<std::int32_t> vertex_colours(const Comm& comm, const HostMesh& mesh, const VertexCells& cells);

/// Per cell lid, the cell across each edge (edge i joins corners i and i + 1):
/// num_vertices entries from offsets[c], -1 where no local cell is there.
/// `cells` is vertex_cells(mesh).
struct CellNeighbours {
  std::vector<Offset> offsets;  ///< cell count + 1
  std::vector<LO> cells;
};
CellNeighbours edge_neighbours(const HostMesh& mesh, const VertexCells& cells);

/// Per owned vertex, its cells and the cells sharing an edge with them (in
/// lid order, each once); ghost vertices get none. Given to vertex_colours,
/// no two vertices of a colour lie in cells sharing an edge. Needs two ghost
/// layers through vertices for the owned vertices' lists to be whole.
VertexCells edge_stencil(const HostMesh& mesh, const VertexCells& cells, const CellNeighbours& neighbours);

}  // namespace detail

struct QualitySmoothOptions {
  int shape_sweeps = 50;   ///< condition-number sweeps
  int angle_sweeps = 5;    ///< least-angle sweeps
  int polish_sweeps = 10;  ///< condition-number sweeps after them
  /// Degrees in (0, 90]: the angle phase moves the vertices of cells whose
  /// least angle is below it. Higher raises the least angle further and
  /// leaves more cells just under 45 degrees.
  Real angle_threshold = 45;
  /// > 0: a phase ends once no vertex moved farther than this times its
  /// patch's shortest edge in a sweep.
  Real tolerance = 0;
  /// >= 1: in every phase a step is kept only if no two cells sharing an
  /// edge, one of them in the vertex's patch, are left with an area ratio
  /// (larger over smaller) above max(area_ratio_bound, the largest such ratio
  /// before the vertex moved). The mesh's largest ratio then ends at most
  /// max(area_ratio_bound, its largest at the start). A cell's area is half
  /// the length of its area vector, from its corners in space. On more than
  /// one rank the mesh needs two ghost layers through vertices; vertices are
  /// coloured so that no two of a colour lie in cells sharing an edge. 0: no
  /// bound.
  Real area_ratio_bound = 0;
  /// Per vertex lid, nonzero for a vertex that stays; empty: the skin stays.
  std::span<const std::uint8_t> fixed;
  /// Move on the sphere of this radius about the origin.
  std::optional<Real> sphere_radius;
};

struct QualitySmoothReport {
  int shape_sweeps = 0;          ///< sweeps made
  int angle_sweeps = 0;          ///< sweeps made
  int polish_sweeps = 0;         ///< sweeps made
  Real least_angle_before = 0;   ///< degrees, over the cells of every rank
  Real least_angle_after = 0;    ///< degrees, over the cells of every rank
  Real area_ratio_before = 0;    ///< largest over the cells of every rank; 0 without a bound
  Real area_ratio_after = 0;     ///< largest over the cells of every rank; 0 without a bound
};

/// Collective. Throws std::invalid_argument on every rank if any rank passes
/// a mesh not of dimension 2, with a cell neither a triangle nor a quad, with
/// a corner whose det A <= 0 (in the tangent plane), with unremoved edits,
/// or, on more than one rank, without a ghost layer through vertices (two
/// with an area ratio bound); negative sweeps, a threshold outside (0, 90], a
/// tolerance negative or not finite, an area ratio bound negative, in (0, 1)
/// or not finite, a mask not one per vertex, a radius not positive and
/// finite; or if ranks differ in any option (or in giving a radius).
QualitySmoothReport quality_smooth(const Comm& comm, HostMesh& mesh, const QualitySmoothOptions& options = {});

}  // namespace soma

#endif  // SOMA_ALGORITHMS_QUALITY_SMOOTH_HPP_
