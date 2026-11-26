// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_REFINE_HPP_
#define SOMA_ALGORITHMS_REFINE_HPP_

/// \file
/// Uniform refinement: every edge of a mesh split in two, each cell into
/// children that tile it, on any number of ranks:
///
/// ```
///   const soma::RefineResult fine = soma::refine_uniform(comm, mesh, {.levels = 2});
///   fine.mesh;       // the refined local part, children on their parent's rank
///   fine.parent[c];  // the cell of `mesh` that fine cell c lies in
/// ```
///
/// Cells and their children (a cell's nodes: its corners, its edges'
/// midpoints, its quad faces' centres, its centre where it has one):
///
/// ```
///   tri      4 tris      the corners' halves and the middle one
///   quad     4 quads     with the centre
///   polygon  n quads     one per corner, with the centroid (corners averaged)
///   tet      8 tets      the corners' halves and four about the shortest
///                        diagonal of the octahedron left (the first in edge
///                        order of those as short)
///   hex      8 hexes     with the faces' centres and the centre
///   prism    8 prisms    with the quad faces' centres
///   pyramid  6 pyramids  with the base's centre: the corners' halves, one
///            + 4 tets    upside down under the apex's, and four tets
/// ```
///
/// A corner's child is the cell halved about it, so children are turned as
/// their parent is; the others are turned so in the parent's reference
/// coordinates. Every quad face gets a centre and no triangle does, so cells
/// of every type meeting at a face split it alike, and the refined mesh is
/// conforming where the mesh is. A new vertex is the average of its corners
/// (an edge's two, a face's four, a cell's all), summed in gid order, so ranks
/// holding it agree bit for bit.
///
/// Gids do not depend on the number of ranks: with V, E and F the largest
/// vertex, edge and face gid on any rank (edges and faces derived as
/// build_distributed_adjacencies derives them), an edge's midpoint is V + its
/// edge gid, a quad face's centre V + E + its face gid (3D); with the cells
/// numbered from 1 in gid order (n = gid - G + 1, G the least cell gid), a
/// cell's centre is V + E + F + n (F = 0 in 2D) and its child c is
/// K (n - 1) + c + 1, K the most children of any cell. The refined mesh is
/// therefore the same, gid for gid and bit for bit, on any number of ranks.
///
/// Each rank refines the cells it owns; the result has no ghosts (add them
/// again as needed). Vertex and cell fields carry over where they are dense
/// (one value per entity of a full domain, per-entity layout, not variable,
/// not handles), with their default and global values and their rule
/// (soma/fields/field_rule.hpp). Vertices and cells the input has (by their
/// vertices' gids) keep their values. The new ones take the field's rule,
/// called once with all of them, when it has one (hanging vertices too);
/// otherwise a child takes its parent's value (so integrals of cell averages
/// are kept), and a new vertex of a floating-point field the average of its
/// corners (exact for linear, bilinear and trilinear fields; for a hanging
/// vertex, the coarse side's interpolation), of an integer one the field's
/// default. Vertex values are read where a rank holds them, shared copies and
/// ghosts included, so these must hold their owners' values (update them
/// first) for ranks to agree. Fields that have no way to give new vertices
/// values (integer vertex fields with neither a rule nor a default), other
/// fields, and edge and face fields are dropped and named in `dropped`; the
/// internal soma: fields are rebuilt. Rules are each rank's own: set the same
/// rules on every rank, or ranks may keep different fields.
///
/// Each level is refine_marked (below) with every cell marked, so it keeps the
/// refinement state: "refine:level" one up, "refine:birth" recorded for each
/// new vertex (so refine_marked can follow), vertices made earlier on the
/// cells' edges and faces reused, a mesh that has hanging vertices keeping
/// them (listed in RefineResult::hanging), and transition cells of a
/// conforming result put back first. On a conforming mesh without that
/// state, the result is as described above.
///
/// A child's corners are its parent's mapped from reference coordinates, so a
/// cell not star-shaped about its corners' average (a non-convex quad or
/// polygon; in 3D, a strongly distorted cell) would have children turned
/// against it: plane cells are checked and refused (refine_marked: those it
/// splits), volume cells are not.
///
/// refine_quads (below) refines a quad mesh where marked and keeps it all-quad
/// and conforming, edges split in three: for spectral elements, in the plane
/// or on a sphere.
///
/// The refinement runs on the host (a HostMesh in, built with
/// build_local_part); a device mesh is mirrored to the host and back.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/point_match.hpp"

namespace soma {

struct RefineOptions {
  int levels = 1;  ///< passes; each splits every edge in two
};

struct RefineResult {
  HostMesh mesh;
  /// For each cell of `mesh`, the owned cell of the input it lies in (a local
  /// id of the input; refine_marked: for a cell in a group of transition
  /// cells the input put back, the group's least cell by gid).
  std::vector<LO> parent;
  /// The children of the input's owned cells: cell p's are
  /// children[child_offsets[p], child_offsets[p + 1]), local ids of `mesh`.
  std::vector<Offset> child_offsets;
  std::vector<LO> children;
  /// Fields not carried over, by name.
  std::vector<std::string> dropped;
  /// Marked refinement: the hanging vertices of `mesh` (local ids), each
  /// the midpoint of an edge or the centre of a quad face that a cell still
  /// has whole: its value is sum_k weights[k] value(masters[k]) over
  /// k in [hanging_offsets[i], hanging_offsets[i + 1]), masters by gid (they
  /// may be hanging themselves, one level up).
  std::vector<LO> hanging;
  std::vector<Offset> hanging_offsets{0};
  std::vector<GO> hanging_masters;
  std::vector<Real> hanging_weights;
};

/// How refine_marked finds the vertices earlier refinement made on the edges
/// and faces of the cells it refines (see refine_marked).
enum class VertexMatching : std::uint8_t {
  recorded,     ///< by the corners recorded in "refine:birth"
  coordinates,  ///< also by coordinates, for vertices with no record (infer_births)
};

struct MarkedOptions {
  /// Conforming refinement: no hanging vertices (see refine_marked).
  bool conforming = false;
  VertexMatching vertices = VertexMatching::recorded;
  PointTolerance tol{};  ///< VertexMatching::coordinates: exact by default
};

/// Collective. Throws std::invalid_argument on every rank if any rank's mesh
/// is not of dimension 2 (tris, quads, polygons) or 3 (tets, hexes, prisms,
/// pyramids), has edits or high-order nodes, or a quad or polygon not
/// star-shaped about its corners' average; if levels < 1 or the ranks differ
/// in levels or dimension; or if a gid of the refined mesh would not fit a GO.
[[nodiscard]] RefineResult refine_uniform(const Comm& comm, const HostMesh& mesh, RefineOptions options = {});

/// Marked refinement: the owned cells with a non-zero mark (one per owned
/// cell, in local id order), and the cells the 2:1 balance needs, split once
/// as refine_uniform splits them; the others stay whole, so their edges and
/// faces can carry hanging vertices (non-conforming). Levels are kept in the
/// cell field "refine:level" (0 where it is missing) and each vertex the
/// refinement makes at an edge's midpoint or a quad face's centre records the
/// corners it lies between in the vertex field "refine:birth" (count, then
/// up to four gids), so that later calls and other ranks reuse it: through a
/// directory keyed by those corners, whatever the partition.
///
/// Balance: a cell is split when a cell touching it at a corner, or at a
/// vertex born on its edges or faces (or on theirs, one level down), would be
/// two levels finer than it.
///
/// options.conforming leaves no hanging vertex. A mesh of triangles alone, or
/// of tets alone, is bisected instead: each cell marked is cut in two at the
/// midpoint of its longest edge (the one of least corner gids among equals),
/// and any cell with a vertex on one of its edges is cut so again, until none
/// is left (longest-edge, Rivara, bisection; each cut counts as a level). In
/// any other mesh the cells are split as above, then each cell left with
/// hanging vertices is replaced by transition cells: in 2D, triangles from
/// its centre to its sides, divided at their hanging vertices; in 3D, a cone
/// from its centre over each face: tets over triangles, pyramids over quads,
/// a face split as its neighbour split it, or else, with hanging vertices on
/// its edges, fanned into triangles from its centre (which both cells on it
/// share). Transition cells keep their parent's level and are cells like any
/// other in the result. Each records the cell it replaced in the cell field
/// "refine:green" (its gid, topology, corner count and corners' gids; zeros
/// on other cells), and the next call to refine_marked puts each group back
/// as that cell before it refines (red-green refinement): a group split
/// between ranks is gathered first on the rank of its least cell by gid,
/// whose gid, fields and level the cell put back takes, and it is marked
/// when any of its cells is.
///
/// options.vertices = VertexMatching::coordinates first finds, as
/// infer_births does, the births of vertices with no record (exactly, or
/// within options.tol); the recorded ones, the default, need no coordinates,
/// so vertices may move between calls. Matching costs one match_points over
/// the owned vertices and the points asked per depth of refinement found.
///
/// Gids, coordinates and fields follow refine_uniform (a vertex made at a
/// centre takes its corners' average); the result is the same on any number
/// of ranks. Collective; throws as refine_uniform does, and
/// std::invalid_argument on every rank if a rank's marks are not one per
/// owned cell.
[[nodiscard]] RefineResult refine_marked(const Comm& comm, const HostMesh& mesh, std::span<const std::int8_t> marks,
                                         MarkedOptions options = {});

struct QuadRefineOptions {
  std::optional<Real> sphere_radius;  ///< project new vertices onto this sphere about the origin
};

/// All-quad conforming refinement: the owned cells with a non-zero mark (one
/// per owned cell, in local id order) split into nine, and their neighbours
/// into transition quads, so that the result has quads alone and no hanging
/// vertex (Schneiders' vertex-based templates, edges split in three). The
/// input must be conforming: hanging vertices are not detected, and a mesh
/// that has them (refine_marked's default output, say) gets more.
///
/// Marks move to corners: a marked cell marks its four. A cell with two
/// opposite or three marked corners is promoted (all four marked), until no
/// cell changes, on every rank alike. Each cell then splits by its marked
/// corners (reference corners v0 (0,0), v1 (1,0), v2 (1,1), v3 (0,1), the
/// cell's corner order; the cases rotated to its marked corners):
///
/// ```
///   none        1 quad    the cell, whole
///   one (v0)    3 quads   a (1/3,0), c (0,1/3), p (1/3,1/3):
///                         [v0 a p c] [a v1 v2 p] [c p v2 v3]
///   two (v0 v1) 7 quads   a (1/3,0) b (2/3,0) d (1,1/3) c (0,1/3) and
///                         p q r s at (1/3,1/3) (2/3,1/3) (1/3,2/3) (2/3,2/3):
///                         [v0 a p c] [a b q p] [b v1 d q] [c p r v3]
///                         [p q s r] [q d v2 s] [r s v2 v3]
///   four        9 quads   the 3 x 3 grid
/// ```
///
/// An edge's new vertices depend on its ends alone (both marked: at 1/3 and
/// 2/3; one: at 1/3 from it), so the cells on an edge split it alike.
///
/// Positions: an edge's new vertex is (2/3) lo + (1/3) hi or (1/3) lo +
/// (2/3) hi, lo and hi its ends by gid; one inside a cell is the bilinear map
/// of the cell's corners at its reference position, its weights summed in
/// the corners' gid order. On a sphere (options.sphere_radius) each new
/// vertex is then projected onto it; the input's vertices stay as they are.
///
/// Gids: with V and E the largest vertex and edge gid on any rank (edges
/// derived as build_distributed_adjacencies derives them) and the cells
/// numbered n = gid - G + 1 (G the least cell gid), an edge's new vertex is
/// V + 2 (edge gid - 1) + k + 1, k = 0 for the one nearer its lower-gid end;
/// a cell's new vertex nearest its corner j is V + 2 E + 4 (n - 1) + j + 1;
/// child c, in the order above, is 9 (n - 1) + c + 1 (a whole cell is its
/// own child 0, so every cell is renumbered). The result is the same mesh,
/// gid for gid and bit for bit, on any number of ranks.
///
/// Levels are kept in the cell field "refine:level" (0 where missing): the
/// nine children of a cell split fully are one level up, the others keep
/// their parent's. No other state is kept; a later call takes the result as
/// a mesh like any other. Fields carry over as refine_uniform carries them,
/// a new vertex's value weighted as its position is (before projection). A
/// geometry the input has (mesh_geometry.hpp) is the result's where the new
/// vertices are on it: a sphere with that sphere_radius, affine with none.
///
/// Collective; throws std::invalid_argument on every rank if any rank's mesh
/// is not of quads alone (dimension 2, no edits, no high-order nodes), if a
/// rank's marks are not one per owned cell, if a rank's sphere radius is not
/// positive and finite or the ranks differ in it (or in giving one), if a cell to be split or one of its children would
/// not turn as the cell does at every corner (a non-convex cell, say: on a
/// sphere, about the cell's area vector), or if a gid of the result would not
/// fit a GO.
[[nodiscard]] RefineResult refine_quads(const Comm& comm, const HostMesh& mesh, std::span<const std::int8_t> marks,
                                        QuadRefineOptions options = {});

/// Records, in "refine:birth", the corners each vertex lies between, found by
/// coordinates (match_points, exactly or within tol) as refinement would
/// have placed it: at the midpoint of an owned cell's edge or the centre of
/// its quad face, then, as refinement cuts them, at the midpoints of those
/// edges' halves, of a triangle's halves about an edge's midpoint and of its
/// four children, and at the centres of a quad's four children, to any
/// depth. For meshes with hanging vertices whose records are missing (read
/// from a format that drops the field, made by other means, from an AMR
/// hierarchy), so that refine_marked reuses those vertices and lists them as
/// hanging. Recorded births stay; cell levels are not inferred ("refine:level"
/// is 0 where missing). A vertex moved since it was made (smoothed, snapped)
/// is not found. Collective; throws as refine_marked does and as match_points
/// does for tol.
void infer_births(const Comm& comm, HostMesh& mesh, PointTolerance tol = {});

/// The hanging vertices of a mesh, each as a combination of vertices that do
/// not hang: a vertex refine_marked made (or infer_births recorded, in
/// "refine:birth") on an edge or face that some cell still has whole takes
/// the average of that side's corners, a corner that hangs itself (one level
/// up) taking its own masters in turn. Exact for the traces of linear,
/// bilinear and trilinear elements, so a finite element space that so
/// constrains them is continuous (FemSpace).
struct HangingConstraints {
  std::vector<LO> vertex;          ///< the hanging vertices here, owned or not, ascending
  std::vector<Offset> offsets{0};  ///< vertex i's masters: [offsets[i], offsets[i + 1])
  std::vector<GO> masters;         ///< by gid, ascending; none hanging
  std::vector<Real> weights;       ///< adding up to 1
};

/// Collective: the hanging constraints of the vertices here. Every rank gets
/// every hanging vertex's record (an allgather of five gids per vertex), to
/// resolve masters that hang. Empty when no rank's mesh has "refine:birth".
[[nodiscard]] HangingConstraints hanging_constraints(const Comm& comm, const HostMesh& mesh);

/// Marks for refine_marked from an indicator per owned cell (local id order).
/// mark_threshold marks the cells whose indicator exceeds `threshold`.
/// mark_fraction marks the cells of largest indicator whose indicators add up
/// to at least `fraction` of their sum over every rank (Doerfler marking; the
/// indicators should be non-negative, squared error estimates, say): those
/// whose indicator is at least the largest cutoff that reaches the fraction,
/// so cells of equal indicator are marked alike. A fraction of 0, or a sum of
/// 0, marks nothing; the sums are rounded as each partition adds them.
/// Collective; throws std::invalid_argument on every rank for a fraction
/// outside [0, 1] or an indicator negative or not finite on any rank.
[[nodiscard]] std::vector<std::int8_t> mark_threshold(std::span<const Real> indicator, Real threshold);
[[nodiscard]] std::vector<std::int8_t> mark_fraction(const Comm& comm, std::span<const Real> indicator, Real fraction);

}  // namespace soma

#endif  // SOMA_ALGORITHMS_REFINE_HPP_
