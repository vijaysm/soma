// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_REFINE_HPP_
#define SOMA_ALGORITHMS_REFINE_HPP_

// Uniform refinement: every edge of a mesh split in two, each cell into
// children that tile it, on any number of ranks:
//
//   const soma::RefineResult fine = soma::refine_uniform(comm, mesh, {.levels = 2});
//   fine.mesh;       // the refined local part, children on their parent's rank
//   fine.parent[c];  // the cell of `mesh` that fine cell c lies in
//
// Cells and their children (a cell's nodes: its corners, its edges'
// midpoints, its quad faces' centres, its centre where it has one):
//
//   tri      4 tris      the corners' halves and the middle one
//   quad     4 quads     with the centre
//   polygon  n quads     one per corner, with the centroid (corners averaged)
//   tet      8 tets      the corners' halves and four about the shortest
//                        diagonal of the octahedron left (the first in edge
//                        order of those as short)
//   hex      8 hexes     with the faces' centres and the centre
//   prism    8 prisms    with the quad faces' centres
//   pyramid  6 pyramids  with the base's centre: the corners' halves, one
//            + 4 tets    upside down under the apex's, and four tets
//
// A corner's child is the cell halved about it, so children are turned as
// their parent is; the others are turned so in the parent's reference
// coordinates. Every quad face gets a centre and no triangle does, so cells
// of every type meeting at a face split it alike, and the refined mesh is
// conforming where the mesh is. A new vertex is the average of its corners
// (an edge's two, a face's four, a cell's all), summed in gid order, so ranks
// holding it agree bit for bit.
//
// Gids do not depend on the number of ranks: with V, E and F the largest
// vertex, edge and face gid on any rank (edges and faces derived as
// build_distributed_adjacencies derives them), an edge's midpoint is V + its
// edge gid, a quad face's centre V + E + its face gid (3D); with the cells
// numbered from 1 in gid order (n = gid - G + 1, G the least cell gid), a
// cell's centre is V + E + F + n (F = 0 in 2D) and its child c is
// K (n - 1) + c + 1, K the most children of any cell. The refined mesh is
// therefore the same, gid for gid and bit for bit, on any number of ranks.
//
// Each rank refines the cells it owns; the result has no ghosts (add them
// again as needed). Fields carry over where they are dense (one value per
// entity of a full domain, per-entity layout, not variable, not handles),
// with their default and global values: a cell field's children take the
// parent's value (so integrals of cell averages are kept), a vertex field of
// floating values keeps the values of the old vertices and gives a new one
// the average of its corners (exact for linear, bilinear and trilinear
// fields). Vertex values are read where a rank holds them, shared copies and
// ghosts included, so these must hold their owners' values (update them
// first) for ranks to agree. Other fields, edge and face fields, and integer
// vertex fields are dropped and named in `dropped`; the internal soma:
// fields are rebuilt.
//
// A child's corners are its parent's mapped from reference coordinates, so a
// cell not star-shaped about its corners' average (a non-convex quad or
// polygon; in 3D, a strongly distorted cell) would have children turned
// against it: plane cells are checked and refused, volume cells are not.
//
// The refinement runs on the host (a HostMesh in, built with
// build_local_part); a device mesh is mirrored to the host and back.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

struct RefineOptions {
  int levels = 1;  // passes; each splits every edge in two
};

struct RefineResult {
  HostMesh mesh;
  // For each cell of `mesh`, the owned cell of the input it lies in (a local
  // id of the input).
  std::vector<LO> parent;
  // The children of the input's owned cells: cell p's are
  // children[child_offsets[p], child_offsets[p + 1]), local ids of `mesh`.
  std::vector<Offset> child_offsets;
  std::vector<LO> children;
  // Fields not carried over, by name.
  std::vector<std::string> dropped;
  // Marked refinement: the hanging vertices of `mesh` (local ids), each
  // the midpoint of an edge or the centre of a quad face that a cell still
  // has whole: its value is sum_k weights[k] value(masters[k]) over
  // k in [hanging_offsets[i], hanging_offsets[i + 1]), masters by gid (they
  // may be hanging themselves, one level up).
  std::vector<LO> hanging;
  std::vector<Offset> hanging_offsets{0};
  std::vector<GO> hanging_masters;
  std::vector<Real> hanging_weights;
};

struct MarkedOptions {
  // Conforming refinement: no hanging vertices (see refine_marked).
  bool conforming = false;
};

// Collective. Throws std::invalid_argument on every rank if any rank's mesh
// is not of dimension 2 (tris, quads, polygons) or 3 (tets, hexes, prisms,
// pyramids), has edits or high-order nodes, or a quad or polygon not
// star-shaped about its corners' average; if levels < 1 or the ranks differ
// in levels or dimension; or if a gid of the refined mesh would not fit a GO.
[[nodiscard]] RefineResult refine_uniform(const Comm& comm, const HostMesh& mesh, RefineOptions options = {});

// Marked refinement: the owned cells with a non-zero mark (one per owned
// cell, in local id order), and the cells the 2:1 balance needs, split once
// as refine_uniform splits them; the others stay whole, so their edges and
// faces can carry hanging vertices (non-conforming). Levels are kept in the
// cell field "refine:level" (0 where it is missing) and each vertex the
// refinement makes at an edge's midpoint or a quad face's centre records the
// corners it lies between in the vertex field "refine:birth" (count, then
// up to four gids), so that later calls and other ranks reuse it: through a
// directory keyed by those corners, whatever the partition.
//
// Balance: a cell is split when a cell touching it at a corner, or at a
// vertex born on its edges or faces (or on theirs, one level down), would be
// two levels finer than it.
//
// options.conforming leaves no hanging vertex. A mesh of triangles alone, or
// of tets alone, is bisected instead: each cell marked is cut in two at the
// midpoint of its longest edge (the one of least corner gids among equals),
// and any cell with a vertex on one of its edges is cut so again, until none
// is left (longest-edge, Rivara, bisection; each cut counts as a level). In
// any other mesh the cells are split as above, then each cell left with
// hanging vertices is replaced by transition cells: in 2D, triangles from
// its centre to its sides, divided at their hanging vertices; in 3D, a cone
// from its centre over each face: tets over triangles, pyramids over quads,
// a face split as its neighbour split it, or else, with hanging vertices on
// its edges, fanned into triangles from its centre (which both cells on it
// share). Transition cells keep their parent's level and are cells like any
// other afterwards.
//
// Gids, coordinates and fields follow refine_uniform (a vertex made at a
// centre takes its corners' average); the result is the same on any number
// of ranks. Collective; throws as refine_uniform does, and
// std::invalid_argument on every rank if a rank's marks are not one per
// owned cell.
[[nodiscard]] RefineResult refine_marked(const Comm& comm, const HostMesh& mesh, std::span<const std::int8_t> marks,
                                         MarkedOptions options = {});

}  // namespace soma

#endif  // SOMA_ALGORITHMS_REFINE_HPP_
