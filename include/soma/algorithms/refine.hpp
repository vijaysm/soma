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
// entity of a full domain, per-entity layout, not variable, not handles):
// a cell field's children take the parent's value (so integrals of cell
// averages are kept), a vertex field of floating values keeps the values of
// the old vertices and gives a new one the average of its corners (exact for
// linear, bilinear and trilinear fields). Other fields, edge and face fields,
// and integer vertex fields are dropped and named in `dropped`; the internal
// soma: fields are rebuilt.

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
};

// Collective. Throws std::invalid_argument on every rank if any rank's mesh
// is not of dimension 2 (tris, quads, polygons) or 3 (tets, hexes, prisms,
// pyramids), has edits or high-order nodes, or if levels < 1; or if a gid
// of the refined mesh would not fit a GO.
[[nodiscard]] RefineResult refine_uniform(const Comm& comm, const HostMesh& mesh, RefineOptions options = {});

}  // namespace soma

#endif  // SOMA_ALGORITHMS_REFINE_HPP_
