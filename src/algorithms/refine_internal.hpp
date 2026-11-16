// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_ALGORITHMS_REFINE_INTERNAL_HPP_
#define SOMA_SRC_ALGORITHMS_REFINE_INTERNAL_HPP_

// What uniform and marked refinement share (src/algorithms/refine*.cpp):
// cell patterns in reference coordinates, the work part of owned cells, gid
// numbering and the assembly of the refined part with its fields.

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"

namespace soma::detail::refine {

using Host = Kokkos::HostSpace;

struct P3 {
  Real x = 0, y = 0, z = 0;
  friend bool operator==(const P3&, const P3&) = default;
};

inline P3 operator+(const P3& a, const P3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline P3 operator-(const P3& a, const P3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline P3 half(const P3& a, const P3& b) { return {(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2}; }

inline P3 average(std::span<const P3> p) {
  P3 s;
  for (const P3& q : p) s = s + q;
  const auto n = static_cast<Real>(p.size());
  return {s.x / n, s.y / n, s.z / n};
}

inline Real det(const P3& a, const P3& b, const P3& c) {
  return a.x * (b.y * c.z - b.z * c.y) - a.y * (b.x * c.z - b.z * c.x) + a.z * (b.x * c.y - b.y * c.x);
}

// A child: its type and its parent's local nodes.
struct Child {
  Topology type = Topology::tri;
  std::vector<int> nodes;
};

// A fixed cell type's nodes, in reference coordinates, and its children:
// nodes are the corners, the edges' midpoints in edge order, the quad faces'
// centres in face order (3D) and the centre (quads, hexes). A tet has its
// four corner children and then four about one diagonal of the octahedron,
// per diagonal.
struct Pattern {
  std::vector<P3> ref;
  std::vector<int> face_node;  // per face, its centre's node or -1 (3D)
  int centre = -1;
  std::vector<Child> children;          // all children (tets: the corners' only)
  std::array<std::vector<Child>, 3> octahedron;  // tets: per diagonal
};

// The pattern of a fixed cell type (not a polygon).
const Pattern& pattern(Topology t);

inline bool supported_cell(Topology t, int dim) {
  if (dim == 2) return t == Topology::tri || t == Topology::quad || t == Topology::polygon;
  return t == Topology::tet || t == Topology::hex || t == Topology::prism || t == Topology::pyramid;
}

// A vertex the refinement adds (or, in marked refinement, reuses): its gid
// and its corners' gids, ascending.
struct NewVertex {
  GO gid = 0;
  std::vector<GO> corners;
};

// gid -> local id, from a sorted table.
inline LO find_lid(const std::vector<std::pair<GO, LO>>& table, GO gid) {
  const auto it = std::lower_bound(table.begin(), table.end(), std::pair<GO, LO>{gid, std::numeric_limits<LO>::min()});
  if (it == table.end() || it->first != gid) throw std::logic_error("refinement: a gid is missing");
  return it->second;
}

inline std::vector<std::pair<GO, LO>> lid_table(const HostMesh& m, int d, LO count) {
  std::vector<std::pair<GO, LO>> t;
  const auto gid = m.gid(d);
  for (LO e = 0; e < count; ++e) t.emplace_back(gid(e), e);
  std::sort(t.begin(), t.end());
  return t;
}

struct Level {
  HostMesh mesh;
  std::vector<LO> parent;
  std::vector<std::string> dropped;
};

// The owned cells of `mesh` alone, as a part of their own, with the extra
// free vertices given (gids, xyz), and their edges (and, in 3D, faces)
// derived: gids that do not depend on the ranks.
HostMesh work_part(const Comm& comm, const HostMesh& mesh, std::span<const GO> extra_gids = {},
                   std::span<const Real> extra_xyz = {});

// Largest vertex, edge and face gids on any rank, the cells' numbers (from
// the least cell gid as 1) and the most children of a cell; throws on every
// rank if the gids they make would not fit a GO.
struct Numbers {
  GO vmax = 0, emax = 0, fmax = 0, kmax = 1, cmin = 1;
  [[nodiscard]] GO number(GO cell) const { return cell - cmin + 1; }
  [[nodiscard]] GO child(GO cell, GO c) const { return kmax * (number(cell) - 1) + c + 1; }
  [[nodiscard]] GO parent(GO child_gid) const { return (child_gid - 1) / kmax + cmin; }
};

template <class Children>
Numbers numbers(const Comm& comm, const HostMesh& work, const Children& children, const char* who) {
  const int dim = work.dim();
  const MeshView<Host> w = work.view();
  const LO cells = work.count(dim);
  const auto wv = work.gid(0);
  const auto we = work.gid(1);
  const auto wc = work.gid(dim);
  // Largest vertex, edge, face and cell gids, most children; then the gid
  // ranges they make, checked against overflow.
  std::array<GO, 5> top{0, 0, 0, std::numeric_limits<GO>::min(), 0};
  GO least = std::numeric_limits<GO>::max();
  for (LO v = 0; v < work.count(0); ++v) top[0] = std::max(top[0], wv(v));
  for (LO e = 0; e < work.count(1); ++e) top[1] = std::max(top[1], we(e));
  if (dim == 3) {
    for (LO f = 0; f < work.count(2); ++f) top[2] = std::max(top[2], work.gid(2)(f));
  }
  for (LO e = 0; e < cells; ++e) {
    top[3] = std::max(top[3], wc(e));
    least = std::min(least, wc(e));
    top[4] = std::max<GO>(top[4], children(w, e));
  }
  comm.allreduce(std::span<GO>(top), Op::max);
  least = comm.allreduce(least, Op::min);
  // Cells are numbered from the least cell gid, cmin, as 1.
  const GO vmax = top[0], emax = top[1], fmax = top[2], kmax = top[4];
  const bool cells_here = top[3] != std::numeric_limits<GO>::min();  // on some rank
  const GO cmin = cells_here ? least : 1;
  constexpr GO most = std::numeric_limits<GO>::max();
  // span = top[3] - cmin + 1 cell numbers, without overflow.
  const bool wide = cells_here && (cmin < 0 ? top[3] >= most + cmin : top[3] - cmin >= most);
  const GO span = cells_here && !wide ? top[3] - cmin + 1 : 0;
  if (wide || vmax > most - emax || vmax + emax > most - fmax || vmax + emax + fmax > most - span ||
      (span > 0 && kmax > most / span)) {
    throw std::invalid_argument(std::string(who) + ": the refined mesh's gids would not fit a GO");
  }
  return {vmax, emax, fmax, std::max<GO>(kmax, 1), cmin};
}

// Where the new nodes of a cell being split come from: the gid of edge s's
// midpoint and of quad face s's centre (sides in canonical order), and of
// the centre (quads, polygons, hexes).
struct NodeGids {
  std::function<GO(int)> edge;
  std::function<GO(int)> face;
  GO centre = 0;
};

// The number of children of a cell split as refine_uniform splits it.
int children_of(Topology t, LO corners);

// The pattern slot of a child type: tri, quad, tet, prism, pyramid, hex.
int slot(Topology t);

// Splits cell e of `work` as refine_uniform does, its children numbered
// base + 1, base + 2, ..., appended to kid_gids and kid_conn by slot; each
// node other than a corner goes to `added` with its corners.
void split_cell(const HostMesh& work, LO e, GO base, const NodeGids& ids, std::array<std::vector<GO>, 6>& kid_gids,
                std::array<std::vector<GO>, 6>& kid_conn, std::vector<NewVertex>& added);

// Polygon cells of the refined part (cells left as they were).
struct Polygons {
  std::vector<GO> gids;
  std::vector<Offset> offsets{0};
  std::vector<GO> conn;
};

// The refined part from the children (by pattern slot: tri, quad, tet,
// prism, pyramid, hex; and polygons) and the vertices added, with the
// vertices the children use; parents and fields (but refine: ones) from
// `mesh`, whose owned cells `work` holds.
Level assemble(const Comm& comm, const HostMesh& mesh, const HostMesh& work, std::array<std::vector<GO>, 6>& kid_gids,
               std::array<std::vector<GO>, 6>& kid_conn, std::vector<NewVertex>& added, const Numbers& nums,
               const Polygons& polygons);

// Collective: throws std::invalid_argument on every rank if any rank's mesh
// cannot be refined or the ranks disagree on the levels or the dimension;
// messages begin with `who`.
void check(const Comm& comm, const HostMesh& mesh, int levels, const char* who, bool shapes = true);

// Whether a plane cell is star-shaped about its corners' average (its
// children, and the triangles from that point to its sides, turn as it does).
bool star_shaped(const HostMesh& mesh, LO e);

}  // namespace soma::detail::refine

#endif  // SOMA_SRC_ALGORITHMS_REFINE_INTERNAL_HPP_
