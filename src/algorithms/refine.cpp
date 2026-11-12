// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/refine.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "soma/core/mesh_builder.hpp"
#include "soma/fields/registry.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/topology/adjacency.hpp"
#include "soma/topology/canonical.hpp"
#include "soma/topology/sides.hpp"

namespace soma {

namespace {

using Host = Kokkos::HostSpace;

struct P3 {
  Real x = 0, y = 0, z = 0;
  friend bool operator==(const P3&, const P3&) = default;
};

P3 operator+(const P3& a, const P3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
P3 operator-(const P3& a, const P3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
P3 half(const P3& a, const P3& b) { return {(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2}; }

P3 average(std::span<const P3> p) {
  P3 s;
  for (const P3& q : p) s = s + q;
  const auto n = static_cast<Real>(p.size());
  return {s.x / n, s.y / n, s.z / n};
}

Real det(const P3& a, const P3& b, const P3& c) {
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

std::vector<P3> reference_corners(Topology t) {
  switch (t) {
    case Topology::tri: return {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    case Topology::quad: return {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    case Topology::tet: return {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    case Topology::prism: return {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
    case Topology::pyramid: return {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {0, 0, 1}};
    case Topology::hex:
      return {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    default: return {};
  }
}

// The sign of a cell's turn in reference coordinates, from its first corner.
Real turn(Topology t, const std::vector<P3>& p) {
  switch (t) {
    case Topology::tri:
    case Topology::quad: {
      const P3 a = p[1] - p[0], b = p.back() - p[0];
      return a.x * b.y - a.y * b.x;
    }
    case Topology::tet:
    case Topology::prism: return det(p[1] - p[0], p[2] - p[0], p[3] - p[0]);
    case Topology::pyramid:
    case Topology::hex: return det(p[1] - p[0], p[3] - p[0], p[4] - p[0]);
    default: return 0;
  }
}

// The same cell turned the other way.
std::vector<P3> turned(Topology t, const std::vector<P3>& p) {
  std::vector<int> by;
  switch (t) {
    case Topology::tri: by = {0, 2, 1}; break;
    case Topology::quad: by = {0, 3, 2, 1}; break;
    case Topology::tet: by = {0, 2, 1, 3}; break;
    case Topology::prism: by = {0, 2, 1, 3, 5, 4}; break;
    case Topology::pyramid: by = {0, 3, 2, 1, 4}; break;
    case Topology::hex: by = {0, 3, 2, 1, 4, 7, 6, 5}; break;
    default: break;
  }
  std::vector<P3> out;
  for (const int i : by) out.push_back(p[static_cast<std::size_t>(i)]);
  return out;
}

Pattern make_pattern(Topology t, int dim) {
  Pattern pat;
  pat.ref = reference_corners(t);
  const std::vector<P3> c = pat.ref;
  const int n = static_cast<int>(c.size());
  const auto at = [&](int i) { return c[static_cast<std::size_t>(i)]; };
  for (int s = 0; s < canonical::num_sides(t, 1); ++s) {
    const canonical::SideDef e = canonical::side(t, 1, s);
    pat.ref.push_back(half(at(e.v[0]), at(e.v[1])));
  }
  if (dim == 3) {
    for (int s = 0; s < canonical::num_sides(t, 2); ++s) {
      const canonical::SideDef f = canonical::side(t, 2, s);
      if (f.num_vertices != 4) {
        pat.face_node.push_back(-1);
        continue;
      }
      pat.face_node.push_back(static_cast<int>(pat.ref.size()));
      const std::array<P3, 4> q{at(f.v[0]), at(f.v[1]), at(f.v[2]), at(f.v[3])};
      pat.ref.push_back(average(q));
    }
  }
  if (t == Topology::quad || t == Topology::hex) {
    pat.centre = static_cast<int>(pat.ref.size());
    pat.ref.push_back(average(c));
  }
  const auto node = [&](const P3& p) {
    const auto it = std::find(pat.ref.begin(), pat.ref.end(), p);
    if (it == pat.ref.end()) throw std::logic_error("refine_uniform: a child's corner is not a node");
    return static_cast<int>(it - pat.ref.begin());
  };
  const auto child = [&](Topology ct, std::vector<P3> p) {
    if (turn(ct, p) < 0) p = turned(ct, p);
    Child out{ct, {}};
    for (const P3& q : p) out.nodes.push_back(node(q));
    return out;
  };
  // The cell halved about each corner.
  for (int i = 0; i < n; ++i) {
    std::vector<P3> p;
    for (int j = 0; j < n; ++j) p.push_back(half(at(i), at(j)));
    pat.children.push_back(child(t, p));
  }
  if (t == Topology::tri) {
    pat.children.push_back(child(t, {half(at(0), at(1)), half(at(1), at(2)), half(at(2), at(0))}));
  } else if (t == Topology::prism) {
    const P3 h{0, 0, 0.5};
    const P3 a = half(at(0), at(1)), b = half(at(1), at(2)), d = half(at(2), at(0));
    pat.children.push_back(child(t, {a, b, d, a + h, b + h, d + h}));
    pat.children.push_back(child(t, {a + h, b + h, d + h, a + h + h, b + h + h, d + h + h}));
  } else if (t == Topology::pyramid) {
    const P3 base = average(std::span<const P3>(c.data(), 4));
    pat.children.push_back(
        child(Topology::pyramid,
              {half(at(0), at(4)), half(at(1), at(4)), half(at(2), at(4)), half(at(3), at(4)), base}));
    for (int i = 0; i < 4; ++i) {
      const int j = (i + 1) % 4;
      pat.children.push_back(child(Topology::tet, {base, half(at(i), at(j)), half(at(i), at(4)), half(at(j), at(4))}));
    }
  } else if (t == Topology::tet) {
    // Opposite edges (p, q) and (r, s); the other four midpoints go round
    // the diagonal between theirs.
    const int pairs[3][4] = {{0, 1, 2, 3}, {1, 2, 0, 3}, {2, 0, 1, 3}};
    for (int k = 0; k < 3; ++k) {
      const auto [p, q, r, s] = std::array<int, 4>{pairs[k][0], pairs[k][1], pairs[k][2], pairs[k][3]};
      const P3 a = half(at(p), at(q)), b = half(at(r), at(s));
      const std::array<P3, 4> ring{half(at(p), at(r)), half(at(r), at(q)), half(at(q), at(s)), half(at(s), at(p))};
      for (std::size_t i = 0; i < 4; ++i) {
        pat.octahedron[static_cast<std::size_t>(k)].push_back(child(Topology::tet, {a, b, ring[i], ring[(i + 1) % 4]}));
      }
    }
  }
  return pat;
}

const Pattern& pattern(Topology t) {
  static const std::array<Pattern, 6> all{make_pattern(Topology::tri, 2),   make_pattern(Topology::quad, 2),
                                          make_pattern(Topology::tet, 3),   make_pattern(Topology::prism, 3),
                                          make_pattern(Topology::pyramid, 3), make_pattern(Topology::hex, 3)};
  switch (t) {
    case Topology::tri: return all[0];
    case Topology::quad: return all[1];
    case Topology::tet: return all[2];
    case Topology::prism: return all[3];
    case Topology::pyramid: return all[4];
    default: return all[5];
  }
}

bool supported_cell(Topology t, int dim) {
  if (dim == 2) return t == Topology::tri || t == Topology::quad || t == Topology::polygon;
  return t == Topology::tet || t == Topology::hex || t == Topology::prism || t == Topology::pyramid;
}

// A vertex the refinement adds: its gid and its corners' gids, ascending.
struct NewVertex {
  GO gid = 0;
  std::vector<GO> corners;
};

// gid -> local id, from a sorted table.
LO find_lid(const std::vector<std::pair<GO, LO>>& table, GO gid) {
  const auto it = std::lower_bound(table.begin(), table.end(), std::pair<GO, LO>{gid, std::numeric_limits<LO>::min()});
  if (it == table.end() || it->first != gid) throw std::logic_error("refine_uniform: a gid is missing");
  return it->second;
}

std::vector<std::pair<GO, LO>> lid_table(const HostMesh& m, int d, LO count) {
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

// Carries a dense field of dimension d from `in` to `out`: a cell's children
// take its value; a vertex keeps its value, a new one takes its corners'
// average.
template <class T>
void carry(const HostMesh& in, HostMesh& out, const FieldInfo& info, int d, const std::vector<LO>& parent,
           const std::vector<std::pair<GO, LO>>& in_vertex, GO vmax, const std::vector<NewVertex>& added) {
  const auto from = in.fields().template get<T>(info.id).view();
  auto to = out.fields().template add_tag<T>(info.name, d, info.ncomp).view();
  const auto ncomp = static_cast<std::size_t>(info.ncomp);
  if (d != 0) {
    for (std::size_t c = 0; c < parent.size(); ++c) {
      for (std::size_t k = 0; k < ncomp; ++k) to(c, k) = from(static_cast<std::size_t>(parent[c]), k);
    }
    return;
  }
  const auto gid = out.gid(0);
  for (LO v = 0; v < out.count(0); ++v) {
    const GO g = gid(v);
    const auto vi = static_cast<std::size_t>(v);
    if (g <= vmax) {
      const auto lid = static_cast<std::size_t>(find_lid(in_vertex, g));
      for (std::size_t k = 0; k < ncomp; ++k) to(vi, k) = from(lid, k);
      continue;
    }
    const auto it = std::lower_bound(added.begin(), added.end(), g,
                                     [](const NewVertex& a, GO b) { return a.gid < b; });
    for (std::size_t k = 0; k < ncomp; ++k) {
      Real s = 0;
      for (const GO cg : it->corners) {
        s += static_cast<Real>(from(static_cast<std::size_t>(find_lid(in_vertex, cg)), k));
      }
      to(vi, k) = static_cast<T>(s / static_cast<Real>(it->corners.size()));
    }
  }
}

Level refine_once(const Comm& comm, const HostMesh& mesh) {
  const int dim = mesh.dim();
  const MeshView<Host> m = mesh.view();
  const LO owned = mesh.segment_table(dim).owned_end;
  const auto vgid = mesh.gid(0);
  const auto cgid = mesh.gid(dim);
  const auto x = mesh.coords();

  // The owned cells alone, with their edges (and faces) derived.
  HostMesh work;
  {
    MeshBuilder b(dim);
    std::vector<LO> used;
    for (LO e = 0; e < owned; ++e) {
      for (LO i = 0; i < m.num_vertices(dim, e); ++i) used.push_back(m.vertex(dim, e, i));
    }
    std::sort(used.begin(), used.end());
    used.erase(std::unique(used.begin(), used.end()), used.end());
    std::vector<GO> gids;
    std::vector<Real> xyz;
    for (const LO v : used) {
      gids.push_back(vgid(v));
      xyz.insert(xyz.end(), {x(v, 0), x(v, 1), x(v, 2)});
    }
    b.add_vertices(gids, xyz);
    for (const Topology t : {Topology::tri, Topology::quad, Topology::tet, Topology::prism, Topology::pyramid,
                             Topology::hex}) {
      std::vector<GO> eg, conn;
      for (LO e = 0; e < owned; ++e) {
        if (m.topology(dim, e) != t) continue;
        eg.push_back(cgid(e));
        for (LO i = 0; i < m.num_vertices(dim, e); ++i) conn.push_back(vgid(m.vertex(dim, e, i)));
      }
      if (!eg.empty()) b.add_elements(t, eg, conn);
    }
    std::vector<GO> pg, pv;
    std::vector<Offset> po{0};
    for (LO e = 0; e < owned; ++e) {
      if (m.topology(dim, e) != Topology::polygon) continue;
      pg.push_back(cgid(e));
      for (LO i = 0; i < m.num_vertices(dim, e); ++i) pv.push_back(vgid(m.vertex(dim, e, i)));
      po.push_back(static_cast<Offset>(pv.size()));
    }
    if (!pg.empty()) b.add_polygons(pg, po, pv);
    work = build_local_part(comm, std::move(b));
  }
  AdjacencyRequest request;
  request.add(dim, 1);
  if (dim == 3) request.add(dim, 2);
  build_distributed_adjacencies(comm, work, request);

  const MeshView<Host> w = work.view();
  const LO cells = work.count(dim);
  const auto wv = work.gid(0);
  const auto we = work.gid(1);
  const auto wc = work.gid(dim);
  const auto wx = work.coords();
  // Largest vertex, edge, face and cell gids, most children; then the gid
  // ranges they make, checked against overflow.
  std::array<GO, 6> top{0, 0, 0, std::numeric_limits<GO>::min(), 0, std::numeric_limits<GO>::min()};
  for (LO v = 0; v < work.count(0); ++v) top[0] = std::max(top[0], wv(v));
  for (LO e = 0; e < work.count(1); ++e) top[1] = std::max(top[1], we(e));
  if (dim == 3) {
    for (LO f = 0; f < work.count(2); ++f) top[2] = std::max(top[2], work.gid(2)(f));
  }
  for (LO e = 0; e < cells; ++e) {
    top[3] = std::max(top[3], wc(e));
    top[5] = std::max(top[5], -wc(e));
    const Topology t = w.topology(dim, e);
    top[4] = std::max<GO>(top[4], t == Topology::polygon ? static_cast<GO>(w.num_vertices(dim, e))
                                  : t == Topology::tri || t == Topology::quad ? 4
                                  : t == Topology::pyramid ? 10
                                                           : 8);
  }
  comm.allreduce(std::span<GO>(top), Op::max);
  // Cells are numbered from the least cell gid, cmin, as 1.
  const GO vmax = top[0], emax = top[1], fmax = top[2], kmax = top[4];
  const GO cmin = top[5] == std::numeric_limits<GO>::min() ? 1 : -top[5];
  constexpr GO most = std::numeric_limits<GO>::max();
  const bool cells_here = top[3] != std::numeric_limits<GO>::min();  // on some rank
  if (cells_here && cmin < 0 && top[3] > most + cmin) {
    throw std::invalid_argument("refine_uniform: the refined mesh's gids would not fit a GO");
  }
  const GO span = cells_here && top[3] - cmin < most ? top[3] - cmin + 1 : 0;
  if (vmax > most - emax || vmax + emax > most - fmax || vmax + emax + fmax > most - span ||
      (span > 0 && kmax > most / span)) {
    throw std::invalid_argument("refine_uniform: the refined mesh's gids would not fit a GO");
  }
  const GO face_base = vmax + emax, centre_base = vmax + emax + fmax;
  const auto number = [&](GO cell) { return cell - cmin + 1; };

  // Children, as (type, gid, corner gids); new vertices.
  const auto by_gid = [&](LO a, LO b) { return wv(a) < wv(b); };
  const auto midpoint = [&](std::vector<LO> corners) {
    std::sort(corners.begin(), corners.end(), by_gid);
    std::vector<P3> p;
    for (const LO v : corners) p.push_back({wx(v, 0), wx(v, 1), wx(v, 2)});
    return average(p);
  };
  std::vector<NewVertex> added;
  const auto add = [&](GO gid, std::vector<LO> corners) {
    std::sort(corners.begin(), corners.end(), by_gid);
    NewVertex nv{gid, {}};
    for (const LO v : corners) nv.corners.push_back(wv(v));
    added.push_back(std::move(nv));
  };
  std::array<std::vector<GO>, 6> kid_gids, kid_conn;  // by pattern slot
  const auto slot = [](Topology t) {
    switch (t) {
      case Topology::tri: return 0;
      case Topology::quad: return 1;
      case Topology::tet: return 2;
      case Topology::prism: return 3;
      case Topology::pyramid: return 4;
      default: return 5;
    }
  };
  const auto edges = work.adjacency(dim, 1);
  for (LO e = 0; e < cells; ++e) {
    const Topology t = w.topology(dim, e);
    const LO n = w.num_vertices(dim, e);
    const GO base = kmax * (number(wc(e)) - 1);
    // The cell's nodes: gids, and corners for the new ones.
    std::vector<GO> node;
    std::vector<std::vector<LO>> corners;
    for (LO i = 0; i < n; ++i) node.push_back(wv(w.vertex(dim, e, i)));
    const Offset e0 = edges.offsets(static_cast<std::size_t>(e));
    for (int s = 0; s < topology::side_count(w, dim, e, 1); ++s) {
      const topology::Side side = topology::side_of(w, dim, e, 1, s);
      node.push_back(vmax + we(edges.targets(static_cast<std::size_t>(e0 + s))));
      corners.push_back({side.v[0], side.v[1]});
      add(node.back(), corners.back());
    }
    if (t == Topology::polygon) {
      std::vector<LO> all;
      for (LO i = 0; i < n; ++i) all.push_back(w.vertex(dim, e, i));
      const GO centre = centre_base + number(wc(e));
      add(centre, all);
      for (LO i = 0; i < n; ++i) {
        const LO prev = (i + n - 1) % n;
        kid_gids[1].push_back(base + i + 1);
        kid_conn[1].insert(kid_conn[1].end(), {node[static_cast<std::size_t>(i)], node[static_cast<std::size_t>(n + i)],
                                               centre, node[static_cast<std::size_t>(n + prev)]});
      }
      continue;
    }
    const Pattern& pat = pattern(t);
    if (dim == 3) {
      const auto faces = work.adjacency(dim, 2);
      const Offset f0 = faces.offsets(static_cast<std::size_t>(e));
      for (int s = 0; s < static_cast<int>(pat.face_node.size()); ++s) {
        if (pat.face_node[static_cast<std::size_t>(s)] < 0) continue;
        const topology::Side side = topology::side_of(w, dim, e, 2, s);
        node.push_back(face_base + work.gid(2)(faces.targets(static_cast<std::size_t>(f0 + s))));
        add(node.back(), {side.v[0], side.v[1], side.v[2], side.v[3]});
      }
    }
    if (pat.centre >= 0) {
      std::vector<LO> all;
      for (LO i = 0; i < n; ++i) all.push_back(w.vertex(dim, e, i));
      node.push_back(centre_base + number(wc(e)));
      add(node.back(), all);
    }
    std::vector<const Child*> kids;
    for (const Child& c : pat.children) kids.push_back(&c);
    if (t == Topology::tet) {
      // The octahedron's shortest diagonal: between the midpoints of edges
      // k and its opposite (5, 3, 4).
      const int opposite[3] = {5, 3, 4};
      int best = 0;
      Real shortest = std::numeric_limits<Real>::infinity();
      for (int k = 0; k < 3; ++k) {
        const P3 d = midpoint(corners[static_cast<std::size_t>(k)]) -
                     midpoint(corners[static_cast<std::size_t>(opposite[k])]);
        const Real l = d.x * d.x + d.y * d.y + d.z * d.z;
        if (l < shortest) {
          shortest = l;
          best = k;
        }
      }
      for (const Child& c : pat.octahedron[static_cast<std::size_t>(best)]) kids.push_back(&c);
    }
    for (std::size_t k = 0; k < kids.size(); ++k) {
      const int s = slot(kids[k]->type);
      kid_gids[static_cast<std::size_t>(s)].push_back(base + static_cast<GO>(k) + 1);
      auto& conn = kid_conn[static_cast<std::size_t>(s)];
      for (const int i : kids[k]->nodes) conn.push_back(node[static_cast<std::size_t>(i)]);
    }
  }
  std::sort(added.begin(), added.end(), [](const NewVertex& a, const NewVertex& b) { return a.gid < b.gid; });
  const auto same = [](const NewVertex& a, const NewVertex& b) { return a.gid == b.gid; };
  added.erase(std::unique(added.begin(), added.end(), same), added.end());

  // The refined part.
  MeshBuilder b(dim);
  {
    std::vector<GO> gids;
    std::vector<Real> xyz;
    for (LO v = 0; v < work.count(0); ++v) {
      gids.push_back(wv(v));
      xyz.insert(xyz.end(), {wx(v, 0), wx(v, 1), wx(v, 2)});
    }
    const auto work_vertex = lid_table(work, 0, work.count(0));
    for (const NewVertex& nv : added) {
      std::vector<P3> p;
      for (const GO g : nv.corners) {
        const LO v = find_lid(work_vertex, g);
        p.push_back({wx(v, 0), wx(v, 1), wx(v, 2)});
      }
      const P3 c = average(p);
      gids.push_back(nv.gid);
      xyz.insert(xyz.end(), {c.x, c.y, c.z});
    }
    b.add_vertices(gids, xyz);
  }
  const Topology types[6] = {Topology::tri, Topology::quad, Topology::tet, Topology::prism, Topology::pyramid,
                             Topology::hex};
  for (std::size_t s = 0; s < 6; ++s) {
    if (!kid_gids[s].empty()) b.add_elements(types[s], kid_gids[s], kid_conn[s]);
  }
  Level out;
  out.mesh = build_local_part(comm, std::move(b));

  // Parents: by gid, among the input's owned cells.
  const auto in_cell = lid_table(mesh, dim, owned);
  const auto fg = out.mesh.gid(dim);
  for (LO c = 0; c < out.mesh.count(dim); ++c) out.parent.push_back(find_lid(in_cell, (fg(c) - 1) / kmax + cmin));

  // Fields.
  const auto in_vertex = lid_table(mesh, 0, mesh.count(0));
  for (const FieldInfo& info : mesh.fields().list()) {
    if (info.name.rfind("soma:", 0) == 0) continue;
    const bool dense = !info.variable && !info.handle && info.layout == Layout::per_entity() && info.dims.size() == 1 &&
                       (info.dims[0] == 0 || info.dims[0] == dim) && info.rows == mesh.count(info.dims[0]);
    const bool real = info.type == DataType::f32 || info.type == DataType::f64;
    if (!dense || (info.dims[0] == 0 && !real)) {
      out.dropped.push_back(info.name);
      continue;
    }
    const int d = info.dims[0];
    switch (info.type) {
      case DataType::i8: carry<std::int8_t>(mesh, out.mesh, info, d, out.parent, in_vertex, vmax, added); break;
      case DataType::u8: carry<std::uint8_t>(mesh, out.mesh, info, d, out.parent, in_vertex, vmax, added); break;
      case DataType::i32: carry<std::int32_t>(mesh, out.mesh, info, d, out.parent, in_vertex, vmax, added); break;
      case DataType::u32: carry<std::uint32_t>(mesh, out.mesh, info, d, out.parent, in_vertex, vmax, added); break;
      case DataType::i64: carry<std::int64_t>(mesh, out.mesh, info, d, out.parent, in_vertex, vmax, added); break;
      case DataType::u64: carry<std::uint64_t>(mesh, out.mesh, info, d, out.parent, in_vertex, vmax, added); break;
      case DataType::f32: carry<float>(mesh, out.mesh, info, d, out.parent, in_vertex, vmax, added); break;
      case DataType::f64: carry<double>(mesh, out.mesh, info, d, out.parent, in_vertex, vmax, added); break;
    }
  }
  return out;
}

}  // namespace

RefineResult refine_uniform(const Comm& comm, const HostMesh& mesh, RefineOptions options) {
  const int dim = mesh.dim();
  int bad = 0;
  if (dim != 2 && dim != 3) {
    bad = 1;
  } else if (mesh.edited() || mesh.has_nodes(dim)) {
    bad = 2;
  } else {
    const MeshView<Host> m = mesh.view();
    for (LO e = 0; e < m.count(dim) && bad == 0; ++e) {
      if (!supported_cell(m.topology(dim, e), dim)) bad = 3;
    }
  }
  if (options.levels < 1) bad = 4;
  bad = comm.allreduce(bad, Op::max);
  if (bad == 1) throw std::invalid_argument("refine_uniform: a rank's mesh is not of dimension 2 or 3");
  if (bad == 2) throw std::invalid_argument("refine_uniform: a rank's mesh has edits or high-order nodes");
  if (bad == 3) {
    throw std::invalid_argument("refine_uniform: a rank's mesh holds a cell other than a tri, quad or polygon (2D) or "
                                "a tet, hex, prism or pyramid (3D)");
  }
  if (bad == 4) throw std::invalid_argument("refine_uniform: levels must be at least 1");

  RefineResult out;
  Level level = refine_once(comm, mesh);
  out.parent = std::move(level.parent);
  out.dropped = std::move(level.dropped);
  for (int l = 1; l < options.levels; ++l) {
    Level next = refine_once(comm, level.mesh);
    for (LO& p : next.parent) p = out.parent[static_cast<std::size_t>(p)];
    out.parent = std::move(next.parent);
    level = std::move(next);
  }
  out.mesh = std::move(level.mesh);
  const LO owned = mesh.segment_table(dim).owned_end;
  out.child_offsets.assign(static_cast<std::size_t>(owned) + 1, 0);
  for (const LO p : out.parent) ++out.child_offsets[static_cast<std::size_t>(p) + 1];
  for (std::size_t p = 0; p < static_cast<std::size_t>(owned); ++p) out.child_offsets[p + 1] += out.child_offsets[p];
  out.children.resize(out.parent.size());
  std::vector<Offset> next(out.child_offsets.begin(), out.child_offsets.end() - 1);
  for (std::size_t c = 0; c < out.parent.size(); ++c) {
    out.children[static_cast<std::size_t>(next[static_cast<std::size_t>(out.parent[c])]++)] = static_cast<LO>(c);
  }
  return out;
}

}  // namespace soma
