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
#include "refine_internal.hpp"

namespace soma::detail::refine {



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


// Carries a dense field of dimension d from `in` to `out`: a cell's children
// take its value; a vertex keeps its value, a new one takes its corners'
// weighted sum (or average).
template <class T>
void carry(const HostMesh& in, HostMesh& out, const FieldInfo& info, int d, const std::vector<LO>& parent,
           const std::vector<std::pair<GO, LO>>& in_vertex, const std::vector<NewVertex>& added) {
  const auto source = in.fields().template get<T>(info.id);
  const auto from = source.view();
  FieldOptions<T> options;
  if (source.has_default()) options.default_value = source.default_value();
  if (source.has_global()) options.global_value = source.global_value();
  auto to = out.fields().template add_tag<T>(info.name, d, info.ncomp, std::move(options)).view();
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
    // A vertex of the input keeps its value; one added (or reused from
    // another rank) takes its corners' average.
    const auto it = std::lower_bound(added.begin(), added.end(), g,
                                     [](const NewVertex& a, GO b) { return a.gid < b; });
    const auto held =
        std::lower_bound(in_vertex.begin(), in_vertex.end(), std::pair<GO, LO>{g, std::numeric_limits<LO>::min()});
    if (it == added.end() || it->gid != g || (held != in_vertex.end() && held->first == g)) {
      const auto lid = static_cast<std::size_t>(find_lid(in_vertex, g));
      for (std::size_t k = 0; k < ncomp; ++k) to(vi, k) = from(lid, k);
      continue;
    }
    for (std::size_t k = 0; k < ncomp; ++k) {
      Real s = 0;
      for (std::size_t j = 0; j < it->corners.size(); ++j) {
        const auto value = static_cast<Real>(from(static_cast<std::size_t>(find_lid(in_vertex, it->corners[j])), k));
        s += it->weights.empty() ? value : it->weights[j] * value;
      }
      to(vi, k) = static_cast<T>(it->weights.empty() ? s / static_cast<Real>(it->corners.size()) : s);
    }
  }
}

// The owned cells of `mesh` alone, as a part of their own, with their edges
// (and, in 3D, faces) derived: gids that do not depend on the ranks.
HostMesh work_part(const Comm& comm, const HostMesh& mesh, std::span<const GO> extra_gids,
                   std::span<const Real> extra_xyz) {
  const int dim = mesh.dim();
  const MeshView<Host> m = mesh.view();
  const LO owned = mesh.segment_table(dim).owned_end;
  const auto vgid = mesh.gid(0);
  const auto cgid = mesh.gid(dim);
  const auto x = mesh.coords();
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
    gids.insert(gids.end(), extra_gids.begin(), extra_gids.end());
    xyz.insert(xyz.end(), extra_xyz.begin(), extra_xyz.end());
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
  return work;
}


// The refined part from the children (by pattern slot: tri, quad, tet,
// prism, pyramid, hex) and the vertices added (projected onto a sphere when
// a radius is given); parents and fields from `mesh`, whose owned cells
// `work` holds.
Level assemble(const Comm& comm, const HostMesh& mesh, const HostMesh& work, std::array<std::vector<GO>, 6>& kid_gids,
               std::array<std::vector<GO>, 6>& kid_conn, std::vector<NewVertex>& added, const Numbers& nums,
               const Polygons& polygons, std::optional<Real> sphere_radius) {
  const int dim = mesh.dim();
  const LO owned = mesh.segment_table(dim).owned_end;
  const auto wv = work.gid(0);
  const auto wx = work.coords();
  std::sort(added.begin(), added.end(), [](const NewVertex& a, const NewVertex& b) { return a.gid < b.gid; });
  const auto same = [](const NewVertex& a, const NewVertex& b) { return a.gid == b.gid; };
  added.erase(std::unique(added.begin(), added.end(), same), added.end());

  // The refined part: the vertices the children use.
  std::vector<GO> used;
  for (const auto& conn : kid_conn) used.insert(used.end(), conn.begin(), conn.end());
  used.insert(used.end(), polygons.conn.begin(), polygons.conn.end());
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  const auto in_use = [&](GO g) { return std::binary_search(used.begin(), used.end(), g); };
  MeshBuilder b(dim);
  {
    std::vector<GO> gids;
    std::vector<Real> xyz;
    for (LO v = 0; v < work.count(0); ++v) {
      if (!in_use(wv(v))) continue;
      gids.push_back(wv(v));
      xyz.insert(xyz.end(), {wx(v, 0), wx(v, 1), wx(v, 2)});
    }
    const auto work_vertex = lid_table(work, 0, work.count(0));
    for (const NewVertex& nv : added) {
      // Reused vertices the work part holds already.
      const auto held = std::lower_bound(work_vertex.begin(), work_vertex.end(),
                                         std::pair<GO, LO>{nv.gid, std::numeric_limits<LO>::min()});
      if ((held != work_vertex.end() && held->first == nv.gid) || !in_use(nv.gid)) continue;
      const P3 c = place(
          nv,
          [&](GO g) {
            const LO v = find_lid(work_vertex, g);
            return P3{wx(v, 0), wx(v, 1), wx(v, 2)};
          },
          sphere_radius);
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
  if (!polygons.gids.empty()) b.add_polygons(polygons.gids, polygons.offsets, polygons.conn);
  Level out;
  out.mesh = build_local_part(comm, std::move(b));

  // Parents: by gid, among the input's owned cells.
  const auto in_cell = lid_table(mesh, dim, owned);
  const auto fg = out.mesh.gid(dim);
  for (LO c = 0; c < out.mesh.count(dim); ++c) out.parent.push_back(find_lid(in_cell, nums.parent(fg(c))));

  // Fields.
  const auto in_vertex = lid_table(mesh, 0, mesh.count(0));
  for (const FieldInfo& info : mesh.fields().list()) {
    if (info.name.rfind("soma:", 0) == 0 || info.name.rfind("refine:", 0) == 0) continue;
    const bool dense = !info.variable && !info.handle && info.layout == Layout::per_entity() && info.dims.size() == 1 &&
                       (info.dims[0] == 0 || info.dims[0] == dim) && info.rows == mesh.count(info.dims[0]);
    const bool real = info.type == DataType::f32 || info.type == DataType::f64;
    if (!dense || (info.dims[0] == 0 && !real)) {
      out.dropped.push_back(info.name);
      continue;
    }
    const int d = info.dims[0];
    switch (info.type) {
      case DataType::i8: carry<std::int8_t>(mesh, out.mesh, info, d, out.parent, in_vertex, added); break;
      case DataType::u8: carry<std::uint8_t>(mesh, out.mesh, info, d, out.parent, in_vertex, added); break;
      case DataType::i32: carry<std::int32_t>(mesh, out.mesh, info, d, out.parent, in_vertex, added); break;
      case DataType::u32: carry<std::uint32_t>(mesh, out.mesh, info, d, out.parent, in_vertex, added); break;
      case DataType::i64: carry<std::int64_t>(mesh, out.mesh, info, d, out.parent, in_vertex, added); break;
      case DataType::u64: carry<std::uint64_t>(mesh, out.mesh, info, d, out.parent, in_vertex, added); break;
      case DataType::f32: carry<float>(mesh, out.mesh, info, d, out.parent, in_vertex, added); break;
      case DataType::f64: carry<double>(mesh, out.mesh, info, d, out.parent, in_vertex, added); break;
    }
  }
  return out;
}

void split_cell(const HostMesh& work, LO e, GO base, const NodeGids& ids, std::array<std::vector<GO>, 6>& kid_gids,
                std::array<std::vector<GO>, 6>& kid_conn, std::vector<NewVertex>& added) {
  const int dim = work.dim();
  const MeshView<Host> w = work.view();
  const auto wv = work.gid(0);
  const auto wx = work.coords();
  const auto by_gid = [&](LO a, LO b) { return wv(a) < wv(b); };
  const auto midpoint = [&](std::vector<LO> corners) {
    std::sort(corners.begin(), corners.end(), by_gid);
    std::vector<P3> p;
    for (const LO v : corners) p.push_back({wx(v, 0), wx(v, 1), wx(v, 2)});
    return average(p);
  };
  const auto add = [&](GO gid, std::vector<LO> corners) {
    std::sort(corners.begin(), corners.end(), by_gid);
    NewVertex nv{gid, {}, {}};
    for (const LO v : corners) nv.corners.push_back(wv(v));
    added.push_back(std::move(nv));
  };
  const Topology t = w.topology(dim, e);
  const LO n = w.num_vertices(dim, e);
  // The cell's nodes: gids, and corners for the new ones.
  std::vector<GO> node;
  std::vector<std::vector<LO>> corners;
  for (LO i = 0; i < n; ++i) node.push_back(wv(w.vertex(dim, e, i)));
  for (int s = 0; s < topology::side_count(w, dim, e, 1); ++s) {
    const topology::Side side = topology::side_of(w, dim, e, 1, s);
    node.push_back(ids.edge(s));
    corners.push_back({side.v[0], side.v[1]});
    add(node.back(), corners.back());
  }
  if (t == Topology::polygon) {
    std::vector<LO> all;
    for (LO i = 0; i < n; ++i) all.push_back(w.vertex(dim, e, i));
    add(ids.centre, all);
    for (LO i = 0; i < n; ++i) {
      const LO prev = (i + n - 1) % n;
      kid_gids[1].push_back(base + i + 1);
      kid_conn[1].insert(kid_conn[1].end(), {node[static_cast<std::size_t>(i)], node[static_cast<std::size_t>(n + i)],
                                             ids.centre, node[static_cast<std::size_t>(n + prev)]});
    }
    return;
  }
  const Pattern& pat = pattern(t);
  for (int s = 0; s < static_cast<int>(pat.face_node.size()); ++s) {
    if (pat.face_node[static_cast<std::size_t>(s)] < 0) continue;
    const topology::Side side = topology::side_of(w, dim, e, 2, s);
    node.push_back(ids.face(s));
    add(node.back(), {side.v[0], side.v[1], side.v[2], side.v[3]});
  }
  if (pat.centre >= 0) {
    std::vector<LO> all;
    for (LO i = 0; i < n; ++i) all.push_back(w.vertex(dim, e, i));
    node.push_back(ids.centre);
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
      const P3 d =
          midpoint(corners[static_cast<std::size_t>(k)]) - midpoint(corners[static_cast<std::size_t>(opposite[k])]);
      const Real l = d.x * d.x + d.y * d.y + d.z * d.z;
      if (l < shortest) {
        shortest = l;
        best = k;
      }
    }
    for (const Child& c : pat.octahedron[static_cast<std::size_t>(best)]) kids.push_back(&c);
  }
  for (std::size_t k = 0; k < kids.size(); ++k) {
    const auto s = static_cast<std::size_t>(slot(kids[k]->type));
    kid_gids[s].push_back(base + static_cast<GO>(k) + 1);
    for (const int i : kids[k]->nodes) kid_conn[s].push_back(node[static_cast<std::size_t>(i)]);
  }
}

int slot(Topology t) {
  switch (t) {
    case Topology::tri: return 0;
    case Topology::quad: return 1;
    case Topology::tet: return 2;
    case Topology::prism: return 3;
    case Topology::pyramid: return 4;
    default: return 5;
  }
}

int children_of(Topology t, LO corners) {
  return t == Topology::polygon                     ? static_cast<int>(corners)
         : t == Topology::tri || t == Topology::quad ? 4
         : t == Topology::pyramid                    ? 10
                                                     : 8;
}

// Whether a plane cell lies on one side of the lines from its corners'
// average to its sides: each triangle (centre, v_i, v_i+1) turns as the cell
// does (as its area vector, so cells off the xy plane are judged in their own
// plane). Otherwise some of its children would turn against it.
bool star_shaped(const HostMesh& mesh, LO e) {
  const MeshView<Host> m = mesh.view();
  const auto x = mesh.coords();
  const LO n = m.num_vertices(2, e);
  std::vector<P3> p;
  for (LO i = 0; i < n; ++i) {
    const LO v = m.vertex(2, e, i);
    p.push_back({x(v, 0), x(v, 1), x(v, 2)});
  }
  const P3 c = average(p);
  const auto cross = [](const P3& a, const P3& b) {
    return P3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
  };
  std::vector<P3> turns;
  P3 normal;
  for (std::size_t i = 0; i < p.size(); ++i) {
    turns.push_back(cross(p[i] - c, p[(i + 1) % p.size()] - c));
    normal = normal + turns.back();
  }
  return std::all_of(turns.begin(), turns.end(), [&](const P3& t) {
    return t.x * normal.x + t.y * normal.y + t.z * normal.z > 0;
  });
}

// Collective: throws std::invalid_argument on every rank if any rank's mesh
// cannot be refined or the ranks disagree on the levels or the dimension.
void check(const Comm& comm, const HostMesh& mesh, int levels, const char* who, bool shapes) {
  const int dim = mesh.dim();
  int bad = 0;
  if (dim != 2 && dim != 3) {
    bad = 1;
  } else if (mesh.edited() || (comm.size() > 1 && mesh.needs_sync()) || mesh.has_nodes(dim)) {
    bad = 2;
  } else {
    const MeshView<Host> m = mesh.view();
    for (LO e = 0; e < m.count(dim) && bad == 0; ++e) {
      if (!supported_cell(m.topology(dim, e), dim)) bad = 3;
    }
    const LO owned = mesh.segment_table(dim).owned_end;
    for (LO e = 0; shapes && dim == 2 && e < owned && bad == 0; ++e) {
      if (m.topology(2, e) != Topology::tri && !star_shaped(mesh, e)) bad = 5;
    }
  }
  if (levels < 1) bad = std::max(bad, 4);
  std::array<int, 5> agree{bad, levels, -levels, dim, -dim};
  comm.allreduce(std::span<int>(agree), Op::max);
  if (agree[1] != -agree[2]) throw std::invalid_argument(std::string(who) + ": the ranks differ in levels");
  if (agree[3] != -agree[4]) throw std::invalid_argument(std::string(who) + ": the ranks' meshes differ in dimension");
  switch (agree[0]) {
    case 1: throw std::invalid_argument(std::string(who) + ": the mesh is not of dimension 2 or 3");
    case 2: throw std::invalid_argument(std::string(who) + ": a rank's mesh has edits or high-order nodes");
    case 3:
      throw std::invalid_argument(std::string(who) +
                                  ": a rank's mesh holds a cell other than a tri, quad or polygon (2D) or a tet, hex, "
                                  "prism or pyramid (3D)");
    case 4: throw std::invalid_argument(std::string(who) + ": levels must be at least 1");
    case 5:
      throw std::invalid_argument(std::string(who) + ": a rank's mesh holds a quad or polygon that is not star-shaped "
                                  "about its corners' average (its children would turn against it)");
    default: break;
  }
}

}  // namespace soma::detail::refine
