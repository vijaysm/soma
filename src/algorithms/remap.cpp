// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/remap.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "soma/algorithms/arborx_index.hpp"
#include "soma/algorithms/convex_clip.hpp"
#include "soma/algorithms/grid_index.hpp"
#include "soma/algorithms/polygon.hpp"
#include "soma/algorithms/sphere_clip.hpp"
#include "soma/algorithms/tet_clip.hpp"
#include "soma/parallel/transport/route.hpp"
#include "soma/topology/sides.hpp"

namespace soma::detail {

namespace {

using Host = Kokkos::HostSpace;
using Exec = Kokkos::DefaultHostExecutionSpace;

bool supported_cell(Topology t, int dim) {
  return dim == 2 ? (t == Topology::tri || t == Topology::quad || t == Topology::polygon)
                  : (t == Topology::tet || t == Topology::hex || t == Topology::prism || t == Topology::pyramid);
}

bool supported_mesh(const Mesh<Host>& mesh) {
  const int dim = mesh.dim();
  if ((dim != 2 && dim != 3) || mesh.has_nodes(dim)) return false;
  const MeshView<Host> m = mesh.view();
  for (LO e = 0; e < m.count(dim); ++e) {
    if (!supported_cell(m.topology(dim, e), dim) || m.num_vertices(dim, e) > polygon_capacity) return false;
  }
  return true;
}

// A cell's convex pieces, as cells -> pieces -> corners (x, y, z each; z 0
// in 2D): polygons in 2D, tets in 3D.
struct Pieces {
  std::vector<Offset> cell{0}, piece{0};
  std::vector<Real> xyz;

  void corner(Real x, Real y, Real z) { xyz.insert(xyz.end(), {x, y, z}); }
  void close_piece() { piece.push_back(static_cast<Offset>(xyz.size() / 3)); }
  void close_cell() { cell.push_back(static_cast<Offset>(piece.size() - 1)); }
  // Drops what was added since the last close_cell().
  void undo_cell() {
    piece.resize(static_cast<std::size_t>(cell.back()) + 1);
    xyz.resize(3 * static_cast<std::size_t>(piece.back()));
  }
  [[nodiscard]] Offset corners(std::size_t c) const {
    return piece[static_cast<std::size_t>(cell[c + 1])] - piece[static_cast<std::size_t>(cell[c])];
  }
};

// Plane cell e's corners counter-clockwise from its least-gid corner; false
// for a cell off a plane z = const or thinner than eps (twice its area over
// its longest side).
bool canonical_polygon(const Mesh<Host>& mesh, LO e, Real eps, Polygon& p) {
  const MeshView<Host> m = mesh.view();
  const auto x = mesh.coords();
  const auto gid = mesh.gid(0);
  const int n = static_cast<int>(m.num_vertices(2, e));
  p.n = 0;
  if (n < 3) return false;
  int least = 0;
  for (int i = 1; i < n; ++i) {
    if (gid(m.vertex(2, e, i)) < gid(m.vertex(2, e, least))) least = i;
  }
  const Real z0 = x(m.vertex(2, e, 0), 2);
  bool flat = true;
  for (int i = 0; i < n; ++i) {
    const LO v = m.vertex(2, e, (least + i) % n);
    flat = flat && std::abs(x(v, 2) - z0) <= eps;
    p.push(x(v, 0), x(v, 1));
  }
  if (twice_area(p) < 0) {
    // The same start, the other way round.
    std::reverse(p.x + 1, p.x + n);
    std::reverse(p.y + 1, p.y + n);
  }
  Real longest = 0;
  for (int i = 0; i < n; ++i) {
    const int j = i + 1 == n ? 0 : i + 1;
    longest = std::max(longest, std::hypot(p.x[j] - p.x[i], p.y[j] - p.y[i]));
  }
  const Real twice = twice_area(p);
  return flat && std::isfinite(twice) && twice >= eps * longest && simple(p);
}

// A plane cell's pieces: itself if convex, else its ear-clipping triangles;
// its area. False where it is bad (crossing itself among them).
bool plane_pieces(const Mesh<Host>& mesh, LO e, Real eps, Pieces& out, Real& measure) {
  Polygon p;
  if (!canonical_polygon(mesh, e, eps, p)) return false;
  measure = twice_area(p) / 2;
  if (convex(p, eps)) {
    for (int i = 0; i < p.n; ++i) out.corner(p.x[i], p.y[i], 0);
    out.close_piece();
    return true;
  }
  int tri[3 * polygon_capacity];
  const int k = triangulate(p, eps, tri);
  for (int t = 0; t < k; ++t) {
    for (int i = 0; i < 3; ++i) out.corner(p.x[tri[3 * t + i]], p.y[tri[3 * t + i]], 0);
    out.close_piece();
  }
  return k == p.n - 2;
}

// The listings of a cell's corners that turn it about itself: a rotation r
// lists corner r[q] as corner q. Closed from two rotations of each type.
std::vector<std::vector<int>> rotations(Topology t) {
  std::vector<std::vector<int>> gens;
  if (t == Topology::hex) gens = {{1, 2, 3, 0, 5, 6, 7, 4}, {3, 2, 6, 7, 0, 1, 5, 4}};
  if (t == Topology::prism) gens = {{1, 2, 0, 4, 5, 3}, {3, 5, 4, 0, 2, 1}};
  if (t == Topology::pyramid) gens = {{1, 2, 3, 0, 4}};
  if (gens.empty()) return {};
  std::vector<int> id(gens[0].size());
  std::iota(id.begin(), id.end(), 0);
  std::vector<std::vector<int>> all{id};
  for (std::size_t i = 0; i < all.size(); ++i) {
    for (const auto& g : gens) {
      std::vector<int> r(g.size());
      for (std::size_t q = 0; q < g.size(); ++q) r[q] = all[i][static_cast<std::size_t>(g[q])];
      if (std::find(all.begin(), all.end(), r) == all.end()) all.push_back(r);
    }
  }
  return all;
}

const std::vector<std::vector<int>>& rotations_of(Topology t) {
  static const std::vector<std::vector<int>> hex = rotations(Topology::hex), prism = rotations(Topology::prism),
                                             pyramid = rotations(Topology::pyramid), none;
  return t == Topology::hex ? hex : t == Topology::prism ? prism : t == Topology::pyramid ? pyramid : none;
}

// A cell on the sphere of radius r: its corners as unit vectors from its
// least-gid corner, counter-clockwise seen from outside; itself if convex,
// else the triangles of its ear clipping in the plane touching the sphere at
// its corners' mean (where great circles are straight lines); its area. eps
// is on the unit sphere. False for a corner off the sphere by more than eps
// r, a corner not in front of that plane, or a cell crossing itself or
// thinner than eps (its area over its longest chord).
bool sphere_pieces(const Mesh<Host>& mesh, LO e, Real eps, Real r, Pieces& out, Real& measure) {
  const MeshView<Host> m = mesh.view();
  const auto x = mesh.coords();
  const auto gid = mesh.gid(0);
  const int n = static_cast<int>(m.num_vertices(2, e));
  if (n < 3) return false;
  int least = 0;
  for (int i = 1; i < n; ++i) {
    if (gid(m.vertex(2, e, i)) < gid(m.vertex(2, e, least))) least = i;
  }
  SpherePolygon s;
  bool on = true;
  for (int i = 0; i < n; ++i) {
    const LO v = m.vertex(2, e, (least + i) % n);
    const Point3 q{x(v, 0), x(v, 1), x(v, 2)};
    on = on && std::abs(std::sqrt(dot(q, q)) - r) <= eps * r;
    s.push(unit(q));
  }
  if (!on) return false;
  if (sphere_area(s) < 0) std::reverse(s.p + 1, s.p + n);  // the same start, the other way round
  Point3 c{};
  for (int i = 0; i < n; ++i) c = {c.x + s.p[i].x, c.y + s.p[i].y, c.z + s.p[i].z};
  c = unit(c);
  const Point3 a = std::abs(c.x) < 0.9 ? Point3{1, 0, 0} : Point3{0, 1, 0};
  const Point3 e1 = unit(cross(c, a)), e2 = cross(c, e1);
  Polygon p;
  Real longest = 0;
  for (int i = 0; i < n; ++i) {
    const Real h = dot(s.p[i], c);
    if (!(h > 0)) return false;
    p.push(dot(s.p[i], e1) / h, dot(s.p[i], e2) / h);
    longest = std::max(longest, chord(s.p[i], s.p[(i + 1) % n]));
  }
  const Real area = sphere_area(s);
  if (!std::isfinite(area) || area < eps * longest || !simple(p)) return false;
  measure = 0;
  if (convex(p, eps)) {
    for (int i = 0; i < n; ++i) out.corner(s.p[i].x, s.p[i].y, s.p[i].z);
    out.close_piece();
    measure = area * r * r;
    return true;
  }
  int tri[3 * polygon_capacity];
  const int k = triangulate(p, eps, tri);
  for (int t = 0; t < k; ++t) {
    for (int i = 0; i < 3; ++i) {
      const Point3& q = s.p[tri[3 * t + i]];
      out.corner(q.x, q.y, q.z);
    }
    out.close_piece();
    measure += sphere_triangle_area(s.p[tri[3 * t]], s.p[tri[3 * t + 1]], s.p[tri[3 * t + 2]]) * r * r;
  }
  return k == n - 2;
}

// A volume cell's tets: a tet itself, its corners in gid order; another cell
// one tet per triangle of its faces, apex its centroid (its corners summed
// in gid order), a quad face cut along the diagonal from its least-gid
// corner, so that two cells cut a face they share alike. Faces start at
// their least-gid corner and go in the order of their corner gids, so that
// the tets do not depend on how the cell's corners are listed. Its volume
// (an inverted cell's, all tets turned the other way, counts as its
// magnitude). False
// where a tet is thinner than eps (below eps times its longest edge squared)
// or turned unlike the others (the cell is not star-shaped about its
// centroid). A pyramid, prism or hex that is convex with flat faces (each
// corner within eps of its faces' planes) is instead one piece, its corners
// listed as the turn of it whose corner gids come least, so that the piece
// does not depend on how the cell's corners are listed; the measure is the
// tets' still.
bool volume_pieces(const Mesh<Host>& mesh, LO e, Real eps, Pieces& out, Real& measure) {
  const MeshView<Host> m = mesh.view();
  const auto x = mesh.coords();
  const auto gid = mesh.gid(0);
  const int n = static_cast<int>(m.num_vertices(3, e));
  const auto at = [&](LO v) { return Point3{x(v, 0), x(v, 1), x(v, 2)}; };
  std::vector<LO> by_gid(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) by_gid[static_cast<std::size_t>(i)] = m.vertex(3, e, i);
  std::sort(by_gid.begin(), by_gid.end(), [&](LO a, LO b) { return gid(a) < gid(b); });
  std::vector<Tet> tets;
  if (m.topology(3, e) == Topology::tet) {
    tets.push_back({{at(by_gid[0]), at(by_gid[1]), at(by_gid[2]), at(by_gid[3])}});
  } else {
    Point3 c{};
    for (const LO v : by_gid) {
      const Point3 p = at(v);
      c = {c.x + p.x, c.y + p.y, c.z + p.z};
    }
    c = {c.x / n, c.y / n, c.z / n};
    // Faces from their least-gid corner, turned as given, in the order of
    // their sorted corner gids.
    struct Face {
      std::array<GO, 4> key{};
      int n = 0;
      std::array<LO, 4> v{};
    };
    std::vector<Face> faces;
    for (int s = 0; s < topology::side_count(m, 3, e, 2); ++s) {
      const topology::Side f = topology::side_of(m, 3, e, 2, s);
      Face face;
      face.n = f.n;
      int k = 0;
      for (int i = 1; i < f.n; ++i) {
        if (gid(f.v[i]) < gid(f.v[k])) k = i;
      }
      for (int i = 0; i < f.n; ++i) {
        face.v[static_cast<std::size_t>(i)] = f.v[(k + i) % f.n];
        face.key[static_cast<std::size_t>(i)] = gid(f.v[i]);
      }
      std::sort(face.key.begin(), face.key.begin() + f.n);
      faces.push_back(face);
    }
    std::sort(faces.begin(), faces.end(), [](const Face& p, const Face& q) { return p.key < q.key; });
    for (const Face& f : faces) {
      tets.push_back({{c, at(f.v[0]), at(f.v[1]), at(f.v[2])}});
      if (f.n == 4) tets.push_back({{c, at(f.v[0]), at(f.v[2]), at(f.v[3])}});
    }
  }
  measure = 0;
  int positive = 0, negative = 0;
  bool thick = true;
  for (const Tet& t : tets) {
    const Real six = six_volume(t);
    const Real l = longest_edge(t);
    (six > 0 ? positive : negative) += 1;
    thick = thick && std::isfinite(six) && std::abs(six) / 6 >= eps * l * l;
    measure += std::abs(six) / 6;
    for (const Point3& p : t.p) out.corner(p.x, p.y, p.z);
    out.close_piece();
  }
  const bool ok = thick && (positive == 0 || negative == 0);
  const auto& turns = rotations_of(m.topology(3, e));
  if (!ok || turns.empty()) return ok;
  const auto gid_less = [&](int a, int b) { return gid(m.vertex(3, e, a)) < gid(m.vertex(3, e, b)); };
  const std::vector<int>* least = &turns.front();
  for (const auto& r : turns) {
    if (std::lexicographical_compare(r.begin(), r.end(), least->begin(), least->end(), gid_less)) least = &r;
  }
  Point3 corner[8];
  for (int q = 0; q < n; ++q) corner[q] = at(m.vertex(3, e, (*least)[static_cast<std::size_t>(q)]));
  if (!is_convex(corner, n, eps)) return ok;
  out.undo_cell();
  for (int q = 0; q < n; ++q) out.corner(corner[q].x, corner[q].y, corner[q].z);
  out.close_piece();
  return ok;
}

void box_of(const Pieces& p, std::size_t c, Real* box) {
  for (int a = 0; a < 3; ++a) {
    box[a] = std::numeric_limits<Real>::infinity();
    box[a + 3] = -std::numeric_limits<Real>::infinity();
  }
  for (Offset k = p.piece[static_cast<std::size_t>(p.cell[c])]; k < p.piece[static_cast<std::size_t>(p.cell[c + 1])]; ++k) {
    for (std::size_t a = 0; a < 3; ++a) {
      box[a] = std::min(box[a], p.xyz[3 * static_cast<std::size_t>(k) + a]);
      box[a + 3] = std::max(box[a + 3], p.xyz[3 * static_cast<std::size_t>(k) + a]);
    }
  }
}

}  // namespace

// The box of the cap that holds a cell on the unit sphere: the corners' mean
// direction c and the widest angle a from it to a corner; on each axis the
// cap reaches 1 where the axis is within a of c, else cos(angle - a). So the
// cell's arcs and inside, not only its corners, are in the box.
void cap_box(const Pieces& p, std::size_t c, Real* box) {
  box_of(p, c, box);
  Point3 sum{};
  const auto first = static_cast<std::size_t>(p.piece[static_cast<std::size_t>(p.cell[c])]);
  const auto last = static_cast<std::size_t>(p.piece[static_cast<std::size_t>(p.cell[c + 1])]);
  const auto at = [&](std::size_t k) { return Point3{p.xyz[3 * k], p.xyz[3 * k + 1], p.xyz[3 * k + 2]}; };
  for (std::size_t k = first; k < last; ++k) sum = {sum.x + at(k).x, sum.y + at(k).y, sum.z + at(k).z};
  const Point3 d = unit(sum);
  Real lowest = 1;
  for (std::size_t k = first; k < last; ++k) lowest = std::min(lowest, dot(d, at(k)));
  const Real a = std::acos(std::clamp<Real>(lowest, -1, 1));
  const Real dc[3] = {d.x, d.y, d.z};
  for (std::size_t k = 0; k < 3; ++k) {
    const Real up = std::acos(std::clamp<Real>(dc[k], -1, 1)), down = std::acos(std::clamp<Real>(-dc[k], -1, 1));
    box[k + 3] = std::max(box[k + 3], up <= a ? Real{1} : std::cos(up - a));
    box[k] = std::min(box[k], down <= a ? Real{-1} : -std::cos(down - a));
  }
}

RemapPlan remap_plan(const Comm& comm, const Mesh<Host>& source, const Mesh<Host>& target,
                     const RemapOptions& options) {
  RemapPlan p;
  p.supported = supported_mesh(source) && supported_mesh(target) && source.dim() == target.dim() &&
                (options.surface == Surface::plane || target.dim() == 2);
  const int dim = target.dim();
  const bool sphere = options.surface == Surface::sphere;
  // On a sphere: its radius, and every length on the unit sphere.
  Real r = options.radius;
  if (sphere && !(r > 0)) {
    r = 0;
    for (const Mesh<Host>* mesh : {&source, &target}) {
      const auto x = mesh->coords();
      for (LO v = 0; v < mesh->count(0); ++v) {
        r = std::max(r, std::sqrt(x(v, 0) * x(v, 0) + x(v, 1) * x(v, 1) + x(v, 2) * x(v, 2)));
      }
    }
    r = comm.allreduce(r, Op::max);
  }
  p.radius = sphere ? r : 0;
  const Real eps = sphere ? options.merge_tolerance / r : options.merge_tolerance;
  const SphereClip method = options.sphere_clip;
  const LO nt = p.supported ? target.segment_table(dim).owned_end : 0;
  const LO ns = p.supported ? source.segment_table(dim).owned_end : 0;
  constexpr Real nan = std::numeric_limits<Real>::quiet_NaN();
  constexpr Real inf = std::numeric_limits<Real>::infinity();
  const auto split = [&](const Mesh<Host>& mesh, LO e, Pieces& out, Real& measure) {
    const bool ok = sphere     ? sphere_pieces(mesh, e, eps, r, out, measure)
                    : dim == 2 ? plane_pieces(mesh, e, eps, out, measure)
                               : volume_pieces(mesh, e, eps, out, measure);
    if (!ok) out.undo_cell();
    out.close_cell();
    return ok;
  };

  // The owned target cells: pieces, boxes (NaN for a bad one, so that it
  // meets nothing), measures; and their span on this rank.
  Pieces tp;
  std::vector<Real> tbox(6 * static_cast<std::size_t>(nt), nan);
  p.target_area.assign(static_cast<std::size_t>(nt), 1);
  Real span[6] = {inf, inf, inf, -inf, -inf, -inf};
  for (LO t = 0; t < nt; ++t) {
    const auto ti = static_cast<std::size_t>(t);
    if (!split(target, t, tp, p.target_area[ti])) {
      ++p.bad_target;
      p.target_area[ti] = 1;
      continue;
    }
    (sphere ? cap_box : box_of)(tp, ti, &tbox[6 * ti]);
    for (std::size_t a = 0; a < 3; ++a) {
      span[a] = std::min(span[a], tbox[6 * ti + a]);
      span[a + 3] = std::max(span[a + 3], tbox[6 * ti + a + 3]);
    }
  }

  // The owned source cells, and the most pieces and corners any has.
  Pieces spieces;
  std::vector<char> sgood(static_cast<std::size_t>(ns), 0);
  std::int64_t most[2] = {1, 3};
  for (LO s = 0; s < ns; ++s) {
    const auto si = static_cast<std::size_t>(s);
    Real measure = 0;
    if (split(source, s, spieces, measure)) {
      sgood[si] = 1;
      most[0] = std::max<std::int64_t>(most[0], spieces.cell[si + 1] - spieces.cell[si]);
      most[1] = std::max<std::int64_t>(most[1], spieces.corners(si));
    } else {
      ++p.bad_source;
    }
  }
  comm.allreduce(std::span<std::int64_t>(most), Op::max);

  // Every rank's target span; each good source cell, as its pieces, to every
  // rank whose span its box, widened by eps, meets.
  const std::vector<Real> spans = comm.allgatherv(std::span<const Real>(span, 6));
  const auto ranks = static_cast<std::size_t>(comm.size());
  const auto mp = static_cast<std::size_t>(most[0]), mc = static_cast<std::size_t>(most[1]);
  // gid, piece count, corners per piece, then x, y, z per corner.
  const std::size_t width = sizeof(GO) + sizeof(std::int64_t) * (1 + mp) + 3 * mc * sizeof(Real);
  std::vector<int> dest;
  for (LO s = 0; s < ns; ++s) {
    const auto si = static_cast<std::size_t>(s);
    if (!sgood[si]) continue;
    Real box[6];
    (sphere ? cap_box : box_of)(spieces, si, box);
    for (std::size_t r = 0; r < ranks; ++r) {
      const Real* q = &spans[6 * r];
      bool meets = true;
      for (std::size_t a = 0; a < 3; ++a) meets = meets && box[a] <= q[a + 3] + eps && q[a] <= box[a + 3] + eps;
      if (meets) {
        dest.push_back(static_cast<int>(r));
        p.sent_cell.push_back(s);
      }
    }
  }
  transport::RecordBatch<Host> sent(dest.size(), width, "soma::remap.sent");
  const auto sgid = source.gid(dim);
  std::vector<std::int64_t> counts(1 + mp);
  std::vector<Real> xyz(3 * mc);
  for (std::size_t i = 0; i < dest.size(); ++i) {
    sent.dest(i) = dest[i];
    const auto s = static_cast<std::size_t>(p.sent_cell[i]);
    std::fill(counts.begin(), counts.end(), 0);
    std::fill(xyz.begin(), xyz.end(), 0);
    counts[0] = spieces.cell[s + 1] - spieces.cell[s];
    for (Offset k = spieces.cell[s]; k < spieces.cell[s + 1]; ++k) {
      const auto ki = static_cast<std::size_t>(k);
      counts[1 + ki - static_cast<std::size_t>(spieces.cell[s])] = spieces.piece[ki + 1] - spieces.piece[ki];
    }
    const auto first = 3 * static_cast<std::size_t>(spieces.piece[static_cast<std::size_t>(spieces.cell[s])]);
    std::copy(spieces.xyz.begin() + static_cast<std::ptrdiff_t>(first),
              spieces.xyz.begin() + static_cast<std::ptrdiff_t>(first + 3 * static_cast<std::size_t>(spieces.corners(s))),
              xyz.begin());
    std::byte* row = &sent.payload(i, 0);
    const GO g = sgid(p.sent_cell[i]);
    std::memcpy(row, &g, sizeof(GO));
    std::memcpy(row + sizeof(GO), counts.data(), counts.size() * sizeof(std::int64_t));
    std::memcpy(row + sizeof(GO) + counts.size() * sizeof(std::int64_t), xyz.data(), xyz.size() * sizeof(Real));
  }
  const transport::RecordBatch<Host> arrived = transport::route(Exec{}, comm, sent);
  p.plan = transport::learn_plan(comm, sent, arrived);

  // The arrived source cells: gids, pieces (as their owners split them), boxes.
  const std::size_t na = arrived.size();
  Pieces sp;
  p.source_gid.resize(na);
  Kokkos::View<Real**, Kokkos::LayoutLeft, Host> sbox("soma::remap.source_boxes", na, 6);
  for (std::size_t i = 0; i < na; ++i) {
    const std::byte* row = &arrived.payload(i, 0);
    std::memcpy(&p.source_gid[i], row, sizeof(GO));
    std::memcpy(counts.data(), row + sizeof(GO), counts.size() * sizeof(std::int64_t));
    std::memcpy(xyz.data(), row + sizeof(GO) + counts.size() * sizeof(std::int64_t), xyz.size() * sizeof(Real));
    std::size_t c = 0;
    for (std::int64_t k = 0; k < counts[0]; ++k) {
      for (std::int64_t j = 0; j < counts[1 + static_cast<std::size_t>(k)]; ++j, ++c) {
        sp.corner(xyz[3 * c], xyz[3 * c + 1], xyz[3 * c + 2]);
      }
      sp.close_piece();
    }
    sp.close_cell();
    Real box[6];
    (sphere ? cap_box : box_of)(sp, i, box);
    for (std::size_t a = 0; a < 6; ++a) sbox(i, a) = box[a];
  }

  // Candidates for each owned target cell, then their overlaps: the sums
  // over the two cells' pieces, in piece order.
  Kokkos::View<Real**, Kokkos::LayoutLeft, Host> qbox("soma::remap.target_boxes", static_cast<std::size_t>(nt), 6);
  for (std::size_t t = 0; t < static_cast<std::size_t>(nt); ++t) {
    for (std::size_t a = 0; a < 6; ++a) qbox(t, a) = tbox[6 * t + a];
  }
  const auto pairs = [&] {
#if SOMA_ENABLE_ARBORX
    if (options.search == SpatialSearch::arborx) return ArborXIndex<Host>(sbox, eps).query(qbox);
#endif
    if (options.search != SpatialSearch::grid) throw std::invalid_argument("remap_plan: ArborX's search is not built");
    return GridIndex<Host>(sbox, eps).query(qbox);
  }();
  std::vector<Real> area(pairs.boxes.extent(0), 0);
  LO overflow = 0;
  // Piece i of a Pieces as a polygon or a tet.
  const auto polygon = [](const Pieces& q, Offset i, Polygon& out) {
    out.n = 0;
    for (Offset c = q.piece[static_cast<std::size_t>(i)]; c < q.piece[static_cast<std::size_t>(i) + 1]; ++c) {
      out.push(q.xyz[3 * static_cast<std::size_t>(c)], q.xyz[3 * static_cast<std::size_t>(c) + 1]);
    }
  };
  // Piece i's corners (up to 8), their count returned.
  const auto corners = [](const Pieces& q, Offset i, Point3* out) {
    const auto first = static_cast<std::size_t>(q.piece[static_cast<std::size_t>(i)]);
    const auto n = static_cast<int>(q.piece[static_cast<std::size_t>(i) + 1] - q.piece[static_cast<std::size_t>(i)]);
    for (std::size_t k = 0; k < static_cast<std::size_t>(n); ++k) {
      out[k] = {q.xyz[3 * (first + k)], q.xyz[3 * (first + k) + 1], q.xyz[3 * (first + k) + 2]};
    }
    return n;
  };
  const auto on_sphere = [](const Pieces& q, Offset i, SpherePolygon& out) {
    out.n = 0;
    for (Offset c = q.piece[static_cast<std::size_t>(i)]; c < q.piece[static_cast<std::size_t>(i) + 1]; ++c) {
      const auto k = 3 * static_cast<std::size_t>(c);
      out.push({q.xyz[k], q.xyz[k + 1], q.xyz[k + 2]});
    }
  };
  const auto tet = [](const Pieces& q, Offset i) {
    Tet out;
    const auto c = 3 * static_cast<std::size_t>(q.piece[static_cast<std::size_t>(i)]);
    for (std::size_t k = 0; k < 4; ++k) out.p[k] = {q.xyz[c + 3 * k], q.xyz[c + 3 * k + 1], q.xyz[c + 3 * k + 2]};
    return out;
  };
  Kokkos::parallel_reduce(
      "soma::remap.clip", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, nt),
      [&](const LO t, LO& over) {
        const auto ti = static_cast<std::size_t>(t);
        Polygon a, b, out;
        SpherePolygon sa, sb, sout;
        for (Offset k = pairs.offsets(t); k < pairs.offsets(t + 1); ++k) {
          const auto s = static_cast<std::size_t>(pairs.boxes(k));
          Real sum = 0;
          for (Offset i = tp.cell[ti]; i < tp.cell[ti + 1]; ++i) {
            if (sphere) {
              on_sphere(tp, i, sa);
            } else if (dim == 2) {
              polygon(tp, i, a);
            }
            for (Offset j = sp.cell[s]; j < sp.cell[s + 1]; ++j) {
              Real piece = 0;
              if (sphere) {
                on_sphere(sp, j, sb);
                piece = method == SphereClip::gnomonic ? clip_gnomonic(sa, sb, eps, sout)
                                                       : clip_great_circle(sa, sb, eps, sout);
                piece = piece < 0 ? piece : piece * r * r;
              } else if (dim == 2) {
                polygon(sp, j, b);
                piece = clip(a, b, eps, out);
              } else if (tp.piece[static_cast<std::size_t>(i) + 1] - tp.piece[static_cast<std::size_t>(i)] == 4 &&
                         sp.piece[static_cast<std::size_t>(j) + 1] - sp.piece[static_cast<std::size_t>(j)] == 4) {
                piece = overlap(tet(tp, i), tet(sp, j), eps);
              } else {
                Point3 tc[8], sc[8];
                const int ntc = corners(tp, i, tc), nsc = corners(sp, j, sc);
                piece = convex_overlap(tc, ntc, sc, nsc, eps);
              }
              if (piece < 0) {
                ++over;
              } else {
                sum += piece;
              }
            }
          }
          area[static_cast<std::size_t>(k)] = sum;
        }
      },
      overflow);
  p.overflow = overflow;

  // Rows of the overlaps that are something, ordered by source gid.
  p.offsets.assign(static_cast<std::size_t>(nt) + 1, 0);
  std::vector<std::pair<GO, LO>> row;
  for (LO t = 0; t < nt; ++t) {
    row.clear();
    for (Offset k = pairs.offsets(t); k < pairs.offsets(t + 1); ++k) {
      if (area[static_cast<std::size_t>(k)] > 0) {
        const LO s = pairs.boxes(k);
        row.emplace_back(p.source_gid[static_cast<std::size_t>(s)], static_cast<LO>(k));
      }
    }
    std::sort(row.begin(), row.end());
    for (const auto& [g, k] : row) {
      p.columns.push_back(pairs.boxes(k));
      p.overlap.push_back(area[static_cast<std::size_t>(k)]);
    }
    p.offsets[static_cast<std::size_t>(t) + 1] = static_cast<Offset>(p.columns.size());
  }
  return p;
}

}  // namespace soma::detail
