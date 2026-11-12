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
#include <vector>

#include "soma/algorithms/grid_index.hpp"
#include "soma/algorithms/polygon.hpp"
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
// centroid).
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
  return thick && (positive == 0 || negative == 0);
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

RemapPlan remap_plan(const Comm& comm, const Mesh<Host>& source, const Mesh<Host>& target, Real eps) {
  RemapPlan p;
  p.supported = supported_mesh(source) && supported_mesh(target) && source.dim() == target.dim();
  const int dim = target.dim();
  const LO nt = p.supported ? target.segment_table(dim).owned_end : 0;
  const LO ns = p.supported ? source.segment_table(dim).owned_end : 0;
  constexpr Real nan = std::numeric_limits<Real>::quiet_NaN();
  constexpr Real inf = std::numeric_limits<Real>::infinity();
  const auto split = [&](const Mesh<Host>& mesh, LO e, Pieces& out, Real& measure) {
    const bool ok = dim == 2 ? plane_pieces(mesh, e, eps, out, measure) : volume_pieces(mesh, e, eps, out, measure);
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
    box_of(tp, ti, &tbox[6 * ti]);
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
    box_of(spieces, si, box);
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
    box_of(sp, i, box);
    for (std::size_t a = 0; a < 6; ++a) sbox(i, a) = box[a];
  }

  // Candidates for each owned target cell, then their overlaps: the sums
  // over the two cells' pieces, in piece order.
  Kokkos::View<Real**, Kokkos::LayoutLeft, Host> qbox("soma::remap.target_boxes", static_cast<std::size_t>(nt), 6);
  for (std::size_t t = 0; t < static_cast<std::size_t>(nt); ++t) {
    for (std::size_t a = 0; a < 6; ++a) qbox(t, a) = tbox[6 * t + a];
  }
  const GridIndex<Host> index(sbox, eps);
  const auto pairs = index.query(qbox);
  std::vector<Real> area(pairs.boxes.extent(0), 0);
  LO overflow = 0;
  // Piece i of a Pieces as a polygon or a tet.
  const auto polygon = [](const Pieces& q, Offset i, Polygon& out) {
    out.n = 0;
    for (Offset c = q.piece[static_cast<std::size_t>(i)]; c < q.piece[static_cast<std::size_t>(i) + 1]; ++c) {
      out.push(q.xyz[3 * static_cast<std::size_t>(c)], q.xyz[3 * static_cast<std::size_t>(c) + 1]);
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
        for (Offset k = pairs.offsets(t); k < pairs.offsets(t + 1); ++k) {
          const auto s = static_cast<std::size_t>(pairs.boxes(k));
          Real sum = 0;
          for (Offset i = tp.cell[ti]; i < tp.cell[ti + 1]; ++i) {
            if (dim == 2) polygon(tp, i, a);
            for (Offset j = sp.cell[s]; j < sp.cell[s + 1]; ++j) {
              Real piece = 0;
              if (dim == 2) {
                polygon(sp, j, b);
                piece = clip(a, b, eps, out);
              } else {
                piece = overlap(tet(tp, i), tet(sp, j), eps);
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
