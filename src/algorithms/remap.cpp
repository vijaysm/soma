// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/remap.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "soma/algorithms/grid_index.hpp"
#include "soma/algorithms/polygon.hpp"
#include "soma/parallel/transport/route.hpp"

namespace soma::detail {

namespace {

using Host = Kokkos::HostSpace;
using Exec = Kokkos::DefaultHostExecutionSpace;

bool supported_mesh(const Mesh<Host>& mesh) {
  if (mesh.dim() != 2 || mesh.has_nodes(2)) return false;
  const MeshView<Host> m = mesh.view();
  for (LO e = 0; e < m.count(2); ++e) {
    const Topology t = m.topology(2, e);
    if (t != Topology::tri && t != Topology::quad && t != Topology::polygon) return false;
    if (m.num_vertices(2, e) > polygon_capacity) return false;
  }
  return true;
}

// Cell e's corners counter-clockwise from its least-gid corner; false for a
// cell off a plane z = const or thinner than eps (twice its area over its
// longest side).
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
  return flat && std::isfinite(twice) && twice >= eps * longest;
}

// A polygon's convex pieces: itself if convex, else its ear-clipping
// triangles. Appended as cells -> pieces -> corners; false where it crosses
// itself (no ears, or triangles not adding up to its area).
struct Pieces {
  std::vector<Offset> cell{0}, piece{0};
  std::vector<Real> xy;

  void add(const Polygon& p, const int* corners, int n) {
    for (int i = 0; i < n; ++i) {
      xy.push_back(p.x[corners[i]]);
      xy.push_back(p.y[corners[i]]);
    }
    piece.push_back(static_cast<Offset>(xy.size() / 2));
  }
  bool add_cell(const Polygon& p, Real eps) {
    bool ok = true;
    if (convex(p, eps)) {
      int all[polygon_capacity];
      std::iota(all, all + p.n, 0);
      add(p, all, p.n);
    } else {
      int tri[3 * polygon_capacity];
      const int k = triangulate(p, eps, tri);
      Real sum = 0;
      for (int t = 0; t < k; ++t) {
        Polygon piece;
        for (int i = 0; i < 3; ++i) piece.push(p.x[tri[3 * t + i]], p.y[tri[3 * t + i]]);
        sum += twice_area(piece);
        add(p, tri + 3 * t, 3);
      }
      const Real area = twice_area(p);
      ok = k > 0 && std::abs(sum - area) <= 1e-10 * area;
    }
    cell.push_back(static_cast<Offset>(piece.size() - 1));
    return ok;
  }
};

void box_of(const Polygon& p, Real* box) {
  box[0] = box[1] = std::numeric_limits<Real>::infinity();
  box[3] = box[4] = -std::numeric_limits<Real>::infinity();
  box[2] = box[5] = 0;
  for (int i = 0; i < p.n; ++i) {
    box[0] = std::min(box[0], p.x[i]);
    box[1] = std::min(box[1], p.y[i]);
    box[3] = std::max(box[3], p.x[i]);
    box[4] = std::max(box[4], p.y[i]);
  }
}

}  // namespace

RemapPlan remap_plan(const Comm& comm, const Mesh<Host>& source, const Mesh<Host>& target, Real eps) {
  RemapPlan p;
  p.supported = supported_mesh(source) && supported_mesh(target);
  const LO nt = p.supported ? target.segment_table(2).owned_end : 0;
  const LO ns = p.supported ? source.segment_table(2).owned_end : 0;
  constexpr Real nan = std::numeric_limits<Real>::quiet_NaN();

  // The owned target cells: polygons, pieces, boxes (NaN for a bad one, so
  // that it meets nothing), areas; and their span on this rank.
  Pieces tp;
  std::vector<Real> tbox(6 * static_cast<std::size_t>(nt), nan);
  p.target_area.assign(static_cast<std::size_t>(nt), 1);
  Real span[6] = {std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity(), 0,
                  -std::numeric_limits<Real>::infinity(), -std::numeric_limits<Real>::infinity(), 0};
  for (LO t = 0; t < nt; ++t) {
    Polygon poly;
    const auto ti = static_cast<std::size_t>(t);
    if (!canonical_polygon(target, t, eps, poly) || !tp.add_cell(poly, eps)) {
      ++p.bad_target;
      if (tp.cell.size() < ti + 2) tp.cell.push_back(tp.cell.back());
      continue;
    }
    p.target_area[ti] = twice_area(poly) / 2;
    box_of(poly, &tbox[6 * ti]);
    for (int a = 0; a < 2; ++a) {
      span[a] = std::min(span[a], tbox[6 * ti + static_cast<std::size_t>(a)]);
      span[a + 3] = std::max(span[a + 3], tbox[6 * ti + static_cast<std::size_t>(a) + 3]);
    }
  }

  // The owned source cells, and the most corners any has.
  std::vector<Polygon> spoly(static_cast<std::size_t>(ns));
  std::vector<char> sgood(static_cast<std::size_t>(ns), 0);
  std::int64_t corners = 3;
  for (LO s = 0; s < ns; ++s) {
    const auto si = static_cast<std::size_t>(s);
    Pieces check;
    if (canonical_polygon(source, s, eps, spoly[si]) && check.add_cell(spoly[si], eps)) {
      sgood[si] = 1;
      corners = std::max<std::int64_t>(corners, spoly[si].n);
    } else {
      ++p.bad_source;
    }
  }
  corners = comm.allreduce(corners, Op::max);

  // Every rank's target span; each good source cell to every rank whose span
  // its box, widened by eps, meets.
  const std::vector<Real> spans = comm.allgatherv(std::span<const Real>(span, 6));
  const auto ranks = static_cast<std::size_t>(comm.size());
  const auto nc = static_cast<std::size_t>(corners);
  const std::size_t width = sizeof(GO) + sizeof(std::int64_t) + 2 * nc * sizeof(Real);
  std::vector<int> dest;
  for (LO s = 0; s < ns; ++s) {
    const auto si = static_cast<std::size_t>(s);
    if (!sgood[si]) continue;
    Real box[6];
    box_of(spoly[si], box);
    for (std::size_t r = 0; r < ranks; ++r) {
      const Real* q = &spans[6 * r];
      if (box[0] <= q[3] + eps && q[0] <= box[3] + eps && box[1] <= q[4] + eps && q[1] <= box[4] + eps) {
        dest.push_back(static_cast<int>(r));
        p.sent_cell.push_back(s);
      }
    }
  }
  transport::RecordBatch<Host> sent(dest.size(), width, "soma::remap.sent");
  const auto sgid = source.gid(2);
  for (std::size_t i = 0; i < dest.size(); ++i) {
    sent.dest(i) = dest[i];
    const Polygon& poly = spoly[static_cast<std::size_t>(p.sent_cell[i])];
    std::byte* row = &sent.payload(i, 0);
    const GO g = sgid(p.sent_cell[i]);
    const std::int64_t n = poly.n;
    std::memcpy(row, &g, sizeof(GO));
    std::memcpy(row + sizeof(GO), &n, sizeof(n));
    std::vector<Real> xy(2 * nc, 0);
    for (int k = 0; k < poly.n; ++k) {
      xy[2 * static_cast<std::size_t>(k)] = poly.x[k];
      xy[2 * static_cast<std::size_t>(k) + 1] = poly.y[k];
    }
    std::memcpy(row + sizeof(GO) + sizeof(n), xy.data(), xy.size() * sizeof(Real));
  }
  const transport::RecordBatch<Host> arrived = transport::route(Exec{}, comm, sent);
  p.plan = transport::learn_plan(comm, sent, arrived);

  // The arrived source cells: gids, pieces, boxes.
  const std::size_t na = arrived.size();
  Pieces sp;
  p.source_gid.resize(na);
  Kokkos::View<Real**, Kokkos::LayoutLeft, Host> sbox("soma::remap.source_boxes", na, 6);
  for (std::size_t i = 0; i < na; ++i) {
    const std::byte* row = &arrived.payload(i, 0);
    std::int64_t n = 0;
    std::memcpy(&p.source_gid[i], row, sizeof(GO));
    std::memcpy(&n, row + sizeof(GO), sizeof(n));
    Polygon poly;
    for (std::int64_t k = 0; k < n; ++k) {
      Real xy[2];
      std::memcpy(xy, row + sizeof(GO) + sizeof(n) + 2 * static_cast<std::size_t>(k) * sizeof(Real), sizeof(xy));
      poly.push(xy[0], xy[1]);
    }
    sp.add_cell(poly, eps);  // as its owner split it
    Real box[6];
    box_of(poly, box);
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
  Kokkos::parallel_for(
      "soma::remap.clip", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, nt), [&](const LO t) {
        const auto ti = static_cast<std::size_t>(t);
        Polygon a, b, out;
        for (Offset k = pairs.offsets(t); k < pairs.offsets(t + 1); ++k) {
          const auto s = static_cast<std::size_t>(pairs.boxes(k));
          Real sum = 0;
          for (Offset i = tp.cell[ti]; i < tp.cell[ti + 1]; ++i) {
            a.n = 0;
            for (Offset c = tp.piece[static_cast<std::size_t>(i)]; c < tp.piece[static_cast<std::size_t>(i) + 1]; ++c) {
              a.push(tp.xy[2 * static_cast<std::size_t>(c)], tp.xy[2 * static_cast<std::size_t>(c) + 1]);
            }
            for (Offset j = sp.cell[s]; j < sp.cell[s + 1]; ++j) {
              b.n = 0;
              for (Offset c = sp.piece[static_cast<std::size_t>(j)]; c < sp.piece[static_cast<std::size_t>(j) + 1]; ++c) {
                b.push(sp.xy[2 * static_cast<std::size_t>(c)], sp.xy[2 * static_cast<std::size_t>(c) + 1]);
              }
              sum += clip(a, b, eps, out);
            }
          }
          area[static_cast<std::size_t>(k)] = sum;
        }
      });
  Exec{}.fence();

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
