// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_CONVEX_CLIP_HPP_
#define SOMA_ALGORITHMS_CONVEX_CLIP_HPP_

/// \file
/// The overlap of two convex cells (tets, pyramids, prisms, hexes with flat
/// faces), for the remap of volume meshes: one cell, as a polyhedron (its
/// faces as polygons), cut by the face planes of the other, and the volume of
/// what is left. Callable in kernels; fixed capacity, no allocation.
///
/// A cut keeps each face's part on the plane's inner side and closes the
/// polyhedron with a cap, the polygon of the points where the faces cross the
/// plane. As in tet_clip.hpp, a corner outside by at most eps (the merge
/// tolerance, a length) is moved onto the plane and kept, so that of two cells
/// sharing a face one, not both, takes the band between; a crossing is found
/// from the edge's inner end, so that the two faces that share an edge find
/// the same point. The volume is summed over tets from the polyhedron's
/// corner average to each face's fan of triangles, and the whole overlap is
/// dropped when it is thinner than eps: below eps L^2, L its longest edge.
///
/// A cell's face planes start at its face's least corner (by x, then y, then
/// z) and go towards the lesser of that corner's two neighbours, so that two
/// cells sharing a face find planes that differ only in sign. Both cells are
/// first moved so that the clipping cell's first corner is the origin.

#include <Kokkos_Core.hpp>

#include "soma/algorithms/tet_clip.hpp"
#include "soma/core/topology.hpp"
#include "soma/topology/canonical.hpp"

namespace soma::detail {

/// Faces a cut polyhedron may hold, and corners per face: a cell's faces (at
/// most 6) and one cap per cut (at most 6), each cut adding at most one
/// corner to a face.
inline constexpr int convex_faces = 16;
inline constexpr int convex_corners = 16;

struct Polyhedron {
  int faces = 0;
  int n[convex_faces] = {};
  Point3 p[convex_faces][convex_corners];
};

/// The cell with `corners` corners (4 tet, 5 pyramid, 6 prism, 8 hex).
KOKKOS_INLINE_FUNCTION Topology convex_topology(int corners) {
  return corners == 4 ? Topology::tet
         : corners == 5 ? Topology::pyramid
         : corners == 6 ? Topology::prism
         : corners == 8 ? Topology::hex
                        : Topology::vertex;
}

KOKKOS_INLINE_FUNCTION bool point_less(const Point3& a, const Point3& b) {
  if (a.x != b.x) return a.x < b.x;
  if (a.y != b.y) return a.y < b.y;
  return a.z < b.z;
}

/// The plane of a face with n corners (3 or 4) of a cell whose corners average
/// to g, its inside g's: Newell's normal over the corners from the least one
/// towards its lesser neighbour. False for a face of no area.
KOKKOS_INLINE_FUNCTION bool convex_face_plane(const Point3* q, int n, const Point3& g, Plane& out) {
  int k = 0;
  for (int i = 1; i < n; ++i) {
    if (point_less(q[i], q[k])) k = i;
  }
  const int step = point_less(q[(k + 1) % n], q[(k + n - 1) % n]) ? 1 : n - 1;
  Point3 r[4];
  for (int i = 0; i < n; ++i) r[i] = q[(k + i * step) % n];
  Point3 nn{};
  for (int i = 0; i < n; ++i) {
    const Point3& a = r[i];
    const Point3& b = r[(i + 1) % n];
    nn.x += (a.y - b.y) * (a.z + b.z);
    nn.y += (a.z - b.z) * (a.x + b.x);
    nn.z += (a.x - b.x) * (a.y + b.y);
  }
  const Real len = Kokkos::sqrt(dot(nn, nn));
  if (!(len > 0)) return false;
  nn = {nn.x / len, nn.y / len, nn.z / len};
  Real c = dot(nn, r[0]);
  if (dot(nn, g) - c > 0) {
    nn = {-nn.x, -nn.y, -nn.z};
    c = -c;
  }
  out.n = nn;
  out.c = c;
  return true;
}

/// The corners of face s of a cell of topology t, given its corners.
KOKKOS_INLINE_FUNCTION int convex_face(Topology t, int s, const Point3* corner, Point3* out) {
  const canonical::SideDef f = canonical::side(t, 2, s);
  for (int i = 0; i < f.num_vertices; ++i) out[i] = corner[f.v[i]];
  return f.num_vertices;
}

KOKKOS_INLINE_FUNCTION Point3 average(const Point3* p, int n) {
  Point3 g{};
  for (int i = 0; i < n; ++i) g = {g.x + p[i].x, g.y + p[i].y, g.z + p[i].z};
  return {g.x / n, g.y / n, g.z / n};
}

/// The face planes of a convex cell (count returned; 0 for a face of no area).
KOKKOS_INLINE_FUNCTION int convex_planes(const Point3* corner, int corners, Plane* out) {
  const Topology t = convex_topology(corners);
  const Point3 g = average(corner, corners);
  const int nf = canonical::num_sides(t, 2);
  for (int s = 0; s < nf; ++s) {
    Point3 q[4];
    const int n = convex_face(t, s, corner, q);
    if (!convex_face_plane(q, n, g, out[s])) return 0;
  }
  return nf;
}

/// Whether a cell's faces are flat and it is convex: each corner within eps of
/// its faces' planes, and no corner outside a face's plane by more than eps.
KOKKOS_INLINE_FUNCTION bool is_convex(const Point3* corner, int corners, Real eps) {
  if (convex_topology(corners) == Topology::vertex) return false;
  Point3 q[8];
  for (int i = 0; i < corners; ++i) q[i] = minus(corner[i], corner[0]);
  Plane h[6];
  const int nf = convex_planes(q, corners, h);
  if (nf == 0) return false;
  const Topology t = convex_topology(corners);
  for (int s = 0; s < nf; ++s) {
    const canonical::SideDef f = canonical::side(t, 2, s);
    for (int i = 0; i < f.num_vertices; ++i) {
      if (Kokkos::fabs(dot(h[s].n, q[f.v[i]]) - h[s].c) > eps) return false;
    }
    for (int i = 0; i < corners; ++i) {
      if (dot(h[s].n, q[i]) - h[s].c > eps) return false;
    }
  }
  return true;
}

/// p cut by plane h, into out; false where out would overflow. A cut that
/// takes nothing away keeps p's faces (those on the plane would else come
/// back as the cap too).
KOKKOS_INLINE_FUNCTION bool cut_convex(const Polyhedron& p, const Plane& h, Real eps, Polyhedron& out) {
  out.faces = 0;
  Point3 cap[convex_corners * 2];
  int nc = 0;
  bool room = true;
  const auto at = [&](const Point3& x, Real& d) {
    d = dot(h.n, x) - h.c;
    if (d > 0 && d <= eps) {
      const Point3 y{x.x - d * h.n.x, x.y - d * h.n.y, x.z - d * h.n.z};
      d = 0;
      return y;
    }
    return x;
  };
  const auto to_cap = [&](const Point3& x) {
    for (int i = 0; i < nc; ++i) {
      if (cap[i].x == x.x && cap[i].y == x.y && cap[i].z == x.z) return;
    }
    if (nc < convex_corners * 2) {
      cap[nc++] = x;
    } else {
      room = false;
    }
  };
  bool beyond = false;
  for (int f = 0; f < p.faces && !beyond; ++f) {
    for (int i = 0; i < p.n[f] && !beyond; ++i) {
      Real d = 0;
      at(p.p[f][i], d);
      beyond = d > 0;
    }
  }
  if (!beyond) {
    out = p;
    return true;
  }
  for (int f = 0; f < p.faces; ++f) {
    Point3 q[convex_corners];
    Real d[convex_corners];
    for (int i = 0; i < p.n[f]; ++i) q[i] = at(p.p[f][i], d[i]);
    if (out.faces == convex_faces) return false;
    Point3* kept = out.p[out.faces];
    int nk = 0;
    for (int i = 0; i < p.n[f] && room; ++i) {
      const int j = (i + 1) % p.n[f];
      if (d[i] <= 0) {
        if (nk == convex_corners) return false;
        kept[nk++] = q[i];
        if (d[i] == 0) to_cap(q[i]);
      }
      if ((d[i] < 0 && d[j] > 0) || (d[i] > 0 && d[j] < 0)) {
        const int a = d[i] < 0 ? i : j, b = d[i] < 0 ? j : i;  // inner end first
        const Real s = d[a] / (d[a] - d[b]);
        const Point3 x{q[a].x + s * (q[b].x - q[a].x), q[a].y + s * (q[b].y - q[a].y), q[a].z + s * (q[b].z - q[a].z)};
        if (nk == convex_corners) return false;
        kept[nk++] = x;
        to_cap(x);
      }
    }
    if (!room) return false;
    if (nk >= 3) out.n[out.faces++] = nk;
  }
  if (nc >= 3) {
    if (out.faces == convex_faces || nc > convex_corners) return false;
    // The cap's corners in turn about their average, in the plane.
    const Point3 g = average(cap, nc);
    const Point3 a = Kokkos::fabs(h.n.x) < 0.9 ? Point3{1, 0, 0} : Point3{0, 1, 0};
    Point3 u = cross(h.n, a);
    const Real lu = Kokkos::sqrt(dot(u, u));
    u = {u.x / lu, u.y / lu, u.z / lu};
    const Point3 v = cross(h.n, u);
    Real angle[convex_corners * 2];
    for (int i = 0; i < nc; ++i) {
      const Point3 r = minus(cap[i], g);
      angle[i] = Kokkos::atan2(dot(r, v), dot(r, u));
    }
    for (int i = 1; i < nc; ++i) {
      const Point3 x = cap[i];
      const Real t = angle[i];
      int k = i - 1;
      for (; k >= 0 && angle[k] > t; --k) {
        cap[k + 1] = cap[k];
        angle[k + 1] = angle[k];
      }
      cap[k + 1] = x;
      angle[k + 1] = t;
    }
    for (int i = 0; i < nc; ++i) out.p[out.faces][i] = cap[i];
    out.n[out.faces++] = nc;
  }
  return true;
}

/// The volume of a convex polyhedron and its longest edge.
KOKKOS_INLINE_FUNCTION Real convex_volume(const Polyhedron& p, Real& longest) {
  Point3 g{};
  int count = 0;
  longest = 0;
  for (int f = 0; f < p.faces; ++f) {
    for (int i = 0; i < p.n[f]; ++i) {
      g = {g.x + p.p[f][i].x, g.y + p.p[f][i].y, g.z + p.p[f][i].z};
      ++count;
      const Point3 e = minus(p.p[f][(i + 1) % p.n[f]], p.p[f][i]);
      longest = Kokkos::fmax(longest, Kokkos::sqrt(dot(e, e)));
    }
  }
  if (count == 0) return 0;
  g = {g.x / count, g.y / count, g.z / count};
  Real v = 0;
  for (int f = 0; f < p.faces; ++f) {
    for (int i = 1; i + 1 < p.n[f]; ++i) v += Kokkos::fabs(six_volume({{g, p.p[f][0], p.p[f][i], p.p[f][i + 1]}})) / 6;
  }
  return v;
}

/// The volume convex cells s and c share (corners in the cells' own order;
/// counts 4, 5, 6 or 8): s cut by c's face planes. Negative where a cut
/// overflows the capacity.
KOKKOS_INLINE_FUNCTION Real convex_overlap(const Point3* s_in, int ns, const Point3* c_in, int nc, Real eps) {
  Point3 s[8], c[8];
  for (int i = 0; i < ns; ++i) s[i] = minus(s_in[i], c_in[0]);
  for (int i = 0; i < nc; ++i) c[i] = minus(c_in[i], c_in[0]);
  Plane h[6];
  const int nh = convex_planes(c, nc, h);
  if (nh == 0) return 0;
  Polyhedron a, b;
  const Topology ts = convex_topology(ns);
  a.faces = canonical::num_sides(ts, 2);
  for (int f = 0; f < a.faces; ++f) a.n[f] = convex_face(ts, f, s, a.p[f]);
  for (int k = 0; k < nh && a.faces > 0; ++k) {
    if (!cut_convex(a, h[k], eps, b)) return -1;
    a = b;
  }
  Real l = 0;
  const Real v = convex_volume(a, l);
  return v >= eps * l * l ? v : 0;
}

}  // namespace soma::detail

#endif  // SOMA_ALGORITHMS_CONVEX_CLIP_HPP_
