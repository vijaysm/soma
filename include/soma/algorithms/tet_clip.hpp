// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_TET_CLIP_HPP_
#define SOMA_ALGORITHMS_TET_CLIP_HPP_

// The overlap of two tetrahedra, for the remap of volume meshes: one tet cut
// by the four planes of the other, each cut keeping the part on the other's
// inner side as at most three tets, and the volumes summed. Callable in
// kernels; fixed capacity, no allocation.
//
// A cut classifies each corner by its signed distance to the plane: within
// eps (the merge tolerance, a length) it is on the plane and kept, so a
// corner never moves by less than eps and no piece thinner than eps is made
// along a face. Of the corners kept (k of 4):
//
//   4  the tet;
//   3  a prism (the three, and the points where their edges to the fourth
//      cross the plane): three tets;
//   2  a wedge (the two, and the four crossings): three tets;
//   1  a tet (the one and its three crossings);
//   0  nothing.
//
// The pieces are only a way to sum the overlap's volume (a split prism or
// wedge may hold thin tets of real volume), so the whole overlap, not a
// piece, is dropped when it is thinner than eps: below eps L^2, L the
// longest edge of its pieces (its extent). Volumes are taken about each
// tet's first corner, so their rounding is relative to the tet's size.

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"

namespace soma::detail {

struct Point3 {
  Real x = 0, y = 0, z = 0;
};

struct Tet {
  Point3 p[4];
};

// Pieces a clip may hold: each of four cuts at most triples them.
inline constexpr int tet_capacity = 81;

KOKKOS_INLINE_FUNCTION Point3 minus(const Point3& a, const Point3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
KOKKOS_INLINE_FUNCTION Real dot(const Point3& a, const Point3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
KOKKOS_INLINE_FUNCTION Point3 cross(const Point3& a, const Point3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Six times the signed volume: positive where b - a, c - a, d - a are
// right-handed.
KOKKOS_INLINE_FUNCTION Real six_volume(const Tet& t) {
  return dot(minus(t.p[1], t.p[0]), cross(minus(t.p[2], t.p[0]), minus(t.p[3], t.p[0])));
}

KOKKOS_INLINE_FUNCTION Real longest_edge(const Tet& t) {
  Real l = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      const Point3 d = minus(t.p[j], t.p[i]);
      l = Kokkos::fmax(l, Kokkos::sqrt(dot(d, d)));
    }
  }
  return l;
}

// A plane: n . x = c, n of unit length; inside where n . x - c <= 0.
struct Plane {
  Point3 n;
  Real c = 0;
};

// The plane of face (a, b, c) of tet t, its inside the tet's (the face's
// opposite corner d). False for a face of no area.
KOKKOS_INLINE_FUNCTION bool face_plane(const Point3& a, const Point3& b, const Point3& c, const Point3& d, Plane& out) {
  Point3 n = cross(minus(b, a), minus(c, a));
  const Real len = Kokkos::sqrt(dot(n, n));
  if (!(len > 0)) return false;
  n = {n.x / len, n.y / len, n.z / len};
  Real off = dot(n, a);
  if (dot(n, d) - off > 0) {  // turn it so that d is inside
    n = {-n.x, -n.y, -n.z};
    off = -off;
  }
  out.n = n;
  out.c = off;
  return true;
}

// The part of tet t inside plane h, as up to three tets appended to out
// (count in, count out); false where out would overflow.
KOKKOS_INLINE_FUNCTION bool cut(const Tet& t, const Plane& h, Real eps, Tet* out, int& count) {
  Real d[4];
  int in[4], outside[4], ni = 0, no = 0;
  for (int i = 0; i < 4; ++i) {
    d[i] = dot(h.n, t.p[i]) - h.c;
    if (d[i] <= eps) {
      in[ni++] = i;
    } else {
      outside[no++] = i;
    }
  }
  // Where edge (i, o) crosses the plane: i inside (d <= eps), o outside (d > eps).
  const auto cross_at = [&](int i, int o) {
    const Real di = d[i] < 0 ? d[i] : 0;  // a corner on the plane is the crossing itself
    if (di == 0) return t.p[i];
    const Real s = di / (di - d[o]);
    return Point3{t.p[i].x + s * (t.p[o].x - t.p[i].x), t.p[i].y + s * (t.p[o].y - t.p[i].y),
                  t.p[i].z + s * (t.p[o].z - t.p[i].z)};
  };
  const int need = ni == 4 || ni == 1 ? 1 : ni == 0 ? 0 : 3;
  if (count + need > tet_capacity) return false;
  if (ni == 4) {
    out[count++] = t;
  } else if (ni == 1) {
    const int a = in[0];
    out[count++] = {{t.p[a], cross_at(a, outside[0]), cross_at(a, outside[1]), cross_at(a, outside[2])}};
  } else if (ni == 3) {
    // Prism: a, b, c and ad, bd, cd.
    const int a = in[0], b = in[1], c = in[2], o = outside[0];
    const Point3 ad = cross_at(a, o), bd = cross_at(b, o), cd = cross_at(c, o);
    out[count++] = {{t.p[a], t.p[b], t.p[c], ad}};
    out[count++] = {{t.p[b], t.p[c], ad, bd}};
    out[count++] = {{t.p[c], ad, bd, cd}};
  } else if (ni == 2) {
    // Wedge: triangle (a, ac, ad) swept to (b, bc, bd).
    const int a = in[0], b = in[1], c = outside[0], e = outside[1];
    const Point3 ac = cross_at(a, c), ad = cross_at(a, e), bc = cross_at(b, c), bd = cross_at(b, e);
    out[count++] = {{t.p[a], ac, ad, t.p[b]}};
    out[count++] = {{ac, ad, t.p[b], bc}};
    out[count++] = {{ad, t.p[b], bc, bd}};
  }
  return true;
}

// The volume tets s and c share: s cut by c's four face planes. Negative
// where the pieces overflow tet_capacity (never for two tets: each cut of a
// convex piece list keeps the count within 3^4).
KOKKOS_INLINE_FUNCTION Real overlap(const Tet& s, const Tet& c, Real eps) {
  Tet a[tet_capacity], b[tet_capacity];
  int na = 1, nb = 0;
  a[0] = s;
  const int faces[4][4] = {{1, 2, 3, 0}, {0, 2, 3, 1}, {0, 1, 3, 2}, {0, 1, 2, 3}};
  for (int f = 0; f < 4 && na > 0; ++f) {
    Plane h;
    if (!face_plane(c.p[faces[f][0]], c.p[faces[f][1]], c.p[faces[f][2]], c.p[faces[f][3]], h)) return 0;
    nb = 0;
    for (int i = 0; i < na; ++i) {
      if (!cut(a[i], h, eps, b, nb)) return -1;
    }
    for (int i = 0; i < nb; ++i) a[i] = b[i];
    na = nb;
  }
  Real v = 0, l = 0;
  for (int i = 0; i < na; ++i) {
    v += Kokkos::fabs(six_volume(a[i])) / 6;
    l = Kokkos::fmax(l, longest_edge(a[i]));
  }
  return v >= eps * l * l ? v : 0;
}

}  // namespace soma::detail

#endif  // SOMA_ALGORITHMS_TET_CLIP_HPP_
