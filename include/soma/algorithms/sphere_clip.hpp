// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_SPHERE_CLIP_HPP_
#define SOMA_ALGORITHMS_SPHERE_CLIP_HPP_

/// \file
/// Polygons on the unit sphere whose edges are great-circle arcs, for the
/// remap of meshes on a sphere: areas, and the overlap of two convex ones by
/// either of two methods. Callable in kernels; fixed capacity, no allocation.
///
/// ```
///   gnomonic      both polygons projected from the centre onto the plane
///                 touching the sphere at the centre of the cube face that the
///                 sum of the two polygons' mean directions points into (great circles
///                 project to straight lines), clipped there as plane polygons
///                 (polygon.hpp), and the overlap's corners taken back onto the
///                 sphere; as MOAB's Intx2MeshOnSphere. Polygons that an edge's
///                 great circle parts overlap by nothing; others reaching the
///                 face's plane's horizon cannot be projected.
///   great_circle  the subject cut by the plane through the centre and each
///                 edge of the clipping polygon in turn (Sutherland and
///                 Hodgman on the sphere): a point is inside where it is on
///                 the edge's left, seen from outside the sphere.
/// ```
///
/// As in the plane (polygon.hpp), one tolerance eps governs every decision,
/// here as a distance on the unit sphere: a corner outside a clipping great
/// circle by at most eps is moved onto it, consecutive corners closer than
/// eps merge, and an overlap of area below eps^2 is nothing. The area of a
/// polygon is the sum of its fan's triangles' spherical excesses from its
/// first corner (Van Oosterom and Strackee 1983), each
/// 2 atan2(a . (b x c), 1 + a . b + b . c + c . a): positive counter-
/// clockwise seen from outside.

#include <Kokkos_Core.hpp>

#include "soma/algorithms/polygon.hpp"
#include "soma/algorithms/tet_clip.hpp"
#include "soma/core/types.hpp"

namespace soma::detail {

struct SpherePolygon {
  int n = 0;
  Point3 p[polygon_storage];

  KOKKOS_INLINE_FUNCTION void push(const Point3& q) {
    if (n < polygon_storage) p[n] = q;
    ++n;  // past storage: the caller sees n > polygon_storage
  }
};

KOKKOS_INLINE_FUNCTION Point3 unit(const Point3& a) {
  const Real l = Kokkos::sqrt(dot(a, a));
  return {a.x / l, a.y / l, a.z / l};
}

/// The signed area of spherical triangle a, b, c (unit vectors).
KOKKOS_INLINE_FUNCTION Real sphere_triangle_area(const Point3& a, const Point3& b, const Point3& c) {
  return 2 * Kokkos::atan2(dot(a, cross(b, c)), 1 + dot(a, b) + dot(b, c) + dot(c, a));
}

/// The signed area of a polygon on the unit sphere: its fan from p[0].
KOKKOS_INLINE_FUNCTION Real sphere_area(const SpherePolygon& s) {
  Real a = 0;
  for (int i = 1; i + 1 < s.n; ++i) a += sphere_triangle_area(s.p[0], s.p[i], s.p[i + 1]);
  return a;
}

KOKKOS_INLINE_FUNCTION Real chord(const Point3& a, const Point3& b) {
  const Point3 d = minus(a, b);
  return Kokkos::sqrt(dot(d, d));
}

/// Consecutive corners closer than eps merged (the first of a run kept).
KOKKOS_INLINE_FUNCTION void merge_corners(SpherePolygon& s, Real eps) {
  if (s.n > polygon_storage) return;
  int m = 0;
  for (int i = 0; i < s.n; ++i) {
    if (m > 0 && chord(s.p[i], s.p[m - 1]) < eps) continue;
    s.p[m++] = s.p[i];
  }
  while (m > 1 && chord(s.p[m - 1], s.p[0]) < eps) --m;
  s.n = m;
}

/// The overlap's area, 0 where it is under eps^2 or has fewer than three
/// corners.
KOKKOS_INLINE_FUNCTION Real overlap_area(SpherePolygon& out, Real eps) {
  if (out.n < 3) {
    out.n = 0;
    return 0;
  }
  const Real area = sphere_area(out);
  if (!(area >= eps * eps)) {
    out.n = 0;
    return 0;
  }
  return area;
}

/// The overlap of convex polygons s and c (unit corners, counter-clockwise
/// seen from outside) by great circles, into out; its area (negative where
/// it overflows polygon_storage).
KOKKOS_INLINE_FUNCTION Real clip_great_circle(const SpherePolygon& s, const SpherePolygon& c, Real eps,
                                              SpherePolygon& out) {
  SpherePolygon a = s, b;
  for (int e = 0; e < c.n && a.n > 0; ++e) {
    const int f = e + 1 == c.n ? 0 : e + 1;
    Point3 nrm = cross(c.p[e], c.p[f]);
    const Real len = Kokkos::sqrt(dot(nrm, nrm));
    if (!(len > 0)) continue;
    nrm = {nrm.x / len, nrm.y / len, nrm.z / len};
    b.n = 0;
    for (int i = 0; i < a.n; ++i) {
      const int j = i + 1 == a.n ? 0 : i + 1;
      const Real di = dot(nrm, a.p[i]), dj = dot(nrm, a.p[j]);
      if (di >= 0) {
        b.push(a.p[i]);
      } else if (di >= -eps) {
        b.push(unit({a.p[i].x - di * nrm.x, a.p[i].y - di * nrm.y, a.p[i].z - di * nrm.z}));  // onto the circle
      }
      if ((di >= 0 && dj < -eps) || (di < -eps && dj >= 0)) {
        // Where the arc crosses the circle: the chord's crossing, from the
        // sphere's centre.
        const Real t = di / (di - dj);
        b.push(unit({a.p[i].x + t * (a.p[j].x - a.p[i].x), a.p[i].y + t * (a.p[j].y - a.p[i].y),
                     a.p[i].z + t * (a.p[j].z - a.p[i].z)}));
      }
    }
    if (b.n > polygon_storage) {
      out.n = 0;
      return -1;
    }
    merge_corners(b, eps);
    a = b;
  }
  out = a;
  return overlap_area(out, eps);
}

/// The cube face a direction points into: its normal f and the plane's axes
/// u, v (right-handed: u x v = f).
KOKKOS_INLINE_FUNCTION void cube_face(const Point3& d, Point3& f, Point3& u, Point3& v) {
  const Real ax = Kokkos::fabs(d.x), ay = Kokkos::fabs(d.y), az = Kokkos::fabs(d.z);
  if (ax >= ay && ax >= az) {
    const Real s = d.x >= 0 ? 1 : -1;
    f = {s, 0, 0};
    u = {0, s, 0};
    v = {0, 0, 1};
  } else if (ay >= az) {
    const Real s = d.y >= 0 ? 1 : -1;
    f = {0, s, 0};
    u = {0, 0, s};
    v = {1, 0, 0};
  } else {
    const Real s = d.z >= 0 ? 1 : -1;
    f = {0, 0, s};
    u = {s, 0, 0};
    v = {0, 1, 0};
  }
}

/// Whether an edge's great circle of a or of b has the other's corners all
/// beyond it by more than eps: the polygons do not overlap.
KOKKOS_INLINE_FUNCTION bool apart(const SpherePolygon& a, const SpherePolygon& b, Real eps) {
  const auto beyond = [&](const SpherePolygon& p, const SpherePolygon& q) {
    for (int e = 0; e < p.n; ++e) {
      const Point3 nrm = cross(p.p[e], p.p[e + 1 == p.n ? 0 : e + 1]);
      const Real len = Kokkos::sqrt(dot(nrm, nrm));
      if (!(len > 0)) continue;
      bool all = true;
      for (int i = 0; i < q.n && all; ++i) all = dot(nrm, q.p[i]) / len < -eps;
      if (all) return true;
    }
    return false;
  };
  return beyond(a, b) || beyond(b, a);
}

/// The overlap of convex polygons s and c by the gnomonic projection onto
/// their cube face, into out (on the sphere); its area: 0 where an edge's
/// great circle parts them, negative where it overflows or a corner of
/// either is not in front of the face's plane.
KOKKOS_INLINE_FUNCTION Real clip_gnomonic(const SpherePolygon& s, const SpherePolygon& c, Real eps,
                                          SpherePolygon& out) {
  // The face both polygons' mean directions, summed, point into.
  const auto mean = [](const SpherePolygon& q) {
    Point3 m{};
    for (int i = 0; i < q.n; ++i) m = {m.x + q.p[i].x, m.y + q.p[i].y, m.z + q.p[i].z};
    return unit(m);
  };
  const Point3 ms = mean(s), mc = mean(c);
  Point3 f, u, v;
  cube_face({ms.x + mc.x, ms.y + mc.y, ms.z + mc.z}, f, u, v);
  Polygon ps, pc, po;
  const auto project = [&](const SpherePolygon& from, Polygon& to) {
    to.n = 0;
    for (int i = 0; i < from.n; ++i) {
      const Real h = dot(from.p[i], f);
      if (!(h > 0)) return false;
      to.push(dot(from.p[i], u) / h, dot(from.p[i], v) / h);
    }
    return true;
  };
  out.n = 0;
  if (apart(s, c, eps)) return 0;
  if (!project(s, ps) || !project(c, pc)) return -1;
  if (clip(ps, pc, eps, po) < 0) return -1;
  for (int i = 0; i < po.n; ++i) {
    out.push(unit({f.x + po.x[i] * u.x + po.y[i] * v.x, f.y + po.x[i] * u.y + po.y[i] * v.y,
                   f.z + po.x[i] * u.z + po.y[i] * v.z}));
  }
  return overlap_area(out, eps);
}

}  // namespace soma::detail

#endif  // SOMA_ALGORITHMS_SPHERE_CLIP_HPP_
