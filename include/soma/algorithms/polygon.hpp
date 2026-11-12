// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_POLYGON_HPP_
#define SOMA_ALGORITHMS_POLYGON_HPP_

// Plane polygons for intersection: a cell's corners as a counter-clockwise
// polygon, the overlap of two convex polygons, and a non-convex polygon split
// into triangles. Callable in kernels; fixed-capacity, no allocation.
//
// One tolerance, the merge tolerance eps (a length), governs every decision:
// a point within eps of a clipping line counts as on it, consecutive corners
// closer than eps merge into one, and a piece with fewer than three corners
// left or an area below eps^2 is no piece. A polygon is convex when no corner
// turns clockwise by more than eps (as a distance from the line through its
// neighbours); corners in a line are allowed.
//
// The overlap of convex polygons is Sutherland and Hodgman's clipping (1974):
// the subject cut by the half-plane left of each edge of the clip polygon in
// turn. Its area depends on the corners' order, so callers start each
// polygon at a fixed corner (soma's remap: the least gid) to get the same
// bits on every rank.
//
// A non-convex polygon is split by ear clipping (Meisters 1975): from its
// first corner on, the first convex corner whose triangle holds no other
// corner is cut off, until three are left. The triangles are then clipped as
// convex polygons and their overlaps summed.

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"

namespace soma::detail {

// Corners a polygon may have; an overlap of two has at most the sum.
inline constexpr int polygon_capacity = 64;

struct Polygon {
  int n = 0;
  Real x[polygon_capacity] = {}, y[polygon_capacity] = {};

  KOKKOS_INLINE_FUNCTION void push(Real px, Real py) {
    if (n < polygon_capacity) {
      x[n] = px;
      y[n] = py;
    }
    ++n;  // past capacity: the caller sees n > polygon_capacity
  }
};

// Twice the signed area (positive counter-clockwise): the shoelace formula
// about the first corner, summed from it, so that its rounding is relative
// to the polygon's size, not to its distance from the origin.
KOKKOS_INLINE_FUNCTION Real twice_area(const Polygon& p) {
  Real s = 0;
  for (int i = 1; i + 1 < p.n; ++i) {
    const Real ax = p.x[i] - p.x[0], ay = p.y[i] - p.y[0], bx = p.x[i + 1] - p.x[0], by = p.y[i + 1] - p.y[0];
    s += ax * by - bx * ay;
  }
  return s;
}

// (b - a) x (c - a): positive where a, b, c turn counter-clockwise.
KOKKOS_INLINE_FUNCTION Real turn(Real ax, Real ay, Real bx, Real by, Real cx, Real cy) {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

// Consecutive corners closer than eps merged (the first of a run kept), in
// place; a last corner within eps of the first goes too.
KOKKOS_INLINE_FUNCTION void merge_corners(Polygon& p, Real eps) {
  if (p.n > polygon_capacity) return;
  int m = 0;
  for (int i = 0; i < p.n; ++i) {
    if (m > 0 && Kokkos::hypot(p.x[i] - p.x[m - 1], p.y[i] - p.y[m - 1]) < eps) continue;
    p.x[m] = p.x[i];
    p.y[m] = p.y[i];
    ++m;
  }
  while (m > 1 && Kokkos::hypot(p.x[m - 1] - p.x[0], p.y[m - 1] - p.y[0]) < eps) --m;
  p.n = m;
}

// A counter-clockwise polygon with no corner turning clockwise by more than
// eps.
KOKKOS_INLINE_FUNCTION bool convex(const Polygon& p, Real eps) {
  if (p.n < 3 || p.n > polygon_capacity) return false;
  for (int i = 0; i < p.n; ++i) {
    const int h = i == 0 ? p.n - 1 : i - 1, j = i + 1 == p.n ? 0 : i + 1;
    const Real base = Kokkos::hypot(p.x[j] - p.x[h], p.y[j] - p.y[h]);
    // Corner i's distance to the left of the line from h to j.
    if (turn(p.x[h], p.y[h], p.x[i], p.y[i], p.x[j], p.y[j]) < -eps * base) return false;
  }
  return true;
}

// The overlap of convex counter-clockwise polygons s and c, into out; its
// area (0 where none). out starts where s's clipping leaves it.
KOKKOS_INLINE_FUNCTION Real clip(const Polygon& s, const Polygon& c, Real eps, Polygon& out) {
  Polygon a = s, b;
  for (int e = 0; e < c.n && a.n > 0; ++e) {
    const int f = e + 1 == c.n ? 0 : e + 1;
    const Real ex = c.x[f] - c.x[e], ey = c.y[f] - c.y[e];
    const Real len = Kokkos::hypot(ex, ey);
    if (!(len > 0)) continue;
    // Signed distance of a point to the left of edge e.
    const auto side = [&](Real px, Real py) { return (ex * (py - c.y[e]) - ey * (px - c.x[e])) / len; };
    b.n = 0;
    for (int i = 0; i < a.n; ++i) {
      const int j = i + 1 == a.n ? 0 : i + 1;
      const Real di = side(a.x[i], a.y[i]), dj = side(a.x[j], a.y[j]);
      if (di >= -eps) b.push(a.x[i], a.y[i]);
      // Crossing strictly from one side to the other: the point on the line.
      if ((di > eps && dj < -eps) || (di < -eps && dj > eps)) {
        const Real t = di / (di - dj);
        b.push(a.x[i] + t * (a.x[j] - a.x[i]), a.y[i] + t * (a.y[j] - a.y[i]));
      }
    }
    if (b.n > polygon_capacity) {
      out.n = polygon_capacity + 1;
      return 0;
    }
    merge_corners(b, eps);
    a = b;
  }
  out = a;
  if (out.n < 3) {
    out.n = 0;
    return 0;
  }
  const Real area = twice_area(out) / 2;
  if (!(area >= eps * eps)) {
    out.n = 0;
    return 0;
  }
  return area;
}

// A counter-clockwise simple polygon's triangles by ear clipping, as corner
// triples into tri (3 per triangle, at most n - 2); their count, or -1 where
// no ear is found (the polygon crosses itself, or has too many corners).
KOKKOS_INLINE_FUNCTION int triangulate(const Polygon& p, Real eps, int* tri) {
  if (p.n < 3 || p.n > polygon_capacity) return -1;
  int left[polygon_capacity];
  int m = p.n, count = 0;
  for (int i = 0; i < m; ++i) left[i] = i;
  while (m > 3) {
    bool cut = false;
    for (int k = 0; k < m && !cut; ++k) {
      const int a = left[k == 0 ? m - 1 : k - 1], b = left[k], c = left[k + 1 == m ? 0 : k + 1];
      const Real base = Kokkos::hypot(p.x[c] - p.x[a], p.y[c] - p.y[a]);
      if (!(turn(p.x[a], p.y[a], p.x[b], p.y[b], p.x[c], p.y[c]) > eps * base)) continue;  // not convex
      // No other corner inside or on the triangle (a, b, c), but for one
      // at the place of one of its corners: a corner on the cut would leave
      // a degenerate piece.
      bool empty = true;
      for (int q = 0; q < m && empty; ++q) {
        const int v = left[q];
        if (v == a || v == b || v == c) continue;
        const Real px = p.x[v], py = p.y[v];
        const bool at = (px == p.x[a] && py == p.y[a]) || (px == p.x[b] && py == p.y[b]) || (px == p.x[c] && py == p.y[c]);
        empty = at || !(turn(p.x[a], p.y[a], p.x[b], p.y[b], px, py) >= 0 &&
                        turn(p.x[b], p.y[b], p.x[c], p.y[c], px, py) >= 0 && turn(p.x[c], p.y[c], p.x[a], p.y[a], px, py) >= 0);
      }
      if (!empty) continue;
      tri[3 * count] = a;
      tri[3 * count + 1] = b;
      tri[3 * count + 2] = c;
      ++count;
      for (int q = k; q + 1 < m; ++q) left[q] = left[q + 1];
      --m;
      cut = true;
    }
    if (!cut) return -1;
  }
  tri[3 * count] = left[0];
  tri[3 * count + 1] = left[1];
  tri[3 * count + 2] = left[2];
  return count + 1;
}

}  // namespace soma::detail

#endif  // SOMA_ALGORITHMS_POLYGON_HPP_
