// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_FEM_HPP_
#define SOMA_ALGORITHMS_FEM_HPP_

// Linear finite elements on single cells: P1 triangles and tets, Q1 quads and
// hexes, with corners in MOAB's canonical order. Reference cells: the unit
// triangle (0,0) (1,0) (0,1) and tet (0,0,0) (1,0,0) (0,1,0) (0,0,1); [-1, 1]^d
// for quads and hexes. Cells of dimension 2 lie in a plane z = const.
//
//   fem::Cell c = fem::cell_of(view, dim, e);  // view = mesh.view()
//   Real k[fem::max_nodes][fem::max_nodes];
//   const Real det = fem::stiffness(c, k);  // !(det > 0): inverted, flat or unsupported
//
// Cells must be positively oriented: a cell whose Jacobian determinant is
// not positive at every corner and Gauss point contributes nothing, and the
// kernels say so by their return value, for the caller to fail on.
// Everything here is device-callable and allocation-free.

#include <Kokkos_Core.hpp>

#include "soma/core/mesh_view.hpp"
#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"

namespace soma::fem {

inline constexpr int max_nodes = 8;
inline constexpr int max_points = 27;

// A quadrature rule on a reference cell: n points xi with weights w, which
// sum to the reference cell's measure.
struct Quadrature {
  int n = 0;
  Real w[max_points] = {};
  Real xi[max_points][3] = {};
};

namespace detail {

KOKKOS_INLINE_FUNCTION void add(Quadrature& q, Real w, Real a, Real b, Real c) {
  q.w[q.n] = w;
  q.xi[q.n][0] = a;
  q.xi[q.n][1] = b;
  q.xi[q.n][2] = c;
  ++q.n;
}

// Gauss-Legendre with m = 1, 2 or 3 points on [-1, 1], exact to degree 2m - 1.
KOKKOS_INLINE_FUNCTION void gauss(int m, Real* x, Real* w) {
  if (m == 1) {
    x[0] = 0;
    w[0] = 2;
  } else if (m == 2) {
    const Real a = Real{1} / Kokkos::sqrt(Real{3});
    x[0] = -a;
    x[1] = a;
    w[0] = w[1] = 1;
  } else {
    const Real a = Kokkos::sqrt(Real{3} / Real{5});
    x[0] = -a;
    x[1] = 0;
    x[2] = a;
    w[0] = w[2] = Real{5} / Real{9};
    w[1] = Real{8} / Real{9};
  }
}

}  // namespace detail

// The smallest rule here exact for polynomials of `degree` (at most 5 for
// quads and hexes, 4 for triangles and tets) on the reference cell.
KOKKOS_INLINE_FUNCTION Quadrature quadrature(Topology t, int degree) {
  Quadrature q;
  if (t == Topology::quad || t == Topology::hex) {
    const int m = degree <= 1 ? 1 : degree <= 3 ? 2 : 3;
    Real x[3], w[3];
    detail::gauss(m, x, w);
    const int mk = t == Topology::hex ? m : 1;
    for (int k = 0; k < mk; ++k) {
      for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
          detail::add(q, w[i] * w[j] * (t == Topology::hex ? w[k] : Real{1}), x[i], x[j],
                      t == Topology::hex ? x[k] : Real{0});
        }
      }
    }
    return q;
  }
  if (t == Topology::tri) {
    if (degree <= 1) {
      detail::add(q, Real{1} / 2, Real{1} / 3, Real{1} / 3, 0);
    } else if (degree == 2) {
      detail::add(q, Real{1} / 6, Real{1} / 6, Real{1} / 6, 0);
      detail::add(q, Real{1} / 6, Real{2} / 3, Real{1} / 6, 0);
      detail::add(q, Real{1} / 6, Real{1} / 6, Real{2} / 3, 0);
    } else {
      // Dunavant's six points, degree 4.
      const Real a = 0.44594849091596488632, wa = 0.22338158967801146570 / 2;
      const Real b = 0.091576213509770743460, wb = 0.10995174365532186764 / 2;
      detail::add(q, wa, a, a, 0);
      detail::add(q, wa, 1 - 2 * a, a, 0);
      detail::add(q, wa, a, 1 - 2 * a, 0);
      detail::add(q, wb, b, b, 0);
      detail::add(q, wb, 1 - 2 * b, b, 0);
      detail::add(q, wb, b, 1 - 2 * b, 0);
    }
    return q;
  }
  if (t == Topology::tet) {
    if (degree <= 1) {
      detail::add(q, Real{1} / 6, Real{1} / 4, Real{1} / 4, Real{1} / 4);
    } else if (degree == 2) {
      const Real a = 0.5854101966249685, b = 0.1381966011250105;
      detail::add(q, Real{1} / 24, b, b, b);
      detail::add(q, Real{1} / 24, a, b, b);
      detail::add(q, Real{1} / 24, b, a, b);
      detail::add(q, Real{1} / 24, b, b, a);
    } else {
      // Keast's eleven points, degree 4 (one negative weight).
      detail::add(q, Real{-74} / 5625, Real{1} / 4, Real{1} / 4, Real{1} / 4);
      const Real a = Real{1} / 14, b = Real{11} / 14, wa = Real{343} / 45000;
      detail::add(q, wa, a, a, a);
      detail::add(q, wa, b, a, a);
      detail::add(q, wa, a, b, a);
      detail::add(q, wa, a, a, b);
      const Real c = 0.399403576166799219, d = 0.100596423833200785, wc = Real{56} / 2250;
      detail::add(q, wc, c, c, d);
      detail::add(q, wc, c, d, c);
      detail::add(q, wc, d, c, c);
      detail::add(q, wc, d, d, c);
      detail::add(q, wc, d, c, d);
      detail::add(q, wc, c, d, d);
    }
    return q;
  }
  return q;
}

// The values N and reference gradients dN (d per node) of the shape
// functions at xi; the number of nodes, 0 for an unsupported topology.
KOKKOS_INLINE_FUNCTION int shape(Topology t, const Real* xi, Real* n, Real (*dn)[3]) {
  if (t == Topology::tri || t == Topology::tet) {
    const int d = t == Topology::tri ? 2 : 3;
    n[0] = 1;
    for (int j = 0; j < 3; ++j) dn[0][j] = j < d ? -1 : 0;
    for (int j = 0; j < d; ++j) n[0] -= xi[j];
    for (int a = 1; a <= d; ++a) {
      n[a] = xi[a - 1];
      for (int j = 0; j < 3; ++j) dn[a][j] = j == a - 1 ? 1 : 0;
    }
    return d + 1;
  }
  if (t == Topology::quad || t == Topology::hex) {
    // Corner signs: (-1,-1) (1,-1) (1,1) (-1,1), then the same at z = 1.
    constexpr Real sx[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
    constexpr Real sy[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
    constexpr Real sz[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
    const bool hex = t == Topology::hex;
    const int nodes = hex ? 8 : 4;
    const Real scale = hex ? Real{1} / 8 : Real{1} / 4;
    for (int a = 0; a < nodes; ++a) {
      const Real fx = 1 + sx[a] * xi[0], fy = 1 + sy[a] * xi[1], fz = hex ? 1 + sz[a] * xi[2] : 1;
      n[a] = scale * fx * fy * fz;
      dn[a][0] = scale * sx[a] * fy * fz;
      dn[a][1] = scale * fx * sy[a] * fz;
      dn[a][2] = hex ? scale * fx * fy * sz[a] : 0;
    }
    return nodes;
  }
  return 0;
}

// The corner count of a topology these elements support, else 0.
KOKKOS_INLINE_FUNCTION constexpr int nodes_of(Topology t) {
  switch (t) {
    case Topology::tri: return 3;
    case Topology::quad: return 4;
    case Topology::tet: return 4;
    case Topology::hex: return 8;
    default: return 0;
  }
}

// A cell's topology, dimension, node count and corner coordinates; n = 0
// when cell_of reads a cell of a topology without these elements.
struct Cell {
  Topology topology = Topology::vertex;
  int dim = 0;
  int n = 0;
  Real x[max_nodes][3] = {};
};

template <class MemSpace>
KOKKOS_INLINE_FUNCTION Cell cell_of(const MeshView<MemSpace>& m, int dim, LO e) {
  Cell c;
  c.topology = m.topology(dim, e);
  c.dim = dim;
  const int n = nodes_of(c.topology);
  if (n == 0) return c;
  c.n = n;
  for (int a = 0; a < c.n; ++a) {
    const LO v = m.vertex(dim, e, a);
    for (int q = 0; q < 3; ++q) c.x[a][q] = m.x(v, q);
  }
  return c;
}

namespace detail {

// The Jacobian j[i][k] = d x_i / d xi_k at reference point xi, with the
// shape values n and reference gradients dn there; the node count.
KOKKOS_INLINE_FUNCTION int jacobian(const Cell& c, const Real* xi, Real* n, Real (*dn)[3], Real (*j)[3]) {
  const int nodes = shape(c.topology, xi, n, dn);
  const int d = c.dim;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) j[i][k] = 0;
  }
  for (int a = 0; a < nodes; ++a) {
    for (int i = 0; i < d; ++i) {
      for (int k = 0; k < d; ++k) j[i][k] += c.x[a][i] * dn[a][k];
    }
  }
  return nodes;
}

// The determinant of the d x d Jacobian j; 0 (not -0) when it vanishes.
KOKKOS_INLINE_FUNCTION Real determinant(int d, const Real (*j)[3]) {
  Real det = 0;
  if (d == 2) {
    det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
  } else {
    const Real c00 = j[1][1] * j[2][2] - j[1][2] * j[2][1];
    const Real c01 = j[1][2] * j[2][0] - j[1][0] * j[2][2];
    const Real c02 = j[1][0] * j[2][1] - j[1][1] * j[2][0];
    det = j[0][0] * c00 + j[0][1] * c01 + j[0][2] * c02;
  }
  return det == 0 ? Real{0} : det;
}

// The Jacobian's determinant at xi alone, as at_point gives it.
KOKKOS_INLINE_FUNCTION Real det_at(const Cell& c, const Real* xi) {
  Real n[max_nodes], dn[max_nodes][3], j[3][3];
  jacobian(c, xi, n, dn, j);
  return determinant(c.dim, j);
}

}  // namespace detail

// At reference point xi: the shape functions n, their physical gradients g,
// and the Jacobian's determinant (negative for an inverted cell, 0 for a
// degenerate one, when g is left unset).
KOKKOS_INLINE_FUNCTION Real at_point(const Cell& c, const Real* xi, Real* n, Real (*g)[3]) {
  Real dn[max_nodes][3], j[3][3];  // j[i][k] = d x_i / d xi_k
  const int nodes = detail::jacobian(c, xi, n, dn, j);
  const int d = c.dim;
  const Real det = detail::determinant(d, j);
  if (det == 0) return 0;
  Real inv[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};  // inv[k][i] = d xi_k / d x_i
  if (d == 2) {
    inv[0][0] = j[1][1] / det;
    inv[0][1] = -j[0][1] / det;
    inv[1][0] = -j[1][0] / det;
    inv[1][1] = j[0][0] / det;
  } else {
    inv[0][0] = (j[1][1] * j[2][2] - j[1][2] * j[2][1]) / det;
    inv[1][0] = (j[1][2] * j[2][0] - j[1][0] * j[2][2]) / det;
    inv[2][0] = (j[1][0] * j[2][1] - j[1][1] * j[2][0]) / det;
    inv[0][1] = (j[0][2] * j[2][1] - j[0][1] * j[2][2]) / det;
    inv[1][1] = (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / det;
    inv[2][1] = (j[0][1] * j[2][0] - j[0][0] * j[2][1]) / det;
    inv[0][2] = (j[0][1] * j[1][2] - j[0][2] * j[1][1]) / det;
    inv[1][2] = (j[0][2] * j[1][0] - j[0][0] * j[1][2]) / det;
    inv[2][2] = (j[0][0] * j[1][1] - j[0][1] * j[1][0]) / det;
  }
  for (int a = 0; a < nodes; ++a) {
    for (int i = 0; i < 3; ++i) {
      Real s = 0;
      for (int k = 0; k < d; ++k) s += dn[a][k] * inv[k][i];
      g[a][i] = i < d ? s : 0;
    }
  }
  return det;
}

namespace detail {

// Whether least_jacobian(c) is NaN before any determinant: n = 0, a
// topology without these elements, a coordinate not finite, or a 2D cell off
// one plane z = const.
KOKKOS_INLINE_FUNCTION bool unusable(const Cell& c) {
  if (c.n == 0 || c.n != nodes_of(c.topology) || c.dim != dimension(c.topology)) return true;
  for (int a = 0; a < c.n; ++a) {
    for (int i = 0; i < 3; ++i) {
      if (!Kokkos::isfinite(c.x[a][i])) return true;
    }
  }
  if (c.dim == 2) {
    for (int a = 1; a < c.n; ++a) {
      if (c.x[a][2] != c.x[0][2]) return true;
    }
  }
  return false;
}

// The lesser of least and det, where a determinant not finite (one that
// overflowed) counts as NaN and a NaN, once in, stays.
KOKKOS_INLINE_FUNCTION Real lesser(Real least, Real det) {
  if (!Kokkos::isfinite(det)) return Kokkos::Experimental::quiet_NaN_v<Real>;
  return Kokkos::isnan(least) || det >= least ? least : det;
}

// The determinant at corner a of a quad or hex. There a shape function's
// reference gradient along xi_k is +-1/2 for the corner and its neighbour
// along k and 0 for the others, so column k of the Jacobian is half the
// edge to that neighbour, signed: the sum at_point forms, bit for bit for
// finite coordinates (the halves are exact, the zero terms add nothing)
// where the compiler contracts each expression alike (-ffp-contract=on or
// off; fast may fuse the determinant differently where it is inlined).
KOKKOS_INLINE_FUNCTION Real det_at_corner(const Cell& c, int a) {
  constexpr Real sx[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
  constexpr Real sy[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
  constexpr Real sz[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
  const int d = c.dim;
  const int along[3] = {a ^ 1, (a & 4) | ((a & 3) ^ 3), a ^ 4};
  const Real sign[3] = {sx[a], sy[a], sz[a]};
  Real j[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  for (int k = 0; k < d; ++k) {
    const Real h = sign[k] / 2;
    for (int i = 0; i < d; ++i) j[i][k] = c.x[a][i] * h + c.x[along[k]][i] * -h;
  }
  return determinant(d, j);
}

// The smallest determinant over a quad's or hex's corners, as lesser folds.
KOKKOS_INLINE_FUNCTION Real least_at_corners(const Cell& c) {
  Real least = Kokkos::Experimental::infinity_v<Real>;
  for (int a = 0; a < c.n; ++a) least = lesser(least, det_at_corner(c, a));
  return least;
}

}  // namespace detail

// The smallest Jacobian determinant over the cell's corners and the Gauss
// points of the rules below: positive for a usable cell. NaN for a cell not
// usable here at all: of n = 0, a topology without these elements, a
// coordinate not finite, a determinant that overflows, or of dimension 2
// with corners off one plane z = const. The determinant of a simplex is
// constant; that of a quad or hex is checked at its corners too, where a
// folded cell inverts first.
KOKKOS_INLINE_FUNCTION Real least_jacobian(const Cell& c) {
  if (detail::unusable(c)) return Kokkos::Experimental::quiet_NaN_v<Real>;
  const Real centre[3] = {Real{1} / 4, Real{1} / 4, Real{1} / 4};
  if (c.topology == Topology::tri || c.topology == Topology::tet) {
    return detail::lesser(Kokkos::Experimental::infinity_v<Real>, detail::det_at(c, centre));
  }
  Real least = detail::least_at_corners(c);
  const Quadrature q = quadrature(c.topology, 2);
  for (int p = 0; p < q.n; ++p) least = detail::lesser(least, detail::det_at(c, q.xi[p]));
  return least;
}

namespace detail {

// Calls body(w, n, g) at each point of the rule of `degree` with w the
// weight times the Jacobian's determinant, and returns least_jacobian(c).
// Unless that is > 0, body may have run on some points and what it made is
// to be discarded. A quad's or hex's degree-2 rule is the one
// least_jacobian checks: its determinants serve both.
template <class Body>
KOKKOS_INLINE_FUNCTION Real integrate(const Cell& c, int degree, const Body& body) {
  const Quadrature q = quadrature(c.topology, degree);
  const bool shared = (c.topology == Topology::quad || c.topology == Topology::hex) && degree >= 2 && degree <= 3;
  Real least = Kokkos::Experimental::quiet_NaN_v<Real>;
  if (!shared) {
    least = least_jacobian(c);
    if (!(least > 0)) return least;
  } else {
    if (unusable(c)) return least;
    least = least_at_corners(c);
    if (!(least > 0)) return least_jacobian(c);
  }
  for (int p = 0; p < q.n; ++p) {
    Real n[max_nodes], g[max_nodes][3];
    const Real det = at_point(c, q.xi[p], n, g);
    if (shared) {
      least = lesser(least, det);
      if (!(least > 0)) return least_jacobian(c);
    }
    body(q.w[p] * det, n, g);
  }
  return least;
}

}  // namespace detail

// The stiffness matrix k[a][b] = integral of grad N_a . grad N_b (exact for
// simplices and for parallelogram-shaped quads and hexes). Returns
// least_jacobian(c); unless it is > 0, k is zero and not to be used.
KOKKOS_INLINE_FUNCTION Real stiffness(const Cell& c, Real (*k)[max_nodes]) {
  for (int a = 0; a < c.n; ++a) {
    for (int b = 0; b < c.n; ++b) k[a][b] = 0;
  }
  const bool simplex = c.topology == Topology::tri || c.topology == Topology::tet;
  // The upper triangle, then its mirror: g_a . g_b is g_b . g_a, bit for bit.
  const Real least = detail::integrate(c, simplex ? 0 : 2, [&](Real w, const Real*, const Real (*g)[3]) {
    for (int a = 0; a < c.n; ++a) {
      for (int b = a; b < c.n; ++b) k[a][b] += w * (g[a][0] * g[b][0] + g[a][1] * g[b][1] + g[a][2] * g[b][2]);
    }
  });
  for (int a = 0; a < c.n; ++a) {
    for (int b = 0; b < c.n; ++b) k[a][b] = !(least > 0) ? 0 : b < a ? k[b][a] : k[a][b];
  }
  return least;
}

// The mass matrix m[a][b] = integral of N_a N_b, exact on affine cells.
// Returns least_jacobian(c); unless it is > 0, m is zero.
KOKKOS_INLINE_FUNCTION Real mass(const Cell& c, Real (*m)[max_nodes]) {
  for (int a = 0; a < c.n; ++a) {
    for (int b = 0; b < c.n; ++b) m[a][b] = 0;
  }
  const Real least = detail::integrate(c, 2, [&](Real w, const Real* n, const Real (*)[3]) {
    for (int a = 0; a < c.n; ++a) {
      for (int b = 0; b < c.n; ++b) m[a][b] += w * n[a] * n[b];
    }
  });
  if (!(least > 0)) {
    for (int a = 0; a < c.n; ++a) {
      for (int b = 0; b < c.n; ++b) m[a][b] = 0;
    }
  }
  return least;
}

// The point x of the cell at reference point xi, given the shape values n.
KOKKOS_INLINE_FUNCTION void point(const Cell& c, const Real* n, Real* x) {
  for (int i = 0; i < 3; ++i) {
    x[i] = 0;
    for (int a = 0; a < c.n; ++a) x[i] += n[a] * c.x[a][i];
  }
}

// The load vector r[a] = integral of f N_a by the degree-2 rule, f(x, y, z)
// device-callable. Returns least_jacobian(c); unless it is > 0, r is zero.
template <class F>
KOKKOS_INLINE_FUNCTION Real load(const Cell& c, const F& f, Real* r) {
  for (int a = 0; a < c.n; ++a) r[a] = 0;
  const Real least = detail::integrate(c, 2, [&](Real w, const Real* n, const Real (*)[3]) {
    Real x[3];
    point(c, n, x);
    const Real fx = f(x[0], x[1], x[2]);
    for (int a = 0; a < c.n; ++a) r[a] += w * fx * n[a];
  });
  if (!(least > 0)) {
    for (int a = 0; a < c.n; ++a) r[a] = 0;
  }
  return least;
}

// The integral over the cell of (sum_a u[a] N_a - exact)^2 by a degree-4
// rule (5 for quads and hexes); NaN unless least_jacobian(c) > 0, so that a
// sum over cells shows a bad one.
template <class F>
KOKKOS_INLINE_FUNCTION Real squared_error(const Cell& c, const Real* u, const F& exact) {
  if (!(least_jacobian(c) > 0)) return Kokkos::Experimental::quiet_NaN_v<Real>;
  const Quadrature q = quadrature(c.topology, 4);
  Real sum = 0;
  for (int p = 0; p < q.n; ++p) {
    Real n[max_nodes], g[max_nodes][3], x[3];
    const Real w = q.w[p] * at_point(c, q.xi[p], n, g);
    point(c, n, x);
    Real uh = 0;
    for (int a = 0; a < c.n; ++a) uh += u[a] * n[a];
    const Real e = uh - exact(x[0], x[1], x[2]);
    sum += w * e * e;
  }
  return sum;
}

}  // namespace soma::fem

#endif  // SOMA_ALGORITHMS_FEM_HPP_
