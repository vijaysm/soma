// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "element_map.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace soma::detail::sample {

namespace {

// Reference coordinates of a quad's nodes in MOAB's order: corners, the
// midpoints of edges 0-1, 1-2, 2-3, 3-0, the centre.
constexpr Real quad_xi[9] = {-1, 1, 1, -1, 0, 1, 0, -1, 0};
constexpr Real quad_eta[9] = {-1, -1, 1, 1, -1, 0, 1, 0, 0};

Real quadratic(Real t, Real at) { return at < 0 ? t * (t - 1) / 2 : at > 0 ? t * (t + 1) / 2 : 1 - t * t; }
Real quadratic_d(Real t, Real at) { return at < 0 ? t - Real{0.5} : at > 0 ? t + Real{0.5} : -2 * t; }

int reference_dim_of(Topology t) {
  switch (t) {
    case Topology::edge: return 1;
    case Topology::tri:
    case Topology::quad: return 2;
    case Topology::tet:
    case Topology::pyramid:
    case Topology::prism:
    case Topology::hex: return 3;
    default: return 0;
  }
}

std::size_t nodes_of(Topology t) {
  switch (t) {
    case Topology::edge: return 2;
    case Topology::tri: return 3;
    case Topology::quad: return 4;
    case Topology::tet: return 4;
    case Topology::pyramid: return 5;
    case Topology::prism: return 6;
    case Topology::hex: return 8;
    default: return 0;
  }
}

// Shape functions n and their reference derivatives dn (3 per node) at xi.
// Returns false at a pyramid's apex, where only n is defined.
bool shape(Topology t, std::size_t count, const Real* xi, Real* n, Real (*dn)[3]) {
  for (std::size_t a = 0; a < count; ++a) dn[a][0] = dn[a][1] = dn[a][2] = 0;
  switch (t) {
    case Topology::edge:
      n[0] = (1 - xi[0]) / 2;
      n[1] = (1 + xi[0]) / 2;
      dn[0][0] = Real{-0.5};
      dn[1][0] = Real{0.5};
      return true;
    case Topology::tri:
    case Topology::tet: {
      const int d = t == Topology::tri ? 2 : 3;
      n[0] = 1;
      for (int j = 0; j < d; ++j) {
        n[j + 1] = (1 + xi[j]) / 2;
        n[0] -= n[j + 1];
        dn[j + 1][j] = Real{0.5};
        dn[0][j] = Real{-0.5};
      }
      return true;
    }
    case Topology::quad: {
      const Real s = xi[0], r = xi[1];
      for (std::size_t a = 0; a < count; ++a) {
        const Real p = quad_xi[a], q = quad_eta[a];
        if (count == 4) {
          n[a] = (1 + s * p) * (1 + r * q) / 4;
          dn[a][0] = p * (1 + r * q) / 4;
          dn[a][1] = q * (1 + s * p) / 4;
        } else if (count == 9) {
          n[a] = quadratic(s, p) * quadratic(r, q);
          dn[a][0] = quadratic_d(s, p) * quadratic(r, q);
          dn[a][1] = quadratic(s, p) * quadratic_d(r, q);
        } else if (a < 4) {
          n[a] = (1 + s * p) * (1 + r * q) * (s * p + r * q - 1) / 4;
          dn[a][0] = p * (1 + r * q) * (2 * s * p + r * q) / 4;
          dn[a][1] = q * (1 + s * p) * (s * p + 2 * r * q) / 4;
        } else if (p == 0) {
          n[a] = (1 - s * s) * (1 + r * q) / 2;
          dn[a][0] = -s * (1 + r * q);
          dn[a][1] = q * (1 - s * s) / 2;
        } else {
          n[a] = (1 + s * p) * (1 - r * r) / 2;
          dn[a][0] = p * (1 - r * r) / 2;
          dn[a][1] = -r * (1 + s * p);
        }
      }
      return true;
    }
    case Topology::hex:
      for (std::size_t a = 0; a < 8; ++a) {
        const Real p = quad_xi[a % 4], q = quad_eta[a % 4], z = a < 4 ? -1 : 1;
        const Real fx = 1 + p * xi[0], fy = 1 + q * xi[1], fz = 1 + z * xi[2];
        n[a] = fx * fy * fz / 8;
        dn[a][0] = p * fy * fz / 8;
        dn[a][1] = fx * q * fz / 8;
        dn[a][2] = fx * fy * z / 8;
      }
      return true;
    case Topology::prism: {
      const Real l[3] = {-(xi[0] + xi[1]) / 2, (1 + xi[0]) / 2, (1 + xi[1]) / 2};
      const Real dl[3][2] = {{Real{-0.5}, Real{-0.5}}, {Real{0.5}, 0}, {0, Real{0.5}}};
      for (std::size_t a = 0; a < 6; ++a) {
        const std::size_t i = a % 3;
        const Real z = a < 3 ? -1 : 1, fz = (1 + z * xi[2]) / 2;
        n[a] = l[i] * fz;
        dn[a][0] = dl[i][0] * fz;
        dn[a][1] = dl[i][1] * fz;
        dn[a][2] = l[i] * z / 2;
      }
      return true;
    }
    case Topology::pyramid: {
      const Real u = 1 - xi[2];
      n[4] = xi[2];
      dn[4][2] = 1;
      if (std::abs(u) < 1e-14) {
        for (std::size_t a = 0; a < 4; ++a) n[a] = 0;
        return false;
      }
      for (std::size_t a = 0; a < 4; ++a) {
        const Real p = quad_xi[a], q = quad_eta[a];
        const Real fx = u + p * xi[0], fy = u + q * xi[1];
        n[a] = fx * fy / (4 * u);
        dn[a][0] = p * fy / (4 * u);
        dn[a][1] = q * fx / (4 * u);
        dn[a][2] = -((fx + fy) / (4 * u) - fx * fy / (4 * u * u));
      }
      return true;
    }
    default: return false;
  }
}

V3 cross(const V3& a, const V3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Real dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

}  // namespace

ElementMap::ElementMap(Topology t, std::vector<V3> nodes, const std::optional<MeshGeometry>& surface)
    : t_(t), rdim_(reference_dim_of(t)), nodes_(std::move(nodes)) {
  const std::size_t want = nodes_of(t);
  const bool quadratic_quad = t == Topology::quad && (nodes_.size() == 8 || nodes_.size() == 9);
  if (want == 0 || (nodes_.size() != want && !quadratic_quad)) {
    throw std::invalid_argument(std::string("sample: no map for a ") + name(t) + " of " +
                                std::to_string(nodes_.size()) + " nodes");
  }
  if (surface && surface->kind == MeshGeometry::Kind::sphere && rdim_ <= 2) {
    sphere_ = true;
    center_ = surface->center;
    radius_ = surface->radius;
  }
}

Mapped ElementMap::at(const Real* xi, bool measure) const {
  Real n[9];
  Real dn[9][3];
  const bool regular = shape(t_, nodes_.size(), xi, n, dn);
  Mapped out;
  std::array<V3, 3> a{};
  for (std::size_t k = 0; k < nodes_.size(); ++k) {
    for (int c = 0; c < 3; ++c) {
      out.x[c] += n[k] * nodes_[k][c];
      for (int j = 0; j < rdim_; ++j) a[j][c] += dn[k][j] * nodes_[k][c];
    }
  }
  if (sphere_) {
    V3 y{out.x[0] - center_[0], out.x[1] - center_[1], out.x[2] - center_[2]};
    const Real r = std::sqrt(dot(y, y));
    for (Real& v : y) v /= r;
    for (int c = 0; c < 3; ++c) out.x[c] = center_[c] + radius_ * y[c];
    for (int j = 0; j < rdim_; ++j) {
      const Real along = dot(y, a[j]);
      for (int c = 0; c < 3; ++c) a[j][c] = radius_ / r * (a[j][c] - along * y[c]);
    }
  }
  if (!measure) return out;
  if (!regular) throw std::logic_error("sample: a measure asked for at a pyramid's apex");
  if (rdim_ == 1) {
    out.measure = std::sqrt(dot(a[0], a[0]));
  } else if (rdim_ == 2) {
    const V3 c = cross(a[0], a[1]);
    out.measure = std::sqrt(dot(c, c));
  } else {
    out.measure = std::abs(dot(a[0], cross(a[1], a[2])));
  }
  return out;
}

void gauss_legendre(int m, std::vector<Real>& x, std::vector<Real>& w) {
  if (m < 1) throw std::invalid_argument("gauss_legendre: " + std::to_string(m) + " points");
  x.assign(static_cast<std::size_t>(m), 0);
  w.assign(static_cast<std::size_t>(m), 0);
  for (int i = 0; i < (m + 1) / 2; ++i) {
    // Newton from Tricomi's estimate of root i (descending), then mirrored.
    Real z = std::cos(std::numbers::pi_v<Real> * (i + Real{0.75}) / (m + Real{0.5}));
    Real dp = 0;
    for (int it = 0; it < 100; ++it) {
      Real p0 = 1, p1 = z;
      for (int k = 2; k <= m; ++k) {
        const Real p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      if (m == 1) p0 = 1;
      dp = m * (z * p1 - p0) / (z * z - 1);
      const Real step = p1 / dp;
      z -= step;
      if (std::abs(step) <= 1e-16) break;
    }
    {
      Real p0 = 1, p1 = z;
      for (int k = 2; k <= m; ++k) {
        const Real p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      if (m == 1) p0 = 1;
      dp = m * (z * p1 - p0) / (z * z - 1);
    }
    const auto lo = static_cast<std::size_t>(i), hi = static_cast<std::size_t>(m - 1 - i);
    const Real weight = 2 / ((1 - z * z) * dp * dp);
    x[lo] = -z;
    x[hi] = z;
    w[lo] = w[hi] = weight;
  }
  if (m % 2 == 1) x[static_cast<std::size_t>(m / 2)] = 0;
}

void reference_rule(Topology t, int degree, std::vector<Real>& xi, std::vector<Real>& w) {
  if (degree < 0 || degree > 60) {
    throw std::invalid_argument("sample: quadrature degree " + std::to_string(degree) + " is not in [0, 60]");
  }
  xi.clear();
  w.clear();
  const auto points = [](int exact) { return exact / 2 + 1; };  // 2m - 1 >= exact
  std::vector<Real> a, wa, b, wb, c, wc;
  switch (t) {
    case Topology::edge:
      gauss_legendre(points(degree), a, wa);
      for (std::size_t i = 0; i < a.size(); ++i) {
        xi.insert(xi.end(), {a[i], 0, 0});
        w.push_back(wa[i]);
      }
      return;
    case Topology::quad:
    case Topology::hex: {
      gauss_legendre(points(degree), a, wa);
      const std::size_t m = a.size(), mk = t == Topology::hex ? m : 1;
      for (std::size_t k = 0; k < mk; ++k) {
        for (std::size_t j = 0; j < m; ++j) {
          for (std::size_t i = 0; i < m; ++i) {
            xi.insert(xi.end(), {a[i], a[j], t == Topology::hex ? a[k] : 0});
            w.push_back(wa[i] * wa[j] * (t == Topology::hex ? wa[k] : 1));
          }
        }
      }
      return;
    }
    case Topology::tri:
    case Topology::prism: {
      // (a, b) in [-1, 1]^2 to the triangle: xi = (1 + a)(1 - b)/2 - 1, eta = b,
      // Jacobian (1 - b)/2, one degree more in b.
      gauss_legendre(points(degree), a, wa);
      gauss_legendre(points(degree + 1), b, wb);
      if (t == Topology::prism) {
        gauss_legendre(points(degree), c, wc);
      } else {
        c = {0};
        wc = {1};
      }
      for (std::size_t k = 0; k < c.size(); ++k) {
        for (std::size_t j = 0; j < b.size(); ++j) {
          for (std::size_t i = 0; i < a.size(); ++i) {
            xi.insert(xi.end(), {(1 + a[i]) * (1 - b[j]) / 2 - 1, b[j], c[k]});
            w.push_back(wa[i] * wb[j] * wc[k] * (1 - b[j]) / 2);
          }
        }
      }
      return;
    }
    case Topology::tet:
      // xi = (1 + a)(1 - b)(1 - c)/4 - 1, eta = (1 + b)(1 - c)/2 - 1, zeta = c,
      // Jacobian (1 - b)/2 ((1 - c)/2)^2.
      gauss_legendre(points(degree), a, wa);
      gauss_legendre(points(degree + 1), b, wb);
      gauss_legendre(points(degree + 2), c, wc);
      for (std::size_t k = 0; k < c.size(); ++k) {
        for (std::size_t j = 0; j < b.size(); ++j) {
          for (std::size_t i = 0; i < a.size(); ++i) {
            xi.insert(xi.end(), {(1 + a[i]) * (1 - b[j]) * (1 - c[k]) / 4 - 1, (1 + b[j]) * (1 - c[k]) / 2 - 1, c[k]});
            const Real h = (1 - c[k]) / 2;
            w.push_back(wa[i] * wb[j] * wc[k] * (1 - b[j]) / 2 * h * h);
          }
        }
      }
      return;
    case Topology::pyramid:
      // z = (1 + c)/2, xi = a (1 - z), eta = b (1 - z), Jacobian (1 - z)^2 / 2.
      gauss_legendre(points(degree), a, wa);
      gauss_legendre(points(degree + 2), c, wc);
      for (std::size_t k = 0; k < c.size(); ++k) {
        const Real z = (1 + c[k]) / 2;
        for (std::size_t j = 0; j < a.size(); ++j) {
          for (std::size_t i = 0; i < a.size(); ++i) {
            xi.insert(xi.end(), {a[i] * (1 - z), a[j] * (1 - z), z});
            w.push_back(wa[i] * wa[j] * wc[k] * (1 - z) * (1 - z) / 2);
          }
        }
      }
      return;
    default:
      throw std::invalid_argument(std::string("sample: no reference cell for a ") + name(t));
  }
}

}  // namespace soma::detail::sample
