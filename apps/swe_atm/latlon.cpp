// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "latlon.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "soma/fields/node_tables.hpp"

namespace swe {

namespace {

constexpr double quarter_pi = std::numbers::pi / 4;
using Vec = std::array<double, 3>;

double dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// An element of the cubed sphere: face axis k and side s, indices along the
// face's lattice axes a = k + 1 and b = k + 2 (mod 3), as cubed_sphere_mesh
// numbers them.
struct Element {
  int k, s, ia, ib;
};

Element element_of(int ne, soma::GO gid) {
  const std::int64_t e = gid - 1, per_face = static_cast<std::int64_t>(ne) * ne;
  const auto f = static_cast<int>(e / per_face), rem = static_cast<int>(e % per_face);
  return {f / 2, f % 2 == 1 ? 1 : -1, rem % ne, rem / ne};
}

soma::GO gid_of(int ne, const Element& el) {
  const int f = 2 * el.k + (el.s > 0 ? 1 : 0);
  return 1 + static_cast<soma::GO>(f) * ne * ne + el.ia + static_cast<soma::GO>(ne) * el.ib;
}

// The unit vector of lattice point (u, w) of the face: equiangular, as the
// mesh's vertices.
Vec lattice(int ne, const Element& el, int u, int w) {
  const auto at = [ne](int t) { return std::tan(quarter_pi * (2.0 * t / ne - 1)); };
  Vec p{};
  p[static_cast<std::size_t>(el.k)] = at(el.s > 0 ? ne : 0);
  p[static_cast<std::size_t>((el.k + 1) % 3)] = at(u);
  p[static_cast<std::size_t>((el.k + 2) % 3)] = at(w);
  const double l = std::sqrt(dot(p, p));
  return {p[0] / l, p[1] / l, p[2] / l};
}

// Corners in the mesh's order: counterclockwise from outside, corner 0 at
// (ia, ib), corner 1 along a on the + side and along b on the - side.
std::array<Vec, 4> corners(int ne, const Element& el) {
  const int i = el.ia, j = el.ib;
  if (el.s > 0) return {lattice(ne, el, i, j), lattice(ne, el, i + 1, j), lattice(ne, el, i + 1, j + 1), lattice(ne, el, i, j + 1)};
  return {lattice(ne, el, i, j), lattice(ne, el, i, j + 1), lattice(ne, el, i + 1, j + 1), lattice(ne, el, i + 1, j)};
}

// The bilinear map at (xi, eta) and its derivatives.
void bilinear(const std::array<Vec, 4>& c, double xi, double eta, Vec& y, Vec& dxi, Vec& deta) {
  const double n[4] = {(1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4, (1 + xi) * (1 + eta) / 4,
                       (1 - xi) * (1 + eta) / 4};
  const double nx[4] = {-(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4};
  const double ne[4] = {-(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4};
  for (std::size_t d = 0; d < 3; ++d) {
    y[d] = dxi[d] = deta[d] = 0;
    for (std::size_t q = 0; q < 4; ++q) {
      y[d] += n[q] * c[q][d];
      dxi[d] += nx[q] * c[q][d];
      deta[d] += ne[q] * c[q][d];
    }
  }
}

// Gauss-Newton on |y(xi, eta)|^-1 y(xi, eta) = x from (xi, eta).
void solve(const std::array<Vec, 4>& c, const Vec& x, double& xi, double& eta) {
  for (int it = 0; it < 20; ++it) {
    Vec y, a, b;
    bilinear(c, xi, eta, y, a, b);
    const double ly = std::sqrt(dot(y, y));
    Vec r, ja, jb;
    for (std::size_t d = 0; d < 3; ++d) r[d] = y[d] / ly - x[d];
    // d(y / |y|) = (I - y y^T / |y|^2) dy / |y|.
    const double ya = dot(y, a) / (ly * ly), yb = dot(y, b) / (ly * ly);
    for (std::size_t d = 0; d < 3; ++d) {
      ja[d] = (a[d] - ya * y[d]) / ly;
      jb[d] = (b[d] - yb * y[d]) / ly;
    }
    const double aa = dot(ja, ja), ab = dot(ja, jb), bb = dot(jb, jb), ra = dot(ja, r), rb = dot(jb, r);
    const double det = aa * bb - ab * ab;
    const double dxi = -(bb * ra - ab * rb) / det, deta = -(aa * rb - ab * ra) / det;
    xi += dxi;
    eta += deta;
    if (std::abs(dxi) + std::abs(deta) < 1e-14) break;
  }
}

}  // namespace

std::array<double, 3> element_point(int ne, soma::GO gid, double xi, double eta) {
  Vec y, a, b;
  bilinear(corners(ne, element_of(ne, gid)), xi, eta, y, a, b);
  const double l = std::sqrt(dot(y, y));
  return {y[0] / l, y[1] / l, y[2] / l};
}

Located locate(int ne, const std::array<double, 3>& point) {
  const double length = std::sqrt(dot(point, point));
  const Vec x = {point[0] / length, point[1] / length, point[2] / length};
  const double largest = std::max({std::abs(x[0]), std::abs(x[1]), std::abs(x[2])});
  Located best;
  for (int k = 0; k < 3; ++k) {
    const double xk = x[static_cast<std::size_t>(k)];
    if (std::abs(xk) < largest * (1 - 1e-12)) continue;  // another face, or a cube edge's neighbour
    const int s = xk > 0 ? 1 : -1;
    // Equiangular lattice coordinates in [0, ne]: elements' edges are their
    // integer lines.
    const auto coordinate = [&](int axis) {
      return ne / 2.0 * (std::atan(x[static_cast<std::size_t>(axis)] / std::abs(xk)) / quarter_pi + 1);
    };
    const double ta = coordinate((k + 1) % 3), tb = coordinate((k + 2) % 3);
    const auto candidates = [ne](double t) {
      std::array<int, 2> c = {std::clamp(static_cast<int>(std::floor(t)), 0, ne - 1), -1};
      const double nearest = std::round(t);
      if (std::abs(t - nearest) < 1e-9) {
        const int below = static_cast<int>(nearest) - 1, above = static_cast<int>(nearest);
        c = {below >= 0 ? below : above, above <= ne - 1 && below >= 0 ? above : -1};
      }
      return c;
    };
    for (const int ia : candidates(ta)) {
      if (ia < 0) continue;
      for (const int ib : candidates(tb)) {
        if (ib < 0) continue;
        const Element el{k, s, ia, ib};
        double xi = 2 * ((s > 0 ? ta : tb) - (s > 0 ? ia : ib)) - 1;
        double eta = 2 * ((s > 0 ? tb : ta) - (s > 0 ? ib : ia)) - 1;
        solve(corners(ne, el), x, xi, eta);
        if (std::abs(xi) > 1 + 1e-9 || std::abs(eta) > 1 + 1e-9) continue;
        const soma::GO gid = gid_of(ne, el);
        if (best.gid == 0 || gid < best.gid) best = {gid, std::clamp(xi, -1.0, 1.0), std::clamp(eta, -1.0, 1.0)};
      }
    }
  }
  if (best.gid == 0) throw std::logic_error("swe::locate: no element holds the point");
  return best;
}

std::vector<double> sample(const soma::Comm& comm, const Space& space, const soma::HostMesh& mesh, int ne,
                           const ConstValues& values, int component, std::span<const double> lat,
                           std::span<const double> lon) {
  const int p = space.order(), np = p + 1;
  const std::vector<double> nodes = soma::gll_points_1d(p);
  const auto gid = mesh.gid(2);
  std::unordered_map<soma::GO, soma::LO> owned;
  for (soma::LO e = 0; e < space.owned_elements(); ++e) owned.emplace(gid(e), e);
  const auto basis = [&](double t, std::vector<double>& l) {
    for (int i = 0; i < np; ++i) {
      double v = 1;
      for (int m = 0; m < np; ++m) {
        if (m != i) v *= (t - nodes[static_cast<std::size_t>(m)]) / (nodes[static_cast<std::size_t>(i)] - nodes[static_cast<std::size_t>(m)]);
      }
      l[static_cast<std::size_t>(i)] = v;
    }
  };
  std::vector<double> out(lat.size() * lon.size(), 0.0), li(static_cast<std::size_t>(np)), lj(static_cast<std::size_t>(np));
  const double degree = std::numbers::pi / 180;
  for (std::size_t i = 0; i < lat.size(); ++i) {
    const double t = lat[i] * degree;
    for (std::size_t j = 0; j < lon.size(); ++j) {
      const double l = lon[j] * degree;
      const Located at = locate(ne, {std::cos(t) * std::cos(l), std::cos(t) * std::sin(l), std::sin(t)});
      const auto e = owned.find(at.gid);
      if (e == owned.end()) continue;
      basis(at.xi, li);
      basis(at.eta, lj);
      const soma::LO first = e->second * np * np;
      double v = 0;
      for (int b = 0; b < np; ++b) {
        double row = 0;
        for (int a = 0; a < np; ++a) row += li[static_cast<std::size_t>(a)] * values(first + a + np * b, component);
        v += lj[static_cast<std::size_t>(b)] * row;
      }
      out[i * lon.size() + j] = v;
    }
  }
  // One rank holds each point; the others' zeros leave its value's bits.
  comm.allreduce(std::span<double>(out), soma::Op::sum);
  return out;
}

void write_npy(const std::filesystem::path& path, std::span<const double> data, std::size_t rows, std::size_t cols) {
  static_assert(std::endian::native == std::endian::little, "write_npy writes the host's doubles as '<f8'");
  if (data.size() != rows * cols) {
    throw std::invalid_argument("swe::write_npy: " + std::to_string(data.size()) + " values for " +
                                std::to_string(rows) + " x " + std::to_string(cols));
  }
  std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " +
                       std::to_string(cols) + "), }";
  // Padded with spaces and a newline to a multiple of 64 bytes with the
  // 10-byte preamble.
  header.append(64 - (10 + header.size() + 1) % 64, ' ');
  header += '\n';
  std::ofstream out(path, std::ios::binary);
  const auto length = static_cast<std::uint16_t>(header.size());
  out.write("\x93NUMPY\x01\x00", 8);
  const char size[2] = {static_cast<char>(length & 0xff), static_cast<char>(length >> 8)};
  out.write(size, 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(double)));
  if (!out) throw std::runtime_error("swe::write_npy: cannot write " + path.string());
}

}  // namespace swe
