// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/conformity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "soma/core/mesh_view.hpp"
#include "soma/topology/sides.hpp"

namespace soma::detail {

namespace {

using Key = std::array<LO, 4>;  // a side's corners, sorted, -1 padded

Key key_of(const topology::Side& s) {
  Key k{-1, -1, -1, -1};
  std::copy(s.v, s.v + s.n, k.begin());
  std::sort(k.begin(), k.begin() + s.n);
  return k;
}

// The edges of one-sided sides that have the end of another of them inside:
// each side of an edge (a, m) on an edge (a, b), m strictly between a and b,
// as a conforming mesh has none. A refined cell across a coarse side shares
// its corner a, so a ghost layer through vertices holds the fine edge on the
// rank that owns the coarse cell.
template <class Coords, class OneSided>
LO hanging_edges(const MeshView<Kokkos::HostSpace>& m, int dim, LO cells, const Coords& x, const OneSided& one_sided,
                 std::vector<LO>* inside) {
  std::vector<std::pair<LO, LO>> ends;  // (a, b) and (b, a) of every edge of a one-sided side
  for (LO e = 0; e < cells; ++e) {
    for (int s = 0; s < topology::side_count(m, dim, e, dim - 1); ++s) {
      const topology::Side sd = topology::side_of(m, dim, e, dim - 1, s);
      if (!one_sided(sd, e)) continue;
      for (int k = 0; k < (sd.n == 2 ? 1 : sd.n); ++k) {
        const LO a = sd.v[k], b = sd.v[(k + 1) % sd.n];
        ends.emplace_back(a, b);
        ends.emplace_back(b, a);
      }
    }
  }
  std::sort(ends.begin(), ends.end());
  ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
  LO found = 0;
  for (std::size_t lo = 0, hi = 0; lo < ends.size(); lo = hi) {
    const LO a = ends[lo].first;
    while (hi < ends.size() && ends[hi].first == a) ++hi;
    for (std::size_t i = lo; i < hi; ++i) {
      std::array<Real, 3> u{};
      Real uu = 0, size = 0;
      for (int q = 0; q < 3; ++q) {
        u[static_cast<std::size_t>(q)] = x(ends[i].second, q) - x(a, q);
        uu += u[static_cast<std::size_t>(q)] * u[static_cast<std::size_t>(q)];
        size = std::max({size, std::abs(x(a, q)), std::abs(x(ends[i].second, q))});
      }
      // Off the edge by rounding only: of its length, and of the coordinates.
      const Real off = 1e-10 * std::sqrt(uu) + 8 * std::numeric_limits<Real>::epsilon() * size;
      for (std::size_t j = lo; j < hi; ++j) {
        if (j == i) continue;
        Real uw = 0, rest = 0;
        for (int q = 0; q < 3; ++q) uw += u[static_cast<std::size_t>(q)] * (x(ends[j].second, q) - x(a, q));
        const Real at = uw / uu;
        if (!(at > 1e-6 && at < 1 - 1e-6)) continue;
        for (int q = 0; q < 3; ++q) {
          const Real r = x(ends[j].second, q) - x(a, q) - at * u[static_cast<std::size_t>(q)];
          rest += r * r;
        }
        if (rest <= off * off) {
          ++found;
          if (inside) inside->push_back(ends[j].second);
        }
      }
    }
  }
  return found;
}

}  // namespace

LO nonconforming_sides(const Mesh<Kokkos::HostSpace>& mesh) {
  const Nonconforming n = nonconforming(mesh);
  return n.hanging + n.split_quads;
}

Nonconforming nonconforming(const Mesh<Kokkos::HostSpace>& mesh) {
  Nonconforming out;
  const int dim = mesh.dim();
  if (dim < 2) return out;
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  const LO cells = m.count(dim);
  std::vector<Key> keys;
  for (LO e = 0; e < cells; ++e) {
    for (int s = 0; s < topology::side_count(m, dim, e, dim - 1); ++s) keys.push_back(key_of(topology::side_of(m, dim, e, dim - 1, s)));
  }
  std::sort(keys.begin(), keys.end());
  const auto once = [&](const Key& k) {
    const auto [lo, hi] = std::equal_range(keys.begin(), keys.end(), k);
    return hi - lo == 1;
  };
  out.hanging = hanging_edges(m, dim, cells, mesh.coords(), [&](const topology::Side& sd, LO) { return once(key_of(sd)); },
                              &out.vertices);
  std::sort(out.vertices.begin(), out.vertices.end());
  out.vertices.erase(std::unique(out.vertices.begin(), out.vertices.end()), out.vertices.end());
  if (dim == 3) {
    // A one-sided quad and a one-sided triangle on three of its corners.
    for (std::size_t i = 0; i < keys.size(); ++i) {
      const Key& q = keys[i];
      if (q[3] < 0 || (i > 0 && keys[i - 1] == q) || (i + 1 < keys.size() && keys[i + 1] == q)) continue;
      for (int skip = 0; skip < 4; ++skip) {
        Key t{-1, -1, -1, -1};
        for (int j = 0, k = 0; j < 4; ++j) {
          if (j != skip) t[static_cast<std::size_t>(k++)] = q[static_cast<std::size_t>(j)];
        }
        if (once(t)) ++out.split_quads;
      }
    }
  }
  return out;
}

SplitSides split_sides(const Mesh<Kokkos::HostSpace>& mesh) {
  SplitSides out;
  const int dim = mesh.dim();
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  const LO cells = m.count(dim);
  out.code.assign(static_cast<std::size_t>(cells), 0);
  if (dim != 3) return out;
  std::vector<Key> keys;
  for (LO e = 0; e < cells; ++e) {
    for (int s = 0; s < topology::side_count(m, dim, e, dim - 1); ++s) keys.push_back(key_of(topology::side_of(m, dim, e, dim - 1, s)));
  }
  std::sort(keys.begin(), keys.end());
  const auto once = [&](const Key& k) {
    const auto [lo, hi] = std::equal_range(keys.begin(), keys.end(), k);
    return hi - lo == 1;
  };
  const auto triangle = [&](LO a, LO b, LO c) {
    topology::Side t;
    t.n = 3;
    t.v[0] = a;
    t.v[1] = b;
    t.v[2] = c;
    return once(key_of(t));
  };
  const LO owned = mesh.segment_table(dim).owned_end;
  for (LO e = 0; e < cells; ++e) {
    if (m.topology(dim, e) != Topology::hex) continue;
    for (int s = 0; s < 6; ++s) {
      const topology::Side q = topology::side_of(m, dim, e, dim - 1, s);
      if (!once(key_of(q))) continue;
      const LO* v = q.v;
      const bool a = triangle(v[0], v[1], v[2]), b = triangle(v[0], v[2], v[3]);
      const bool c = triangle(v[0], v[1], v[3]), d = triangle(v[1], v[2], v[3]);
      const bool by02 = a && b && !c && !d, by13 = c && d && !a && !b;
      if (by02 || by13) {
        out.code[static_cast<std::size_t>(e)] |= static_cast<std::uint16_t>((by02 ? 1u : 2u) << (2 * s));
      } else if ((a || b || c || d) && e < owned) {
        ++out.odd;
      }
    }
  }
  return out;
}

std::vector<std::uint8_t> conforming_skin(const Mesh<Kokkos::HostSpace>& mesh) {
  const int dim = mesh.dim();
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  const LO cells = m.count(dim), nv = m.count(0);
  const auto x = mesh.coords();
  std::vector<Key> keys;
  for (LO e = 0; e < cells; ++e) {
    for (int s = 0; s < topology::side_count(m, dim, e, dim - 1); ++s) keys.push_back(key_of(topology::side_of(m, dim, e, dim - 1, s)));
  }
  std::sort(keys.begin(), keys.end());
  const auto once = [&](const Key& k) {
    const auto [lo, hi] = std::equal_range(keys.begin(), keys.end(), k);
    return hi - lo == 1;
  };
  struct Lone {
    topology::Side side;
    LO cell;
  };
  std::vector<Lone> lone;
  std::vector<std::vector<std::size_t>> at(static_cast<std::size_t>(nv));
  for (LO e = 0; e < cells; ++e) {
    for (int s = 0; s < topology::side_count(m, dim, e, dim - 1); ++s) {
      const topology::Side sd = topology::side_of(m, dim, e, dim - 1, s);
      if (!once(key_of(sd))) continue;
      for (int j = 0; j < sd.n; ++j) at[static_cast<std::size_t>(sd.v[j])].push_back(lone.size());
      lone.push_back({sd, e});
    }
  }
  // Whether vertex v lies on side sd, to rounding of its size (a warped
  // quad's warp allowed off its mean plane).
  const auto on_side = [&](const topology::Side& sd, LO v) {
    const auto p = [&](int j, int i) { return x(sd.v[j], i); };
    Real size = 0;
    for (int j = 1; j < sd.n; ++j) {
      Real d = 0;
      for (int i = 0; i < 3; ++i) d += (p(j, i) - p(0, i)) * (p(j, i) - p(0, i));
      size = std::max(size, std::sqrt(d));
    }
    const Real tol = 1e-9 * size;
    if (dim == 2) {
      const Real u0 = p(1, 0) - p(0, 0), u1 = p(1, 1) - p(0, 1), w0 = x(v, 0) - p(0, 0), w1 = x(v, 1) - p(0, 1);
      const Real uu = u0 * u0 + u1 * u1, t = (u0 * w0 + u1 * w1) / uu;
      return t >= -1e-9 && t <= 1 + 1e-9 && std::abs(u0 * w1 - u1 * w0) <= tol * std::sqrt(uu);
    }
    std::array<Real, 3> a{}, c{};
    const int u0 = sd.n == 3 ? 1 : 2, w0 = sd.n == 3 ? 2 : 3, w1 = sd.n == 3 ? 0 : 1;
    const std::array<Real, 3> u{p(u0, 0) - p(0, 0), p(u0, 1) - p(0, 1), p(u0, 2) - p(0, 2)};
    const std::array<Real, 3> w{p(w0, 0) - p(w1, 0), p(w0, 1) - p(w1, 1), p(w0, 2) - p(w1, 2)};
    a = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
    const Real norm = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    for (int i = 0; i < 3; ++i) {
      Real sum = 0;
      for (int j = 0; j < sd.n; ++j) sum += p(j, i);
      c[static_cast<std::size_t>(i)] = sum / sd.n;
      a[static_cast<std::size_t>(i)] /= norm;
    }
    const auto off = [&](LO q) {
      Real d = 0;
      for (int i = 0; i < 3; ++i) d += (x(q, i) - c[static_cast<std::size_t>(i)]) * a[static_cast<std::size_t>(i)];
      return std::abs(d);
    };
    Real warp = 0;
    for (int j = 0; j < sd.n; ++j) warp = std::max(warp, off(sd.v[j]));
    if (off(v) > warp + tol) return false;
    for (int j = 0; j < sd.n; ++j) {
      const int k = (j + 1) % sd.n;
      std::array<Real, 3> e{}, r{};
      for (int i = 0; i < 3; ++i) {
        e[static_cast<std::size_t>(i)] = p(k, i) - p(j, i);
        r[static_cast<std::size_t>(i)] = x(v, i) - p(j, i);
      }
      const std::array<Real, 3> cr{e[1] * r[2] - e[2] * r[1], e[2] * r[0] - e[0] * r[2], e[0] * r[1] - e[1] * r[0]};
      if (cr[0] * a[0] + cr[1] * a[1] + cr[2] * a[2] < -tol * size) return false;
    }
    return true;
  };
  // A one-sided side that another lies in, or that lies in another, is
  // between cells: not skin.
  std::vector<std::uint8_t> between(lone.size(), 0);
  for (std::size_t i = 0; i < lone.size(); ++i) {
    const topology::Side& big = lone[i].side;
    for (int j = 0; j < big.n; ++j) {
      for (const std::size_t k : at[static_cast<std::size_t>(big.v[j])]) {
        if (k == i || lone[k].cell == lone[i].cell) continue;
        const topology::Side& small = lone[k].side;
        bool inside = true;
        for (int q = 0; q < small.n && inside; ++q) inside = on_side(big, small.v[q]);
        if (inside) between[i] = between[k] = 1;
      }
    }
  }
  std::vector<std::uint8_t> skin(static_cast<std::size_t>(nv), 0);
  for (std::size_t i = 0; i < lone.size(); ++i) {
    if (between[i] != 0) continue;
    for (int j = 0; j < lone[i].side.n; ++j) skin[static_cast<std::size_t>(lone[i].side.v[j])] = 1;
  }
  return skin;
}

}  // namespace soma::detail
