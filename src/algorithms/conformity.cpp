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
LO hanging_edges(const MeshView<Kokkos::HostSpace>& m, int dim, LO cells, const Coords& x, const OneSided& one_sided) {
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
        if (rest <= off * off) ++found;
      }
    }
  }
  return found;
}

}  // namespace

LO nonconforming_sides(const Mesh<Kokkos::HostSpace>& mesh) {
  const int dim = mesh.dim();
  if (dim < 2) return 0;
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
  LO found = hanging_edges(m, dim, cells, mesh.coords(), [&](const topology::Side& sd, LO) { return once(key_of(sd)); });
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
        if (once(t)) ++found;
      }
    }
  }
  return found;
}

}  // namespace soma::detail
