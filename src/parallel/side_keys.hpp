// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_PARALLEL_SIDE_KEYS_HPP_
#define SOMA_SRC_PARALLEL_SIDE_KEYS_HPP_

// The sides of an element named by their corners' gids, whether or not the
// mesh holds them: an entity of dimension k shared by two elements has the
// same key in both.

#include <algorithm>
#include <array>
#include <limits>

#include "soma/core/mesh.hpp"
#include "soma/topology/canonical.hpp"

namespace soma::detail {

// Sorted corner gids, padded with the largest GO.
using SideKey = std::array<GO, 4>;

// Calls f(key) for each side of dimension k of element e of dimension d: its
// corners for k = 0, a polygon's edges between consecutive corners.
template <class F>
void for_each_side(const HostMesh& mesh, int d, LO e, int k, F&& f) {
  const auto off = mesh.conn_offsets(d);
  const auto conn = mesh.conn(d);
  const auto vgid = mesh.gid(0);
  const Offset b = off(e), n = off(e + 1) - off(e);
  constexpr GO pad = std::numeric_limits<GO>::max();
  const auto at = [&](Offset q) { return vgid(conn(b + q)); };
  if (k == 0) {
    for (Offset q = 0; q < n; ++q) f(SideKey{at(q), pad, pad, pad});
    return;
  }
  const auto t = static_cast<Topology>(mesh.topology(d)(e));
  if (t == Topology::polygon) {
    if (k != 1) return;
    for (Offset q = 0; q < n; ++q) {
      SideKey key{at(q), at((q + 1) % n), pad, pad};
      std::sort(key.begin(), key.end());
      f(key);
    }
    return;
  }
  for (int i = 0; i < canonical::num_sides(t, k); ++i) {
    const canonical::SideDef s = canonical::side(t, k, i);
    SideKey key{pad, pad, pad, pad};
    for (int j = 0; j < s.num_vertices; ++j) key[static_cast<std::size_t>(j)] = at(s.v[j]);
    std::sort(key.begin(), key.end());
    f(key);
  }
}

// The key of entity e of dimension k from its own corners.
inline SideKey key_of(const HostMesh& mesh, int k, LO e) {
  const auto off = mesh.conn_offsets(k);
  const auto conn = mesh.conn(k);
  const auto vgid = mesh.gid(0);
  SideKey key;
  key.fill(std::numeric_limits<GO>::max());
  for (Offset q = off(e); q < off(e + 1) && q - off(e) < 4; ++q) key[static_cast<std::size_t>(q - off(e))] = vgid(conn(q));
  std::sort(key.begin(), key.end());
  return key;
}

}  // namespace soma::detail

#endif  // SOMA_SRC_PARALLEL_SIDE_KEYS_HPP_
