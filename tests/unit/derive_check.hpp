// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_DERIVE_CHECK_HPP_
#define SOMA_TESTS_DERIVE_CHECK_HPP_

// Independent checks of derived sub-entities: a std::set of sorted vertex
// tuples from the canonical tables, and orientation by brute-force rotation.

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "adjacency_check.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/topology.hpp"
#include "soma/topology/canonical.hpp"

namespace soma_test {

using Tuple = std::vector<soma::LO>;

inline Tuple sorted(Tuple t) {
  std::sort(t.begin(), t.end());
  return t;
}

// The vertices of side s of dimension k of element e, in the element's order.
inline Tuple side_tuple(const std::vector<soma::Offset>& off, const std::vector<soma::LO>& conn,
                        soma::Topology t, soma::LO e, int k, int s) {
  const auto b = static_cast<std::size_t>(off[static_cast<std::size_t>(e)]);
  const auto n = static_cast<std::size_t>(off[static_cast<std::size_t>(e) + 1]) - b;
  if (t == soma::Topology::polygon) return {conn[b + static_cast<std::size_t>(s)], conn[b + (static_cast<std::size_t>(s) + 1) % n]};
  const auto def = soma::canonical::side(t, k, s);
  Tuple out;
  for (int i = 0; i < def.num_vertices; ++i) out.push_back(conn[b + def.v[i]]);
  return out;
}

inline int side_total(soma::Topology t, soma::LO arity, int k) {
  if (t == soma::Topology::polygon) return k == 1 ? static_cast<int>(arity) : 0;
  return soma::canonical::num_sides(t, k);
}

// (offset << 1) | reversed of b against a, found by trying every rotation in
// both directions.
inline int brute_force_code(const Tuple& a, const Tuple& b) {
  const int n = static_cast<int>(a.size());
  for (int r = 0; r < 2; ++r) {
    for (int off = 0; off < n; ++off) {
      bool ok = true;
      for (int i = 0; i < n && ok; ++i) {
        const int at = r == 0 ? (off + i) % n : (off - i + n) % n;
        ok = a[static_cast<std::size_t>(at)] == b[static_cast<std::size_t>(i)];
      }
      if (ok && !(n == 2 && r == 1 && off == 0) && !(n == 2 && r == 0 && off == 1)) {
        return (off << 1) | r;
      }
    }
  }
  return -1;
}

struct Expected {
  std::set<Tuple> unique;      // sorted vertex tuples
  std::map<Tuple, int> users;  // incidences per tuple
};

inline Expected expected_sides(const soma::HostMesh& mesh, int k) {
  const int d = mesh.dim();
  const auto off = to_vector(mesh.conn_offsets(d));
  const auto conn = to_vector(mesh.conn(d));
  const auto topo = to_vector(mesh.topology(d));
  Expected out;
  for (soma::LO e = 0; e < mesh.count(d); ++e) {
    const auto t = static_cast<soma::Topology>(topo[static_cast<std::size_t>(e)]);
    const auto arity = static_cast<soma::LO>(off[static_cast<std::size_t>(e) + 1] - off[static_cast<std::size_t>(e)]);
    for (int s = 0; s < side_total(t, arity, k); ++s) {
      const Tuple key = sorted(side_tuple(off, conn, t, e, k, s));
      out.unique.insert(key);
      ++out.users[key];
    }
  }
  return out;
}

}  // namespace soma_test

#endif  // SOMA_TESTS_DERIVE_CHECK_HPP_
