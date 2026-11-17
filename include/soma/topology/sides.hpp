// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TOPOLOGY_SIDES_HPP_
#define SOMA_TOPOLOGY_SIDES_HPP_

/// \file
/// The sides of an entity as mesh vertex ids, in the entity's canonical order:
/// the canonical tables for fixed topologies, the vertex cycle for polygon
/// edges. Device-callable.

#include <Kokkos_Core.hpp>

#include "soma/core/mesh_view.hpp"
#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"
#include "soma/topology/canonical.hpp"

namespace soma::topology {

struct Side {
  Topology topology = Topology::vertex;
  int n = 0;
  LO v[4] = {0, 0, 0, 0};
};

/// Number of sides of dimension k < d of entity e of dimension d.
template <class M>
KOKKOS_INLINE_FUNCTION int side_count(const MeshView<M>& m, int d, LO e, int k) {
  const Topology t = m.topology(d, e);
  if (t == Topology::polygon) return k == 1 ? static_cast<int>(m.num_vertices(d, e)) : 0;
  return canonical::num_sides(t, k);
}

template <class M>
KOKKOS_INLINE_FUNCTION Side side_of(const MeshView<M>& m, int d, LO e, int k, int s) {
  Side out;
  const Topology t = m.topology(d, e);
  if (t == Topology::polygon) {
    const LO n = m.num_vertices(d, e);
    out.topology = Topology::edge;
    out.n = 2;
    out.v[0] = m.vertex(d, e, s);
    out.v[1] = m.vertex(d, e, (s + 1) % n);
    return out;
  }
  const canonical::SideDef def = canonical::side(t, k, s);
  out.topology = def.topology;
  out.n = def.num_vertices;
  for (int i = 0; i < out.n; ++i) out.v[i] = m.vertex(d, e, def.v[i]);
  return out;
}

/// The entity itself as a side: its first n <= 4 vertices.
template <class M>
KOKKOS_INLINE_FUNCTION Side entity_as_side(const MeshView<M>& m, int d, LO e) {
  Side out;
  out.topology = m.topology(d, e);
  out.n = static_cast<int>(m.num_vertices(d, e));
  if (out.n > 4) return out;
  for (int i = 0; i < out.n; ++i) out.v[i] = m.vertex(d, e, i);
  return out;
}

KOKKOS_INLINE_FUNCTION LO min_vertex(const Side& s) {
  LO m = s.v[0];
  for (int i = 1; i < s.n; ++i) m = s.v[i] < m ? s.v[i] : m;
  return m;
}

/// Same vertex set, for n <= 4 vertices without repeats.
KOKKOS_INLINE_FUNCTION bool same_set(const LO* a, const LO* b, int n) {
  for (int i = 0; i < n; ++i) {
    bool found = false;
    for (int j = 0; j < n; ++j) found = found || a[i] == b[j];
    if (!found) return false;
  }
  return true;
}

}  // namespace soma::topology

#endif  // SOMA_TOPOLOGY_SIDES_HPP_
