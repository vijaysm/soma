// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_NODE_TABLES_HPP_
#define SOMA_FIELDS_NODE_TABLES_HPP_

// Reference-element node coordinates for high-order layouts.
//
// Tables are generated on the host once per key. Simplex GLL nodes use the warp
// and blend construction of T. Warburton, "An explicit construction of
// interpolation nodes on the simplex", J. Eng. Math. 56 (2006) 247-262, with
// the optimized parameters tabulated in Hesthaven and Warburton, Nodal
// Discontinuous Galerkin Methods (2008), section 6.1. The face warp of a
// tetrahedron uses the triangle parameter, so every triangular face carries the
// same node set whichever element it belongs to.
//
// Reference elements: edge, quad and hex are [-1,1]^d; triangle has vertices
// (-1,-1), (1,-1), (-1,1); tet has (-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1);
// prism is the triangle times [-1,1] with vertices 0,1,2 at z=-1 and 3,4,5 at
// z=+1; pyramid has a quad base at z=0 (vertices 0..3, counter-clockwise from
// above) and the apex 4 at (0,0,1). Corner i is canonical local vertex i.

#include <compare>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"
#include "soma/fields/layout.hpp"

namespace soma {

inline constexpr int max_optimized_simplex_order = 15;

struct NodeTableKey {
  Topology topology = Topology::edge;
  NodeFamily family = NodeFamily::gll;
  int order = 1;
  NodeOrdering ordering = NodeOrdering::tensor_lexicographic;
  bool serendipity = false;

  friend auto operator<=>(const NodeTableKey&, const NodeTableKey&) = default;
};

struct NodeTableHost {
  int dim = 0;
  int count = 0;
  std::vector<Real> x;  // column-major (count x dim)

  Real coord(int node, int d) const { return x[static_cast<std::size_t>(d * count + node)]; }
};

// The p+1 Gauss-Lobatto-Legendre points on [-1,1], ascending, exactly symmetric.
// Throws std::invalid_argument for p outside [1, max_node_order] and
// std::runtime_error if Newton's method fails to converge.
std::vector<Real> gll_points_1d(int p);

// Throws std::invalid_argument naming the combination when it is unsupported.
NodeTableHost make_node_table(const NodeTableKey& key);

// perm[i] is the position, in ordering from.ordering, of the node at position i of
// ordering `to`: coords_to(i) == coords_from(perm[i]).
std::vector<int> make_node_permutation(const NodeTableKey& from, NodeOrdering to);

template <class MemSpace>
struct NodeTableView {
  using Coords = Kokkos::View<const Real**, Kokkos::LayoutLeft, MemSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Coords x;
  int count = 0;
  int dim = 0;

  KOKKOS_INLINE_FUNCTION Real operator()(int node, int d) const { return x(node, d); }
};

template <class MemSpace>
class NodeTables {
 public:
  using Table = NodeTableView<MemSpace>;
  using Permutation = Kokkos::View<const int*, Kokkos::LayoutLeft, MemSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  // The returned views stay valid until clear() or Kokkos::finalize.
  static Table get(const NodeTableKey& key) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.tables.find(key);
    if (it == s.tables.end()) {
      const NodeTableHost host = make_node_table(key);
      Kokkos::View<Real**, Kokkos::LayoutLeft, MemSpace> d(
          Kokkos::view_alloc("soma::node_table", Kokkos::WithoutInitializing), static_cast<std::size_t>(host.count),
          static_cast<std::size_t>(host.dim));
      Kokkos::View<const Real**, Kokkos::LayoutLeft, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          h(host.x.data(), static_cast<std::size_t>(host.count), static_cast<std::size_t>(host.dim));
      Kokkos::deep_copy(d, h);
      it = s.tables.emplace(key, d).first;
      s.register_hook();
    }
    return Table{it->second, it->second.extent_int(0), it->second.extent_int(1)};
  }

  static Permutation permutation(const NodeTableKey& from, NodeOrdering to) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto k = std::make_pair(from, to);
    auto it = s.permutations.find(k);
    if (it == s.permutations.end()) {
      const std::vector<int> host = make_node_permutation(from, to);
      Kokkos::View<int*, Kokkos::LayoutLeft, MemSpace> d(
          Kokkos::view_alloc("soma::node_permutation", Kokkos::WithoutInitializing),
          host.size());
      Kokkos::View<const int*, Kokkos::LayoutLeft, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          h(host.data(), host.size());
      Kokkos::deep_copy(d, h);
      it = s.permutations.emplace(k, d).first;
      s.register_hook();
    }
    return Permutation(it->second);
  }

  static void clear() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.tables.clear();
    s.permutations.clear();
  }

 private:
  struct State {
    std::mutex mutex;
    std::map<NodeTableKey, Kokkos::View<Real**, Kokkos::LayoutLeft, MemSpace>> tables;
    std::map<std::pair<NodeTableKey, NodeOrdering>,
             Kokkos::View<int*, Kokkos::LayoutLeft, MemSpace>>
        permutations;
    bool hooked = false;

    void register_hook() {
      if (!hooked) {
        hooked = true;
        Kokkos::push_finalize_hook([] { NodeTables::clear_unlocked(); });
      }
    }
  };

  static State& state() {
    static State s;
    return s;
  }

  static void clear_unlocked() {
    State& s = state();
    s.tables.clear();
    s.permutations.clear();
  }
};

}  // namespace soma

#endif  // SOMA_FIELDS_NODE_TABLES_HPP_
