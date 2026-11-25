// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/components.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "soma/parallel/halo_exchange.hpp"

namespace soma {

std::vector<std::uint8_t> edge_reachable(const Comm& comm, HostMesh& mesh, const std::vector<std::uint8_t>& keep,
                                         const std::vector<std::uint8_t>& seed) {
  if (comm.size() == 1 && mesh.dim() == 2 && !mesh.has_adjacency(2, 1)) {
    build_adjacencies(mesh, AdjacencyRequest{}.add(2, 1));
  }
  const LO owned = mesh.segment_table(2).owned_end, total = mesh.count(2);
  const auto n = static_cast<std::size_t>(owned);
  // 1: sizes, 2: no adjacency, 3: an element without edges; the largest any rank finds is reported.
  int error = 0;
  if (keep.size() != n || seed.size() != n) {
    error = 1;
  } else if (!mesh.has_adjacency(2, 1)) {
    error = 2;
  }
  const auto view = mesh.view();
  const LO edges = mesh.count(1);
  if (error == 0) {
    for (LO e = 0; e < total && error == 0; ++e) {
      if (view.adj(2, 1, e).size == 0) error = 3;
    }
  }
  error = comm.allreduce(error, Op::max);
  if (error == 1) {
    throw std::invalid_argument("edge_reachable: keep and seed must have one flag per owned element on every rank");
  }
  if (error == 2) throw std::invalid_argument("edge_reachable: the mesh has no (2, 1) adjacency");
  if (error == 3) throw std::invalid_argument("edge_reachable: an element has no edges (ghosts or edges missing)");

  // The (up to two) elements at each edge, over every local element.
  std::vector<LO> at_edge(2 * static_cast<std::size_t>(edges), -1);
  for (LO e = 0; e < total; ++e) {
    const auto row = view.adj(2, 1, e);
    for (LO s = 0; s < row.size; ++s) {
      LO* slot = &at_edge[2 * static_cast<std::size_t>(row[s])];
      if (slot[0] < 0) {
        slot[0] = e;
      } else if (slot[0] != e && slot[1] < 0) {
        slot[1] = e;
      }
    }
  }

  View2<int, Kokkos::HostSpace> kept("soma::edge_reachable::keep", static_cast<std::size_t>(total), 1);
  View2<int, Kokkos::HostSpace> reach("soma::edge_reachable::reach", static_cast<std::size_t>(total), 1);
  for (std::size_t e = 0; e < n; ++e) {
    kept(e, 0) = keep[e] != 0;
    reach(e, 0) = keep[e] != 0 && seed[e] != 0;
  }
  // Flood from every reached element across the edges both of whose elements are kept.
  const auto flood = [&] {
    std::vector<LO> stack;
    for (LO e = 0; e < total; ++e) {
      if (reach(e, 0)) stack.push_back(e);
    }
    while (!stack.empty()) {
      const LO e = stack.back();
      stack.pop_back();
      const auto row = view.adj(2, 1, e);
      for (LO s = 0; s < row.size; ++s) {
        for (int k = 0; k < 2; ++k) {
          const LO f = at_edge[2 * static_cast<std::size_t>(row[s]) + static_cast<std::size_t>(k)];
          if (f >= 0 && kept(f, 0) && !reach(f, 0)) {
            reach(f, 0) = 1;
            stack.push_back(f);
          }
        }
      }
    }
  };
  if (comm.size() > 1) {
    const HaloExchange<Kokkos::HostSpace> halo(comm, mesh, 2);
    halo.update(comm, kept);
    for (;;) {
      halo.update(comm, reach);
      std::vector<int> before(n);
      for (std::size_t e = 0; e < n; ++e) before[e] = reach(e, 0);
      flood();
      int changed = 0;
      for (std::size_t e = 0; e < n; ++e) changed |= reach(e, 0) != before[e];
      if (comm.allreduce(changed, Op::max) == 0) break;
    }
  } else {
    flood();
  }
  std::vector<std::uint8_t> out(n);
  for (std::size_t e = 0; e < n; ++e) out[e] = static_cast<std::uint8_t>(reach(e, 0));
  return out;
}

}  // namespace soma
