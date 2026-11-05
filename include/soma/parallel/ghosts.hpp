// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_GHOSTS_HPP_
#define SOMA_PARALLEL_GHOSTS_HPP_

// Ghost layers of a distributed mesh (design §21.4). Two elements of the
// mesh's dimension are joined when they have a common side of dimension
// bridge_dim (by its corners, held or not); with layer 0 the elements a rank
// holds other than as ghosts, layer i is every element held elsewhere joined
// to layer i - 1. The rule does not depend on the partition.

#include <cstddef>
#include <span>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

struct GhostOptions {
  int layers = 1;      // 0: no ghosts
  int bridge_dim = 0;  // below the mesh's dimension
  bool sets = false;   // ghosts join the sets their owners' entities are in
};

namespace detail {

// For each element of the mesh's dimension, by lid, the ranks whose ghosts
// include it, ascending; only owners list ranks.
struct GhostTargets {
  std::vector<std::size_t> offsets;
  std::vector<int> ranks;

  [[nodiscard]] std::span<const int> of(LO e) const {
    const auto i = static_cast<std::size_t>(e);
    return {ranks.data() + offsets[i], offsets[i + 1] - offsets[i]};
  }
};

// Collective: one round per layer. A rank sends each side of dimension
// bridge_dim of its frontier elements to every other holder of the side's
// lowest vertex (from soma:sharing, or as told with a ghost); each holder
// answers with the elements it owns that have the side and the holders of
// their vertices; the asker keeps those it does not hold. Every owner of an
// answer holds that vertex, so answers come from any rank, however thin the
// parts between. Throws std::invalid_argument on every rank for options that
// differ between ranks, a negative layer count, a bridge_dim outside
// [0, dim), or a mesh without sharing at P > 1.
[[nodiscard]] GhostTargets ghost_targets(const Comm& comm, const HostMesh& mesh, const GhostOptions& options);

}  // namespace detail

}  // namespace soma

#endif  // SOMA_PARALLEL_GHOSTS_HPP_
