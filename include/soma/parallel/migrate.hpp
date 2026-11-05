// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_MIGRATE_HPP_
#define SOMA_PARALLEL_MIGRATE_HPP_

// Moving the elements of a distributed mesh between ranks with their closure
// (design §21.9).

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

namespace detail {

// Where the entities of a distributed mesh go when each element of its
// dimension goes to the ranks `to_ranks` lists for it, as rows over its lids
// (`to_offsets` has count + 1 entries from 0). Each entity is placed by its
// owner, and the others place nothing:
//   an element         on the ranks listed for it;
//   a lower entity     of a dimension given to the builder on any rank: on
//                      every rank receiving all its corners;
//   a vertex           on every rank receiving an entity it is a corner or a
//                      node of.
// What no rank receives stays with its owner. Collective. Throws
// std::invalid_argument on every rank if a rank's rows do not cover its
// elements or name a rank outside comm, or if the mesh has no sharing at
// P > 1 (build_local_part gives it).
struct Placement {
  std::array<std::vector<std::size_t>, 4> offsets;  // count(d) + 1 each, empty above the mesh's dimension
  std::array<std::vector<int>, 4> ranks;            // ascending within an entity

  [[nodiscard]] std::span<const int> of(int d, LO e) const {
    const auto& o = offsets[static_cast<std::size_t>(d)];
    const auto i = static_cast<std::size_t>(e);
    return {ranks[static_cast<std::size_t>(d)].data() + o[i], o[i + 1] - o[i]};
  }
};

[[nodiscard]] Placement place_closure(const Comm& comm, const HostMesh& mesh, std::span<const std::size_t> to_offsets,
                                      std::span<const int> to_ranks);

}  // namespace detail

}  // namespace soma

#endif  // SOMA_PARALLEL_MIGRATE_HPP_
