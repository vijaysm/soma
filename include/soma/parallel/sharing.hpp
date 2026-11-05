// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_SHARING_HPP_
#define SOMA_PARALLEL_SHARING_HPP_

// Which ranks hold an entity, from the gids each rank holds: every gid goes
// to a directory rank (transport::owner of the gid), which sends each holder
// of a gid held more than once the other holders. Traffic and memory scale
// with the shared gids; a gid held once costs one record. The owner of a
// shared gid is the lowest rank holding it. A gid may also be held as a
// ghost: ghost holders learn its owner, and no one else learns of them.

#include <cstdint>
#include <span>
#include <vector>

#include "soma/core/segments.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

struct Sharing {
  // Per gid given, in order: the other ranks holding it, ascending, rows
  // offsets[i] .. offsets[i + 1] of `ranks` (empty for an interior gid).
  std::vector<std::uint32_t> offsets;
  std::vector<int> ranks;
  // Per gid given, the owner of a gid held here as a ghost, else -1; empty
  // when no ghosts were given.
  std::vector<int> ghost_owner;

  // interior, shared (owned here), shared_copy (owned by ranks[offsets[i]])
  // or ghost.
  [[nodiscard]] Ownership kind(std::size_t i, int me) const;
  // The lowest rank holding gid i other than as a ghost.
  [[nodiscard]] int owner(std::size_t i, int me) const;
  // Per gid, a group equal for gids held by the same other ranks: groups
  // number the distinct rank sets in ascending lexicographic order, so they
  // agree with the order of the sets on every rank. A ghost's set is its
  // owner.
  [[nodiscard]] std::vector<std::uint32_t> groups() const;
};

// Collective: the sharing of the gids this rank holds; ghost[i] != 0 when
// gid i is held here only as a ghost (no flags: none is). Throws
// std::invalid_argument on every rank if any rank names a gid twice, gives
// flags but not one per gid, or holds as a ghost a gid no rank holds
// otherwise.
[[nodiscard]] Sharing resolve_sharing(const Comm& comm, std::span<const GO> gids,
                                      std::span<const std::uint8_t> ghost = {});

}  // namespace soma

#endif  // SOMA_PARALLEL_SHARING_HPP_
