// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_TRANSPORT_SORT_HPP_
#define SOMA_PARALLEL_TRANSPORT_SORT_HPP_

// Global ranks of keys held across ranks, by a sample sort: each rank sends
// regular samples of its sorted keys to rank 0, which picks P - 1 splitters
// and broadcasts them; every key is routed to the rank of its key range,
// ranked there, offset by an exclusive scan of the counts, and the index goes
// back to its place. A key is `words` std::uint64_t compared
// lexicographically.
//
//   distinct = true:  equal keys, on any ranks, share one index; indices
//                     number the distinct keys in ascending order, so they
//                     do not depend on P (derived-entity numbering).
//   distinct = false: every key has its own index; equal keys are ordered by
//                     sender rank, then the sender's order.
//
// Memory per rank is the keys it holds and receives plus P - 1 splitters.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "soma/parallel/comm.hpp"
#include "soma/parallel/transport/route.hpp"

namespace soma::transport {

struct KeyRanks {
  std::vector<std::uint64_t> index;  // per key, in the order given
  std::uint64_t total = 0;           // the number of indices on all ranks
};

// Collective. Throws std::invalid_argument on every rank if any rank passes
// words == 0, keys that are not whole rows, or a different words or
// `distinct` than another rank.
[[nodiscard]] KeyRanks rank_keys(const Comm& comm, std::span<const std::uint64_t> keys, std::size_t words,
                                 bool distinct, Strategy strategy = Strategy::automatic);

// Distinct ranks, and per key given, in order, the other ranks that gave an
// equal key, ascending: ranks[offsets[i]] .. ranks[offsets[i + 1] - 1]. The
// rank of a key's range sees every copy, so it tells their holders; this
// costs one more exchange, of the keys held on several ranks only.
struct HeldKeyRanks : KeyRanks {
  std::vector<std::size_t> offsets;
  std::vector<int> ranks;
};

// Collective. Throws as rank_keys.
[[nodiscard]] HeldKeyRanks rank_keys_held(const Comm& comm, std::span<const std::uint64_t> keys, std::size_t words,
                                          Strategy strategy = Strategy::automatic);

}  // namespace soma::transport

#endif  // SOMA_PARALLEL_TRANSPORT_SORT_HPP_
