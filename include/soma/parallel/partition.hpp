// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_PARTITION_HPP_
#define SOMA_PARALLEL_PARTITION_HPP_

/// \file
/// Partitions without a graph: elements ordered along a Morton curve of their
/// centroids in the box of every rank's centroids (ties by gid), then cut into
/// P runs whose sizes differ by at most one. The order is the same whatever
/// the ranks hold, so the result depends only on the elements and P.

#include <span>
#include <vector>

#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

/// Collective: the destination rank of each element given here, from its
/// centroid (x, y, z interleaved) and gid. Throws std::invalid_argument on
/// every rank if any rank passes a centroid count other than its gid count or
/// a centroid that is not finite.
[[nodiscard]] std::vector<int> partition_by_curve(const Comm& comm, std::span<const Real> centroids,
                                                  std::span<const GO> gids);

}  // namespace soma

#endif  // SOMA_PARALLEL_PARTITION_HPP_
