// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/partition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "soma/core/sfc.hpp"
#include "soma/parallel/transport/sort.hpp"

namespace soma {

std::vector<int> partition_by_curve(const Comm& comm, std::span<const Real> centroids, std::span<const GO> gids) {
  const std::size_t n = gids.size();
  const bool bad = centroids.size() != 3 * n ||
                   !std::all_of(centroids.begin(), centroids.end(), [](Real v) { return std::isfinite(v); });
  if (comm.allreduce(static_cast<std::int32_t>(bad), Op::max)) {
    throw std::invalid_argument(
        "partition_by_curve: a rank passes a centroid count other than its gid count, or a centroid that is not "
        "finite");
  }
  // The box of every rank's centroids: maxima of the highs and of the negated lows.
  std::array<Real, 6> box;
  box.fill(-std::numeric_limits<Real>::infinity());
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t d = 0; d < 3; ++d) {
      box[d] = std::max(box[d], -centroids[3 * i + d]);
      box[3 + d] = std::max(box[3 + d], centroids[3 * i + d]);
    }
  }
  comm.allreduce(std::span<Real>(box), Op::max);
  const Real lo[3] = {-box[0], -box[1], -box[2]}, hi[3] = {box[3], box[4], box[5]};
  // With no element on any rank the box is empty (its high corner -inf).
  const sfc::KeyBox key_box = std::isfinite(hi[0]) ? sfc::KeyBox::make(lo, hi) : sfc::KeyBox{};

  // Keys (Morton key, gid) are unique, so their global ranks are the curve order.
  std::vector<std::uint64_t> keys(2 * n);
  for (std::size_t i = 0; i < n; ++i) {
    keys[2 * i] = sfc::morton_key(key_box, centroids[3 * i], centroids[3 * i + 1], centroids[3 * i + 2]);
    // Offset so negative gids keep their order as unsigned words.
    keys[2 * i + 1] = static_cast<std::uint64_t>(gids[i]) ^ (std::uint64_t{1} << 63);
  }
  const transport::KeyRanks ranks = transport::rank_keys(comm, keys, 2, false);
  // The first total % P runs take one element more; no product can overflow.
  const auto p = static_cast<std::uint64_t>(comm.size());
  const std::uint64_t q = ranks.total / p, longer = ranks.total % p, split = longer * (q + 1);
  std::vector<int> dest(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t at = ranks.index[i];
    dest[i] = static_cast<int>(at < split ? at / (q + 1) : longer + (at - split) / q);
  }
  return dest;
}

}  // namespace soma
