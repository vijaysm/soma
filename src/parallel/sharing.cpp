// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/sharing.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include "soma/parallel/transport/rendezvous.hpp"
#include "soma/parallel/transport/route.hpp"

namespace soma {

Ownership Sharing::kind(std::size_t i, int me) const {
  if (offsets[i] == offsets[i + 1]) return Ownership::interior;
  return owner(i, me) == me ? Ownership::shared : Ownership::shared_copy;
}

int Sharing::owner(std::size_t i, int me) const {
  return offsets[i] == offsets[i + 1] ? me : std::min(me, ranks[offsets[i]]);
}

std::vector<std::uint32_t> Sharing::groups() const {
  const std::size_t n = offsets.empty() ? 0 : offsets.size() - 1;
  const auto row = [&](std::size_t i) {
    return std::span<const int>(ranks.data() + offsets[i], offsets[i + 1] - offsets[i]);
  };
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  const auto less = [&](std::size_t a, std::size_t b) {
    const auto x = row(a), y = row(b);
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
  };
  std::stable_sort(order.begin(), order.end(), less);
  std::vector<std::uint32_t> out(n);
  std::uint32_t group = 0;
  for (std::size_t k = 0; k < n; ++k) {
    if (k > 0 && less(order[k - 1], order[k])) ++group;
    out[order[k]] = group;
  }
  return out;
}

Sharing resolve_sharing(const Comm& comm, std::span<const GO> gids) {
  using Host = Kokkos::DefaultHostExecutionSpace;
  using Batch = transport::RecordBatch<Kokkos::HostSpace>;
  std::vector<GO> sorted(gids.begin(), gids.end());
  std::sort(sorted.begin(), sorted.end());
  const bool repeated = std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end();
  if (comm.allreduce(static_cast<std::int32_t>(repeated), Op::max)) {
    throw std::invalid_argument("resolve_sharing: a rank names a gid twice");
  }
  // Every gid to its directory rank.
  Batch sent(gids.size(), sizeof(GO), "soma::resolve_sharing");
  for (std::size_t i = 0; i < gids.size(); ++i) {
    sent.dest(i) = transport::owner(static_cast<std::uint64_t>(gids[i]), comm.size());
    std::memcpy(&sent.payload(i, 0), &gids[i], sizeof(GO));
  }
  const Batch at_directory = transport::route(Host{}, comm, sent);
  // Holders of each gid (arrivals come by source, so ascending).
  std::vector<std::pair<GO, int>> held(at_directory.size());
  for (std::size_t i = 0; i < held.size(); ++i) {
    std::memcpy(&held[i].first, &at_directory.payload(i, 0), sizeof(GO));
    held[i].second = at_directory.source(i);
  }
  std::stable_sort(held.begin(), held.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  // To each holder of a shared gid, one record (gid, other holder) per other holder.
  std::vector<std::tuple<int, GO, int>> notes;
  for (std::size_t a = 0; a < held.size();) {
    std::size_t b = a;
    while (b < held.size() && held[b].first == held[a].first) ++b;
    for (std::size_t h = a; b - a > 1 && h < b; ++h) {
      for (std::size_t o = a; o < b; ++o) {
        if (o != h) notes.emplace_back(held[h].second, held[a].first, held[o].second);
      }
    }
    a = b;
  }
  Batch back(notes.size(), sizeof(GO) + sizeof(int), "soma::resolve_sharing.back");
  for (std::size_t i = 0; i < notes.size(); ++i) {
    const auto& [to, gid, other] = notes[i];
    back.dest(i) = to;
    std::memcpy(&back.payload(i, 0), &gid, sizeof(GO));
    std::memcpy(&back.payload(i, sizeof(GO)), &other, sizeof(int));
  }
  const Batch told = transport::route(Host{}, comm, back);
  std::vector<std::pair<GO, int>> others(told.size());
  for (std::size_t i = 0; i < others.size(); ++i) {
    std::memcpy(&others[i].first, &told.payload(i, 0), sizeof(GO));
    std::memcpy(&others[i].second, &told.payload(i, sizeof(GO)), sizeof(int));
  }
  std::sort(others.begin(), others.end());
  // Rows in the order of the gids given.
  Sharing out;
  out.offsets.assign(gids.size() + 1, 0);
  out.ranks.reserve(others.size());
  for (std::size_t i = 0; i < gids.size(); ++i) {
    auto it = std::lower_bound(others.begin(), others.end(), std::pair<GO, int>{gids[i], -1});
    for (; it != others.end() && it->first == gids[i]; ++it) out.ranks.push_back(it->second);
    if (out.ranks.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error("resolve_sharing: more than 2^32 - 1 sharing entries on one rank");
    }
    out.offsets[i + 1] = static_cast<std::uint32_t>(out.ranks.size());
  }
  return out;
}

}  // namespace soma
