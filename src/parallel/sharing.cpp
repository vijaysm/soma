// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/sharing.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include "../core/dense_ids.hpp"
#include "soma/parallel/transport/rendezvous.hpp"
#include "soma/parallel/transport/route.hpp"

namespace soma {

namespace {

bool is_ghost(const Sharing& s, std::size_t i) { return !s.ghost_owner.empty() && s.ghost_owner[i] >= 0; }

}  // namespace

Ownership Sharing::kind(std::size_t i, int me) const {
  if (is_ghost(*this, i)) return Ownership::ghost;
  if (offsets[i] == offsets[i + 1]) return Ownership::interior;
  return owner(i, me) == me ? Ownership::shared : Ownership::shared_copy;
}

int Sharing::owner(std::size_t i, int me) const {
  if (is_ghost(*this, i)) return ghost_owner[i];
  return offsets[i] == offsets[i + 1] ? me : std::min(me, ranks[offsets[i]]);
}

std::vector<std::uint32_t> Sharing::groups() const {
  const std::size_t n = offsets.empty() ? 0 : offsets.size() - 1;
  const auto row = [&](std::size_t i) {
    if (is_ghost(*this, i)) return std::span<const int>(&ghost_owner[i], 1);
    return std::span<const int>(ranks.data() + offsets[i], offsets[i + 1] - offsets[i]);
  };
  // Empty rows (interior entities, most of them) come first, as group 0;
  // only the others are sorted.
  std::vector<std::size_t> order;
  bool any_empty = false;
  for (std::size_t i = 0; i < n; ++i) {
    if (row(i).empty()) {
      any_empty = true;
    } else {
      order.push_back(i);
    }
  }
  const auto less = [&](std::size_t a, std::size_t b) {
    const auto x = row(a), y = row(b);
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
  };
  std::sort(order.begin(), order.end(), less);
  std::vector<std::uint32_t> out(n, 0);
  std::uint32_t group = any_empty ? 1 : 0;
  for (std::size_t k = 0; k < order.size(); ++k) {
    if (k > 0 && less(order[k - 1], order[k])) ++group;
    out[order[k]] = group;
  }
  return out;
}

Sharing resolve_sharing(const Comm& comm, std::span<const GO> gids, std::span<const std::uint8_t> ghost) {
  using Host = Kokkos::DefaultHostExecutionSpace;
  using Batch = transport::RecordBatch<Kokkos::HostSpace>;
  const bool repeated = detail::GidIndex(gids.size(), [&](std::size_t i) { return gids[i]; }).repeated.has_value();
  if (comm.allreduce(static_cast<std::int32_t>(repeated), Op::max)) {
    throw std::invalid_argument("resolve_sharing: a rank names a gid twice");
  }
  const bool bad_flags = !ghost.empty() && ghost.size() != gids.size();
  const std::int32_t any_ghost = std::any_of(ghost.begin(), ghost.end(), [](std::uint8_t g) { return g != 0; }) ? 1 : 0;
  std::array<std::int32_t, 2> f{bad_flags ? 1 : 0, any_ghost};
  comm.allreduce(std::span<std::int32_t>(f), Op::max);
  if (f[0] != 0) throw std::invalid_argument("resolve_sharing: a rank gives ghost flags but not one per gid");
  const auto ghost_at = [&](std::size_t i) { return !ghost.empty() && ghost[i] != 0; };
  if (comm.size() == 1) {
    // A lone rank shares nothing, and no rank owns what it ghosts.
    if (f[1] != 0) {
      throw std::invalid_argument("resolve_sharing: a rank holds as a ghost a gid no rank holds otherwise");
    }
    Sharing out;
    out.offsets.assign(gids.size() + 1, 0);
    return out;
  }
  // Every gid to its directory rank, with whether it is a ghost here.
  constexpr std::size_t row = sizeof(GO) + sizeof(std::int32_t);
  Batch sent(gids.size(), row, "soma::resolve_sharing");
  for (std::size_t i = 0; i < gids.size(); ++i) {
    sent.dest(i) = transport::owner(static_cast<std::uint64_t>(gids[i]), comm.size());
    const std::int32_t g = ghost_at(i) ? 1 : 0;
    std::memcpy(&sent.payload(i, 0), &gids[i], sizeof(GO));
    std::memcpy(&sent.payload(i, sizeof(GO)), &g, sizeof(g));
  }
  const Batch at_directory = transport::route(Host{}, comm, sent);
  // Holders of each gid, other holders before ghost holders, each ascending
  // (arrivals come by source).
  std::vector<std::tuple<GO, std::int32_t, int>> held(at_directory.size());
  for (std::size_t i = 0; i < held.size(); ++i) {
    GO g;
    std::int32_t k;
    std::memcpy(&g, &at_directory.payload(i, 0), sizeof(GO));
    std::memcpy(&k, &at_directory.payload(i, sizeof(GO)), sizeof(k));
    held[i] = {g, k, at_directory.source(i)};
  }
  if (!std::is_sorted(held.begin(), held.end())) {
    // By gid, then the few holders of each by (ghost, rank).
    detail::stable_sort_by(held, [](const auto& h) { return std::get<0>(h); });
    for (std::size_t a = 0, b = 0; a < held.size(); a = b) {
      while (b < held.size() && std::get<0>(held[b]) == std::get<0>(held[a])) ++b;
      if (b - a > 1) {
        std::sort(held.begin() + static_cast<std::ptrdiff_t>(a), held.begin() + static_cast<std::ptrdiff_t>(b));
      }
    }
  }
  // To each holder of a shared gid, one record (gid, other holder, 0) per
  // other holder; to each ghost holder one record (gid, owner, 1).
  std::vector<std::tuple<int, GO, int, std::int32_t>> notes;
  bool orphan = false;
  for (std::size_t a = 0; a < held.size();) {
    const GO g = std::get<0>(held[a]);
    std::size_t b = a, m = a;  // [a, m) hold it, [m, b) ghost it
    while (b < held.size() && std::get<0>(held[b]) == g) ++b;
    while (m < b && std::get<1>(held[m]) == 0) ++m;
    for (std::size_t h = a; m - a > 1 && h < m; ++h) {
      for (std::size_t o = a; o < m; ++o) {
        if (o != h) notes.emplace_back(std::get<2>(held[h]), g, std::get<2>(held[o]), 0);
      }
    }
    if (m == a && b > m) orphan = true;
    for (std::size_t h = m; m > a && h < b; ++h) notes.emplace_back(std::get<2>(held[h]), g, std::get<2>(held[a]), 1);
    a = b;
  }
  if (comm.allreduce(static_cast<std::int32_t>(orphan), Op::max)) {
    throw std::invalid_argument("resolve_sharing: a rank holds as a ghost a gid no rank holds otherwise");
  }
  Batch back(notes.size(), sizeof(GO) + sizeof(int) + sizeof(std::int32_t), "soma::resolve_sharing.back");
  for (std::size_t i = 0; i < notes.size(); ++i) {
    const auto& [to, gid, other, k] = notes[i];
    back.dest(i) = to;
    std::memcpy(&back.payload(i, 0), &gid, sizeof(GO));
    std::memcpy(&back.payload(i, sizeof(GO)), &other, sizeof(int));
    std::memcpy(&back.payload(i, sizeof(GO) + sizeof(int)), &k, sizeof(k));
  }
  const Batch told = transport::route(Host{}, comm, back);
  std::vector<std::pair<GO, int>> others, owners;
  for (std::size_t i = 0; i < told.size(); ++i) {
    GO g;
    int other;
    std::int32_t k;
    std::memcpy(&g, &told.payload(i, 0), sizeof(GO));
    std::memcpy(&other, &told.payload(i, sizeof(GO)), sizeof(int));
    std::memcpy(&k, &told.payload(i, sizeof(GO) + sizeof(int)), sizeof(k));
    (k != 0 ? owners : others).emplace_back(g, other);
  }
  std::sort(others.begin(), others.end());
  std::sort(owners.begin(), owners.end());
  // Rows in the order of the gids given.
  Sharing out;
  out.offsets.assign(gids.size() + 1, 0);
  out.ranks.reserve(others.size());
  if (f[1] != 0) out.ghost_owner.assign(gids.size(), -1);
  for (std::size_t i = 0; i < gids.size(); ++i) {
    if (ghost_at(i)) {
      out.ghost_owner[i] = std::lower_bound(owners.begin(), owners.end(), std::pair<GO, int>{gids[i], -1})->second;
    } else {
      auto it = std::lower_bound(others.begin(), others.end(), std::pair<GO, int>{gids[i], -1});
      for (; it != others.end() && it->first == gids[i]; ++it) out.ranks.push_back(it->second);
    }
    if (out.ranks.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error("resolve_sharing: more than 2^32 - 1 sharing entries on one rank");
    }
    out.offsets[i + 1] = static_cast<std::uint32_t>(out.ranks.size());
  }
  return out;
}

}  // namespace soma
