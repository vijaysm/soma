// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/merge.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "soma/parallel/transport/sort.hpp"

namespace soma {

namespace {

// Applies edit to a copy of the builder and keeps it only when the edit
// succeeds on every rank: the failing ranks rethrow theirs, the others
// throw std::runtime_error naming `who`.
template <class Edit>
void on_every_rank(const Comm& comm, MeshBuilder& builder, const char* who, Edit&& edit) {
  MeshBuilder trial = builder;
  std::exception_ptr failure;
  try {
    edit(trial);
  } catch (...) {
    failure = std::current_exception();
  }
  if (comm.allreduce(static_cast<std::int32_t>(failure ? 1 : 0), Op::max) != 0) {
    if (failure) std::rethrow_exception(failure);
    throw std::runtime_error(std::string(who) + ": failed on another rank");
  }
  builder = std::move(trial);
}

}  // namespace

void merge_vertices(const Comm& comm, MeshBuilder& builder, PointTolerance tol, GO base) {
  const PointIds m = match_points(comm, builder.vertex_coordinates(), tol);
  std::vector<GO> gids(m.ids.size());
  for (std::size_t i = 0; i < gids.size(); ++i) gids[i] = base + m.ids[i];
  on_every_rank(comm, builder, "merge_vertices", [&](MeshBuilder& b) { b.rename_vertices(gids, m.xyz); });
}

void number_elements(const Comm& comm, MeshBuilder& builder, GO base) {
  std::array<std::vector<GO>, 4> gids;
  for (int d = 1; d <= 3; ++d) {
    std::vector<Offset> offsets;
    std::vector<GO> corners;
    builder.element_corners(d, offsets, corners);
    Offset widest = 0;
    for (std::size_t e = 0; e + 1 < offsets.size(); ++e) widest = std::max(widest, offsets[e + 1] - offsets[e]);
    const auto words = static_cast<std::size_t>(comm.allreduce(static_cast<std::int64_t>(widest), Op::max));
    if (words == 0) continue;
    // Sorted corner gids, as words that sort alike, padded with the largest.
    std::vector<std::uint64_t> keys(words * (offsets.size() - 1), ~std::uint64_t{0});
    for (std::size_t e = 0; e + 1 < offsets.size(); ++e) {
      const auto key = keys.begin() + static_cast<std::ptrdiff_t>(e * words);
      for (Offset q = offsets[e]; q < offsets[e + 1]; ++q) {
        key[q - offsets[e]] = static_cast<std::uint64_t>(corners[static_cast<std::size_t>(q)]) ^ (std::uint64_t{1} << 63);
      }
      std::sort(key, key + static_cast<std::ptrdiff_t>(offsets[e + 1] - offsets[e]));
    }
    const transport::KeyRanks ranks = transport::rank_keys(comm, keys, words, true);
    auto& g = gids[static_cast<std::size_t>(d)];
    g.resize(ranks.index.size());
    for (std::size_t e = 0; e < g.size(); ++e) g[e] = base + static_cast<GO>(ranks.index[e]);
  }
  on_every_rank(comm, builder, "number_elements", [&](MeshBuilder& b) {
    for (int d = 1; d <= 3; ++d) {
      if (!gids[static_cast<std::size_t>(d)].empty()) b.rename_elements(d, gids[static_cast<std::size_t>(d)]);
    }
  });
}

}  // namespace soma
