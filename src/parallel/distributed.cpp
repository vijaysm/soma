// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/distributed.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "soma/fields/domain_mesh.hpp"
#include "soma/parallel/sharing.hpp"
#include "soma/parallel/transport/sort.hpp"
#include "soma/topology/build.hpp"

namespace soma {

namespace {

// Installs soma:sharing and soma:owner on dimension `dim` from the sharing of
// `gids` (the gids given to resolve_sharing, in its order).
void install(HostMesh& mesh, const Comm& comm, int dim, std::span<const GO> gids, const Sharing& sharing) {
  std::vector<std::pair<GO, std::size_t>> row_of(gids.size());
  for (std::size_t i = 0; i < gids.size(); ++i) row_of[i] = {gids[i], i};
  std::sort(row_of.begin(), row_of.end());
  const auto find = [&](GO g) {
    return std::lower_bound(row_of.begin(), row_of.end(), std::pair<GO, std::size_t>{g, 0})->second;
  };
  const SegmentTable& t = mesh.segment_table(dim);
  const auto gid = mesh.gid(dim);
  std::vector<Offset> offsets{0};
  std::vector<int> ranks;
  for (LO e = t.interior_end; e < t.shared_end; ++e) {
    const std::size_t i = find(gid(e));
    ranks.insert(ranks.end(), sharing.ranks.begin() + sharing.offsets[i], sharing.ranks.begin() + sharing.offsets[i + 1]);
    offsets.push_back(static_cast<Offset>(ranks.size()));
  }
  std::vector<int> owners;
  for (LO e = t.owned_end; e < t.shared_end; ++e) owners.push_back(sharing.owner(find(gid(e)), comm.rank()));

  for (LO e = t.shared_end; e < mesh.count(dim); ++e) owners.push_back(sharing.owner(find(gid(e)), comm.rank()));

  SegmentFilter shared;
  shared.ownership = SegmentFilter::Own::shared;
  std::vector<LO> others(static_cast<std::size_t>(mesh.count(dim) - t.owned_end));
  std::iota(others.begin(), others.end(), t.owned_end);
  auto& reg = mesh.fields();
  reg.add_var<int>(sharing_field, make_space(mesh, Domain<Kokkos::HostSpace>::segments(mesh, dim, shared), Layout::per_entity()),
                   offsets, ranks);
  auto owner = reg.add<int>(owner_field,
                            make_space(mesh, Domain<Kokkos::HostSpace>::from_list(mesh, dim, others), Layout::per_entity()), 1);
  for (std::size_t r = 0; r < owners.size(); ++r) owner.view()(r, 0) = owners[r];
}

bool has_sharing(const HostMesh& mesh, int dim) {
  try {
    (void)mesh.fields().get_var<int>(sharing_field, dim);
    return true;
  } catch (const std::out_of_range&) {
    return false;
  }
}

// Drops soma:sharing and soma:owner from dimension `dim`.
void uninstall(HostMesh& mesh, int dim) {
  auto& reg = mesh.fields();
  for (const FieldInfo& f : reg.list()) {  // stale after a derivation, so not by get
    const bool named = f.name == sharing_field || f.name == owner_field;
    if (named && std::find(f.dims.begin(), f.dims.end(), dim) != f.dims.end()) reg.remove(f.id);
  }
}

// Runs f, then reports a failure on any rank on every rank.
template <class F>
void collectively(const Comm& comm, const char* what, F&& f) {
  std::exception_ptr failure;
  try {
    f();
  } catch (...) {
    failure = std::current_exception();
  }
  if (comm.allreduce(static_cast<std::int32_t>(failure != nullptr), Op::max)) {
    if (failure) std::rethrow_exception(failure);
    throw std::runtime_error(std::string(what) + ": failed on another rank");
  }
}

constexpr std::size_t key_words = 4;  // corners of a derived entity at most

// Gives derived dimension k global gids, shares it, orders it by ownership.
void distribute_dimension(const Comm& comm, HostMesh& mesh, int k, const std::vector<GO>& explicit_gids) {
  const LO n = mesh.count(k);
  const auto gid = mesh.gid(k);
  const auto off = mesh.conn_offsets(k);
  const auto conn = mesh.conn(k);
  const auto vgid = mesh.gid(0);
  std::vector<std::uint64_t> keys;
  std::vector<LO> fresh;  // entities derived here, not explicit
  bool too_many_corners = false;
  for (LO e = 0; e < n; ++e) {
    if (std::binary_search(explicit_gids.begin(), explicit_gids.end(), gid(e))) continue;
    fresh.push_back(e);
    std::array<std::uint64_t, key_words> key;
    key.fill(~std::uint64_t{0});
    const Offset len = off(e + 1) - off(e);
    too_many_corners = too_many_corners || len > static_cast<Offset>(key_words);
    for (Offset q = 0; q < len && q < static_cast<Offset>(key_words); ++q) {
      // Signed gids as words that sort alike.
      key[static_cast<std::size_t>(q)] = static_cast<std::uint64_t>(vgid(conn(off(e) + q))) ^ (std::uint64_t{1} << 63);
    }
    std::sort(key.begin(), key.end());
    keys.insert(keys.end(), key.begin(), key.end());
  }
  if (comm.allreduce(static_cast<std::int32_t>(too_many_corners), Op::max)) {
    throw std::invalid_argument("build_distributed_adjacencies: a derived entity has more than four corners");
  }
  const GO largest = explicit_gids.empty() ? 0 : explicit_gids.back();
  const GO base = std::max<GO>(comm.allreduce(largest, Op::max), 0);
  const transport::KeyRanks ranks = transport::rank_keys(comm, keys, key_words, true);
  std::vector<GO> gids(static_cast<std::size_t>(n));
  for (LO e = 0; e < n; ++e) gids[static_cast<std::size_t>(e)] = gid(e);
  for (std::size_t i = 0; i < fresh.size(); ++i) {
    gids[static_cast<std::size_t>(fresh[i])] = base + 1 + static_cast<GO>(ranks.index[i]);
  }

  const Sharing sharing = resolve_sharing(comm, gids);
  const std::vector<std::uint32_t> groups = sharing.groups();
  collectively(comm, "build_distributed_adjacencies", [&] {
    // By (kind, topology, arity, group), the derived order within.
    const auto topo = mesh.topology(k);
    std::vector<LO> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), LO{0});
    const auto key_of = [&](LO e) {
      const auto i = static_cast<std::size_t>(e);
      return std::tuple(sharing.kind(i, comm.rank()), static_cast<Topology>(topo(e)),
                        static_cast<std::uint16_t>(off(e + 1) - off(e)), groups[i]);
    };
    std::stable_sort(order.begin(), order.end(), [&](LO a, LO b) { return key_of(a) < key_of(b); });
    std::vector<LO> new_of_old(order.size());
    std::vector<Ownership> kinds(order.size());
    std::vector<Topology> topology(order.size());
    std::vector<std::uint16_t> arity(order.size());
    for (std::size_t r = 0; r < order.size(); ++r) {
      new_of_old[static_cast<std::size_t>(order[r])] = static_cast<LO>(r);
      std::tie(kinds[r], topology[r], arity[r], std::ignore) = key_of(order[r]);
    }
    permute_entities(mesh, k, new_of_old, make_segments(kinds, topology, arity), gids);
    install(mesh, comm, k, gids, sharing);
  });
}

}  // namespace

void build_distributed_adjacencies(const Comm& comm, HostMesh& mesh, AdjacencyRequest request) {
  // Explicit entities of each dimension (in lid order, then sorted), and
  // dimensions already distributed.
  std::array<std::vector<GO>, 4> explicit_gids, before;
  std::array<bool, 4> done{};
  for (int k = 1; k < mesh.dim(); ++k) {
    const auto kk = static_cast<std::size_t>(k);
    done[kk] = has_sharing(mesh, k);
    const auto g = mesh.gid(k);
    for (LO e = 0; e < mesh.count(k); ++e) before[kk].push_back(g(e));
    explicit_gids[kk] = before[kk];
    std::sort(explicit_gids[kk].begin(), explicit_gids[kk].end());
  }
  collectively(comm, "build_distributed_adjacencies", [&] { build_adjacencies(mesh, request); });
  for (int k = 1; k < mesh.dim(); ++k) {
    // A distributed dimension the derivation left alone stays; one it added
    // to or reordered is distributed again. Every rank takes the same branch.
    const auto kk = static_cast<std::size_t>(k);
    const auto g = mesh.gid(k);
    const bool same = static_cast<std::size_t>(mesh.count(k)) == before[kk].size() &&
                      std::equal(before[kk].begin(), before[kk].end(), g.data());
    const std::int32_t todo = (!done[kk] && mesh.count(k) > 0) || (done[kk] && !same) ? 1 : 0;
    if (comm.allreduce(todo, Op::max) == 0) continue;
    if (done[kk]) uninstall(mesh, k);
    distribute_dimension(comm, mesh, k, explicit_gids[kk]);
  }
}

HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder, Ordering ordering) {
  return build_local_part(comm, std::move(builder), std::array<std::vector<GO>, 4>{}, ordering);
}

HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder, const std::array<std::vector<GO>, 4>& ghosts,
                          Ordering ordering) {
  // Every dimension the builder holds on some rank, the vertices always.
  std::array<std::vector<GO>, 4> gids;
  gids[0].assign(builder.vertex_gids().begin(), builder.vertex_gids().end());
  std::array<std::int32_t, 4> held{1, 0, 0, 0};
  for (int d = 1; d <= 3; ++d) {
    gids[static_cast<std::size_t>(d)] = builder.element_gids(d);
    held[static_cast<std::size_t>(d)] = gids[static_cast<std::size_t>(d)].empty() ? 0 : 1;
  }
  comm.allreduce(std::span<std::int32_t>(held), Op::max);
  // Ghost flags per gid given; a ghost the builder does not hold fails.
  std::array<std::vector<std::uint8_t>, 4> flags;
  bool stray = false;
  for (std::size_t d = 0; d < 4; ++d) {
    if (ghosts[d].empty()) continue;
    std::vector<GO> g = ghosts[d];
    std::sort(g.begin(), g.end());
    std::vector<GO> h = gids[d];
    std::sort(h.begin(), h.end());
    stray = stray || !std::includes(h.begin(), h.end(), g.begin(), g.end());
    flags[d].resize(gids[d].size());
    for (std::size_t i = 0; i < gids[d].size(); ++i) flags[d][i] = std::binary_search(g.begin(), g.end(), gids[d][i]) ? 1 : 0;
  }
  if (comm.allreduce(static_cast<std::int32_t>(stray), Op::max)) {
    throw std::invalid_argument("build_local_part: a rank names a ghost it does not hold");
  }
  std::array<std::optional<Sharing>, 4> sharing;
  for (std::size_t d = 0; d < 4; ++d) {
    if (held[d]) sharing[d] = resolve_sharing(comm, gids[d], flags[d]);
  }

  // The box of every rank's vertices: maxima of the highs and of the negated lows.
  const auto xyz = builder.vertex_coordinates();
  std::array<Real, 6> box;
  box.fill(-std::numeric_limits<Real>::infinity());
  for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) {
    for (std::size_t d = 0; d < 3; ++d) {
      box[d] = std::max(box[d], -xyz[i + d]);
      box[3 + d] = std::max(box[3 + d], xyz[i + d]);
    }
  }
  comm.allreduce(std::span<Real>(box), Op::max);

  // A failure on one rank is reported on every rank before anything else
  // collective runs.
  HostMesh mesh;
  collectively(comm, "build_local_part", [&] {
    for (std::size_t d = 0; d < 4; ++d) {
      if (!sharing[d]) continue;
      std::vector<Ownership> kinds(gids[d].size());
      for (std::size_t i = 0; i < kinds.size(); ++i) kinds[i] = sharing[d]->kind(i, comm.rank());
      builder.set_ownership(static_cast<int>(d), gids[d], kinds, sharing[d]->groups());
    }
    if (std::isfinite(box[3])) builder.set_key_box({-box[0], -box[1], -box[2]}, {box[3], box[4], box[5]});
    mesh = std::move(builder).build(ordering);
    for (std::size_t d = 0; d < 4; ++d) {
      if (sharing[d] && static_cast<int>(d) <= mesh.dim()) install(mesh, comm, static_cast<int>(d), gids[d], *sharing[d]);
    }
  });
  return mesh;
}

}  // namespace soma
