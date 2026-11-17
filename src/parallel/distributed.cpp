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

#include "closure.hpp"
#include "side_keys.hpp"

namespace soma {

namespace {

using detail::add_sharing_fields;
void order_and_install(const Comm& comm, HostMesh& mesh, int k, const std::vector<GO>& gids, const Sharing& sharing);

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
  add_sharing_fields(mesh, dim, offsets, ranks, owners);
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

// A dimension's soma:sharing and soma:owner by lid, to install again when a
// derivation renews the epoch without changing the dimension.
struct Kept {
  std::vector<GO> gids;
  std::vector<Offset> offsets{0};
  std::vector<int> ranks;
  std::vector<int> owners;
};

Kept keep(const HostMesh& mesh, int dim) {
  Kept k;
  const SegmentTable& t = mesh.segment_table(dim);
  const auto g = mesh.gid(dim);
  k.gids.assign(g.data(), g.data() + mesh.count(dim));
  const auto s = mesh.fields().get_var<int>(sharing_field, dim);
  for (LO e = t.interior_end; e < t.shared_end; ++e) {
    const auto row = s.value_or_default(dim, e);
    k.ranks.insert(k.ranks.end(), row.begin(), row.end());
    k.offsets.push_back(static_cast<Offset>(k.ranks.size()));
  }
  const auto o = mesh.fields().get<int>(owner_field, dim);
  for (LO e = t.owned_end; e < mesh.count(dim); ++e) k.owners.push_back(o.value_or_default(dim, e));
  return k;
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
// An explicit entity is a ghost if it was one before (`explicit_ghosts`,
// sorted), a derived one if it is a side of no element held other than as a
// ghost.
void distribute_dimension(const Comm& comm, HostMesh& mesh, int k, const std::vector<GO>& explicit_gids,
                          const std::vector<GO>& explicit_ghosts) {
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

  // Ghosts: only with ghost elements on some rank.
  const int top = mesh.dim();
  const LO elements = mesh.count(top), held_elements = mesh.segment_table(top).shared_end;
  std::vector<std::uint8_t> ghost;
  if (comm.allreduce(static_cast<std::int32_t>(held_elements < elements || !explicit_ghosts.empty()), Op::max)) {
    std::vector<detail::SideKey> sides;
    for (LO e = 0; e < held_elements; ++e) {
      detail::for_each_side(mesh, top, e, k, [&](const detail::SideKey& key) { sides.push_back(key); });
    }
    std::sort(sides.begin(), sides.end());
    ghost.resize(static_cast<std::size_t>(n));
    std::size_t f = 0;
    for (LO e = 0; e < n; ++e) {
      const bool derived = f < fresh.size() && fresh[f] == e;
      if (derived) ++f;
      ghost[static_cast<std::size_t>(e)] =
          derived ? !std::binary_search(sides.begin(), sides.end(), detail::key_of(mesh, k, e))
                  : std::binary_search(explicit_ghosts.begin(), explicit_ghosts.end(), gid(e));
    }
  }
  order_and_install(comm, mesh, k, gids, resolve_sharing(comm, gids, ghost));
}

// What add_ghosts knows of a dimension it derives again (all of them on
// every rank): its gids and sharing taken from what the rank held and from
// what the owners of its ghosts sent, instead of a sort and a directory. An
// entity is a ghost by the same rule as in distribute_dimension.
void distribute_known(const Comm& comm, HostMesh& mesh, int k, const std::vector<GO>& explicit_gids,
                      const std::vector<GO>& explicit_ghosts, const detail::KnownEntities& known) {
  const LO n = mesh.count(k);
  const auto gid = mesh.gid(k);
  const int top = mesh.dim();
  std::vector<detail::SideKey> sides;
  for (LO e = 0; e < mesh.segment_table(top).shared_end; ++e) {
    detail::for_each_side(mesh, top, e, k, [&](const detail::SideKey& key) { sides.push_back(key); });
  }
  std::sort(sides.begin(), sides.end());
  std::vector<GO> gids(static_cast<std::size_t>(n));
  Sharing sharing;
  sharing.offsets.push_back(0);
  sharing.ghost_owner.assign(static_cast<std::size_t>(n), -1);
  std::string missing;
  for (LO e = 0; e < n; ++e) {
    const auto i = static_cast<std::size_t>(e);
    const bool given = std::binary_search(explicit_gids.begin(), explicit_gids.end(), gid(e));
    const detail::SideKey key = detail::key_of(mesh, k, e);
    if (given) {
      gids[i] = gid(e);
    } else if (const auto g = known.gid(key)) {
      gids[i] = *g;
    } else {
      missing = "no gid is known for an entity of dimension " + std::to_string(k);
      sharing.offsets.push_back(static_cast<std::uint32_t>(sharing.ranks.size()));
      continue;
    }
    const bool ghost = given ? std::binary_search(explicit_ghosts.begin(), explicit_ghosts.end(), gids[i])
                             : !std::binary_search(sides.begin(), sides.end(), key);
    if (ghost) {
      const auto owner = known.owner(gids[i]);
      if (!owner) {
        missing = "no owner is known for ghost " + std::to_string(gids[i]) + " of dimension " + std::to_string(k);
      }
      sharing.ghost_owner[i] = owner.value_or(comm.rank());
    } else if (const auto row = known.holders(gids[i])) {
      sharing.ranks.insert(sharing.ranks.end(), row->begin(), row->end());
    } else {
      missing = "no holders are known for " + std::to_string(gids[i]) + " of dimension " + std::to_string(k);
    }
    sharing.offsets.push_back(static_cast<std::uint32_t>(sharing.ranks.size()));
  }
  collectively(comm, "add_ghosts", [&] {
    if (!missing.empty()) throw std::logic_error("add_ghosts: " + missing);
  });
  order_and_install(comm, mesh, k, gids, sharing);
}

// Orders dimension k by (kind, topology, arity, group), keeping the order
// within, gives it `gids` and installs its sharing.
void order_and_install(const Comm& comm, HostMesh& mesh, int k, const std::vector<GO>& gids, const Sharing& sharing) {
  const LO n = mesh.count(k);
  const auto off = mesh.conn_offsets(k);
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

// build_distributed_adjacencies, each dimension distributed by
// distribute(k, explicit gids, explicit ghosts).
template <class Distribute>
void build_and_distribute(const Comm& comm, HostMesh& mesh, AdjacencyRequest request, Distribute&& distribute) {
  // Explicit entities of each dimension (in lid order, then sorted), and
  // dimensions already distributed.
  std::array<std::vector<GO>, 4> explicit_gids, before, ghosts;
  std::array<bool, 4> done{};
  std::array<std::optional<Kept>, 4> kept;
  const std::uint64_t epoch = mesh.epoch();
  for (int d = 0; d <= mesh.dim(); ++d) {
    if (has_sharing(mesh, d)) kept[static_cast<std::size_t>(d)] = keep(mesh, d);
  }
  for (int k = 1; k < mesh.dim(); ++k) {
    const auto kk = static_cast<std::size_t>(k);
    done[kk] = has_sharing(mesh, k);
    const auto g = mesh.gid(k);
    for (LO e = 0; e < mesh.count(k); ++e) before[kk].push_back(g(e));
    explicit_gids[kk] = before[kk];
    std::sort(explicit_gids[kk].begin(), explicit_gids[kk].end());
    for (LO e = mesh.segment_table(k).shared_end; e < mesh.count(k); ++e) ghosts[kk].push_back(g(e));
    std::sort(ghosts[kk].begin(), ghosts[kk].end());
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
    distribute(k, explicit_gids[kk], ghosts[kk]);
    kept[kk].reset();
  }
  // Dimensions left alone keep their sharing at the new epoch.
  if (mesh.epoch() == epoch) return;
  for (int d = 0; d <= mesh.dim(); ++d) {
    const auto& k = kept[static_cast<std::size_t>(d)];
    if (!k) continue;
    const auto g = mesh.gid(d);
    if (static_cast<std::size_t>(mesh.count(d)) != k->gids.size() || !std::equal(k->gids.begin(), k->gids.end(), g.data())) {
      throw std::logic_error("build_distributed_adjacencies: a derivation changed a distributed dimension it did not extend");
    }
    uninstall(mesh, d);
    add_sharing_fields(mesh, d, k->offsets, k->ranks, k->owners);
  }
}

}  // namespace

void detail::add_sharing_fields(HostMesh& mesh, int dim, const std::vector<Offset>& offsets,
                                const std::vector<int>& ranks, const std::vector<int>& owners) {
  const SegmentTable& t = mesh.segment_table(dim);
  SegmentFilter shared;
  shared.ownership = SegmentFilter::Own::shared;
  std::vector<LO> others(static_cast<std::size_t>(mesh.count(dim) - t.owned_end));
  std::iota(others.begin(), others.end(), t.owned_end);
  auto& reg = mesh.fields();
  reg.add_var<int>(sharing_field,
                   make_space(mesh, Domain<Kokkos::HostSpace>::segments(mesh, dim, shared), Layout::per_entity()),
                   offsets, ranks);
  auto owner = reg.add<int>(
      owner_field, make_space(mesh, Domain<Kokkos::HostSpace>::from_list(mesh, dim, others), Layout::per_entity()), 1);
  for (std::size_t r = 0; r < owners.size(); ++r) owner.view()(r, 0) = owners[r];
}

void build_distributed_adjacencies(const Comm& comm, HostMesh& mesh, AdjacencyRequest request) {
  build_and_distribute(comm, mesh, request, [&](int k, const std::vector<GO>& given, const std::vector<GO>& ghosts) {
    distribute_dimension(comm, mesh, k, given, ghosts);
  });
}

void detail::build_known_adjacencies(const Comm& comm, HostMesh& mesh, AdjacencyRequest request,
                                     const std::array<KnownEntities, 4>& known) {
  build_and_distribute(comm, mesh, request, [&](int k, const std::vector<GO>& given, const std::vector<GO>& ghosts) {
    distribute_known(comm, mesh, k, given, ghosts, known[static_cast<std::size_t>(k)]);
  });
}

HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder, Ordering ordering) {
  return build_local_part(comm, std::move(builder), std::array<std::vector<GO>, 4>{}, ordering);
}

namespace {

// build_local_part with the sharing of some dimensions given: its checks and
// the box of every rank's vertices in one reduction, with `before`, this
// rank's failure so far (every rank throws if any rank failed). A failure of
// the build itself is reported in one more reduction, or left in `deferred`
// when given, for the caller to report with its next check.
HostMesh build_part(const Comm& comm, MeshBuilder&& builder, const std::array<std::vector<GO>, 4>& ghosts,
                    const std::array<std::optional<Sharing>, 4>& given, Ordering ordering,
                    std::exception_ptr before = nullptr, std::exception_ptr* deferred = nullptr,
                    const char* who = "build_local_part") {
  // Every dimension the builder holds (the vertices always), and the box.
  std::array<std::vector<GO>, 4> gids;
  gids[0].assign(builder.vertex_gids().begin(), builder.vertex_gids().end());
  // Maxima of: held per dimension, a stray ghost, a given sharing that does
  // not cover its gids, an earlier failure, then the negated lows and highs.
  std::array<Real, 13> f;
  f.fill(0);
  f[0] = 1;
  for (int d = 1; d <= 3; ++d) {
    gids[static_cast<std::size_t>(d)] = builder.element_gids(d);
    f[static_cast<std::size_t>(d)] = gids[static_cast<std::size_t>(d)].empty() ? 0 : 1;
  }
  // Ghost flags per gid given; a ghost the builder does not hold fails.
  std::array<std::vector<std::uint8_t>, 4> flags;
  for (std::size_t d = 0; d < 4; ++d) {
    if (ghosts[d].empty()) continue;
    std::vector<GO> g = ghosts[d];
    std::sort(g.begin(), g.end());
    std::vector<GO> h = gids[d];
    std::sort(h.begin(), h.end());
    if (!std::includes(h.begin(), h.end(), g.begin(), g.end())) f[4] = 1;
    flags[d].resize(gids[d].size());
    for (std::size_t i = 0; i < gids[d].size(); ++i) flags[d][i] = std::binary_search(g.begin(), g.end(), gids[d][i]) ? 1 : 0;
  }
  for (std::size_t d = 0; d < 4; ++d) {
    const auto& g = given[d];
    if (g && (g->offsets.size() != gids[d].size() + 1 || g->offsets.back() != g->ranks.size())) f[5] = 1;
  }
  f[6] = before ? 1 : 0;
  const auto xyz = builder.vertex_coordinates();
  for (std::size_t k = 7; k < 13; ++k) f[k] = -std::numeric_limits<Real>::infinity();
  for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) {
    for (std::size_t d = 0; d < 3; ++d) {
      f[7 + d] = std::max(f[7 + d], -xyz[i + d]);
      f[10 + d] = std::max(f[10 + d], xyz[i + d]);
    }
  }
  comm.allreduce(std::span<Real>(f), Op::max);
  if (f[6] != 0) {
    if (before) std::rethrow_exception(before);
    throw std::runtime_error(std::string(who) + ": failed on another rank");
  }
  if (f[4] != 0) throw std::invalid_argument("build_local_part: a rank names a ghost it does not hold");
  if (f[5] != 0) throw std::invalid_argument("build_local_part: a rank's given sharing does not cover its gids");
  std::array<std::optional<Sharing>, 4> sharing;
  for (std::size_t d = 0; d < 4; ++d) {
    if (f[d] != 0) sharing[d] = given[d] ? *given[d] : resolve_sharing(comm, gids[d], flags[d]);
  }

  HostMesh mesh;
  const auto build = [&] {
    for (std::size_t d = 0; d < 4; ++d) {
      if (!sharing[d]) continue;
      std::vector<Ownership> kinds(gids[d].size());
      for (std::size_t i = 0; i < kinds.size(); ++i) kinds[i] = sharing[d]->kind(i, comm.rank());
      builder.set_ownership(static_cast<int>(d), gids[d], kinds, sharing[d]->groups());
    }
    if (std::isfinite(f[10])) builder.set_key_box({-f[7], -f[8], -f[9]}, {f[10], f[11], f[12]});
    mesh = std::move(builder).build(ordering);
    for (std::size_t d = 0; d < 4; ++d) {
      if (sharing[d] && static_cast<int>(d) <= mesh.dim()) install(mesh, comm, static_cast<int>(d), gids[d], *sharing[d]);
    }
  };
  if (deferred == nullptr) {
    // A failure on one rank is reported on every rank before anything else
    // collective runs.
    collectively(comm, "build_local_part", build);
  } else {
    try {
      build();
    } catch (...) {
      *deferred = std::current_exception();
    }
  }
  return mesh;
}

}  // namespace

HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder, const std::array<std::vector<GO>, 4>& ghosts,
                          Ordering ordering) {
  return build_part(comm, std::move(builder), ghosts, {}, ordering);
}

HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder, const std::array<std::optional<Sharing>, 4>& sharing,
                          Ordering ordering) {
  return build_part(comm, std::move(builder), {}, sharing, ordering);
}

HostMesh detail::build_local_part_deferred(const Comm& comm, MeshBuilder&& builder,
                                           const std::array<std::optional<Sharing>, 4>& sharing, Ordering ordering,
                                           std::exception_ptr before, std::exception_ptr& failure, const char* who) {
  return build_part(comm, std::move(builder), {}, sharing, ordering, std::move(before), &failure, who);
}

std::vector<int> owner_ranks(const Comm& comm, const HostMesh& mesh, int d) {
  if (d < 0 || d > mesh.dim()) {
    throw std::invalid_argument("owner_ranks: dimension " + std::to_string(d) + " is outside [0, " +
                                std::to_string(mesh.dim()) + "]");
  }
  const LO count = mesh.count(d);
  const LO owned = mesh.segment_table(d).owned_end;
  std::vector<int> out(static_cast<std::size_t>(count), comm.rank());
  if (owned < count) {
    const auto o = mesh.fields().get<int>(owner_field, d);
    for (LO e = owned; e < count; ++e) out[static_cast<std::size_t>(e)] = o.value_or_default(d, e);
  }
  return out;
}

}  // namespace soma
