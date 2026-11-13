// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/merge.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "closure.hpp"
#include "soma/core/mesh_view.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/parallel/transport/sort.hpp"
#include "soma/topology/build.hpp"

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

// The rows of `s` for the entities a rename keeps: the first given each gid.
Sharing kept_rows(const std::vector<GO>& gids, const Sharing& s) {
  std::vector<std::size_t> order(gids.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return gids[a] < gids[b]; });
  std::vector<std::uint8_t> keep(gids.size(), 0);
  for (std::size_t k = 0; k < order.size(); ++k) {
    if (k == 0 || gids[order[k - 1]] != gids[order[k]]) keep[order[k]] = 1;
  }
  Sharing out;
  out.offsets.assign(1, 0);
  for (std::size_t i = 0; i < gids.size(); ++i) {
    if (keep[i] == 0) continue;
    out.ranks.insert(out.ranks.end(), s.ranks.begin() + s.offsets[i], s.ranks.begin() + s.offsets[i + 1]);
    out.offsets.push_back(static_cast<std::uint32_t>(out.ranks.size()));
  }
  return out;
}

// merge_vertices, with each vertex's new gid, in the order added, in `gids`.
Sharing merge_into(const Comm& comm, MeshBuilder& builder, PointTolerance tol, GO base, std::vector<GO>& gids) {
  const PointIds m = match_points(comm, builder.vertex_coordinates(), tol);
  gids.resize(m.ids.size());
  for (std::size_t i = 0; i < gids.size(); ++i) gids[i] = base + m.ids[i];
  on_every_rank(comm, builder, "merge_vertices", [&](MeshBuilder& b) { b.rename_vertices(gids, m.xyz); });
  return kept_rows(gids, m.sharing);
}

// The sorted corner gids of entity e of dimension d.
std::vector<GO> corner_key(const MeshView<Kokkos::HostSpace>& v, int d, LO e, const std::vector<GO>& gid_of_vertex) {
  std::vector<GO> key(static_cast<std::size_t>(v.num_vertices(d, e)));
  for (LO k = 0; k < v.num_vertices(d, e); ++k) key[static_cast<std::size_t>(k)] = gid_of_vertex[static_cast<std::size_t>(v.vertex(d, e, k))];
  std::sort(key.begin(), key.end());
  return key;
}

// soma:sharing and soma:owner of dimension d, as add_sharing_fields takes them.
struct SharingFields {
  std::vector<Offset> offsets;
  std::vector<int> ranks, owners;
};

std::optional<SharingFields> sharing_fields(const HostMesh& mesh, int d) {
  if (!mesh.fields().contains(sharing_field) || mesh.count(d) == 0) return std::nullopt;
  try {
    const auto s = mesh.fields().get_var<int>(sharing_field, d);
    const auto o = mesh.fields().get<int>(owner_field, d);
    SharingFields out;
    const auto off = s.offsets();
    const auto val = s.values();
    for (std::size_t r = 0; r < off.extent(0); ++r) out.offsets.push_back(off(r));
    for (std::size_t r = 0; r < val.extent(0); ++r) out.ranks.push_back(val(r));
    for (std::size_t r = 0; r < o.view().extent(0); ++r) out.owners.push_back(o.view()(r, 0));
    return out;
  } catch (const std::out_of_range&) {
    return std::nullopt;
  }
}

}  // namespace

Sharing merge_vertices(const Comm& comm, MeshBuilder& builder, PointTolerance tol, GO base) {
  std::vector<GO> gids;
  return merge_into(comm, builder, tol, base, gids);
}

Renumbering merge_mesh(const Comm& comm, HostMesh& mesh, PointTolerance tol) {
  using Access = detail::TopologyAccess;
  using Host = Kokkos::HostSpace;
  bool ghosts = false;
  for (int d = 0; d <= mesh.dim(); ++d) ghosts = ghosts || mesh.segment_table(d).shared_end < mesh.count(d);
  if (comm.allreduce(static_cast<std::int32_t>(ghosts), Op::max) != 0) {
    throw std::invalid_argument("merge_mesh: a rank holds ghosts; merge before adding them");
  }
  const int dim = comm.allreduce(static_cast<std::int32_t>(mesh.dim()), Op::max);
  const bool parts = comm.allreduce(static_cast<std::int32_t>(comm.size() > 1 || mesh.fields().contains(sharing_field)),
                                    Op::max) != 0;
  const auto v = mesh.view();

  // The live vertices, and the live elements and explicit sides, as a builder.
  MeshBuilder b(std::max(dim, 1));
  std::vector<LO> vlid;
  {
    std::vector<GO> gids;
    std::vector<Real> xyz;
    for (LO i = 0; i < mesh.count(0); ++i) {
      if (!v.alive(0, i)) continue;
      vlid.push_back(i);
      gids.push_back(v.gid(0, i));
      for (int q = 0; q < 3; ++q) xyz.push_back(v.x(i, q));
    }
    b.add_vertices(gids, xyz);
  }
  for (int d = 1; d <= mesh.dim(); ++d) {
    if (d < mesh.dim() && !mesh.input(d)) continue;  // derived again below
    std::map<std::pair<std::uint8_t, LO>, std::vector<LO>> groups;  // by (topology, nodes)
    for (LO e = 0; e < mesh.count(d); ++e) {
      if (!v.alive(d, e)) continue;
      const Topology t = v.topology(d, e);
      groups[{static_cast<std::uint8_t>(t), t == Topology::polygon ? 0 : v.num_nodes(d, e)}].push_back(e);
    }
    for (const auto& [key, ents] : groups) {
      const auto t = static_cast<Topology>(key.first);
      std::vector<GO> gids, nodes;
      std::vector<Offset> offsets{0};
      for (const LO e : ents) {
        gids.push_back(v.gid(d, e));
        const LO n = t == Topology::polygon ? v.num_vertices(d, e) : v.num_nodes(d, e);
        for (LO k = 0; k < n; ++k) nodes.push_back(v.gid(0, v.node(d, e, k)));
        offsets.push_back(static_cast<Offset>(nodes.size()));
      }
      if (t == Topology::polygon) {
        b.add_polygons(gids, offsets, nodes);
      } else {
        b.add_elements(t, gids, nodes, static_cast<int>(key.second));
      }
    }
  }

  // Merged and numbered as merge_vertices and number_elements, then built.
  std::vector<GO> merged;
  Sharing vertices = merge_into(comm, b, tol, 1, merged);
  std::array<std::optional<Sharing>, 4> sharing = number_elements(comm, b);
  sharing[0] = std::move(vertices);
  const AdjacencyRequest graphs = detail::graphs_to_rebuild(comm, mesh);
  HostMesh next;
  if (parts) {
    next = build_local_part(comm, std::move(b), sharing, mesh.ordering());
    if (!graphs.empty()) build_distributed_adjacencies(comm, next, graphs);
  } else {
    next = std::move(b).build(mesh.ordering());
    if (!graphs.empty()) build_adjacencies(next, graphs);
  }

  // Old lids to new: a vertex by its merged gid, an entity by its merged
  // corners; where several old entities become one, the least lid keeps its
  // values and memberships.
  Renumbering out;
  std::vector<GO> gid_of_vertex(static_cast<std::size_t>(mesh.count(0)), 0);
  for (std::size_t i = 0; i < vlid.size(); ++i) gid_of_vertex[static_cast<std::size_t>(vlid[i])] = merged[i];
  for (int d = 0; d <= 3; ++d) out.new_lid[static_cast<std::size_t>(d)].assign(static_cast<std::size_t>(mesh.count(d)), invalid_lo);
  {
    std::vector<std::pair<GO, LO>> at;
    for (LO i = 0; i < next.count(0); ++i) at.emplace_back(next.gid(0)(i), i);
    std::sort(at.begin(), at.end());
    std::vector<std::uint8_t> taken(at.size(), 0);
    for (const LO i : vlid) {
      const auto it = std::lower_bound(at.begin(), at.end(), std::pair<GO, LO>{gid_of_vertex[static_cast<std::size_t>(i)], 0});
      const auto k = static_cast<std::size_t>(it - at.begin());
      if (it == at.end() || it->first != gid_of_vertex[static_cast<std::size_t>(i)] || taken[k] != 0) continue;
      taken[k] = 1;
      out.new_lid[0][static_cast<std::size_t>(i)] = it->second;
    }
  }
  const auto nv = next.view();
  std::vector<GO> next_gid_of_vertex(static_cast<std::size_t>(next.count(0)));
  for (LO i = 0; i < next.count(0); ++i) next_gid_of_vertex[static_cast<std::size_t>(i)] = next.gid(0)(i);
  for (int d = 1; d <= std::min(mesh.dim(), next.dim()); ++d) {
    std::vector<std::pair<std::vector<GO>, LO>> at;
    for (LO e = 0; e < next.count(d); ++e) at.emplace_back(corner_key(nv, d, e, next_gid_of_vertex), e);
    std::sort(at.begin(), at.end());
    std::vector<std::uint8_t> taken(at.size(), 0);
    for (LO e = 0; e < mesh.count(d); ++e) {
      if (!v.alive(d, e)) continue;
      const std::vector<GO> key = corner_key(v, d, e, gid_of_vertex);
      const auto it = std::lower_bound(at.begin(), at.end(), std::pair<std::vector<GO>, LO>{key, 0});
      const auto k = static_cast<std::size_t>(it - at.begin());
      if (it == at.end() || it->first != key || taken[k] != 0) continue;
      taken[k] = 1;
      out.new_lid[static_cast<std::size_t>(d)][static_cast<std::size_t>(e)] = it->second;
    }
  }

  // Sets, fields and mesh values through the lids. The part's own sharing
  // fields replace those carried over.
  std::array<std::optional<SharingFields>, 4> own;
  for (int d = 0; d <= 3; ++d) own[static_cast<std::size_t>(d)] = sharing_fields(next, d);
  std::array<std::optional<FieldRegistry<Host>::SlotRemap>, num_field_slots> maps;
  for (int d = 0; d <= 3; ++d) {
    maps[static_cast<std::size_t>(d)] = FieldRegistry<Host>::SlotRemap{out.new_lid[static_cast<std::size_t>(d)], next.count(d)};
  }
  std::optional<EntitySets<Host>> sets;
  for (int d = 0; d <= 3; ++d) {
    if (mesh.sets().count() == 0 || !mesh.sets().has_members(d)) continue;
    sets = (sets ? *sets : mesh.sets()).renumbered(d, out.new_lid[static_cast<std::size_t>(d)], next.count(d), true);
  }
  Access::sets(next) = sets ? std::move(*sets) : mesh.sets().template mirror<Host>({true, true, true, true}, true);
  next.values() = mesh.values();
  Access::epoch(next) = mesh.epoch() + 1;
  next.fields() = FieldRegistry<Host>{};
  Access::attach_fields(next);
  mesh.fields().remap_into(next.fields(), maps, next.epoch(), element_info(next));
  for (int d = 0; d <= 3; ++d) {
    if (next.fields().contains(sharing_field)) {
      try {
        next.fields().remove(next.fields().get_var<int>(sharing_field, d).id());
        next.fields().remove(next.fields().get<int>(owner_field, d).id());
      } catch (const std::out_of_range&) {
        // not on this dimension
      }
    }
    if (const auto& s = own[static_cast<std::size_t>(d)]) detail::add_sharing_fields(next, d, s->offsets, s->ranks, s->owners);
  }
  mesh.swap(next);
  return out;
}

HostMesh build_merged_part(const Comm& comm, MeshBuilder&& builder, PointTolerance tol, Ordering ordering) {
  // Vertices first: elements are named by their merged corners.
  Sharing vertices = merge_vertices(comm, builder, tol);
  std::array<std::optional<Sharing>, 4> sharing = number_elements(comm, builder);
  sharing[0] = std::move(vertices);
  return build_local_part(comm, std::move(builder), sharing, ordering);
}

std::array<std::optional<Sharing>, 4> number_elements(const Comm& comm, MeshBuilder& builder, GO base) {
  std::array<std::optional<Sharing>, 4> sharing;
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
    const transport::HeldKeyRanks ranks = transport::rank_keys_held(comm, keys, words);
    auto& g = gids[static_cast<std::size_t>(d)];
    g.resize(ranks.index.size());
    for (std::size_t e = 0; e < g.size(); ++e) g[e] = base + static_cast<GO>(ranks.index[e]);
    Sharing s;
    s.offsets.assign(ranks.offsets.begin(), ranks.offsets.end());
    s.ranks = ranks.ranks;
    sharing[static_cast<std::size_t>(d)] = kept_rows(g, s);
  }
  on_every_rank(comm, builder, "number_elements", [&](MeshBuilder& b) {
    for (int d = 1; d <= 3; ++d) {
      if (!gids[static_cast<std::size_t>(d)].empty()) b.rename_elements(d, gids[static_cast<std::size_t>(d)]);
    }
  });
  return sharing;
}

}  // namespace soma
