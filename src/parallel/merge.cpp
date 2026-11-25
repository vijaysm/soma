// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/merge.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "closure.hpp"
#include "soma/core/mesh_edit.hpp"
#include "soma/core/mesh_view.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "records.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/parallel/transport/rendezvous.hpp"
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

using Bytes = Kokkos::View<std::byte**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using Arrived = transport::RecordBatch<Kokkos::HostSpace>;

// transport::rendezvous_to over host vectors: row i of `width` bytes, filled
// by fill(i, row), goes with keys[i] to dest_of(keys[i]); answer(keys,
// arrived) answers each arrival in `answer_width` bytes.
template <class Fill, class Answer, class DestOf>
Bytes rendezvous_to(const Comm& comm, const std::vector<std::uint64_t>& keys, std::size_t width, Fill&& fill,
                    std::size_t answer_width, Answer&& answer, DestOf&& dest_of) {
  Kokkos::View<std::uint64_t*, Kokkos::HostSpace> k("soma::merge.keys", keys.size());
  Bytes p("soma::merge.payload", keys.size(), width);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    k(i) = keys[i];
    fill(i, &p(i, 0));
  }
  return transport::rendezvous_to(Kokkos::DefaultHostExecutionSpace{}, comm,
                                  Kokkos::View<const std::uint64_t*, Kokkos::HostSpace>(k),
                                  Kokkos::View<const std::byte**, Kokkos::LayoutRight, Kokkos::HostSpace>(p),
                                  answer_width, std::forward<Answer>(answer), std::forward<DestOf>(dest_of));
}

// rendezvous_to each key's owner, transport::owner.
template <class Fill, class Answer>
Bytes rendezvous(const Comm& comm, const std::vector<std::uint64_t>& keys, std::size_t width, Fill&& fill,
                 std::size_t answer_width, Answer&& answer) {
  const int size = comm.size();
  return rendezvous_to(comm, keys, width, std::forward<Fill>(fill), answer_width, std::forward<Answer>(answer),
                       [size](std::uint64_t key) { return transport::owner(key, size); });
}

// Arrivals by key: positions in key order.
template <class KeyView>
std::vector<std::size_t> by_key(const KeyView& keys) {
  std::vector<std::size_t> o(keys.extent(0));
  std::iota(o.begin(), o.end(), std::size_t{0});
  std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return keys(a) < keys(b); });
  return o;
}

// The gid of each entity, entities grouped by `group` (one value in [0,
// groups) for every copy, on any rank, of one merged entity), `old` their
// gids and trusted[i] != 0 where old[i] may stay. keep: a group takes the
// least trusted gid of its members that no other group's trusted member has;
// a group without one takes top + 1 + its rank among such groups, top the
// largest of `top_here` over the ranks. renumber: 1 + group. With `strict`, a
// trusted gid that several groups have throws std::invalid_argument on every
// rank, naming the least such gid, instead of being given up. Collective.
std::vector<GO> choose_gids(const Comm& comm, const std::vector<std::uint64_t>& group, std::uint64_t groups,
                            std::span<const GO> old, const std::vector<std::uint8_t>& trusted, GO top_here,
                            MergeGids how, bool strict = false) {
  const std::size_t n = group.size();
  std::vector<GO> out(n);
  if (how == MergeGids::renumber) {
    for (std::size_t i = 0; i < n; ++i) out[i] = 1 + static_cast<GO>(group[i]);
    return out;
  }
  // At each trusted gid's directory: whether one group alone has it.
  std::vector<std::size_t> held;
  std::vector<std::uint64_t> keys;
  for (std::size_t i = 0; i < n; ++i) {
    if (trusted[i] == 0) continue;
    held.push_back(i);
    keys.push_back(static_cast<std::uint64_t>(old[i]));
  }
  GO clash = std::numeric_limits<GO>::max();  // the least gid several groups have, seen here
  const Bytes alone = rendezvous(
      comm, keys, sizeof(std::uint64_t), [&](std::size_t i, std::byte* row) { detail::put(row, 0, group[held[i]]); },
      1, [&](const auto& arrived_keys, const Arrived& arrived) {
        Bytes rows("soma::merge.alone", arrived.size(), 1);
        const std::vector<std::size_t> o = by_key(arrived_keys);
        for (std::size_t a = 0, b = 0; a < o.size(); a = b) {
          bool one = true;
          const auto g0 = detail::take<std::uint64_t>(&arrived.payload(o[a], 0), 0);
          for (b = a; b < o.size() && arrived_keys(o[b]) == arrived_keys(o[a]); ++b) {
            one = one && detail::take<std::uint64_t>(&arrived.payload(o[b], 0), 0) == g0;
          }
          if (!one) clash = std::min(clash, static_cast<GO>(arrived_keys(o[a])));
          for (std::size_t k = a; k < b; ++k) rows(o[k], 0) = std::byte{one ? std::uint8_t{1} : std::uint8_t{0}};
        }
        return rows;
      });
  if (strict) {
    clash = comm.allreduce(clash, Op::min);
    if (clash != std::numeric_limits<GO>::max()) {
      throw std::invalid_argument("merge_mesh: gid " + std::to_string(clash) +
                                  " names different points on different ranks; pass the entities added since the "
                                  "mesh was built (MergeOptions::added) or renumber");
    }
  }
  std::vector<std::uint8_t> candidate(n, 0);
  for (std::size_t k = 0; k < held.size(); ++k) candidate[held[k]] = alone(k, 0) == std::byte{1} ? 1 : 0;
  // At each group's directory, the rank holding its block of group ids (in
  // order, so blocks of lower ranks hold lower ids): the least candidate gid
  // of its members or, for a group without one, its rank among all such
  // groups, those of lower blocks (a scan) and those before it in this.
  constexpr std::size_t flag_at = sizeof(std::uint64_t);
  const auto size = static_cast<std::uint64_t>(comm.size());
  const std::uint64_t block = std::max<std::uint64_t>(1, groups / size + (groups % size != 0 ? 1 : 0));
  std::uint64_t fresh_here = 0;
  const Bytes chosen = rendezvous_to(
      comm, group, sizeof(GO) + 1,
      [&](std::size_t i, std::byte* row) {
        detail::put(row, 0, old[i]);
        detail::put(row, sizeof(GO), candidate[i]);
      },
      flag_at + 1,
      [&](const auto& arrived_keys, const Arrived& arrived) {
        Bytes rows;
        std::vector<std::size_t> o;
        // Every rank scans, even one whose answer failed, then rethrows.
        std::exception_ptr failure;
        try {
          rows = Bytes("soma::merge.chosen", arrived.size(), flag_at + 1);
          o = by_key(arrived_keys);
          for (std::size_t a = 0, b = 0; a < o.size(); a = b) {
            bool any = false;
            GO least = 0;
            for (b = a; b < o.size() && arrived_keys(o[b]) == arrived_keys(o[a]); ++b) {
              if (detail::take<std::uint8_t>(&arrived.payload(o[b], 0), sizeof(GO)) == 0) continue;
              const auto g = detail::take<GO>(&arrived.payload(o[b], 0), 0);
              least = any ? std::min(least, g) : g;
              any = true;
            }
            if (!any) ++fresh_here;
            for (std::size_t k = a; k < b; ++k) {
              detail::put(&rows(o[k], 0), 0, static_cast<std::uint64_t>(least));
              detail::put(&rows(o[k], 0), flag_at, static_cast<std::uint8_t>(any ? 1 : 0));
            }
          }
        } catch (...) {
          failure = std::current_exception();
        }
        std::uint64_t next = comm.exscan(failure ? std::uint64_t{0} : fresh_here);
        if (failure) std::rethrow_exception(failure);
        for (std::size_t a = 0, b = 0; a < o.size(); a = b) {
          for (b = a; b < o.size() && arrived_keys(o[b]) == arrived_keys(o[a]); ++b) {}
          if (detail::take<std::uint8_t>(&rows(o[a], 0), flag_at) != 0) continue;
          for (std::size_t k = a; k < b; ++k) detail::put(&rows(o[k], 0), 0, next);
          ++next;
        }
        return rows;
      },
      [block](std::uint64_t g) { return static_cast<int>(g / block); });
  // New gids for the groups without one, above every gid given.
  const GO top = std::max<GO>(comm.allreduce(top_here, Op::max), 0);
  const std::uint64_t fresh = comm.allreduce(fresh_here, Op::sum);
  if (fresh > static_cast<std::uint64_t>(std::numeric_limits<GO>::max() - top)) {
    throw std::overflow_error("merge: new gids would not fit a GO");
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto v = detail::take<std::uint64_t>(&chosen(i, 0), 0);
    out[i] = detail::take<std::uint8_t>(&chosen(i, 0), flag_at) != 0 ? static_cast<GO>(v)
                                                                       : top + 1 + static_cast<GO>(v);
  }
  return out;
}

// All trusted, and the largest gid given: what a builder's gids are.
std::vector<std::uint8_t> all_trusted(std::size_t n) { return std::vector<std::uint8_t>(n, 1); }
GO largest(std::span<const GO> gids) {
  GO top = 0;
  for (const GO g : gids) top = std::max(top, g);
  return top;
}

// merge_vertices with the vertices whose gids may stay and the largest gid
// in use here; each vertex's new gid, in the order added, in `gids`.
Sharing merge_into(const Comm& comm, MeshBuilder& builder, PointTolerance tol, MergeGids how,
                   const std::vector<std::uint8_t>& trusted, GO top_here, std::vector<GO>& gids, bool strict = false) {
  const PointIds m = match_points(comm, builder.vertex_coordinates(), tol);
  const std::vector<std::uint64_t> group(m.ids.begin(), m.ids.end());
  gids = choose_gids(comm, group, static_cast<std::uint64_t>(m.count), builder.vertex_gids(), trusted, top_here, how,
                     strict);
  on_every_rank(comm, builder, "merge_vertices", [&](MeshBuilder& b) { b.rename_vertices(gids, m.xyz); });
  return kept_rows(gids, m.sharing);
}

// number_elements with, per dimension, the elements whose gids may stay and
// the largest gid in use here.
std::array<std::optional<Sharing>, 4> number_into(const Comm& comm, MeshBuilder& builder, MergeGids how,
                                                  const std::array<std::vector<std::uint8_t>, 4>& trusted,
                                                  const std::array<GO, 4>& top_here, bool strict = false);

// The sorted corner gids of entity e of dimension d.
std::vector<GO> corner_key(const MeshView<Kokkos::HostSpace>& v, int d, LO e, const std::vector<GO>& gid_of_vertex) {
  std::vector<GO> key(static_cast<std::size_t>(v.num_vertices(d, e)));
  for (LO k = 0; k < v.num_vertices(d, e); ++k) {
    key[static_cast<std::size_t>(k)] = gid_of_vertex[static_cast<std::size_t>(v.vertex(d, e, k))];
  }
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

Sharing merge_vertices(const Comm& comm, MeshBuilder& builder, PointTolerance tol, MergeGids how) {
  std::vector<GO> gids;
  return merge_into(comm, builder, tol, how, all_trusted(builder.vertex_gids().size()), largest(builder.vertex_gids()),
                    gids);
}

namespace {

// merge_mesh; with `by_gid`, entities are one where their gids are, whatever
// their places. `known` keeps gids strictly where no rank's part has sharing
// any more.
Renumbering merge_live(const Comm& comm, HostMesh& mesh, const MergeOptions& options, bool by_gid, bool known) {
  using Access = detail::TopologyAccess;
  const PointTolerance tol = options.tol;
  const MergeGids how = options.gids;
  if (options.added) {
    bool bad = false;
    for (int d = 0; d <= 3; ++d) {
      for (const LO e : (*options.added)[static_cast<std::size_t>(d)]) bad = bad || e < 0 || e >= mesh.count(d);
    }
    if (comm.allreduce(static_cast<std::int32_t>(bad), Op::max) != 0) {
      throw std::invalid_argument("merge_mesh: an added entity is not a lid of the mesh");
    }
  }
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

  // The live vertices, and the live elements and explicit sides, as a
  // builder. Gids of entities added since the mesh was built (given, or its
  // tail: MeshEdit's, chosen by each rank alone) may not stay; the largest
  // gid of each dimension, removed entities included, bounds the new ones.
  std::array<std::vector<std::uint8_t>, 4> is_added;
  for (int d = 0; d <= 3; ++d) {
    auto& f = is_added[static_cast<std::size_t>(d)];
    f.assign(static_cast<std::size_t>(mesh.count(d)), 0);
    if (options.added) {
      for (const LO e : (*options.added)[static_cast<std::size_t>(d)]) f[static_cast<std::size_t>(e)] = 1;
    } else {
      for (LO e = mesh.count(d) - mesh.tail(d); e < mesh.count(d); ++e) f[static_cast<std::size_t>(e)] = 1;
    }
  }
  const auto added = [&](int d, LO e) { return is_added[static_cast<std::size_t>(d)][static_cast<std::size_t>(e)] != 0; };
  // On a distributed mesh the gids kept must already agree: one naming
  // different points on two ranks is an added entity not named as one.
  const bool strict =
      comm.allreduce(static_cast<std::int32_t>(known || mesh.fields().contains(sharing_field)), Op::max) != 0;
  std::array<std::vector<std::uint8_t>, 4> trusted;
  std::array<GO, 4> top{};
  for (int d = 0; d <= mesh.dim(); ++d) {
    for (LO e = 0; e < mesh.count(d); ++e) top[static_cast<std::size_t>(d)] = std::max(top[static_cast<std::size_t>(d)], v.gid(d, e));
  }
  MeshBuilder b(std::max(dim, 1));
  std::vector<LO> vlid;
  {
    std::vector<GO> gids;
    std::vector<Real> xyz;
    for (LO i = 0; i < mesh.count(0); ++i) {
      if (!v.alive(0, i)) continue;
      vlid.push_back(i);
      gids.push_back(v.gid(0, i));
      trusted[0].push_back(added(0, i) ? 0 : 1);
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
        trusted[static_cast<std::size_t>(d)].push_back(added(d, e) ? 0 : 1);
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
  std::array<std::optional<Sharing>, 4> sharing;
  if (by_gid) {
    merged = std::vector<GO>(b.vertex_gids().begin(), b.vertex_gids().end());
  } else {
    Sharing vertices = merge_into(comm, b, tol, how, trusted[0], top[0], merged, strict);
    sharing = number_into(comm, b, how, trusted, top, strict);
    sharing[0] = std::move(vertices);
  }
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

// Each field with one row per entity of a whole dimension: its owners' values
// to the copies and ghosts. Fields in name order; which qualify is agreed.
void owners_values(const Comm& comm, HostMesh& mesh) {
  const std::uint64_t h = detail::fields_hash(mesh);
  std::array<std::uint64_t, 2> same{h, ~h};
  comm.allreduce(std::span<std::uint64_t>(same), Op::max);
  if (same[0] != ~same[1]) throw std::invalid_argument("sync_mesh: the ranks' fields differ");
  std::vector<FieldInfo> list;  // not SOMA's own (soma:sharing, soma:owner), held only where entities are shared
  for (FieldInfo& f : mesh.fields().list()) {
    if (!f.name.starts_with("soma:")) list.push_back(std::move(f));
  }
  std::sort(list.begin(), list.end(), [](const FieldInfo& a, const FieldInfo& b) { return a.name < b.name; });
  std::vector<std::int32_t> fit(list.size(), 0);
  for (std::size_t i = 0; i < list.size(); ++i) {
    const FieldInfo& f = list[i];
    if (f.variable || f.handle || f.dims.size() != 1 || f.dims[0] > mesh.dim() || f.rows != mesh.count(f.dims[0])) continue;
    bool rows = true;
    dispatch_data_type(f.type, [&]<class T>() {
      const auto g = mesh.fields().template get<T>(f.id);
      for (LO e = 0; e < mesh.count(f.dims[0]) && rows; ++e) {
        const RowRange r = g.space().entity_rows(f.dims[0], e);
        rows = r.begin == e && r.end == e + 1;
      }
    });
    fit[i] = rows ? 1 : 0;
  }
  if (!fit.empty()) comm.allreduce(std::span<std::int32_t>(fit), Op::min);
  std::array<std::optional<HaloExchange<Kokkos::HostSpace>>, 4> halo;
  for (std::size_t i = 0; i < list.size(); ++i) {
    if (fit[i] == 0) continue;
    const int d = list[i].dims[0];
    if (!halo[static_cast<std::size_t>(d)]) halo[static_cast<std::size_t>(d)].emplace(comm, mesh, d);
    dispatch_data_type(list[i].type, [&]<class T>() {
      halo[static_cast<std::size_t>(d)]->update(comm, mesh.fields().template get<T>(list[i].id).view());
    });
  }
}

}  // namespace

Renumbering merge_mesh(const Comm& comm, HostMesh& mesh, const MergeOptions& options) {
  return merge_live(comm, mesh, options, false, false);
}

Renumbering sync_mesh(const Comm& comm, HostMesh& mesh, const SyncOptions& options) {
  using Access = detail::TopologyAccess;
  const detail::UnsyncedEdits edits = Access::edits(std::as_const(mesh));
  std::array<std::int32_t, 4> f{edits.unsynced ? 1 : 0, edits.adds ? 1 : 0};
  const std::array<int, 2> had = edits.ghosts[0] > 0 ? edits.ghosts : std::array{mesh.ghost_layers(), mesh.ghost_bridge_dim()};
  f[2] = had[0];
  f[3] = had[1];
  comm.allreduce(std::span<std::int32_t>(f), Op::max);
  Renumbering out;
  if (f[0] == 0) return out;

  // On a copy, so that a throw leaves the mesh as it was. The parts not edited
  // drop their ghosts too.
  HostMesh next = mesh.clone();
  std::exception_ptr failure;
  Renumbering packed;
  MergeOptions merge{options.tol, MergeGids::keep, std::array<std::vector<LO>, 4>{}};
  try {
    detail::drop_ghosts(next);
    packed = next.compress();
    for (int d = 0; d <= 3; ++d) {
      const auto& chosen = edits.chosen[static_cast<std::size_t>(d)];
      const std::set<GO> mine(chosen.begin(), chosen.end());
      if (mine.empty()) continue;
      for (LO e = 0; e < next.count(d); ++e) {
        if (mine.contains(next.gid(d)(e))) (*merge.added)[static_cast<std::size_t>(d)].push_back(e);
      }
    }
  } catch (...) {
    failure = std::current_exception();
  }
  if (comm.allreduce(static_cast<std::int32_t>(failure ? 1 : 0), Op::max) != 0) {
    if (failure) std::rethrow_exception(failure);
    throw std::runtime_error("sync_mesh: another rank failed");
  }
  out = merge_live(comm, next, merge, f[1] == 0, true);
  for (int d = 0; d <= 3; ++d) {
    const auto& first = packed.new_lid[static_cast<std::size_t>(d)];
    auto& then = out.new_lid[static_cast<std::size_t>(d)];
    if (first.empty()) continue;
    std::vector<LO> both(first.size(), invalid_lo);
    for (std::size_t i = 0; i < first.size(); ++i) {
      if (first[i] != invalid_lo) both[i] = then[static_cast<std::size_t>(first[i])];
    }
    then = std::move(both);
  }
  if (options.ghosts && f[2] > 0 && comm.size() > 1) next.add_ghosts(comm, GhostOptions{f[2], f[3], options.ghost_sets});
  if (options.fields && comm.size() > 1) owners_values(comm, next);
  mesh.swap(next);
  return out;
}

HostMesh build_merged_part(const Comm& comm, MeshBuilder&& builder, PointTolerance tol, MergeGids how,
                           Ordering ordering) {
  // Vertices first: elements are named by their merged corners.
  Sharing vertices = merge_vertices(comm, builder, tol, how);
  std::array<std::optional<Sharing>, 4> sharing = number_elements(comm, builder, how);
  sharing[0] = std::move(vertices);
  return build_local_part(comm, std::move(builder), sharing, ordering);
}

std::array<std::optional<Sharing>, 4> number_elements(const Comm& comm, MeshBuilder& builder, MergeGids how) {
  std::array<std::vector<std::uint8_t>, 4> trusted;
  std::array<GO, 4> top{};
  for (int d = 1; d <= 3; ++d) {
    const std::vector<GO> g = builder.element_gids(d);
    trusted[static_cast<std::size_t>(d)] = all_trusted(g.size());
    top[static_cast<std::size_t>(d)] = largest(g);
  }
  return number_into(comm, builder, how, trusted, top);
}

namespace {

std::array<std::optional<Sharing>, 4> number_into(const Comm& comm, MeshBuilder& builder, MergeGids how,
                                                  const std::array<std::vector<std::uint8_t>, 4>& trusted,
                                                  const std::array<GO, 4>& top_here, bool strict) {
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
    const auto k = static_cast<std::size_t>(d);
    const std::vector<GO> old = builder.element_gids(d);
    auto& g = gids[k];
    g = choose_gids(comm, ranks.index, ranks.total, old, trusted[k], top_here[k], how, strict);
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

}  // namespace

}  // namespace soma
