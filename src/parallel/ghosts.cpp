// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/ghosts.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/parallel/sharing.hpp"
#include "soma/topology/build.hpp"

#include "closure.hpp"
#include "records.hpp"
#include "side_keys.hpp"

namespace soma::detail {

namespace {

struct Element {
  Topology topology;
  std::vector<GO> corners;
};

}  // namespace

GhostTargets ghost_targets(const Comm& comm, const HostMesh& mesh, const GhostOptions& options) {
  const int top = mesh.dim(), me = comm.rank(), b = options.bridge_dim;
  const bool sharing = comm.size() == 1 || mesh.fields().contains(sharing_field);
  // With no vertex shared on any rank no element is joined to another
  // rank's, so there are no targets.
  const SegmentTable& st = mesh.segment_table(0);
  std::array<std::int64_t, 6> f{options.layers, -options.layers, b, -b, sharing ? 0 : 1,
                                st.shared_end > st.interior_end ? 1 : 0};
  comm.allreduce(std::span<std::int64_t>(f), Op::max);
  if (f[0] != -f[1] || f[2] != -f[3]) throw std::invalid_argument("ghost_targets: the ranks' options differ");
  if (options.layers < 0) throw std::invalid_argument("ghost_targets: a negative layer count");
  if (b < 0 || b >= top) throw std::invalid_argument("ghost_targets: bridge_dim is not below the mesh's dimension");
  if (f[4] != 0) throw std::invalid_argument("ghost_targets: a rank's mesh has no sharing (build_local_part gives it)");

  GhostTargets out;
  out.offsets.assign(static_cast<std::size_t>(mesh.count(top)) + 1, 0);
  if (options.layers == 0 || f[5] == 0) return out;

  // The other holders of each shared vertex, then of vertices learnt with ghosts.
  const auto vgid = mesh.gid(0);
  std::map<GO, std::vector<int>> holders;
  const SegmentTable& vt = mesh.segment_table(0);
  if (vt.shared_end > vt.interior_end) {
    const auto s = mesh.fields().get_var<int>(sharing_field, 0);
    for (LO v = vt.interior_end; v < vt.shared_end; ++v) {
      const auto row = s.value_or_default(0, v);
      holders[vgid(v)].assign(row.begin(), row.end());
    }
  }
  // Only sides whose lowest vertex is held elsewhere are asked about, so the
  // first frontier is the held elements with a shared vertex. An owned
  // element answered in round i has a side of an element of the asker's
  // frontier: held elsewhere too, its corners are shared and the element is
  // on the boundary; else owned here and answered in round i - 1. So the
  // answers are within layers - 1 sides of the boundary, and only those
  // owned elements (by vertices, a wider set) are indexed by their sides.
  const LO owned = mesh.segment_table(top).owned_end, held = mesh.segment_table(top).shared_end;
  const auto off = mesh.conn_offsets(top);
  const auto conn = mesh.conn(top);
  const auto shared = [&](LO v) { return v >= vt.interior_end && v < vt.shared_end; };
  std::vector<char> near(static_cast<std::size_t>(held), 0);
  std::vector<LO> boundary;
  for (LO e = 0; e < held; ++e) {
    for (Offset q = off(e); q < off(e + 1); ++q) {
      if (!shared(conn(q))) continue;
      near[static_cast<std::size_t>(e)] = 1;
      boundary.push_back(e);
      break;
    }
  }
  if (options.layers > 1) {
    // The owned elements of each vertex, then layers - 1 steps out.
    std::vector<std::size_t> first(static_cast<std::size_t>(mesh.count(0)) + 1, 0);
    for (Offset q = 0; q < off(owned); ++q) ++first[static_cast<std::size_t>(conn(q)) + 1];
    for (std::size_t v = 1; v < first.size(); ++v) first[v] += first[v - 1];
    std::vector<LO> of(first.back());
    {
      std::vector<std::size_t> at(first.begin(), first.end() - 1);
      for (LO e = 0; e < owned; ++e) {
        for (Offset q = off(e); q < off(e + 1); ++q) of[at[static_cast<std::size_t>(conn(q))]++] = e;
      }
    }
    std::vector<LO> step;
    for (const LO e : boundary) {
      if (e < owned) step.push_back(e);
    }
    for (int i = 1; i < options.layers && !step.empty(); ++i) {
      std::vector<LO> next;
      for (const LO e : step) {
        for (Offset q = off(e); q < off(e + 1); ++q) {
          const auto v = static_cast<std::size_t>(conn(q));
          for (std::size_t j = first[v]; j < first[v + 1]; ++j) {
            if (near[static_cast<std::size_t>(of[j])] != 0) continue;
            near[static_cast<std::size_t>(of[j])] = 1;
            next.push_back(of[j]);
          }
        }
      }
      step = std::move(next);
    }
  }
  std::vector<std::pair<SideKey, LO>> index;
  for (LO e = 0; e < owned; ++e) {
    if (near[static_cast<std::size_t>(e)] != 0) {
      for_each_side(mesh, top, e, b, [&](const SideKey& k) { index.emplace_back(k, e); });
    }
  }
  std::sort(index.begin(), index.end());
  // Elements held here that another rank may answer with: the copies, then
  // the ghosts taken.
  const auto egid = mesh.gid(top);
  std::vector<GO> known;
  for (LO e = owned; e < held; ++e) known.push_back(egid(e));
  std::sort(known.begin(), known.end());
  std::vector<Element> frontier;
  for (const LO e : boundary) {
    frontier.push_back({static_cast<Topology>(mesh.topology(top)(e)), corner_gids(mesh, top, e)});
  }
  std::int64_t widest = 0;
  for (LO e = 0; e < owned; ++e) {
    widest = std::max<std::int64_t>(widest, static_cast<std::int64_t>(off(e + 1) - off(e)));
  }
  widest = comm.allreduce(widest, Op::max);
  // An answer: gid, topology, corner count, lid on its owner, corners.
  constexpr std::size_t head = sizeof(GO) + 2 * sizeof(std::int32_t) + sizeof(LO);
  const std::size_t width = head + static_cast<std::size_t>(widest) * sizeof(GO);

  std::vector<std::pair<int, LO>> taken;  // (owner of a ghost element, its lid there)
  for (int layer = 0; layer < options.layers; ++layer) {
    // Each side of the frontier to the other holders of its lowest vertex.
    std::vector<std::pair<int, SideKey>> asks;
    for (const Element& x : frontier) {
      for_each_side(x.topology, std::span<const GO>(x.corners), b, [&](const SideKey& k) {
        const auto it = holders.find(k[0]);
        if (it == holders.end()) return;
        for (const int r : it->second) {
          if (r != me) asks.emplace_back(r, k);
        }
      });
    }
    std::sort(asks.begin(), asks.end());
    asks.erase(std::unique(asks.begin(), asks.end()), asks.end());
    std::vector<int> to(asks.size());
    for (std::size_t i = 0; i < asks.size(); ++i) to[i] = asks[i].first;
    const Batch in = route(comm, to, sizeof(SideKey), [&](std::size_t i, std::byte* row) { put(row, 0, asks[i].second); });

    // Owned elements with each side asked about, once per asker.
    std::vector<std::pair<int, LO>> answers;
    for (std::size_t i = 0; i < in.size(); ++i) {
      const auto k = take<SideKey>(&in.payload(i, 0), 0);
      for (auto it = std::lower_bound(index.begin(), index.end(), std::pair<SideKey, LO>{k, std::numeric_limits<LO>::min()});
           it != index.end() && it->first == k; ++it) {
        answers.emplace_back(in.source(i), it->second);
      }
    }
    std::sort(answers.begin(), answers.end());
    answers.erase(std::unique(answers.begin(), answers.end()), answers.end());
    to.resize(answers.size());
    for (std::size_t i = 0; i < answers.size(); ++i) to[i] = answers[i].first;
    const Batch got = route(comm, to, width, [&](std::size_t i, std::byte* row) {
      const LO e = answers[i].second;
      const std::vector<GO> c = corner_gids(mesh, top, e);
      put(row, 0, egid(e));
      put(row, sizeof(GO), static_cast<std::int32_t>(mesh.topology(top)(e)));
      put(row, sizeof(GO) + sizeof(std::int32_t), static_cast<std::int32_t>(c.size()));
      put(row, sizeof(GO) + 2 * sizeof(std::int32_t), e);
      for (std::size_t q = 0; q < c.size(); ++q) put(row, head + q * sizeof(GO), c[q]);
    });
    // The holders of the answers' vertices, for the next round: this rank
    // and its sharing.
    std::vector<std::tuple<int, GO, int>> told;  // (asker, vertex, holder)
    const bool last = layer + 1 == options.layers;
    for (const auto& [asker, e] : last ? decltype(answers){} : answers) {
      for (const GO v : corner_gids(mesh, top, e)) {
        told.emplace_back(asker, v, me);
        const auto it = holders.find(v);
        if (it == holders.end()) continue;
        for (const int r : it->second) told.emplace_back(asker, v, r);
      }
    }
    std::sort(told.begin(), told.end());
    told.erase(std::unique(told.begin(), told.end()), told.end());
    to.resize(told.size());
    for (std::size_t i = 0; i < told.size(); ++i) to[i] = std::get<0>(told[i]);
    const Batch heard = last ? Batch{} : route(comm, to, sizeof(GO) + sizeof(int), [&](std::size_t i, std::byte* row) {
      put(row, 0, std::get<1>(told[i]));
      put(row, sizeof(GO), std::get<2>(told[i]));
    });

    // The next layer: answers not held here yet (each element comes once a
    // round, from its owner).
    for (std::size_t i = 0; i < heard.size(); ++i) {
      std::vector<int>& h = holders[take<GO>(&heard.payload(i, 0), 0)];
      const int r = take<int>(&heard.payload(i, 0), sizeof(GO));
      if (std::find(h.begin(), h.end(), r) == h.end()) h.push_back(r);
    }
    frontier.clear();
    std::vector<GO> fresh;
    for (std::size_t i = 0; i < got.size(); ++i) {
      const std::byte* row = &got.payload(i, 0);
      const GO g = take<GO>(row, 0);
      if (std::binary_search(known.begin(), known.end(), g)) continue;
      Element x{static_cast<Topology>(take<std::int32_t>(row, sizeof(GO))), {}};
      const auto n = static_cast<std::size_t>(take<std::int32_t>(row, sizeof(GO) + sizeof(std::int32_t)));
      for (std::size_t q = 0; q < n; ++q) x.corners.push_back(take<GO>(row, head + q * sizeof(GO)));
      fresh.push_back(g);
      taken.emplace_back(got.source(i), take<LO>(row, sizeof(GO) + 2 * sizeof(std::int32_t)));
      frontier.push_back(std::move(x));
    }
    std::sort(fresh.begin(), fresh.end());
    const auto middle = known.insert(known.end(), fresh.begin(), fresh.end());
    std::inplace_merge(known.begin(), middle, known.end());
  }

  // Each owner learns who ghosts its elements.
  std::vector<int> to(taken.size());
  for (std::size_t i = 0; i < taken.size(); ++i) to[i] = taken[i].first;
  const Batch asked = route(comm, to, sizeof(LO), [&](std::size_t i, std::byte* row) { put(row, 0, taken[i].second); });
  std::vector<std::pair<LO, int>> rows;
  for (std::size_t i = 0; i < asked.size(); ++i) rows.emplace_back(take<LO>(&asked.payload(i, 0), 0), asked.source(i));
  std::sort(rows.begin(), rows.end());
  out.ranks.reserve(rows.size());
  std::size_t k = 0;
  for (LO e = 0; e < mesh.count(top); ++e) {
    for (; k < rows.size() && rows[k].first == e; ++k) out.ranks.push_back(rows[k].second);
    out.offsets[static_cast<std::size_t>(e) + 1] = out.ranks.size();
  }
  return out;
}

namespace {

// The part's dimension d permuted so the entities `old` held (not as ghosts)
// take their ids there; ghosts stay after them, as the part has them.
void keep_ids(const HostMesh& old, HostMesh& part, int d) {
  const SegmentTable& was = old.segment_table(d);
  const SegmentTable& now = part.segment_table(d);
  if (now.shared_end != was.shared_end || now.owned_end != was.owned_end || now.interior_end != was.interior_end) {
    throw std::logic_error("add_ghosts: the part does not hold what the rank held before");
  }
  bool in_place = true;
  for (LO e = 0; e < was.shared_end && in_place; ++e) in_place = part.gid(d)(e) == old.gid(d)(e);
  if (in_place) return;
  std::vector<std::pair<GO, LO>> lid(static_cast<std::size_t>(was.shared_end));
  for (LO e = 0; e < was.shared_end; ++e) lid[static_cast<std::size_t>(e)] = {old.gid(d)(e), e};
  std::sort(lid.begin(), lid.end());
  std::vector<LO> new_of_old(static_cast<std::size_t>(part.count(d)));
  bool same = true;
  for (LO e = 0; e < part.count(d); ++e) {
    LO to = e;
    if (e < now.shared_end) {
      const GO g = part.gid(d)(e);
      const auto it = std::lower_bound(lid.begin(), lid.end(), std::pair<GO, LO>{g, std::numeric_limits<LO>::min()});
      if (it == lid.end() || it->first != g) throw std::logic_error("add_ghosts: the part does not hold what the rank held before");
      to = it->second;
    }
    new_of_old[static_cast<std::size_t>(e)] = to;
    same = same && to == e;
  }
  if (same) return;
  SegmentTable table;
  table.count = now.count;
  table.interior_end = was.interior_end;
  table.owned_end = was.owned_end;
  table.shared_end = was.shared_end;
  for (const Segment& s : was.segments) {
    if (s.begin < was.shared_end) table.segments.push_back(s);
  }
  for (const Segment& s : now.segments) {
    if (s.begin >= now.shared_end) table.segments.push_back(s);
  }
  permute_entities(part, d, new_of_old, table);
}

// Ghost closures on their way from the owners of the ghost elements. An
// entity: gid, topology, corner count, node count beyond the corners, owner,
// then the corners' and nodes' gids, padded to the widest.
constexpr std::size_t entity_header = sizeof(GO) + 4 * sizeof(std::int32_t);
// A vertex: gid, owner, coordinates.
constexpr std::size_t vertex_width = sizeof(GO) + sizeof(std::int32_t) + 3 * sizeof(Real);
// An entity below the mesh's dimension: dimension, owner, gid, key.
constexpr std::size_t side_width = 2 * sizeof(std::int32_t) + sizeof(GO) + sizeof(SideKey);

struct Arrived {
  GO gid = 0;
  Topology topology = Topology::vertex;
  std::int32_t corners = 0, extra = 0, owner = 0;
  std::vector<GO> nodes;  // corners, then nodes
  std::array<Real, 3> xyz{};
};

// The owner of each held entity (other than as a ghost) of dimension d.
std::vector<int> held_owners(const HostMesh& mesh, int d, int me) {
  const SegmentTable& t = mesh.segment_table(d);
  std::vector<int> out(static_cast<std::size_t>(t.shared_end), me);
  if (t.owned_end == t.shared_end) return out;
  const auto o = mesh.fields().get<int>(owner_field, d);
  for (LO e = t.owned_end; e < t.shared_end; ++e) out[static_cast<std::size_t>(e)] = o.value_or_default(d, e);
  return out;
}

// The held entities of dimension d (gid, lid), sorted.
std::vector<std::pair<GO, LO>> held_by_gid(const HostMesh& mesh, int d) {
  std::vector<std::pair<GO, LO>> out;
  for (LO e = 0; e < mesh.segment_table(d).shared_end; ++e) out.emplace_back(mesh.gid(d)(e), e);
  std::sort(out.begin(), out.end());
  return out;
}

LO find(const std::vector<std::pair<GO, LO>>& v, GO g) {
  const auto it = std::lower_bound(v.begin(), v.end(), std::pair<GO, LO>{g, std::numeric_limits<LO>::min()});
  return it == v.end() || it->first != g ? -1 : it->second;
}

// Entity e of dimension d (a vertex with its coordinates) as this rank holds
// it, with its owner.
Arrived arrived_of(const HostMesh& mesh, int d, LO e, int owner) {
  Arrived a;
  a.gid = mesh.gid(d)(e);
  a.owner = owner;
  if (d == 0) {
    for (std::size_t q = 0; q < 3; ++q) a.xyz[q] = mesh.coords()(e, q);
    return a;
  }
  a.topology = static_cast<Topology>(mesh.topology(d)(e));
  a.nodes = corner_gids(mesh, d, e);
  a.corners = static_cast<std::int32_t>(a.nodes.size());
  if (mesh.has_nodes(d)) {
    const auto noff = mesh.node_offsets(d);
    const auto nodes = mesh.nodes(d);
    for (Offset k = noff(e); k < noff(e + 1); ++k) a.nodes.push_back(mesh.gid(0)(nodes(k)));
    a.extra = static_cast<std::int32_t>(noff(e + 1) - noff(e));
  }
  return a;
}

void put_entity(std::byte* row, const Arrived& a) {
  put(row, 0, a.gid);
  put(row, sizeof(GO), static_cast<std::int32_t>(a.topology));
  put(row, sizeof(GO) + sizeof(std::int32_t), a.corners);
  put(row, sizeof(GO) + 2 * sizeof(std::int32_t), a.extra);
  put(row, sizeof(GO) + 3 * sizeof(std::int32_t), a.owner);
  for (std::size_t k = 0; k < a.nodes.size(); ++k) put(row, entity_header + k * sizeof(GO), a.nodes[k]);
}

Arrived take_entity(const std::byte* row) {
  Arrived a;
  a.gid = take<GO>(row, 0);
  a.topology = static_cast<Topology>(take<std::int32_t>(row, sizeof(GO)));
  a.corners = take<std::int32_t>(row, sizeof(GO) + sizeof(std::int32_t));
  a.extra = take<std::int32_t>(row, sizeof(GO) + 2 * sizeof(std::int32_t));
  a.owner = take<std::int32_t>(row, sizeof(GO) + 3 * sizeof(std::int32_t));
  a.nodes.resize(static_cast<std::size_t>(a.corners + a.extra));
  for (std::size_t k = 0; k < a.nodes.size(); ++k) a.nodes[k] = take<GO>(row, entity_header + k * sizeof(GO));
  return a;
}

void put_vertex(std::byte* row, const Arrived& a) {
  put(row, 0, a.gid);
  put(row, sizeof(GO), a.owner);
  for (std::size_t q = 0; q < 3; ++q) put(row, sizeof(GO) + sizeof(std::int32_t) + q * sizeof(Real), a.xyz[q]);
}

Arrived take_vertex(const std::byte* row) {
  Arrived a;
  a.gid = take<GO>(row, 0);
  a.owner = take<std::int32_t>(row, sizeof(GO));
  for (std::size_t q = 0; q < 3; ++q) a.xyz[q] = take<Real>(row, sizeof(GO) + sizeof(std::int32_t) + q * sizeof(Real));
  return a;
}

// Records by gid, once each.
void by_gid(std::vector<Arrived>& v) {
  std::sort(v.begin(), v.end(), [](const Arrived& x, const Arrived& y) { return x.gid < y.gid; });
  v.erase(std::unique(v.begin(), v.end(), [](const Arrived& x, const Arrived& y) { return x.gid == y.gid; }), v.end());
}

const Arrived* find(const std::vector<Arrived>& v, GO g) {
  const auto it = std::lower_bound(v.begin(), v.end(), g, [](const Arrived& x, GO y) { return x.gid < y; });
  return it == v.end() || it->gid != g ? nullptr : &*it;
}

// Rows of (rank, dimension, lid) once each, sorted.
void sort_unique(std::vector<std::tuple<int, int, LO>>& v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

// A side of a ghost element known from its owner: key, gid, owner.
struct KnownSide {
  SideKey key;
  GO gid = 0;
  int owner = 0;
  friend bool operator<(const KnownSide& x, const KnownSide& y) { return x.key < y.key; }
};

// What `mesh` holds other than as ghosts, in place, its copies taking their
// owners' versions (`theirs`), then the ghosts of `tail` in its order: a mesh
// of the ghosts after the held vertices they use (`held`: the held entities
// by gid). The dimensions `given` take the tail's entities; those `derived`
// keep their sides held (`by_key`) and add the ghost elements' other sides
// (`known`, by key) as derive_sides numbers them: each by its lowest
// (element, side), with its vertices in that side's order, by topology and
// arity, then (as distribute_known orders them) by owner. The others are
// left to derive.
template <class Given, class Derived>
HostMesh splice(const HostMesh& mesh, const HostMesh& tail, Given&& given, Derived&& derived,
                const std::array<std::vector<Arrived>, 4>& theirs,
                const std::array<std::vector<std::pair<GO, LO>>, 4>& held,
                const std::array<std::vector<std::pair<SideKey, LO>>, 4>& by_key,
                const std::array<std::vector<KnownSide>, 4>& known) {
  using Access = TopologyAccess;
  const int top = mesh.dim();
  const auto uninit = [](const char* label) {
    return Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing);
  };
  HostMesh part;
  Access::shape(part, top, mesh.ordering());
  const LO held_vertices = mesh.segment_table(0).shared_end, first_ghost = tail.segment_table(0).shared_end;
  std::vector<LO> vertex_of(static_cast<std::size_t>(tail.count(0)));
  for (LO l = 0; l < tail.count(0); ++l) {
    vertex_of[static_cast<std::size_t>(l)] =
        l >= first_ghost ? held_vertices + (l - first_ghost) : find(held[0], tail.gid(0)(l));
  }
  for (int d = 0; d <= top; ++d) {
    if (!given(d)) continue;
    const auto dd = static_cast<std::size_t>(d);
    const SegmentTable& t = mesh.segment_table(d);
    const SegmentTable& u = tail.segment_table(d);
    const LO nh = t.shared_end, t0 = u.shared_end, n = nh + (tail.count(d) - t0);
    const auto at = [&](LO e) { return static_cast<std::size_t>(nh + (e - t0)); };
    EntityBlock<Kokkos::HostSpace>& blk = Access::block(part, d);
    blk.count = n;
    blk.input = d == 0 || n > 0;
    blk.gid = View1<GO, Kokkos::HostSpace>(uninit("soma::mesh::gid"), static_cast<std::size_t>(n));
    for (LO e = 0; e < nh; ++e) blk.gid(static_cast<std::size_t>(e)) = mesh.gid(d)(e);
    for (LO e = t0; e < tail.count(d); ++e) blk.gid(at(e)) = tail.gid(d)(e);
    SegmentTable table;
    table.count = n;
    table.interior_end = t.interior_end;
    table.owned_end = t.owned_end;
    table.shared_end = nh;
    for (const Segment& g : t.segments) {
      if (g.begin < nh) table.segments.push_back(g);
    }
    for (Segment g : u.segments) {
      if (g.begin < t0) continue;
      g.begin = nh + (g.begin - t0);
      g.end = nh + (g.end - t0);
      table.segments.push_back(g);
    }
    blk.segments = std::move(table);
    if (d == 0) {
      CoordView<Kokkos::HostSpace>& xyz = Access::coords(part);
      xyz = CoordView<Kokkos::HostSpace>(uninit("soma::mesh::coords"), static_cast<std::size_t>(n));
      for (LO v = 0; v < nh; ++v) {
        for (std::size_t q = 0; q < 3; ++q) xyz(static_cast<std::size_t>(v), q) = mesh.coords()(v, q);
      }
      for (const Arrived& a : theirs[0]) {
        const auto v = static_cast<std::size_t>(find(held[0], a.gid));
        for (std::size_t q = 0; q < 3; ++q) xyz(v, q) = a.xyz[q];
      }
      for (LO v = t0; v < tail.count(0); ++v) {
        for (std::size_t q = 0; q < 3; ++q) xyz(at(v), q) = tail.coords()(v, q);
      }
      std::array<LO, 4>& ends = Access::corner_ends(part);
      for (std::size_t k = 0; k < 3; ++k) ends[k] = mesh.corner_end(static_cast<Ownership>(k));
      ends[3] = nh + (tail.corner_end(Ownership::ghost) - t0);
      continue;
    }
    blk.topology = View1<std::uint8_t, Kokkos::HostSpace>(uninit("soma::mesh::topology"), static_cast<std::size_t>(n));
    for (LO e = 0; e < nh; ++e) blk.topology(static_cast<std::size_t>(e)) = mesh.topology(d)(e);
    for (LO e = t0; e < tail.count(d); ++e) blk.topology(at(e)) = tail.topology(d)(e);
    // Corners, then the nodes beyond them: the held as they are, a copy's as
    // its owner has them, the ghosts' renamed.
    const auto join = [&](View1<Offset, Kokkos::HostSpace>& offsets, View1<LO, Kokkos::HostSpace>& ids, bool nodes) {
      const auto off = nodes ? mesh.node_offsets(d) : mesh.conn_offsets(d);
      const auto from = nodes ? mesh.nodes(d) : mesh.conn(d);
      const auto uoff = nodes ? tail.node_offsets(d) : tail.conn_offsets(d);
      const auto ufrom = nodes ? tail.nodes(d) : tail.conn(d);
      const bool mine = off.extent(0) > 0, theirs_too = uoff.extent(0) > 0;
      const Offset kept = mine ? off(nh) : 0, more = theirs_too ? uoff(tail.count(d)) - uoff(t0) : 0;
      if (nodes && kept + more == 0) return;
      offsets = View1<Offset, Kokkos::HostSpace>(uninit("soma::mesh::offsets"), static_cast<std::size_t>(n) + 1);
      ids = View1<LO, Kokkos::HostSpace>(uninit("soma::mesh::conn"), static_cast<std::size_t>(kept + more));
      for (LO e = 0; e <= nh; ++e) offsets(static_cast<std::size_t>(e)) = mine ? off(e) : 0;
      for (Offset k = 0; k < kept; ++k) ids(static_cast<std::size_t>(k)) = from(k);
      for (const Arrived& a : theirs[dd]) {
        const LO e = find(held[dd], a.gid);
        const Offset begin = offsets(static_cast<std::size_t>(e)), end = offsets(static_cast<std::size_t>(e) + 1);
        const std::size_t skip = nodes ? static_cast<std::size_t>(a.corners) : 0;
        if (end - begin != (nodes ? a.extra : a.corners)) {
          throw std::logic_error("add_ghosts: a copy and its owner's entity differ in their nodes");
        }
        for (Offset k = begin; k < end; ++k) {
          ids(static_cast<std::size_t>(k)) = find(held[0], a.nodes[skip + static_cast<std::size_t>(k - begin)]);
        }
      }
      Offset w = kept;
      for (LO e = t0; e < tail.count(d); ++e) {
        if (theirs_too) {
          for (Offset k = uoff(e); k < uoff(e + 1); ++k) {
            ids(static_cast<std::size_t>(w++)) = vertex_of[static_cast<std::size_t>(ufrom(k))];
          }
        }
        offsets(at(e) + 1) = w;
      }
    };
    join(blk.conn_offsets, blk.conn, false);
    join(blk.node_offsets, blk.nodes, true);
  }
  const SegmentTable& et = part.segment_table(top);
  for (int k = 1; k < top; ++k) {
    if (given(k) || !derived(k)) continue;
    const auto kk = static_cast<std::size_t>(k);
    if (mesh.has_nodes(k)) throw std::logic_error("add_ghosts: derived sides with nodes");
    struct Fresh {
      SideKey key;
      std::size_t slot;  // (element, side), in order
      Topology topology;
      std::array<LO, 4> v;
      int n;
    };
    std::vector<Fresh> fresh;
    const auto off = part.conn_offsets(top);
    const auto conn = part.conn(top);
    const auto vgid = part.gid(0);
    for (LO e = et.shared_end; e < part.count(top); ++e) {
      const auto t = static_cast<Topology>(part.topology(top)(e));
      const auto corners = static_cast<int>(off(e + 1) - off(e));
      const int sides = t == Topology::polygon ? (k == 1 ? corners : 0) : canonical::num_sides(t, k);
      for (int q = 0; q < sides; ++q) {
        Fresh x{{}, fresh.size(), Topology::edge, {}, 2};
        if (t == Topology::polygon) {
          x.v = {conn(off(e) + q), conn(off(e) + (q + 1) % corners), 0, 0};
        } else {
          const canonical::SideDef def = canonical::side(t, k, q);
          x.topology = def.topology;
          x.n = def.num_vertices;
          for (int i = 0; i < x.n; ++i) x.v[static_cast<std::size_t>(i)] = conn(off(e) + def.v[i]);
        }
        x.key.fill(std::numeric_limits<GO>::max());
        for (int i = 0; i < x.n; ++i) x.key[static_cast<std::size_t>(i)] = vgid(x.v[static_cast<std::size_t>(i)]);
        std::sort(x.key.begin(), x.key.end());
        const auto& h = by_key[kk];
        const auto it =
            std::lower_bound(h.begin(), h.end(), std::pair<SideKey, LO>{x.key, std::numeric_limits<LO>::min()});
        if (it == h.end() || it->first != x.key) fresh.push_back(x);
      }
    }
    // Each side once, by its lowest (element, side); then numbered.
    std::stable_sort(fresh.begin(), fresh.end(), [](const Fresh& x, const Fresh& y) { return x.key < y.key; });
    fresh.erase(std::unique(fresh.begin(), fresh.end(), [](const Fresh& x, const Fresh& y) { return x.key == y.key; }),
                fresh.end());
    std::vector<std::pair<const Fresh*, const KnownSide*>> sides;
    for (const Fresh& x : fresh) {
      const auto it = std::lower_bound(known[kk].begin(), known[kk].end(), KnownSide{x.key, 0, 0});
      if (it == known[kk].end() || it->key != x.key) {
        throw std::logic_error("add_ghosts: no gid is known for a side of a ghost of dimension " + std::to_string(k));
      }
      sides.emplace_back(&x, &*it);
    }
    std::sort(sides.begin(), sides.end(), [](const auto& x, const auto& y) {
      return std::tuple(x.first->topology, x.first->n, x.second->owner, x.first->slot) <
             std::tuple(y.first->topology, y.first->n, y.second->owner, y.first->slot);
    });
    const SegmentTable& t = mesh.segment_table(k);
    const LO nh = t.shared_end, n = nh + static_cast<LO>(sides.size());
    const auto moff = mesh.conn_offsets(k);
    const Offset kept = nh > 0 ? moff(nh) : 0;
    Offset entries = kept;
    for (const auto& x : sides) entries += x.first->n;
    EntityBlock<Kokkos::HostSpace>& blk = Access::block(part, k);
    blk.count = n;
    blk.derived = true;
    blk.gid = View1<GO, Kokkos::HostSpace>(uninit("soma::mesh::gid"), static_cast<std::size_t>(n));
    blk.topology = View1<std::uint8_t, Kokkos::HostSpace>(uninit("soma::mesh::topology"), static_cast<std::size_t>(n));
    blk.conn_offsets = View1<Offset, Kokkos::HostSpace>(uninit("soma::mesh::offsets"), static_cast<std::size_t>(n) + 1);
    blk.conn = View1<LO, Kokkos::HostSpace>(uninit("soma::mesh::conn"), static_cast<std::size_t>(entries));
    blk.conn_offsets(0) = 0;
    for (LO e = 0; e < nh; ++e) {
      const auto i = static_cast<std::size_t>(e);
      blk.gid(i) = mesh.gid(k)(e);
      blk.topology(i) = mesh.topology(k)(e);
      blk.conn_offsets(i + 1) = moff(e + 1);
    }
    for (Offset q = 0; q < kept; ++q) blk.conn(static_cast<std::size_t>(q)) = mesh.conn(k)(q);
    SegmentTable table;
    table.count = n;
    table.interior_end = t.interior_end;
    table.owned_end = t.owned_end;
    table.shared_end = nh;
    for (const Segment& g : t.segments) {
      if (g.begin < nh) table.segments.push_back(g);
    }
    Offset w = kept;
    for (std::size_t r = 0; r < sides.size(); ++r) {
      const Fresh& x = *sides[r].first;
      const auto i = static_cast<std::size_t>(nh) + r;
      blk.gid(i) = sides[r].second->gid;
      blk.topology(i) = static_cast<std::uint8_t>(x.topology);
      for (int q = 0; q < x.n; ++q) blk.conn(static_cast<std::size_t>(w++)) = x.v[static_cast<std::size_t>(q)];
      blk.conn_offsets(i + 1) = w;
      const auto arity = static_cast<std::uint16_t>(x.n);
      const auto e = static_cast<LO>(i);
      if (table.segments.empty() || table.segments.back().begin < nh ||
          table.segments.back().topology != x.topology || table.segments.back().arity != arity) {
        table.segments.push_back(Segment{e, e + 1, x.topology, arity, Ownership::ghost});
      } else {
        table.segments.back().end = e + 1;
      }
    }
    blk.segments = std::move(table);
  }
  Access::attach_fields(part);
  return part;
}

// The part with the ghosts given by `targets`, built as build_placed builds
// it from what place_closure places, but from what this rank holds and what
// owners send it, with the sharing known: the same part, without a
// directory. As there, every entity is as its owner has it: owners first
// send the copies of their shared entities (vertices and the dimensions
// given to the builder) what they hold, then send the ranks ghosting their
// elements the closures, as their copies now hold them.
HostMesh append_ghosts(const Comm& comm, const HostMesh& mesh, const GhostTargets& targets,
                       const GhostOptions& options) {
  const int top = mesh.dim(), me = comm.rank();
  // Lower dimensions given to the builder on some rank, as place_closure
  // finds them; the box of the owned vertices of every rank (the negated
  // lows, the highs): the box of the part built whole; dimensions held on
  // some rank.
  std::array<Real, 14> f{};
  for (int d = 1; d < top; ++d) f[static_cast<std::size_t>(d)] = mesh.input(d) && mesh.count(d) > 0 ? 1 : 0;
  for (std::size_t q = 4; q < 10; ++q) f[q] = -std::numeric_limits<Real>::infinity();
  for (int d = 0; d <= top; ++d) f[10 + static_cast<std::size_t>(d)] = mesh.count(d) > 0 ? 1 : 0;
  for (LO v = 0; v < mesh.segment_table(0).owned_end; ++v) {
    for (std::size_t q = 0; q < 3; ++q) {
      f[4 + q] = std::max(f[4 + q], -mesh.coords()(v, q));
      f[7 + q] = std::max(f[7 + q], mesh.coords()(v, q));
    }
  }
  comm.allreduce(std::span<Real>(f), Op::max);
  const auto given = [&](int d) { return d == 0 || d == top || f[static_cast<std::size_t>(d)] != 0; };
  const auto derived = [&](int d) { return !given(d) && f[10 + static_cast<std::size_t>(d)] != 0; };
  std::array<std::vector<int>, 4> owner;
  for (int d = 0; d <= top; ++d) owner[static_cast<std::size_t>(d)] = held_owners(mesh, d, me);
  const auto local = [&](int d, LO e) {
    return arrived_of(mesh, d, e, owner[static_cast<std::size_t>(d)][static_cast<std::size_t>(e)]);
  };

  // What owners send: their shared entities to the copies; and the closures
  // of the elements other ranks ghost: the element, its sides held here (of
  // given dimensions whole), the vertices of what goes whole.
  std::vector<std::tuple<int, int, LO>> shared, ents, sides, verts;  // (rank, dimension, lid)
  for (int d = 0; d <= top; ++d) {
    const SegmentTable& t = mesh.segment_table(d);
    if (!given(d) || t.owned_end == t.interior_end) continue;
    const auto rows = mesh.fields().get_var<int>(sharing_field, d);
    for (LO e = t.interior_end; e < t.owned_end; ++e) {
      for (const int r : rows.value_or_default(d, e)) shared.emplace_back(r, d, e);
    }
  }
  std::array<std::vector<std::pair<SideKey, LO>>, 4> by_key;
  for (int k = 1; k < top; ++k) {
    auto& v = by_key[static_cast<std::size_t>(k)];
    for (LO e = 0; e < mesh.segment_table(k).shared_end; ++e) v.emplace_back(key_of(mesh, k, e), e);
    std::sort(v.begin(), v.end());
  }
  const auto add_vertices = [&](int r, int d, LO e) {
    const auto off = mesh.conn_offsets(d);
    for (Offset k = off(e); k < off(e + 1); ++k) verts.emplace_back(r, 0, mesh.conn(d)(k));
    if (!mesh.has_nodes(d)) return;
    const auto noff = mesh.node_offsets(d);
    for (Offset k = noff(e); k < noff(e + 1); ++k) verts.emplace_back(r, 0, mesh.nodes(d)(k));
  };
  for (LO e = 0; e < mesh.segment_table(top).owned_end; ++e) {
    const std::span<const int> to = targets.of(e);
    if (to.empty()) continue;
    std::vector<std::pair<int, LO>> closure{{top, e}};
    for (int k = 1; k < top; ++k) {
      const auto& v = by_key[static_cast<std::size_t>(k)];
      if (v.empty()) continue;
      for_each_side(mesh, top, e, k, [&](const SideKey& key) {
        const auto it =
            std::lower_bound(v.begin(), v.end(), std::pair<SideKey, LO>{key, std::numeric_limits<LO>::min()});
        if (it != v.end() && it->first == key) closure.emplace_back(k, it->second);
      });
    }
    for (const int r : to) {
      for (const auto& [d, l] : closure) {
        if (d < top) sides.emplace_back(r, d, l);
        if (!given(d)) continue;
        ents.emplace_back(r, d, l);
        add_vertices(r, d, l);
      }
    }
  }
  for (auto* v : {&shared, &ents, &sides, &verts}) sort_unique(*v);
  std::int64_t widest = 0;
  for (const auto* v : {&shared, &ents}) {
    for (const auto& [r, d, e] : *v) {
      if (d == 0) continue;
      Offset n = mesh.conn_offsets(d)(e + 1) - mesh.conn_offsets(d)(e);
      if (mesh.has_nodes(d)) n += mesh.node_offsets(d)(e + 1) - mesh.node_offsets(d)(e);
      widest = std::max<std::int64_t>(widest, static_cast<std::int64_t>(n));
    }
  }
  widest = comm.allreduce(widest, Op::max);
  const std::size_t width = entity_header + static_cast<std::size_t>(widest) * sizeof(GO);

  // Rows of entities (d > 0) and of vertices to their ranks.
  const auto send = [&](const std::vector<std::tuple<int, int, LO>>& rows, bool vertices, auto&& version) {
    std::vector<std::size_t> which;
    std::vector<int> to;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if ((std::get<1>(rows[i]) == 0) != vertices) continue;
      which.push_back(i);
      to.push_back(std::get<0>(rows[i]));
    }
    return route(comm, to, vertices ? vertex_width : width, [&](std::size_t i, std::byte* row) {
      const auto [r, d, e] = rows[which[i]];
      const Arrived a = version(d, e);
      if (vertices) {
        put_vertex(row, a);
      } else {
        put_entity(row, a);
      }
    });
  };
  // The copies here take their owners' entities.
  std::array<std::vector<Arrived>, 4> theirs;
  {
    const Batch vin = send(shared, true, local), ein = send(shared, false, local);
    for (std::size_t i = 0; i < vin.size(); ++i) theirs[0].push_back(take_vertex(&vin.payload(i, 0)));
    for (std::size_t i = 0; i < ein.size(); ++i) {
      Arrived a = take_entity(&ein.payload(i, 0));
      theirs[static_cast<std::size_t>(dimension(a.topology))].push_back(std::move(a));
    }
    for (auto& v : theirs) by_gid(v);
  }
  const auto version = [&](int d, LO e) {
    const auto dd = static_cast<std::size_t>(d);
    const Arrived* a = e >= mesh.segment_table(d).owned_end ? find(theirs[dd], mesh.gid(d)(e)) : nullptr;
    return a != nullptr ? *a : local(d, e);
  };
  const Batch vin = send(verts, true, version), ein = send(ents, false, version);
  std::vector<int> to(sides.size());
  for (std::size_t i = 0; i < sides.size(); ++i) to[i] = std::get<0>(sides[i]);
  const Batch sin = route(comm, to, side_width, [&](std::size_t i, std::byte* row) {
    const auto [r, d, e] = sides[i];
    put(row, 0, static_cast<std::int32_t>(d));
    put(row, sizeof(std::int32_t),
        static_cast<std::int32_t>(owner[static_cast<std::size_t>(d)][static_cast<std::size_t>(e)]));
    put(row, 2 * sizeof(std::int32_t), mesh.gid(d)(e));
    put(row, 2 * sizeof(std::int32_t) + sizeof(GO), key_of(mesh, d, e));
  });

  // What arrived and is not held here: the ghosts, once each by gid.
  std::array<std::vector<std::pair<GO, LO>>, 4> held;
  for (int d = 0; d <= top; ++d) held[static_cast<std::size_t>(d)] = held_by_gid(mesh, d);
  const auto fresh = [&](int d, GO g) { return find(held[static_cast<std::size_t>(d)], g) < 0; };
  std::array<std::vector<Arrived>, 4> ghosts;
  for (std::size_t i = 0; i < ein.size(); ++i) {
    Arrived a = take_entity(&ein.payload(i, 0));
    const int d = dimension(a.topology);
    if (fresh(d, a.gid)) ghosts[static_cast<std::size_t>(d)].push_back(std::move(a));
  }
  for (std::size_t i = 0; i < vin.size(); ++i) {
    Arrived a = take_vertex(&vin.payload(i, 0));
    if (fresh(0, a.gid)) ghosts[0].push_back(std::move(a));
  }
  std::array<std::vector<std::pair<GO, int>>, 4> ghost_owner;
  for (std::size_t d = 0; d <= static_cast<std::size_t>(top); ++d) {
    by_gid(ghosts[d]);
    for (const Arrived& a : ghosts[d]) ghost_owner[d].emplace_back(a.gid, a.owner);
  }

  // The sides of the ghost elements as their owners hold them, by key.
  std::array<std::vector<KnownSide>, 4> known_sides;
  std::array<std::vector<std::pair<GO, int>>, 4> side_owner;
  for (std::size_t i = 0; i < sin.size(); ++i) {
    const std::byte* row = &sin.payload(i, 0);
    const auto k = static_cast<std::size_t>(take<std::int32_t>(row, 0));
    const KnownSide x{take<SideKey>(row, 2 * sizeof(std::int32_t) + sizeof(GO)),
                      take<GO>(row, 2 * sizeof(std::int32_t)), take<std::int32_t>(row, sizeof(std::int32_t))};
    known_sides[k].push_back(x);
    side_owner[k].emplace_back(x.gid, x.owner);
  }
  for (std::size_t k = 0; k < 4; ++k) {
    std::sort(known_sides[k].begin(), known_sides[k].end());
    known_sides[k].erase(std::unique(known_sides[k].begin(), known_sides[k].end(),
                                     [](const KnownSide& x, const KnownSide& y) { return x.key == y.key; }),
                         known_sides[k].end());
    std::sort(side_owner[k].begin(), side_owner[k].end());
  }

  // The ghosts as the builder orders the part built whole: each ownership
  // kind on its own, ghosts by owner, each entity by itself in the box above.
  // So it orders the ghosts alone, after the held vertices they use, which
  // stay in place with the rest of what is held.
  MeshBuilder b(top);
  {
    std::vector<GO> used;
    for (int d = 1; d <= top; ++d) {
      for (const Arrived& a : ghosts[static_cast<std::size_t>(d)]) {
        for (const GO g : a.nodes) {
          if (!fresh(0, g)) used.push_back(g);
        }
      }
    }
    std::sort(used.begin(), used.end());
    used.erase(std::unique(used.begin(), used.end()), used.end());
    std::vector<GO> g;
    std::vector<Real> c;
    for (const GO u : used) {
      const Arrived a = version(0, find(held[0], u));
      g.push_back(u);
      c.insert(c.end(), a.xyz.begin(), a.xyz.end());
    }
    for (const Arrived& a : ghosts[0]) {
      g.push_back(a.gid);
      c.insert(c.end(), a.xyz.begin(), a.xyz.end());
    }
    b.add_vertices(g, c);
  }
  for (int d = 1; d <= top; ++d) {
    if (!given(d)) continue;
    std::vector<Arrived> all = ghosts[static_cast<std::size_t>(d)];
    std::stable_sort(all.begin(), all.end(), [](const Arrived& x, const Arrived& y) {
      return std::tuple(x.topology, x.corners, x.extra) < std::tuple(y.topology, y.corners, y.extra);
    });
    for (std::size_t i = 0; i < all.size();) {
      std::size_t j = i;
      std::vector<GO> gids, nodes;
      std::vector<Offset> offs{0};
      for (; j < all.size() && all[j].topology == all[i].topology &&
             (all[i].topology == Topology::polygon ||
              (all[j].corners == all[i].corners && all[j].extra == all[i].extra));
           ++j) {
        gids.push_back(all[j].gid);
        nodes.insert(nodes.end(), all[j].nodes.begin(), all[j].nodes.end());
        offs.push_back(static_cast<Offset>(nodes.size()));
      }
      if (all[i].topology == Topology::polygon) {
        b.add_polygons(gids, offs, nodes);
      } else if (all[i].extra == 0) {
        b.add_elements(all[i].topology, gids, nodes);
      } else {
        b.add_elements(all[i].topology, gids, nodes, all[i].corners + all[i].extra);
      }
      i = j;
    }
  }
  const auto owner_of_ghost = [&](std::size_t d, GO g) {
    const auto& v = given(static_cast<int>(d)) ? ghost_owner[d] : side_owner[d];
    const auto it = std::lower_bound(v.begin(), v.end(), std::pair<GO, int>{g, std::numeric_limits<int>::min()});
    return it == v.end() || it->first != g ? -1 : it->second;
  };
  for (int d = 0; d <= top; ++d) {
    if (!given(d)) continue;
    const auto dd = static_cast<std::size_t>(d);
    const std::vector<GO> gids =
        d == 0 ? std::vector<GO>(b.vertex_gids().begin(), b.vertex_gids().end()) : b.element_gids(d);
    Sharing s;
    s.offsets.assign(gids.size() + 1, 0);
    s.ghost_owner.resize(gids.size());
    for (std::size_t i = 0; i < gids.size(); ++i) s.ghost_owner[i] = owner_of_ghost(dd, gids[i]);
    std::vector<Ownership> kinds(gids.size());
    for (std::size_t i = 0; i < gids.size(); ++i) kinds[i] = s.kind(i, me);
    b.set_ownership(d, gids, kinds, s.groups());
  }
  if (std::isfinite(f[7])) b.set_key_box({-f[4], -f[5], -f[6]}, {f[7], f[8], f[9]});

  // The part: what is held, then the ghosts; each given dimension's sharing
  // as held, the ghosts' owners as sent. A failure is reported on every rank.
  HostMesh part;
  std::exception_ptr failure;
  try {
    part = splice(mesh, std::move(b).build(mesh.ordering()), given, derived, theirs, held, by_key, known_sides);
    for (int d = 0; d <= top; ++d) {
      if (!given(d) && !derived(d)) continue;
      const auto dd = static_cast<std::size_t>(d);
      const SegmentTable& t = mesh.segment_table(d);
      std::vector<Offset> offsets{0};
      std::vector<int> ranks;
      if (t.shared_end > t.interior_end) {
        const auto rows = mesh.fields().get_var<int>(sharing_field, d);
        for (LO e = t.interior_end; e < t.shared_end; ++e) {
          const auto r = rows.value_or_default(d, e);
          ranks.insert(ranks.end(), r.begin(), r.end());
          offsets.push_back(static_cast<Offset>(ranks.size()));
        }
      }
      std::vector<int> owners(owner[dd].begin() + t.owned_end, owner[dd].end());
      for (LO e = t.shared_end; e < part.count(d); ++e) owners.push_back(owner_of_ghost(dd, part.gid(d)(e)));
      add_sharing_fields(part, d, offsets, ranks, owners);
    }
  } catch (...) {
    failure = std::current_exception();
  }
  if (comm.allreduce(std::int32_t{failure ? 1 : 0}, Op::max) != 0) {
    if (failure) std::rethrow_exception(failure);
    throw std::runtime_error("add_ghosts: failed on another rank");
  }

  // The graphs held, built again. The derived dimensions are on the part
  // whole; a given one may gain derived sides, distributed with the gids and
  // sharing held here or sent.
  const AdjacencyRequest graphs = graphs_to_rebuild(comm, mesh);
  if (!graphs.empty()) {
    std::array<KnownEntities, 4> known;
    for (int k = 1; k < top; ++k) {
      if (!given(k)) continue;
      KnownEntities& x = known[static_cast<std::size_t>(k)];
      const SegmentTable& t = mesh.segment_table(k);
      std::optional<VarField<int, Kokkos::HostSpace>> rows;
      if (t.shared_end > t.interior_end) rows = mesh.fields().get_var<int>(sharing_field, k);
      for (const auto& [g, e] : held[static_cast<std::size_t>(k)]) {
        x.by_key.emplace_back(key_of(mesh, k, e), g);
        x.held.push_back(g);
        if (e >= t.interior_end) {
          const auto r = rows->value_or_default(k, e);
          x.ranks.insert(x.ranks.end(), r.begin(), r.end());
        }
        x.offsets.push_back(x.ranks.size());
      }
    }
    for (std::size_t i = 0; i < sin.size(); ++i) {
      const std::byte* row = &sin.payload(i, 0);
      const int k = take<std::int32_t>(row, 0);
      const GO g = take<GO>(row, 2 * sizeof(std::int32_t));
      if (!given(k) || !fresh(k, g)) continue;
      KnownEntities& x = known[static_cast<std::size_t>(k)];
      x.by_key.emplace_back(take<SideKey>(row, 2 * sizeof(std::int32_t) + sizeof(GO)), g);
      x.ghosts.emplace_back(g, take<std::int32_t>(row, sizeof(std::int32_t)));
    }
    for (KnownEntities& x : known) {
      std::sort(x.by_key.begin(), x.by_key.end());
      x.by_key.erase(std::unique(x.by_key.begin(), x.by_key.end()), x.by_key.end());
      std::sort(x.ghosts.begin(), x.ghosts.end());
      x.ghosts.erase(std::unique(x.ghosts.begin(), x.ghosts.end()), x.ghosts.end());
    }
    build_known_adjacencies(comm, part, graphs, known);
  }
  move_entity_data(comm, mesh, part, options.sets, true);
  return part;
}

// What add_ghosts does once the targets are known: the part built again from
// the closures placed through directories (`rebuild`), or by append_ghosts;
// then everything held keeps its id.
void place_ghosts(const Comm& comm, HostMesh& mesh, const GhostTargets& targets, const GhostOptions& options,
                  bool rebuild) {
  const int top = mesh.dim(), me = comm.rank();
  HostMesh part;
  if (rebuild) {
    // Each owned element stays here, on the ranks sharing it and on those
    // ghosting it.
    const SegmentTable& t = mesh.segment_table(top);
    std::optional<VarField<int, Kokkos::HostSpace>> sharing;
    if (t.owned_end > t.interior_end) sharing = mesh.fields().get_var<int>(sharing_field, top);
    std::vector<std::size_t> offsets{0};
    std::vector<int> ranks;
    for (LO e = 0; e < mesh.count(top); ++e) {
      if (e < t.owned_end) {
        std::vector<int> r{me};
        if (e >= t.interior_end) {
          const auto s = sharing->value_or_default(top, e);
          r.insert(r.end(), s.begin(), s.end());
        }
        const auto g = targets.of(e);
        r.insert(r.end(), g.begin(), g.end());
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        ranks.insert(ranks.end(), r.begin(), r.end());
      }
      offsets.push_back(ranks.size());
    }
    const Placement where = place_closure(comm, mesh, offsets, ranks);
    const GhostMarks marks{options.sets};
    part = build_placed(comm, mesh, where, mesh.ordering(), &marks);
  } else {
    part = append_ghosts(comm, mesh, targets, options);
  }
  for (int d = 0; d <= top; ++d) keep_ids(mesh, part, d);
  mesh = std::move(part);
}

// Checks, finds the targets and, unless nothing changes, places the ghosts.
void ghost(const Comm& comm, HostMesh& mesh, const GhostOptions& options, bool rebuild) {
  check_movable(comm, mesh, "add_ghosts", false, "");
  const GhostTargets targets = ghost_targets(comm, mesh, options);
  // Nothing to place when no rank holds copies or ghosts and none gains a
  // ghost (one rank, say): the rebuild would give the same mesh.
  std::int32_t change = targets.ranks.empty() ? 0 : 1;
  for (int d = 0; d <= mesh.dim(); ++d) change = mesh.segment_table(d).owned_end < mesh.count(d) ? 1 : change;
  if (comm.allreduce(change, Op::max) == 0) return;
  place_ghosts(comm, mesh, targets, options, rebuild);
}

}  // namespace

void add_ghosts(const Comm& comm, HostMesh& mesh, const GhostOptions& options) { ghost(comm, mesh, options, false); }

void add_ghosts_by_rebuild(const Comm& comm, HostMesh& mesh, const GhostOptions& options) {
  ghost(comm, mesh, options, true);
}

void remove_ghosts(const Comm& comm, HostMesh& mesh) { add_ghosts(comm, mesh, GhostOptions{0, 0, false}); }

}  // namespace soma::detail
