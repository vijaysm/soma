// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/ghosts.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"

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
  // Elements owned here by their bridge sides; elements held here (not as ghosts).
  const LO owned = mesh.segment_table(top).owned_end, held = mesh.segment_table(top).shared_end;
  std::vector<std::pair<SideKey, LO>> index;
  for (LO e = 0; e < owned; ++e) for_each_side(mesh, top, e, b, [&](const SideKey& k) { index.emplace_back(k, e); });
  std::sort(index.begin(), index.end());
  const auto egid = mesh.gid(top);
  std::vector<GO> known;
  std::vector<Element> frontier;
  std::int64_t widest = 0;
  for (LO e = 0; e < held; ++e) {
    known.push_back(egid(e));
    frontier.push_back({static_cast<Topology>(mesh.topology(top)(e)), corner_gids(mesh, top, e)});
    if (e < owned) widest = std::max<std::int64_t>(widest, static_cast<std::int64_t>(frontier.back().corners.size()));
  }
  std::sort(known.begin(), known.end());
  widest = comm.allreduce(widest, Op::max);
  const std::size_t width = sizeof(GO) + 2 * sizeof(std::int32_t) + static_cast<std::size_t>(widest) * sizeof(GO);

  std::vector<std::pair<GO, int>> taken;  // (ghost element, its owner)
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
      for (std::size_t q = 0; q < c.size(); ++q) put(row, sizeof(GO) + 2 * sizeof(std::int32_t) + q * sizeof(GO), c[q]);
    });
    // The holders of the answers' vertices: this rank and its sharing.
    std::vector<std::tuple<int, GO, int>> told;  // (asker, vertex, holder)
    for (const auto& [asker, e] : answers) {
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
    const Batch heard = route(comm, to, sizeof(GO) + sizeof(int), [&](std::size_t i, std::byte* row) {
      put(row, 0, std::get<1>(told[i]));
      put(row, sizeof(GO), std::get<2>(told[i]));
    });

    // The next layer: answers not held here yet.
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
      for (std::size_t q = 0; q < n; ++q) x.corners.push_back(take<GO>(row, sizeof(GO) + 2 * sizeof(std::int32_t) + q * sizeof(GO)));
      fresh.push_back(g);
      known.insert(std::upper_bound(known.begin(), known.end(), g), g);
      taken.emplace_back(g, got.source(i));
      frontier.push_back(std::move(x));
    }
  }

  // Each owner learns who ghosts its elements.
  std::vector<int> to(taken.size());
  for (std::size_t i = 0; i < taken.size(); ++i) to[i] = taken[i].second;
  const Batch asked = route(comm, to, sizeof(GO), [&](std::size_t i, std::byte* row) { put(row, 0, taken[i].first); });
  std::vector<std::pair<GO, LO>> lid(static_cast<std::size_t>(owned));
  for (LO e = 0; e < owned; ++e) lid[static_cast<std::size_t>(e)] = {egid(e), e};
  std::sort(lid.begin(), lid.end());
  std::vector<std::pair<LO, int>> rows;
  for (std::size_t i = 0; i < asked.size(); ++i) {
    const GO g = take<GO>(&asked.payload(i, 0), 0);
    rows.emplace_back(std::lower_bound(lid.begin(), lid.end(), std::pair<GO, LO>{g, std::numeric_limits<LO>::min()})->second,
                      asked.source(i));
  }
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

}  // namespace

void add_ghosts(const Comm& comm, HostMesh& mesh, const GhostOptions& options) {
  check_movable(comm, mesh, "add_ghosts", false, "");
  const GhostTargets targets = ghost_targets(comm, mesh, options);
  const int top = mesh.dim(), me = comm.rank();
  // Nothing to place when no rank holds copies or ghosts and none gains a
  // ghost (one rank, say): the rebuild would give the same mesh.
  std::int32_t change = targets.ranks.empty() ? 0 : 1;
  for (int d = 0; d <= top; ++d) change = mesh.segment_table(d).owned_end < mesh.count(d) ? 1 : change;
  if (comm.allreduce(change, Op::max) == 0) return;
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
  HostMesh part = build_placed(comm, mesh, where, mesh.ordering(), &marks);
  for (int d = 0; d <= top; ++d) keep_ids(mesh, part, d);
  mesh = std::move(part);
}

void remove_ghosts(const Comm& comm, HostMesh& mesh) { add_ghosts(comm, mesh, GhostOptions{0, 0, false}); }

}  // namespace soma::detail
