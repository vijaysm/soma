// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/fields/domain_mesh.hpp"

namespace soma {

namespace {

using Host = Kokkos::HostSpace;

[[noreturn]] void reject(const std::string& what) { throw std::invalid_argument("permute_entities: " + what); }

template <class T>
View1<T, Host> uninit(const std::string& label, std::size_t n) {
  return View1<T, Host>(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), n);
}

// One value per entity, in the new order; empty stays empty.
template <class T>
View1<T, Host> moved(const View1<T, Host>& old, const std::vector<LO>& old_of, const std::string& label) {
  if (old.extent(0) == 0) return old;
  View1<T, Host> out = uninit<T>(label, old_of.size());
  for (std::size_t i = 0; i < old_of.size(); ++i) out(i) = old(static_cast<std::size_t>(old_of[i]));
  return out;
}

// CSR rows in the new order; empty offsets stay empty.
void moved_rows(const View1<Offset, Host>& offsets, const View1<LO, Host>& values, const std::vector<LO>& old_of,
                View1<Offset, Host>& out_offsets, View1<LO, Host>& out_values, const std::string& label) {
  if (offsets.extent(0) == 0) {
    out_offsets = offsets;
    out_values = values;
    return;
  }
  out_offsets = uninit<Offset>(label + "_offsets", old_of.size() + 1);
  out_values = uninit<LO>(label, values.extent(0));
  out_offsets(0) = 0;
  for (std::size_t i = 0; i < old_of.size(); ++i) {
    const auto o = static_cast<std::size_t>(old_of[i]);
    const Offset b = offsets(o);
    const Offset e = offsets(o + 1);
    std::copy(values.data() + b, values.data() + e, out_values.data() + out_offsets(i));
    out_offsets(i + 1) = out_offsets(i) + (e - b);
  }
}

// Every value v becomes new_of_old[v]; empty stays empty.
View1<LO, Host> renamed(const View1<LO, Host>& old, const std::vector<LO>& new_of_old, const std::string& label) {
  if (old.extent(0) == 0) return old;
  View1<LO, Host> out = uninit<LO>(label, old.extent(0));
  for (std::size_t i = 0; i < old.extent(0); ++i) out(i) = new_of_old[static_cast<std::size_t>(old(i))];
  return out;
}

bool same(const SegmentTable& a, const SegmentTable& b) {
  return a.segments == b.segments && a.count == b.count && a.interior_end == b.interior_end &&
         a.owned_end == b.owned_end && a.shared_end == b.shared_end;
}

// The table must cover the entities with segments in order, and be what
// make_segments gives for the ownership it names and the reordered entities'
// topology and arity. Vertices must keep each kind's corners first (`corner_ends`
// are the old ones), and the table split at the new corner ends, which it
// returns (the old ones for other dimensions).
std::array<LO, 4> check_segments(const EntityBlock<Host>& b, int dim, const std::vector<LO>& old_of,
                                 const SegmentTable& t, const std::array<LO, 4>& corner_ends) {
  const auto n = old_of.size();
  if (t.count != static_cast<LO>(n)) {
    reject("the segment table counts " + std::to_string(t.count) + " entities, not " + std::to_string(n));
  }
  std::vector<Ownership> own(n);
  LO at = 0;
  for (const Segment& s : t.segments) {
    if (s.begin != at || s.end <= s.begin || s.end > static_cast<LO>(n)) {
      reject("segment [" + std::to_string(s.begin) + ", " + std::to_string(s.end) + ") does not follow at " +
             std::to_string(at));
    }
    std::fill(own.begin() + s.begin, own.begin() + s.end, s.ownership);
    at = s.end;
  }
  if (at != static_cast<LO>(n)) reject("the segments end at " + std::to_string(at) + ", not " + std::to_string(n));
  std::vector<Topology> topo(n, Topology::vertex);
  std::vector<std::uint16_t> arity(n, 1);
  if (dim > 0) {
    for (std::size_t i = 0; i < n; ++i) {
      const auto o = static_cast<std::size_t>(old_of[i]);
      topo[i] = static_cast<Topology>(b.topology(o));
      arity[i] = static_cast<std::uint16_t>(b.conn_offsets(o + 1) - b.conn_offsets(o));
    }
  }
  SegmentTable want;
  try {
    want = make_segments(own, topo, arity);
  } catch (const std::invalid_argument& e) {
    reject(std::string("the reordered entities do not follow the segments (") + e.what() + ")");
  }
  std::array<LO, 4> ends = corner_ends;
  if (dim == 0) {
    const std::array<LO, 5> was = kind_bounds(b.segments);
    const auto corner = [&](LO o) {
      std::size_t k = 0;
      while (o >= was[k + 1]) ++k;
      return o < corner_ends[k];
    };
    const std::array<LO, 5> bounds = kind_bounds(want);
    for (std::size_t k = 0; k < 4; ++k) {
      ends[k] = bounds[k];
      bool nodes = false;
      for (LO i = bounds[k]; i < bounds[k + 1]; ++i) {
        const LO o = old_of[static_cast<std::size_t>(i)];
        if (!corner(o)) {
          nodes = true;
        } else if (nodes) {
          reject("corner vertex " + std::to_string(o) + " goes to " + std::to_string(i) +
                 ", after a node-only vertex of its kind");
        } else {
          ++ends[k];
        }
      }
    }
    split_segments(want, ends);
  }
  if (!same(want, t)) reject("the segment table does not describe the reordered entities");
  return ends;
}

// A held graph with its rows and targets in the new order.
Adjacency<Host> permuted_graph(const Adjacency<Host>& g, bool rows, bool targets, bool sorted_rows,
                               const std::vector<LO>& old_of, const std::vector<LO>& new_of_old) {
  if (g.offsets.extent(0) == 0) return g;
  const bool coded = g.codes.extent(0) > 0;
  const std::size_t nrows = g.offsets.extent(0) - 1;
  Adjacency<Host> out;
  out.offsets = uninit<Offset>("soma::adjacency::offsets", nrows + 1);
  out.targets = uninit<LO>("soma::adjacency::targets", g.targets.extent(0));
  if (coded) out.codes = uninit<std::uint8_t>("soma::adjacency::codes", g.codes.extent(0));
  out.offsets(0) = 0;
  std::vector<std::pair<LO, std::uint8_t>> row;
  for (std::size_t r = 0; r < nrows; ++r) {
    const auto o = rows ? static_cast<std::size_t>(old_of[r]) : r;
    row.clear();
    for (auto k = static_cast<std::size_t>(g.offsets(o)); k < static_cast<std::size_t>(g.offsets(o + 1)); ++k) {
      const LO t = g.targets(k);
      row.emplace_back(targets ? new_of_old[static_cast<std::size_t>(t)] : t, coded ? g.codes(k) : std::uint8_t{0});
    }
    if (targets && sorted_rows) std::sort(row.begin(), row.end());
    auto at = static_cast<std::size_t>(out.offsets(r));
    for (const auto& [t, c] : row) {
      out.targets(at) = t;
      if (coded) out.codes(at) = c;
      ++at;
    }
    out.offsets(r + 1) = static_cast<Offset>(at);
  }
  return out;
}

}  // namespace

void permute_entities(HostMesh& mesh, int dim, std::span<const LO> new_of_old, SegmentTable segments,
                      std::span<const GO> gids) {
  if (dim < 0 || dim > mesh.dim()) {
    reject("dimension " + std::to_string(dim) + " is not in [0, " + std::to_string(mesh.dim()) + "]");
  }
  if (mesh.edited()) throw std::logic_error("permute_entities: the mesh holds edits; compress() it first");
  const auto d = static_cast<std::size_t>(dim);
  const EntityBlock<Host>& old = mesh.ents_[d];
  const auto n = static_cast<std::size_t>(old.count);
  if (new_of_old.size() != n) {
    reject("new_of_old holds " + std::to_string(new_of_old.size()) + " ids for " + std::to_string(n) + " entities");
  }
  std::vector<LO> old_of(n, invalid_lo);
  for (std::size_t i = 0; i < n; ++i) {
    const LO l = new_of_old[i];
    if (l < 0 || static_cast<std::size_t>(l) >= n) {
      reject("entity " + std::to_string(i) + " goes to " + std::to_string(l) + ", not in [0, " + std::to_string(n) + ")");
    }
    if (old_of[static_cast<std::size_t>(l)] != invalid_lo) {
      reject("entities " + std::to_string(old_of[static_cast<std::size_t>(l)]) + " and " + std::to_string(i) +
             " both go to " + std::to_string(l));
    }
    old_of[static_cast<std::size_t>(l)] = static_cast<LO>(i);
  }
  const std::array<LO, 4> corner_ends = check_segments(old, dim, old_of, segments, mesh.corner_ends_);
  View1<GO, Host> given;
  if (!gids.empty()) {
    if (gids.size() != n) reject(std::to_string(gids.size()) + " gids for " + std::to_string(n) + " entities");
    std::vector<GO> sorted(gids.begin(), gids.end());
    std::sort(sorted.begin(), sorted.end());
    const auto dup = std::adjacent_find(sorted.begin(), sorted.end());
    if (dup != sorted.end()) reject("gid " + std::to_string(*dup) + " is given twice");
    given = uninit<GO>("soma::mesh::gid", n);
    std::copy(gids.begin(), gids.end(), given.data());
  }
  const std::vector<LO> map(new_of_old.begin(), new_of_old.end());

  // The reordered dimension.
  std::array<EntityBlock<Host>, 4> blocks = mesh.ents_;
  EntityBlock<Host>& b = blocks[d];
  b.gid = moved(gids.empty() ? old.gid : given, old_of, "soma::mesh::gid");
  b.topology = moved(old.topology, old_of, "soma::mesh::topology");
  moved_rows(old.conn_offsets, old.conn, old_of, b.conn_offsets, b.conn, "soma::mesh::conn");
  moved_rows(old.node_offsets, old.nodes, old_of, b.node_offsets, b.nodes, "soma::mesh::nodes");
  b.segments = std::move(segments);
  CoordView<Host> coords = mesh.coords_;
  if (dim == 0) {
    // Vertex lids: coordinates move, every connectivity and node list is renamed.
    coords = CoordView<Host>(Kokkos::view_alloc(std::string("soma::mesh::coords"), Kokkos::WithoutInitializing),
                             n);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t q = 0; q < 3; ++q) coords(i, q) = mesh.coords_(static_cast<std::size_t>(old_of[i]), q);
    }
    for (std::size_t k = 1; k <= 3; ++k) {
      blocks[k].conn = renamed(blocks[k].conn, map, "soma::mesh::conn");
      blocks[k].nodes = renamed(blocks[k].nodes, map, "soma::mesh::nodes");
    }
  }

  HostMesh next;
  next.dim_ = mesh.dim_;
  next.corner_ends_ = corner_ends;
  next.ordering_ = mesh.ordering_;
  next.ghosts_ = mesh.ghosts_;
  next.ents_ = std::move(blocks);
  next.coords_ = coords;
  next.values_ = mesh.values_;
  next.epoch_ = mesh.epoch_ + 1;
  for (int s = 0; s < adjacency_slots; ++s) {
    if (!mesh.adj_.has(s)) continue;
    const bool pair = s < 16;
    const int from = pair ? s / 4 : (s - 16) / 4;
    const int to = pair ? s % 4 : from;
    const Adjacency<Host>& g = mesh.adj_.graphs[static_cast<std::size_t>(s)];
    if (from != dim && to != dim) {
      next.adj_.put(s, g);
      continue;
    }
    // Upward and bridge rows are sorted; downward rows follow the side order.
    next.adj_.put(s, permuted_graph(g, from == dim, to == dim, !pair || from < to, old_of, map));
  }
  next.sets_ = mesh.sets_.count() > 0 && mesh.sets_.has_members(dim)
                   ? mesh.sets_.renumbered(dim, map, old.count)
                   : mesh.sets_.template mirror<Host>({true, true, true, true}, true);
  next.attach_fields();
  std::array<std::optional<FieldRegistry<Host>::SlotRemap>, num_field_slots> maps;
  maps[d] = FieldRegistry<Host>::SlotRemap{map, old.count};
  mesh.fields_.remap_into(next.fields_, maps, next.epoch_, element_info(next));
  mesh.swap(next);
}

}  // namespace soma
