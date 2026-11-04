// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/core/mesh_edit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "soma/core/mesh_builder.hpp"
#include "soma/runtime/options.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/topology/build.hpp"
#include "soma/topology/nodes.hpp"
#include "soma/topology/sides.hpp"

namespace soma {

namespace {

using Host = Kokkos::HostSpace;

[[noreturn]] void reject(const std::string& what) { throw std::invalid_argument("MeshEdit: " + what); }

template <class T>
View1<T, Host> grown(const View1<T, Host>& old, std::size_t keep, std::size_t n, const std::string& label) {
  View1<T, Host> out(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), n);
  std::copy(old.data(), old.data() + keep, out.data());
  return out;
}

// A side by its vertex count and its vertices sorted.
std::array<LO, 5> side_key(const topology::Side& s) {
  std::array<LO, 5> x{static_cast<LO>(s.n), -1, -1, -1, -1};
  std::copy(s.v, s.v + s.n, x.begin() + 1);
  std::sort(x.begin() + 1, x.begin() + 1 + s.n);
  return x;
}

// The sides of dimension k of every live entity above k, sorted.
std::vector<std::array<LO, 5>> side_keys(const HostMesh& mesh, int k) {
  const auto m = mesh.view();
  std::vector<std::array<LO, 5>> out;
  for (int d = k + 1; d <= mesh.dim(); ++d) {
    for (LO e = 0; e < mesh.count(d); ++e) {
      if (!m.alive(d, e)) continue;
      const int n = topology::side_count(m, d, e, k);
      for (int s = 0; s < n; ++s) out.push_back(side_key(topology::side_of(m, d, e, k, s)));
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// Each live entity's lid in `to`, found by its gid; invalid_lo for removed
// entities and gids `to` lacks.
std::array<std::vector<LO>, 4> new_lids(const HostMesh& from, const HostMesh& to) {
  std::array<std::vector<LO>, 4> out;
  const auto v = from.view();
  for (int d = 0; d <= 3; ++d) {
    auto& map = out[static_cast<std::size_t>(d)];
    map.assign(static_cast<std::size_t>(from.count(d)), invalid_lo);
    const auto to_gid = to.gid(d);
    const auto n = static_cast<std::size_t>(to.count(d));
    GO lo = 0, hi = -1;
    for (std::size_t l = 0; l < n; ++l) {
      lo = l == 0 ? to_gid(l) : std::min(lo, to_gid(l));
      hi = l == 0 ? to_gid(l) : std::max(hi, to_gid(l));
    }
    // Gids spanning at most a few times the count: a table indexed by gid.
    if (n > 0 && hi - lo < 4 * static_cast<GO>(n) + 1024) {
      std::vector<LO> lid_of(static_cast<std::size_t>(hi - lo + 1), invalid_lo);
      for (std::size_t l = 0; l < n; ++l) lid_of[static_cast<std::size_t>(to_gid(l) - lo)] = static_cast<LO>(l);
      for (LO l = 0; l < from.count(d); ++l) {
        const GO g = v.gid(d, l);
        if (v.alive(d, l) && g >= lo && g <= hi) map[static_cast<std::size_t>(l)] = lid_of[static_cast<std::size_t>(g - lo)];
      }
      continue;
    }
    std::vector<std::pair<GO, LO>> by_gid(n);
    for (std::size_t l = 0; l < n; ++l) by_gid[l] = {to_gid(l), static_cast<LO>(l)};
    std::sort(by_gid.begin(), by_gid.end());
    for (LO l = 0; l < from.count(d); ++l) {
      if (!v.alive(d, l)) continue;
      const GO g = v.gid(d, l);
      const auto it = std::lower_bound(by_gid.begin(), by_gid.end(), std::pair<GO, LO>{g, 0});
      if (it != by_gid.end() && it->first == g) map[static_cast<std::size_t>(l)] = it->second;
    }
  }
  return out;
}

}  // namespace

void MeshEdit::check_vertex(LO v) const {
  if (v < 0 || v >= next_lid(0)) {
    reject("vertex " + std::to_string(v) + " is not in [0, " + std::to_string(next_lid(0)) + ")");
  }
}

LO MeshEdit::add_vertices(std::span<const Real> xyz) {
  if (xyz.size() % 3 != 0) reject("add_vertices: xyz must hold 3 values per vertex");
  const std::size_t n = xyz.size() / 3;
  const LO first = add_vertices(std::vector<GO>(n), xyz);
  std::fill(added_[0].given.end() - static_cast<std::ptrdiff_t>(n), added_[0].given.end(), 0);
  return first;
}

LO MeshEdit::add_vertices(std::span<const GO> gids, std::span<const Real> xyz) {
  if (xyz.size() != 3 * gids.size()) reject("add_vertices: xyz must hold 3 values per vertex");
  for (std::size_t i = 0; i < xyz.size(); ++i) {
    if (!std::isfinite(xyz[i])) reject("add_vertices: vertex " + std::to_string(i / 3) + " has a non-finite coordinate");
  }
  const LO first = next_lid(0);
  Added& a = added_[0];
  a.xyz.insert(a.xyz.end(), xyz.begin(), xyz.end());
  a.gids.insert(a.gids.end(), gids.begin(), gids.end());
  a.given.insert(a.given.end(), gids.size(), 1);
  a.count += static_cast<LO>(gids.size());
  return first;
}

LO MeshEdit::add_elements(Topology t, std::span<const LO> node_lids, int nodes_per_element) {
  const int n = nodes_per_element == 0 ? num_vertices(t) : nodes_per_element;
  if (n <= 0 || node_lids.size() % static_cast<std::size_t>(n) != 0) {
    reject("add_elements: the node list does not hold whole elements");
  }
  const LO first = add_elements(t, std::vector<GO>(node_lids.size() / static_cast<std::size_t>(n)), node_lids, n);
  Added& a = added_[static_cast<std::size_t>(dimension(t))];
  std::fill(a.given.end() - static_cast<std::ptrdiff_t>(node_lids.size() / static_cast<std::size_t>(n)), a.given.end(), 0);
  return first;
}

LO MeshEdit::add_elements(Topology t, std::span<const GO> gids, std::span<const LO> node_lids,
                          int nodes_per_element) {
  if (t == Topology::polygon) reject("use add_polygons for polygons");
  if (t == Topology::polyhedron || t == Topology::vertex) reject(std::string(name(t)) + " elements cannot be added");
  const int d = dimension(t);
  const int n = nodes_per_element == 0 ? num_vertices(t) : nodes_per_element;
  if (mid_node_bits(t, n) < 0) reject("no " + std::string(name(t)) + " element has " + std::to_string(n) + " nodes");
  if (node_lids.size() != gids.size() * static_cast<std::size_t>(n)) {
    reject("add_elements: the node list does not match the elements");
  }
  for (const LO v : node_lids) check_vertex(v);
  const int corners = num_vertices(t);
  const LO first = next_lid(d);
  Added& a = added_[static_cast<std::size_t>(d)];
  for (std::size_t e = 0; e < gids.size(); ++e) {
    const auto at = node_lids.begin() + static_cast<std::ptrdiff_t>(e * static_cast<std::size_t>(n));
    a.topology.push_back(static_cast<std::uint8_t>(t));
    a.conn.insert(a.conn.end(), at, at + corners);
    a.offsets.push_back(static_cast<Offset>(a.conn.size()));
    a.nodes.insert(a.nodes.end(), at + corners, at + n);
    a.node_offsets.push_back(static_cast<Offset>(a.nodes.size()));
  }
  a.gids.insert(a.gids.end(), gids.begin(), gids.end());
  a.given.insert(a.given.end(), gids.size(), 1);
  a.count += static_cast<LO>(gids.size());
  return first;
}

LO MeshEdit::add_polygons(std::span<const Offset> offsets, std::span<const LO> vertex_lids) {
  if (offsets.empty() || offsets[0] != 0 || static_cast<std::size_t>(offsets.back()) != vertex_lids.size()) {
    reject("add_polygons: offsets must start at 0 and end at the number of vertices");
  }
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    if (offsets[i + 1] - offsets[i] < 3 || offsets[i + 1] - offsets[i] > std::numeric_limits<std::uint16_t>::max()) {
      reject("add_polygons: polygon " + std::to_string(i) + " has " + std::to_string(offsets[i + 1] - offsets[i]) +
             " vertices");
    }
  }
  for (const LO v : vertex_lids) check_vertex(v);
  const LO first = next_lid(2);
  Added& a = added_[2];
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    a.topology.push_back(static_cast<std::uint8_t>(Topology::polygon));
    a.conn.insert(a.conn.end(), vertex_lids.begin() + offsets[i], vertex_lids.begin() + offsets[i + 1]);
    a.offsets.push_back(static_cast<Offset>(a.conn.size()));
    a.node_offsets.push_back(static_cast<Offset>(a.nodes.size()));
    a.gids.push_back(0);
    a.given.push_back(0);
    ++a.count;
  }
  return first;
}

void MeshEdit::remove(int d, std::span<const LO> lids) {
  if (d < 0 || d > mesh_->dim()) reject("remove: dimension " + std::to_string(d) + " is not in [0, mesh dimension]");
  const auto alive = mesh_->alive(d);
  for (const LO l : lids) {
    if (l < 0 || l >= mesh_->count(d)) {
      reject("remove: entity " + std::to_string(l) + " of dimension " + std::to_string(d) + " is not in [0, " +
             std::to_string(mesh_->count(d)) + ")");
    }
    if (alive.extent(0) > 0 && alive(static_cast<std::size_t>(l)) == 0) {
      reject("remove: entity " + std::to_string(l) + " of dimension " + std::to_string(d) + " is already removed");
    }
  }
  auto& r = removed_[static_cast<std::size_t>(d)];
  r.insert(r.end(), lids.begin(), lids.end());
}

EditReport MeshEdit::apply() {
  HostMesh& m = *mesh_;
  // Elements above the mesh dimension raise it.
  int D = m.dim();
  for (int d = D + 1; d <= 3; ++d) {
    if (added_[static_cast<std::size_t>(d)].count > 0) D = d;
  }
  EditReport report;

  // Removed per dimension, as a mask over the existing entities.
  std::array<std::vector<std::uint8_t>, 4> gone;
  for (int d = 0; d <= 3; ++d) {
    auto& r = removed_[static_cast<std::size_t>(d)];
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    report.removed[static_cast<std::size_t>(d)] = static_cast<LO>(r.size());
    report.added[static_cast<std::size_t>(d)] = added_[static_cast<std::size_t>(d)].count;
    if (r.empty()) continue;
    gone[static_cast<std::size_t>(d)].assign(static_cast<std::size_t>(m.count(d)), 0);
    for (const LO l : r) gone[static_cast<std::size_t>(d)][static_cast<std::size_t>(l)] = 1;
  }
  const auto live = [&](int d, LO e) {
    const auto alive = m.alive(d);
    if (alive.extent(0) > 0 && alive(static_cast<std::size_t>(e)) == 0) return false;
    const auto& g = gone[static_cast<std::size_t>(d)];
    return g.empty() || g[static_cast<std::size_t>(e)] == 0;
  };

  // Vertices in use stay: by live existing elements and by added ones.
  if (!removed_[0].empty() || m.alive(0).extent(0) > 0) {
    const auto v = m.view();
    const auto vertex_gid = [&](LO l) { return std::to_string(m.gid(0)(static_cast<std::size_t>(l))); };
    for (int d = 1; d <= D; ++d) {
      for (LO e = 0; e < m.count(d); ++e) {
        if (!live(d, e)) continue;
        for (LO k = 0; k < v.num_nodes(d, e); ++k) {
          const LO x = v.node(d, e, k);
          if (!live(0, x)) {
            reject("vertex gid " + vertex_gid(x) + " cannot be removed: element gid " +
                   std::to_string(m.gid(d)(static_cast<std::size_t>(e))) + " of dimension " + std::to_string(d) +
                   " uses it");
          }
        }
      }
      const Added& a = added_[static_cast<std::size_t>(d)];
      for (const auto* list : {&a.conn, &a.nodes}) {
        for (const LO x : *list) {
          if (x < m.count(0) && !live(0, x)) reject("an added element uses vertex gid " + vertex_gid(x) + ", which is removed");
        }
      }
    }
  }

  // Gids: given ones new to their dimension, the others after the largest.
  std::array<std::vector<GO>, 4> gids;
  for (int d = 0; d <= 3; ++d) {
    const Added& a = added_[static_cast<std::size_t>(d)];
    if (a.count == 0) continue;
    std::vector<GO> taken(m.gid(d).data(), m.gid(d).data() + m.count(d));
    GO next = 0;
    for (const GO g : taken) next = std::max(next, g + 1);
    for (std::size_t i = 0; i < a.gids.size(); ++i) {
      if (a.given[i] != 0) next = std::max(next, a.gids[i] + 1);
    }
    auto& out = gids[static_cast<std::size_t>(d)];
    for (std::size_t i = 0; i < a.gids.size(); ++i) {
      out.push_back(a.given[i] != 0 ? a.gids[i] : next++);
      if (a.given[i] != 0) taken.push_back(a.gids[i]);
    }
    std::sort(taken.begin(), taken.end());
    const auto dup = std::adjacent_find(taken.begin(), taken.end());
    if (dup != taken.end()) reject("gid " + std::to_string(*dup) + " is already used in dimension " + std::to_string(d));
  }

  // The edited blocks, built aside.
  std::array<EntityBlock<Host>, 4> blocks;
  std::array<bool, 4> changed{};
  CoordView<Host> coords = m.coords_;
  for (int d = 0; d <= 3; ++d) {
    const auto k = static_cast<std::size_t>(d);
    const EntityBlock<Host>& old = m.ents_[k];
    EntityBlock<Host> b = old;
    const Added& a = added_[k];
    changed[k] = a.count > 0 || !removed_[k].empty();
    if (!changed[k]) {
      blocks[k] = std::move(b);
      continue;
    }
    const auto n0 = static_cast<std::size_t>(old.count);
    const std::size_t n = n0 + static_cast<std::size_t>(a.count);
    if (n > static_cast<std::size_t>(std::numeric_limits<LO>::max())) reject("too many entities in dimension " + std::to_string(d));
    b.count = static_cast<LO>(n);
    if (a.count > 0) {
      b.gid = grown(old.gid, n0, n, "soma::mesh::gid");
      std::copy(gids[k].begin(), gids[k].end(), b.gid.data() + n0);
      b.tail = old.tail + a.count;
      b.input = true;
      if (d == 0) {
        coords = CoordView<Host>(Kokkos::view_alloc(std::string("soma::mesh::coords"), Kokkos::WithoutInitializing), n);
        for (std::size_t i = 0; i < n0; ++i) {
          for (std::size_t q = 0; q < 3; ++q) coords(i, q) = m.coords_(i, q);
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(a.count); ++i) {
          for (std::size_t q = 0; q < 3; ++q) coords(n0 + i, q) = a.xyz[3 * i + q];
        }
      } else {
        const Offset base = n0 > 0 ? old.conn_offsets(n0) : 0;
        b.conn_offsets = grown(old.conn_offsets, n0 > 0 ? n0 + 1 : 0, n + 1, "soma::mesh::offsets");
        b.conn_offsets(0) = 0;
        for (std::size_t i = 1; i < a.offsets.size(); ++i) b.conn_offsets(n0 + i) = base + a.offsets[i];
        b.conn = grown(old.conn, static_cast<std::size_t>(base), static_cast<std::size_t>(base) + a.conn.size(), "soma::mesh::conn");
        std::copy(a.conn.begin(), a.conn.end(), b.conn.data() + base);
        b.topology = grown(old.topology, n0, n, "soma::mesh::topology");
        std::copy(a.topology.begin(), a.topology.end(), b.topology.data() + n0);
        if (!a.nodes.empty() || old.node_offsets.extent(0) > 0) {
          const Offset nbase = old.node_offsets.extent(0) > 0 ? old.node_offsets(n0) : 0;
          View1<Offset, Host> no(Kokkos::view_alloc(std::string("soma::mesh::node_offsets"), Kokkos::WithoutInitializing), n + 1);
          for (std::size_t i = 0; i <= n0; ++i) no(i) = old.node_offsets.extent(0) > 0 ? old.node_offsets(i) : 0;
          for (std::size_t i = 1; i < a.node_offsets.size(); ++i) no(n0 + i) = nbase + a.node_offsets[i];
          b.node_offsets = no;
          b.nodes = grown(old.nodes, static_cast<std::size_t>(nbase), static_cast<std::size_t>(nbase) + a.nodes.size(),
                          "soma::mesh::nodes");
          std::copy(a.nodes.begin(), a.nodes.end(), b.nodes.data() + nbase);
        }
      }
      // The tail's segments: one per run of (topology, arity) in added order.
      for (std::size_t i = 0; i < static_cast<std::size_t>(a.count); ++i) {
        const Topology t = d == 0 ? Topology::vertex : static_cast<Topology>(a.topology[i]);
        const auto arity = static_cast<std::uint16_t>(d == 0 ? 1 : a.offsets[i + 1] - a.offsets[i]);
        const auto at = static_cast<LO>(n0 + i);
        auto& segs = b.segments.segments;
        if (i == 0 || segs.back().topology != t || segs.back().arity != arity) {
          segs.push_back(Segment{at, at, t, arity, Ownership::interior});
        }
        segs.back().end = at + 1;
      }
      b.segments.count = b.count;
      b.segments.interior_end = b.count;
      b.segments.owned_end = b.count;
      b.segments.shared_end = b.count;
    }
    if (!removed_[k].empty() || old.alive.extent(0) > 0) {
      View1<std::uint8_t, Host> alive(Kokkos::view_alloc(std::string("soma::mesh::alive"), Kokkos::WithoutInitializing), n);
      for (std::size_t i = 0; i < n; ++i) alive(i) = i < n0 ? (live(d, static_cast<LO>(i)) ? 1 : 0) : 1;
      b.alive = alive;
    }
    blocks[k] = std::move(b);
  }
  // Sides are no longer known to be complete once elements or sides change.
  for (int k = 1; k < D; ++k) {
    if (changed[static_cast<std::size_t>(D)] || changed[static_cast<std::size_t>(k)]) {
      blocks[static_cast<std::size_t>(k)].derived = false;
    }
  }

  HostMesh next;
  next.dim_ = D;
  next.corners_ = m.corners_;
  next.ordering_ = m.ordering_;
  next.epoch_ = m.epoch_;
  next.ents_ = std::move(blocks);
  next.coords_ = coords;
  next.values_ = m.values_;
  for (int s = 0; s < adjacency_slots; ++s) {
    if (!m.adj_.has(s)) continue;
    bool touched = false;
    for (int d = 0; d <= 3; ++d) touched = touched || (changed[static_cast<std::size_t>(d)] && topology::detail::touches(s, d));
    if (!touched) next.adj_.put(s, m.adj_.graphs[static_cast<std::size_t>(s)]);
  }
  // Sets: removed members leave; slots follow the new counts.
  std::optional<EntitySets<Host>> sets;
  for (int d = 0; d <= 3; ++d) {
    if (!changed[static_cast<std::size_t>(d)] || m.sets_.count() == 0 || !m.sets_.has_members(d)) continue;
    std::vector<LO> map(static_cast<std::size_t>(m.count(d)));
    for (LO l = 0; l < m.count(d); ++l) {
      const auto& g = gone[static_cast<std::size_t>(d)];
      map[static_cast<std::size_t>(l)] = g.empty() || g[static_cast<std::size_t>(l)] == 0 ? l : invalid_lo;
    }
    sets = (sets ? *sets : m.sets_).renumbered(d, map, next.count(d), true);
  }
  next.sets_ = sets ? std::move(*sets) : m.sets_.template mirror<Host>({true, true, true, true}, true);
  next.attach_fields();
  // Fields: same lids, more entities in the edited dimensions.
  std::array<std::optional<FieldRegistry<Host>::SlotRemap>, num_field_slots> maps;
  for (int d = 0; d <= 3; ++d) {
    if (!changed[static_cast<std::size_t>(d)]) continue;
    FieldRegistry<Host>::SlotRemap r;
    r.new_lid.resize(static_cast<std::size_t>(m.count(d)));
    for (LO l = 0; l < m.count(d); ++l) r.new_lid[static_cast<std::size_t>(l)] = l;
    r.count = next.count(d);
    maps[static_cast<std::size_t>(d)] = std::move(r);
  }
  m.fields_.remap_into(next.fields_, maps, m.epoch_, element_info(next));
  // Graphs touching edited dimensions, built again over the live entities.
  const AdjacencyRequest dropped = AdjacencyRequest::from_bits(m.adjacencies().bits() & ~next.adjacencies().bits());
  if (!dropped.empty()) build_adjacencies(next, dropped);

  // Compressed before the swap, so a throw leaves the mesh as it was.
  const double t = threshold_.value_or(compress_threshold());
  if (t > 0 && removed_fraction(next) >= t) report.compressed = next.compress();

  m.swap(next);
  added_ = {};
  removed_ = {};
  return report;
}

void MeshEdit::set_compress_threshold(double t) {
  if (!(t >= 0.0 && t <= 1.0)) throw std::invalid_argument("MeshEdit::set_compress_threshold: " + std::to_string(t) + " is not in [0, 1]");
  threshold_ = t;
}

double removed_fraction(const HostMesh& mesh) {
  double worst = 0;
  for (int d = 0; d <= 3; ++d) {
    const LO n = mesh.count(d);
    if (n == 0) continue;
    const double f = static_cast<double>(n - mesh.num_live(d)) / static_cast<double>(n);
    worst = f > worst ? f : worst;
  }
  return worst;
}

namespace detail {

namespace {

// The mesh MeshBuilder builds from the live entities, in the mesh's ordering,
// with its input and completeness marks.
HostMesh live_mesh(const HostMesh& mesh) {
  const int D = mesh.dim();
  const auto v = mesh.view();
  MeshBuilder b(D);
  {
    std::vector<GO> gids;
    std::vector<Real> xyz;
    for (LO i = 0; i < mesh.count(0); ++i) {
      if (!v.alive(0, i)) continue;
      gids.push_back(v.gid(0, i));
      for (int q = 0; q < 3; ++q) xyz.push_back(v.x(i, q));
    }
    b.add_vertices(gids, xyz);
  }
  for (int d = 1; d <= 3; ++d) {
    // Sides derived alone that no live element has any more go with it; a
    // side given as input stays, and derivation rejects it if it is no side.
    using Access = TopologyAccess;
    const bool prune = d < D && mesh.count(d) > 0 && !Access::block(mesh, d).input && !Access::block(mesh, d).derived;
    const std::vector<std::array<LO, 5>> sides = prune ? side_keys(mesh, d) : std::vector<std::array<LO, 5>>{};
    // Live entities by (topology, node count), each group in lid order.
    std::map<std::pair<std::uint8_t, LO>, std::vector<LO>> groups;
    for (LO e = 0; e < mesh.count(d); ++e) {
      if (!v.alive(d, e)) continue;
      if (prune && !std::binary_search(sides.begin(), sides.end(), side_key(topology::entity_as_side(v, d, e)))) continue;
      const Topology t = v.topology(d, e);
      groups[{static_cast<std::uint8_t>(t), t == Topology::polygon ? 0 : v.num_nodes(d, e)}].push_back(e);
    }
    for (const auto& [key, ents] : groups) {
      const auto t = static_cast<Topology>(key.first);
      std::vector<GO> gids;
      std::vector<GO> nodes;
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
  HostMesh next = std::move(b).build(mesh.ordering());
  for (int d = 0; d <= 3; ++d) {
    using Access = TopologyAccess;
    Access::block(next, d).input = Access::block(mesh, d).input;
    Access::block(next, d).derived = Access::block(mesh, d).derived && next.count(d) > 0;
  }
  return next;
}

}  // namespace

Renumbering compress(HostMesh& mesh) {
  Renumbering out;
  if (!mesh.edited()) return out;
  HostMesh next = live_mesh(mesh);
  // Held graphs again; sides that lost their completeness are derived anew
  // (keeping the gids of the explicit ones) before anything else holds lids.
  const AdjacencyRequest held = mesh.adjacencies();
  if (!held.empty()) build_adjacencies(next, held);

  // Old lid to new lid through the gids (derivation may have renumbered sides).
  out.new_lid = new_lids(mesh, next);
  std::array<std::optional<FieldRegistry<Host>::SlotRemap>, num_field_slots> maps;
  for (int d = 0; d <= 3; ++d) {
    maps[static_cast<std::size_t>(d)] = FieldRegistry<Host>::SlotRemap{out.new_lid[static_cast<std::size_t>(d)], next.count(d)};
  }
  std::optional<EntitySets<Host>> sets;
  for (int d = 0; d <= 3; ++d) {
    if (mesh.sets_.count() == 0 || !mesh.sets_.has_members(d)) continue;
    sets = (sets ? *sets : mesh.sets_).renumbered(d, out.new_lid[static_cast<std::size_t>(d)], next.count(d), true);
  }
  next.sets_ = sets ? std::move(*sets) : mesh.sets_.template mirror<Host>({true, true, true, true}, true);
  next.values_ = mesh.values_;
  next.epoch_ = mesh.epoch_ + 1;
  next.attach_fields();
  mesh.fields_.remap_into(next.fields_, maps, next.epoch_, element_info(next));
  mesh.swap(next);
  return out;
}

void build_edited(HostMesh& mesh, AdjacencyRequest req) {
  const int D = mesh.dim();
  std::uint32_t bits = 0;
  for (int s = 0; s < adjacency_slots; ++s) {
    const bool conn = s < 16 && s % 4 == 0;
    if (req.has_slot(s) && !conn && !mesh.adj_.has(s)) bits |= std::uint32_t{1} << s;
  }
  if (bits == 0) return;
  // Built over the live entities, then carried back to the edited lids.
  HostMesh live = live_mesh(mesh);
  build_adjacencies(live, AdjacencyRequest::from_bits(bits));
  // After the build: derivation renumbers the sides it derives again.
  const std::array<std::vector<LO>, 4> fwd = new_lids(mesh, live);
  std::array<bool, 4> rederived{};
  for (int k = 1; k < D; ++k) {
    const auto i = static_cast<std::size_t>(k);
    rederived[i] = !mesh.ents_[i].derived && live.ents_[i].derived;
  }
  std::array<std::vector<LO>, 4> back;  // live lid -> edited lid
  for (int d = 0; d <= 3; ++d) {
    const auto i = static_cast<std::size_t>(d);
    if (rederived[i]) continue;
    back[i].assign(static_cast<std::size_t>(live.count(d)), invalid_lo);
    for (std::size_t l = 0; l < fwd[i].size(); ++l) {
      if (fwd[i][l] != invalid_lo) back[i][static_cast<std::size_t>(fwd[i][l])] = static_cast<LO>(l);
    }
  }
  const auto source = [&](int d, LO e) { return rederived[static_cast<std::size_t>(d)] ? e : fwd[static_cast<std::size_t>(d)][static_cast<std::size_t>(e)]; };
  const auto target = [&](int d, LO t) { return rederived[static_cast<std::size_t>(d)] ? t : back[static_cast<std::size_t>(d)][static_cast<std::size_t>(t)]; };

  std::array<Adjacency<Host>, adjacency_slots> made{};
  for (int s = 0; s < adjacency_slots; ++s) {
    if ((bits >> s & 1U) == 0) continue;
    const bool pair = s < 16;
    const int from = pair ? s / 4 : (s - 16) / 4;
    const int to = pair ? s % 4 : from;
    const bool sorted_rows = !pair || from < to;
    const Adjacency<Host>& g = live.adj_.graphs[static_cast<std::size_t>(s)];
    const bool coded = g.codes.extent(0) > 0;
    const LO rows = rederived[static_cast<std::size_t>(from)] ? live.count(from) : mesh.count(from);
    Adjacency<Host> out;
    out.offsets = View1<Offset, Host>("soma::adjacency::offsets", static_cast<std::size_t>(rows) + 1);
    std::vector<LO> targets;
    std::vector<std::uint8_t> codes;
    std::vector<std::pair<LO, std::uint8_t>> row;
    for (LO e = 0; e < rows; ++e) {
      const LO le = source(from, e);
      row.clear();
      if (le != invalid_lo) {
        for (Offset k = g.offsets(static_cast<std::size_t>(le)); k < g.offsets(static_cast<std::size_t>(le) + 1); ++k) {
          row.emplace_back(target(to, g.targets(static_cast<std::size_t>(k))), coded ? g.codes(static_cast<std::size_t>(k)) : 0);
        }
      }
      if (sorted_rows) std::sort(row.begin(), row.end());
      for (const auto& [t, c] : row) {
        targets.push_back(t);
        codes.push_back(c);
      }
      out.offsets(static_cast<std::size_t>(e) + 1) = static_cast<Offset>(targets.size());
    }
    out.targets = View1<LO, Host>(Kokkos::view_alloc(std::string("soma::adjacency::targets"), Kokkos::WithoutInitializing), targets.size());
    std::copy(targets.begin(), targets.end(), out.targets.data());
    if (coded) {
      out.codes = View1<std::uint8_t, Host>(Kokkos::view_alloc(std::string("soma::adjacency::codes"), Kokkos::WithoutInitializing),
                                            codes.size());
      std::copy(codes.begin(), codes.end(), out.codes.data());
    }
    made[static_cast<std::size_t>(s)] = std::move(out);
  }
  // Sets follow re-derived sides through their gids.
  std::optional<EntitySets<Host>> sets;
  for (int k = 1; k < D; ++k) {
    const auto i = static_cast<std::size_t>(k);
    if (!rederived[i] || mesh.sets_.count() == 0 || !mesh.sets_.has_members(k)) continue;
    sets = (sets ? *sets : mesh.sets_).renumbered(k, fwd[i], live.count(k), true);
  }

  // Fields on re-derived sides follow them through their gids; the epoch stays.
  FieldRegistry<Host> fields;
  {
    std::array<std::optional<FieldRegistry<Host>::SlotRemap>, num_field_slots> maps;
    ElementInfo<Host> info;
    for (int d = 0; d <= 3; ++d) {
      const auto i = static_cast<std::size_t>(d);
      const EntityBlock<Host>& b = rederived[i] ? live.ents_[i] : mesh.ents_[i];
      fields.set_entities(d, b.count, b.gid);
      if (d > 0) {
        info.topology[i] = b.topology;
        info.conn_offsets[i] = b.conn_offsets;
      }
      if (rederived[i]) maps[i] = FieldRegistry<Host>::SlotRemap{fwd[i], live.count(d)};
    }
    fields.set_entities(set_dim, sets ? sets->count() : mesh.sets_.count(), sets ? sets->gid() : mesh.sets_.gid());
    mesh.fields_.remap_into(fields, maps, mesh.epoch_, info);
  }

  // Install; nothing below throws.
  if (sets) mesh.sets_ = std::move(*sets);
  mesh.fields_ = std::move(fields);
  for (int k = 1; k < D; ++k) {
    const auto i = static_cast<std::size_t>(k);
    if (!rederived[i]) continue;
    const bool input = mesh.ents_[i].input;
    mesh.ents_[i] = std::move(live.ents_[i]);
    mesh.ents_[i].input = input;
    for (int s = 0; s < adjacency_slots; ++s) {
      if (topology::detail::touches(s, k)) mesh.adj_.drop(s);
    }
  }
  for (int s = 0; s < adjacency_slots; ++s) {
    if ((bits >> s & 1U) != 0) mesh.adj_.put(s, std::move(made[static_cast<std::size_t>(s)]));
  }
  mesh.attach_fields();
}

}  // namespace detail

}  // namespace soma
