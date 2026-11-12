// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_edit.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/testing/box_mesh.hpp"
#include "soma/topology/build.hpp"

using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Ownership;
using soma::Real;
using soma::SegmentTable;
using soma::Topology;

namespace {

using Host = Kokkos::HostSpace;
using Values = std::map<GO, std::vector<double>>;

// What a mesh says about each entity, by gid: what must survive a reordering.
struct Snapshot {
  std::array<std::map<GO, std::pair<int, std::vector<GO>>>, 4> entities;  // topology, node gids
  std::map<GO, std::array<Real, 3>> coords;
  std::uint32_t graphs_held = 0;
  // Per held slot, each row by gids with its codes: downward rows in stored
  // order, the others sorted.
  std::map<int, std::map<GO, std::vector<std::pair<GO, int>>>> graphs;
  std::vector<std::array<std::vector<GO>, 4>> set_members;
  std::vector<std::vector<std::pair<int, GO>>> sequences;
  std::map<std::string, Values> fields;
  friend bool operator==(const Snapshot&, const Snapshot&) = default;
};

GO gid(const HostMesh& m, int d, LO l) { return m.gid(d)(static_cast<std::size_t>(l)); }

Values fixed_by_gid(const HostMesh& m, const std::string& name, int d) {
  const auto f = m.fields().get<double>(name, d);
  Values out;
  for (LO e = 0; e < m.count(d); ++e) {
    for (int c = 0; c < f.ncomp(); ++c) out[gid(m, d, e)].push_back(f.value_or_default(d, e, c));
  }
  return out;
}

Values var_by_gid(const HostMesh& m, const std::string& name, int d) {
  const auto f = m.fields().get_var<double>(name, d);
  Values out;
  for (LO e = 0; e < m.count(d); ++e) {
    const auto v = f.value_or_default(d, e);
    out[gid(m, d, e)] = std::vector<double>(v.begin(), v.end());
  }
  return out;
}

Snapshot snapshot(const HostMesh& m) {
  Snapshot s;
  const auto v = m.view();
  for (int d = 0; d <= m.dim(); ++d) {
    for (LO e = 0; e < m.count(d); ++e) {
      auto& rec = s.entities[static_cast<std::size_t>(d)][gid(m, d, e)];
      if (d == 0) continue;
      rec.first = static_cast<int>(v.topology(d, e));
      for (LO k = 0; k < v.num_nodes(d, e); ++k) rec.second.push_back(gid(m, 0, v.node(d, e, k)));
    }
  }
  for (LO i = 0; i < m.count(0); ++i) {
    s.coords[gid(m, 0, i)] = {m.coords()(static_cast<std::size_t>(i), 0), m.coords()(static_cast<std::size_t>(i), 1),
                              m.coords()(static_cast<std::size_t>(i), 2)};
  }
  s.graphs_held = m.adjacencies().bits();
  for (int slot = 0; slot < soma::adjacency_slots; ++slot) {
    if (!m.adjacencies().has_slot(slot)) continue;
    const bool pair = slot < 16;
    const int from = pair ? slot / 4 : (slot - 16) / 4;
    const int to = pair ? slot % 4 : from;
    const auto g = pair ? m.adjacency(from, to) : m.bridge(from, (slot - 16) % 4);
    auto& rows = s.graphs[slot];
    for (LO e = 0; e < m.count(from); ++e) {
      auto& row = rows[gid(m, from, e)];
      for (auto k = g.offsets(static_cast<std::size_t>(e)); k < g.offsets(static_cast<std::size_t>(e) + 1); ++k) {
        const auto at = static_cast<std::size_t>(k);
        row.emplace_back(gid(m, to, g.targets(at)), g.codes.extent(0) > 0 ? g.codes(at) : -1);
      }
      if (!pair || from < to) std::sort(row.begin(), row.end());
    }
  }
  const auto& sets = m.sets();
  for (LO i = 0; i < sets.count(); ++i) {
    auto& mine = s.set_members.emplace_back();
    for (int d = 0; d <= 3; ++d) {
      if (!sets.has_members(d)) continue;
      for (const LO l : sets.members_of(i, d)) mine[static_cast<std::size_t>(d)].push_back(gid(m, d, l));
      std::sort(mine[static_cast<std::size_t>(d)].begin(), mine[static_cast<std::size_t>(d)].end());
    }
    auto& seq = s.sequences.emplace_back();
    if (!sets.ordered(i)) continue;
    for (const soma::SetEntry& x : sets.sequence(i)) {
      seq.emplace_back(x.slot, x.slot == soma::set_dim ? GO{x.lid} : gid(m, x.slot, x.lid));
    }
  }
  using Named = std::pair<const char*, int>;
  for (const auto& [name, d] : {Named{"vertex", 0}, Named{"edge", 1}, Named{"face", 2}, Named{"cell", 3}}) {
    if (d <= m.dim() && m.fields().contains(name)) s.fields[name] = fixed_by_gid(m, name, d);
    const std::string var = std::string(name) + "_var";
    if (d <= m.dim() && m.fields().contains(var)) s.fields[var] = var_by_gid(m, var, d);
  }
  return s;
}

// Upward and bridge rows sorted by (target, code), downward rows of the
// held graphs exactly as a fresh build gives them.
void expect_graphs_canonical(const HostMesh& m) {
  soma::TransferOptions o = soma::TransferOptions::everything();
  o.adjacencies = m.adjacencies();
  o.adj_source = soma::TransferOptions::AdjSource::rebuild;
  const HostMesh fresh = m.mirror<Host>(o);
  for (int slot = 0; slot < soma::adjacency_slots; ++slot) {
    if (!m.adjacencies().has_slot(slot)) continue;
    const bool pair = slot < 16;
    const int a = pair ? slot / 4 : (slot - 16) / 4;
    const int b = pair ? slot % 4 : (slot - 16) % 4;
    const auto g = pair ? m.adjacency(a, b) : m.bridge(a, b);
    const auto f = pair ? fresh.adjacency(a, b) : fresh.bridge(a, b);
    ASSERT_EQ(g.offsets.extent(0), f.offsets.extent(0)) << "slot " << slot;
    ASSERT_EQ(g.targets.extent(0), f.targets.extent(0)) << "slot " << slot;
    ASSERT_EQ(g.codes.extent(0), f.codes.extent(0)) << "slot " << slot;
    for (std::size_t i = 0; i < g.offsets.extent(0); ++i) EXPECT_EQ(g.offsets(i), f.offsets(i)) << "slot " << slot;
    for (std::size_t i = 0; i < g.targets.extent(0); ++i) EXPECT_EQ(g.targets(i), f.targets(i)) << "slot " << slot;
    for (std::size_t i = 0; i < g.codes.extent(0); ++i) EXPECT_EQ(g.codes(i), f.codes(i)) << "slot " << slot;
  }
}

std::vector<LO> shuffled(LO n, unsigned seed) {
  std::vector<LO> p(static_cast<std::size_t>(n));
  std::iota(p.begin(), p.end(), 0);
  std::mt19937 rng(seed);
  std::shuffle(p.begin(), p.end(), rng);
  return p;
}

// The table of n entities of one topology and arity, ownership by the given
// counts of each kind.
SegmentTable uniform(Topology t, std::uint16_t arity, std::array<LO, 4> kinds) {
  std::vector<Ownership> own;
  for (std::size_t k = 0; k < 4; ++k) own.insert(own.end(), static_cast<std::size_t>(kinds[k]), static_cast<Ownership>(k));
  const std::vector<Topology> topo(own.size(), t);
  const std::vector<std::uint16_t> ar(own.size(), arity);
  return soma::make_segments(own, topo, ar);
}

void expect_table(const HostMesh& m, int d, const SegmentTable& t) {
  const SegmentTable& s = m.segment_table(d);
  EXPECT_EQ(s.segments, t.segments);
  EXPECT_EQ(s.count, t.count);
  EXPECT_EQ(s.interior_end, t.interior_end);
  EXPECT_EQ(s.owned_end, t.owned_end);
  EXPECT_EQ(s.shared_end, t.shared_end);
}

// Fixed and variable-length fields over a dimension, valued by gid.
void add_fields(HostMesh& m, const std::string& name, int d) {
  auto f = m.fields().add<double>(name, soma::make_space(m, soma::Domain<Host>::all(m, d), soma::Layout::per_entity()), 2);
  for (LO e = 0; e < m.count(d); ++e) {
    f.view()(static_cast<std::size_t>(e), 0) = static_cast<double>(gid(m, d, e));
    f.view()(static_cast<std::size_t>(e), 1) = 0.5 * static_cast<double>(e);
  }
  // Every other entity, holding e % 3 values.
  std::vector<LO> members;
  for (LO e = 0; e < m.count(d); e += 2) members.push_back(e);
  std::vector<soma::Offset> offsets{0};
  std::vector<double> values;
  for (const LO e : members) {
    for (LO k = 0; k < e % 3; ++k) values.push_back(static_cast<double>(100 * gid(m, d, e) + k));
    offsets.push_back(static_cast<soma::Offset>(values.size()));
  }
  m.fields().add_var<double>(name + "_var",
                             soma::make_space(m, soma::Domain<Host>::from_list(m, d, members), soma::Layout::per_entity()),
                             offsets, values);
}

// Two sets: one of every other entity of each given dimension, one ordered
// with repeats.
void add_sets(HostMesh& m, const std::vector<int>& dims) {
  soma::SetSpec a{1, soma::set_unique, {}, {}, {}};
  soma::SetSpec b{2, soma::set_ordered, {}, {}, {}};
  for (const int d : dims) {
    for (LO e = 0; e < m.count(d); e += 2) a.contents.push_back({d, e});
    const LO n = m.count(d);
    b.contents.push_back({d, n - 1});
    b.contents.push_back({d, 0});
    b.contents.push_back({d, n - 1});
  }
  soma::SetBuilder sb;
  sb.add(std::move(a));
  sb.add(std::move(b));
  std::move(sb).install(m);
}

// Permutes one dimension and checks that the mesh says the same by gid.
void permute_and_check(HostMesh& m, int d, const std::vector<LO>& p, const SegmentTable& t) {
  const Snapshot before = snapshot(m);
  const auto epoch = m.epoch();
  soma::permute_entities(m, d, p, t);
  EXPECT_EQ(m.epoch(), epoch + 1);
  expect_table(m, d, t);
  const Snapshot after = snapshot(m);
  EXPECT_EQ(after.entities, before.entities);
  EXPECT_EQ(after.coords, before.coords);
  EXPECT_EQ(after.graphs_held, before.graphs_held);
  EXPECT_EQ(after.graphs, before.graphs);
  EXPECT_EQ(after.set_members, before.set_members);
  EXPECT_EQ(after.sequences, before.sequences);
  EXPECT_EQ(after.fields, before.fields);
  EXPECT_FALSE(before.fields.empty());
  expect_graphs_canonical(m);
}

soma::AdjacencyRequest planar_request() {
  return soma::AdjacencyRequest{}.add(2, 1).add(1, 2).add(0, 1).add(0, 2).add_bridge(2, 0).add_bridge(2, 1).add_bridge(
      0, 1).add_bridge(1, 0);
}

soma::AdjacencyRequest solid_request() {
  soma::AdjacencyRequest r;
  for (int a = 0; a <= 3; ++a) {
    for (int b = 0; b <= 3; ++b) {
      if (a == b) continue;
      if (b != 0) r.add(a, b);
      r.add_bridge(a, b);
    }
  }
  return r;
}

// Vertices of the mixed planar mesh, relabelled with every ownership: the
// connectivity, coordinates, graphs, sets and fields follow.
TEST(PermuteEntities, VerticesOfAPlanarMesh) {
  HostMesh m = soma_test::two_quads_and_a_triangle();
  soma::build_adjacencies(m, planar_request());
  add_fields(m, "vertex", 0);
  add_fields(m, "edge", 1);
  add_sets(m, {0, 1, 2});
  permute_and_check(m, 0, shuffled(m.count(0), 1), uniform(Topology::vertex, 1, {3, 2, 1, 1}));
  EXPECT_EQ(m.segment_table(0).interior_end, 3);
  EXPECT_EQ(m.segment_table(0).owned_end, 5);
  EXPECT_EQ(m.segment_table(0).shared_end, 6);
}

// Derived edges and the elements of the mixed mesh (triangle first, as the
// table requires).
TEST(PermuteEntities, EdgesAndElementsOfAPlanarMesh) {
  HostMesh m = soma_test::two_quads_and_a_triangle();
  soma::build_adjacencies(m, planar_request());
  ASSERT_EQ(m.count(1), 9);
  add_fields(m, "edge", 1);
  add_fields(m, "face", 2);
  add_sets(m, {1, 2});
  permute_and_check(m, 1, shuffled(9, 2), uniform(Topology::edge, 2, {4, 3, 0, 2}));
  // The triangle sorts first; the quads trade places, one of them shared.
  ASSERT_EQ(m.view().topology(2, 0), Topology::tri);
  const GO quad = gid(m, 2, 1);
  const std::vector<Ownership> own{Ownership::interior, Ownership::interior, Ownership::shared};
  const std::vector<Topology> topo{Topology::tri, Topology::quad, Topology::quad};
  const std::vector<std::uint16_t> arity{3, 4, 4};
  permute_and_check(m, 2, std::vector<LO>{0, 2, 1}, soma::make_segments(own, topo, arity));
  EXPECT_EQ(gid(m, 2, 2), quad);
  EXPECT_EQ(m.segment_table(2).owned_end, 3);
  EXPECT_EQ(m.segment_table(2).interior_end, 2);
}

// A hex box with every graph: each dimension in turn, derived faces and edges
// included; the down graphs from the hexes keep their side order and codes.
TEST(PermuteEntities, EveryDimensionOfAHexBox) {
  HostMesh m = soma::testing::box_mesh({2, 2, 2, Topology::hex});
  soma::build_adjacencies(m, solid_request());
  add_fields(m, "vertex", 0);
  add_fields(m, "edge", 1);
  add_fields(m, "face", 2);
  add_fields(m, "cell", 3);
  add_sets(m, {0, 1, 2, 3});
  const LO nv = m.count(0);
  const LO ne = m.count(1);
  const LO nf = m.count(2);
  ASSERT_EQ(nv, 27);
  ASSERT_EQ(ne, 54);
  ASSERT_EQ(nf, 36);
  permute_and_check(m, 0, shuffled(nv, 3), uniform(Topology::vertex, 1, {10, 7, 6, 4}));
  permute_and_check(m, 2, shuffled(nf, 4), uniform(Topology::quad, 4, {20, 6, 6, 4}));
  permute_and_check(m, 1, shuffled(ne, 5), uniform(Topology::edge, 2, {30, 10, 10, 4}));
  permute_and_check(m, 3, shuffled(8, 6), uniform(Topology::hex, 8, {2, 2, 2, 2}));
  // Building again changes nothing: the derived dimensions stay complete.
  const Snapshot s = snapshot(m);
  soma::build_adjacencies(m, solid_request());
  EXPECT_EQ(snapshot(m), s);
}

// The vertices of a hex box laid out in gid order again: the identity of gids.
TEST(PermuteEntities, InverseRestoresTheOrder) {
  HostMesh m = soma::testing::box_mesh({2, 1, 1, Topology::hex});
  soma::build_adjacencies(m, soma::AdjacencyRequest{}.add(3, 2).add(0, 3));
  add_fields(m, "vertex", 0);
  const std::vector<LO> p = shuffled(m.count(0), 7);
  const SegmentTable t = uniform(Topology::vertex, 1, {m.count(0), 0, 0, 0});
  const std::vector<GO> gids0(m.gid(0).data(), m.gid(0).data() + m.count(0));
  soma::permute_entities(m, 0, p, t);
  std::vector<LO> inv(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) inv[static_cast<std::size_t>(p[i])] = static_cast<LO>(i);
  permute_and_check(m, 0, inv, t);
  EXPECT_EQ(std::vector<GO>(m.gid(0).data(), m.gid(0).data() + m.count(0)), gids0);
}

// High-order nodes are vertex lids: they are renamed with the corners, and
// node-only vertices stay after the corners.
TEST(PermuteEntities, HighOrderNodesFollowTheVertices) {
  soma::MeshBuilder b(2);
  const std::vector<GO> gids{10, 20, 30, 40, 50, 60, 15, 35, 45, 25, 55};
  const std::vector<Real> xyz{0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0,
                              0.5, 0, 0, 1, 0.5, 0, 0.5, 1, 0, 0, 0.5, 0, 0.5, 0.5, 0};
  b.add_vertices(gids, xyz);
  b.add_elements(Topology::quad, std::vector<GO>{1}, std::vector<GO>{10, 20, 50, 40, 15, 35, 45, 25, 55}, 9);
  b.add_elements(Topology::quad, std::vector<GO>{2}, std::vector<GO>{20, 30, 60, 50}, 4);
  HostMesh m = std::move(b).build(soma::Ordering::file);
  ASSERT_EQ(m.corners(), 6);
  ASSERT_TRUE(m.has_nodes(2));
  soma::build_adjacencies(m, soma::AdjacencyRequest{}.add(2, 1).add(0, 2));
  add_fields(m, "vertex", 0);
  const std::vector<LO> p{4, 0, 5, 2, 1, 3, 9, 6, 10, 8, 7};
  SegmentTable t = uniform(Topology::vertex, 1, {11, 0, 0, 0});
  EXPECT_THROW(soma::permute_entities(m, 0, p, t), std::invalid_argument);  // not split at the corners
  soma::split_segments(t, std::vector<LO>{6});
  permute_and_check(m, 0, p, t);
  EXPECT_EQ(m.corners(), 6);
  // Corners and nodes may not trade places.
  const std::vector<LO> across{6, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10};
  EXPECT_THROW(soma::permute_entities(m, 0, across, t), std::invalid_argument);
}

// A tri6 whose vertices have every ownership: built as
//   interior 3 | 6, shared 1 2 | 4, shared_copy | 5
// (corners | node-only vertices). Vertices move within and across kinds as
// long as each kind keeps its corners first; the corner ends follow.
std::vector<GO> vertex_gids(const HostMesh& m) {
  std::vector<GO> out;
  for (LO v = 0; v < m.count(0); ++v) out.push_back(m.gid(0)(v));
  return out;
}

TEST(PermuteEntities, EachOwnershipKindKeepsItsCornersFirst) {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6},
                 std::vector<Real>{0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0});
  b.add_elements(Topology::tri, std::vector<GO>{9}, std::vector<GO>{1, 2, 3, 4, 5, 6}, 6);
  b.set_ownership(0, std::vector<GO>{1, 2, 4, 5},
                  std::vector<Ownership>{Ownership::shared, Ownership::shared, Ownership::shared, Ownership::shared_copy},
                  std::vector<std::uint32_t>(4, 0));
  HostMesh m = std::move(b).build(soma::Ordering::file);
  ASSERT_EQ(vertex_gids(m), (std::vector<GO>{3, 6, 1, 2, 4, 5}));
  add_fields(m, "vertex", 0);
  const auto table = [](std::vector<Ownership> own, std::vector<LO> cuts) {
    SegmentTable t = soma::make_segments(own, std::vector<Topology>(own.size(), Topology::vertex),
                                         std::vector<std::uint16_t>(own.size(), 1));
    soma::split_segments(t, cuts);
    return t;
  };
  const auto ends = [&] {
    return std::vector<LO>{m.corner_end(Ownership::interior), m.corner_end(Ownership::shared),
                           m.corner_end(Ownership::shared_copy), m.corner_end(Ownership::ghost)};
  };
  using O = Ownership;
  // 1 and 2 trade places.
  permute_and_check(m, 0, {0, 1, 3, 2, 4, 5}, table({O::interior, O::interior, O::shared, O::shared, O::shared, O::shared_copy}, {1, 4, 5}));
  EXPECT_EQ(ends(), (std::vector<LO>{1, 4, 5, 6}));
  EXPECT_EQ(vertex_gids(m), (std::vector<GO>{3, 6, 2, 1, 4, 5}));
  // 6 becomes a shared copy: interior 3, shared 2 1 | 4, shared_copy | 6 5.
  permute_and_check(m, 0, {0, 4, 1, 2, 3, 5}, table({O::interior, O::shared, O::shared, O::shared, O::shared_copy, O::shared_copy}, {1, 3, 4}));
  EXPECT_EQ(ends(), (std::vector<LO>{1, 3, 4, 6}));
  EXPECT_EQ(vertex_gids(m), (std::vector<GO>{3, 2, 1, 4, 6, 5}));
  EXPECT_EQ(m.corners(), 1);
  // Node 4 before corner 1 in the shared kind, whatever the table.
  const std::vector<LO> node_first{0, 2, 3, 1, 4, 5};
  EXPECT_THROW(soma::permute_entities(m, 0, node_first, table({O::interior, O::shared, O::shared, O::shared, O::shared_copy, O::shared_copy}, {1, 3, 4})),
               std::invalid_argument);
  EXPECT_THROW(soma::permute_entities(m, 0, node_first, table({O::interior, O::shared, O::shared, O::shared, O::shared_copy, O::shared_copy}, {1, 2, 4})),
               std::invalid_argument);
  EXPECT_EQ(ends(), (std::vector<LO>{1, 3, 4, 6}));
}

// Bad input throws std::invalid_argument and changes nothing.
TEST(PermuteEntities, InvalidInputLeavesTheMeshUnchanged) {
  HostMesh m = soma_test::two_quads_and_a_triangle();
  soma::build_adjacencies(m, planar_request());
  add_fields(m, "vertex", 0);
  add_sets(m, {0, 2});
  const Snapshot s = snapshot(m);
  const auto epoch = m.epoch();
  const SegmentTable tv = uniform(Topology::vertex, 1, {7, 0, 0, 0});
  const std::vector<LO> good = shuffled(7, 8);
  const auto unchanged = [&] {
    EXPECT_EQ(snapshot(m), s);
    EXPECT_EQ(m.epoch(), epoch);
    expect_table(m, 0, tv);
  };
  using V = std::vector<LO>;
  EXPECT_THROW(soma::permute_entities(m, -1, good, tv), std::invalid_argument);
  EXPECT_THROW(soma::permute_entities(m, 3, V{}, SegmentTable{}), std::invalid_argument);
  EXPECT_THROW(soma::permute_entities(m, 4, V{}, SegmentTable{}), std::invalid_argument);
  EXPECT_THROW(soma::permute_entities(m, 0, V{0, 1, 2, 3, 4, 5}, tv), std::invalid_argument);        // short
  EXPECT_THROW(soma::permute_entities(m, 0, V{0, 1, 2, 3, 4, 5, 6, 7}, tv), std::invalid_argument);  // long
  EXPECT_THROW(soma::permute_entities(m, 0, V{0, 1, 2, 3, 4, 5, 5}, tv), std::invalid_argument);     // repeat
  EXPECT_THROW(soma::permute_entities(m, 0, V{0, 1, 2, 3, 4, 5, 7}, tv), std::invalid_argument);     // past the end
  EXPECT_THROW(soma::permute_entities(m, 0, V{0, 1, 2, 3, 4, 5, -1}, tv), std::invalid_argument);
  unchanged();
  // Tables: the wrong count, a gap, unsorted ownership, the wrong topology or
  // arity, ends that disagree with the segments, a split segment.
  EXPECT_THROW(soma::permute_entities(m, 0, good, uniform(Topology::vertex, 1, {6, 0, 0, 0})), std::invalid_argument);
  SegmentTable gap = tv;
  gap.segments[0].begin = 1;
  EXPECT_THROW(soma::permute_entities(m, 0, good, gap), std::invalid_argument);
  SegmentTable unsorted;
  unsorted.count = 7;
  unsorted.segments = {soma::Segment{0, 3, Topology::vertex, 1, Ownership::shared},
                       soma::Segment{3, 7, Topology::vertex, 1, Ownership::interior}};
  EXPECT_THROW(soma::permute_entities(m, 0, good, unsorted), std::invalid_argument);
  EXPECT_THROW(soma::permute_entities(m, 0, good, uniform(Topology::edge, 2, {7, 0, 0, 0})), std::invalid_argument);
  SegmentTable ends = tv;
  ends.interior_end = 3;
  EXPECT_THROW(soma::permute_entities(m, 0, good, ends), std::invalid_argument);
  SegmentTable split = tv;
  split.segments = {soma::Segment{0, 3, Topology::vertex, 1, Ownership::interior},
                    soma::Segment{3, 7, Topology::vertex, 1, Ownership::interior}};
  EXPECT_THROW(soma::permute_entities(m, 0, good, split), std::invalid_argument);
  // Elements: the triangle must stay where the table puts triangles.
  const std::vector<Ownership> own(3, Ownership::interior);
  const std::vector<Topology> topo{Topology::tri, Topology::quad, Topology::quad};
  const std::vector<std::uint16_t> arity{3, 4, 4};
  const SegmentTable te = soma::make_segments(own, topo, arity);
  ASSERT_EQ(m.view().topology(2, 0), Topology::tri);
  EXPECT_THROW(soma::permute_entities(m, 2, V{1, 0, 2}, te), std::invalid_argument);
  unchanged();
}

// An edited mesh is compressed first.
TEST(PermuteEntities, EditedMeshesAreRefused) {
  HostMesh m = soma::testing::box_mesh({2, 1, 1, Topology::quad});
  soma::MeshEdit edit(m);
  edit.set_compress_threshold(0);
  edit.remove(2, std::vector<LO>{0});
  edit.apply();
  ASSERT_TRUE(m.edited());
  EXPECT_THROW(soma::permute_entities(m, 0, shuffled(m.count(0), 9), uniform(Topology::vertex, 1, {m.count(0), 0, 0, 0})),
               std::logic_error);
  m.compress();
  add_fields(m, "vertex", 0);
  const LO n = m.count(0);
  permute_and_check(m, 0, shuffled(n, 10), uniform(Topology::vertex, 1, {n, 0, 0, 0}));
}

}  // namespace

// Gids given with the permutation follow their entities; repeated gids are
// refused and change nothing.
TEST(PermuteEntities, GivenGidsFollowTheirEntities) {
  soma::HostMesh m = soma_test::two_quads_and_a_triangle();
  const soma::LO n = m.count(2);
  std::vector<soma::GO> old_gids;
  for (soma::LO e = 0; e < n; ++e) old_gids.push_back(m.gid(2)(e));
  const std::vector<soma::LO> identity{0, 1, 2};
  const soma::SegmentTable table = m.segment_table(2);
  const std::vector<soma::GO> twice{70, 70, 71};
  EXPECT_THROW(soma::permute_entities(m, 2, identity, table, twice), std::invalid_argument);
  EXPECT_THROW(soma::permute_entities(m, 2, identity, table, std::vector<soma::GO>{1}), std::invalid_argument);
  for (soma::LO e = 0; e < n; ++e) EXPECT_EQ(m.gid(2)(e), old_gids[static_cast<std::size_t>(e)]);
  const std::vector<soma::GO> fresh{-7, 900, 31};
  soma::permute_entities(m, 2, identity, table, fresh);
  for (soma::LO e = 0; e < n; ++e) EXPECT_EQ(m.gid(2)(e), fresh[static_cast<std::size_t>(e)]);
}
