// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "distributed_fixture.hpp"
#include "distributed_parts.hpp"
#include "mesh_text.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_edit.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/topology/build.hpp"

using namespace soma_test;
using soma::LO;
using soma::Ownership;

namespace {

Ownership kind_of(const soma::HostMesh& m, int d, LO e) {
  const soma::SegmentTable& t = m.segment_table(d);
  if (e < t.interior_end) return Ownership::interior;
  if (e < t.owned_end) return Ownership::shared;
  if (e < t.shared_end) return Ownership::shared_copy;
  return Ownership::ghost;
}

std::vector<int> sharing_of(const soma::HostMesh& m, int d, LO e) {
  const auto v = m.fields().get_var<int>(soma::sharing_field, d).value_or_default(d, e);
  return {v.begin(), v.end()};
}

int owner_of(const soma::HostMesh& m, int d, LO e, int me) {
  if (e < m.segment_table(d).owned_end) return me;
  return m.fields().get<int>(soma::owner_field, d).value_or_default(d, e);
}

std::map<GO, LO> lids(const soma::HostMesh& m, int d) {
  std::map<GO, LO> out;
  for (LO e = 0; e < m.count(d); ++e) out[m.gid(d)(e)] = e;
  return out;
}

// The cells of the next rank touching this rank's cells.
std::vector<GO> next_rank_neighbours(const std::map<GO, int>& rank_of, int me, int p) {
  std::set<GO> mine;
  for (const auto& [cell, r] : rank_of) {
    if (r == me) {
      for (const GO v : corners(static_cast<int>(cell - 1000))) mine.insert(v);
    }
  }
  std::vector<GO> out;
  for (const auto& [cell, r] : rank_of) {
    if (r != (me + 1) % p || r == me) continue;
    const auto c = corners(static_cast<int>(cell - 1000));
    if (std::any_of(c.begin(), c.end(), [&](GO v) { return mine.contains(v); })) out.push_back(cell);
  }
  return out;
}

}  // namespace

// Each owner lists, for each of its cells, exactly the ranks whose k-ring has
// it, however thin the parts: sliver columns, an empty rank, a lone cell.
TEST(Ghosts, LayersFollowTheGlobalRuleOnSliverPartitions) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  for (int kind = 0; kind < 4; ++kind) {
    const std::map<GO, int> rank_of = partition(c, kind);
    const soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, me));
    for (int bridge = 0; bridge < 2; ++bridge) {
      std::vector<std::set<GO>> rings;
      for (int layers = 1; layers <= 3; ++layers) {
        SCOPED_TRACE(testing::Message() << "partition " << kind << ", bridge " << bridge << ", layers " << layers);
        rings.clear();
        for (int r = 0; r < p; ++r) rings.push_back(ring(rank_of, r, layers, bridge));
        const soma::detail::GhostTargets t = soma::detail::ghost_targets(c, m, soma::GhostOptions{layers, bridge});
        std::int64_t sent = 0;
        for (LO e = 0; e < m.count(2); ++e) {
          const GO g = m.gid(2)(e);
          std::vector<int> want;
          for (int r = 0; r < p; ++r) {
            if (rings[static_cast<std::size_t>(r)].contains(g)) want.push_back(r);
          }
          const auto got = t.of(e);
          EXPECT_EQ(std::vector<int>(got.begin(), got.end()), want) << "cell " << g;
          sent += static_cast<std::int64_t>(got.size());
        }
        std::set<int> used;
        for (const auto& [cell, r] : rank_of) used.insert(r);
        if (used.size() > 1) EXPECT_GT(c.allreduce(sent, soma::Op::sum), 0);  // the guard of a vacuous pass
      }
    }
  }
}

TEST(Ghosts, BadOptionsThrowOnEveryRank) {
  const Comm c = Comm::world();
  const soma::HostMesh m = soma::build_local_part(c, local_builder(cell_ranks(c), c.rank()));
  EXPECT_THROW((void)soma::detail::ghost_targets(c, m, soma::GhostOptions{-1, 0}), std::invalid_argument);
  EXPECT_THROW((void)soma::detail::ghost_targets(c, m, soma::GhostOptions{1, 2}), std::invalid_argument);
  const int layers = c.rank() == c.size() - 1 ? 2 : 1;
  if (c.size() == 1) return;
  try {
    (void)soma::detail::ghost_targets(c, m, soma::GhostOptions{layers, 0});
    ADD_FAILURE() << "options that differ were accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("options differ"), std::string::npos) << e.what();
  }
}

// Entities held only through ghost cells come last, owned elsewhere; the
// others are as without the ghosts.
TEST(Ghosts, BuildLocalPartPutsGhostsLast) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  const soma::HostMesh ref = soma::build_local_part(c, local_builder(rank_of, me));

  std::vector<GO> cells, ghost_cells = next_rank_neighbours(rank_of, me, p);
  std::set<GO> own_vertices;
  for (const auto& [cell, r] : rank_of) {
    if (r != me) continue;
    cells.push_back(cell);
    for (const GO v : corners(static_cast<int>(cell - 1000))) own_vertices.insert(v);
  }
  std::array<std::vector<GO>, 4> ghosts;
  std::set<GO> gv;
  for (const GO cell : ghost_cells) {
    for (const GO v : corners(static_cast<int>(cell - 1000))) {
      if (!own_vertices.contains(v)) gv.insert(v);
    }
  }
  ghosts[0].assign(gv.begin(), gv.end());
  ghosts[2] = ghost_cells;
  std::vector<GO> all = cells;
  all.insert(all.end(), ghost_cells.begin(), ghost_cells.end());
  const soma::HostMesh m = soma::build_local_part(c, cells_builder(all), ghosts);

  const auto users = vertex_users(rank_of);
  for (const int d : {0, 2}) {
    const auto want = lids(ref, d);
    const std::set<GO> ghost_set(ghosts[static_cast<std::size_t>(d)].begin(), ghosts[static_cast<std::size_t>(d)].end());
    ASSERT_EQ(static_cast<std::size_t>(m.count(d)), want.size() + ghost_set.size());
    EXPECT_EQ(m.segment_table(d).shared_end, ref.segment_table(d).shared_end);
    int last_owner = -1;
    for (LO e = 0; e < m.count(d); ++e) {
      const GO g = m.gid(d)(e);
      if (ghost_set.contains(g)) {
        EXPECT_EQ(kind_of(m, d, e), Ownership::ghost) << "gid " << g;
        const int owner = d == 2 ? rank_of.at(g) : *users.at(g).begin();
        EXPECT_EQ(owner_of(m, d, e, me), owner) << "gid " << g;
        EXPECT_GE(owner, last_owner);  // one topology: grouped by owner
        last_owner = owner;
        continue;
      }
      ASSERT_TRUE(want.contains(g)) << "gid " << g;
      const LO r = want.at(g);
      EXPECT_EQ(kind_of(m, d, e), kind_of(ref, d, r)) << "gid " << g;
      EXPECT_EQ(owner_of(m, d, e, me), owner_of(ref, d, r, me)) << "gid " << g;
      if (e >= m.segment_table(d).interior_end) EXPECT_EQ(sharing_of(m, d, e), sharing_of(ref, d, r)) << "gid " << g;
    }
  }
}

// Edges derived on a part with ghost cells: those of no own cell are
// ghosts, owned by the lowest rank whose cells have them; the others are as
// without the ghosts, with the same gids.
TEST(Ghosts, DerivedEntitiesOfGhostsOnlyAreGhosts) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  soma::HostMesh ref = soma::build_local_part(c, local_builder(rank_of, me));
  soma::build_distributed_adjacencies(c, ref, soma::AdjacencyRequest{}.add(2, 1));

  std::vector<GO> all, ghost_cells = next_rank_neighbours(rank_of, me, p);
  std::set<GO> own_vertices;
  for (const auto& [cell, r] : rank_of) {
    if (r != me) continue;
    all.push_back(cell);
    for (const GO v : corners(static_cast<int>(cell - 1000))) own_vertices.insert(v);
  }
  all.insert(all.end(), ghost_cells.begin(), ghost_cells.end());
  std::array<std::vector<GO>, 4> ghosts;
  std::set<GO> gv;
  for (const GO cell : ghost_cells) {
    for (const GO v : corners(static_cast<int>(cell - 1000))) {
      if (!own_vertices.contains(v)) gv.insert(v);
    }
  }
  ghosts[0].assign(gv.begin(), gv.end());
  ghosts[2] = ghost_cells;
  soma::HostMesh m = soma::build_local_part(c, cells_builder(all), ghosts);
  soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));

  // Edges by their corners: the lowest rank with a cell having each.
  const auto ends = [](const soma::HostMesh& x, LO e) {
    std::pair<GO, GO> v{x.gid(0)(x.conn(1)(x.conn_offsets(1)(e))), x.gid(0)(x.conn(1)(x.conn_offsets(1)(e) + 1))};
    if (v.first > v.second) std::swap(v.first, v.second);
    return v;
  };
  std::map<std::pair<GO, GO>, int> lowest;
  for (const auto& [cell, r] : rank_of) {
    const auto q = corners(static_cast<int>(cell - 1000));
    for (int i = 0; i < 4; ++i) {
      std::pair<GO, GO> v{q[static_cast<std::size_t>(i)], q[static_cast<std::size_t>((i + 1) % 4)]};
      if (v.first > v.second) std::swap(v.first, v.second);
      const auto it = lowest.find(v);
      lowest[v] = it == lowest.end() ? r : std::min(it->second, r);
    }
  }
  std::map<std::pair<GO, GO>, LO> in_ref;
  for (LO e = 0; e < ref.count(1); ++e) in_ref[ends(ref, e)] = e;
  LO ghost_edges = 0;
  for (LO e = 0; e < m.count(1); ++e) {
    const auto v = ends(m, e);
    if (!in_ref.contains(v)) {
      ++ghost_edges;
      EXPECT_EQ(kind_of(m, 1, e), Ownership::ghost);
      EXPECT_EQ(owner_of(m, 1, e, me), lowest.at(v));
      continue;
    }
    const LO r = in_ref.at(v);
    EXPECT_EQ(m.gid(1)(e), ref.gid(1)(r));
    EXPECT_EQ(kind_of(m, 1, e), kind_of(ref, 1, r));
    EXPECT_EQ(owner_of(m, 1, e, me), owner_of(ref, 1, r, me));
    if (e >= m.segment_table(1).interior_end && e < m.segment_table(1).shared_end) {
      EXPECT_EQ(sharing_of(m, 1, e), sharing_of(ref, 1, r));
    }
  }
  EXPECT_EQ(m.count(1), ref.count(1) + ghost_edges);
  EXPECT_EQ(m.segment_table(1).shared_end, ref.segment_table(1).shared_end);
  if (p > 1) EXPECT_GT(c.allreduce(static_cast<std::int64_t>(ghost_edges), soma::Op::sum), 0);
}

TEST(Ghosts, BuildLocalPartRejectsGhostsItDoesNotHoldOnEveryRank) {
  const Comm c = Comm::world();
  const std::map<GO, int> rank_of = cell_ranks(c);
  std::array<std::vector<GO>, 4> ghosts;
  if (c.rank() == c.size() - 1) ghosts[2] = {999999};
  EXPECT_THROW((void)soma::build_local_part(c, local_builder(rank_of, c.rank()), ghosts), std::invalid_argument);
}

namespace {

// A snapshot of what ghosting must not change: per dimension, the gids and
// kinds by lid of the entities held before, and the sharing of the shared.
struct Held {
  std::array<std::vector<GO>, 3> gids;
  std::array<soma::SegmentTable, 3> table;
  std::array<std::vector<std::vector<int>>, 3> sharing;
};

Held held(const soma::HostMesh& m) {
  Held h;
  for (int d = 0; d <= 2; ++d) {
    const auto k = static_cast<std::size_t>(d);
    h.table[k] = m.segment_table(d);
    for (LO e = 0; e < m.segment_table(d).shared_end; ++e) {
      h.gids[k].push_back(m.gid(d)(e));
      if (e >= m.segment_table(d).interior_end) h.sharing[k].push_back(sharing_of(m, d, e));
    }
  }
  return h;
}

void expect_held(const soma::HostMesh& m, const Held& h) {
  for (int d = 0; d <= 2; ++d) {
    const auto k = static_cast<std::size_t>(d);
    const soma::SegmentTable& t = m.segment_table(d);
    EXPECT_EQ(t.interior_end, h.table[k].interior_end) << "dimension " << d;
    EXPECT_EQ(t.owned_end, h.table[k].owned_end) << "dimension " << d;
    ASSERT_EQ(t.shared_end, h.table[k].shared_end) << "dimension " << d;
    for (LO e = 0; e < t.shared_end; ++e) {
      ASSERT_EQ(m.gid(d)(e), h.gids[k][static_cast<std::size_t>(e)]) << "dimension " << d << ", lid " << e;
      if (e >= t.interior_end) EXPECT_EQ(sharing_of(m, d, e), h.sharing[k][static_cast<std::size_t>(e - t.interior_end)]);
    }
  }
}

// A field on all vertices (2 gid, copies holding -1, which must not count)
// and one on the even cells (gid + 0.5); a set of the even cells.
void decorate(soma::HostMesh& m) {
  auto f = m.fields().add_tag<double>("twice", 0, 1);
  for (LO e = 0; e < m.count(0); ++e) {
    f.view()(e, 0) = e < m.segment_table(0).owned_end ? 2.0 * static_cast<double>(m.gid(0)(e)) : -1.0;
  }
  std::vector<LO> even;
  for (LO e = 0; e < m.count(2); ++e) {
    if (m.gid(2)(e) % 2 == 0) even.push_back(e);
  }
  auto g = m.fields().add<double>("half", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::from_list(m, 2, even), soma::Layout::per_entity()), 1);
  soma::SetSpec s;
  s.gid = 77;
  for (LO e : even) {
    g.view()(g.space().entity_rows(2, e).begin, 0) = static_cast<double>(m.gid(2)(e)) + 0.5;
    s.contents.push_back({2, e});
  }
  soma::SetBuilder sb;
  sb.add(std::move(s));
  std::move(sb).install(m);
}

}  // namespace

// In place: what was held keeps its ids; ghost cells are the k-ring, their
// vertices and edges the closure; owners, fields and (when asked) sets come
// from the owners; ghosting again changes nothing; remove_ghosts restores.
TEST(Ghosts, AddGhostsKeepsIdsAndAppendsTheRing) {
  const Comm c = Comm::world();
  const int me = c.rank();
  for (int kind = 0; kind < 4; ++kind) {
    const std::map<GO, int> rank_of = partition(c, kind);
    std::set<int> used;
    for (const auto& [cell, r] : rank_of) used.insert(r);
    for (int layers = 1; layers <= 2; ++layers) {
      for (const bool sets : {false, true}) {
        SCOPED_TRACE(testing::Message() << "partition " << kind << ", layers " << layers << ", sets " << sets);
        soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, me));
        soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
        decorate(m);
        const Held before = held(m);
        soma::HostMesh& r = m.add_ghosts(c, soma::GhostOptions{layers, 0, sets});
        EXPECT_EQ(&r, &m);
        expect_held(m, before);

        // Ghost cells: the ring, grouped by owner.
        const std::set<GO> want = ring(rank_of, me, layers, 0);
        std::set<GO> cells, own_vertices, vertices, own_edges, edges;
        int last = -1;
        for (LO e = m.segment_table(2).shared_end; e < m.count(2); ++e) {
          const GO g = m.gid(2)(e);
          cells.insert(g);
          EXPECT_EQ(owner_of(m, 2, e, me), rank_of.at(g));
          EXPECT_GE(rank_of.at(g), last);
          last = rank_of.at(g);
        }
        EXPECT_EQ(cells, want);
        const auto edge_key = [](GO a, GO b) { return std::min(a, b) * 100000 + std::max(a, b); };
        for (const auto& [cell, q] : rank_of) {
          if (q != me && !want.contains(cell)) continue;
          const auto k = corners(static_cast<int>(cell - 1000));
          for (int i = 0; i < 4; ++i) {
            const GO a = k[static_cast<std::size_t>(i)], b = k[static_cast<std::size_t>((i + 1) % 4)];
            (q == me ? own_vertices : vertices).insert(a);
            (q == me ? own_edges : edges).insert(edge_key(a, b));
          }
        }
        std::set<GO> ghost_vertices, ghost_edges, want_vertices, want_edges;
        for (LO e = m.segment_table(0).shared_end; e < m.count(0); ++e) ghost_vertices.insert(m.gid(0)(e));
        for (LO e = m.segment_table(1).shared_end; e < m.count(1); ++e) {
          const auto off = m.conn_offsets(1);
          ghost_edges.insert(edge_key(m.gid(0)(m.conn(1)(off(e))), m.gid(0)(m.conn(1)(off(e) + 1))));
        }
        std::set_difference(vertices.begin(), vertices.end(), own_vertices.begin(), own_vertices.end(),
                            std::inserter(want_vertices, want_vertices.end()));
        std::set_difference(edges.begin(), edges.end(), own_edges.begin(), own_edges.end(),
                            std::inserter(want_edges, want_edges.end()));
        EXPECT_EQ(ghost_vertices, want_vertices);
        EXPECT_EQ(ghost_edges, want_edges);
        if (used.size() > 1) EXPECT_GT(c.allreduce(static_cast<std::int64_t>(cells.size()), soma::Op::sum), 0);

        // Fields from the owners; set members on ghosts only when asked.
        const auto twice = m.fields().get<double>("twice");
        for (LO e = m.segment_table(0).owned_end; e < m.count(0); ++e) {
          EXPECT_EQ(twice.value_or_default(0, e), 2.0 * static_cast<double>(m.gid(0)(e)));
        }
        const auto half = m.fields().get<double>("half");
        std::set<GO> members, want_members;
        for (const LO e : m.sets().members_of(*m.sets().index_of(77), 2)) members.insert(m.gid(2)(e));
        for (LO e = 0; e < m.count(2); ++e) {
          const GO g = m.gid(2)(e);
          const bool ghost = e >= m.segment_table(2).shared_end;
          EXPECT_EQ(half.space().entity_rows(2, e).valid(), g % 2 == 0) << "cell " << g;
          if (g % 2 == 0) {
            EXPECT_EQ(half.value_or_default(2, e), static_cast<double>(g) + 0.5);
            if (!ghost || sets) want_members.insert(g);
          }
        }
        EXPECT_EQ(members, want_members);

        // Again: the same ghosts; then none.
        std::vector<GO> first(m.gid(2).data(), m.gid(2).data() + m.count(2));
        m.add_ghosts(c, soma::GhostOptions{layers, 0, sets});
        EXPECT_EQ(std::vector<GO>(m.gid(2).data(), m.gid(2).data() + m.count(2)), first);
        m.remove_ghosts(c);
        expect_held(m, before);
        for (int d = 0; d <= 2; ++d) EXPECT_EQ(m.count(d), before.table[static_cast<std::size_t>(d)].shared_end);
      }
    }
  }
}

TEST(Ghosts, AnEditedMeshIsRefusedOnEveryRank) {
  const Comm c = Comm::world();
  soma::HostMesh m = soma::build_local_part(c, local_builder(cell_ranks(c), c.rank()));
  if (c.rank() == c.size() - 1) {
    soma::MeshEdit edit(m);
    (void)edit.add_vertices(std::vector<soma::Real>{-5, -5, 0});
    edit.apply();
  }
  EXPECT_THROW(m.add_ghosts(c, soma::GhostOptions{}), std::logic_error);
}

// A part reordered after its build keeps that order: ghosting restores each
// entity's id, with its values.
TEST(Ghosts, AddGhostsKeepsTheIdsOfAReorderedPart) {
  const Comm c = Comm::world();
  const int me = c.rank();
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(testing::Message() << "partition " << kind);
    const std::map<GO, int> rank_of = partition(c, kind);
    soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, me));
    soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
    for (int d = 0; d <= 2; ++d) {  // each segment reversed
      const soma::SegmentTable t = m.segment_table(d);
      std::vector<LO> p(static_cast<std::size_t>(m.count(d)));
      for (const soma::Segment& g : t.segments) {
        for (LO e = g.begin; e < g.end; ++e) p[static_cast<std::size_t>(e)] = g.begin + (g.end - 1 - e);
      }
      soma::permute_entities(m, d, p, t);
    }
    decorate(m);
    const Held before = held(m);
    m.add_ghosts(c, soma::GhostOptions{2, 0, false});
    expect_held(m, before);
    const auto twice = m.fields().get<double>("twice");
    for (LO e = 0; e < m.count(0); ++e) EXPECT_EQ(twice.value_or_default(0, e), 2.0 * static_cast<double>(m.gid(0)(e)));
  }
}

// Explicit edges (the horizontal ones) and derived ones (the vertical):
// those of ghost cells only are ghosts either way, and stay so when the
// edges are derived again.
TEST(Ghosts, ExplicitAndDerivedSidesOfGhostsAreGhosts) {
  const Comm c = Comm::world();
  const int me = c.rank();
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(testing::Message() << "partition " << kind);
    const std::map<GO, int> rank_of = partition(c, kind);
    std::vector<GO> cells;
    for (const auto& [cell, r] : rank_of) {
      if (r == me) cells.push_back(cell);
    }
    soma::HostMesh m = soma::build_local_part(c, cells_with_edges_builder(cells));
    soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
    const Held before = held(m);
    m.add_ghosts(c, soma::GhostOptions{1, 0, false});
    expect_held(m, before);
    const std::set<GO> ring1 = ring(rank_of, me, 1, 0);
    const auto key = [](GO a, GO b2) { return std::min(a, b2) * 100000 + std::max(a, b2); };
    std::set<GO> own, theirs, got;
    for (const auto& [cell, r] : rank_of) {
      if (r != me && !ring1.contains(cell)) continue;
      const auto q = corners(static_cast<int>(cell - 1000));
      for (int i = 0; i < 4; ++i) (r == me ? own : theirs).insert(key(q[static_cast<std::size_t>(i)], q[static_cast<std::size_t>((i + 1) % 4)]));
    }
    std::set<GO> want;
    std::set_difference(theirs.begin(), theirs.end(), own.begin(), own.end(), std::inserter(want, want.end()));
    const auto off = m.conn_offsets(1);
    for (LO e = m.segment_table(1).shared_end; e < m.count(1); ++e) got.insert(key(m.gid(0)(m.conn(1)(off(e))), m.gid(0)(m.conn(1)(off(e) + 1))));
    EXPECT_EQ(got, want);
  }
}

// update writes ghosts from their owners, as copies; reduce leaves them, and
// combines only the holders other than ghosts. Split or not.
TEST(Ghosts, UpdateReachesGhostsAndReduceDoesNot) {
  const Comm c = Comm::world();
  const int me = c.rank();
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(testing::Message() << "partition " << kind);
    soma::HostMesh m = soma::build_local_part(c, local_builder(partition(c, kind), me));
    soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
    m.add_ghosts(c, soma::GhostOptions{2, 0, false});
    for (int d = 0; d <= 2; ++d) {
      const soma::HaloExchange<Kokkos::HostSpace> x(c, m, d);
      const soma::SegmentTable& t = m.segment_table(d);
      soma::View2<double, Kokkos::HostSpace> v("v", static_cast<std::size_t>(m.count(d)), 2);
      for (LO e = 0; e < m.count(d); ++e) {
        const double g = static_cast<double>(m.gid(d)(e));
        v(e, 0) = e < t.owned_end ? g : -1.0;
        v(e, 1) = e < t.owned_end ? 3 * g : -1.0;
      }
      if (d == 1) {
        x.begin_update(c, v).end();
      } else {
        x.update(c, v);
      }
      for (LO e = 0; e < m.count(d); ++e) {
        const double g = static_cast<double>(m.gid(d)(e));
        EXPECT_EQ(v(e, 0), g) << "dimension " << d << ", gid " << m.gid(d)(e);
        EXPECT_EQ(v(e, 1), 3 * g) << "dimension " << d << ", gid " << m.gid(d)(e);
      }
      soma::View2<int, Kokkos::HostSpace> n("n", static_cast<std::size_t>(m.count(d)), 1);
      Kokkos::deep_copy(n, 1);
      x.reduce(c, n, soma::Op::sum);
      for (LO e = 0; e < m.count(d); ++e) {
        const int holders = e < t.interior_end || e >= t.shared_end ? 1 : 1 + static_cast<int>(sharing_of(m, d, e).size());
        EXPECT_EQ(n(e, 0), holders) << "dimension " << d << ", gid " << m.gid(d)(e);
      }
    }
  }
}

// A ghost whose recorded owner does not own it is refused on every rank.
TEST(Ghosts, HaloExchangeRefusesAGhostItsOwnerDoesNotOwn) {
  const Comm c = Comm::world();
  if (c.size() < 3) GTEST_SKIP() << "needs a rank that does not own the ghost";
  soma::HostMesh m = soma::build_local_part(c, local_builder(cell_ranks(c), c.rank()));
  m.add_ghosts(c, soma::GhostOptions{1, 0, false});
  if (c.rank() == 0 && m.count(2) > m.segment_table(2).shared_end) {
    auto owner = m.fields().get<int>(soma::owner_field, 2);
    const auto row = static_cast<std::size_t>(m.segment_table(2).shared_end - m.segment_table(2).owned_end);
    owner.view()(row, 0) = (owner.view()(row, 0) + 1) % c.size() == 0 ? 1 : (owner.view()(row, 0) + 1) % c.size();
  }
  EXPECT_THROW((soma::HaloExchange<Kokkos::HostSpace>(c, m, 2)), std::invalid_argument);
}

// With every cell on one rank no rank shares, ghosts or gains anything, so
// add_ghosts leaves the mesh as it is: it routes no more than finding the
// (empty) ghost targets does, ids and gids stay, and field storage is kept.
TEST(Ghosts, NothingToGhostLeavesTheMeshAsItIs) {
  const Comm c = Comm::world();
  std::map<GO, int> rank_of = cell_ranks(c);
  for (auto& [cell, r] : rank_of) r = 0;
  soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, c.rank()));
  auto f = m.fields().add_tag<double>("t", 0, 1);
  const double* before_data = f.view().data();
  std::array<std::vector<GO>, 3> gids;
  for (int d = 0; d <= 2; ++d) gids[static_cast<std::size_t>(d)].assign(m.gid(d).data(), m.gid(d).data() + m.count(d));
  const soma::GhostOptions o{1, 0, false};
  const auto c0 = soma::detail::comm_counts();
  (void)soma::detail::ghost_targets(c, m, o);
  const auto c1 = soma::detail::comm_counts();
  m.add_ghosts(c, o);
  const auto c2 = soma::detail::comm_counts();
  EXPECT_EQ(c2.routes - c1.routes, c1.routes - c0.routes);
  for (int d = 0; d <= 2; ++d) {
    EXPECT_EQ(std::vector<GO>(m.gid(d).data(), m.gid(d).data() + m.count(d)), gids[static_cast<std::size_t>(d)]) << d;
    EXPECT_EQ(m.segment_table(d).shared_end, m.count(d)) << d;
  }
  EXPECT_EQ(m.fields().get<double>("t", 0).view().data(), before_data);
  m.remove_ghosts(c);
  EXPECT_EQ(m.fields().get<double>("t", 0).view().data(), before_data);
}

namespace {

// Where two parts differ entity for entity, or "": per dimension the gids by
// lid, the segments, the sharing and owners, then the held graphs by lid and
// everything else by gid.
std::string exact_difference(const soma::HostMesh& a, const soma::HostMesh& b, int me) {
  if (a.dim() != b.dim()) return "dimension";
  for (int d = 0; d <= a.dim(); ++d) {
    const std::string at = " of dimension " + std::to_string(d);
    if (a.count(d) != b.count(d)) return "count" + at;
    const soma::SegmentTable &x = a.segment_table(d), &y = b.segment_table(d);
    if (x.segments != y.segments || x.interior_end != y.interior_end || x.owned_end != y.owned_end ||
        x.shared_end != y.shared_end) {
      return "segments" + at;
    }
    if (d == 0) {
      for (const soma::Ownership k : {soma::Ownership::interior, soma::Ownership::shared, soma::Ownership::shared_copy,
                                      soma::Ownership::ghost}) {
        if (a.corner_end(k) != b.corner_end(k)) return "corner end of kind " + std::to_string(static_cast<int>(k));
      }
    }
    for (LO e = 0; e < a.count(d); ++e) {
      if (a.gid(d)(e) != b.gid(d)(e)) return "gid of " + std::to_string(e) + at;
      if (owner_of(a, d, e, me) != owner_of(b, d, e, me)) return "owner of " + std::to_string(e) + at;
      if (e >= x.interior_end && e < x.shared_end && sharing_of(a, d, e) != sharing_of(b, d, e)) {
        return "sharing of " + std::to_string(e) + at;
      }
    }
  }
  if (a.adjacencies().bits() != b.adjacencies().bits()) return "graphs held";
  for (int from = 0; from <= a.dim(); ++from) {
    for (int to = 0; to <= a.dim(); ++to) {
      if (from == to || to == 0 || !a.has_adjacency(from, to)) continue;
      const auto g = a.adjacency(from, to), h = b.adjacency(from, to);
      const std::string graph = "graph " + std::to_string(from) + " -> " + std::to_string(to);
      const auto differ = [](const auto& u, const auto& v) -> std::string {
        if (u.extent(0) != v.extent(0)) return " in length";
        for (std::size_t i = 0; i < u.extent(0); ++i) {
          if (u(i) != v(i)) return " at " + std::to_string(i);
        }
        return "";
      };
      if (const std::string w = differ(g.offsets, h.offsets); !w.empty()) return graph + " offsets" + w;
      if (const std::string w = differ(g.targets, h.targets); !w.empty()) return graph + " targets" + w;
      if (const std::string w = differ(g.codes, h.codes); !w.empty()) return graph + " codes" + w;
    }
  }
  return mesh_difference(a, b);
}

}  // namespace

// add_ghosts sends each ghosting rank its ghosts' closures and builds the
// part from them: the part, entity for entity, that rebuilding it through
// the directories gives, on every partition and kind of part, adding ghosts
// to a part that has some and removing them. (The grid with extras only on
// the curve partition: on the others the rebuild moves the diagonal edge to
// a rank ghosting both its corners; see the next test.)
TEST(Ghosts, AddGhostsBuildsWhatTheRebuildBuilds) {
  const Comm c = Comm::world();
  const int me = c.rank();
  for (int kind = 0; kind < 4; ++kind) {
    const std::map<GO, int> rank_of = partition(c, kind);
    std::vector<GO> cells;
    for (const auto& [cell, r] : rank_of) {
      if (r == me) cells.push_back(cell);
    }
    for (int part = 0; part < 6; ++part) {
      if (part == 2 && kind != 0) continue;
      soma::HostMesh m;
      if (part == 0 || part == 5) {
        m = soma::build_local_part(c, local_builder(rank_of, me));
        soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
        decorate(m);
      } else if (part == 1) {
        m = soma::build_local_part(c, local_builder(rank_of, me));
        soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1).add(0, 2));
        add_fields(m, me);
      } else if (part == 2) {
        m = grid_with_extras(c, rank_of);
        add_sets(m);
      } else if (part == 3) {
        m = soma::build_local_part(c, cells_with_edges_builder(cells));
        soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
      } else {
        m = soma::build_local_part(c, quad9_builder(rank_of, me));
      }
      if (part == 5) {  // each segment reversed
        for (int d = 0; d <= 2; ++d) {
          const soma::SegmentTable t = m.segment_table(d);
          std::vector<LO> p(static_cast<std::size_t>(m.count(d)));
          for (const soma::Segment& g : t.segments) {
            for (LO e = g.begin; e < g.end; ++e) p[static_cast<std::size_t>(e)] = g.begin + (g.end - 1 - e);
          }
          soma::permute_entities(m, d, p, t);
        }
      }
      for (const soma::GhostOptions o : {soma::GhostOptions{1, 0, false}, soma::GhostOptions{2, 1, true},
                                         soma::GhostOptions{1, 1, true}, soma::GhostOptions{2, 0, false},
                                         soma::GhostOptions{0, 0, false}}) {
        SCOPED_TRACE(testing::Message() << "partition " << kind << ", part " << part << ", layers " << o.layers
                                        << ", bridge " << o.bridge_dim << ", sets " << o.sets);
        soma::HostMesh want = m.clone();
        soma::detail::add_ghosts_by_rebuild(c, want, o);
        m.add_ghosts(c, o);  // on top of the ghosts the last options gave
        EXPECT_EQ(exact_difference(m, want, me), "");
      }
    }
  }
}

// Entities held other than as ghosts stay held, with their ids, whatever
// else holds their vertices as ghosts: the diagonal edge (whose far corner a
// rank holds only for it) and the free vertex too.
TEST(Ghosts, AddGhostsKeepsWhatARankHoldsOnlyForAnEdge) {
  const Comm c = Comm::world();
  for (int kind = 0; kind < 4; ++kind) {
    const std::map<GO, int> rank_of = partition(c, kind);
    for (int layers = 1; layers <= 2; ++layers) {
      SCOPED_TRACE(testing::Message() << "partition " << kind << ", layers " << layers);
      soma::HostMesh m = grid_with_extras(c, rank_of);
      add_sets(m);
      const Held before = held(m);
      m.add_ghosts(c, soma::GhostOptions{layers, 0, true});
      expect_held(m, before);
      m.remove_ghosts(c);
      expect_held(m, before);
      for (int d = 0; d <= 2; ++d) EXPECT_EQ(m.count(d), before.table[static_cast<std::size_t>(d)].shared_end);
    }
  }
}


// Cells held on two ranks (each rank also holds the first cell of the next):
// no rank holding a cell ghosts it, and the part is the rebuild's.
TEST(Ghosts, CellsHeldOnTwoRanksAreNotGhostedThere) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  for (int kind = 0; kind < 4; ++kind) {
    const std::map<GO, int> rank_of = partition(c, kind);
    std::vector<GO> cells;
    std::map<int, GO> first;
    for (const auto& [cell, r] : rank_of) {
      if (r == me) cells.push_back(cell);
      first.emplace(r, cell);
    }
    const auto next = first.find((me + 1) % p);
    if (next != first.end() && rank_of.at(next->second) != me) cells.push_back(next->second);
    for (int layers = 1; layers <= 2; ++layers) {
      SCOPED_TRACE(testing::Message() << "partition " << kind << ", layers " << layers);
      soma::HostMesh m = soma::build_local_part(c, cells_builder(cells));
      const soma::detail::GhostTargets t = soma::detail::ghost_targets(c, m, soma::GhostOptions{layers, 0});
      const soma::SegmentTable& s = m.segment_table(2);
      for (LO e = 0; e < s.owned_end; ++e) {
        const std::vector<int> holders = e < s.interior_end ? std::vector<int>{} : sharing_of(m, 2, e);
        for (const int r : t.of(e)) {
          EXPECT_NE(r, me) << "cell " << m.gid(2)(e);
          EXPECT_EQ(std::count(holders.begin(), holders.end(), r), 0) << "cell " << m.gid(2)(e) << ", rank " << r;
        }
      }
      soma::HostMesh want = m.clone();
      soma::detail::add_ghosts_by_rebuild(c, want, soma::GhostOptions{layers, 0, false});
      m.add_ghosts(c, soma::GhostOptions{layers, 0, false});
      EXPECT_EQ(exact_difference(m, want, me), "");
    }
  }
}
