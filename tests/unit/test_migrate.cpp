// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "distributed_fixture.hpp"
#include "distributed_parts.hpp"
#include "soma/core/handle_ref.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_edit.hpp"
#include "soma/core/mesh_values.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/core/sets.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/parallel/transport/rendezvous.hpp"
#include "soma/topology/build.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;

using namespace soma_test;

namespace {


std::vector<int> sorted_ranks(std::span<const int> r) {
  std::vector<int> v(r.begin(), r.end());
  std::sort(v.begin(), v.end());
  return v;
}


}  // namespace

// Each entity is placed by its owner: an element at its destination, a
// vertex on every rank receiving an element that uses it, an explicit edge on
// every rank holding both its corners. What no rank receives stays with its
// owner, and its vertices with it.
TEST(Migrate, PlacementFollowsTheElements) {
  const Comm c = Comm::world();
  const int p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  const soma::HostMesh m = grid_with_extras(c, rank_of);
  std::map<GO, int> moved;  // every cell one rank up
  for (const auto& [cell, r] : rank_of) moved[cell] = (r + 1) % p;
  std::map<GO, std::set<int>> holders;  // vertex -> ranks receiving a cell using it
  for (const auto& [cell, r] : moved) {
    for (const GO v : corners(static_cast<int>(cell - 1000))) holders[v].insert(r);
  }
  const int d = m.dim();
  const auto egid = m.gid(d);
  std::vector<std::size_t> off{0};
  std::vector<int> to;
  for (LO e = 0; e < m.count(d); ++e) {
    to.push_back(moved.at(egid(e)));
    off.push_back(to.size());
  }
  const soma::detail::Placement where = soma::detail::place_closure(c, m, off, to);

  for (LO e = 0; e < m.count(d); ++e) {
    EXPECT_EQ(sorted_ranks(where.of(d, e)), std::vector<int>{moved.at(egid(e))}) << "cell " << egid(e);
  }
  // The diagonal stays with its owner when no rank receives both its corners.
  const std::set<int>& first = holders.at(vertex(0, 0));
  const std::set<int>& last = holders.at(vertex(nx, ny));
  std::vector<int> diag_to;
  std::set_intersection(first.begin(), first.end(), last.begin(), last.end(), std::back_inserter(diag_to));
  const int diag_owner = rank_of.at(1000);
  if (diag_to.empty()) diag_to = {diag_owner};
  const auto& t1 = m.segment_table(1);
  const auto e1 = m.gid(1);
  std::int64_t placed_edges = 0;
  for (LO e = 0; e < m.count(1); ++e) {
    if (e >= t1.owned_end) {
      EXPECT_TRUE(where.of(1, e).empty()) << "a copy of edge " << e1(e) << " is placed";
      continue;
    }
    ++placed_edges;
    std::vector<int> want;
    if (e1(e) == diagonal) {
      want = diag_to;
    } else {
      const GO j = e1(e) - 5000;
      const std::set<int>& a = holders.at(vertex(6, static_cast<int>(j)));
      const std::set<int>& b = holders.at(vertex(6, static_cast<int>(j) + 1));
      std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(want));
    }
    EXPECT_EQ(sorted_ranks(where.of(1, e)), want) << "edge " << e1(e);
  }
  EXPECT_EQ(c.allreduce(placed_edges, soma::Op::sum), ny + 1);
  // Vertices go where their cells go, and with the diagonal where it goes.
  for (const int r : diag_to) {
    holders[vertex(0, 0)].insert(r);
    holders[vertex(nx, ny)].insert(r);
  }
  holders[free_vertex] = {0};
  const auto& t0 = m.segment_table(0);
  const auto v0 = m.gid(0);
  std::int64_t placed_vertices = 0;
  for (LO e = 0; e < m.count(0); ++e) {
    if (e >= t0.owned_end) {
      EXPECT_TRUE(where.of(0, e).empty()) << "a copy of vertex " << v0(e) << " is placed";
      continue;
    }
    ++placed_vertices;
    const std::set<int>& h = holders.at(v0(e));
    EXPECT_EQ(sorted_ranks(where.of(0, e)), std::vector<int>(h.begin(), h.end())) << "vertex " << v0(e);
  }
  EXPECT_EQ(c.allreduce(placed_vertices, soma::Op::sum), (nx + 1) * (ny + 1) + 1);
}

TEST(Migrate, PlacementRejectsBadDestinations) {
  const Comm c = Comm::world();
  const soma::HostMesh m = grid_with_extras(c, cell_ranks(c));
  const auto n = static_cast<std::size_t>(m.count(m.dim()));
  std::vector<std::size_t> off(n + 1);
  for (std::size_t i = 0; i <= n; ++i) off[i] = i;
  std::vector<int> to(n, c.size());  // a rank outside comm
  EXPECT_THROW((void)soma::detail::place_closure(c, m, off, to), std::invalid_argument);
  std::vector<std::size_t> short_off(n, 0);  // one offset short
  std::vector<int> none;
  EXPECT_THROW((void)soma::detail::place_closure(c, m, short_off, none), std::invalid_argument);
}

namespace {

// Rows sending each element of m to the rank `moved` names for its gid.
std::pair<std::vector<std::size_t>, std::vector<int>> rows_to(const soma::HostMesh& m, const std::map<GO, int>& moved) {
  const int d = m.dim();
  const auto egid = m.gid(d);
  std::vector<std::size_t> off{0};
  std::vector<int> to;
  for (LO e = 0; e < m.count(d); ++e) {
    to.push_back(moved.at(egid(e)));
    off.push_back(to.size());
  }
  return {off, to};
}

}  // namespace

// Edge midpoints and centres go wherever a cell using them goes.
TEST(Migrate, PlacementCarriesHighOrderNodes) {
  const Comm c = Comm::world();
  const int p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  const soma::HostMesh m = soma::build_local_part(c, quad9_builder(rank_of, c.rank()));
  std::map<GO, int> moved;
  std::map<GO, std::set<int>> holders;
  for (const auto& [cell, r] : rank_of) {
    moved[cell] = (r + 1) % p;
    for (const GO v : quad9(static_cast<int>(cell - 1000))) holders[v].insert((r + 1) % p);
  }
  const auto [off, to] = rows_to(m, moved);
  const soma::detail::Placement where = soma::detail::place_closure(c, m, off, to);
  const auto gid = m.gid(0);
  std::int64_t nodes = 0;
  for (LO e = 0; e < m.segment_table(0).owned_end; ++e) {
    const std::set<int>& h = holders.at(gid(e));
    EXPECT_EQ(sorted_ranks(where.of(0, e)), std::vector<int>(h.begin(), h.end())) << "point " << gid(e);
    const auto [i, j] = lattice(gid(e));
    if (i % 2 != 0 || j % 2 != 0) ++nodes;
  }
  EXPECT_EQ(c.allreduce(nodes, soma::Op::sum), static_cast<std::int64_t>((2 * nx + 1) * (2 * ny + 1) - (nx + 1) * (ny + 1)));
}

// An element held by two ranks is placed once, by its owner.
TEST(Migrate, CopiesOfElementsPlaceNothing) {
  const Comm c = Comm::world();
  const int me = c.rank(), last = c.size() - 1;
  const std::map<GO, int> rank_of = cell_ranks(c);
  soma::MeshBuilder b = local_builder(rank_of, me);
  if (me == last && rank_of.at(1000) != me) {  // a copy of the first cell
    const std::set<GO> here(b.vertex_gids().begin(), b.vertex_gids().end());
    std::vector<GO> add;
    std::vector<soma::Real> xyz;
    for (const GO v : corners(0)) {
      if (here.contains(v)) continue;
      add.push_back(v);
      xyz.insert(xyz.end(), {static_cast<soma::Real>((v - 1) % (nx + 1)), static_cast<soma::Real>((v - 1) / (nx + 1)), 0});
    }
    b.add_vertices(add, xyz);
    b.add_elements(soma::Topology::quad, std::vector<GO>{1000}, corners(0));
  }
  const soma::HostMesh m = soma::build_local_part(c, std::move(b));
  std::map<GO, int> moved;
  for (const auto& [cell, r] : rank_of) moved[cell] = (r + 1) % c.size();
  const auto [off, to] = rows_to(m, moved);
  const soma::detail::Placement where = soma::detail::place_closure(c, m, off, to);
  const auto gid = m.gid(2);
  std::int64_t placed = 0;
  for (LO e = 0; e < m.count(2); ++e) {
    if (where.of(2, e).empty()) continue;
    ++placed;
    EXPECT_EQ(sorted_ranks(where.of(2, e)), std::vector<int>{moved.at(gid(e))});
  }
  EXPECT_EQ(c.allreduce(placed, soma::Op::sum), static_cast<std::int64_t>(nx * ny));
}

namespace {

std::vector<GO> gids_of(const soma::HostMesh& m, int d) {
  std::vector<GO> out;
  const auto g = m.gid(d);
  for (LO e = 0; e < m.count(d); ++e) out.push_back(g(e));
  return out;
}


std::vector<int> one_rank_up(const soma::HostMesh& m, const std::map<GO, int>& rank_of, int p) {
  std::vector<int> dest;
  const auto g = m.gid(m.dim());
  for (LO e = 0; e < m.count(m.dim()); ++e) dest.push_back((rank_of.at(g(e)) + 1) % p);
  return dest;
}

std::int64_t owned(const soma::HostMesh& m, int d) { return m.segment_table(d).owned_end; }

}  // namespace

// Moved one rank up, each part holds the cells sent to it with their corners,
// the explicit edges whose corners it holds, and whatever stayed; every
// entity is owned once and the part has its sharing.
TEST(Migrate, MovedEntitiesKeepTheirGidsCoordinatesAndConnectivity) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  const soma::HostMesh m = grid_with_extras(c, rank_of);
  const soma::HostMesh part = soma::migrate(c, m, one_rank_up(m, rank_of, p));

  std::set<GO> cells, verts;
  for (const auto& [cell, r] : rank_of) {
    if ((r + 1) % p != me) continue;
    cells.insert(cell);
    for (const GO v : corners(static_cast<int>(cell - 1000))) verts.insert(v);
  }
  const std::vector<GO> got_cells = gids_of(part, 2);
  EXPECT_EQ(std::set<GO>(got_cells.begin(), got_cells.end()), cells);
  for (LO e = 0; e < part.count(2); ++e) {
    EXPECT_EQ(nodes_of(part, 2, e), corners(static_cast<int>(part.gid(2)(e) - 1000))) << "cell " << part.gid(2)(e);
  }
  std::set<GO> edges;
  for (int j = 0; j < ny; ++j) {
    if (verts.contains(vertex(6, j)) && verts.contains(vertex(6, j + 1))) edges.insert(5000 + j);
  }
  const auto vg = part.gid(0);
  const auto xyz = part.coords();
  for (LO e = 0; e < part.count(0); ++e) {
    const GO g = vg(e);
    const double x = g == free_vertex ? -1 : static_cast<double>((g - 1) % (nx + 1));
    const double y = g == free_vertex ? -1 : static_cast<double>((g - 1) / (nx + 1));
    EXPECT_EQ(xyz(e, 0), x) << "vertex " << g;
    EXPECT_EQ(xyz(e, 1), y) << "vertex " << g;
  }
  const std::vector<GO> got_vertices = gids_of(part, 0);
  const std::set<GO> vset(got_vertices.begin(), got_vertices.end());
  const std::vector<GO> got_edges = gids_of(part, 1);
  for (const GO e : got_edges) {
    if (e != diagonal) EXPECT_TRUE(edges.contains(e)) << "edge " << e;
  }
  for (const GO e : edges) EXPECT_NE(std::find(got_edges.begin(), got_edges.end(), e), got_edges.end()) << "edge " << e;
  for (const GO v : verts) EXPECT_TRUE(vset.contains(v)) << "vertex " << v;
  EXPECT_EQ(vset.contains(free_vertex), me == 0);
  for (LO e = 0; e < part.count(1); ++e) {
    const std::vector<GO> ends = nodes_of(part, 1, e);
    if (part.gid(1)(e) == diagonal) {
      EXPECT_EQ(ends, (std::vector<GO>{vertex(0, 0), vertex(nx, ny)}));
    } else {
      const int j = static_cast<int>(part.gid(1)(e) - 5000);
      EXPECT_EQ(ends, (std::vector<GO>{vertex(6, j), vertex(6, j + 1)}));
    }
  }
  EXPECT_TRUE(part.input(1) || part.count(1) == 0);
  EXPECT_EQ(c.allreduce(owned(part, 0), soma::Op::sum), static_cast<std::int64_t>((nx + 1) * (ny + 1) + 1));
  EXPECT_EQ(c.allreduce(owned(part, 1), soma::Op::sum), static_cast<std::int64_t>(ny + 1));
  EXPECT_EQ(c.allreduce(owned(part, 2), soma::Op::sum), static_cast<std::int64_t>(nx * ny));
  EXPECT_NO_THROW((void)part.fields().get_var<int>(soma::sharing_field, 0));
}

// Sent to their own ranks, the parts are what they were.
TEST(Migrate, StayingPutKeepsThePart) {
  const Comm c = Comm::world();
  const std::map<GO, int> rank_of = cell_ranks(c);
  const soma::HostMesh m = grid_with_extras(c, rank_of);
  const soma::HostMesh part = soma::migrate(c, m, std::vector<int>(static_cast<std::size_t>(m.count(2)), c.rank()));
  for (int d = 0; d <= 2; ++d) {
    EXPECT_EQ(gids_of(part, d), gids_of(m, d)) << "dimension " << d;
    EXPECT_EQ(part.segment_table(d).segments, m.segment_table(d).segments) << "dimension " << d;
  }
  for (LO e = 0; e < m.count(0); ++e) {
    for (int k = 0; k < 3; ++k) EXPECT_EQ(part.coords()(e, k), m.coords()(e, k));
  }
}

// Corner and node gids travel whole: quad9s and polygons.
TEST(Migrate, HighOrderElementsAndPolygonsKeepTheirNodes) {
  const Comm c = Comm::world();
  const int p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  const soma::HostMesh q = soma::build_local_part(c, quad9_builder(rank_of, c.rank()));
  const soma::HostMesh qp = soma::migrate(c, q, one_rank_up(q, rank_of, p));
  for (LO e = 0; e < qp.count(2); ++e) {
    EXPECT_EQ(nodes_of(qp, 2, e), quad9(static_cast<int>(qp.gid(2)(e) - 1000))) << "cell " << qp.gid(2)(e);
  }
  EXPECT_EQ(c.allreduce(owned(qp, 0), soma::Op::sum), static_cast<std::int64_t>((2 * nx + 1) * (2 * ny + 1)));
  EXPECT_EQ(c.allreduce(owned(qp, 2), soma::Op::sum), static_cast<std::int64_t>(nx * ny));

  soma::MeshBuilder b(2);
  {
    const soma::MeshBuilder lin = local_builder(rank_of, c.rank());
    b.add_vertices(lin.vertex_gids(), lin.vertex_coordinates());
    std::vector<GO> cells, conn;
    std::vector<soma::Offset> off{0};
    for (const auto& [cell, r] : rank_of) {
      if (r != c.rank()) continue;
      cells.push_back(cell);
      for (const GO v : corners(static_cast<int>(cell - 1000))) conn.push_back(v);
      off.push_back(static_cast<soma::Offset>(conn.size()));
    }
    b.add_polygons(cells, off, conn);
  }
  const soma::HostMesh poly = soma::build_local_part(c, std::move(b));
  const soma::HostMesh pp = soma::migrate(c, poly, one_rank_up(poly, rank_of, p));
  for (LO e = 0; e < pp.count(2); ++e) {
    EXPECT_EQ(pp.topology(2)(e), static_cast<std::uint8_t>(soma::Topology::polygon));
    EXPECT_EQ(nodes_of(pp, 2, e), corners(static_cast<int>(pp.gid(2)(e) - 1000))) << "cell " << pp.gid(2)(e);
  }
  EXPECT_EQ(c.allreduce(owned(pp, 2), soma::Op::sum), static_cast<std::int64_t>(nx * ny));
}

// Derived edges are derived again on the new parts, with the same gids.
TEST(Migrate, DerivedEntitiesComeBackWithTheirGids) {
  const Comm c = Comm::world();
  const std::map<GO, int> rank_of = cell_ranks(c);
  soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, c.rank()));
  soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
  // A sum over owned edges of a hash of (corners, gid): each edge counts once.
  const auto signature = [&](const soma::HostMesh& x) {
    std::uint64_t s = 0;
    for (LO e = 0; e < x.segment_table(1).owned_end; ++e) {
      std::vector<GO> ends = nodes_of(x, 1, e);
      std::sort(ends.begin(), ends.end());
      const auto h = static_cast<std::uint64_t>(ends[0]) * 1000003u + static_cast<std::uint64_t>(ends[1]);
      s += soma::transport::mix(h ^ static_cast<std::uint64_t>(x.gid(1)(e)));
    }
    return c.allreduce(s, soma::Op::sum);
  };
  const soma::HostMesh part = soma::migrate(c, m, one_rank_up(m, rank_of, c.size()));
  EXPECT_FALSE(part.input(1));
  EXPECT_TRUE(part.has_adjacency(2, 1));
  EXPECT_EQ(signature(part), signature(m));
}

TEST(Migrate, BadDestinationsAndEditedMeshesThrowOnEveryRank) {
  const Comm c = Comm::world();
  soma::HostMesh m = grid_with_extras(c, cell_ranks(c));
  const auto n = static_cast<std::size_t>(m.count(2));
  EXPECT_THROW((void)soma::migrate(c, m, std::vector<int>(n + 1, 0)), std::invalid_argument);
  EXPECT_THROW((void)soma::migrate(c, m, std::vector<int>(n, c.size())), std::invalid_argument);
  if (c.rank() == c.size() - 1) {
    soma::MeshEdit edit(m);
    (void)edit.add_vertices(std::vector<soma::Real>{-5, -5, 0});
    edit.apply();
  }
  EXPECT_THROW((void)soma::migrate(c, m, std::vector<int>(n, 0)), std::logic_error);
}

// quad9s and quads in one mesh (the last rank only quads, so ranks send
// records of different widths), off the plane, with a mesh value: node counts
// stay apart, every coordinate travels, mesh values are copied.
TEST(Migrate, MixedNodeCountsCoordinatesAndMeshValues) {
  const Comm c = Comm::world();
  const std::map<GO, int> rank_of = cell_ranks(c);
  const auto height = [](GO v) { return 0.125 * lattice(v).second + 0.5; };
  std::set<GO> used;
  std::array<std::vector<GO>, 2> cells, conn;  // quads, quad9s
  for (const auto& [cell, r] : rank_of) {
    if (r != c.rank()) continue;
    const std::vector<GO> n = quad9(static_cast<int>(cell - 1000));
    const std::size_t k = cell % 2 == 0 && (r != c.size() - 1 || c.size() == 1) ? 1 : 0;
    cells[k].push_back(cell);
    for (std::size_t q = 0; q < (k == 1 ? 9u : 4u); ++q) {
      conn[k].push_back(n[q]);
      used.insert(n[q]);
    }
  }
  std::vector<GO> vg(used.begin(), used.end());
  std::vector<soma::Real> xyz;
  for (const GO v : vg) xyz.insert(xyz.end(), {0.5 * lattice(v).first, 0.5 * lattice(v).second, height(v)});
  soma::MeshBuilder b(2);
  b.add_vertices(vg, xyz);
  b.add_elements(soma::Topology::quad, cells[0], conn[0]);
  b.add_elements(soma::Topology::quad, cells[1], conn[1], 9);
  soma::HostMesh m = soma::build_local_part(c, std::move(b));
  soma::MeshValue answer;
  answer.type = soma::DataType::i32;
  answer.global_value = soma::MeshValue::bytes_of(std::span<const std::int32_t>(std::vector<std::int32_t>{42}));
  m.values().set("answer", answer);

  const soma::HostMesh part = soma::migrate(c, m, one_rank_up(m, rank_of, c.size()));
  for (LO e = 0; e < part.count(2); ++e) {
    const GO cell = part.gid(2)(e);
    std::vector<GO> want = quad9(static_cast<int>(cell - 1000));
    if (cell % 2 != 0 || (rank_of.at(cell) == c.size() - 1 && c.size() > 1)) want.resize(4);
    EXPECT_EQ(nodes_of(part, 2, e), want) << "cell " << cell;
  }
  for (LO e = 0; e < part.count(0); ++e) EXPECT_EQ(part.coords()(e, 2), height(part.gid(0)(e)));
  ASSERT_EQ(part.values().names(), std::vector<std::string>{"answer"});
  EXPECT_EQ(part.values().at("answer").global_value, answer.global_value);
  EXPECT_EQ(c.allreduce(owned(part, 2), soma::Op::sum), static_cast<std::int64_t>(nx * ny));
}

// Edges derived with no graph held (only their connectivity) come back too.
TEST(Migrate, DimensionsDerivedWithoutGraphsComeBack) {
  const Comm c = Comm::world();
  const std::map<GO, int> rank_of = cell_ranks(c);
  soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, c.rank()));
  soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(1, 0));
  ASSERT_EQ(m.adjacencies().bits(), 0u);
  const soma::HostMesh part = soma::migrate(c, m, one_rank_up(m, rank_of, c.size()));
  EXPECT_EQ(part.adjacencies().bits(), 0u);
  EXPECT_EQ(c.allreduce(owned(part, 1), soma::Op::sum), c.allreduce(owned(m, 1), soma::Op::sum));
  EXPECT_GT(c.allreduce(owned(part, 1), soma::Op::sum), 0);
}

namespace {


std::set<GO> member_gids(const soma::HostMesh& m, GO set, int slot) {
  const LO s = *m.sets().index_of(set);
  std::set<GO> out;
  for (const LO l : m.sets().members_of(s, slot)) out.insert(slot == soma::set_dim ? m.sets().gid()(l) : m.gid(slot)(l));
  return out;
}

// Whether f throws std::invalid_argument saying `what`.
template <class F>
bool throws_saying(F&& f, const std::string& what) {
  try {
    f();
  } catch (const std::invalid_argument& e) {
    return std::string(e.what()).find(what) != std::string::npos;
  }
  return false;
}

}  // namespace

// Members follow their entities; the set list, flags and links stay; an
// ordered set lists each sending rank's entries in its order, ranks in order,
// with the set entries as rank 0 had them.
TEST(Migrate, SetsFollowTheirMembers) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  soma::HostMesh m = grid_with_extras(c, rank_of);
  add_sets(m);
  const soma::HostMesh part = soma::migrate(c, m, one_rank_up(m, rank_of, p));
  ASSERT_EQ(part.sets().count(), 3);
  EXPECT_TRUE(part.sets().ordered(*part.sets().index_of(set_b)));
  EXPECT_FALSE(part.sets().ordered(*part.sets().index_of(set_a)));

  std::set<GO> even, x0, edges;
  for (LO e = 0; e < part.count(2); ++e) {
    if (part.gid(2)(e) % 2 == 0) even.insert(part.gid(2)(e));
  }
  for (LO e = 0; e < part.count(0); ++e) {
    const GO g = part.gid(0)(e);
    if (g != free_vertex && (g - 1) % (nx + 1) == 0) x0.insert(g);
  }
  for (LO e = 0; e < part.count(1); ++e) edges.insert(part.gid(1)(e));
  EXPECT_EQ(member_gids(part, set_a, 2), even);
  EXPECT_EQ(member_gids(part, set_a, 0), x0);
  EXPECT_EQ(member_gids(part, set_c, 1), edges);
  EXPECT_EQ(member_gids(part, set_c, 0), me == 0 ? std::set<GO>{free_vertex} : std::set<GO>{});
  EXPECT_EQ(member_gids(part, set_c, soma::set_dim), std::set<GO>{set_a});
  const LO a = *part.sets().index_of(set_a), b = *part.sets().index_of(set_b);
  EXPECT_EQ(std::vector<LO>(part.sets().children(b).begin(), part.sets().children(b).end()), std::vector<LO>{a});
  EXPECT_EQ(std::vector<LO>(part.sets().parents(a).begin(), part.sets().parents(a).end()), std::vector<LO>{b});

  std::vector<std::pair<int, GO>> want;  // (slot, gid)
  for (int s = 0; s < p; ++s) {
    std::vector<GO> sent;
    for (const auto& [cell, r] : rank_of) {
      if (r == s && (r + 1) % p == me) sent.push_back(cell);
    }
    for (auto it = sent.rbegin(); it != sent.rend(); ++it) want.emplace_back(2, *it);
    if (s == 0) want.emplace_back(soma::set_dim, set_a);
  }
  std::vector<std::pair<int, GO>> got;
  for (const soma::SetEntry& x : part.sets().sequence(b)) {
    got.emplace_back(x.slot, x.slot == soma::set_dim ? part.sets().gid()(x.lid) : part.gid(x.slot)(x.lid));
  }
  EXPECT_EQ(got, want);
}

// A mesh mirrored without its edges holds no edge members; its sets keep
// their other members.
TEST(Migrate, SetsOfAMirrorWithoutEdgesKeepTheirOtherMembers) {
  const Comm c = Comm::world();
  if (c.size() != 1) GTEST_SKIP() << "a mirror holds no sharing";
  soma::HostMesh m = grid_with_extras(c, cell_ranks(c));
  add_sets(m);
  soma::TransferOptions o;
  o.dims = {true, false, true, true};
  const soma::HostMesh lean = m.mirror<Kokkos::HostSpace>(o);
  ASSERT_FALSE(lean.sets().has_members(1));
  const soma::HostMesh part = soma::migrate(c, lean, std::vector<int>(static_cast<std::size_t>(lean.count(2)), 0));
  EXPECT_EQ(member_gids(part, set_a, 2), member_gids(m, set_a, 2));
  EXPECT_EQ(member_gids(part, set_a, 0), member_gids(m, set_a, 0));
  EXPECT_EQ(member_gids(part, set_c, 0), member_gids(m, set_c, 0));
}

// Two sets, 100 and 101, that the last rank changes one way at a time.
TEST(Migrate, SetsThatDifferBetweenRanksThrowOnEveryRank) {
  const Comm c = Comm::world();
  if (c.size() == 1) GTEST_SKIP() << "needs two ranks";
  const std::map<GO, int> rank_of = cell_ranks(c);
  const bool last = c.rank() == c.size() - 1;
  for (int change = 0; change < 5; ++change) {
    SCOPED_TRACE(change);
    soma::HostMesh m = grid_with_extras(c, rank_of);
    soma::SetSpec x, y;
    x.gid = last && change == 0 ? 200 : 100;
    y.gid = 101;
    if (last && change == 1) y.flags = soma::set_ordered;
    if (!(last && change == 2)) x.children = {101};
    if (!(last && change == 3)) y.parents = {x.gid};
    if (last && change == 4) x.contents.push_back({soma::set_dim, 101});
    soma::SetBuilder sb;
    sb.add(std::move(x));
    sb.add(std::move(y));
    std::move(sb).install(m);
    EXPECT_TRUE(throws_saying([&] { (void)soma::migrate(c, m, one_rank_up(m, rank_of, c.size())); }, "sets differ"));
  }
}

// Members of a dimension only derived follow their entities: the edges of
// even lowest corner gid, an unordered set, and the same edges in an ordered
// one (by decreasing lowest corner on each rank, copies too, which only their
// owners send).
TEST(Migrate, SetsOfDerivedEntitiesFollowThem) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, me));
  soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
  const auto low = [](const soma::HostMesh& x, LO e) {
    const std::vector<GO> ends = nodes_of(x, 1, e);
    return std::min(ends[0], ends[1]);
  };
  const auto chosen = [&](const soma::HostMesh& x) {
    std::vector<std::pair<GO, LO>> out;  // (lowest corner, lid), decreasing
    for (LO e = 0; e < x.count(1); ++e) {
      if (low(x, e) % 2 == 0) out.emplace_back(low(x, e), e);
    }
    std::sort(out.rbegin(), out.rend());
    return out;
  };
  soma::SetSpec u, o;
  u.gid = 300;
  o.gid = 301;
  o.flags = soma::set_ordered;
  for (const auto& [g, e] : chosen(m)) {
    u.contents.push_back({1, e});
    o.contents.push_back({1, e});
  }
  soma::SetBuilder sb;
  sb.add(std::move(u));
  sb.add(std::move(o));
  std::move(sb).install(m);

  const soma::HostMesh part = soma::migrate(c, m, one_rank_up(m, rank_of, p));
  ASSERT_FALSE(part.input(1));
  std::set<GO> want;
  for (const auto& [g, e] : chosen(part)) want.insert(part.gid(1)(e));
  EXPECT_EQ(member_gids(part, 300, 1), want);
  const auto seq = part.sets().sequence(*part.sets().index_of(301));
  std::set<GO> in_order;
  for (const soma::SetEntry& x : seq) {
    ASSERT_EQ(x.slot, 1);
    in_order.insert(part.gid(1)(x.lid));
  }
  EXPECT_EQ(in_order, want);
  EXPECT_EQ(seq.size(), want.size());
}

namespace {

// The lowest corner gid of entity (d, e).




}  // namespace

// Fields keep their options, layout and tag format; each entity's values
// follow it; a domain over every entity stays full, one over some holds the
// members that arrived.
TEST(Migrate, FieldsFollowTheirEntities) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, me));
  soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
  add_fields(m, me);
  const soma::HostMesh part = soma::migrate(c, m, one_rank_up(m, rank_of, p));
  const auto& reg = part.fields();
  const auto members = [&](const auto& f, int d) {
    std::vector<GO> out;
    for (LO e = 0; e < part.count(d); ++e) {
      if (rows_of(f, d, e).valid()) out.push_back(part.gid(d)(e));
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  const auto every = [&](int d, auto keep) {
    std::vector<GO> out;
    for (LO e = 0; e < part.count(d); ++e) {
      if (keep(part.gid(d)(e))) out.push_back(part.gid(d)(e));
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  const auto all = [](GO) { return true; };

  const auto temp = reg.get<double>("temp");
  EXPECT_EQ(temp.default_value(), (std::vector<double>{-1, -2}));
  EXPECT_EQ(temp.global_value(), (std::vector<double>{7, 8}));
  EXPECT_EQ(temp.domain().info(0).kind, soma::DomainKind::full);
  EXPECT_EQ(reg.tag_format(reg.family<double>("temp", 2)), (soma::TagFormat{2, soma::TagFormat::Kind::plain, 0}));
  for (LO e = 0; e < part.count(0); ++e) {
    EXPECT_EQ(temp.value_or_default(0, e, 0), static_cast<double>(part.gid(0)(e)));
    EXPECT_EQ(temp.value_or_default(0, e, 1), 2.0 * static_cast<double>(part.gid(0)(e)));
  }

  const auto flag = reg.get<std::int32_t>("flag");
  EXPECT_FALSE(flag.has_default());
  EXPECT_EQ(members(flag, 2), every(2, [](GO g) { return g % 2 == 0; }));
  for (LO e = 0; e < part.count(2); ++e) {
    if (part.gid(2)(e) % 2 == 0) EXPECT_EQ(flag.value_or_default(2, e), 3 * part.gid(2)(e));
  }

  const auto low = reg.get<float>("low");
  EXPECT_EQ(low.domain().info(0).kind, soma::DomainKind::full);
  for (LO e = 0; e < part.count(1); ++e) EXPECT_EQ(low.value_or_default(1, e), static_cast<float>(low_corner(part, 1, e)));

  const auto q = reg.get<double>("q");
  EXPECT_EQ(q.space().layout(), soma::Layout::element_nodes(soma::NodeFamily::gll, 2));
  for (LO e = 0; e < part.count(2); ++e) {
    const auto r = rows_of(q, 2, e);
    ASSERT_EQ(r.end - r.begin, 9);
    for (LO k = r.begin; k < r.end; ++k) EXPECT_EQ(q.view()(k, 0), 100.0 * static_cast<double>(part.gid(2)(e)) + static_cast<double>(k - r.begin));
  }

  const auto hist = reg.get_var<std::int64_t>("hist");
  EXPECT_TRUE(hist.has_default());
  EXPECT_EQ(members(hist, 0), every(0, [](GO g) { return g % 3 == 0; }));
  for (LO e = 0; e < part.count(0); ++e) {
    const GO g = part.gid(0)(e);
    if (g % 3 != 0) continue;
    const LO r = hist.domain().row(0, e);
    std::vector<std::int64_t> want, got;
    for (GO k = 0; k < g % 4; ++k) want.push_back(g + k);
    for (auto k = hist.offsets()(r); k < hist.offsets()(r + 1); ++k) got.push_back(hist.values()(k));
    EXPECT_EQ(got, want);
  }

  const auto ref = reg.get<GO>("ref");
  EXPECT_TRUE(reg.info(ref.id()).handle);
  for (LO e = 0; e < part.count(2); ++e) EXPECT_EQ(ref.value_or_default(2, e), soma::make_handle(0, low_corner(part, 2, e)));

  const auto multi = reg.get<std::int32_t>("multi");
  EXPECT_EQ(members(multi, 0), every(0, all));
  EXPECT_EQ(members(multi, 2), every(2, [](GO g) { return g % 3 == 0; }));
  for (LO e = 0; e < part.count(0); ++e) EXPECT_EQ(multi.value_or_default(0, e), part.gid(0)(e) + 1);
  for (LO e = 0; e < part.count(2); ++e) {
    if (part.gid(2)(e) % 3 == 0) EXPECT_EQ(multi.value_or_default(2, e), part.gid(2)(e) + 1);
  }

  // Full on rank 0 only: a list everywhere after.
  const auto some = reg.get<std::int32_t>("some");
  EXPECT_EQ(members(some, 2), every(2, [&](GO g) { return g % 2 == 0 || rank_of.at(g) == 0; }));

  const auto on_sets = reg.get<std::int32_t>("on_sets");
  EXPECT_EQ(on_sets.value_or_default(soma::set_dim, *part.sets().index_of(field_set)), 11);
  const auto var_on_sets = reg.get_var<std::int32_t>("var_on_sets").value_or_default(soma::set_dim, *part.sets().index_of(field_set));
  EXPECT_EQ(std::vector<std::int32_t>(var_on_sets.begin(), var_on_sets.end()), (std::vector<std::int32_t>{1, 2, 3}));
}

// A field or mesh value the last rank declares differently: components,
// default, global value, name, mesh value, dimension, tag format, values on
// sets, layout, value type.
TEST(Migrate, FieldsThatDifferBetweenRanksThrowOnEveryRank) {
  const Comm c = Comm::world();
  if (c.size() == 1) GTEST_SKIP() << "needs two ranks";
  const std::map<GO, int> rank_of = cell_ranks(c);
  const bool last = c.rank() == c.size() - 1;
  for (int change = 0; change < 10; ++change) {
    SCOPED_TRACE(change);
    soma::HostMesh m = grid_with_extras(c, rank_of);
    auto& reg = m.fields();
    soma::FieldOptions<double> o;
    if (last && change == 1) o.default_value = std::vector<double>{1};
    if (last && change == 2) o.global_value = std::vector<double>{1};
    const int dim = last && change == 5 ? 2 : 0;
    if (last && change == 9) {
      (void)reg.add_tag<float>("x", dim, 1);
    } else {
      (void)reg.add_tag<double>(last && change == 3 ? "other" : "x", dim, last && change == 0 ? 2 : 1, o);
    }
    if (last && change == 6) reg.set_tag_format(reg.family<double>("x", 1), soma::TagFormat{1, soma::TagFormat::Kind::plain, 0});
    if (change == 7) {
      soma::SetSpec s;
      s.gid = field_set;
      soma::SetBuilder sb;
      sb.add(std::move(s));
      std::move(sb).install(m);
      reg.add_tag<std::int32_t>("on_sets", soma::set_dim, 1).view()(0, 0) = last ? 1 : 0;
    }
    if (change == 8) {
      const auto l = soma::Layout::element_nodes(soma::NodeFamily::gll, last ? 2 : 1);
      (void)reg.add<double>("y", soma::make_space(m, soma::Domain<Kokkos::HostSpace>::all(m, 2), l), 1);
    }
    soma::MeshValue v;
    v.type = soma::DataType::i32;
    v.default_value = soma::MeshValue::bytes_of<std::int32_t>(std::vector<std::int32_t>{last && change == 4 ? 1 : 0});
    m.values().set("answer", v);
    EXPECT_TRUE(throws_saying([&] { (void)soma::migrate(c, m, one_rank_up(m, rank_of, c.size())); },
                              "fields or mesh values differ"));
  }
}
