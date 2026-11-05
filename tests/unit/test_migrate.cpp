// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "distributed_fixture.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_edit.hpp"
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

constexpr GO diagonal = 6000;  // an explicit edge from the grid's first corner to its last
constexpr GO free_vertex = 9999;

std::vector<int> sorted_ranks(std::span<const int> r) {
  std::vector<int> v(r.begin(), r.end());
  std::sort(v.begin(), v.end());
  return v;
}

// The grid with explicit edges on x = 6 (gids 5000 + j) on every rank
// holding both their corners, the diagonal edge on the rank holding the
// first cell, and a free vertex on rank 0.
soma::HostMesh grid_with_extras(const Comm& c, const std::map<GO, int>& rank_of) {
  const int me = c.rank();
  const std::map<GO, std::set<int>> users = vertex_users(rank_of);
  soma::MeshBuilder b = local_builder(rank_of, me);
  std::vector<GO> eg, ev;
  for (int j = 0; j < ny; ++j) {
    const std::pair<GO, GO> e{vertex(6, j), vertex(6, j + 1)};
    if (!users.at(e.first).contains(me) || !users.at(e.second).contains(me)) continue;
    eg.push_back(5000 + j);
    ev.insert(ev.end(), {e.first, e.second});
  }
  if (rank_of.at(1000) == me) {
    if (rank_of.at(1000 + nx * ny - 1) != me) {  // the last corner is not here yet
      b.add_vertices(std::vector<GO>{vertex(nx, ny)}, std::vector<soma::Real>{nx, ny, 0});
    }
    eg.push_back(diagonal);
    ev.insert(ev.end(), {vertex(0, 0), vertex(nx, ny)});
  }
  b.add_elements(soma::Topology::edge, eg, ev);
  if (me == 0) b.add_vertices(std::vector<GO>{free_vertex}, std::vector<soma::Real>{-1, -1, 0});
  return soma::build_local_part(c, std::move(b));
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

// The vertex gids of entity e of dimension d, corners then nodes.
std::vector<GO> nodes_of(const soma::HostMesh& m, int d, LO e) {
  const auto v = m.view();
  const auto g = m.gid(0);
  std::vector<GO> out;
  for (LO k = 0; k < v.num_nodes(d, e); ++k) out.push_back(g(v.node(d, e, k)));
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
