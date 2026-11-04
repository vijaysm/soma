// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mesh_fixture.hpp"
#include "mesh_text.hpp"
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
using soma::MeshEdit;
using soma::Real;
using soma::Topology;

namespace {

using soma_test::lid_of;

template <class S>
std::vector<LO> vec(S span) {
  return {span.begin(), span.end()};
}

std::string what_of(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

std::vector<GO> vertex_gids(const HostMesh& m, int d, LO e) {
  const auto v = m.view();
  std::vector<GO> out;
  for (LO k = 0; k < v.num_nodes(d, e); ++k) out.push_back(v.gid(0, v.node(d, e, k)));
  return out;
}

// Each live row of a graph by the gid of its entity: the gids of its targets.
std::map<GO, std::vector<GO>> graph_by_gid(const HostMesh& m, int from, int to) {
  std::map<GO, std::vector<GO>> out;
  const auto g = m.adjacency(from, to);
  const auto v = m.view();
  for (LO e = 0; e < m.count(from); ++e) {
    if (!v.alive(from, e)) continue;
    auto& row = out[m.gid(from)(static_cast<std::size_t>(e))];
    for (auto k = g.offsets(static_cast<std::size_t>(e)); k < g.offsets(static_cast<std::size_t>(e) + 1); ++k) {
      row.push_back(m.gid(to)(static_cast<std::size_t>(g.targets(static_cast<std::size_t>(k)))));
    }
    std::sort(row.begin(), row.end());
  }
  return out;
}

}  // namespace

// The fixture (vertices 10..70, quads 7 and 3, tri 5) with a quad on its left:
// two new vertices and a new quad follow the existing entities, with the next
// gids, in one tail segment each.
TEST(MeshEdit, AddedEntitiesFollowTheOthers) {
  HostMesh m = soma_test::two_quads_and_a_triangle();
  MeshEdit edit(m);
  const LO v = edit.add_vertices(std::vector<Real>{-1, 0, 0, -1, 1, 0});
  EXPECT_EQ(v, 7);
  const LO q = edit.add_elements(Topology::quad, std::vector<LO>{v, lid_of(m, 0, 10), lid_of(m, 0, 40), v + 1});
  EXPECT_EQ(q, 3);
  EXPECT_EQ(m.count(2), 3);  // nothing visible before apply
  const soma::EditReport r = edit.apply();
  EXPECT_EQ(r.added[0], 2);
  EXPECT_EQ(r.added[2], 1);
  EXPECT_EQ(m.count(0), 9);
  EXPECT_EQ(m.count(2), 4);
  EXPECT_EQ(m.tail(0), 2);
  EXPECT_EQ(m.tail(2), 1);
  EXPECT_TRUE(m.edited());
  EXPECT_EQ(m.num_live(2), 4);
  EXPECT_EQ(m.gid(0)(7), 71);
  EXPECT_EQ(m.gid(0)(8), 72);
  EXPECT_EQ(m.gid(2)(3), 8);
  EXPECT_EQ(vertex_gids(m, 2, 3), (std::vector<GO>{71, 10, 40, 72}));
  EXPECT_EQ(m.coords()(8, 0), -1.0);
  EXPECT_EQ(m.coords()(8, 1), 1.0);
  const auto segs = m.segments(2);
  ASSERT_EQ(segs.size(), 3u);
  EXPECT_EQ(segs.back(), (soma::Segment{3, 4, Topology::quad, 4, soma::Ownership::interior}));
  EXPECT_EQ(m.segment_table(2).count, 4);
  // Existing entities keep their lids.
  EXPECT_EQ(m.gid(2)(0), 5);
  EXPECT_EQ(m.gid(0)(0), 10);
}

// Removed entities stay in place, masked; a vertex a live element uses cannot
// go unless the element goes too.
TEST(MeshEdit, RemovedEntitiesAreMasked) {
  HostMesh m = soma_test::two_quads_and_a_triangle();
  const LO tri = lid_of(m, 2, 5);
  const LO v70 = lid_of(m, 0, 70);
  const LO v30 = lid_of(m, 0, 30);
  {
    MeshEdit edit(m);
    edit.remove(0, std::vector<LO>{v70});
    EXPECT_EQ(what_of([&] { edit.apply(); }),
              "MeshEdit: vertex gid 70 cannot be removed: element gid 5 of dimension 2 uses it");
    EXPECT_FALSE(m.edited());
    edit.remove(2, std::vector<LO>{tri});
    edit.apply();
  }
  EXPECT_EQ(m.count(2), 3);
  EXPECT_EQ(m.num_live(2), 2);
  EXPECT_EQ(m.num_live(0), 6);
  EXPECT_FALSE(m.view().alive(2, tri));
  EXPECT_FALSE(m.view().alive(0, v70));
  EXPECT_TRUE(m.view().alive(0, v30));
  MeshEdit again(m);
  EXPECT_EQ(what_of([&] { again.remove(2, std::vector<LO>{tri}); }),
            "MeshEdit: remove: entity " + std::to_string(tri) + " of dimension 2 is already removed");
  EXPECT_EQ(what_of([&] { again.remove(2, std::vector<LO>{3}); }),
            "MeshEdit: remove: entity 3 of dimension 2 is not in [0, 3)");
  // An added element cannot use a removed vertex.
  again.add_elements(Topology::tri, std::vector<LO>{v30, v70, lid_of(m, 0, 60)});
  EXPECT_EQ(what_of([&] { again.apply(); }), "MeshEdit: an added element uses vertex gid 70, which is removed");
}

TEST(MeshEdit, InputIsCheckedAsItIsGiven) {
  HostMesh m = soma_test::two_quads_and_a_triangle();
  MeshEdit edit(m);
  EXPECT_THROW(edit.add_vertices(std::vector<Real>{0, 0}), std::invalid_argument);
  EXPECT_THROW(edit.add_vertices(std::vector<Real>{0, 0, std::nan("")}), std::invalid_argument);
  EXPECT_THROW(edit.add_elements(Topology::quad, std::vector<LO>{0, 1, 2, 7}), std::invalid_argument);
  EXPECT_THROW(edit.add_elements(Topology::quad, std::vector<LO>{0, 1, 2, 3, 4, 5}, 6), std::invalid_argument);
  EXPECT_THROW(edit.add_elements(Topology::quad, std::vector<LO>{0, 1, 2, 3, 4, 5, 6}, 7), std::invalid_argument);
  EXPECT_THROW(edit.add_elements(Topology::hex, std::vector<LO>(8, 0)), std::invalid_argument);
  EXPECT_THROW(edit.add_elements(Topology::polygon, std::vector<LO>{0, 1, 2}), std::invalid_argument);
  EXPECT_THROW(edit.add_polygons(std::vector<soma::Offset>{0, 2}, std::vector<LO>{0, 1}), std::invalid_argument);
  EXPECT_THROW(edit.remove(3, std::vector<LO>{0}), std::invalid_argument);
  EXPECT_EQ(edit.apply().added[2], 0);
  EXPECT_FALSE(m.edited());
}

// Given gids must be new to their dimension; a failed apply changes nothing
// and keeps the edit, which applies once the clash is gone.
TEST(MeshEdit, GivenGidsMustBeNew) {
  HostMesh m = soma_test::two_quads_and_a_triangle();
  MeshEdit edit(m);
  edit.add_vertices(std::vector<GO>{70}, std::vector<Real>{5, 5, 0});
  EXPECT_EQ(what_of([&] { edit.apply(); }), "MeshEdit: gid 70 is already used in dimension 0");
  EXPECT_EQ(m.count(0), 7);
  MeshEdit ok(m);
  ok.add_vertices(std::vector<GO>{100}, std::vector<Real>{5, 5, 0});
  ok.add_vertices(std::vector<Real>{6, 6, 0});
  ok.add_polygons(std::vector<soma::Offset>{0, 3}, std::vector<LO>{7, 8, 0});
  ok.apply();
  EXPECT_EQ(m.gid(0)(7), 100);
  EXPECT_EQ(m.gid(0)(8), 101);  // after the largest, given ones included
  EXPECT_EQ(static_cast<Topology>(m.topology(2)(3)), Topology::polygon);
  EXPECT_EQ(m.gid(2)(3), 8);
}

// Fields over every entity grow, the added ones at the default; subsets keep
// their members; values stay; the epoch does not move. Removed entities leave
// sets. Graphs touching edited dimensions go, and sides are no longer complete.
TEST(MeshEdit, FieldsSetsAndGraphsFollow) {
  HostMesh m = soma::testing::box_mesh({2, 1, 1, Topology::quad});
  soma::build_adjacencies(m, soma::AdjacencyRequest{}.add(2, 1).add(0, 2));
  const auto epoch = m.epoch();
  auto all = m.fields().add<double>("a", soma::make_space(m, soma::Domain<Kokkos::HostSpace>::all(m, 0),
                                                         soma::Layout::per_entity()),
                                    1, {.default_value = std::vector<double>{-5}});
  for (std::size_t i = 0; i < 6; ++i) all.view()(i, 0) = static_cast<double>(10 * i);
  auto some = m.fields().add<std::int32_t>(
      "s", soma::make_space(m, soma::Domain<Kokkos::HostSpace>::from_list(m, 2, std::vector<LO>{1}),
                            soma::Layout::per_entity()),
      1);
  some.view()(0, 0) = 3;
  soma::SetBuilder sb;
  sb.add(soma::SetSpec{1, soma::set_unique, {{2, 0}, {2, 1}}, {}, {}});
  std::move(sb).install(m);
  ASSERT_TRUE(m.has_adjacency(2, 1));

  MeshEdit edit(m);
  const LO v = edit.add_vertices(std::vector<Real>{3, 0, 0, 3, 1, 0});
  edit.add_elements(Topology::quad, std::vector<LO>{2, v, v + 1, 5});
  edit.remove(2, std::vector<LO>{0});
  edit.apply();

  EXPECT_EQ(m.epoch(), epoch);
  const auto a = m.fields().get<double>("a");
  EXPECT_EQ(a.rows(), 8);
  EXPECT_DOUBLE_EQ(a.view()(5, 0), 50.0);
  EXPECT_DOUBLE_EQ(a.view()(6, 0), -5.0);
  EXPECT_DOUBLE_EQ(a.view()(7, 0), -5.0);
  const auto s = m.fields().get<std::int32_t>("s");
  EXPECT_EQ(s.domain().members(0), (std::vector<LO>{1}));
  EXPECT_EQ(s.view()(0, 0), 3);
  EXPECT_EQ(vec(m.sets().members_of(0, 2)), (std::vector<LO>{1}));
  EXPECT_EQ(m.sets().entities(2), 3);
  // Held graphs are built again over the live entities: the edges of the
  // removed quad that no live quad has go, the new quad's are added, the
  // shared edge keeps its gid; the graphs agree with those of the compressed
  // mesh, by gid.
  ASSERT_TRUE(m.has_adjacency(2, 1));
  ASSERT_TRUE(m.has_adjacency(0, 2));
  EXPECT_TRUE(soma::detail::TopologyAccess::block(m, 1).derived);
  EXPECT_EQ(m.count(1), 7);
  EXPECT_EQ(m.adjacency(2, 1).offsets(1) - m.adjacency(2, 1).offsets(0), 0);  // the removed quad
  HostMesh c = m.clone();
  c.compress();
  EXPECT_EQ(graph_by_gid(m, 2, 1), graph_by_gid(c, 2, 1));
  EXPECT_EQ(graph_by_gid(m, 0, 2), graph_by_gid(c, 0, 2));
  EXPECT_EQ(graph_by_gid(m, 0, 2).at(m.gid(0)(2)).size(), 2u);
  EXPECT_EQ(m.epoch(), epoch);
}

// Additions alone leave every entity live, so graphs build as on any mesh.
TEST(MeshEdit, GraphsBuildOnAMeshWithAdditionsOnly) {
  HostMesh m = soma::testing::box_mesh({2, 1, 1, Topology::quad});
  MeshEdit edit(m);
  const LO v = edit.add_vertices(std::vector<Real>{3, 0, 0, 3, 1, 0});
  edit.add_elements(Topology::quad, std::vector<LO>{2, v, v + 1, 5});
  edit.apply();
  soma::build_adjacencies(m, soma::AdjacencyRequest{}.add(2, 1).add(1, 2));
  EXPECT_EQ(m.count(1), 10);
  EXPECT_EQ(m.adjacency(2, 1).offsets(3) - m.adjacency(2, 1).offsets(2), 4);
}

// compress() gives the mesh MeshBuilder builds from the live entities, in the
// mesh's ordering; fields keep their values by gid and stay current; sets and
// held graphs follow; the epoch advances.
TEST(MeshEdit, CompressGivesTheMeshBuiltFromTheLiveEntities) {
  for (const auto ordering : {soma::Ordering::file, soma::Ordering::type_major_sfc}) {
    soma::MeshBuilder b0(2);
    std::vector<GO> vg;
    std::vector<Real> xyz;
    for (GO j = 0; j < 3; ++j) {
      for (GO i = 0; i < 4; ++i) {
        vg.push_back(1 + i + 4 * j);
        xyz.insert(xyz.end(), {static_cast<Real>(i), static_cast<Real>(j), 0});
      }
    }
    b0.add_vertices(vg, xyz);
    std::vector<GO> eg;
    std::vector<GO> conn;
    for (GO j = 0; j < 2; ++j) {
      for (GO i = 0; i < 3; ++i) {
        eg.push_back(1 + i + 3 * j);
        const GO a = 1 + i + 4 * j;
        conn.insert(conn.end(), {a, a + 1, a + 5, a + 4});
      }
    }
    b0.add_elements(Topology::quad, eg, conn);
    HostMesh m = std::move(b0).build(ordering);
    soma::build_adjacencies(m, soma::AdjacencyRequest{}.add(0, 2));
    auto f = m.fields().add<double>("x", soma::make_space(m, soma::Domain<Kokkos::HostSpace>::all(m, 2),
                                                           soma::Layout::per_entity()),
                                    1, {.default_value = std::vector<double>{-1}});
    for (LO e = 0; e < 6; ++e) f.view()(static_cast<std::size_t>(e), 0) = static_cast<double>(m.gid(2)(static_cast<std::size_t>(e)));
    soma::SetBuilder sb;
    sb.add(soma::SetSpec{9, soma::set_ordered, {{2, lid_of(m, 2, 6)}, {2, lid_of(m, 2, 2)}, {0, lid_of(m, 0, 12)}}, {}, {}});
    std::move(sb).install(m);
    const auto epoch = m.epoch();

    // Remove element 2 and vertex... none; add a triangle on top, gid 7.
    MeshEdit edit(m);
    const LO top = edit.add_vertices(std::vector<Real>{1.5, 3, 0});
    edit.add_elements(Topology::tri, std::vector<LO>{lid_of(m, 0, 10), lid_of(m, 0, 11), top});
    edit.remove(2, std::vector<LO>{lid_of(m, 2, 2)});
    edit.apply();
    const auto old_gid2 = std::vector<GO>(m.gid(2).data(), m.gid(2).data() + m.count(2));
    const soma::Renumbering r = m.compress();

    EXPECT_FALSE(m.edited());
    EXPECT_EQ(m.epoch(), epoch + 1);
    EXPECT_EQ(m.ordering(), ordering);
    // The same mesh as built from scratch, lid for lid.
    soma::MeshBuilder b1(2);
    vg.push_back(13);
    xyz.insert(xyz.end(), {1.5, 3, 0});
    b1.add_vertices(vg, xyz);
    std::vector<GO> eg1;
    std::vector<GO> conn1;
    for (std::size_t e = 0; e < eg.size(); ++e) {
      if (eg[e] == 2) continue;
      eg1.push_back(eg[e]);
      conn1.insert(conn1.end(), conn.begin() + static_cast<std::ptrdiff_t>(4 * e),
                   conn.begin() + static_cast<std::ptrdiff_t>(4 * e + 4));
    }
    b1.add_elements(Topology::quad, eg1, conn1);
    b1.add_elements(Topology::tri, std::vector<GO>{7}, std::vector<GO>{10, 11, 13});
    const HostMesh want = std::move(b1).build(ordering);
    ASSERT_EQ(m.count(2), want.count(2));
    for (int d : {0, 2}) {
      for (LO e = 0; e < m.count(d); ++e) {
        EXPECT_EQ(m.gid(d)(static_cast<std::size_t>(e)), want.gid(d)(static_cast<std::size_t>(e))) << d << " " << e;
      }
    }
    EXPECT_EQ(soma_test::entities(m, 2), soma_test::entities(want, 2));
    EXPECT_EQ(soma_test::coordinates(m), soma_test::coordinates(want));
    EXPECT_EQ(m.segments(2).size(), want.segments(2).size());
    // The renumbering names each survivor's new lid.
    for (std::size_t l = 0; l < old_gid2.size(); ++l) {
      const LO n = r.new_lid[2][l];
      if (old_gid2[l] == 2) {
        EXPECT_EQ(n, soma::invalid_lo);
      } else {
        EXPECT_EQ(m.gid(2)(static_cast<std::size_t>(n)), old_gid2[l]);
      }
    }
    // Fields by gid, the new element at the default.
    const auto g = m.fields().get<double>("x");
    for (LO e = 0; e < m.count(2); ++e) {
      const GO gid = m.gid(2)(static_cast<std::size_t>(e));
      EXPECT_DOUBLE_EQ(g.view()(static_cast<std::size_t>(e), 0), gid == 7 ? -1.0 : static_cast<double>(gid));
    }
    // The ordered set without element 2, its order kept.
    std::vector<std::pair<int, GO>> seq;
    for (const auto& e : m.sets().sequence(0)) seq.emplace_back(e.slot, m.gid(e.slot)(static_cast<std::size_t>(e.lid)));
    EXPECT_EQ(seq, (std::vector<std::pair<int, GO>>{{2, 6}, {0, 12}}));
    // The held graph, rebuilt.
    ASSERT_TRUE(m.has_adjacency(0, 2));
    const auto adj = m.adjacency(0, 2);
    const LO v10 = lid_of(m, 0, 10);
    EXPECT_EQ(adj.offsets(static_cast<std::size_t>(v10) + 1) - adj.offsets(static_cast<std::size_t>(v10)), 3);
    // Compressing again does nothing.
    EXPECT_TRUE(m.compress().new_lid[0].empty());
    EXPECT_EQ(m.epoch(), epoch + 1);
  }
}

// Removing elements and a vertex they alone used, then compressing, drops
// them; a high-order element added with its nodes puts its node-only vertices
// after the corners again.
TEST(MeshEdit, CompressDropsRemovedEntitiesAndOrdersNodes) {
  HostMesh m = soma::testing::box_mesh({2, 1, 1, Topology::quad});
  MeshEdit edit(m);
  // Quad 1 (vertices 1 2 5 4) goes with vertex 2, which it alone uses.
  const LO quad = lid_of(m, 2, m.gid(2)(1));
  const auto corners = vertex_gids(m, 2, quad);
  edit.remove(2, std::vector<LO>{quad});
  edit.remove(0, std::vector<LO>{lid_of(m, 0, corners[1])});
  // A quad8 on the left: corners 0, 3 and two new vertices; four new nodes.
  const LO v = edit.add_vertices(std::vector<Real>{-1, 0, 0, -1, 1, 0, -0.5, 0, 0, -0.5, 1, 0, -1, 0.5, 0, 0, 0.5, 0});
  edit.add_elements(Topology::quad,
                    std::vector<LO>{v, lid_of(m, 0, corners[0]), lid_of(m, 0, corners[3]), v + 1, v + 2, v + 5, v + 3, v + 4},
                    8);
  edit.apply();
  m.compress();
  EXPECT_EQ(m.count(2), 2);
  EXPECT_EQ(m.count(0), 6 - 1 + 6);
  EXPECT_EQ(m.corners(), 7);
  EXPECT_TRUE(m.has_nodes(2));
  EXPECT_EQ(soma_test::lid_of(m, 0, corners[1]), soma::invalid_lo);
}

// Two hexes; the first goes. Its faces that the second lacks go with it, the
// shared face keeps its gid, lid aside; fields and sets on faces follow by
// gid, and graphs not held before build over the live entities.
TEST(MeshEdit, SidesDerivedAgainKeepFieldsAndSetsByGid) {
  HostMesh m = soma::testing::box_mesh({2, 1, 1, Topology::hex});
  soma::build_adjacencies(m, soma::AdjacencyRequest{}.add(3, 2));
  ASSERT_EQ(m.count(2), 11);
  // The face both hexes have.
  const auto down = m.adjacency(3, 2);
  GO shared = -1;
  for (auto i = down.offsets(0); i < down.offsets(1); ++i) {
    for (auto j = down.offsets(1); j < down.offsets(2); ++j) {
      if (down.targets(static_cast<std::size_t>(i)) == down.targets(static_cast<std::size_t>(j))) {
        shared = m.gid(2)(static_cast<std::size_t>(down.targets(static_cast<std::size_t>(i))));
      }
    }
  }
  ASSERT_GE(shared, 0);
  auto f = m.fields().add<double>("face", soma::make_space(m, soma::Domain<Kokkos::HostSpace>::all(m, 2),
                                                           soma::Layout::per_entity()),
                                  1);
  for (LO e = 0; e < 11; ++e) f.view()(static_cast<std::size_t>(e), 0) = static_cast<double>(m.gid(2)(static_cast<std::size_t>(e)));
  soma::SetBuilder sb;
  sb.add(soma::SetSpec{1, soma::set_unique, {{2, 0}, {2, lid_of(m, 2, shared)}}, {}, {}});
  std::move(sb).install(m);
  const auto epoch = m.epoch();
  const GO first = m.gid(2)(0);

  MeshEdit edit(m);
  edit.remove(3, std::vector<LO>{0});
  edit.apply();
  EXPECT_EQ(m.epoch(), epoch);
  EXPECT_EQ(m.count(2), 6);
  const LO s = lid_of(m, 2, shared);
  ASSERT_GE(s, 0);
  const auto g = m.fields().get<double>("face");
  EXPECT_EQ(g.rows(), 6);
  for (LO e = 0; e < 6; ++e) {
    EXPECT_DOUBLE_EQ(g.view()(static_cast<std::size_t>(e), 0), static_cast<double>(m.gid(2)(static_cast<std::size_t>(e))));
  }
  const bool first_kept = lid_of(m, 2, first) >= 0;
  std::vector<LO> want{s};
  if (first_kept) want.push_back(lid_of(m, 2, first));
  std::sort(want.begin(), want.end());
  EXPECT_EQ(vec(m.sets().members_of(0, 2)), want);
  soma::build_adjacencies(m, soma::AdjacencyRequest{}.add(2, 3).add(0, 3));
  EXPECT_EQ(graph_by_gid(m, 2, 3).at(shared), (std::vector<GO>{m.gid(3)(1)}));
  EXPECT_EQ(graph_by_gid(m, 0, 3).at(m.gid(0)(0)), (std::vector<GO>{}));
  EXPECT_EQ(m.adjacency(0, 3).offsets(1), 0);
}
