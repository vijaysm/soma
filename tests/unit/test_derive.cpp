// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "adjacency_check.hpp"
#include "derive_check.hpp"
#include "soma/core/mesh.hpp"
#include "soma/topology/derive.hpp"
#include "soma/topology/orientation.hpp"
#include "soma/topology/transpose.hpp"
#include "topology_meshes.hpp"

using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::Topology;

using DevExec = Kokkos::DefaultExecutionSpace;
using DevMem = DevExec::memory_space;
using HostExec = Kokkos::DefaultHostExecutionSpace;

namespace soma_test {

// Everything derive produces, on the host.
struct DerivedHost {
  LO count = 0;
  std::vector<Offset> offsets;
  std::vector<LO> conn;
  std::vector<std::uint8_t> topology;
  std::vector<GO> gid;
  std::vector<soma::Segment> segments;
  HostGraph down;
  friend bool operator==(const DerivedHost&, const DerivedHost&) = default;
};

template <class Exec>
DerivedHost derive_on(const soma::HostMesh& host, int k, const Exec& exec) {
  using M = typename Exec::memory_space;
  const auto mesh = host.mirror<M>();
  const int d = mesh.dim();
  const auto up = soma::topology::transpose(exec, mesh.conn_offsets(d), mesh.conn(d),
                                            mesh.count(0), "test");
  const auto r = soma::topology::derive(exec, mesh, k, up);
  return {r.block.count,         to_vector(r.block.conn_offsets), to_vector(r.block.conn),
          to_vector(r.block.topology), to_vector(r.block.gid), r.block.segments.segments,
          to_host(r.down)};
}

Tuple entity(const DerivedHost& h, LO id) {
  return {h.conn.begin() + h.offsets[static_cast<std::size_t>(id)],
          h.conn.begin() + h.offsets[static_cast<std::size_t>(id) + 1]};
}

// Checks h against the independent expectation: the set of entities, every
// incidence's vertex set and code, and the creator connectivity.
void check_against_expected(const soma::HostMesh& mesh, int k, const DerivedHost& h) {
  const Expected ex = expected_sides(mesh, k);
  ASSERT_EQ(static_cast<std::size_t>(h.count), ex.unique.size());
  std::set<Tuple> got;
  for (LO id = 0; id < h.count; ++id) got.insert(sorted(entity(h, id)));
  ASSERT_EQ(got, ex.unique);

  const int d = mesh.dim();
  const auto off = to_vector(mesh.conn_offsets(d));
  const auto conn = to_vector(mesh.conn(d));
  const auto topo = to_vector(mesh.topology(d));
  for (LO e = 0; e < mesh.count(d); ++e) {
    const auto t = static_cast<Topology>(topo[static_cast<std::size_t>(e)]);
    const auto arity = static_cast<LO>(off[static_cast<std::size_t>(e) + 1] - off[static_cast<std::size_t>(e)]);
    const auto row = h.down.row(e);
    const auto codes = h.down.row_codes(e);
    ASSERT_EQ(static_cast<int>(row.size()), side_total(t, arity, k));
    for (std::size_t s = 0; s < row.size(); ++s) {
      const Tuple side = side_tuple(off, conn, t, e, k, static_cast<int>(s));
      const Tuple stored = entity(h, row[s]);
      ASSERT_EQ(sorted(stored), sorted(side)) << "element " << e << " side " << s;
      ASSERT_EQ(static_cast<int>(codes[s]), brute_force_code(side, stored))
          << "element " << e << " side " << s;
    }
  }
}

// Incidences per entity with their senses, from the down graph.
std::vector<std::vector<int>> senses(const DerivedHost& h) {
  std::vector<std::vector<int>> out(static_cast<std::size_t>(h.count));
  for (std::size_t i = 0; i < h.down.targets.size(); ++i) {
    out[static_cast<std::size_t>(h.down.targets[i])].push_back(
        soma::topology::code_sense(h.down.codes[i]));
  }
  return out;
}

}  // namespace soma_test

TEST(Orientation, MatchesTheMoabConventions) {
  using soma::topology::connectivity_match;
  const LO e[2] = {10, 20};
  const LO er[2] = {20, 10};
  auto m = connectivity_match(e, er, 2);
  EXPECT_TRUE(m.same);
  EXPECT_EQ(m.offset, 1);
  EXPECT_TRUE(m.reversed);
  m = connectivity_match(e, e, 2);
  EXPECT_TRUE(m.same);
  EXPECT_EQ(m.offset, 0);
  EXPECT_FALSE(m.reversed);
  const LO q[4] = {1, 2, 3, 4};
  const LO rot2[4] = {3, 4, 1, 2};
  const LO rev3[4] = {3, 2, 1, 4};
  const LO rev1[4] = {1, 4, 3, 2};
  const LO other[4] = {1, 3, 2, 4};
  m = connectivity_match(q, rot2, 4);
  EXPECT_TRUE(m.same && m.offset == 2 && !m.reversed);
  m = connectivity_match(q, rev3, 4);
  EXPECT_TRUE(m.same && m.offset == 2 && m.reversed);
  m = connectivity_match(q, rev1, 4);
  EXPECT_TRUE(m.same && m.offset == 0 && m.reversed);
  EXPECT_FALSE(connectivity_match(q, other, 4).same);
  EXPECT_EQ(soma::topology::orientation_code(3, true), 7);
  EXPECT_EQ(soma::topology::code_offset(7), 3);
  EXPECT_EQ(soma::topology::code_sense(7), -1);
  EXPECT_EQ(soma::topology::code_sense(6), 1);
}

TEST(Derive, BoxCountsAreClosedForm) {
  const auto mesh = soma_test::box(8, 16, 16, Topology::hex);
  const auto edges = soma_test::derive_on(mesh, 1, DevExec{});
  const auto faces = soma_test::derive_on(mesh, 2, DevExec{});
  EXPECT_EQ(edges.count, 7208);
  EXPECT_EQ(faces.count, 6656);
  const auto s = soma_test::senses(faces);
  int boundary = 0;
  for (const auto& inc : s) {
    if (inc.size() == 1) {
      ++boundary;
      EXPECT_EQ(inc[0], 1);  // the only element is the creator
    } else {
      ASSERT_EQ(inc.size(), 2u);
      EXPECT_EQ(inc[0] + inc[1], 0);  // one +1 and one -1
    }
  }
  EXPECT_EQ(boundary, 1024);
  ASSERT_EQ(faces.segments.size(), 1u);
  EXPECT_EQ(faces.segments[0].topology, Topology::quad);
  EXPECT_EQ(edges.gid.front(), 1);
  EXPECT_EQ(edges.gid.back(), 7208);
}

TEST(Derive, MatchesTheIndependentSetsOnBoxes) {
  for (const auto ordering : {soma::Ordering::file, soma::Ordering::type_major_sfc}) {
    const auto hex = soma_test::box(3, 4, 5, Topology::hex, ordering);
    soma_test::check_against_expected(hex, 1, soma_test::derive_on(hex, 1, DevExec{}));
    soma_test::check_against_expected(hex, 2, soma_test::derive_on(hex, 2, DevExec{}));
    for (const Topology t : {Topology::quad, Topology::tri}) {
      const auto flat = soma_test::box(4, 3, 1, t, ordering);
      soma_test::check_against_expected(flat, 1, soma_test::derive_on(flat, 1, DevExec{}));
    }
  }
}

TEST(Derive, MixedElementsGetOneSegmentPerSideTopology) {
  const auto mesh = soma_test::mixed_3d();
  const auto faces = soma_test::derive_on(mesh, 2, DevExec{});
  soma_test::check_against_expected(mesh, 2, faces);
  ASSERT_EQ(faces.segments.size(), 2u);
  EXPECT_EQ(faces.segments[0], (soma::Segment{0, 9, Topology::tri, 3, soma::Ownership::interior}));
  EXPECT_EQ(faces.segments[1], (soma::Segment{9, 17, Topology::quad, 4, soma::Ownership::interior}));
  for (LO id = 0; id < faces.count; ++id) {
    EXPECT_EQ(faces.topology[static_cast<std::size_t>(id)],
              static_cast<std::uint8_t>(id < 9 ? Topology::tri : Topology::quad));
  }
  soma_test::check_against_expected(mesh, 1, soma_test::derive_on(mesh, 1, DevExec{}));
}

TEST(Derive, PolygonEdgesFollowTheVertexCycle) {
  const auto mesh = soma_test::merged_polygons();
  const auto edges = soma_test::derive_on(mesh, 1, DevExec{});
  soma_test::check_against_expected(mesh, 1, edges);
  const auto s = soma_test::senses(edges);
  int boundary = 0;
  for (const auto& inc : s) boundary += inc.size() == 1;
  const LO arity_sum = static_cast<LO>(edges.down.targets.size());
  EXPECT_EQ(arity_sum, 64);
  EXPECT_EQ(boundary, 18);
  EXPECT_EQ(edges.count, (arity_sum + boundary) / 2);
  EXPECT_EQ(edges.count, 41);
}

TEST(Derive, IsBitwiseEqualAcrossExecutionSpaces) {
  const auto hex = soma_test::box(5, 4, 3, Topology::hex, soma::Ordering::type_major_sfc);
  const auto mixed = soma_test::mixed_3d();
  const auto poly = soma_test::merged_polygons();
  for (const auto* mesh : {&hex, &mixed, &poly}) {
    for (int k = 1; k < mesh->dim(); ++k) {
      const auto ref = soma_test::derive_on(*mesh, k, HostExec{});
      EXPECT_EQ(soma_test::derive_on(*mesh, k, DevExec{}), ref);
#ifdef KOKKOS_ENABLE_SERIAL
      EXPECT_EQ(soma_test::derive_on(*mesh, k, Kokkos::Serial{}), ref);
#endif
    }
  }
}

TEST(Derive, RejectsDimensionsWithoutElementsAbove) {
  const auto mesh = soma_test::box(2, 2, 1, Topology::quad);
  const auto up = soma::topology::transpose(HostExec{}, mesh.conn_offsets(2), mesh.conn(2),
                                            mesh.count(0), "test");
  EXPECT_THROW((void)soma::topology::derive(HostExec{}, mesh, 2, up), std::invalid_argument);
  EXPECT_THROW((void)soma::topology::derive(HostExec{}, mesh, 0, up), std::invalid_argument);
}

// Bottom faces of the 2 x 2 x 2 box (z = 0), listed in the hex side-4 order
// (v0, v3, v2, v1), which is outward; the box gid of vertex (i, j, 0) is
// 1 + i + 3 j.
TEST(Derive, AdoptsExplicitFacesKeepingGidsAndOrder) {
  const auto g = [](int i, int j) { return GO{1 + i + 3 * j}; };
  std::vector<GO> quads;
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      quads.insert(quads.end(), {g(i, j), g(i, j + 1), g(i + 1, j + 1), g(i + 1, j)});
    }
  }
  const std::vector<GO> qgids{100, 101, 102, 103};
  const auto mesh = soma_test::box_with_explicit(2, qgids, quads);
  ASSERT_EQ(mesh.count(2), 4);
  const auto faces = soma_test::derive_on(mesh, 2, DevExec{});
  EXPECT_EQ(faces.count, 36);
  soma_test::check_against_expected(mesh, 2, faces);

  const auto vgid = soma_test::to_vector(mesh.gid(0));
  std::set<GO> seen;
  int adopted = 0;
  for (LO id = 0; id < faces.count; ++id) {
    const GO gid = faces.gid[static_cast<std::size_t>(id)];
    EXPECT_TRUE(seen.insert(gid).second) << "gid " << gid << " repeats";
    if (gid > 103) continue;
    ++adopted;
    const auto q = static_cast<std::ptrdiff_t>(gid - 100);
    std::vector<GO> got;
    for (const LO v : soma_test::entity(faces, id)) got.push_back(vgid[static_cast<std::size_t>(v)]);
    EXPECT_EQ(got, std::vector<GO>(quads.begin() + 4 * q, quads.begin() + 4 * q + 4));
  }
  EXPECT_EQ(adopted, 4);
  // explicit gids are kept and new ones count up from the largest + 1
  EXPECT_EQ(seen.size(), 36u);
  EXPECT_EQ(*seen.begin(), 100);
  EXPECT_EQ(*seen.rbegin(), 135);
}

TEST(Derive, AdoptsExplicitEdges) {
  const std::vector<GO> egids{7, 3};
  const std::vector<GO> edges{2, 1, 5, 6};  // reversed (1,2), and (5,6)
  const auto mesh = soma_test::box_with_explicit(2, {}, {}, egids, edges);
  const auto derived = soma_test::derive_on(mesh, 1, DevExec{});
  soma_test::check_against_expected(mesh, 1, derived);
  const auto vgid = soma_test::to_vector(mesh.gid(0));
  int adopted = 0;
  for (LO id = 0; id < derived.count; ++id) {
    const GO gid = derived.gid[static_cast<std::size_t>(id)];
    if (gid != 7 && gid != 3) {
      EXPECT_GE(gid, 8);
      continue;
    }
    ++adopted;
    const auto e = soma_test::entity(derived, id);
    const std::vector<GO> got{vgid[static_cast<std::size_t>(e[0])], vgid[static_cast<std::size_t>(e[1])]};
    EXPECT_EQ(got, gid == 7 ? (std::vector<GO>{2, 1}) : (std::vector<GO>{5, 6}));
  }
  EXPECT_EQ(adopted, 2);
}

TEST(Derive, RejectsOrphanAndRepeatedExplicitEntities) {
  // vertex (i, j, k) has gid 1 + i + 3 j + 9 k; (1, 5, 14, 10) cuts a hex
  // diagonally and is no face
  const auto orphan = soma_test::box_with_explicit(2, {42}, {1, 5, 14, 10});
  try {
    (void)soma_test::derive_on(orphan, 2, HostExec{});
    ADD_FAILURE() << "no exception";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("gid 42"), std::string::npos) << e.what();
  }
  const auto twins = soma_test::box_with_explicit(2, {50, 51}, {1, 2, 5, 4, 2, 5, 4, 1});
  try {
    (void)soma_test::derive_on(twins, 2, HostExec{});
    ADD_FAILURE() << "no exception";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("gid 51"), std::string::npos) << e.what();
  }
}

TEST(Derive, RejectsExplicitFacesWhoseVerticesAreNoCycle) {
  // the bottom face of the first hex is (1, 2, 5, 4); (1, 5, 2, 4) crosses it
  const auto bowtie = soma_test::box_with_explicit(2, {100}, {1, 5, 2, 4});
  try {
    (void)soma_test::derive_on(bowtie, 2, HostExec{});
    ADD_FAILURE() << "no exception";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("gid 100"), std::string::npos) << e.what();
  }
}
