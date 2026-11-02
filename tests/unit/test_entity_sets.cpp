// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/core/sets.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/topology/build.hpp"

using soma::GO;
using soma::LO;
using soma::SetEntry;
using soma::SetItem;
using soma::SetSpec;
using soma::set_dim;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace soma_test {

// The fixture (7 vertices, faces: lid 0 the triangle, 1 and 2 the quads) with
// its edges derived, and four sets:
//   gid 100 unordered: faces 2, 0, 2 and vertex 6, set 300
//   gid 200 ordered:   vertex 3, face 1, vertex 3, edge 0, set 100
//   gid 300 unordered: set 200 (a containment cycle 100 -> 300 -> 200 -> 100)
//   gid 400 unordered and empty; child of 100, whose parent it is too
inline soma::HostMesh mesh_with_sets() {
  soma::HostMesh mesh = two_quads_and_a_triangle();
  soma::build_adjacencies(mesh, soma::AdjacencyRequest{}.add(2, 1));
  soma::SetBuilder b;
  b.add(SetSpec{300, soma::set_unique, {{set_dim, 200}}, {}, {}});
  b.add(SetSpec{100, soma::set_unique | soma::set_track_owner,
                {{2, 2}, {2, 0}, {2, 2}, {0, 6}, {set_dim, 300}}, {400}, {400}});
  b.add(SetSpec{200, soma::set_ordered, {{0, 3}, {2, 1}, {0, 3}, {1, 0}, {set_dim, 100}}, {}, {}});
  b.add(SetSpec{400, soma::set_unique, {}, {}, {100}});
  std::move(b).install(mesh);
  return mesh;
}

template <class M>
Kokkos::View<int*, Kokkos::HostSpace> contains_on_space(soma::SetsView<M> v, int slot, LO set, LO n) {
  Kokkos::View<int*, M> out("soma::test::set_contains", static_cast<std::size_t>(n));
  Kokkos::parallel_for(
      "soma::test::set_contains", Kokkos::RangePolicy<typename M::execution_space>(0, n),
      KOKKOS_LAMBDA(const LO l) { out(l) = v.contains(slot, set, l) ? 1 : 0; });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
}

template <class S>
std::vector<LO> vec(S span) {
  return {span.begin(), span.end()};
}

}  // namespace soma_test

TEST(EntitySets, AMeshHasNoneUntilInstalled) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  EXPECT_EQ(mesh.sets().count(), 0);
  EXPECT_EQ(mesh.sets().gid().extent(0), 0u);
  EXPECT_TRUE(mesh.sets().has_members(set_dim));
  EXPECT_FALSE(mesh.sets().index_of(1).has_value());
  EXPECT_THROW((void)mesh.sets().flags(0), std::out_of_range);
}

TEST(EntitySets, RowsAreSortedAndUniquePerSlot) {
  const auto mesh = soma_test::mesh_with_sets();
  const auto& s = mesh.sets();
  ASSERT_EQ(s.count(), 4);
  const auto g = Kokkos::create_mirror_view_and_copy(HostMem{}, s.gid());
  EXPECT_EQ((std::vector<GO>{g(0), g(1), g(2), g(3)}), (std::vector<GO>{100, 200, 300, 400}));
  EXPECT_EQ(soma_test::vec(s.members_of(0, 2)), (std::vector<LO>{0, 2}));
  EXPECT_EQ(soma_test::vec(s.members_of(0, 0)), (std::vector<LO>{6}));
  EXPECT_EQ(soma_test::vec(s.members_of(0, set_dim)), (std::vector<LO>{2}));
  EXPECT_TRUE(s.members_of(0, 1).empty());
  EXPECT_EQ(soma_test::vec(s.members_of(1, 0)), (std::vector<LO>{3}));
  EXPECT_EQ(soma_test::vec(s.members_of(1, 1)), (std::vector<LO>{0}));
  EXPECT_EQ(soma_test::vec(s.members_of(1, set_dim)), (std::vector<LO>{0}));
  for (int k = 0; k <= set_dim; ++k) EXPECT_TRUE(s.members_of(3, k).empty()) << k;
  const auto off = Kokkos::create_mirror_view_and_copy(HostMem{}, s.offsets(2));
  ASSERT_EQ(off.extent(0), 5u);
  EXPECT_EQ(off(4), 3);
  EXPECT_EQ(s.index_of(300), 2);
  EXPECT_FALSE(s.index_of(250).has_value());
}

TEST(EntitySets, FlagsOrderAndLinksAreKept) {
  const auto mesh = soma_test::mesh_with_sets();
  const auto& s = mesh.sets();
  EXPECT_EQ(s.flags(0), soma::set_unique | soma::set_track_owner);
  EXPECT_FALSE(s.ordered(0));
  EXPECT_TRUE(s.ordered(1));
  const auto seq = s.sequence(1);
  const std::vector<SetEntry> want{{0, 3}, {2, 1}, {0, 3}, {1, 0}, {set_dim, 0}};
  EXPECT_EQ((std::vector<SetEntry>(seq.begin(), seq.end())), want);
  EXPECT_THROW((void)s.sequence(0), std::logic_error);
  EXPECT_EQ(soma_test::vec(s.children(0)), (std::vector<LO>{3}));
  EXPECT_EQ(soma_test::vec(s.parents(0)), (std::vector<LO>{3}));
  EXPECT_EQ(soma_test::vec(s.parents(3)), (std::vector<LO>{0}));
  EXPECT_TRUE(s.children(3).empty());
  EXPECT_TRUE(s.children(1).empty());
}

TEST(EntitySets, RecursiveMembersFollowContainedSetsAndStopAtCycles) {
  const auto mesh = soma_test::mesh_with_sets();
  const auto& s = mesh.sets();
  EXPECT_EQ(s.members_recursive(0, 0), (std::vector<LO>{3, 6}));
  EXPECT_EQ(s.members_recursive(0, 2), (std::vector<LO>{0, 1, 2}));
  EXPECT_EQ(s.members_recursive(2, 1), (std::vector<LO>{0}));
  EXPECT_TRUE(s.members_recursive(3, 0).empty());
  EXPECT_THROW((void)s.members_recursive(0, set_dim), std::invalid_argument);
}

TEST(EntitySets, DeviceRowsMatchTheHost) {
  const auto host = soma_test::mesh_with_sets();
  const auto dev = host.mirror<DevMem>();
  ASSERT_EQ(dev.sets().count(), 4);
  const auto view = dev.sets().view();
  for (LO set = 0; set < 4; ++set) {
    for (int slot = 0; slot <= set_dim; ++slot) {
      const LO n = slot == set_dim ? 4 : host.count(slot);
      const auto got = soma_test::contains_on_space<DevMem>(view, slot, set, n);
      const auto want = host.sets().members_of(set, slot);
      for (LO l = 0; l < n; ++l) {
        const bool in = std::find(want.begin(), want.end(), l) != want.end();
        EXPECT_EQ(got(static_cast<std::size_t>(l)), in ? 1 : 0)
            << "set " << set << ", slot " << slot << ", lid " << l;
      }
    }
  }
}

TEST(EntitySets, InstallRejectsBadSetsAndChangesNothing) {
  const auto rejects = [](std::vector<SetSpec> specs, const std::string& needle) {
    soma::HostMesh mesh = soma_test::two_quads_and_a_triangle();
    soma::SetBuilder b;
    for (auto& s : specs) b.add(std::move(s));
    try {
      std::move(b).install(mesh);
      ADD_FAILURE() << "accepted; expected: " << needle;
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
    EXPECT_EQ(mesh.sets().count(), 0);
    EXPECT_EQ(mesh.fields().template add_tag<int>("t", set_dim, 1).rows(), 0);
  };
  rejects({{1, 0x8, {}, {}, {}}}, "set gid 1 has flags 0x08, outside 0x07");
  rejects({{1, soma::set_unique, {}, {}, {}}, {1, soma::set_unique, {}, {}, {}}}, "set gid 1 is repeated");
  rejects({{1, soma::set_unique, {{2, 3}}, {}, {}}}, "set gid 1 lists entity 3 of dimension 2, which has 3 entities");
  rejects({{1, soma::set_unique, {{0, -1}}, {}, {}}}, "lists entity -1 of dimension 0");
  rejects({{1, soma::set_unique, {{3, 0}}, {}, {}}}, "of dimension 3, which has 0 entities");
  rejects({{1, soma::set_unique, {{5, 0}}, {}, {}}}, "lists a member of slot 5, not in [0, 4]");
  rejects({{1, soma::set_unique, {{set_dim, 2}}, {}, {}}}, "set gid 1 contains gid 2, which is not a set");
  rejects({{1, soma::set_unique, {}, {9}, {}}}, "set gid 1 has child gid 9, which is not a set");
  rejects({{1, soma::set_unique, {}, {}, {9}}}, "set gid 1 has parent gid 9, which is not a set");
}

TEST(EntitySets, AMeshTakesItsSetsOnce) {
  auto mesh = soma_test::mesh_with_sets();
  soma::SetBuilder b;
  b.add(SetSpec{1, soma::set_unique, {}, {}, {}});
  EXPECT_THROW(std::move(b).install(mesh), std::logic_error);
  EXPECT_EQ(mesh.sets().count(), 4);
}

TEST(EntitySets, MovesAndSwapsCarryTheSets) {
  auto mesh = soma_test::mesh_with_sets();
  soma::HostMesh moved(std::move(mesh));
  EXPECT_EQ(moved.sets().count(), 4);
  EXPECT_EQ(mesh.sets().count(), 0);  // NOLINT(bugprone-use-after-move): moved-from is empty
  soma::HostMesh other = soma_test::two_quads_and_a_triangle();
  other.swap(moved);
  EXPECT_EQ(other.sets().count(), 4);
  EXPECT_EQ(moved.sets().count(), 0);
}

TEST(EntitySets, MirrorsCarryTheSetsByDefault) {
  const auto mesh = soma_test::mesh_with_sets();
  const auto check = [&](const soma::HostMesh& m) {
    ASSERT_EQ(m.sets().count(), 4);
    EXPECT_EQ(soma_test::vec(m.sets().members_of(0, 2)), (std::vector<LO>{0, 2}));
    EXPECT_EQ(m.sets().sequence(1).size(), 5u);
    EXPECT_EQ(soma_test::vec(m.sets().children(0)), (std::vector<LO>{3}));
    EXPECT_NE(m.sets().members(2).data(), mesh.sets().members(2).data());
  };
  check(mesh.mirror<HostMem>());
  check(mesh.clone());
  const auto alias = mesh.mirror_view<HostMem>();
  EXPECT_EQ(alias.sets().members(2).data(), mesh.sets().members(2).data());
  soma::TransferOptions none;
  none.sets = false;
  EXPECT_EQ(mesh.mirror<HostMem>(none).sets().count(), 0);
}

TEST(EntitySets, MembersOfADroppedDimensionThrowInTheMirror) {
  const auto mesh = soma_test::mesh_with_sets();
  soma::TransferOptions o;
  o.dims = {true, false, true, true};
  const auto m = mesh.mirror<HostMem>(o);
  const auto& s = m.sets();
  EXPECT_FALSE(s.has_members(1));
  EXPECT_TRUE(s.has_members(set_dim));
  EXPECT_THROW((void)s.members(1), soma::MissingSetMembers);
  EXPECT_THROW((void)s.offsets(1), soma::MissingSetMembers);
  EXPECT_THROW((void)s.members_of(0, 1), soma::MissingSetMembers);
  EXPECT_THROW((void)s.members_recursive(0, 1), soma::MissingSetMembers);
  EXPECT_THROW((void)s.sequence(1), soma::MissingSetMembers);  // holds edge 0
  try {
    (void)s.members(1);
  } catch (const soma::MissingSetMembers& e) {
    EXPECT_NE(std::string(e.what()).find("set members of dimension 1 were not transferred"),
              std::string::npos) << e.what();
  }
  EXPECT_EQ(soma_test::vec(s.members_of(0, 2)), (std::vector<LO>{0, 2}));
}

TEST(EntitySets, FieldsOnSetsUseTheSetSlot) {
  auto mesh = soma_test::mesh_with_sets();
  using Dom = soma::Domain<HostMem>;
  const auto all = Dom::all(mesh, set_dim);
  EXPECT_EQ(all.size(), 4);
  EXPECT_EQ(all.fingerprint(0), Dom::all(set_dim, 4, mesh.epoch(), mesh.sets().gid()).fingerprint(0));
  auto ids = mesh.fields().add_tag<int>("GLOBAL_ID", set_dim, 1);
  EXPECT_EQ(ids.rows(), 4);
  mesh.fields().add_tag<int>("GLOBAL_ID", 0, 1);
  const auto fam = mesh.fields().family<int>("GLOBAL_ID", 1);
  EXPECT_EQ(mesh.fields().family_members(fam, set_dim), std::vector<soma::FieldId>{ids.id()});
  const auto face_members = Dom::from_set(mesh, 0, 2);
  EXPECT_EQ(face_members.members(0), (std::vector<LO>{0, 2}));
  EXPECT_EQ(Dom::from_set(mesh, 1, set_dim).members(0), (std::vector<LO>{0}));
  EXPECT_THROW((void)Dom::segments(mesh, set_dim, {}), std::invalid_argument);

  soma::TransferOptions with = soma::TransferOptions::everything();
  EXPECT_TRUE(mesh.mirror<HostMem>(with).fields().contains("GLOBAL_ID"));
  EXPECT_EQ(mesh.mirror<HostMem>(with).fields().get<int>("GLOBAL_ID", set_dim).rows(), 4);
  with.sets = false;
  const auto bare = mesh.mirror<HostMem>(with);
  EXPECT_THROW((void)bare.fields().get<int>("GLOBAL_ID", set_dim), std::out_of_range);
  EXPECT_EQ(bare.fields().get<int>("GLOBAL_ID", 0).rows(), 7);
}
