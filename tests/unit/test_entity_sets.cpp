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
#include "soma/testing/box_mesh.hpp"
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
    EXPECT_NE(m.sets().data(2).data(), mesh.sets().data(2).data());
  };
  check(mesh.mirror<HostMem>());
  check(mesh.clone());
  const auto alias = mesh.mirror_view<HostMem>();
  EXPECT_EQ(alias.sets().data(2).data(), mesh.sets().data(2).data());
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
  EXPECT_THROW((void)s.data(1), soma::MissingSetMembers);
  EXPECT_THROW((void)s.offsets(1), soma::MissingSetMembers);
  EXPECT_THROW((void)s.members_of(0, 1), soma::MissingSetMembers);
  EXPECT_THROW((void)s.members_recursive(0, 1), soma::MissingSetMembers);
  EXPECT_THROW((void)s.sequence(1), soma::MissingSetMembers);  // holds edge 0
  try {
    (void)s.data(1);
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

namespace {

// A 10 x 10 quad box (100 faces, 121 vertices) with five sets of faces:
//   gid 10 every face (full), 20 faces 0-9 and 50-59 (intervals),
//   30 the even faces below 20 (list), 40 empty, 50 every vertex and no face.
soma::HostMesh box_with_shaped_sets() {
  soma::HostMesh mesh = soma::testing::box_mesh({10, 10, 1, soma::Topology::quad});
  std::vector<SetItem> all;
  std::vector<SetItem> two_runs;
  std::vector<SetItem> evens;
  std::vector<SetItem> vertices;
  for (GO f = 0; f < 100; ++f) all.push_back({2, f});
  for (GO f = 0; f < 10; ++f) two_runs.push_back({2, 50 + f});
  for (GO f = 0; f < 10; ++f) two_runs.push_back({2, f});
  for (GO f = 0; f < 20; f += 2) evens.push_back({2, f});
  for (GO v = 0; v < 121; ++v) vertices.push_back({0, v});
  soma::SetBuilder b;
  b.add(SetSpec{10, soma::set_unique, all, {}, {}});
  b.add(SetSpec{20, soma::set_unique, two_runs, {}, {}});
  b.add(SetSpec{30, soma::set_unique, evens, {}, {}});
  b.add(SetSpec{40, soma::set_unique, {}, {}, {}});
  b.add(SetSpec{50, soma::set_unique, vertices, {}, {}});
  std::move(b).install(mesh);
  return mesh;
}

std::vector<LO> face_rows(LO set) {
  std::vector<LO> out;
  if (set == 0) {
    for (LO f = 0; f < 100; ++f) out.push_back(f);
  } else if (set == 1) {
    for (LO f = 0; f < 10; ++f) out.push_back(f);
    for (LO f = 50; f < 60; ++f) out.push_back(f);
  } else if (set == 2) {
    for (LO f = 0; f < 20; f += 2) out.push_back(f);
  }
  return out;
}

}  // namespace

// Each row is held in its smallest form and reads back the same through
// indexing, lookup and iteration.
TEST(EntitySets, RowsTakeTheSmallestForm) {
  using soma::DomainKind;
  const auto mesh = box_with_shaped_sets();
  const auto& s = mesh.sets();
  ASSERT_EQ(s.count(), 5);
  const auto kinds = s.kinds(2);
  std::vector<DomainKind> got;
  for (std::size_t i = 0; i < kinds.extent(0); ++i) got.push_back(static_cast<DomainKind>(kinds(i)));
  EXPECT_EQ(got, (std::vector<DomainKind>{DomainKind::full, DomainKind::intervals, DomainKind::list,
                                          DomainKind::list, DomainKind::list}));
  const auto off = s.offsets(2);
  EXPECT_EQ((std::vector<soma::Offset>{off(0), off(1), off(2), off(3), off(4), off(5)}),
            (std::vector<soma::Offset>{0, 0, 5, 15, 15, 15}));
  EXPECT_EQ(s.data(2).extent(0), 15u);
  EXPECT_EQ(s.entities(2), 100);
  EXPECT_EQ(static_cast<DomainKind>(s.kinds(0)(4)), DomainKind::full);
  EXPECT_EQ(s.data(0).extent(0), 0u);

  for (LO set = 0; set < 5; ++set) {
    const std::vector<LO> want = face_rows(set);
    const soma::SetRow r = s.members_of(set, 2);
    EXPECT_EQ(r.size, static_cast<LO>(want.size())) << set;
    EXPECT_EQ(soma_test::vec(r), want) << set;
    for (LO k = 0; k < r.size; ++k) EXPECT_EQ(r[k], want[static_cast<std::size_t>(k)]) << set << " " << k;
    EXPECT_EQ(r[r.size], soma::invalid_lo);
    for (LO l = -1; l <= 100; ++l) {
      const auto it = std::find(want.begin(), want.end(), l);
      const LO row = it == want.end() ? soma::invalid_lo : static_cast<LO>(it - want.begin());
      EXPECT_EQ(r.row(l), row) << set << " " << l;
      EXPECT_EQ(r.contains(l), row != soma::invalid_lo) << set << " " << l;
    }
    const auto dom = soma::Domain<HostMem>::from_set(mesh, set, 2);
    EXPECT_EQ(dom.size(), static_cast<LO>(want.size())) << set;
    for (LO k = 0; k < dom.size(); ++k) {
      EXPECT_EQ(dom.view().locate(k).lid, want[static_cast<std::size_t>(k)]) << set << " " << k;
    }
  }
  EXPECT_EQ(soma::Domain<HostMem>::from_set(mesh, 0, 2).info(0).kind, DomainKind::full);
  EXPECT_EQ(soma::Domain<HostMem>::from_set(mesh, 1, 2).info(0).kind, DomainKind::intervals);
}

// The kernel lives in a named function: CUDA refuses device lambdas in a
// test body (a private member function).
template <class V>
void read_shaped_rows(const V& v, const Kokkos::View<LO**, DevMem>& kth, const Kokkos::View<LO**, DevMem>& row,
                      const Kokkos::View<LO*, DevMem>& walked) {
  Kokkos::parallel_for(
      "soma::test::shaped_rows", Kokkos::RangePolicy<DevMem::execution_space>(0, 5), KOKKOS_LAMBDA(const LO set) {
        const soma::SetRow r = v.row(2, set);
        for (LO k = 0; k <= 100; ++k) {
          kth(set, k) = r[k];
          row(set, k) = r.row(k);
        }
        LO sum = 0;
        LO n = 0;
        for (const LO l : r) {
          sum += l;
          ++n;
        }
        walked(set) = n == r.size ? sum : -1;
      });
}

// On the device, rows index, look up and iterate as on the host.
TEST(EntitySets, ShapedRowsReadTheSameInKernels) {
  const auto host = box_with_shaped_sets();
  const auto dev = host.mirror<DevMem>();
  Kokkos::View<LO**, DevMem> kth("soma::test::kth", 5, 101);
  Kokkos::View<LO**, DevMem> row("soma::test::row", 5, 101);
  Kokkos::View<LO*, DevMem> walked("soma::test::walked", 5);
  read_shaped_rows(dev.sets().view(), kth, row, walked);
  const auto hk = Kokkos::create_mirror_view_and_copy(HostMem{}, kth);
  const auto hr = Kokkos::create_mirror_view_and_copy(HostMem{}, row);
  const auto hw = Kokkos::create_mirror_view_and_copy(HostMem{}, walked);
  for (LO set = 0; set < 5; ++set) {
    const soma::SetRow r = host.sets().members_of(set, 2);
    LO sum = 0;
    for (const LO l : face_rows(set)) sum += l;
    EXPECT_EQ(hw(static_cast<std::size_t>(set)), sum) << set;
    for (LO k = 0; k <= 100; ++k) {
      EXPECT_EQ(hk(static_cast<std::size_t>(set), static_cast<std::size_t>(k)), r[k]) << set << " " << k;
      EXPECT_EQ(hr(static_cast<std::size_t>(set), static_cast<std::size_t>(k)), r.row(k)) << set << " " << k;
    }
  }
}

// Renumbering takes each row's form again, out of the new count.
TEST(EntitySets, RenumberedRowsTakeTheirFormAgain) {
  using soma::DomainKind;
  const auto mesh = box_with_shaped_sets();
  std::vector<LO> shift(100);
  for (LO f = 0; f < 100; ++f) shift[static_cast<std::size_t>(f)] = f + 3;
  const auto grown = mesh.sets().renumbered(2, shift, 110);
  EXPECT_EQ(grown.entities(2), 110);
  const soma::SetRow all = grown.members_of(0, 2);
  EXPECT_EQ(all.kind, DomainKind::intervals);  // 100 of 110: one run
  EXPECT_EQ(soma_test::vec(all).front(), 3);
  EXPECT_EQ(all.size, 100);
  std::vector<LO> halve(100, soma::invalid_lo);
  for (LO f = 0; f < 20; f += 2) halve[static_cast<std::size_t>(f)] = f / 2;
  const auto packed = [&] {
    // Only the evens set survives this map; the others name faces it drops.
    soma::HostMesh m = soma::testing::box_mesh({10, 10, 1, soma::Topology::quad});
    soma::SetBuilder b;
    std::vector<SetItem> evens;
    for (GO f = 0; f < 20; f += 2) evens.push_back({2, f});
    b.add(SetSpec{30, soma::set_unique, evens, {}, {}});
    std::move(b).install(m);
    return m.sets().renumbered(2, halve, 10);
  }();
  EXPECT_EQ(packed.members_of(0, 2).kind, DomainKind::full);
  EXPECT_EQ(soma_test::vec(packed.members_of(0, 2)), (std::vector<LO>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  EXPECT_THROW((void)mesh.sets().renumbered(2, halve, 10), std::logic_error);
}

// With drop, members without a new lid leave their rows and the sequences of
// ordered sets, whose other entries keep their order and repeats.
TEST(EntitySets, RenumberedWithDropLeavesOutUnmappedMembers) {
  soma::HostMesh m = soma::testing::box_mesh({4, 1, 1, soma::Topology::quad});
  soma::SetBuilder b;
  b.add(SetSpec{1, soma::set_ordered, {{2, 0}, {2, 2}, {0, 1}, {2, 0}, {2, 3}}, {}, {}});
  b.add(SetSpec{2, soma::set_unique, {{2, 1}, {2, 2}, {2, 3}}, {}, {}});
  std::move(b).install(m);
  // Face 2 goes; 3 becomes 2 and 0, 1 stay.
  const std::vector<LO> map{0, 1, soma::invalid_lo, 2};
  EXPECT_THROW((void)m.sets().renumbered(2, map, 3), std::logic_error);
  const auto out = m.sets().renumbered(2, map, 3, true);
  std::vector<std::pair<int, LO>> seq;
  for (const auto& e : out.sequence(0)) seq.emplace_back(e.slot, e.lid);
  EXPECT_EQ(seq, (std::vector<std::pair<int, LO>>{{2, 0}, {0, 1}, {2, 0}, {2, 2}}));
  EXPECT_EQ(soma_test::vec(out.members_of(0, 2)), (std::vector<LO>{0, 2}));
  EXPECT_EQ(soma_test::vec(out.members_of(1, 2)), (std::vector<LO>{1, 2}));
  EXPECT_EQ(soma_test::vec(out.members_of(0, 0)), (std::vector<LO>{1}));
  EXPECT_EQ(out.entities(2), 3);
}

// Runs build rows without listing their members, merge with the contents
// (face 7 lies inside the run 5..14), and name sets by consecutive gids.
TEST(EntitySets, RunsBuildRowsWithoutListingThem) {
  using soma::DomainKind;
  using soma::SetRun;
  soma::HostMesh mesh = soma::testing::box_mesh({10, 10, 1, soma::Topology::quad});
  soma::SetBuilder b;
  b.add(SetSpec{1, soma::set_unique, {}, {}, {}}, {{2, 0, 100}});
  b.add(SetSpec{2, soma::set_unique, {}, {}, {}}, {{2, 50, 10}, {2, 0, 10}});
  b.add(SetSpec{3, soma::set_unique, {{2, 3}, {2, 15}, {2, 40}, {2, 7}}, {}, {}}, {{2, 5, 10}, {2, 4, 2}});
  b.add(SetSpec{4, soma::set_unique, {}, {}, {}}, {{set_dim, 1, 3}, {0, 0, 121}});
  b.add(SetSpec{7, soma::set_unique, {}, {}, {}});
  std::move(b).install(mesh);
  const auto& s = mesh.sets();
  EXPECT_EQ(s.members_of(0, 2).kind, DomainKind::full);
  EXPECT_EQ(s.members_of(1, 2).kind, DomainKind::intervals);
  EXPECT_EQ(s.members_of(1, 2).size, 20);
  std::vector<LO> merged;
  for (LO f = 3; f < 16; ++f) merged.push_back(f);
  merged.push_back(40);
  EXPECT_EQ(soma_test::vec(s.members_of(2, 2)), merged);
  EXPECT_EQ(s.members_of(2, 2).kind, DomainKind::intervals);  // 2 runs, 14 members
  EXPECT_EQ(soma_test::vec(s.members_of(3, set_dim)), (std::vector<LO>{0, 1, 2}));
  EXPECT_EQ(s.members_of(3, 0).kind, DomainKind::full);
  EXPECT_EQ(s.data(2).extent(0), 5u + 5u);  // two intervals rows of 2 runs each
  EXPECT_EQ(s.data(0).extent(0), 0u);
}

TEST(EntitySets, InstallRejectsBadRuns) {
  using soma::SetRun;
  const auto rejected = [](const std::vector<SetRun>& runs, std::uint8_t flags, const std::string& needle) {
    soma::HostMesh mesh = soma::testing::box_mesh({2, 2, 1, soma::Topology::quad});
    soma::SetBuilder b;
    b.add(SetSpec{1, flags, {}, {}, {}}, runs);
    b.add(SetSpec{2, soma::set_unique, {}, {}, {}});
    b.add(SetSpec{3, soma::set_unique, {}, {}, {}});
    b.add(SetSpec{5, soma::set_unique, {}, {}, {}});
    try {
      std::move(b).install(mesh);
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
      EXPECT_EQ(mesh.sets().count(), 0);
      return;
    }
    ADD_FAILURE() << "accepted: " << needle;
  };
  rejected({{2, 2, 3}}, soma::set_unique, "set gid 1 lists the 3 entities from 2 of dimension 2, which has 4 entities");
  rejected({{2, -1, 2}}, soma::set_unique, "lists the 2 entities from -1 of dimension 2");
  rejected({{0, 9, 1}}, soma::set_unique, "lists the 1 entities from 9 of dimension 0, which has 9 entities");
  rejected({{2, 0, 0}}, soma::set_unique, "set gid 1 has a run of 0 members");
  rejected({{5, 0, 1}}, soma::set_unique, "set gid 1 has a run of slot 5, not in [0, 4]");
  rejected({{set_dim, 2, 3}}, soma::set_unique, "set gid 1 contains the 3 gids from 2, which are not all sets");
  rejected({{set_dim, 3, 3}}, soma::set_unique, "contains the 3 gids from 3, which are not all sets");
  rejected({{set_dim, 4, 1}}, soma::set_unique, "set gid 1 contains gid 4, which is not a set");
  rejected({{2, 0, 1}}, soma::set_ordered, "set gid 1 is ordered and takes no runs");
}
