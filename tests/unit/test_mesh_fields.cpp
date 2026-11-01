// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/errors.hpp"
#include "soma/fields/space.hpp"

using soma::LO;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

TEST(MeshFields, MeshStartsAtEpochOneAndTheRegistryAgrees) {
  const auto mesh = soma_test::two_quads_and_a_triangle();
  EXPECT_EQ(mesh.epoch(), 1u);
  EXPECT_EQ(mesh.fields().epoch(), 1u);
  EXPECT_TRUE(mesh.fields().has_entities(0));
  EXPECT_TRUE(mesh.fields().has_entities(2));
}

// An empty mesh has zero entities of every dimension, whether it was default
// constructed, moved from or cloned, so add_tag behaves the same on all three.
TEST(MeshFields, AnEmptyMeshTakesTagsLikeItsClone) {
  soma::HostMesh fresh;
  auto source = soma_test::two_quads_and_a_triangle();
  const soma::HostMesh taken = std::move(source);
  auto copy = fresh.clone();
  for (soma::HostMesh* m : {&fresh, &source, &copy}) {  // NOLINT(bugprone-use-after-move)
    for (int d = 0; d < 4; ++d) EXPECT_TRUE(m->fields().has_entities(d)) << d;
    const auto t = m->fields().add_tag<double>("t", 0, 1);
    EXPECT_EQ(t.rows(), 0);
    EXPECT_EQ(t.domain(), soma::Domain<HostMem>::all(0, 0, 1));
  }
  EXPECT_EQ(taken.count(2), 3);
}

TEST(MeshFields, AddTagThroughTheMeshUsesCountsAndGlobalIds) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  auto t = mesh.fields().add_tag<double>("temperature", 0, 1);
  EXPECT_EQ(t.rows(), 7);
  auto q = mesh.fields().add_tag<std::int32_t>("material", 2, 1);
  EXPECT_EQ(q.rows(), 3);
  EXPECT_EQ(q.domain(), soma::Domain<HostMem>::all(2, 3, mesh.epoch(), mesh.gid(2)));
  EXPECT_FALSE(q.domain() == soma::Domain<HostMem>::all(2, 3, mesh.epoch()));
}

TEST(MeshFields, MirrorCopiesTheSelectedFieldsAndTheEpoch) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  auto t = mesh.fields().add_tag<double>("temperature", 0, 1);
  mesh.fields().add_tag<double>("flux", 2, 1);
  t.view()(3, 0) = 4.5;
  soma::TransferOptions opt;
  opt.fields = {"temperature"};
  const auto dev = mesh.mirror<DevMem>(opt);
  EXPECT_TRUE(dev.fields().contains("temperature"));
  EXPECT_FALSE(dev.fields().contains("flux"));
  EXPECT_EQ(dev.epoch(), mesh.epoch());
  EXPECT_EQ(dev.fields().epoch(), mesh.epoch());
  const auto m = dev.fields().get<double>("temperature");
  const auto h = Kokkos::create_mirror_view_and_copy(HostMem{}, m.view());
  EXPECT_DOUBLE_EQ(h(3, 0), 4.5);
  const auto lean = mesh.mirror<DevMem>();
  EXPECT_TRUE(lean.fields().list().empty());
  EXPECT_THROW(mesh.mirror<DevMem>(soma::TransferOptions{{true, true, true, true}, true, {"nope"}}),
               std::out_of_range);
}

TEST(MeshFields, MirrorKeepsABumpedEpochAndItsFields) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  mesh.bump_epoch();
  mesh.bump_epoch();
  mesh.fields().add_tag<double>("temperature", 0, 1);
  const auto dev = mesh.mirror<DevMem>(soma::TransferOptions::everything());
  EXPECT_EQ(dev.epoch(), 3u);
  EXPECT_EQ(dev.fields().epoch(), 3u);
  EXPECT_NO_THROW(dev.fields().get<double>("temperature"));
}

TEST(MeshFields, MirrorOmitsFieldsOfDroppedDimensions) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  mesh.fields().add_tag<double>("v", 0, 1);
  mesh.fields().add_tag<double>("c", 2, 1);
  auto opt = soma::TransferOptions::everything();
  opt.dims = {true, true, false, false};
  const auto m = mesh.mirror<HostMem>(opt);
  EXPECT_TRUE(m.fields().contains("v"));
  EXPECT_FALSE(m.fields().contains("c"));
}

// In the same memory space mirror_view shares the field data with the source,
// so a write through either is seen by both; mirror copies it.
TEST(MeshFields, MirrorViewSharesFieldDataInTheSameSpace) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  auto t = mesh.fields().add_tag<double>("temperature", 0, 1);
  t.view()(2, 0) = 1.5;
  soma::TransferOptions opt;
  opt.fields = {"temperature"};
  const auto alias = mesh.mirror_view<HostMem>(opt);
  const auto a = alias.fields().get<double>("temperature");
  EXPECT_EQ(a.view().data(), t.view().data());
  a.view()(2, 0) = 7.0;
  EXPECT_DOUBLE_EQ(t.view()(2, 0), 7.0);
  const auto deep = mesh.mirror<HostMem>(opt);
  const auto c = deep.fields().get<double>("temperature");
  EXPECT_NE(c.view().data(), t.view().data());
  EXPECT_DOUBLE_EQ(c.view()(2, 0), 7.0);
  c.view()(2, 0) = -1.0;
  EXPECT_DOUBLE_EQ(t.view()(2, 0), 7.0);
}

TEST(MeshFields, CloneKeepsFieldsDeepAndEpoch) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  auto t = mesh.fields().add_tag<double>("temperature", 0, 1);
  t.view()(0, 0) = 2.0;
  auto copy = mesh.clone();
  auto c = copy.fields().get<double>("temperature");
  EXPECT_NE(c.view().data(), t.view().data());
  EXPECT_DOUBLE_EQ(c.view()(0, 0), 2.0);
  c.view()(0, 0) = 9.0;
  EXPECT_DOUBLE_EQ(t.view()(0, 0), 2.0);
  EXPECT_EQ(copy.epoch(), mesh.epoch());
  copy.fields().add_tag<double>("later", 2, 1);
}

TEST(MeshFields, BumpEpochMakesFieldsStaleOnTheMeshAndItsRegistry) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  mesh.fields().add_tag<double>("temperature", 0, 1);
  mesh.bump_epoch();
  EXPECT_EQ(mesh.epoch(), 2u);
  EXPECT_EQ(mesh.fields().epoch(), 2u);
  EXPECT_THROW(mesh.fields().get<double>("temperature"), soma::StaleField);
  EXPECT_NO_THROW(mesh.fields().add_tag<double>("fresh", 0, 1));
}

TEST(MeshFields, MoveEmptiesTheMeshAndItsFields) {
  auto a = soma_test::two_quads_and_a_triangle();
  a.fields().add_tag<double>("t", 0, 1);
  soma::HostMesh b = std::move(a);
  EXPECT_EQ(b.fields().list().size(), 1u);
  EXPECT_TRUE(a.fields().list().empty());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(a.count(2), 0);                // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(b.count(2), 3);
  EXPECT_EQ(b.epoch(), 1u);
}

TEST(MeshFields, MoveCarriesTheEpochAndLeavesTheSourceAtTheInitialOne) {
  auto a = soma_test::two_quads_and_a_triangle();
  a.bump_epoch();
  a.bump_epoch();
  soma::HostMesh b = std::move(a);
  EXPECT_EQ(b.epoch(), 3u);
  EXPECT_EQ(b.fields().epoch(), 3u);
  EXPECT_EQ(a.epoch(), 1u);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(a.fields().epoch(), 1u);
  EXPECT_NO_THROW(b.fields().add_tag<double>("t", 0, 1));
}

TEST(MeshFields, MirrorRefusesAStaleSelectedFieldAndIgnoresOthers) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  mesh.fields().add_tag<double>("old", 0, 1);
  mesh.bump_epoch();
  mesh.fields().add_tag<double>("fresh", 0, 1);
  EXPECT_THROW(mesh.mirror<DevMem>(soma::TransferOptions::everything()), soma::StaleField);
  EXPECT_THROW(mesh.clone(), soma::StaleField);
  soma::TransferOptions fresh_only;
  fresh_only.fields = {"fresh"};
  const auto m = mesh.mirror<DevMem>(fresh_only);
  EXPECT_TRUE(m.fields().contains("fresh"));
  EXPECT_FALSE(m.fields().contains("old"));
  EXPECT_EQ(m.epoch(), 2u);
  const auto lean = mesh.mirror<DevMem>();
  EXPECT_TRUE(lean.fields().list().empty());
  EXPECT_EQ(lean.epoch(), 2u);
}

TEST(MeshFields, MirrorCommitsThePendingWritesOfTheSelectedFieldsOnly) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  const auto members =
      soma::Domain<HostMem>::from_list(2, 3, mesh.epoch(), std::vector<LO>{0}, mesh.gid(2));
  auto& reg = mesh.fields();
  const auto q = reg.add<double>("q", soma::Space<HostMem>(members, soma::Layout::per_entity()), 1);
  const auto r = reg.add<double>("r", soma::Space<HostMem>(members, soma::Layout::per_entity()), 1);
  const double v = 7.0;
  reg.set_pending<double>(q.id(), 2, std::span<const double>(&v, 1));
  reg.set_pending<double>(r.id(), 2, std::span<const double>(&v, 1));
  ASSERT_EQ(reg.pending_count(q.id()), 1u);

  const auto lean = mesh.mirror<DevMem>();
  EXPECT_EQ(reg.pending_count(q.id()), 1u);
  EXPECT_TRUE(reg.is_current(q));

  soma::TransferOptions only_q;
  only_q.fields = {"q"};
  const auto dev = mesh.mirror<DevMem>(only_q);
  EXPECT_EQ(reg.pending_count(q.id()), 0u);
  EXPECT_FALSE(reg.is_current(q));
  EXPECT_EQ(reg.pending_count(r.id()), 1u);
  EXPECT_TRUE(reg.is_current(r));

  const auto copy = dev.fields().get<double>("q");
  EXPECT_EQ(copy.rows(), 2);
  const auto host = Kokkos::create_mirror_view_and_copy(HostMem{}, copy.view());
  EXPECT_DOUBLE_EQ(host(static_cast<std::size_t>(copy.space().entity_rows(2, 2).begin), 0), 7.0);
}
