// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mesh_fixture.hpp"
#include "soma/core/handle_ref.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/registry.hpp"
#include "soma/testing/box_mesh.hpp"
#include "soma/topology/build.hpp"

using soma::GO;
using soma::Layout;
using soma::LO;
using soma::set_dim;
using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace {

// The fixture mesh (vertex gids 10..70, quads 7 and 3, triangle 5) with sets
// 100 and 200.
soma::HostMesh mesh_with_two_sets() {
  soma::HostMesh mesh = soma_test::two_quads_and_a_triangle();
  soma::SetBuilder b;
  b.add(soma::SetSpec{200, soma::set_unique, {}, {}, {}});
  b.add(soma::SetSpec{100, soma::set_unique, {{2, 0}}, {}, {}});
  std::move(b).install(mesh);
  return mesh;
}

LO lid_of(const soma::HostMesh& m, int d, GO gid) {
  for (LO l = 0; l < m.count(d); ++l) {
    if (m.gid(d)(static_cast<std::size_t>(l)) == gid) return l;
  }
  return soma::invalid_lo;
}

std::string what_of(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST(HandleRef, HandlesHoldASlotAndAGid) {
  using soma::handle_gid;
  using soma::handle_slot;
  using soma::make_handle;
  using soma::max_handle_gid;
  for (const int slot : {0, 1, 2, 3, set_dim}) {
    for (const GO gid : {GO{0}, GO{1}, GO{70}, max_handle_gid}) {
      const GO h = make_handle(slot, gid);
      EXPECT_GT(h, 0) << slot << " " << gid;
      EXPECT_EQ(handle_slot(h), slot) << gid;
      EXPECT_EQ(handle_gid(h), gid) << slot;
    }
  }
  // One gid in two slots: two handles.
  EXPECT_NE(make_handle(0, 1), make_handle(1, 1));
  EXPECT_NE(make_handle(2, 1), make_handle(set_dim, 1));
  for (const auto& [slot, gid] : std::vector<std::pair<int, GO>>{
           {-1, 1}, {set_dim + 1, 1}, {0, -1}, {2, max_handle_gid + 1}, {0, std::numeric_limits<GO>::max()}}) {
    EXPECT_EQ(make_handle(slot, gid), 0) << slot << " " << gid;
  }
  for (const GO h : {GO{0}, GO{-1}, GO{70}, max_handle_gid, std::numeric_limits<GO>::min(),
                     std::numeric_limits<GO>::max(), GO{set_dim + 2} << soma::handle_gid_bits}) {
    EXPECT_EQ(handle_slot(h), -1) << h;
  }
}

TEST(HandleRef, ResolvesEntitiesAndSetsOnTheHost) {
  using soma::make_handle;
  const auto mesh = mesh_with_two_sets();
  const auto ref = soma::HandleRef<HostMem>::build(mesh);
  EXPECT_EQ(ref.size(), 7u + 3u + 2u);
  for (const GO g : {10, 40, 70}) {
    const auto t = ref.resolve(make_handle(0, g));
    EXPECT_EQ(t.slot, 0) << g;
    EXPECT_EQ(t.lid, lid_of(mesh, 0, g)) << g;
  }
  for (const GO g : {3, 5, 7}) {
    const auto t = ref.resolve(make_handle(2, g));
    EXPECT_EQ(t.slot, 2) << g;
    EXPECT_EQ(t.lid, lid_of(mesh, 2, g)) << g;
  }
  EXPECT_EQ(ref.resolve(make_handle(set_dim, 100)).slot, set_dim);
  EXPECT_EQ(ref.resolve(make_handle(set_dim, 100)).lid, 0);
  EXPECT_EQ(ref.resolve(make_handle(set_dim, 200)).lid, 1);
  // Bare gids, gids of another slot, and gids no entity has name nothing.
  for (const GO h : {GO{0}, GO{-10}, GO{10}, GO{100}, make_handle(2, 10), make_handle(0, 7), make_handle(1, 7),
                     make_handle(3, 7), make_handle(set_dim, 7), make_handle(0, 1), make_handle(0, 99),
                     make_handle(set_dim, 300)}) {
    EXPECT_FALSE(ref.resolve(h).valid()) << h;
  }
}

// The kernel lives in a named function: CUDA refuses device lambdas in a
// test body (a private member function).
template <class R, class I>
void resolve_all(const R& ref, const I& in, const Kokkos::View<int*, DevMem>& slot,
                 const Kokkos::View<LO*, DevMem>& lid) {
  Kokkos::parallel_for(
      "soma::test::resolve", Kokkos::RangePolicy<DevMem::execution_space>(0, in.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        const soma::HandleTarget t = ref.resolve(in(i));
        slot(i) = t.slot;
        lid(i) = t.lid;
      });
}

TEST(HandleRef, ResolvesInKernels) {
  using soma::make_handle;
  const auto host = mesh_with_two_sets();
  const auto dev = host.mirror<DevMem>();
  const auto ref = soma::HandleRef<DevMem>::build(dev);
  const std::vector<GO> handles{make_handle(0, 70), make_handle(2, 3), make_handle(set_dim, 200), 0,
                                make_handle(0, 99), make_handle(2, 70)};
  const auto in = Kokkos::create_mirror_view_and_copy(
      DevMem{}, soma::View1<const GO, HostMem>(handles.data(), handles.size()));
  Kokkos::View<int*, DevMem> slot("soma::test::slot", handles.size());
  Kokkos::View<LO*, DevMem> lid("soma::test::lid", handles.size());
  resolve_all(ref, in, slot, lid);
  const auto hs = Kokkos::create_mirror_view_and_copy(HostMem{}, slot);
  const auto hl = Kokkos::create_mirror_view_and_copy(HostMem{}, lid);
  EXPECT_EQ((std::vector<int>{hs(0), hs(1), hs(2), hs(3), hs(4), hs(5)}),
            (std::vector<int>{0, 2, set_dim, -1, -1, -1}));
  EXPECT_EQ(hl(0), lid_of(host, 0, 70));
  EXPECT_EQ(hl(1), lid_of(host, 2, 3));
  EXPECT_EQ(hl(2), 1);
  EXPECT_EQ(hl(3), soma::invalid_lo);
}

// box_mesh numbers vertices and elements from 1 alike; a set may share a gid
// with a vertex. The slot tells them apart.
TEST(HandleRef, GidsRepeatAcrossSlots) {
  using soma::make_handle;
  const auto box = soma::testing::box_mesh({2, 2, 1, soma::Topology::quad});
  const auto ref = soma::HandleRef<HostMem>::build(box);
  EXPECT_EQ(ref.size(), static_cast<std::size_t>(box.count(0) + box.count(2)));
  for (const int d : {0, 2}) {
    for (LO l = 0; l < box.count(d); ++l) {
      const GO g = box.gid(d)(static_cast<std::size_t>(l));
      const auto t = ref.resolve(make_handle(d, g));
      EXPECT_EQ(t.slot, d) << g;
      EXPECT_EQ(t.lid, l) << g;
    }
  }
  soma::HostMesh mesh = soma_test::two_quads_and_a_triangle();
  soma::SetBuilder b;
  b.add(soma::SetSpec{70, soma::set_unique, {}, {}, {}});
  std::move(b).install(mesh);
  const auto r = soma::HandleRef<HostMem>::build(mesh);
  EXPECT_EQ(r.resolve(make_handle(0, 70)).slot, 0);
  EXPECT_EQ(r.resolve(make_handle(0, 70)).lid, lid_of(mesh, 0, 70));
  EXPECT_EQ(r.resolve(make_handle(set_dim, 70)).slot, set_dim);
  EXPECT_EQ(r.resolve(make_handle(set_dim, 70)).lid, 0);
}

// Derived edges are numbered within their dimension; their gids meet the
// vertex and element gids of the fixture, and each still has its own handle.
TEST(HandleRef, ResolvesDerivedEntities) {
  using soma::make_handle;
  soma::HostMesh mesh = soma_test::two_quads_and_a_triangle();
  soma::build_adjacencies(mesh, soma::AdjacencyRequest{}.add(2, 1));
  ASSERT_GT(mesh.count(1), 0);
  const auto ref = soma::HandleRef<HostMem>::build(mesh);
  EXPECT_EQ(ref.size(), static_cast<std::size_t>(7 + mesh.count(1) + 3));
  bool shared = false;
  for (int d = 0; d <= 2; ++d) {
    for (LO l = 0; l < mesh.count(d); ++l) {
      const GO g = mesh.gid(d)(static_cast<std::size_t>(l));
      if (d == 1 && (lid_of(mesh, 0, g) != soma::invalid_lo || lid_of(mesh, 2, g) != soma::invalid_lo)) {
        shared = true;
      }
      const auto t = ref.resolve(make_handle(d, g));
      EXPECT_EQ(t.slot, d) << g;
      EXPECT_EQ(t.lid, l) << g;
    }
  }
  EXPECT_TRUE(shared);
}

// Gid 0 is a gid like any other; the null handle is 0.
TEST(HandleRef, GidZeroHasAHandle) {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{0, 1, 2}, std::vector<soma::Real>{0, 0, 0, 1, 0, 0, 0, 1, 0});
  b.add_elements(soma::Topology::tri, std::vector<GO>{5}, std::vector<GO>{0, 1, 2});
  const auto mesh = std::move(b).build(soma::Ordering::file);
  const auto ref = soma::HandleRef<HostMem>::build(mesh);
  EXPECT_EQ(ref.resolve(soma::make_handle(0, 0)).slot, 0);
  EXPECT_EQ(ref.resolve(soma::make_handle(0, 0)).lid, lid_of(mesh, 0, 0));
  EXPECT_FALSE(ref.resolve(0).valid());
}

TEST(HandleRef, RejectsGidsNoHandleCanHold) {
  for (const GO bad : {GO{-3}, soma::max_handle_gid + 1}) {
    soma::HostMesh mesh = soma_test::two_quads_and_a_triangle();
    soma::MeshTestAccess::set_gid(mesh, 2, lid_of(mesh, 2, 3), bad);
    const std::string what = what_of([&] { soma::HandleRef<HostMem>::build(mesh); });
    EXPECT_NE(what.find("HandleRef: dimension 2 has the gid " + std::to_string(bad) + ", which no handle can hold"),
              std::string::npos)
        << what;
  }
}

TEST(HandleRef, RejectsAGidRepeatedWithinASlot) {
  soma::HostMesh mesh = soma_test::two_quads_and_a_triangle();
  soma::MeshTestAccess::set_gid(mesh, 0, lid_of(mesh, 0, 40), 20);
  const std::string what = what_of([&] { soma::HandleRef<HostMem>::build(mesh); });
  EXPECT_NE(what.find("HandleRef: dimension 0 has two entities of gid 20"), std::string::npos) << what;
}

// Handle fields hold gids; their family is apart from plain GO fields.
TEST(HandleFields, AreGidFieldsOfTheirOwnFamily) {
  soma::FieldRegistry<HostMem> reg;
  reg.set_entities(0, 4);
  soma::FieldOptions<GO> h;
  h.handle = true;
  const auto f = reg.add_tag<GO>("ADJ", 0, 1, h);
  const auto plain = reg.add_tag<GO>("ADJ", 0, 1);
  EXPECT_TRUE(reg.info(f.id()).handle);
  EXPECT_FALSE(reg.info(plain.id()).handle);
  EXPECT_NE(reg.info(f.id()).family, reg.info(plain.id()).family);
  EXPECT_EQ(reg.family<GO>("ADJ", 1, Layout::per_entity(), true), reg.info(f.id()).family);
  EXPECT_EQ(reg.family<GO>("ADJ", 1), reg.info(plain.id()).family);
  EXPECT_THROW(reg.add_tag<GO>("ADJ", 0, 1, h), soma::FieldClash);

  soma::FieldOptions<double> wrong;
  wrong.handle = true;
  EXPECT_NE(what_of([&] { reg.add_tag<double>("X", 0, 1, wrong); })
                .find("field 'X': handle values need the GO value type"),
            std::string::npos);
  const soma::Space<HostMem> sp(soma::Domain<HostMem>::all(0, 4, 1), Layout::per_entity());
  const std::vector<soma::Offset> off{0, 1, 1, 1, 3};
  const std::vector<GO> v{7, 3, 5};
  const auto var = reg.add_var<GO>("ADJ", sp, off, v, h);
  EXPECT_TRUE(reg.info(var.id()).handle);
  EXPECT_TRUE(reg.info(var.id()).variable);
  EXPECT_THROW(reg.add_var<double>("Y", sp, off, std::vector<double>{1, 2, 3}, wrong), std::invalid_argument);

  // On one space, the handle and plain fields are told apart by id only; the
  // variable-length one is not a fixed field.
  EXPECT_NE(what_of([&] { reg.get<GO>("ADJ", sp); }).find("has several fields of this value type on this space"),
            std::string::npos);
  reg.remove(plain.id());
  EXPECT_EQ(reg.get<GO>("ADJ", sp).id(), f.id());

  // Mirrors keep the mark.
  soma::FieldRegistry<HostMem> copy;
  reg.mirror_into<HostMem>(copy, {"*"}, {true, true, true, true, true}, false);
  EXPECT_TRUE(copy.info(f.id()).handle);
  EXPECT_TRUE(copy.info(var.id()).handle);
  EXPECT_FALSE(copy.contains("Y"));
}
