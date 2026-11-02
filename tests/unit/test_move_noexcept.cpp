// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Moves of Domain, Space, Field, the registry and the mesh are noexcept, so
// they must not allocate: the empty snapshot a move leaves behind has to exist
// by the time any object does. A throw out of a noexcept move is a terminate, which the harness
// reports as a crash. Each test runs in its own process under ctest, so nothing
// before the armed region has built an empty snapshot; the objects are made
// through the non-default constructors for the same reason. Only move
// construction is armed: move assignment also releases the target's old
// storage, and releasing Kokkos storage allocates in Kokkos' own bookkeeping.

#include <cstdlib>
#include <new>
#include <utility>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/field.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/registry.hpp"
#include "soma/fields/space.hpp"

namespace soma_test {

bool fail_every_allocation = false;

struct ArmedScope {
  ArmedScope() { fail_every_allocation = true; }
  ~ArmedScope() { fail_every_allocation = false; }
  ArmedScope(const ArmedScope&) = delete;
  ArmedScope& operator=(const ArmedScope&) = delete;
};

}  // namespace soma_test

// nvcc compiles a replaced global operator new for the device too, where
// throwing is not allowed: the host body is host-only, the device gets malloc.
void* operator new(std::size_t n) {
  KOKKOS_IF_ON_HOST((
    if (soma_test::fail_every_allocation) throw std::bad_alloc();
    if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc();
  ))
  KOKKOS_IF_ON_DEVICE((return malloc(n == 0 ? 1 : n);))
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using HostMem = Kokkos::HostSpace;

TEST(MoveNoexcept, MovingADomainDoesNotAllocate) {
  soma::Domain<HostMem> a = soma::Domain<HostMem>::all(2, 5, 1);
  soma::Domain<HostMem>* b = nullptr;
  alignas(soma::Domain<HostMem>) unsigned char storage[sizeof(soma::Domain<HostMem>)];
  {
    const soma_test::ArmedScope armed;
    b = new (storage) soma::Domain<HostMem>(std::move(a));
  }
  EXPECT_EQ(b->size(), 5);
  EXPECT_TRUE(a.empty());
  b->~Domain();
}

TEST(MoveNoexcept, MovingASpaceDoesNotAllocate) {
  const auto domain = soma::Domain<HostMem>::all(2, 5, 1);
  soma::Space<HostMem> a(domain, soma::Layout::per_entity());
  soma::Space<HostMem>* b = nullptr;
  alignas(soma::Space<HostMem>) unsigned char storage[sizeof(soma::Space<HostMem>)];
  {
    const soma_test::ArmedScope armed;
    b = new (storage) soma::Space<HostMem>(std::move(a));
  }
  EXPECT_EQ(b->rows(), 5);
  EXPECT_EQ(a.rows(), 0);
  b->~Space();
}

TEST(MoveNoexcept, MovingAFieldDoesNotAllocate) {
  soma::FieldRegistry<HostMem> reg;
  reg.set_entities(2, 5);
  soma::Field<double, HostMem> a = reg.add_tag<double>("q", 2, 1);
  soma::Field<double, HostMem>* b = nullptr;
  alignas(soma::Field<double, HostMem>) unsigned char storage[sizeof(soma::Field<double, HostMem>)];
  {
    const soma_test::ArmedScope armed;
    b = new (storage) soma::Field<double, HostMem>(std::move(a));
  }
  EXPECT_EQ(b->rows(), 5);
  EXPECT_EQ(a.rows(), 0);
  b->~Field();
}

TEST(MoveNoexcept, MovingARegistryDoesNotAllocate) {
  soma::FieldRegistry<HostMem> a;
  a.set_entities(2, 5);
  a.add_tag<double>("q", 2, 1);
  soma::FieldRegistry<HostMem>* b = nullptr;
  alignas(soma::FieldRegistry<HostMem>) unsigned char storage[sizeof(soma::FieldRegistry<HostMem>)];
  {
    const soma_test::ArmedScope armed;
    b = new (storage) soma::FieldRegistry<HostMem>(std::move(a));
  }
  EXPECT_EQ(b->list().size(), 1u);
  b->~FieldRegistry();
}

TEST(MoveNoexcept, MovingAMeshDoesNotAllocate) {
  soma::HostMesh a = soma_test::two_quads_and_a_triangle();
  a.fields().add_tag<double>("q", 2, 1);
  soma::HostMesh* b = nullptr;
  alignas(soma::HostMesh) unsigned char storage[sizeof(soma::HostMesh)];
  {
    const soma_test::ArmedScope armed;
    b = new (storage) soma::HostMesh(std::move(a));
  }
  EXPECT_EQ(b->count(2), 3);
  EXPECT_EQ(b->fields().list().size(), 1u);
  EXPECT_EQ(a.count(2), 0);  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(a.fields().has_entities(2));
  b->~Mesh();
}
