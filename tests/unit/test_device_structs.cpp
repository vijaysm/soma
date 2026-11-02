// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Everything a kernel captures or receives by value is copied byte by byte to
// the device, so it must stay trivially copyable (no std::string, vector or
// owning View members).

#include <cstring>
#include <vector>
#include <type_traits>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/segments.hpp"
#include "soma/core/types.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/node_tables.hpp"
#include "soma/fields/registry.hpp"
#include "soma/fields/resolver.hpp"
#include "soma/fields/space.hpp"
#include "soma/topology/canonical.hpp"

namespace soma_test {

template <template <class...> class Pred, class... Ts>
inline constexpr bool all_of = (Pred<Ts>::value && ...);

template <class... Ts>
inline constexpr bool all_trivially_copyable = all_of<std::is_trivially_copyable, Ts...>;

template <class M, template <class...> class Pred>
inline constexpr bool every_view_struct = all_of<
    Pred, soma::DomainBlockView<M>, soma::DomainView<M>, typename soma::DomainView<M>::Location,
    soma::SpaceBlockView<M>, soma::SpaceView<M>, typename soma::SpaceView<M>::Location,
    soma::FamilyDimView<M>, soma::FamilyView<M>, soma::NodeTableView<M>,
    typename soma::NodeTables<M>::Permutation, soma::UView1<soma::LO, M>>;

// Under nvcc (and NVHPC) Kokkos gives View a user-provided copy constructor, a
// workaround for defaulted constructors with constraints, so there no struct
// holding a View is trivially copyable; Kokkos still copies Views bytewise into
// kernels (ViewsSurviveAByteCopy below). Where this Kokkos build's own unmanaged
// View is trivially copyable, every SOMA view struct must be; everywhere, none
// may own anything (trivially destructible).
template <class M>
inline constexpr bool kokkos_view_is_trivially_copyable =
    std::is_trivially_copyable_v<Kokkos::View<int*, M, Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

static_assert(every_view_struct<HostMem, std::is_trivially_destructible>);
static_assert(every_view_struct<DevMem, std::is_trivially_destructible>);
static_assert(!kokkos_view_is_trivially_copyable<HostMem> ||
              every_view_struct<HostMem, std::is_trivially_copyable>);
static_assert(!kokkos_view_is_trivially_copyable<DevMem> ||
              every_view_struct<DevMem, std::is_trivially_copyable>);

static_assert(all_trivially_copyable<soma::RowRange, soma::FamilyHit, soma::detail::MemberRec,
                                     soma::Fingerprint, soma::Interval, soma::Layout,
                                     soma::NodeTableKey, soma::canonical::SideDef, soma::Segment>);

}  // namespace soma_test

TEST(DeviceStructs, ViewsSurviveAByteCopy) {
  using M = Kokkos::DefaultExecutionSpace::memory_space;
  const auto domain = soma::Domain<M>::from_list(2, 40, 1, std::vector<soma::LO>{3, 9, 20});
  const soma::Space<M> space(domain, soma::Layout::per_entity());
  const soma::SpaceView<M> view = space.view();
  alignas(soma::SpaceView<M>) unsigned char raw[sizeof(soma::SpaceView<M>)];
  std::memcpy(raw, &view, sizeof(view));
  soma::SpaceView<M> copy;
  std::memcpy(&copy, raw, sizeof(copy));
  EXPECT_EQ(copy.rows, 3);
  EXPECT_EQ(copy.num_blocks, 1);
}
