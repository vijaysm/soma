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

#include "soma/core/types.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/node_tables.hpp"
#include "soma/fields/registry.hpp"
#include "soma/fields/resolver.hpp"
#include "soma/fields/space.hpp"

namespace soma_test {

template <class... Ts>
inline constexpr bool all_trivially_copyable = (std::is_trivially_copyable_v<Ts> && ...);

template <class M>
inline constexpr bool views_are_trivially_copyable = all_trivially_copyable<
    soma::DomainBlockView<M>, soma::DomainView<M>, typename soma::DomainView<M>::Location,
    soma::SpaceBlockView<M>, soma::SpaceView<M>, typename soma::SpaceView<M>::Location,
    soma::FamilyDimView<M>, soma::FamilyView<M>, soma::NodeTableView<M>,
    typename soma::NodeTables<M>::Permutation, soma::UView1<soma::LO, M>>;

static_assert(views_are_trivially_copyable<Kokkos::HostSpace>);
static_assert(views_are_trivially_copyable<Kokkos::DefaultExecutionSpace::memory_space>);

static_assert(all_trivially_copyable<soma::RowRange, soma::FamilyHit, soma::detail::MemberRec,
                                     soma::Fingerprint, soma::Interval>);

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
