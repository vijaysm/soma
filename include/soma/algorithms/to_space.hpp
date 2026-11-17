// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_TO_SPACE_HPP_
#define SOMA_ALGORITHMS_TO_SPACE_HPP_

/// \file
/// A host vector copied into a view of a memory space: how the algorithms move
/// the plans they build on the host.

#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/view_types.hpp"

namespace soma::detail {

template <class T, class MemSpace>
View1<T, MemSpace> to_space(const std::vector<T>& v, const char* label) {
  View1<T, MemSpace> out(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), v.size());
  Kokkos::deep_copy(out, Kokkos::View<const T*, Kokkos::LayoutLeft, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(v.data(), v.size()));
  return out;
}

}  // namespace soma::detail

#endif  // SOMA_ALGORITHMS_TO_SPACE_HPP_
