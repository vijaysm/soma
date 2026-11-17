// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_VIEW_TYPES_HPP_
#define SOMA_CORE_VIEW_TYPES_HPP_

#include <Kokkos_Core.hpp>

namespace soma {

/// Every View in SOMA is LayoutLeft, rank 1 included, so that host and device
/// types differ only in memory space and mirrors assign without conversion.
template <class T, class MemSpace>
using View1 = Kokkos::View<T*, Kokkos::LayoutLeft, MemSpace>;

template <class T, class MemSpace>
using View2 = Kokkos::View<T**, Kokkos::LayoutLeft, MemSpace>;

/// Read-only, non-owning: what kernels and device-copyable handles hold.
template <class T, class MemSpace>
using UView1 = Kokkos::View<const T*, Kokkos::LayoutLeft, MemSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>;

}  // namespace soma

#endif  // SOMA_CORE_VIEW_TYPES_HPP_
