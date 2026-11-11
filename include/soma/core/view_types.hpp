// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_VIEW_TYPES_HPP_
#define SOMA_CORE_VIEW_TYPES_HPP_

#include <cstddef>

#include <Kokkos_Core.hpp>

namespace soma {

// Every View in SOMA is LayoutLeft, rank 1 included, so that host and device
// types differ only in memory space and mirrors assign without conversion.
template <class T, class MemSpace>
using View1 = Kokkos::View<T*, Kokkos::LayoutLeft, MemSpace>;

template <class T, class MemSpace>
using View2 = Kokkos::View<T**, Kokkos::LayoutLeft, MemSpace>;

// Read-only, non-owning: what kernels and device-copyable handles hold.
template <class T, class MemSpace>
using UView1 = Kokkos::View<const T*, Kokkos::LayoutLeft, MemSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>;

namespace detail {

// Touches one byte per page of a host View from every host thread. A fresh
// large allocation is unmapped until written, and a device-to-host copy into
// it is staged by the driver, which faults the pages one at a time: measured
// at 1.85 GB/s against 5.4 GB/s into mapped pages on a PCIe 2.0 x16 link
// (bench_backends mirror_out, GTX 1050 Ti). Faulting them here first runs
// the faults on the host backend's threads.
template <class V>
void prefault(const V& v) {
  constexpr std::size_t page = 4096;
  constexpr std::size_t min_bytes = std::size_t{1} << 20;
  const std::size_t bytes = v.span() * sizeof(typename V::value_type);
  if (bytes < min_bytes || !v.span_is_contiguous()) return;
  char* const base = reinterpret_cast<char*>(v.data());
  const std::size_t pages = (bytes + page - 1) / page;
  Kokkos::parallel_for(
      "soma::prefault", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, pages),
      [=](const std::size_t i) { base[i * page] = 0; });
  Kokkos::DefaultHostExecutionSpace{}.fence();
}

// A deep copy of src into a new uninitialized View in Dst, with the pages of
// a host destination mapped before a copy from a space the host cannot read.
template <class Dst, class V>
auto mirror_and_copy(const V& src) {
  auto out = Kokkos::create_mirror(Kokkos::view_alloc(Dst{}, Kokkos::WithoutInitializing), src);
  if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, Dst>::accessible &&
                !Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                            typename V::memory_space>::accessible) {
    prefault(out);
  }
  Kokkos::deep_copy(out, src);
  return out;
}

}  // namespace detail
}  // namespace soma

#endif  // SOMA_CORE_VIEW_TYPES_HPP_
