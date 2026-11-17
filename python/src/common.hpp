// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

/// \file
/// What every binding source shares: the count of live objects holding Kokkos
/// memory, NumPy arrays over Kokkos views, and the registration functions of
/// each source.

#ifndef SOMA_PYTHON_COMMON_HPP_
#define SOMA_PYTHON_COMMON_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"

namespace soma_py {

namespace nb = nanobind;

/// The number of live objects that hold Kokkos memory (a test hook).
std::atomic<long>& live_count() noexcept;

/// Counts itself among the live objects for as long as it exists.
struct Live {
  Live() noexcept { ++live_count(); }
  Live(const Live&) noexcept { ++live_count(); }
  Live& operator=(const Live&) noexcept = default;
  ~Live() { --live_count(); }
};

/// Whether this module finalized Kokkos (at interpreter exit). Objects freed
/// after that leak their Kokkos memory to the operating system instead of
/// freeing it: Kokkos forbids deallocation after finalize, and a process that
/// never finalizes Kokkos aborts at exit.
bool kokkos_finalized() noexcept;

/// Owns a T holding Kokkos memory; frees it unless Kokkos is finalized.
template <class T>
struct Guarded {
  std::unique_ptr<T> value;
  Live live;
  explicit Guarded(std::unique_ptr<T> v) noexcept : value(std::move(v)) {}
  Guarded(Guarded&&) noexcept = default;
  ~Guarded() {
    if (kokkos_finalized()) (void)value.release();  // left to the operating system
  }
};

/// A NumPy array over a host view's memory, without a copy. The array's owner
/// holds a copy of the view, so the allocation lives as long as the array,
/// whatever happens to the object the view came from. `T` const gives a
/// read-only array. Strides are the view's, in elements.
template <class T, class View>
nb::ndarray<nb::numpy, T> view_array(const View& view) {
  static_assert(View::rank <= 2, "view_array: rank 1 or 2");
  auto* holder = new Guarded<View>(std::make_unique<View>(view));
  nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<Guarded<View>*>(p); });
  std::array<std::size_t, 2> shape{};
  std::array<std::int64_t, 2> strides{};
  for (std::size_t r = 0; r < View::rank; ++r) {
    shape[r] = view.extent(r);
    strides[r] = static_cast<std::int64_t>(view.stride(r));
  }
  return nb::ndarray<nb::numpy, T>(const_cast<std::remove_const_t<T>*>(view.data()), View::rank, shape.data(),
                                   owner, strides.data());
}

/// A NumPy array owning a vector's elements (a copy made by the caller).
template <class T>
nb::ndarray<nb::numpy, T> vector_array(std::vector<T>&& values) {
  auto* holder = new std::vector<T>(std::move(values));
  nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  const std::size_t n = holder->size();
  return nb::ndarray<nb::numpy, T>(holder->data(), 1, &n, owner);
}

/// A mesh owned by Python: the C++ mesh behind a guard (above).
struct PyMesh {
  Guarded<soma::HostMesh> guard;
  explicit PyMesh(soma::HostMesh&& m) : guard(std::make_unique<soma::HostMesh>(std::move(m))) {}
  soma::HostMesh& get() noexcept { return *guard.value; }
  const soma::HostMesh& get() const noexcept { return *guard.value; }
};

void bind_runtime(nb::module_& m);
nb::class_<PyMesh> bind_mesh(nb::module_& m);
void bind_edit(nb::module_& m, nb::class_<PyMesh>& mesh);
void bind_fields(nb::module_& m, nb::class_<PyMesh>& mesh);
void bind_io(nb::module_& m);
void bind_parallel(nb::module_& m, nb::class_<PyMesh>& mesh);

}  // namespace soma_py

#endif  // SOMA_PYTHON_COMMON_HPP_
