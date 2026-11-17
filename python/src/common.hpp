// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

/// \file
/// What every binding source shares: the count of live objects holding Kokkos
/// memory, NumPy arrays over Kokkos views, and the registration functions of
/// each source.

#ifndef SOMA_PYTHON_COMMON_HPP_
#define SOMA_PYTHON_COMMON_HPP_

#include <array>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
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
#include "soma/parallel/comm.hpp"

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

/// Runs `check` (argument checks and conversions before a collective call)
/// and agrees on the outcome: if it threw on any rank, it throws on every rank
/// (its own exception where it failed), so no rank enters the collective alone.
template <class F>
void agree(const soma::Comm& comm, F&& check) {
  std::exception_ptr failure;
  try {
    check();
  } catch (...) {
    failure = std::current_exception();
  }
  if (comm.allreduce(static_cast<std::int32_t>(failure != nullptr), soma::Op::max) != 0) {
    if (failure) std::rethrow_exception(failure);
    throw std::invalid_argument("invalid arguments on another rank");
  }
}

template <class T>
constexpr const char* numpy_name() {
  if constexpr (std::is_same_v<T, std::int64_t>) return "int64";
  else if constexpr (std::is_same_v<T, std::int32_t>) return "int32";
  else if constexpr (std::is_same_v<T, std::int8_t>) return "int8";
  else if constexpr (std::is_same_v<T, std::uint8_t>) return "uint8";
  else static_assert(sizeof(T) == 0, "numpy_name: add the type");
}

template <class T>
using IntArray = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using PointArray = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// A 1-D array of integers as T, refusing what a cast would misread: floats
/// (TypeError) and values outside T's range (ValueError). Booleans and every
/// integer type convert. The result references a converted copy when needed.
template <class T>
IntArray<T> checked_ints(nb::handle values, const std::string& what) {
  const nb::module_ np = nb::module_::import_("numpy");
  const nb::object a = np.attr("asarray")(values);
  const std::string kind = nb::cast<std::string>(a.attr("dtype").attr("kind"));
  if (kind != "i" && kind != "u" && kind != "b") {
    throw nb::type_error((what + ": expected integers, got " + nb::cast<std::string>(a.attr("dtype").attr("name"))).c_str());
  }
  if (nb::cast<int>(a.attr("ndim")) != 1) throw std::invalid_argument(what + ": expected a 1-D array");
  if (kind != "b" && nb::cast<std::size_t>(a.attr("size")) > 0) {
    const nb::object info = np.attr("iinfo")(numpy_name<T>());
    const nb::module_ op = nb::module_::import_("operator");
    const nb::object lo = nb::module_::import_("builtins").attr("int")(a.attr("min")());
    const nb::object hi = nb::module_::import_("builtins").attr("int")(a.attr("max")());
    if (nb::cast<bool>(op.attr("lt")(lo, info.attr("min"))) || nb::cast<bool>(op.attr("gt")(hi, info.attr("max")))) {
      throw std::invalid_argument(what + ": values outside the range of " + numpy_name<T>());
    }
  }
  return nb::cast<IntArray<T>>(np.attr("ascontiguousarray")(a, nb::arg("dtype") = numpy_name<T>()));
}

/// Points as a C-ordered (n, 3) float64 array; an empty array of any shape is
/// no points.
inline PointArray points(nb::handle values, const std::string& what) {
  const nb::module_ np = nb::module_::import_("numpy");
  nb::object a = np.attr("asarray")(values, nb::arg("dtype") = "float64");
  if (nb::cast<std::size_t>(a.attr("size")) == 0) a = a.attr("reshape")(0, 3);
  const nb::object shape = a.attr("shape");
  if (nb::cast<int>(a.attr("ndim")) != 2 || nb::cast<std::size_t>(shape[1]) != 3) {
    throw std::invalid_argument(what + ": expected points of shape (n, 3)");
  }
  return nb::cast<PointArray>(np.attr("ascontiguousarray")(a));
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
void bind_algorithms(nb::module_& m);
void bind_parallel(nb::module_& m, nb::class_<PyMesh>& mesh);

}  // namespace soma_py

#endif  // SOMA_PYTHON_COMMON_HPP_
