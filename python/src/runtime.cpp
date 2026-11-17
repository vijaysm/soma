// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Kokkos's lifetime, communicators from mpi4py, and the translation of C++
// exceptions. The only binding source that tests SOMA_ENABLE_MPI.

#include <stdexcept>
#include <string>
#include <string_view>

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include "common.hpp"
#include "soma/config.hpp"
#include "soma/parallel/comm.hpp"

#if SOMA_ENABLE_MPI
#include <mpi.h>
#include <mpi4py/mpi4py.h>
#endif

namespace soma_py {

namespace {
std::atomic<bool> finalized{false};
}  // namespace

std::atomic<long>& live_count() noexcept {
  static std::atomic<long> count{0};
  return count;
}

bool kokkos_finalized() noexcept { return finalized.load(); }

namespace {

bool initialized_here = false;
PyObject* soma_error = nullptr;

// Holds Kokkos memory for the exit test: the smallest live object.
using TestView = Kokkos::View<double*, Kokkos::HostSpace>;
struct LiveObject {
  Guarded<TestView> values{std::make_unique<TestView>("soma_py::LiveObject", 4)};
};

void translate(const std::exception_ptr& p, void*) {
  try {
    std::rethrow_exception(p);
  } catch (const nb::builtin_exception&) {
    throw;  // nanobind's own (TypeError, ...): its translator handles them
  } catch (const nb::python_error&) {
    throw;
  } catch (const std::invalid_argument& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
  } catch (const std::out_of_range& e) {
    PyErr_SetString(PyExc_IndexError, e.what());
  } catch (const std::exception& e) {
    PyErr_SetString(soma_error, e.what());
  }
}

}  // namespace

void bind_runtime(nb::module_& m) {
  if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
    Kokkos::initialize(Kokkos::InitializationSettings().set_disable_warnings(true));
    initialized_here = true;
  }

  soma_error = PyErr_NewExceptionWithDoc("soma.Error", "An error from SOMA's C++ library.", PyExc_RuntimeError,
                                         nullptr);
  m.attr("Error") = nb::handle(soma_error);
  nb::register_exception_translator(translate);

  m.def("live_objects", [] { return live_count().load(); },
        "The number of live objects holding Kokkos memory (meshes, fields, arrays, ...).");
  m.def(
      "_finalize",
      [] {
        if (initialized_here && Kokkos::is_initialized()) {
          finalized = true;
          Kokkos::finalize();
        }
      },
      "Finalizes Kokkos if this module initialized it; run at exit. Objects still alive then leak their "
      "memory to the operating system.");

  m.def(
      "testing_raise",
      [](std::string_view kind) {
        const std::string what = "soma test error: " + std::string(kind);
        if (kind == "invalid_argument") throw std::invalid_argument(what);
        if (kind == "out_of_range") throw std::out_of_range(what);
        throw std::runtime_error(what);
      },
      "Test hook: throws the named C++ exception.");
  nb::class_<LiveObject>(m, "_LiveObject");
  m.def("testing_live_object", [] { return new LiveObject(); }, nb::rv_policy::take_ownership,
        "Test hook: an object holding Kokkos memory.");

#if SOMA_ENABLE_MPI
  m.attr("has_mpi") = true;
  if (import_mpi4py() < 0) throw nb::python_error();
  m.def(
      "mpi_library_version",
      [] {
        char version[MPI_MAX_LIBRARY_VERSION_STRING];
        int length = 0;
        MPI_Get_library_version(version, &length);
        return std::string(version, static_cast<std::size_t>(length));
      },
      "The version string of the MPI library SOMA calls.");
#else
  m.attr("has_mpi") = false;
#endif

  nb::class_<soma::Comm>(m, "Comm", "A communicator: the ranks a collective call runs on.")
      .def(
          "__init__",
          [](soma::Comm* self, nb::handle comm) {
#if SOMA_ENABLE_MPI
            if (!PyObject_TypeCheck(comm.ptr(), &PyMPIComm_Type)) {
              throw nb::type_error("soma.Comm: expected an mpi4py communicator (mpi4py.MPI.Comm)");
            }
            new (self) soma::Comm(*PyMPIComm_Get(comm.ptr()));
#else
            (void)self;
            (void)comm;
            throw nb::type_error("soma.Comm: SOMA was built without MPI; use soma.Comm.self()");
#endif
          },
          nb::arg("comm"), nb::keep_alive<1, 2>(),
          "Wraps an mpi4py communicator, which must outlive this object (it is kept alive).")
      .def_static("world", &soma::Comm::world, "Every rank (MPI_COMM_WORLD), or the one rank without MPI.")
      .def_static("self", &soma::Comm::self, "This rank alone.")
      .def_prop_ro("rank", &soma::Comm::rank)
      .def_prop_ro("size", &soma::Comm::size)
      .def("barrier", &soma::Comm::barrier);
}

}  // namespace soma_py
