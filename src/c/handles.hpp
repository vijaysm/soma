// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's internals: handle structs, the error boundary, and collective
// agreement on argument checks.

#ifndef SOMA_SRC_C_HANDLES_HPP_
#define SOMA_SRC_C_HANDLES_HPP_

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "soma/c/soma.h"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/comm.hpp"

namespace soma::c {

/// A value of the wrong data type (SOMA_ERR_TYPE).
struct TypeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
/// A null, consumed or stale handle (SOMA_ERR_STATE).
struct StateError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// The number of live handles (a test hook).
std::atomic<long>& live_handles() noexcept;

/// Counts itself among the live handles for as long as it exists.
struct Live {
  Live() noexcept { ++live_handles(); }
  Live(const Live&) noexcept { ++live_handles(); }
  Live& operator=(const Live&) noexcept = default;
  ~Live() { --live_handles(); }
};

/// Whether soma_finalize finalized Kokkos. Objects freed after that leak their
/// Kokkos memory to the operating system: Kokkos forbids deallocation after
/// finalize.
bool kokkos_finalized() noexcept;

/// Shared ownership of a T that may hold Kokkos memory: deleted with its last
/// owner unless Kokkos is finalized by then.
template <class T>
std::shared_ptr<T> guarded(T&& value) {
  return std::shared_ptr<T>(new T(std::forward<T>(value)), [](T* p) {
    if (!kokkos_finalized()) delete p;
  });
}

/// Records `message` as the calling thread's last error and returns `status`.
int fail(int status, const std::string& message) noexcept;

/// The status of an exception, and its message.
int status_of(const std::exception_ptr& e, std::string& message) noexcept;

/// Throws the exception matching `status` with `message`.
[[noreturn]] void raise(int status, const std::string& message);

/// Throws StateError unless Kokkos is running: before soma_initialize (or the
/// caller's own Kokkos::initialize) and after finalization, no entry point
/// that touches a mesh may run.
void need_runtime();

/// Copies a name, whole and NUL-terminated, into name[len], and its length
/// (without the terminator) into *length; either may be null. A buffer too
/// short for the name is out_of_range: a name is never cut off.
inline void copy_name(const std::string& s, char* name, std::size_t len, std::size_t* length, const char* what) {
  if (length != nullptr) *length = s.size();
  if (name == nullptr) return;
  if (len <= s.size()) {
    throw std::out_of_range(std::string(what) + ": a name of " + std::to_string(s.size()) +
                            " characters does not fit in a buffer of " + std::to_string(len) +
                            " (it needs its length + 1)");
  }
  std::memcpy(name, s.data(), s.size());
  name[s.size()] = '\0';
}

/// The error boundary of the runtime and communicator entry points: runs f,
/// mapping exceptions to status codes.
template <class F>
int call_any(F&& f) noexcept {
  try {
    f();
    return SOMA_OK;
  } catch (...) {
    std::string message;
    const int status = status_of(std::current_exception(), message);
    return fail(status, message);
  }
}

/// The error boundary of every other entry point: call_any, with Kokkos
/// required to be running.
template <class F>
int call(F&& f) noexcept {
  return call_any([&] {
    need_runtime();
    f();
  });
}

/// Runs `check` (argument checks and conversions before a collective step) and
/// agrees on the outcome: when it threw on any rank, every rank throws, with
/// the worst status; ranks that did not fail say so in their message.
template <class F>
void agree(const Comm& comm, const char* what, F&& check) {
  std::exception_ptr failure;
  int status = SOMA_OK;
  std::string message;
  try {
    check();
  } catch (...) {
    failure = std::current_exception();
    status = status_of(failure, message);
  }
  const int worst = comm.allreduce(static_cast<std::int32_t>(status), Op::max);
  if (worst == SOMA_OK) return;
  if (failure && status == worst) std::rethrow_exception(failure);
  raise(worst, std::string(what) + ": " + (failure ? message : "invalid arguments on another rank"));
}

/// The object behind a handle, or StateError for a null handle.
template <class H>
H& need(H* handle, const char* what) {
  if (handle == nullptr) throw StateError(std::string(what) + ": null handle");
  return *handle;
}

/// An output pointer, or invalid_argument when it is null.
template <class T>
T& out(T* p, const char* what) {
  if (p == nullptr) throw std::invalid_argument(std::string(what) + ": null output pointer");
  return *p;
}

/// Throws invalid_argument unless 0 <= n.
void check_count(std::int64_t n, const char* what);
/// Throws invalid_argument when p is null and n > 0 (an empty array may be null).
void check_array(const void* p, std::int64_t n, const char* what);
/// A topology value (SOMA_VERTEX .. SOMA_POLYHEDRON), or invalid_argument.
Topology topology_of(int t, const char* what);
/// An ordering value; SOMA_ORDERING_DEFAULT is the library's default.
Ordering ordering_of(int o, const char* what);
/// Throws invalid_argument unless 0 <= d <= mesh.dim().
void check_dim(const HostMesh& mesh, int d);

/// A dimension's connectivity, 1-based, for Fortran.
struct Conn1 {
  std::vector<LO> conn;
  std::vector<Offset> offsets;
};

/// A mesh and what the C layer keeps beside it, shared by the mesh's handles
/// and by the handles built from it (edits, halo exchanges, remap operators).
struct MeshState {
  std::shared_ptr<HostMesh> mesh;
  /// Bumped by every call that changes topology or numbering.
  std::uint64_t generation = 0;
  /// The 1-based copies, made on first request. Readers on several threads
  /// share one copy: conn1_mutex guards making it. Calls that change the mesh
  /// (and so drop the copies) must not run alongside readers, as in C++.
  std::array<std::optional<Conn1>, 4> conn1;
  std::mutex conn1_mutex;

  explicit MeshState(HostMesh&& m);
  /// Records a change of topology or numbering: drops the 1-based copies.
  void changed();
};

}  // namespace soma::c

struct soma_comm {
  std::shared_ptr<soma::Comm> comm;
  soma::c::Live live;
};

struct soma_builder {
  std::shared_ptr<std::optional<soma::MeshBuilder>> builder;
  soma::c::Live live;
  /// The builder, or StateError once build_local_part consumed it.
  soma::MeshBuilder& get(const char* what) const;
};

struct soma_mesh {
  std::shared_ptr<soma::c::MeshState> state;
  soma::c::Live live;
  soma::HostMesh& get() const { return *state->mesh; }
};

/// A new mesh handle on a mesh the C layer now owns.
soma_mesh_t* soma_c_new_mesh(soma::HostMesh&& mesh);

#endif  // SOMA_SRC_C_HANDLES_HPP_
