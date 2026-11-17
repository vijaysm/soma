// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's internals: handle structs, the error boundary, and collective
// agreement on argument checks.

#ifndef SOMA_SRC_C_HANDLES_HPP_
#define SOMA_SRC_C_HANDLES_HPP_

#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "soma/c/soma.h"
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

/// The error boundary of every entry point: runs f, mapping exceptions to
/// status codes.
template <class F>
int call(F&& f) noexcept {
  try {
    f();
    return SOMA_OK;
  } catch (...) {
    std::string message;
    const int status = status_of(std::current_exception(), message);
    return fail(status, message);
  }
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

}  // namespace soma::c

struct soma_comm {
  std::shared_ptr<soma::Comm> comm;
  soma::c::Live live;
};

#endif  // SOMA_SRC_C_HANDLES_HPP_
