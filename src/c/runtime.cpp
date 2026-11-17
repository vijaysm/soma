// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's runtime: errors, initialization and communicators.

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "handles.hpp"

namespace soma::c {

namespace {

thread_local std::string last_message;
std::atomic<bool> finalized{false};
std::atomic<bool> initialized_here{false};

}  // namespace

std::atomic<long>& live_handles() noexcept {
  static std::atomic<long> count{0};
  return count;
}

bool kokkos_finalized() noexcept { return finalized.load(); }

void need_runtime() {
  if (finalized || Kokkos::is_finalized()) throw StateError("SOMA was finalized; no call may follow soma_finalize");
  if (!Kokkos::is_initialized()) throw StateError("SOMA is not initialized: call soma_initialize first");
}

int fail(int status, const std::string& message) noexcept {
  try {
    last_message = message;
  } catch (...) {
    last_message.clear();  // out of memory: no message
  }
  return status;
}

int status_of(const std::exception_ptr& e, std::string& message) noexcept {
  try {
    std::rethrow_exception(e);
  } catch (const TypeError& x) {
    message = x.what();
    return SOMA_ERR_TYPE;
  } catch (const StateError& x) {
    message = x.what();
    return SOMA_ERR_STATE;
  } catch (const std::invalid_argument& x) {
    message = x.what();
    return SOMA_ERR_INVALID_ARGUMENT;
  } catch (const std::out_of_range& x) {
    message = x.what();
    return SOMA_ERR_OUT_OF_RANGE;
  } catch (const std::exception& x) {
    message = x.what();
    return SOMA_ERR_RUNTIME;
  } catch (...) {
    message = "unknown error";
    return SOMA_ERR_RUNTIME;
  }
}

void raise(int status, const std::string& message) {
  switch (status) {
    case SOMA_ERR_INVALID_ARGUMENT:
      throw std::invalid_argument(message);
    case SOMA_ERR_OUT_OF_RANGE:
      throw std::out_of_range(message);
    case SOMA_ERR_TYPE:
      throw TypeError(message);
    case SOMA_ERR_STATE:
      throw StateError(message);
    default:
      throw std::runtime_error(message);
  }
}

namespace {

soma_comm_t* new_comm(soma::Comm comm) { return new soma_comm{std::make_shared<soma::Comm>(std::move(comm)), {}}; }

}  // namespace

}  // namespace soma::c

using soma::c::call_any;
using soma::c::need;
using soma::c::out;

extern "C" {

size_t soma_last_error(char* buf, size_t len) {
  const std::string& m = soma::c::last_message;
  if (buf != nullptr && len > 0) {
    const std::size_t n = std::min(len - 1, m.size());
    std::memcpy(buf, m.data(), n);
    buf[n] = '\0';
  }
  return m.size();
}

int soma_set_last_error(int status, const char* message) {
  return soma::c::fail(status, message == nullptr ? std::string() : std::string(message));
}

int soma_initialize(void) {
  return call_any([] {
    if (soma::c::finalized) throw soma::c::StateError("soma_initialize: SOMA was finalized; Kokkos cannot restart");
    if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
      Kokkos::initialize(Kokkos::InitializationSettings().set_disable_warnings(true));
      soma::c::initialized_here = true;
    }
  });
}

int soma_finalize(void) {
  return call_any([] {
    if (soma::c::initialized_here && !soma::c::finalized) {
      soma::c::finalized = true;  // first: objects freed from here on leak instead
      Kokkos::finalize();
    }
  });
}

int soma_testing_live_handles(long* count) {
  return call_any([&] { out(count, "soma_testing_live_handles") = soma::c::live_handles().load(); });
}

int soma_testing_raise(int status) {
  return call_any([&] {
    if (status != SOMA_OK) soma::c::raise(status, "soma test error: status " + std::to_string(status));
  });
}

#if SOMA_C_HAVE_MPI
int soma_comm_create(MPI_Comm comm, soma_comm_t** result) {
  return call_any([&] {
    soma_comm_t*& r = out(result, "soma_comm_create");
    r = nullptr;
    if (comm == MPI_COMM_NULL) throw std::invalid_argument("soma_comm_create: MPI_COMM_NULL is not a communicator");
    r = soma::c::new_comm(soma::Comm(comm));
  });
}
#endif

int soma_comm_create_f(int fcomm, soma_comm_t** result) {
#if SOMA_C_HAVE_MPI
  return soma_comm_create(MPI_Comm_f2c(static_cast<MPI_Fint>(fcomm)), result);
#else
  (void)fcomm;
  return call_any([&] {
    out(result, "soma_comm_create_f") = nullptr;
    throw std::invalid_argument("soma_comm_create_f: SOMA was built without MPI; use soma_comm_self");
  });
#endif
}

int soma_comm_world(soma_comm_t** result) {
  return call_any([&] { out(result, "soma_comm_world") = soma::c::new_comm(soma::Comm::world()); });
}

int soma_comm_self(soma_comm_t** result) {
  return call_any([&] { out(result, "soma_comm_self") = soma::c::new_comm(soma::Comm::self()); });
}

int soma_comm_rank(const soma_comm_t* comm, int* rank) {
  return call_any([&] {
    const soma_comm& c = need(comm, "soma_comm_rank");
    out(rank, "soma_comm_rank") = c.comm->rank();
  });
}

int soma_comm_size(const soma_comm_t* comm, int* size) {
  return call_any([&] {
    const soma_comm& c = need(comm, "soma_comm_size");
    out(size, "soma_comm_size") = c.comm->size();
  });
}

int soma_comm_barrier(const soma_comm_t* comm) {
  return call_any([&] { need(comm, "soma_comm_barrier").comm->barrier(); });
}

int soma_comm_clone(const soma_comm_t* comm, soma_comm_t** result) {
  return call_any([&] {
    const soma_comm& c = need(comm, "soma_comm_clone");
    out(result, "soma_comm_clone") = new soma_comm{c.comm, {}};
  });
}

void soma_comm_free(soma_comm_t* comm) { delete comm; }

}  // extern "C"
