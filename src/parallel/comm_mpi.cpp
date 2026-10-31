// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/comm.hpp"

#include <climits>
#include <limits>
#include <stdexcept>
#include <string>

#if !SOMA_ENABLE_MPI
#error "comm_mpi.cpp requires SOMA_ENABLE_MPI"
#endif

namespace soma {

namespace {

void check(int rc, const char* what) {
  if (rc != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING] = {};
    int len = 0;
    MPI_Error_string(rc, msg, &len);
    throw std::runtime_error(std::string("soma::Comm: ") + what + " failed: " +
                             std::string(msg, static_cast<std::size_t>(len)));
  }
}

int checked_count(std::size_t n) {
  if (n > static_cast<std::size_t>(INT_MAX)) {
    throw std::length_error("soma::Comm: message exceeds INT_MAX elements");
  }
  return static_cast<int>(n);
}

MPI_Op to_mpi(Op op) {
  switch (op) {
    case Op::sum: return MPI_SUM;
    case Op::min: return MPI_MIN;
    case Op::max: return MPI_MAX;
    case Op::prod: return MPI_PROD;
  }
  return MPI_OP_NULL;
}

template <class T> MPI_Datatype datatype();
template <> MPI_Datatype datatype<std::int32_t>() { return MPI_INT32_T; }
template <> MPI_Datatype datatype<std::int64_t>() { return MPI_INT64_T; }
template <> MPI_Datatype datatype<std::uint64_t>() { return MPI_UINT64_T; }
template <> MPI_Datatype datatype<double>() { return MPI_DOUBLE; }

template <class T>
T reduce_one(MPI_Comm comm, T value, Op op) {
  T out{};
  check(MPI_Allreduce(&value, &out, 1, datatype<T>(), to_mpi(op), comm), "MPI_Allreduce");
  return out;
}

template <class T>
void reduce_span(MPI_Comm comm, std::span<T> values, Op op) {
  check(MPI_Allreduce(MPI_IN_PLACE, values.data(), checked_count(values.size()),
                      datatype<T>(), to_mpi(op), comm),
        "MPI_Allreduce");
}

}  // namespace

Comm::Comm(MPI_Comm comm) : comm_(comm) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::runtime_error(
        "soma::Comm used before MPI_Init; construct a soma::ScopeGuard first");
  }
  check(MPI_Comm_rank(comm, &rank_), "MPI_Comm_rank");
  check(MPI_Comm_size(comm, &size_), "MPI_Comm_size");
}

Comm Comm::world() { return Comm(MPI_COMM_WORLD); }
Comm Comm::self() { return Comm(MPI_COMM_SELF); }

void Comm::barrier() const { check(MPI_Barrier(comm_), "MPI_Barrier"); }

template <class T>
T Comm::allreduce(T value, Op op) const {
  return reduce_one(comm_, value, op);
}

template <class T>
void Comm::allreduce(std::span<T> values, Op op) const {
  reduce_span(comm_, values, op);
}

template <class T>
T Comm::exscan(T value) const {
  T out = std::numeric_limits<T>::max();
  check(MPI_Exscan(&value, &out, 1, datatype<T>(), MPI_SUM, comm_), "MPI_Exscan");
  // MPI leaves rank 0's receive buffer undefined.
  return rank_ == 0 ? T{} : out;
}

void Comm::bcast_bytes(std::span<std::byte> buffer, int root) const {
  check(MPI_Bcast(buffer.data(), checked_count(buffer.size()), MPI_BYTE, root, comm_),
        "MPI_Bcast");
}

#define SOMA_INSTANTIATE_COMM(T)                              \
  template T Comm::allreduce<T>(T, Op) const;                 \
  template void Comm::allreduce<T>(std::span<T>, Op) const;   \
  template T Comm::exscan<T>(T) const;

SOMA_INSTANTIATE_COMM(std::int32_t)
SOMA_INSTANTIATE_COMM(std::int64_t)
SOMA_INSTANTIATE_COMM(std::uint64_t)
SOMA_INSTANTIATE_COMM(double)

#undef SOMA_INSTANTIATE_COMM

}  // namespace soma
