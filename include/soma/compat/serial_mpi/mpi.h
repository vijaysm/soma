// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_COMPAT_SERIAL_MPI_H_
#define SOMA_COMPAT_SERIAL_MPI_H_

// A one-rank stand-in for the MPI calls MOAB's serial examples make, on the
// drop-in include path of a SOMA built without MPI. Every communicator has
// one rank; collectives copy their input.

#include <cstdlib>
#include <cstring>

using MPI_Comm = int;
using MPI_Datatype = int;
using MPI_Op = int;

inline constexpr MPI_Comm MPI_COMM_WORLD = 0;
inline constexpr MPI_Comm MPI_COMM_SELF = 1;
inline constexpr int MPI_SUCCESS = 0;
inline constexpr MPI_Op MPI_SUM = 0;
inline constexpr MPI_Op MPI_MAX = 1;
inline constexpr MPI_Op MPI_MIN = 2;
inline constexpr MPI_Datatype MPI_CHAR = 1;
inline constexpr MPI_Datatype MPI_BYTE = 1;
inline constexpr MPI_Datatype MPI_UNSIGNED_CHAR = 1;
inline constexpr MPI_Datatype MPI_INT = 4;
inline constexpr MPI_Datatype MPI_UNSIGNED = 4;
inline constexpr MPI_Datatype MPI_FLOAT = 4;
inline constexpr MPI_Datatype MPI_LONG = 8;
inline constexpr MPI_Datatype MPI_UNSIGNED_LONG = 8;
inline constexpr MPI_Datatype MPI_LONG_LONG = 8;
inline constexpr MPI_Datatype MPI_UNSIGNED_LONG_LONG = 8;
inline constexpr MPI_Datatype MPI_DOUBLE = 8;

namespace soma_serial_mpi {
inline int& state() {
  static int s = 0;  // 0 before MPI_Init, 1 after, 2 after MPI_Finalize
  return s;
}
}  // namespace soma_serial_mpi

inline int MPI_Init(int*, char***) {
  soma_serial_mpi::state() = 1;
  return MPI_SUCCESS;
}
inline int MPI_Initialized(int* flag) {
  *flag = soma_serial_mpi::state() >= 1 ? 1 : 0;
  return MPI_SUCCESS;
}
inline int MPI_Finalize() {
  soma_serial_mpi::state() = 2;
  return MPI_SUCCESS;
}
inline int MPI_Finalized(int* flag) {
  *flag = soma_serial_mpi::state() == 2 ? 1 : 0;
  return MPI_SUCCESS;
}
inline int MPI_Comm_rank(MPI_Comm, int* rank) {
  *rank = 0;
  return MPI_SUCCESS;
}
inline int MPI_Comm_size(MPI_Comm, int* size) {
  *size = 1;
  return MPI_SUCCESS;
}
inline int MPI_Barrier(MPI_Comm) { return MPI_SUCCESS; }
inline int MPI_Bcast(void*, int, MPI_Datatype, int, MPI_Comm) { return MPI_SUCCESS; }
inline int MPI_Allreduce(const void* in, void* out, int count, MPI_Datatype type, MPI_Op, MPI_Comm) {
  if (in != out) std::memcpy(out, in, static_cast<std::size_t>(count) * static_cast<std::size_t>(type));
  return MPI_SUCCESS;
}
inline int MPI_Reduce(const void* in, void* out, int count, MPI_Datatype type, MPI_Op op, int, MPI_Comm comm) {
  return MPI_Allreduce(in, out, count, type, op, comm);
}
[[noreturn]] inline int MPI_Abort(MPI_Comm, int code) { std::exit(code); }

#endif  // SOMA_COMPAT_SERIAL_MPI_H_
