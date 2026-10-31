// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_COMM_HPP_
#define SOMA_PARALLEL_COMM_HPP_

// The single communication seam. Algorithms call these unconditionally; in a
// build without MPI they are single-rank identities (rank 0 of 1), so the
// data model and every algorithm are identical in both builds.
//
// This header, src/parallel/comm_*.cpp, and src/runtime/scope_guard.cpp are
// the only files permitted to test SOMA_ENABLE_MPI.

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "soma/config.hpp"

#if SOMA_ENABLE_MPI
#include <mpi.h>
#endif

namespace soma {

enum class Op : std::uint8_t { sum, min, max, prod };

// Non-owning: a Comm never frees the communicator it wraps, which must
// outlive every copy of the Comm.
//
// Reductions are defined for std::int32_t, std::int64_t, std::uint64_t and
// double; other types fail to link.
class Comm {
 public:
  static Comm world();
  static Comm self();

  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int size() const noexcept { return size_; }

  void barrier() const;

  template <class T>
  [[nodiscard]] T allreduce(T value, Op op) const;
  template <class T>
  void allreduce(std::span<T> values, Op op) const;

  // Exclusive prefix sum. Rank 0 receives T{}.
  template <class T>
  [[nodiscard]] T exscan(T value) const;

  template <class T>
  void bcast(std::span<T> buffer, int root) const {
    static_assert(std::is_trivially_copyable_v<T>);
    bcast_bytes(std::as_writable_bytes(buffer), root);
  }

  // Untyped primitive behind bcast.
  void bcast_bytes(std::span<std::byte> buffer, int root) const;

#if SOMA_ENABLE_MPI
  explicit Comm(MPI_Comm comm);
  [[nodiscard]] MPI_Comm native() const noexcept { return comm_; }
#endif

 private:
  Comm(int rank, int size) noexcept : rank_(rank), size_(size) {}

  int rank_ = 0;
  int size_ = 1;
#if SOMA_ENABLE_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
#endif
};

}  // namespace soma

#endif  // SOMA_PARALLEL_COMM_HPP_
