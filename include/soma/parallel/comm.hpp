// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_COMM_HPP_
#define SOMA_PARALLEL_COMM_HPP_

// The single communication seam. Algorithms call these unconditionally; in a
// build without MPI they are single-rank identities (rank 0 of 1), so the
// data model and every algorithm are identical in both builds.
//
// This header, src/parallel/comm_*.cpp, and src/runtime/scope_guard.cpp are
// the only files permitted to test SOMA_ENABLE_MPI (enforced by the
// policy.mpi_seam test).

#include <cstddef>
#include <cstdint>
#include <span>

#include "soma/config.hpp"

#if SOMA_ENABLE_MPI
#include <mpi.h>
#endif

namespace soma {

enum class Op : std::uint8_t { sum, min, max, prod };

// Non-owning: a Comm never frees the communicator it wraps.
class Comm {
 public:
  static Comm world();
  static Comm self();

  int rank() const noexcept { return rank_; }
  int size() const noexcept { return size_; }

  void barrier() const;

  std::int32_t allreduce(std::int32_t value, Op op) const;
  std::int64_t allreduce(std::int64_t value, Op op) const;
  double allreduce(double value, Op op) const;
  void allreduce(std::span<std::int64_t> values, Op op) const;
  void allreduce(std::span<double> values, Op op) const;

  // Exclusive prefix sum. Rank 0 receives 0.
  std::int64_t exscan(std::int64_t value) const;

  void bcast(std::span<std::byte> buffer, int root) const;

#if SOMA_ENABLE_MPI
  explicit Comm(MPI_Comm comm);
  MPI_Comm native() const noexcept { return comm_; }
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
