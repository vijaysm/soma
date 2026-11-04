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

  // Block r of `send` (send.size() / size() elements) to rank r, block s of
  // `recv` from rank s. Throws std::invalid_argument unless both sizes are
  // the same multiple of size().
  template <class T>
  void alltoall(std::span<const T> send, std::span<T> recv) const {
    static_assert(std::is_trivially_copyable_v<T>);
    alltoall_bytes(std::as_bytes(send), std::as_writable_bytes(recv), sizeof(T));
  }
  // send_counts[r] elements to rank r and recv_counts[s] from rank s, each
  // contiguous in rank order. Throws std::invalid_argument unless there is a
  // count per rank and the counts add up to the buffer sizes.
  template <class T>
  void alltoallv(std::span<const T> send, std::span<const std::size_t> send_counts, std::span<T> recv,
                 std::span<const std::size_t> recv_counts) const {
    static_assert(std::is_trivially_copyable_v<T>);
    alltoallv_bytes(std::as_bytes(send), send_counts, std::as_writable_bytes(recv), recv_counts, sizeof(T));
  }

  // Untyped primitives behind alltoall and alltoallv; counts are in elements
  // of `width` bytes.
  void alltoall_bytes(std::span<const std::byte> send, std::span<std::byte> recv, std::size_t width) const;
  void alltoallv_bytes(std::span<const std::byte> send, std::span<const std::size_t> send_counts,
                       std::span<std::byte> recv, std::span<const std::size_t> recv_counts, std::size_t width) const;

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
