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
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "soma/config.hpp"

#if SOMA_ENABLE_MPI
#include <mpi.h>
#endif

namespace soma {

enum class Op : std::uint8_t { sum, min, max, prod };

namespace detail {
struct RequestState;

// What this process has called since it started, for tests that check how
// many collectives an operation makes: `collectives` counts every collective
// call of Comm (barrier, allreduce, exscan, bcast, alltoall, alltoallv,
// ibarrier), `routes` every transport::route.
struct CommCounts {
  std::uint64_t collectives = 0;
  std::uint64_t routes = 0;
};
[[nodiscard]] CommCounts comm_counts() noexcept;
void count_collective() noexcept;
void count_route() noexcept;
}  // namespace detail

// A non-blocking operation in flight. Move-only; destroying one that has
// not completed waits for it. Tests and waits throw std::runtime_error when
// MPI fails, and in the build without MPI std::logic_error when waiting for
// a message nothing sends (it would wait forever).
class Request {
 public:
  Request() noexcept;
  Request(Request&& other) noexcept;
  Request& operator=(Request&& other) noexcept;
  Request(const Request&) = delete;
  Request& operator=(const Request&) = delete;
  ~Request();

  // A default-constructed or finished request is complete.
  [[nodiscard]] bool test();
  void wait();
  static void wait_all(std::span<Request> requests);

 private:
  friend class Comm;
#if SOMA_ENABLE_MPI
  MPI_Request request_;
#else
  std::shared_ptr<detail::RequestState> state_;
#endif
};

// What iprobe finds: the next message's sender, tag and size in bytes.
struct Envelope {
  int source = 0;
  int tag = 0;
  std::size_t bytes = 0;
};

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
  // Exclusive prefix sums of each entry, in place, in one call.
  template <class T>
  void exscan(std::span<T> values) const;

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

  // Every rank's `mine`, concatenated in rank order, on every rank; with
  // `counts`, also how many elements each rank gave. One collective.
  template <class T>
  [[nodiscard]] std::vector<T> allgatherv(std::span<const T> mine, std::vector<std::size_t>* counts = nullptr) const {
    static_assert(std::is_trivially_copyable_v<T>);
    std::vector<std::size_t> n;
    const std::vector<std::byte> all = allgatherv_bytes(std::as_bytes(mine), sizeof(T), n);
    std::vector<T> out(all.size() / sizeof(T));
    if (!all.empty()) std::memcpy(static_cast<void*>(out.data()), all.data(), all.size());
    if (counts != nullptr) *counts = std::move(n);
    return out;
  }

  // Untyped primitives behind alltoall, alltoallv and allgatherv; counts are
  // in elements of `width` bytes.
  void alltoall_bytes(std::span<const std::byte> send, std::span<std::byte> recv, std::size_t width) const;
  void alltoallv_bytes(std::span<const std::byte> send, std::span<const std::size_t> send_counts,
                       std::span<std::byte> recv, std::span<const std::size_t> recv_counts, std::size_t width) const;
  [[nodiscard]] std::vector<std::byte> allgatherv_bytes(std::span<const std::byte> mine, std::size_t width,
                                                       std::vector<std::size_t>& counts) const;

  // Point-to-point messages of bytes; the buffer must stay valid, and a
  // receive buffer untouched, until the request completes. isend may
  // complete before the message is received, issend only once it is. A
  // message longer than the receive buffer is an error. Tags are in
  // [0, 32767], the least MPI guarantees. Without MPI, rank 0 sends to
  // itself through a queue matched by tag in send order.
  static constexpr int any_source = -1;
  [[nodiscard]] Request isend_bytes(std::span<const std::byte> buffer, int dest, int tag) const;
  [[nodiscard]] Request issend_bytes(std::span<const std::byte> buffer, int dest, int tag) const;
  [[nodiscard]] Request irecv_bytes(std::span<std::byte> buffer, int source, int tag) const;
  template <class T>
  [[nodiscard]] Request isend(std::span<const T> buffer, int dest, int tag) const {
    static_assert(std::is_trivially_copyable_v<T>);
    return isend_bytes(std::as_bytes(buffer), dest, tag);
  }
  template <class T>
  [[nodiscard]] Request issend(std::span<const T> buffer, int dest, int tag) const {
    static_assert(std::is_trivially_copyable_v<T>);
    return issend_bytes(std::as_bytes(buffer), dest, tag);
  }
  template <class T>
  [[nodiscard]] Request irecv(std::span<T> buffer, int source, int tag) const {
    static_assert(std::is_trivially_copyable_v<T>);
    return irecv_bytes(std::as_writable_bytes(buffer), source, tag);
  }
  // The next message from `source` (or any_source) with `tag`, if one has
  // arrived; it stays queued until received.
  [[nodiscard]] std::optional<Envelope> iprobe(int source, int tag) const;
  // Completes once every rank has called it.
  [[nodiscard]] Request ibarrier() const;

  // A count kept per communicator (not per Comm object), returned and
  // advanced by each call: ranks that call it equally often agree on it,
  // which lets successive exchanges use different tags.
  [[nodiscard]] std::uint64_t next_sequence() const;

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
