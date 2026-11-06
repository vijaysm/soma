// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_TRANSPORT_COMM_PLAN_HPP_
#define SOMA_PARALLEL_TRANSPORT_COMM_PLAN_HPP_

// The steady state of an exchange found by route(): who sends how many
// records to whom, replayed with one round of point-to-point messages to
// the known neighbours. A plan is learnt from both halves of one routed
// exchange, the destinations a rank sent and the sources it received, so no
// handshake is needed:
//
//   auto arrived = transport::route(exec, comm, sent);
//   const auto plan = transport::learn_plan(comm, sent, arrived);
//   plan.forward(comm, values_per_sent_record, values_per_arrived_record);
//   plan.reverse(comm, answers_per_arrived_record, answers_per_sent_record);
//
// forward moves one value per sent record to the rank it was routed to, in
// route()'s order there (by source rank, then the sender's order); reverse
// sends one value per arrived record back to its place among the sender's
// records. A rank's records to itself are copied. Messages use tags 32600
// and 32601 of the communicator; both are collective over the neighbours.
//
// A plan can also be made from counts alone, when both sides already know
// them and hold their records grouped by rank (from_counts); each
// neighbour's run then leaves as one message without being copied.
// begin_forward starts a forward on a tag of the caller's choosing and
// returns a Pending whose wait() (or destructor) finishes it, so several
// exchanges can be in flight and work can overlap them; the spans must
// outlive the Pending.

#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/parallel/comm.hpp"
#include "soma/parallel/transport/route.hpp"

namespace soma::transport {

class CommPlan {
 public:
  // A rank and how many records go to it or come from it.
  struct Run {
    int rank = 0;
    std::size_t count = 0;
  };
  class Pending;

  static constexpr int forward_tag = 32600;
  static constexpr int reverse_tag = 32601;

  CommPlan() = default;
  // Throws std::invalid_argument for a destination outside the communicator,
  // sources that are not grouped in ascending rank order, or a different
  // count of records to this rank among the sent and the arrived.
  CommPlan(const Comm& comm, std::span<const int> sent_dest, std::span<const int> arrived_source);
  // Sent records grouped by destination and arrived records by source, the
  // ranks of each list strictly ascending (runs of no records allowed).
  // Throws std::invalid_argument for a rank outside the communicator, ranks
  // not strictly ascending, or a different count to this rank sent and
  // arrived.
  [[nodiscard]] static CommPlan from_counts(const Comm& comm, std::span<const Run> sent, std::span<const Run> arrived);

  // The other ranks this one sends to and receives from, ascending.
  [[nodiscard]] std::vector<int> send_ranks() const;
  [[nodiscard]] std::vector<int> recv_ranks() const;
  [[nodiscard]] std::size_t num_sent() const { return send_perm_.size(); }
  [[nodiscard]] std::size_t num_arrived() const { return recv_offsets_.empty() ? 0 : recv_offsets_.back(); }

  // Values of `width` bytes. Throws std::invalid_argument, before any
  // message, unless the spans hold a value per record.
  void forward_bytes(const Comm& comm, std::span<const std::byte> sent, std::span<std::byte> arrived,
                     std::size_t width) const;
  void reverse_bytes(const Comm& comm, std::span<const std::byte> arrived, std::span<std::byte> sent,
                     std::size_t width) const;

  // Posts the receives, sends and copies this rank's own records; the
  // Pending finishes the exchange. Throws std::invalid_argument, before any
  // message, unless the spans hold a value per record.
  [[nodiscard]] Pending begin_forward_bytes(const Comm& comm, std::span<const std::byte> sent,
                                            std::span<std::byte> arrived, std::size_t width, int tag) const;

  template <class T>
  [[nodiscard]] Pending begin_forward(const Comm& comm, std::span<const T> sent, std::span<T> arrived, int tag) const;
  template <class T>
  void forward(const Comm& comm, std::span<const T> sent, std::span<T> arrived) const {
    static_assert(std::is_trivially_copyable_v<T>);
    forward_bytes(comm, std::as_bytes(sent), std::as_writable_bytes(arrived), sizeof(T));
  }
  template <class T>
  void reverse(const Comm& comm, std::span<const T> arrived, std::span<T> sent) const {
    static_assert(std::is_trivially_copyable_v<T>);
    reverse_bytes(comm, std::as_bytes(arrived), std::as_writable_bytes(sent), sizeof(T));
  }

 private:
  int rank_ = 0;
  // Every rank sent to or received from, this one included, ascending, with
  // offsets into the sent records packed by destination and into the
  // arrived records.
  std::vector<int> send_peers_, recv_peers_;
  std::vector<std::size_t> send_offsets_, recv_offsets_;
  std::vector<std::size_t> send_perm_;  // packed position -> sent record
  bool grouped_ = false;                // send_perm_ is the identity
  // This rank's records to itself: how many, and where they start among the
  // packed sent records and among the arrived.
  std::size_t self_count_ = 0, self_packed_ = 0, self_arrived_ = 0;
};

// A forward in flight. wait() finishes it (a second call does nothing); the
// destructor waits too.
class CommPlan::Pending {
 public:
  Pending() = default;
  void wait() { Request::wait_all(requests_); }

 private:
  friend class CommPlan;
  // Declared before the requests, so a destroyed Pending waits for them
  // before freeing the buffer they send.
  std::vector<std::byte> packed_;
  std::vector<Request> requests_;
};

template <class T>
CommPlan::Pending CommPlan::begin_forward(const Comm& comm, std::span<const T> sent, std::span<T> arrived, int tag) const {
  static_assert(std::is_trivially_copyable_v<T>);
  return begin_forward_bytes(comm, std::as_bytes(sent), std::as_writable_bytes(arrived), sizeof(T), tag);
}

// The plan of the exchange that routed `sent` into `arrived`.
template <class MemSpace>
CommPlan learn_plan(const Comm& comm, const RecordBatch<MemSpace>& sent, const RecordBatch<MemSpace>& arrived) {
  const auto dest = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sent.dest);
  const auto source = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, arrived.source);
  return CommPlan(comm, std::span<const int>(dest.data(), dest.extent(0)),
                  std::span<const int>(source.data(), source.extent(0)));
}

}  // namespace soma::transport

#endif  // SOMA_PARALLEL_TRANSPORT_COMM_PLAN_HPP_
