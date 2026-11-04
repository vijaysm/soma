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
  CommPlan() = default;
  // Throws std::invalid_argument for a destination outside the communicator,
  // sources that are not grouped in ascending rank order, or a different
  // count of records to this rank among the sent and the arrived.
  CommPlan(const Comm& comm, std::span<const int> sent_dest, std::span<const int> arrived_source);

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
  // This rank's records to itself: how many, and where they start among the
  // packed sent records and among the arrived.
  std::size_t self_count_ = 0, self_packed_ = 0, self_arrived_ = 0;
};

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
