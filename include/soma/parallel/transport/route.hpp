// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_TRANSPORT_ROUTE_HPP_
#define SOMA_PARALLEL_TRANSPORT_ROUTE_HPP_

// Sparse personalized exchange: every rank holds records, each naming the
// rank it goes to, and route() delivers them. Records are fixed-width rows
// of bytes, the width the same on every rank.
//
//   transport::RecordBatch<Kokkos::HostSpace> out(n, sizeof(Answer));
//   ... fill out.dest and out.payload ...
//   auto in = transport::route(Kokkos::DefaultHostExecutionSpace{}, comm, out);
//   // in.source(i) sent row i; rows are grouped by source rank, in order.
//
// Three strategies return the same records in the same order:
//   crystal    the crystal router (Tufo and Fischer; gslib, BSD-3): the
//              active ranks are halved ceil(log2 P) times, each rank sending
//              the records bound for the other half to one partner (with an
//              odd count, the last rank of the lower half sends to the first
//              of the upper half, which so receives from two). O(log P)
//              messages per rank and no P-sized arrays; records travel up to
//              log2 P hops.
//   alltoallv  counts by alltoall, then one MPI_Alltoallv.
//   nbx        a synchronous send per destination, then a non-blocking
//              barrier once they are received (Hoefler, Siebert and
//              Lumsdaine 2010); no forwarding and no P-sized counts.
// `automatic` takes alltoallv up to 64 ranks and the crystal router above,
// a placeholder until measurements at scale set the threshold.
//
// Records are staged through host memory: one copy in and one out, whatever
// the memory space. Messages use tags 32000 to 32767 of the communicator.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/parallel/comm.hpp"

namespace soma::transport {

enum class Strategy : std::uint8_t { automatic, crystal, alltoallv, nbx };

template <class MemSpace>
struct RecordBatch {
  Kokkos::View<int*, MemSpace> dest;    // the rank each record goes to
  Kokkos::View<int*, MemSpace> source;  // the rank it came from, filled by route()
  Kokkos::View<std::byte**, Kokkos::LayoutRight, MemSpace> payload;  // one row per record

  RecordBatch() = default;
  RecordBatch(std::size_t n, std::size_t width, const std::string& label = "transport::RecordBatch")
      : dest(label + ".dest", n), source(label + ".source", n), payload(label + ".payload", n, width) {}

  [[nodiscard]] std::size_t size() const { return dest.extent(0); }
  [[nodiscard]] std::size_t width() const { return payload.extent(1); }
};

namespace detail {

// The records that arrived here from other ranks, grouped by source rank and
// in each source's order: rows of `stride` bytes, the destination and source
// ranks (two std::int32_t) then the payload; the first `before` rows come
// from lower ranks. The records this rank sent itself do not travel: `self`
// holds runs (first, count) of their positions in what it sent, in order,
// and they arrive between the lower ranks' and the higher ranks' rows.
struct Arrivals {
  std::size_t stride = 0;
  std::vector<std::byte> bytes;
  std::size_t before = 0;
  std::vector<std::pair<std::size_t, std::size_t>> self;

  [[nodiscard]] std::size_t num_self() const {
    std::size_t n = 0;
    for (const auto& r : self) n += r.second;
    return n;
  }

  static constexpr std::size_t header = 2 * sizeof(std::int32_t);
  [[nodiscard]] std::size_t size() const { return stride == 0 ? 0 : bytes.size() / stride; }
};

[[nodiscard]] Strategy resolve(Strategy s, int size);

// Checks `bytes` received when `held` rows are already here, before any
// buffer is sized from it: whole rows of `stride` bytes, and no more than
// `limit` rows held in all (P times the largest batch of any rank, which no
// exchange can exceed). Throws std::runtime_error otherwise, so a corrupt or
// mismatched message cannot drive an unbounded allocation.
void check_incoming(std::size_t held, std::uint64_t bytes, std::size_t stride, std::size_t limit);

// Collective. Throws std::invalid_argument on every rank if any rank names
// a destination outside the communicator, passes a payload that is not a
// row of `width` bytes per record, or differs from another rank in width or
// strategy. Throws std::runtime_error from check_incoming if a message is
// inconsistent with what the ranks sent.
[[nodiscard]] Arrivals route_host(const Comm& comm, std::span<const int> dest, std::span<const std::byte> payload,
                                  std::size_t width, Strategy strategy);

}  // namespace detail

// Collective: the records of every rank's `in` bound for this rank, dest
// set to this rank and source to the sender's. Throws as route_host.
template <class ExecSpace>
RecordBatch<typename ExecSpace::memory_space> route(const ExecSpace&, const Comm& comm,
                                                    const RecordBatch<typename ExecSpace::memory_space>& in,
                                                    Strategy strategy = Strategy::automatic) {
  using MemSpace = typename ExecSpace::memory_space;
  const auto dest = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, in.dest);
  const auto payload = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, in.payload);
  const std::size_t width = in.width();
  const detail::Arrivals a =
      detail::route_host(comm, std::span<const int>(dest.data(), dest.extent(0)),
                         std::span<const std::byte>(payload.data(), payload.extent(0) * width), width, strategy);
  const std::size_t own = a.num_self(), n = a.size() + own;
  RecordBatch<MemSpace> out;
  out.dest = decltype(out.dest)(Kokkos::view_alloc(Kokkos::WithoutInitializing, "transport::route.dest"), n);
  out.source = decltype(out.source)(Kokkos::view_alloc(Kokkos::WithoutInitializing, "transport::route.source"), n);
  out.payload =
      decltype(out.payload)(Kokkos::view_alloc(Kokkos::WithoutInitializing, "transport::route.payload"), n, width);
  const auto hd = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, out.dest);
  const auto hs = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, out.source);
  const auto hp = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, out.payload);
  const auto arrived = [&](std::size_t i, std::size_t at) {
    const std::byte* row = a.bytes.data() + i * a.stride;
    hd(at) = comm.rank();
    std::memcpy(&hs(at), row + sizeof(std::int32_t), sizeof(std::int32_t));
    if (width > 0) std::memcpy(hp.data() + at * width, row + detail::Arrivals::header, width);
  };
  for (std::size_t i = 0; i < a.before; ++i) arrived(i, i);
  std::size_t at = a.before;
  for (const auto& [first, count] : a.self) {
    if (width > 0) std::memcpy(hp.data() + at * width, payload.data() + first * width, count * width);
    at += count;
  }
  for (std::size_t i = a.before; i < a.before + own; ++i) {
    hd(i) = comm.rank();
    hs(i) = comm.rank();
  }
  for (std::size_t i = a.before; i < a.size(); ++i) arrived(i, i + own);
  Kokkos::deep_copy(out.dest, hd);
  Kokkos::deep_copy(out.source, hs);
  Kokkos::deep_copy(out.payload, hp);
  return out;
}

}  // namespace soma::transport

#endif  // SOMA_PARALLEL_TRANSPORT_ROUTE_HPP_
