// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_TRANSPORT_RENDEZVOUS_HPP_
#define SOMA_PARALLEL_TRANSPORT_RENDEZVOUS_HPP_

// A distributed directory: each record goes to the owner of its key,
// owner(key, P) = mix(key) % P, which sees every record sent for its keys
// and answers each one; the answers come back in the order of the keys.
//
//   auto answers = transport::rendezvous(exec, comm, gids, payload, sizeof(int),
//       [](const auto& keys, const auto& arrived) { ... return rows; });
//
// answer(keys, arrived) receives the keys that arrived here (a
// Kokkos::View<std::uint64_t*>) and a RecordBatch of their senders and
// payloads, grouped by source rank in each source's order, and returns a
// row of `answer_width` bytes per arrival. mix is the splitmix64 finalizer
// (Steele, Lea and Flood 2014), so a key's owner depends only on the key
// and P.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "soma/parallel/comm.hpp"
#include "soma/parallel/transport/comm_plan.hpp"
#include "soma/parallel/transport/route.hpp"

namespace soma::transport {

[[nodiscard]] constexpr std::uint64_t mix(std::uint64_t z) noexcept {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

[[nodiscard]] constexpr int owner(std::uint64_t key, int size) noexcept {
  return static_cast<int>(mix(key) % static_cast<std::uint64_t>(size));
}

// rendezvous_to sends each key to dest_of(key), a rank of comm, instead of
// its owner: a directory placed by the caller (blocks of ids, say).
//
// Collective. Throws std::invalid_argument on every rank if any rank passes
// a payload without a row per key, names a rank outside comm, or returns
// answers that are not a row of answer_width bytes per arrival; an exception
// from answer on one rank is rethrown there and std::runtime_error is thrown
// on the others.
template <class ExecSpace, class Answer, class DestOf>
Kokkos::View<std::byte**, Kokkos::LayoutRight, typename ExecSpace::memory_space> rendezvous_to(
    const ExecSpace&, const Comm& comm, const Kokkos::View<const std::uint64_t*, typename ExecSpace::memory_space>& keys,
    const Kokkos::View<const std::byte**, Kokkos::LayoutRight, typename ExecSpace::memory_space>& payload,
    std::size_t answer_width, const Answer& answer, const DestOf& dest_of, Strategy strategy = Strategy::automatic) {
  using MemSpace = typename ExecSpace::memory_space;
  using Host = Kokkos::DefaultHostExecutionSpace;
  constexpr std::size_t key_bytes = sizeof(std::uint64_t);
  const std::size_t n = keys.extent(0), width = payload.extent(1);
  if (comm.allreduce(static_cast<std::int32_t>(payload.extent(0) != n), Op::max)) {
    throw std::invalid_argument("transport::rendezvous: a rank's payload does not have a row per key");
  }
  const auto hk = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, keys);
  const auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, payload);
  RecordBatch<Host::memory_space> sent(n, key_bytes + width, "transport::rendezvous");
  for (std::size_t i = 0; i < n; ++i) {
    sent.dest(i) = dest_of(hk(i));
    std::memcpy(&sent.payload(i, 0), &hk(i), key_bytes);
    if (width > 0) std::memcpy(&sent.payload(i, key_bytes), &hp(i, 0), width);
  }
  const RecordBatch<Host::memory_space> routed = route(Host{}, comm, sent, strategy);
  const CommPlan plan = learn_plan(comm, sent, routed);

  const std::size_t m = routed.size();
  Kokkos::View<std::uint64_t*, MemSpace> arrived_keys("transport::rendezvous.keys", m);
  RecordBatch<MemSpace> arrived(m, width, "transport::rendezvous.arrived");
  {
    const auto k = Kokkos::create_mirror_view(arrived_keys);
    const auto d = Kokkos::create_mirror_view(arrived.dest);
    const auto s = Kokkos::create_mirror_view(arrived.source);
    const auto p = Kokkos::create_mirror_view(arrived.payload);
    for (std::size_t i = 0; i < m; ++i) {
      std::memcpy(&k(i), &routed.payload(i, 0), key_bytes);
      d(i) = routed.dest(i);
      s(i) = routed.source(i);
      if (width > 0) std::memcpy(&p(i, 0), &routed.payload(i, key_bytes), width);
    }
    Kokkos::deep_copy(arrived_keys, k);
    Kokkos::deep_copy(arrived.dest, d);
    Kokkos::deep_copy(arrived.source, s);
    Kokkos::deep_copy(arrived.payload, p);
  }

  Kokkos::View<std::byte**, Kokkos::LayoutRight, MemSpace> rows;
  std::exception_ptr failure;
  try {
    rows = answer(arrived_keys, arrived);
  } catch (...) {
    failure = std::current_exception();
  }
  const bool bad_rows = !failure && (rows.extent(0) != m || rows.extent(1) != answer_width);
  const std::int32_t status = comm.allreduce(static_cast<std::int32_t>(failure ? 2 : bad_rows ? 1 : 0), Op::max);
  if (failure) std::rethrow_exception(failure);
  if (status == 2) throw std::runtime_error("transport::rendezvous: answering failed on another rank");
  if (status == 1) {
    throw std::invalid_argument("transport::rendezvous: a rank's answers are not a row of answer_width bytes per arrival");
  }

  const auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rows);
  Kokkos::View<std::byte**, Kokkos::LayoutRight, MemSpace> out("transport::rendezvous.answers", n, answer_width);
  const auto ho = Kokkos::create_mirror_view(out);
  plan.reverse_bytes(comm, std::span<const std::byte>(ha.data(), m * answer_width),
                     std::span<std::byte>(ho.data(), n * answer_width), answer_width);
  Kokkos::deep_copy(out, ho);
  return out;
}

// Collective. Throws as rendezvous_to.
template <class ExecSpace, class Answer>
Kokkos::View<std::byte**, Kokkos::LayoutRight, typename ExecSpace::memory_space> rendezvous(
    const ExecSpace& exec, const Comm& comm,
    const Kokkos::View<const std::uint64_t*, typename ExecSpace::memory_space>& keys,
    const Kokkos::View<const std::byte**, Kokkos::LayoutRight, typename ExecSpace::memory_space>& payload,
    std::size_t answer_width, const Answer& answer, Strategy strategy = Strategy::automatic) {
  const int size = comm.size();
  return rendezvous_to(exec, comm, keys, payload, answer_width, answer,
                       [size](std::uint64_t key) { return owner(key, size); }, strategy);
}

}  // namespace soma::transport

#endif  // SOMA_PARALLEL_TRANSPORT_RENDEZVOUS_HPP_
