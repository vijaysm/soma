// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/comm.hpp"

#include <cstring>

#include "comm_checks.hpp"

#if SOMA_ENABLE_MPI
#error "comm_serial.cpp must not be compiled into an MPI build"
#endif

namespace soma {

Comm Comm::world() { return Comm(0, 1); }
Comm Comm::self() { return Comm(0, 1); }

void Comm::barrier() const {}

template <class T>
T Comm::allreduce(T value, Op) const {
  return value;
}

template <class T>
void Comm::allreduce(std::span<T>, Op) const {}

template <class T>
T Comm::exscan(T) const {
  return T{};
}

void Comm::bcast_bytes(std::span<std::byte>, int) const {}

void Comm::alltoall_bytes(std::span<const std::byte> send, std::span<std::byte> recv, std::size_t width) const {
  detail::check_alltoall(send.size(), recv.size(), width, size_);
  if (!send.empty()) std::memmove(recv.data(), send.data(), send.size());
}

void Comm::alltoallv_bytes(std::span<const std::byte> send, std::span<const std::size_t> send_counts,
                           std::span<std::byte> recv, std::span<const std::size_t> recv_counts,
                           std::size_t width) const {
  detail::check_counts("send", send_counts, send.size(), width, size_);
  detail::check_counts("receive", recv_counts, recv.size(), width, size_);
  if (send_counts[0] != recv_counts[0]) {
    throw std::invalid_argument("soma::Comm::alltoallv: rank 0 sends itself a different count than it receives");
  }
  if (!send.empty()) std::memmove(recv.data(), send.data(), send.size());
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
