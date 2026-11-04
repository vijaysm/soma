// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_PARALLEL_COMM_CHECKS_HPP_
#define SOMA_SRC_PARALLEL_COMM_CHECKS_HPP_

// Argument checks shared by both Comm builds, so they reject the same calls.

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

namespace soma::detail {

inline void check_alltoall(std::size_t send_bytes, std::size_t recv_bytes, std::size_t width, int size) {
  const std::size_t block = width * static_cast<std::size_t>(size);
  if (width == 0 || send_bytes != recv_bytes || send_bytes % block != 0) {
    throw std::invalid_argument("soma::Comm::alltoall: the buffers are not the same whole number of blocks per rank");
  }
}

inline void check_counts(const char* side, std::span<const std::size_t> counts, std::size_t bytes, std::size_t width,
                         int size) {
  std::size_t total = 0;
  for (const std::size_t c : counts) total += c;
  if (width == 0 || counts.size() != static_cast<std::size_t>(size) || total * width != bytes) {
    throw std::invalid_argument(std::string("soma::Comm::alltoallv: the ") + side +
                                " counts are not one per rank adding up to the buffer");
  }
}

constexpr int max_tag = 32767;

inline void check_peer(const char* what, int peer, int tag, int size, bool any_allowed) {
  if (!(peer >= 0 && peer < size) && !(any_allowed && peer == -1)) {
    throw std::invalid_argument(std::string("soma::Comm::") + what + ": rank " + std::to_string(peer) + " is not in [0, " +
                                std::to_string(size) + ")");
  }
  if (tag < 0 || tag > max_tag) {
    throw std::invalid_argument(std::string("soma::Comm::") + what + ": tag " + std::to_string(tag) + " is not in [0, " +
                                std::to_string(max_tag) + "]");
  }
}

}  // namespace soma::detail

#endif  // SOMA_SRC_PARALLEL_COMM_CHECKS_HPP_
