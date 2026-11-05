// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <atomic>

#include "soma/parallel/comm.hpp"

namespace soma::detail {

namespace {
std::atomic<std::uint64_t> collectives{0};
std::atomic<std::uint64_t> routes{0};
}  // namespace

CommCounts comm_counts() noexcept {
  return {collectives.load(std::memory_order_relaxed), routes.load(std::memory_order_relaxed)};
}

void count_collective() noexcept { collectives.fetch_add(1, std::memory_order_relaxed); }

void count_route() noexcept { routes.fetch_add(1, std::memory_order_relaxed); }

}  // namespace soma::detail
