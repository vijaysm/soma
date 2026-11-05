// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_PARALLEL_RECORDS_HPP_
#define SOMA_SRC_PARALLEL_RECORDS_HPP_

// Host records as byte rows: fields written and read at byte offsets, and
// rows routed to ranks.

#include <cstddef>
#include <cstring>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/parallel/comm.hpp"
#include "soma/parallel/transport/route.hpp"

namespace soma::detail {

using Batch = transport::RecordBatch<Kokkos::HostSpace>;
using HostExec = Kokkos::DefaultHostExecutionSpace;

template <class T>
void put(std::byte* row, std::size_t at, const T& v) {
  std::memcpy(row + at, &v, sizeof(T));
}

template <class T>
T take(const std::byte* row, std::size_t at) {
  T v;
  std::memcpy(&v, row + at, sizeof(T));
  return v;
}

// Rows of payload `width` to the ranks `dest`, filled by fill(i, row).
template <class Fill>
Batch route(const Comm& comm, const std::vector<int>& dest, std::size_t width, Fill&& fill) {
  Batch out(dest.size(), width, "soma::records");
  for (std::size_t i = 0; i < dest.size(); ++i) {
    out.dest(i) = dest[i];
    fill(i, &out.payload(i, 0));
  }
  return transport::route(HostExec{}, comm, out);
}

}  // namespace soma::detail

#endif  // SOMA_SRC_PARALLEL_RECORDS_HPP_
