// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Out-of-tree consumer: must build against installed headers and library only.
#include <cstdint>
#include <cstdio>

#include <Kokkos_Core.hpp>

#include "soma/core/topology.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/topology/canonical.hpp"

namespace soma_consumer {

int hex_face_count() {
  int sides = 0;
  Kokkos::parallel_reduce(
      "consumer", 1,
      KOKKOS_LAMBDA(const int, int& acc) {
        acc += soma::canonical::num_sides(soma::Topology::hex, 2);
      },
      sides);
  return sides;
}

}  // namespace soma_consumer

int main(int argc, char** argv) {
  soma::ScopeGuard guard(argc, argv);
  const int sides = soma_consumer::hex_face_count();
  const auto comm = soma::Comm::world();
  if (sides != 6 || comm.allreduce(std::int32_t{1}, soma::Op::sum) != comm.size()) {
    std::printf("FAIL sides=%d\n", sides);
    return 1;
  }
  std::printf("OK soma consumer: hex has %d faces, %d rank(s)\n", sides, comm.size());
  return 0;
}
