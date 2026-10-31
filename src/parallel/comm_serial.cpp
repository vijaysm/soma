// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/comm.hpp"

#if SOMA_ENABLE_MPI
#error "comm_serial.cpp must not be compiled into an MPI build"
#endif

namespace soma {

Comm Comm::world() { return Comm(0, 1); }
Comm Comm::self() { return Comm(0, 1); }

void Comm::barrier() const {}

std::int32_t Comm::allreduce(std::int32_t value, Op) const { return value; }
std::int64_t Comm::allreduce(std::int64_t value, Op) const { return value; }
double Comm::allreduce(double value, Op) const { return value; }
void Comm::allreduce(std::span<std::int64_t>, Op) const {}
void Comm::allreduce(std::span<double>, Op) const {}

std::int64_t Comm::exscan(std::int64_t) const { return 0; }

void Comm::bcast(std::span<std::byte>, int) const {}

}  // namespace soma
