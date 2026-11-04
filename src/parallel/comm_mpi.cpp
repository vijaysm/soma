// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/comm.hpp"

#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "comm_checks.hpp"

#if !SOMA_ENABLE_MPI
#error "comm_mpi.cpp requires SOMA_ENABLE_MPI"
#endif

namespace soma {

namespace {

void check(int rc, const char* what) {
  if (rc != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING] = {};
    int len = 0;
    MPI_Error_string(rc, msg, &len);
    throw std::runtime_error(std::string("soma::Comm: ") + what + " failed: " +
                             std::string(msg, static_cast<std::size_t>(len)));
  }
}

int checked_count(std::size_t n) {
  if (n > static_cast<std::size_t>(INT_MAX)) {
    throw std::length_error("soma::Comm: message exceeds INT_MAX elements");
  }
  return static_cast<int>(n);
}

MPI_Op to_mpi(Op op) {
  switch (op) {
    case Op::sum: return MPI_SUM;
    case Op::min: return MPI_MIN;
    case Op::max: return MPI_MAX;
    case Op::prod: return MPI_PROD;
  }
  return MPI_OP_NULL;
}

template <class T> MPI_Datatype datatype();
template <> MPI_Datatype datatype<std::int32_t>() { return MPI_INT32_T; }
template <> MPI_Datatype datatype<std::int64_t>() { return MPI_INT64_T; }
template <> MPI_Datatype datatype<std::uint64_t>() { return MPI_UINT64_T; }
template <> MPI_Datatype datatype<double>() { return MPI_DOUBLE; }

template <class T>
T reduce_one(MPI_Comm comm, T value, Op op) {
  T out{};
  check(MPI_Allreduce(&value, &out, 1, datatype<T>(), to_mpi(op), comm), "MPI_Allreduce");
  return out;
}

template <class T>
void reduce_span(MPI_Comm comm, std::span<T> values, Op op) {
  check(MPI_Allreduce(MPI_IN_PLACE, values.data(), checked_count(values.size()),
                      datatype<T>(), to_mpi(op), comm),
        "MPI_Allreduce");
}

int sequence_keyval = MPI_KEYVAL_INVALID;

int delete_sequence(MPI_Comm, int, void* value, void*) {
  delete static_cast<std::uint64_t*>(value);
  return MPI_SUCCESS;
}

int source_of(int source) { return source == Comm::any_source ? MPI_ANY_SOURCE : source; }

}  // namespace

Request::Request() noexcept : request_(MPI_REQUEST_NULL) {}

Request::Request(Request&& other) noexcept : request_(other.request_) { other.request_ = MPI_REQUEST_NULL; }

Request& Request::operator=(Request&& other) noexcept {
  if (this != &other) {
    if (request_ != MPI_REQUEST_NULL) MPI_Wait(&request_, MPI_STATUS_IGNORE);
    request_ = other.request_;
    other.request_ = MPI_REQUEST_NULL;
  }
  return *this;
}

Request::~Request() {
  if (request_ != MPI_REQUEST_NULL) MPI_Wait(&request_, MPI_STATUS_IGNORE);
}

bool Request::test() {
  int done = 0;
  check(MPI_Test(&request_, &done, MPI_STATUS_IGNORE), "MPI_Test");
  return done != 0;
}

void Request::wait() { check(MPI_Wait(&request_, MPI_STATUS_IGNORE), "MPI_Wait"); }

void Request::wait_all(std::span<Request> requests) {
  std::vector<MPI_Request> r;
  r.reserve(requests.size());
  for (Request& x : requests) r.push_back(x.request_);
  check(MPI_Waitall(checked_count(r.size()), r.data(), MPI_STATUSES_IGNORE), "MPI_Waitall");
  for (Request& x : requests) x.request_ = MPI_REQUEST_NULL;
}

Comm::Comm(MPI_Comm comm) : comm_(comm) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::runtime_error(
        "soma::Comm used before MPI_Init; construct a soma::ScopeGuard first");
  }
  check(MPI_Comm_rank(comm, &rank_), "MPI_Comm_rank");
  check(MPI_Comm_size(comm, &size_), "MPI_Comm_size");
}

Comm Comm::world() { return Comm(MPI_COMM_WORLD); }
Comm Comm::self() { return Comm(MPI_COMM_SELF); }

void Comm::barrier() const { check(MPI_Barrier(comm_), "MPI_Barrier"); }

template <class T>
T Comm::allreduce(T value, Op op) const {
  return reduce_one(comm_, value, op);
}

template <class T>
void Comm::allreduce(std::span<T> values, Op op) const {
  reduce_span(comm_, values, op);
}

template <class T>
T Comm::exscan(T value) const {
  T out = std::numeric_limits<T>::max();
  check(MPI_Exscan(&value, &out, 1, datatype<T>(), MPI_SUM, comm_), "MPI_Exscan");
  // MPI leaves rank 0's receive buffer undefined.
  return rank_ == 0 ? T{} : out;
}

void Comm::bcast_bytes(std::span<std::byte> buffer, int root) const {
  check(MPI_Bcast(buffer.data(), checked_count(buffer.size()), MPI_BYTE, root, comm_),
        "MPI_Bcast");
}

void Comm::alltoall_bytes(std::span<const std::byte> send, std::span<std::byte> recv, std::size_t width) const {
  detail::check_alltoall(send.size(), recv.size(), width, size_);
  const int block = checked_count(send.size() / static_cast<std::size_t>(size_));
  check(MPI_Alltoall(send.data(), block, MPI_BYTE, recv.data(), block, MPI_BYTE, comm_), "MPI_Alltoall");
}

void Comm::alltoallv_bytes(std::span<const std::byte> send, std::span<const std::size_t> send_counts,
                           std::span<std::byte> recv, std::span<const std::size_t> recv_counts,
                           std::size_t width) const {
  detail::check_counts("send", send_counts, send.size(), width, size_);
  detail::check_counts("receive", recv_counts, recv.size(), width, size_);
  const auto p = static_cast<std::size_t>(size_);
  std::vector<int> sc(p), sd(p), rc(p), rd(p);
  std::size_t so = 0, ro = 0;
  for (std::size_t r = 0; r < p; ++r) {
    sc[r] = checked_count(send_counts[r] * width);
    rc[r] = checked_count(recv_counts[r] * width);
    sd[r] = checked_count(so);
    rd[r] = checked_count(ro);
    so += send_counts[r] * width;
    ro += recv_counts[r] * width;
  }
  check(MPI_Alltoallv(send.data(), sc.data(), sd.data(), MPI_BYTE, recv.data(), rc.data(), rd.data(), MPI_BYTE, comm_),
        "MPI_Alltoallv");
}

Request Comm::isend_bytes(std::span<const std::byte> buffer, int dest, int tag) const {
  detail::check_peer("isend", dest, tag, size_, false);
  Request r;
  check(MPI_Isend(buffer.data(), checked_count(buffer.size()), MPI_BYTE, dest, tag, comm_, &r.request_), "MPI_Isend");
  return r;
}

Request Comm::issend_bytes(std::span<const std::byte> buffer, int dest, int tag) const {
  detail::check_peer("issend", dest, tag, size_, false);
  Request r;
  check(MPI_Issend(buffer.data(), checked_count(buffer.size()), MPI_BYTE, dest, tag, comm_, &r.request_), "MPI_Issend");
  return r;
}

Request Comm::irecv_bytes(std::span<std::byte> buffer, int source, int tag) const {
  detail::check_peer("irecv", source, tag, size_, true);
  Request r;
  check(MPI_Irecv(buffer.data(), checked_count(buffer.size()), MPI_BYTE, source_of(source), tag, comm_, &r.request_),
        "MPI_Irecv");
  return r;
}

std::optional<Envelope> Comm::iprobe(int source, int tag) const {
  detail::check_peer("iprobe", source, tag, size_, true);
  int found = 0;
  MPI_Status status;
  check(MPI_Iprobe(source_of(source), tag, comm_, &found, &status), "MPI_Iprobe");
  if (!found) return std::nullopt;
  int bytes = 0;
  check(MPI_Get_count(&status, MPI_BYTE, &bytes), "MPI_Get_count");
  return Envelope{status.MPI_SOURCE, status.MPI_TAG, static_cast<std::size_t>(bytes)};
}

Request Comm::ibarrier() const {
  Request r;
  check(MPI_Ibarrier(comm_, &r.request_), "MPI_Ibarrier");
  return r;
}

std::uint64_t Comm::next_sequence() const {
  if (sequence_keyval == MPI_KEYVAL_INVALID) {
    check(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, delete_sequence, &sequence_keyval, nullptr),
          "MPI_Comm_create_keyval");
  }
  void* value = nullptr;
  int found = 0;
  check(MPI_Comm_get_attr(comm_, sequence_keyval, &value, &found), "MPI_Comm_get_attr");
  if (!found) {
    value = new std::uint64_t{0};
    check(MPI_Comm_set_attr(comm_, sequence_keyval, value), "MPI_Comm_set_attr");
  }
  return (*static_cast<std::uint64_t*>(value))++;
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
