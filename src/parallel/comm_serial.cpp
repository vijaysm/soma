// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/comm.hpp"

#include <algorithm>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <vector>

#include "comm_checks.hpp"

#if SOMA_ENABLE_MPI
#error "comm_serial.cpp must not be compiled into an MPI build"
#endif

namespace soma {

// A receive posted before its message, or an issend not yet received.
struct detail::RequestState {
  bool done = false;
  int tag = 0;
  std::span<std::byte> buffer;
};

namespace {

// Rank 0's messages to itself: sent and not received, and receives posted
// and not matched, each in order.
struct Message {
  int tag;
  std::vector<std::byte> bytes;
  std::shared_ptr<detail::RequestState> sender;  // an issend's, else null
};
std::deque<Message> messages;
std::deque<std::shared_ptr<detail::RequestState>> receives;
std::uint64_t sequence = 0;

void deliver(std::span<const std::byte> from, std::span<std::byte> to) {
  if (from.size() > to.size()) {
    throw std::runtime_error("soma::Comm: a message is longer than the buffer receiving it");
  }
  if (!from.empty()) std::memcpy(to.data(), from.data(), from.size());
}

// Delivers to the first receive posted for `tag`, else queues the message;
// returns the state of a queued issend.
std::shared_ptr<detail::RequestState> send(std::span<const std::byte> buffer, int tag, bool synchronous) {
  const auto it = std::find_if(receives.begin(), receives.end(), [&](const auto& x) { return x->tag == tag; });
  if (it != receives.end()) {
    deliver(buffer, (*it)->buffer);
    (*it)->done = true;
    receives.erase(it);
    return nullptr;
  }
  std::shared_ptr<detail::RequestState> state;
  if (synchronous) state = std::make_shared<detail::RequestState>(detail::RequestState{false, tag, {}});
  messages.push_back(Message{tag, std::vector<std::byte>(buffer.begin(), buffer.end()), state});
  return state;
}

// Forgets a request dropped before it completed.
void abandon(const std::shared_ptr<detail::RequestState>& state) {
  if (!state || state->done) return;
  std::erase(receives, state);
  for (Message& m : messages) {
    if (m.sender == state) m.sender.reset();
  }
}

}  // namespace

Request::Request() noexcept = default;
Request::Request(Request&& other) noexcept = default;

Request& Request::operator=(Request&& other) noexcept {
  if (this != &other) {
    abandon(state_);
    state_ = std::move(other.state_);
  }
  return *this;
}

Request::~Request() { abandon(state_); }

bool Request::test() { return !state_ || state_->done; }

void Request::wait() {
  if (!test()) throw std::logic_error("soma::Request: waits for a message nothing sends or receives");
  state_.reset();
}

void Request::wait_all(std::span<Request> requests) {
  for (Request& r : requests) r.wait();
}

Comm Comm::world() { return Comm(0, 1); }
Comm Comm::self() { return Comm(0, 1); }

void Comm::barrier() const { detail::count_collective(); }

template <class T>
T Comm::allreduce(T value, Op) const {
  detail::count_collective();
  return value;
}

template <class T>
void Comm::allreduce(std::span<T>, Op) const { detail::count_collective(); }

template <class T>
T Comm::exscan(T) const {
  detail::count_collective();
  return T{};
}

template <class T>
void Comm::exscan(std::span<T> values) const {
  detail::count_collective();
  std::fill(values.begin(), values.end(), T{});
}

void Comm::bcast_bytes(std::span<std::byte>, int) const { detail::count_collective(); }

void Comm::alltoall_bytes(std::span<const std::byte> send, std::span<std::byte> recv, std::size_t width) const {
  detail::count_collective();
  detail::check_alltoall(send.size(), recv.size(), width, size_);
  if (!send.empty()) std::memmove(recv.data(), send.data(), send.size());
}

void Comm::alltoallv_bytes(std::span<const std::byte> send, std::span<const std::size_t> send_counts,
                           std::span<std::byte> recv, std::span<const std::size_t> recv_counts,
                           std::size_t width) const {
  detail::count_collective();
  detail::check_counts("send", send_counts, send.size(), width, size_);
  detail::check_counts("receive", recv_counts, recv.size(), width, size_);
  if (send_counts[0] != recv_counts[0]) {
    throw std::invalid_argument("soma::Comm::alltoallv: rank 0 sends itself a different count than it receives");
  }
  if (!send.empty()) std::memmove(recv.data(), send.data(), send.size());
}

Request Comm::isend_bytes(std::span<const std::byte> buffer, int dest, int tag) const {
  detail::check_peer("isend", dest, tag, size_, false);
  send(buffer, tag, false);
  return Request{};
}

Request Comm::issend_bytes(std::span<const std::byte> buffer, int dest, int tag) const {
  detail::check_peer("issend", dest, tag, size_, false);
  Request r;
  r.state_ = send(buffer, tag, true);
  return r;
}

Request Comm::irecv_bytes(std::span<std::byte> buffer, int source, int tag) const {
  detail::check_peer("irecv", source, tag, size_, true);
  Request r;
  const auto it = std::find_if(messages.begin(), messages.end(), [&](const Message& m) { return m.tag == tag; });
  if (it != messages.end()) {
    deliver(it->bytes, buffer);
    if (it->sender) it->sender->done = true;
    messages.erase(it);
    return r;
  }
  r.state_ = std::make_shared<detail::RequestState>(detail::RequestState{false, tag, buffer});
  receives.push_back(r.state_);
  return r;
}

std::optional<Envelope> Comm::iprobe(int source, int tag) const {
  detail::check_peer("iprobe", source, tag, size_, true);
  const auto it = std::find_if(messages.begin(), messages.end(), [&](const Message& m) { return m.tag == tag; });
  if (it == messages.end()) return std::nullopt;
  return Envelope{0, tag, it->bytes.size()};
}

Request Comm::ibarrier() const {
  detail::count_collective();
  return Request{};
}

std::uint64_t Comm::next_sequence() const { return sequence++; }

#define SOMA_INSTANTIATE_COMM(T)                              \
  template T Comm::allreduce<T>(T, Op) const;                 \
  template void Comm::allreduce<T>(std::span<T>, Op) const;   \
  template T Comm::exscan<T>(T) const;                        \
  template void Comm::exscan<T>(std::span<T>) const;

SOMA_INSTANTIATE_COMM(std::int32_t)
SOMA_INSTANTIATE_COMM(std::int64_t)
SOMA_INSTANTIATE_COMM(std::uint64_t)
SOMA_INSTANTIATE_COMM(double)

#undef SOMA_INSTANTIATE_COMM

}  // namespace soma
