// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/transport/comm_plan.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace soma::transport {

namespace {

constexpr int forward_tag = 32600;
constexpr int reverse_tag = 32601;

// The distinct values of a sorted list and where each one's run starts, with
// the end appended.
void runs(std::span<const int> sorted, std::vector<int>& values, std::vector<std::size_t>& offsets) {
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (i == 0 || sorted[i] != sorted[i - 1]) {
      values.push_back(sorted[i]);
      offsets.push_back(i);
    }
  }
  offsets.push_back(sorted.size());
}

void copy(std::span<const std::byte> from, std::size_t i, std::span<std::byte> to, std::size_t j, std::size_t width) {
  if (width > 0) std::memcpy(to.data() + j * width, from.data() + i * width, width);
}

}  // namespace

CommPlan::CommPlan(const Comm& comm, std::span<const int> sent_dest, std::span<const int> arrived_source)
    : rank_(comm.rank()) {
  for (const int d : sent_dest) {
    if (d < 0 || d >= comm.size()) throw std::invalid_argument("transport::CommPlan: a destination is not a rank");
  }
  for (std::size_t i = 0; i < arrived_source.size(); ++i) {
    const int s = arrived_source[i];
    if (s < 0 || s >= comm.size() || (i > 0 && s < arrived_source[i - 1])) {
      throw std::invalid_argument("transport::CommPlan: the sources are not ranks grouped in ascending order");
    }
  }
  send_perm_.resize(sent_dest.size());
  std::iota(send_perm_.begin(), send_perm_.end(), std::size_t{0});
  std::stable_sort(send_perm_.begin(), send_perm_.end(),
                   [&](std::size_t a, std::size_t b) { return sent_dest[a] < sent_dest[b]; });
  std::vector<int> packed(sent_dest.size());
  for (std::size_t k = 0; k < packed.size(); ++k) packed[k] = sent_dest[send_perm_[k]];
  runs(packed, send_peers_, send_offsets_);
  runs(arrived_source, recv_peers_, recv_offsets_);
  const auto first = [](std::span<const int> v, int r) { return static_cast<std::size_t>(std::lower_bound(v.begin(), v.end(), r) - v.begin()); };
  self_packed_ = first(packed, rank_);
  self_arrived_ = first(arrived_source, rank_);
  self_count_ = static_cast<std::size_t>(std::count(packed.begin(), packed.end(), rank_));
  if (static_cast<std::size_t>(std::count(arrived_source.begin(), arrived_source.end(), rank_)) != self_count_) {
    throw std::invalid_argument("transport::CommPlan: the records to this rank differ in count sent and arrived");
  }
}

std::vector<int> CommPlan::send_ranks() const {
  std::vector<int> out;
  std::copy_if(send_peers_.begin(), send_peers_.end(), std::back_inserter(out), [&](int r) { return r != rank_; });
  return out;
}

std::vector<int> CommPlan::recv_ranks() const {
  std::vector<int> out;
  std::copy_if(recv_peers_.begin(), recv_peers_.end(), std::back_inserter(out), [&](int r) { return r != rank_; });
  return out;
}

void CommPlan::forward_bytes(const Comm& comm, std::span<const std::byte> sent, std::span<std::byte> arrived,
                             std::size_t width) const {
  if (sent.size() != num_sent() * width || arrived.size() != num_arrived() * width) {
    throw std::invalid_argument("transport::CommPlan::forward: the spans do not hold a value per record");
  }
  std::vector<std::byte> packed(sent.size());
  for (std::size_t k = 0; k < send_perm_.size(); ++k) copy(sent, send_perm_[k], packed, k, width);
  std::vector<Request> r;
  for (std::size_t k = 0; k < recv_peers_.size(); ++k) {
    if (recv_peers_[k] == rank_) continue;
    r.push_back(comm.irecv_bytes(arrived.subspan(recv_offsets_[k] * width, (recv_offsets_[k + 1] - recv_offsets_[k]) * width),
                                 recv_peers_[k], forward_tag));
  }
  for (std::size_t k = 0; k < send_peers_.size(); ++k) {
    if (send_peers_[k] == rank_) continue;
    r.push_back(comm.isend_bytes(std::span<const std::byte>(packed).subspan(send_offsets_[k] * width,
                                                                            (send_offsets_[k + 1] - send_offsets_[k]) * width),
                                 send_peers_[k], forward_tag));
  }
  for (std::size_t k = 0; k < self_count_; ++k) copy(packed, self_packed_ + k, arrived, self_arrived_ + k, width);
  Request::wait_all(r);
}

void CommPlan::reverse_bytes(const Comm& comm, std::span<const std::byte> arrived, std::span<std::byte> sent,
                             std::size_t width) const {
  if (sent.size() != num_sent() * width || arrived.size() != num_arrived() * width) {
    throw std::invalid_argument("transport::CommPlan::reverse: the spans do not hold a value per record");
  }
  std::vector<std::byte> packed(sent.size());
  std::vector<Request> r;
  for (std::size_t k = 0; k < send_peers_.size(); ++k) {
    if (send_peers_[k] == rank_) continue;
    r.push_back(comm.irecv_bytes(std::span<std::byte>(packed).subspan(send_offsets_[k] * width,
                                                                      (send_offsets_[k + 1] - send_offsets_[k]) * width),
                                 send_peers_[k], reverse_tag));
  }
  for (std::size_t k = 0; k < self_count_; ++k) copy(arrived, self_arrived_ + k, packed, self_packed_ + k, width);
  for (std::size_t k = 0; k < recv_peers_.size(); ++k) {
    if (recv_peers_[k] == rank_) continue;
    r.push_back(comm.isend_bytes(arrived.subspan(recv_offsets_[k] * width, (recv_offsets_[k + 1] - recv_offsets_[k]) * width),
                                 recv_peers_[k], reverse_tag));
  }
  Request::wait_all(r);
  for (std::size_t k = 0; k < send_perm_.size(); ++k) copy(packed, k, sent, send_perm_[k], width);
}

}  // namespace soma::transport
