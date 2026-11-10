// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/transport/comm_plan.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace soma::transport {

namespace {

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
  // Records stably by destination: already grouped, as they often are, they
  // are sent in place; else counted by rank, unless ranks outnumber them.
  send_perm_.resize(sent_dest.size());
  std::iota(send_perm_.begin(), send_perm_.end(), std::size_t{0});
  grouped_ = std::is_sorted(sent_dest.begin(), sent_dest.end());
  if (!grouped_ && static_cast<std::size_t>(comm.size()) > sent_dest.size()) {
    std::stable_sort(send_perm_.begin(), send_perm_.end(),
                     [&](std::size_t a, std::size_t b) { return sent_dest[a] < sent_dest[b]; });
  } else if (!grouped_) {
    std::vector<std::size_t> start(static_cast<std::size_t>(comm.size()) + 1, 0);
    for (const int d : sent_dest) ++start[static_cast<std::size_t>(d) + 1];
    std::partial_sum(start.begin(), start.end(), start.begin());
    for (std::size_t i = 0; i < sent_dest.size(); ++i) send_perm_[start[static_cast<std::size_t>(sent_dest[i])]++] = i;
  }
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

CommPlan CommPlan::from_counts(const Comm& comm, std::span<const Run> sent, std::span<const Run> arrived) {
  const auto check = [&](std::span<const Run> runs) {
    for (std::size_t i = 0; i < runs.size(); ++i) {
      if (runs[i].rank < 0 || runs[i].rank >= comm.size() || (i > 0 && runs[i].rank <= runs[i - 1].rank)) {
        throw std::invalid_argument("transport::CommPlan::from_counts: the runs are not of ranks in strictly ascending order");
      }
    }
  };
  check(sent);
  check(arrived);
  CommPlan plan;
  plan.rank_ = comm.rank();
  const auto fill = [&](std::span<const Run> runs, std::vector<int>& peers, std::vector<std::size_t>& offsets,
                        std::size_t& self_start, std::size_t& self_count) {
    std::size_t at = 0;
    for (const Run& r : runs) {
      if (r.rank == plan.rank_) {
        self_start = at;
        self_count = r.count;
      }
      if (r.count == 0) continue;
      peers.push_back(r.rank);
      offsets.push_back(at);
      at += r.count;
    }
    offsets.push_back(at);
    return at;
  };
  std::size_t sent_self = 0, arrived_self = 0;
  const std::size_t n = fill(sent, plan.send_peers_, plan.send_offsets_, plan.self_packed_, sent_self);
  fill(arrived, plan.recv_peers_, plan.recv_offsets_, plan.self_arrived_, arrived_self);
  if (sent_self != arrived_self) {
    throw std::invalid_argument("transport::CommPlan::from_counts: the records to this rank differ in count sent and arrived");
  }
  plan.self_count_ = sent_self;
  plan.send_perm_.resize(n);
  std::iota(plan.send_perm_.begin(), plan.send_perm_.end(), std::size_t{0});
  plan.grouped_ = true;
  return plan;
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
  begin_forward_bytes(comm, sent, arrived, width, forward_tag).wait();
}

CommPlan::Pending CommPlan::begin_forward_bytes(const Comm& comm, std::span<const std::byte> sent,
                                                std::span<std::byte> arrived, std::size_t width, int tag) const {
  if (sent.size() != num_sent() * width || arrived.size() != num_arrived() * width) {
    throw std::invalid_argument("transport::CommPlan::forward: the spans do not hold a value per record");
  }
  Pending p;
  std::span<const std::byte> packed = sent;
  if (!grouped_) {
    p.packed_.resize(sent.size());
    for (std::size_t k = 0; k < send_perm_.size(); ++k) copy(sent, send_perm_[k], p.packed_, k, width);
    packed = p.packed_;
  }
  p.requests_.reserve(recv_peers_.size() + send_peers_.size());
  for (std::size_t k = 0; k < recv_peers_.size(); ++k) {
    if (recv_peers_[k] == rank_) continue;
    p.requests_.push_back(comm.irecv_bytes(
        arrived.subspan(recv_offsets_[k] * width, (recv_offsets_[k + 1] - recv_offsets_[k]) * width), recv_peers_[k], tag));
  }
  for (std::size_t k = 0; k < send_peers_.size(); ++k) {
    if (send_peers_[k] == rank_) continue;
    p.requests_.push_back(comm.isend_bytes(
        packed.subspan(send_offsets_[k] * width, (send_offsets_[k + 1] - send_offsets_[k]) * width), send_peers_[k], tag));
  }
  if (self_count_ > 0) std::memcpy(arrived.data() + self_arrived_ * width, packed.data() + self_packed_ * width, self_count_ * width);
  return p;
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
