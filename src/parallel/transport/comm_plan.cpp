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
    : rank_(comm.rank()), num_sent_(sent_dest.size()) {
  for (std::size_t i = 0; i < sent_dest.size(); ++i) {
    const int d = sent_dest[i];
    if (d < 0 || d >= comm.size()) throw std::invalid_argument("transport::CommPlan: a destination is not a rank");
    if (i > 0 && d < sent_dest[i - 1]) grouped_ = false;
  }
  for (std::size_t i = 0; i < arrived_source.size(); ++i) {
    const int s = arrived_source[i];
    if (s < 0 || s >= comm.size() || (i > 0 && s < arrived_source[i - 1])) {
      throw std::invalid_argument("transport::CommPlan: the sources are not ranks grouped in ascending order");
    }
  }
  // Records stably by destination: already grouped, as they often are, they
  // are sent in place; else counted by rank, unless ranks outnumber them.
  if (grouped_) {
    runs(sent_dest, send_peers_, send_offsets_);
  } else {
    send_perm_.resize(sent_dest.size());
    if (static_cast<std::size_t>(comm.size()) > sent_dest.size()) {
      std::iota(send_perm_.begin(), send_perm_.end(), std::size_t{0});
      std::stable_sort(send_perm_.begin(), send_perm_.end(),
                       [&](std::size_t a, std::size_t b) { return sent_dest[a] < sent_dest[b]; });
    } else {
      std::vector<std::size_t> start(static_cast<std::size_t>(comm.size()) + 1, 0);
      for (const int d : sent_dest) ++start[static_cast<std::size_t>(d) + 1];
      std::partial_sum(start.begin(), start.end(), start.begin());
      for (std::size_t i = 0; i < sent_dest.size(); ++i) {
        send_perm_[start[static_cast<std::size_t>(sent_dest[i])]++] = i;
      }
    }
    std::vector<int> packed(sent_dest.size());
    for (std::size_t k = 0; k < packed.size(); ++k) packed[k] = sent_dest[send_perm_[k]];
    runs(packed, send_peers_, send_offsets_);
  }
  runs(arrived_source, recv_peers_, recv_offsets_);
  // This rank's run among the sent and the arrived: where it is, or would be.
  const auto own = [&](const std::vector<int>& peers, const std::vector<std::size_t>& offsets, std::size_t& start) {
    const auto k = static_cast<std::size_t>(std::lower_bound(peers.begin(), peers.end(), rank_) - peers.begin());
    start = offsets[k];
    return k < peers.size() && peers[k] == rank_ ? offsets[k + 1] - offsets[k] : std::size_t{0};
  };
  self_count_ = own(send_peers_, send_offsets_, self_packed_);
  if (own(recv_peers_, recv_offsets_, self_arrived_) != self_count_) {
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
  plan.num_sent_ = n;
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
    for (std::size_t k = 0; k < num_sent_; ++k) copy(sent, send_perm_[k], p.packed_, k, width);
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
  // Grouped records are answered in place; others through a packed copy.
  std::vector<std::byte> buffer(grouped_ ? 0 : sent.size());
  const std::span<std::byte> packed = grouped_ ? sent : std::span<std::byte>(buffer);
  std::vector<Request> r;
  for (std::size_t k = 0; k < send_peers_.size(); ++k) {
    if (send_peers_[k] == rank_) continue;
    const std::size_t count = send_offsets_[k + 1] - send_offsets_[k];
    r.push_back(comm.irecv_bytes(packed.subspan(send_offsets_[k] * width, count * width), send_peers_[k], reverse_tag));
  }
  if (self_count_ > 0) {
    std::memcpy(packed.data() + self_packed_ * width, arrived.data() + self_arrived_ * width, self_count_ * width);
  }
  for (std::size_t k = 0; k < recv_peers_.size(); ++k) {
    if (recv_peers_[k] == rank_) continue;
    r.push_back(comm.isend_bytes(arrived.subspan(recv_offsets_[k] * width, (recv_offsets_[k + 1] - recv_offsets_[k]) * width),
                                 recv_peers_[k], reverse_tag));
  }
  Request::wait_all(r);
  if (!grouped_) {
    for (std::size_t k = 0; k < num_sent_; ++k) copy(packed, k, sent, send_perm_[k], width);
  }
}

}  // namespace soma::transport
