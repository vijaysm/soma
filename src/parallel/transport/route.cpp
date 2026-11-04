// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/transport/route.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace soma::transport {

namespace {

constexpr int crystal_tag = 32000;  // + 2 * stage, + 1 for the rows
constexpr int nbx_tag = 32700;      // + the exchange's parity

// In transit a record is a row: its destination, source and place among the
// source's records, then the payload.
struct Header {
  std::int32_t dest;
  std::int32_t source;
  std::int64_t seq;
};
static_assert(sizeof(Header) == 16);

class Rows {
 public:
  explicit Rows(std::size_t width) : stride_(sizeof(Header) + width) {}

  [[nodiscard]] std::size_t size() const { return bytes_.size() / stride_; }
  [[nodiscard]] std::size_t stride() const { return stride_; }
  [[nodiscard]] Header header(std::size_t i) const {
    Header h;
    std::memcpy(&h, bytes_.data() + i * stride_, sizeof h);
    return h;
  }
  [[nodiscard]] std::span<const std::byte> row(std::size_t i) const { return {bytes_.data() + i * stride_, stride_}; }
  [[nodiscard]] std::vector<std::byte>& bytes() { return bytes_; }
  [[nodiscard]] const std::vector<std::byte>& bytes() const { return bytes_; }

  void push(const Header& h, std::span<const std::byte> payload) {
    const std::size_t at = bytes_.size();
    bytes_.resize(at + stride_);
    std::memcpy(bytes_.data() + at, &h, sizeof h);
    if (!payload.empty()) std::memcpy(bytes_.data() + at + sizeof h, payload.data(), payload.size());
  }
  void push(std::span<const std::byte> row) { bytes_.insert(bytes_.end(), row.begin(), row.end()); }
  void append(std::span<const std::byte> rows) { bytes_.insert(bytes_.end(), rows.begin(), rows.end()); }

 private:
  std::size_t stride_;
  std::vector<std::byte> bytes_;
};

// The rows split stably by destination: per destination rank, in order.
std::vector<Rows> by_dest(const Rows& rows, int size) {
  std::vector<Rows> out(static_cast<std::size_t>(size), Rows(rows.stride() - sizeof(Header)));
  for (std::size_t i = 0; i < rows.size(); ++i) out[static_cast<std::size_t>(rows.header(i).dest)].push(rows.row(i));
  return out;
}

void exchange_crystal(const Comm& comm, Rows& rows, int stage, int send_to, std::span<const int> recv_from,
                      const Rows& outgoing) {
  const int tag = crystal_tag + 2 * stage;
  const std::uint64_t count = outgoing.bytes().size();
  std::array<std::uint64_t, 2> counts{};
  std::vector<Request> r;
  for (std::size_t k = 0; k < recv_from.size(); ++k) r.push_back(comm.irecv(std::span<std::uint64_t>(&counts[k], 1), recv_from[k], tag));
  r.push_back(comm.isend(std::span<const std::uint64_t>(&count, 1), send_to, tag));
  Request::wait_all(r);
  r.clear();
  std::array<std::vector<std::byte>, 2> in;
  for (std::size_t k = 0; k < recv_from.size(); ++k) {
    in[k].resize(counts[k]);
    r.push_back(comm.irecv_bytes(in[k], recv_from[k], tag + 1));
  }
  r.push_back(comm.isend_bytes(outgoing.bytes(), send_to, tag + 1));
  Request::wait_all(r);
  for (std::size_t k = 0; k < recv_from.size(); ++k) rows.append(in[k]);
}

void crystal(const Comm& comm, Rows& rows) {
  const int id = comm.rank();
  int bl = 0, n = comm.size();
  for (int stage = 0; n > 1; ++stage) {
    const int nl = (n + 1) / 2, bh = bl + nl;
    const bool low = id < bh;
    Rows keep(rows.stride() - sizeof(Header)), send(rows.stride() - sizeof(Header));
    for (std::size_t i = 0; i < rows.size(); ++i) ((rows.header(i).dest < bh) == low ? keep : send).push(rows.row(i));
    std::array<int, 2> from{};
    std::size_t nfrom = 0;
    int to = 0;
    if (low) {
      // With an odd count the last lower rank has no partner above it.
      to = id + nl < bl + n ? id + nl : bh;
      if (id + nl < bl + n) from[nfrom++] = id + nl;
    } else {
      to = id - nl;
      from[nfrom++] = id - nl;
      if (n % 2 == 1 && id == bh) from[nfrom++] = bh - 1;
    }
    rows = std::move(keep);
    exchange_crystal(comm, rows, stage, to, std::span<const int>(from.data(), nfrom), send);
    if (low) {
      n = nl;
    } else {
      bl = bh;
      n -= nl;
    }
  }
}

void alltoallv(const Comm& comm, Rows& rows) {
  const auto p = static_cast<std::size_t>(comm.size());
  const std::vector<Rows> parts = by_dest(rows, comm.size());
  std::vector<std::uint64_t> send_counts(p), recv_counts(p);
  Rows out(rows.stride() - sizeof(Header));
  for (std::size_t r = 0; r < p; ++r) {
    send_counts[r] = parts[r].size();
    out.append(parts[r].bytes());
  }
  comm.alltoall(std::span<const std::uint64_t>(send_counts), std::span<std::uint64_t>(recv_counts));
  std::vector<std::size_t> sc(send_counts.begin(), send_counts.end()), rc(recv_counts.begin(), recv_counts.end());
  rows.bytes().assign(std::accumulate(rc.begin(), rc.end(), std::size_t{0}) * rows.stride(), std::byte{});
  comm.alltoallv_bytes(out.bytes(), sc, rows.bytes(), rc, rows.stride());
}

void nbx(const Comm& comm, Rows& rows) {
  const int tag = nbx_tag + static_cast<int>(comm.next_sequence() % 2);
  const std::vector<Rows> parts = by_dest(rows, comm.size());
  rows = parts[static_cast<std::size_t>(comm.rank())];
  std::vector<Request> sends;
  for (int r = 0; r < comm.size(); ++r) {
    if (r != comm.rank() && parts[static_cast<std::size_t>(r)].size() > 0) {
      sends.push_back(comm.issend_bytes(parts[static_cast<std::size_t>(r)].bytes(), r, tag));
    }
  }
  Request barrier;
  bool barrier_started = false;
  std::vector<std::byte> in;
  while (true) {
    if (const auto e = comm.iprobe(Comm::any_source, tag)) {
      in.resize(e->bytes);
      comm.irecv_bytes(in, e->source, tag).wait();
      rows.append(in);
    }
    if (!barrier_started) {
      if (std::all_of(sends.begin(), sends.end(), [](Request& s) { return s.test(); })) {
        barrier = comm.ibarrier();
        barrier_started = true;
      }
    } else if (barrier.test()) {
      break;
    }
  }
}

}  // namespace

Strategy detail::resolve(Strategy s, int size) {
  if (s != Strategy::automatic) return s;
  return size <= 64 ? Strategy::alltoallv : Strategy::crystal;
}

detail::Arrivals detail::route_host(const Comm& comm, std::span<const int> dest, std::span<const std::byte> payload,
                                    std::size_t width, Strategy strategy) {
  const bool bad_dest = std::any_of(dest.begin(), dest.end(), [&](int d) { return d < 0 || d >= comm.size(); });
  const bool bad_rows = payload.size() != dest.size() * width;
  std::array<std::uint64_t, 3> agree{static_cast<std::uint64_t>(width), static_cast<std::uint64_t>(strategy), 0};
  comm.bcast(std::span<std::uint64_t>(agree.data(), 2), 0);
  std::int32_t bad = bad_dest || bad_rows || agree[0] != width || agree[1] != static_cast<std::uint64_t>(strategy);
  bad = comm.allreduce(bad, Op::max);
  if (bad) {
    throw std::invalid_argument(
        "transport::route: a rank names a destination outside the communicator, passes a payload that is not a row "
        "per record, or differs from rank 0 in width or strategy");
  }
  Rows rows(width);
  for (std::size_t i = 0; i < dest.size(); ++i) {
    rows.push(Header{dest[i], comm.rank(), static_cast<std::int64_t>(i)}, payload.subspan(i * width, width));
  }
  switch (resolve(strategy, comm.size())) {
    case Strategy::crystal: crystal(comm, rows); break;
    case Strategy::alltoallv: alltoallv(comm, rows); break;
    case Strategy::nbx: nbx(comm, rows); break;
    case Strategy::automatic: break;
  }
  // Every strategy delivers the same rows; the order is fixed here.
  std::vector<std::size_t> order(rows.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::vector<Header> h(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) h[i] = rows.header(i);
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return h[a].source != h[b].source ? h[a].source < h[b].source : h[a].seq < h[b].seq;
  });
  Arrivals out;
  out.source.reserve(rows.size());
  out.payload.resize(rows.size() * width);
  for (std::size_t k = 0; k < order.size(); ++k) {
    out.source.push_back(h[order[k]].source);
    if (width > 0) std::memcpy(out.payload.data() + k * width, rows.row(order[k]).data() + sizeof(Header), width);
  }
  return out;
}

}  // namespace soma::transport
