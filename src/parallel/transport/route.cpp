// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/transport/route.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <numeric>
#include <stdexcept>

namespace soma::transport {

namespace {

constexpr int crystal_tag = 32000;  // + 2 * stage, + 1 for the rows
constexpr int nbx_tag = 32700;      // + the exchange's parity

// In transit a record is a row: its destination and source, then the
// payload. Rows of one source and destination always travel together and in
// order (each strategy moves them by destination, stably), so ordering the
// arrivals by source alone restores each sender's order.
struct Header {
  std::int32_t dest;
  std::int32_t source;
};
static_assert(sizeof(Header) == detail::Arrivals::header);

struct Rows {
  std::size_t stride;
  std::vector<std::byte> bytes;
  std::size_t limit;  // the most rows any rank can hold (detail::check_incoming)

  [[nodiscard]] std::size_t size() const { return bytes.size() / stride; }
  [[nodiscard]] Header header(std::size_t i) const {
    Header h;
    std::memcpy(&h, bytes.data() + i * stride, sizeof h);
    return h;
  }
};

// Throws unless `rows` more rows fit beside `held` within `limit`.
void check_rows(std::size_t held, std::uint64_t rows, std::size_t limit) {
  if (rows > limit || held > limit - rows) {
    throw std::runtime_error("transport::route: a message of " + std::to_string(rows) +
                             " records is more than the ranks sent; the exchange is corrupt");
  }
}

// The rows reordered stably by key(header) in [0, buckets), with where each
// bucket starts (and the end) in `starts`.
template <class Key>
Rows bucket(const Rows& rows, std::size_t buckets, Key key, std::vector<std::size_t>& starts) {
  const std::size_t n = rows.size();
  std::vector<std::size_t> keys(n);
  starts.assign(buckets + 1, 0);
  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = key(rows.header(i));
    ++starts[keys[i] + 1];
  }
  for (std::size_t b = 0; b < buckets; ++b) starts[b + 1] += starts[b];
  std::vector<std::size_t> at(starts.begin(), starts.end() - 1);
  Rows out{rows.stride, std::vector<std::byte>(rows.bytes.size()), rows.limit};
  for (std::size_t i = 0; i < n; ++i) {
    std::memcpy(out.bytes.data() + at[keys[i]]++ * rows.stride, rows.bytes.data() + i * rows.stride, rows.stride);
  }
  return out;
}

std::span<const std::byte> part(const Rows& rows, const std::vector<std::size_t>& starts, std::size_t b) {
  return {rows.bytes.data() + starts[b] * rows.stride, (starts[b + 1] - starts[b]) * rows.stride};
}

// Sends `outgoing` to `send_to` and appends what `recv_from` send to `rows`.
void exchange_crystal(const Comm& comm, Rows& rows, int stage, int send_to, std::span<const int> recv_from,
                      std::span<const std::byte> outgoing) {
  const int tag = crystal_tag + 2 * stage;
  const std::uint64_t count = outgoing.size();
  std::array<std::uint64_t, 2> counts{};
  std::vector<Request> r;
  for (std::size_t k = 0; k < recv_from.size(); ++k) {
    r.push_back(comm.irecv(std::span<std::uint64_t>(&counts[k], 1), recv_from[k], tag));
  }
  r.push_back(comm.isend(std::span<const std::uint64_t>(&count, 1), send_to, tag));
  Request::wait_all(r);
  r.clear();
  std::size_t at = rows.bytes.size();
  detail::check_incoming(rows.size(), counts[0], rows.stride, rows.limit);
  detail::check_incoming(rows.size() + counts[0] / rows.stride, counts[1], rows.stride, rows.limit);
  rows.bytes.resize(at + counts[0] + counts[1]);
  for (std::size_t k = 0; k < recv_from.size(); ++k) {
    r.push_back(comm.irecv_bytes(std::span<std::byte>(rows.bytes.data() + at, counts[k]), recv_from[k], tag + 1));
    at += counts[k];
  }
  r.push_back(comm.isend_bytes(outgoing, send_to, tag + 1));
  Request::wait_all(r);
}

void crystal(const Comm& comm, Rows& rows) {
  const int id = comm.rank();
  int bl = 0, n = comm.size();
  for (int stage = 0; n > 1; ++stage) {
    const int nl = (n + 1) / 2, bh = bl + nl;
    const bool low = id < bh;
    // A stable partition in place: rows for this half move forward, the
    // others are copied out.
    std::size_t leaving = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) leaving += (rows.header(i).dest < bh) != low;
    std::vector<std::byte> outgoing(leaving * rows.stride);
    std::size_t kept = 0, left = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const std::byte* row = rows.bytes.data() + i * rows.stride;
      if ((rows.header(i).dest < bh) == low) {
        if (kept != i) std::memmove(rows.bytes.data() + kept * rows.stride, row, rows.stride);
        ++kept;
      } else {
        std::memcpy(outgoing.data() + left++ * rows.stride, row, rows.stride);
      }
    }
    rows.bytes.resize(kept * rows.stride);
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
    exchange_crystal(comm, rows, stage, to, std::span<const int>(from.data(), nfrom), outgoing);
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
  std::vector<std::size_t> starts;
  const Rows out = bucket(rows, p, [](const Header& h) { return static_cast<std::size_t>(h.dest); }, starts);
  std::vector<std::uint64_t> send_counts(p), recv_counts(p);
  for (std::size_t r = 0; r < p; ++r) send_counts[r] = starts[r + 1] - starts[r];
  comm.alltoall(std::span<const std::uint64_t>(send_counts), std::span<std::uint64_t>(recv_counts));
  std::vector<std::size_t> sc(send_counts.begin(), send_counts.end()), rc(recv_counts.begin(), recv_counts.end());
  std::size_t held = 0;
  for (const std::uint64_t c : recv_counts) {
    check_rows(held, c, rows.limit);
    held += c;
  }
  rows.bytes.resize(std::accumulate(rc.begin(), rc.end(), std::size_t{0}) * rows.stride);
  comm.alltoallv_bytes(out.bytes, sc, rows.bytes, rc, rows.stride);
}

void nbx(const Comm& comm, Rows& rows) {
  const int tag = nbx_tag + static_cast<int>(comm.next_sequence() % 2);
  const auto me = static_cast<std::size_t>(comm.rank());
  std::vector<std::size_t> starts;
  const Rows out = bucket(rows, static_cast<std::size_t>(comm.size()),
                          [](const Header& h) { return static_cast<std::size_t>(h.dest); }, starts);
  const std::span<const std::byte> self = part(out, starts, me);
  rows.bytes.assign(self.begin(), self.end());
  std::vector<Request> sends;
  for (std::size_t r = 0; r < starts.size() - 1; ++r) {
    if (r != me && starts[r + 1] > starts[r]) sends.push_back(comm.issend_bytes(part(out, starts, r), static_cast<int>(r), tag));
  }
  Request barrier;
  bool barrier_started = false;
  while (true) {
    if (const auto e = comm.iprobe(Comm::any_source, tag)) {
      detail::check_incoming(rows.size(), e->bytes, rows.stride, rows.limit);
      const std::size_t at = rows.bytes.size();
      rows.bytes.resize(at + e->bytes);
      comm.irecv_bytes(std::span<std::byte>(rows.bytes.data() + at, e->bytes), e->source, tag).wait();
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

void detail::check_incoming(std::size_t held, std::uint64_t bytes, std::size_t stride, std::size_t limit) {
  if (bytes % stride != 0) {
    throw std::runtime_error("transport::route: a message of " + std::to_string(bytes) +
                             " bytes is not whole records; the exchange is corrupt");
  }
  check_rows(held, bytes / stride, limit);
}

Strategy detail::resolve(Strategy s, int size) {
  if (s != Strategy::automatic) return s;
  return size <= 64 ? Strategy::alltoallv : Strategy::crystal;
}

detail::Arrivals detail::route_host(const Comm& comm, std::span<const int> dest, std::span<const std::byte> payload,
                                    std::size_t width, Strategy strategy) {
  const bool bad_dest = std::any_of(dest.begin(), dest.end(), [&](int d) { return d < 0 || d >= comm.size(); });
  const bool bad_rows = payload.size() != dest.size() * width;
  // One reduction: the maxima of the flag, of the width and the strategy, and
  // of their negatives, which agree only if every rank passed the same; and
  // the largest batch, which bounds what any rank can receive.
  const auto w = static_cast<std::int64_t>(width), st = static_cast<std::int64_t>(strategy);
  std::array<std::int64_t, 6> v{bad_dest || bad_rows, w, -w, st, -st, static_cast<std::int64_t>(dest.size())};
  comm.allreduce(std::span<std::int64_t>(v), Op::max);
  if (v[0] != 0 || v[1] != -v[2] || v[3] != -v[4]) {
    throw std::invalid_argument(
        "transport::route: a rank names a destination outside the communicator, passes a payload that is not a row "
        "per record, or differs from another in width or strategy");
  }
  const auto largest = static_cast<std::size_t>(v[5]);
  const auto p = static_cast<std::size_t>(comm.size());
  const std::size_t limit = largest > std::numeric_limits<std::size_t>::max() / p ? std::numeric_limits<std::size_t>::max()
                                                                                   : largest * p;
  Rows rows{sizeof(Header) + width, std::vector<std::byte>(dest.size() * (sizeof(Header) + width)), limit};
  for (std::size_t i = 0; i < dest.size(); ++i) {
    const Header h{dest[i], comm.rank()};
    std::byte* row = rows.bytes.data() + i * rows.stride;
    std::memcpy(row, &h, sizeof h);
    if (width > 0) std::memcpy(row + sizeof h, payload.data() + i * width, width);
  }
  switch (resolve(strategy, comm.size())) {
    case Strategy::crystal: crystal(comm, rows); break;
    case Strategy::alltoallv: alltoallv(comm, rows); break;
    case Strategy::nbx: nbx(comm, rows); break;
    case Strategy::automatic: break;
  }
  const auto by_source = [](const Header& h) { return static_cast<std::size_t>(h.source); };
  bool sorted = true;
  for (std::size_t i = 1; i < rows.size() && sorted; ++i) sorted = rows.header(i - 1).source <= rows.header(i).source;
  std::vector<std::size_t> starts;
  Rows done = sorted ? std::move(rows) : bucket(rows, static_cast<std::size_t>(comm.size()), by_source, starts);
  return Arrivals{done.stride, std::move(done.bytes)};
}

}  // namespace soma::transport
