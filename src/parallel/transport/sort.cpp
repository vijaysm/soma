// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/transport/sort.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "soma/parallel/transport/comm_plan.hpp"

namespace soma::transport {

namespace {

constexpr std::size_t samples_per_rank = 32;

using Word = std::uint64_t;

bool less(const Word* a, const Word* b, std::size_t words) {
  return std::lexicographical_compare(a, a + words, b, b + words);
}

bool equal(const Word* a, const Word* b, std::size_t words) { return std::equal(a, a + words, b); }

// Positions of the rows of `keys` in ascending key order, ties in place.
std::vector<std::size_t> order(std::span<const Word> keys, std::size_t words) {
  std::vector<std::size_t> o(keys.size() / words);
  std::iota(o.begin(), o.end(), std::size_t{0});
  std::stable_sort(o.begin(), o.end(),
                   [&](std::size_t a, std::size_t b) { return less(&keys[a * words], &keys[b * words], words); });
  return o;
}

// Rows of `words` keys as a batch for route(), each to dest[i].
RecordBatch<Kokkos::HostSpace> rows(std::span<const Word> keys, std::size_t words, const std::vector<int>& dest) {
  RecordBatch<Kokkos::HostSpace> b(dest.size(), words * sizeof(Word), "transport::rank_keys");
  for (std::size_t i = 0; i < dest.size(); ++i) {
    b.dest(i) = dest[i];
    if (words > 0) std::memcpy(&b.payload(i, 0), &keys[i * words], words * sizeof(Word));
  }
  return b;
}

std::vector<Word> words_of(const RecordBatch<Kokkos::HostSpace>& b, std::size_t words) {
  std::vector<Word> out(b.size() * words);
  if (!out.empty()) std::memcpy(out.data(), b.payload.data(), out.size() * sizeof(Word));
  return out;
}

}  // namespace

KeyRanks rank_keys(const Comm& comm, std::span<const std::uint64_t> keys, std::size_t words, bool distinct,
                   Strategy strategy) {
  using Host = Kokkos::DefaultHostExecutionSpace;
  const auto w = static_cast<std::int64_t>(words), d = static_cast<std::int64_t>(distinct);
  std::array<std::int64_t, 5> v{words == 0 || keys.size() % std::max<std::size_t>(words, 1) != 0, w, -w, d, -d};
  comm.allreduce(std::span<std::int64_t>(v), Op::max);
  if (v[0] != 0 || v[1] != -v[2] || v[3] != -v[4]) {
    throw std::invalid_argument(
        "transport::rank_keys: a rank passes no words per key, keys that are not whole rows, or a different words or "
        "distinct than another rank");
  }
  const std::size_t n = keys.size() / words;
  const auto p = static_cast<std::size_t>(comm.size());

  // Regular samples of the local order to rank 0, which picks the splitters.
  const std::vector<std::size_t> local = order(keys, words);
  std::vector<Word> sample;
  const std::size_t ns = std::min(n, samples_per_rank);
  for (std::size_t k = 0; k < ns; ++k) {
    const std::size_t at = local[(2 * k + 1) * n / (2 * ns)];
    sample.insert(sample.end(), keys.begin() + static_cast<std::ptrdiff_t>(at * words),
                  keys.begin() + static_cast<std::ptrdiff_t>((at + 1) * words));
  }
  const auto gathered = route(Host{}, comm, rows(sample, words, std::vector<int>(ns, 0)), strategy);
  std::vector<Word> splitters;
  if (comm.rank() == 0) {
    const std::vector<Word> all = words_of(gathered, words);
    const std::vector<std::size_t> o = order(all, words);
    const std::size_t m = o.size();
    for (std::size_t r = 1; r < p && m > 0; ++r) {
      const std::size_t at = o[r * m / p];
      splitters.insert(splitters.end(), all.begin() + static_cast<std::ptrdiff_t>(at * words),
                       all.begin() + static_cast<std::ptrdiff_t>((at + 1) * words));
    }
  }
  std::uint64_t ns_all = splitters.size();
  comm.bcast(std::span<std::uint64_t>(&ns_all, 1), 0);
  splitters.resize(ns_all);
  comm.bcast(std::span<std::uint64_t>(splitters), 0);
  const std::size_t nsplit = splitters.size() / words;

  // Each key to the rank of its range; equal keys meet on one rank.
  std::vector<int> dest(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t lo = 0, hi = nsplit;  // first splitter greater than the key
    while (lo < hi) {
      const std::size_t mid = (lo + hi) / 2;
      if (less(&keys[i * words], &splitters[mid * words], words)) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    dest[i] = static_cast<int>(lo);
  }
  const auto sent = rows(keys, words, dest);
  const auto arrived = route(Host{}, comm, sent, strategy);
  const CommPlan plan = learn_plan(comm, sent, arrived);

  // Arrivals come by source, then the sender's order; a stable sort keeps that
  // order among equal keys.
  const std::vector<Word> got = words_of(arrived, words);
  const std::vector<std::size_t> o = order(got, words);
  std::vector<std::uint64_t> local_index(o.size());
  std::uint64_t count = 0;
  for (std::size_t k = 0; k < o.size(); ++k) {
    if (distinct && k > 0 && equal(&got[o[k] * words], &got[o[k - 1] * words], words)) {
      local_index[o[k]] = count - 1;
      continue;
    }
    local_index[o[k]] = count++;
  }
  const std::uint64_t offset = comm.exscan(count);
  for (std::uint64_t& x : local_index) x += offset;
  KeyRanks out;
  out.total = comm.allreduce(count, Op::sum);
  out.index.resize(n);
  plan.reverse(comm, std::span<const std::uint64_t>(local_index), std::span<std::uint64_t>(out.index));
  return out;
}

}  // namespace soma::transport
