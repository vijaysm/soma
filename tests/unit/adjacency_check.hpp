// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_ADJACENCY_CHECK_HPP_
#define SOMA_TESTS_ADJACENCY_CHECK_HPP_

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/topology/adjacency.hpp"

namespace soma_test {

// A graph copied to plain vectors, for exact comparison across spaces.
struct HostGraph {
  std::vector<soma::Offset> offsets;
  std::vector<soma::LO> targets;
  std::vector<std::uint8_t> codes;

  soma::LO rows() const { return offsets.empty() ? 0 : static_cast<soma::LO>(offsets.size() - 1); }
  std::vector<soma::LO> row(soma::LO r) const {
    return {targets.begin() + offsets[static_cast<std::size_t>(r)],
            targets.begin() + offsets[static_cast<std::size_t>(r) + 1]};
  }
  std::vector<std::uint8_t> row_codes(soma::LO r) const {
    return {codes.begin() + offsets[static_cast<std::size_t>(r)],
            codes.begin() + offsets[static_cast<std::size_t>(r) + 1]};
  }
  friend bool operator==(const HostGraph&, const HostGraph&) = default;
};

template <class V>
std::vector<typename V::non_const_value_type> to_vector(const V& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  return {h.data(), h.data() + h.extent(0)};
}

template <class G>
HostGraph to_host(const G& g) {
  return {to_vector(g.offsets), to_vector(g.targets), to_vector(g.codes)};
}

}  // namespace soma_test

#endif  // SOMA_TESTS_ADJACENCY_CHECK_HPP_
