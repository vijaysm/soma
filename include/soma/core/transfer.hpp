// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_TRANSFER_HPP_
#define SOMA_CORE_TRANSFER_HPP_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "soma/topology/adjacency.hpp"

namespace soma {

// What a mirror copies. The default is lean: all topology dimensions,
// coordinates and entity sets, no fields, no adjacency graphs. Element
// connectivity and coordinates refer to vertices, so keeping either without
// dims[0] is rejected with std::invalid_argument. Sets keep their members of
// the kept dimensions; asking for the members of a dropped one throws
// MissingSetMembers.
//
// adjacencies names the graphs the destination must hold. With copy they are
// copied from the source, which must hold each of them; with rebuild the
// destination builds them from its own topology. A graph between dimensions
// that are not both kept is rejected with std::invalid_argument.
struct TransferOptions {
  enum class AdjSource : std::uint8_t { copy, rebuild };

  std::array<bool, 4> dims{true, true, true, true};
  bool coordinates = true;
  std::vector<std::string> fields{};  // field names; {"*"} = all
  AdjacencyRequest adjacencies{};
  AdjSource adj_source = AdjSource::rebuild;
  bool sets = true;  // last, so that positional initializers keep their meaning

  static TransferOptions everything() {
    TransferOptions o;
    o.fields = {"*"};
    return o;
  }
};

}  // namespace soma

#endif  // SOMA_CORE_TRANSFER_HPP_
