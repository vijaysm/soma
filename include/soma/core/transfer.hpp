// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_TRANSFER_HPP_
#define SOMA_CORE_TRANSFER_HPP_

#include <array>
#include <string>
#include <vector>

namespace soma {

// What a mirror copies. The default is lean: all topology dimensions and
// coordinates, no fields. Adjacency selection is not an option yet.
struct TransferOptions {
  std::array<bool, 4> dims{true, true, true, true};
  bool coordinates = true;
  std::vector<std::string> fields{};  // field names; {"*"} = all

  static TransferOptions everything() {
    TransferOptions o;
    o.fields = {"*"};
    return o;
  }
};

}  // namespace soma

#endif  // SOMA_CORE_TRANSFER_HPP_
