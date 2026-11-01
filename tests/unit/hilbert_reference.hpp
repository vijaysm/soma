// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_HILBERT_REFERENCE_HPP_
#define SOMA_TESTS_HILBERT_REFERENCE_HPP_

#include <cstdint>
#include <utility>

namespace soma_test {

// Independent 2D reference: the classic rotate-and-flip conversion of (x, y) to
// the distance along the curve of an n x n grid, n a power of two up to 2^32.
// It shares no code with soma::sfc::hilbert_index.
inline std::uint64_t hilbert_reference_2d(std::uint64_t n, std::uint32_t x, std::uint32_t y) {
  std::uint64_t d = 0;
  for (std::uint64_t s = n / 2; s > 0; s /= 2) {
    const std::uint64_t rx = (x & s) != 0 ? 1U : 0U;
    const std::uint64_t ry = (y & s) != 0 ? 1U : 0U;
    d += s * s * ((3U * rx) ^ ry);
    if (ry == 0) {
      if (rx == 1) {
        x = static_cast<std::uint32_t>(n - 1 - x);
        y = static_cast<std::uint32_t>(n - 1 - y);
      }
      std::swap(x, y);
    }
  }
  return d;
}

}  // namespace soma_test

#endif  // SOMA_TESTS_HILBERT_REFERENCE_HPP_
