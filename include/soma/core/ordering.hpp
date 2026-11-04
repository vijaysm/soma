// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_ORDERING_HPP_
#define SOMA_CORE_ORDERING_HPP_

#include <cstdint>

namespace soma {

enum class Ordering : std::uint8_t { file, type_major_sfc, tiled };

// Ordering used when the caller does not choose one.
inline constexpr Ordering default_ordering = Ordering::type_major_sfc;

}  // namespace soma

#endif  // SOMA_CORE_ORDERING_HPP_
