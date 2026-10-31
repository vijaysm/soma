// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_TYPES_HPP_
#define SOMA_CORE_TYPES_HPP_

#include <cstdint>

#include "soma/config.hpp"

namespace soma {

// Local ordinal: an entity index on this rank.
#if SOMA_LO_64
using LO = std::int64_t;
#else
using LO = std::int32_t;
#endif

// CSR offsets. Separate from LO because nnz overflows 32 bits long before
// entity counts do.
#if SOMA_OFFSET_64
using Offset = std::int64_t;
#else
using Offset = std::int32_t;
#endif

// Global ordinal: cross-rank identity (the .h5m file id for read entities).
using GO = std::int64_t;

using Real = double;

inline constexpr LO invalid_lo = -1;
inline constexpr GO invalid_go = -1;

}  // namespace soma

#endif  // SOMA_CORE_TYPES_HPP_
