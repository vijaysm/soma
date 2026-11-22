// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_ATM_CONSTANTS_HPP_
#define SWE_ATM_CONSTANTS_HPP_

// The Earth's constants of Williamson et al. (1992): radius (m), rotation rate
// (1/s) and gravity (m/s^2).

namespace swe {

inline constexpr double radius = 6.37122e6;
inline constexpr double omega = 7.292e-5;
inline constexpr double gravity = 9.80616;

}  // namespace swe

#endif  // SWE_ATM_CONSTANTS_HPP_
