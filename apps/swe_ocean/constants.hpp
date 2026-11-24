// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_CONSTANTS_HPP_
#define SWE_OCEAN_CONSTANTS_HPP_

// The ocean's constants: the Earth's radius (m), rotation rate (1/s) and gravity (m/s^2) of the atmosphere
// (swe_atm/constants.hpp), the reference density (kg/m^3), the quadratic bottom drag coefficient and the
// least ocean depth (m) that the bathymetry is raised to.

namespace ocean {

inline constexpr double a = 6.37122e6;
inline constexpr double g = 9.80616;
inline constexpr double Omega = 7.292e-5;
inline constexpr double rho0 = 1025.0;
inline constexpr double C_d = 2.5e-3;
inline constexpr double H_min = 10.0;

}  // namespace ocean

#endif  // SWE_OCEAN_CONSTANTS_HPP_
