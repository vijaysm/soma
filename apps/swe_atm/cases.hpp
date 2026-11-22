// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_ATM_CASES_HPP_
#define SWE_ATM_CASES_HPP_

// The shallow-water test cases: Williamson et al. (1992) cases 2, 5 and 6, and
// Galewsky et al. (2004) with and without its perturbation. Points are given
// in Cartesian coordinates on (or radially off) the sphere of radius
// swe::radius; lambda is longitude, theta latitude.

#include <array>
#include <cstdint>
#include <string_view>

namespace swe {

enum class Case : std::uint8_t { williamson2, williamson5, williamson6, galewsky, galewsky_steady };

// "w2", "w5", "w6", "galewsky", "galewsky-steady"; throws std::invalid_argument
// listing them for any other name.
Case case_from_name(std::string_view name);
std::string_view case_name(Case c);

// One point's state: depth h (m; the free surface less the topography),
// topography hs (m), velocity u (m/s, Cartesian, tangent) and the Coriolis
// parameter f (1/s).
struct PointState {
  double h = 0, hs = 0;
  std::array<double, 3> u{};
  double f = 0;
};

// The state at t = 0. alpha is case 2's angle between the flow's axis and the
// pole (0 for the other cases); f = 2 Omega (-cos lambda cos theta sin alpha +
// sin theta cos alpha).
PointState initial(Case c, const std::array<double, 3>& x, double alpha = 0);

// The exact state at time t: case 2 only (steady); throws std::invalid_argument
// for the others.
PointState exact(Case c, const std::array<double, 3>& x, double alpha, double t);

}  // namespace swe

#endif  // SWE_ATM_CASES_HPP_
