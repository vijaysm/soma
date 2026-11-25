// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_TESTS_SYNTHETIC_EARTH_HPP_
#define SWE_OCEAN_TESTS_SYNTHETIC_EARTH_HPP_

// The synthetic Earth of the ocean mesh tests (heights in m, latitude and longitude in degrees): ocean 1000 m
// deep; continent A, the box |lat|, |lon| <= 40, less a bay open to the north and east and holding an inland
// basin 500 m deep, {-16 < lat < -4, -16 < lon < -4}; continent B, the box |lat| <= 35, |lon - 90| <= 35,
// holding a lake 500 m deep, the disc of radius 8 degrees about (0, 90); land 500 m high. test_ocean_mesh.cpp
// says why each feature lies where it does on the cubed sphere at ne = 4.

#include <array>
#include <cmath>
#include <numbers>

namespace ocean::testing {

using P3 = std::array<double, 3>;

inline constexpr double deg = std::numbers::pi / 180;

// The unit vector at a latitude and longitude, and the angle between two unit vectors, in degrees.
inline P3 at(double lat, double lon) {
  return {std::cos(lat * deg) * std::cos(lon * deg), std::cos(lat * deg) * std::sin(lon * deg), std::sin(lat * deg)};
}
inline double angle_deg(const P3& u, const P3& v) {
  const P3 c{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
  return std::atan2(std::hypot(c[0], c[1], c[2]), u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / deg;
}

// The basin, the lake and the heights of the synthetic Earth, as described above.
inline bool in_basin(double lat, double lon) { return lat > -16 && lat < -4 && lon > -16 && lon < -4; }
inline constexpr double lake_lat = 0, lake_lon = 90, lake_radius = 8;
inline bool in_lake(double lat, double lon) {
  return angle_deg(at(lat, lon), at(lake_lat, lake_lon)) < lake_radius;
}

inline double heights(double lat, double lon) {
  constexpr double ocean = -1000, inland = -500, land = 500;
  if (std::abs(lat) <= 40 && std::abs(lon) <= 40) {
    const bool bay = (lat > 4 && lon > 4) || (lon > 18 && lat > -18) || (lat > 18 && lon > -18);
    if (bay) return ocean;
    return in_basin(lat, lon) ? inland : land;
  }
  if (std::abs(lat) <= 35 && std::abs(lon - 90) <= 35) return in_lake(lat, lon) ? inland : land;
  return ocean;
}

}  // namespace ocean::testing

#endif  // SWE_OCEAN_TESTS_SYNTHETIC_EARTH_HPP_
