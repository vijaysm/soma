// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_VERTICAL_HPP_
#define SWE_OCEAN_VERTICAL_HPP_

// The ocean's vertical coordinate: K levels in every column, level k (k = 1..K, top first) of thickness
// h_k = h0_k(H) (1 + eta / H), where the reference thicknesses h0_k(H) sum to the depth H. Two coordinates
// differ only in h0_k: ROMS's terrain-following s-coordinate (Vtransform 2, Vstretching 4) and z-star
// geopotential levels cut off at the bottom, kept at least h_min thick. Nothing here communicates; every
// function is local and deterministic.

#include <cstdint>
#include <span>
#include <vector>

namespace ocean {

enum class Vertical : std::uint8_t { s, zstar };

// The coordinate and its parameters. For s: theta_s in (0, 10] (surface stretching), theta_b in [0, 4]
// (bottom stretching), hc >= 0 (m, the depth of the transition to uniform sigma). For zstar: z holds the K + 1
// interfaces z[0] = 0 > z[1] > ... > z[K] (m, negative down); empty selects a default stretched profile of
// K levels reaching 5500 m, z_k = -5500 (exp(3 k / K) - 1) / (exp(3) - 1), the levels thickening with depth
// by a factor of about e^(3/K) per level. h_min (m) is the least thickness of a z-star level.
struct VerticalSpec {
  Vertical kind = Vertical::s;
  int levels = 10;
  double theta_s = 5, theta_b = 2, hc = 100, h_min = 1e-3;
  std::vector<double> z;
};

// The K = spec.levels reference thicknesses h0 (m, top first) of a column of depth H > 0 (m), summing to H to
// roundoff; the z-star interfaces are z0_0 = 0, z0_K = -H and z0_k = max(z_k, -H + (K - k) h_min) in between,
// so the bottom wet level is a partial cell and the levels below it have the thickness h_min. Not collective.
// Throws std::invalid_argument for K < 1, H <= 0, theta_s outside (0, 10], theta_b outside [0, 4], hc < 0,
// h0.size() != K, z-star with H < K h_min, or a z that is not K + 1 descending interfaces from 0.
void reference_thicknesses(const VerticalSpec& spec, double H, std::span<double> h0);

// The slope factor |Ha - Hb| / (Ha + Hb) of two depths (ROMS's rx0; 0 when both are 0).
double rx0(double Ha, double Hb);

// The Haney number of two columns whose K + 1 interfaces are za and zb (m): the largest over levels k of
// |za_k - zb_k + za_(k-1) - zb_(k-1)| / |za_k + zb_k - za_(k-1) - zb_(k-1)|, levels with no thickness in
// both columns skipped. Throws std::invalid_argument if the sizes differ or are below 2.
double rx1(std::span<const double> za, std::span<const double> zb);

}  // namespace ocean

#endif  // SWE_OCEAN_VERTICAL_HPP_
