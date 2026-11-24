// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_BATHYMETRY_HPP_
#define SWE_OCEAN_BATHYMETRY_HPP_

// The ocean's depth H at every GLL point of a spectral-element space on the sphere, from a latitude-longitude
// grid of heights z (ETOPO1: m, negative below sea level) read by window (grid_file.hpp).
//
// average: each point's footprint is the disc (great-circle radius rho = sqrt(m / pi), m the assembled mass at
// the point) of the area the point represents. Its depth is the mean of the depths -z of the grid cells whose
// centres lie in the disc, each weighted by its area cos(lat) dlat dlon (the cell centred on its coordinates),
// land (z > 0) counting as depth 0; its roughness is the variance of those depths about the mean, with the
// same weights (m^2). A footprint with rho < bilinear_below_m, or holding no cell centre, takes the bilinear
// interpolation of the depths at the point instead, and roughness 0. A point is land where its averaged H is
// at most 0. Each rank reads, strip by strip of strip_rows rows, only the columns its points' footprints
// reach in that strip (a disc reaching a pole reaches every longitude; a disc across the antimeridian wraps),
// and every row once. Each point sums its cells in a fixed order (rows south to north, then columns east from
// the west of its footprint), whatever the windows, and dss makes shared points continuous: H is the same,
// bit for bit, on any number of ranks.
//
// smooth: H = max(H, H_min) at every point (land too: with land at 0 the slope factor is 1 at every coast and
// the smoothing could never finish), then sweeps H = max(H + kappa m L(H), H_min), L the space's weak
// Laplacian (soma::sem::laplacian, DSS-continuous), kappa = 0.5 / lambda_max (lambda_max of -L from 20 power
// iterations from a fixed start), until max_rx0 <= r_max or sweep_limit sweeps; then H = max(H, H_min) again.
// The mask m is 1 at the points of the elements whose largest rx0 exceeds r_max and of the elements touching
// them (one element further: the steep elements alone can settle to a steady state still above r_max, at
// order 4 and up), 0 elsewhere. The clamp at H_min in every sweep keeps H positive, so rx0, which steers the
// mask and the stop, means what it says: diffusion of a jump undershoots, below 0 next to a coast. smoothed is
// true exactly when the final rx0 is at most r_max. Every reduction goes through the space (dss, integral) or
// an allreduce of a maximum, so the result is the same, bit for bit, on any number of ranks.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "grid_file.hpp"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "types.hpp"

namespace ocean {

// grid: the height file. r_max: the largest slope factor smoothing leaves; H_min (m): the least depth;
// sweep_limit: the most smoothing sweeps; bilinear_below_m (m): footprint radii below this interpolate;
// strip_rows: rows read and processed at a time (540, nine degrees of ETOPO1's arc minutes).
struct BathymetryOptions {
  GridSpec grid;
  double r_max = 0.2, H_min = 10;
  int sweep_limit = 10000;
  double bilinear_below_m = 2000;
  std::size_t strip_rows = 540;
};

// H and roughness: one component per row of space() (every element's points, ghosts included, continuous);
// land: per row, 1 where the averaged H was at most 0. sweeps, rx0 and smoothed: what smooth did (the sweeps
// taken, max_rx0 at the end and whether it reached r_max). peak_window: the most values in one window this
// rank read.
struct Bathymetry {
  Values H, roughness;
  std::vector<std::uint8_t> land;
  int sweeps = 0;
  double rx0 = 0;
  bool smoothed = true;
  std::size_t peak_window = 0;
};

// The footprint average at every point. Collective. Throws std::invalid_argument on every rank if the space
// is not on a sphere or not over this mesh, strip_rows is 0, or the grid has fewer than 2 rows or columns;
// std::runtime_error as open_grid does, and on every rank if any rank fails to read its windows.
Bathymetry average(const soma::Comm& comm, const Space& space, const soma::HostMesh& mesh,
                   const BathymetryOptions& options);

// Raises H to H_min, smooths it until max_rx0 <= r_max (or sweep_limit sweeps: smoothed false) and raises it
// again; sets sweeps, rx0 and smoothed. Collective. Throws std::invalid_argument for H not one component per
// row of the space, r_max not in (0, 1), H_min not positive or sweep_limit negative; std::runtime_error if
// the power iterations find no positive eigenvalue.
void smooth(const soma::Comm& comm, const Space& space, Bathymetry& bathymetry, const BathymetryOptions& options);

// The largest slope factor rx0(H_a, H_b) = |H_a - H_b| / (H_a + H_b) between neighbouring GLL points of an
// element (along i and along j) over the owned elements of every rank (0 between two zeros). Collective.
// Throws std::invalid_argument for H not one component per row of the space.
double max_rx0(const soma::Comm& comm, const Space& space, const ConstValues& H);

}  // namespace ocean

#endif  // SWE_OCEAN_BATHYMETRY_HPP_
