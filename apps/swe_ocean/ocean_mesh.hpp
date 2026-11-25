// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_OCEAN_MESH_HPP_
#define SWE_OCEAN_OCEAN_MESH_HPP_

// The ocean's mesh, its spectral-element space and its bathymetry, built from a height file in one pipeline:
//
//   1. the cubed sphere at ne on the unit sphere;
//   2. refined toward the coast (soma::refine_graded, np = order + 1, spacing coast_km, lengths in km on the
//      Earth, target coast_target of the coastline decimated at coast_km); coast_km <= 0: no refinement;
//   3. smoothed by soma::quality_smooth on the unit sphere, every vertex free, with the sweeps of `optimize`
//      (shape, angle, polish) and, for area_ratio >= 1, that area ratio bound;
//   4. scaled to the Earth's radius a, rebalanced, given a ghost layer and distributed edges and a space of
//      the order; the bathymetry averaged on it (ocean::average);
//   5. land elements (no point with averaged H > 0) removed, then every element not joined through shared
//      edges to the seed element (the owned element containing (seed_lat, seed_lon), the least gid on a tie);
//      the removed areas reported in km^2, land and disconnected apart;
//   6. rebalanced, given a ghost layer, edges and a space again; the bathymetry averaged again for H and
//      roughness, the land flags kept from step 4 by (element gid, point): a point land there takes H_min;
//      then smoothed (ocean::smooth).
//
// Elements keep their gids throughout; every step is the same, gid for gid and bit for bit, on any number of
// ranks, so are the mesh, H and the removed areas. The space refers to *mesh: the mesh lives behind a pointer so
// that an OceanMesh can move without the space's views dangling.
//
// write_ocean_mesh writes the mesh as a SOMA file (soma::write_soma) with H, roughness and the land flags as
// fields of the space's GLL points (ocean:H, ocean:roughness, ocean:land; set on *mesh, replacing any already
// there) and the removed areas as mesh values. read_ocean_mesh reads it back at any number of ranks: the mesh,
// a ghost layer, edges, the space of the order given and the bathymetry by (element gid, point), the same bits
// as written; sweeps 0, rx0 recomputed (max_rx0), smoothed true.

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>

#include "bathymetry.hpp"
#include "soma/algorithms/refine_graded.hpp"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "types.hpp"

namespace ocean {

// ne: cube-edge elements; order: the space's order; transition, reruns: refine_graded's; coast_km: the GLL
// spacing (km) refined to near the coast (<= 0: none); optimize: quality_smooth's shape, angle and polish
// sweeps; area_ratio: its area ratio bound (0: none, else >= 1); seed_lat, seed_lon (degrees): a point of the
// world ocean; bathymetry: the height file and the averaging and smoothing options.
struct MeshOptions {
  int ne = 30, order = 3, transition = 3, reruns = 4;
  double coast_km = 10;
  std::array<int, 3> optimize{50, 5, 10};
  double area_ratio = 0;
  double seed_lat = 0, seed_lon = -150;
  BathymetryOptions bathymetry;
};

// mesh: the ocean's elements on the sphere of radius a, a ghost layer and distributed edges on more than one
// rank; space: the space of the order over *mesh; bathymetry: per row of the space; removed_*_km2: the area of
// the elements removed as land and as not joined to the world ocean; refined: what refine_graded reported
// (its mesh moved out; default when read from a file); coastline_points: the decimated coastline's points (every
// rank holds them all; 0 without refinement or when read from a file).
struct OceanMesh {
  std::unique_ptr<soma::HostMesh> mesh;
  std::unique_ptr<Space> space;
  Bathymetry bathymetry;
  double removed_land_km2 = 0, removed_disconnected_km2 = 0;
  soma::GradedResult refined;
  std::size_t coastline_points = 0;
};

// The pipeline above. Collective. Throws std::invalid_argument on every rank for ne < 1, order < 1, a seed
// latitude outside [-90, 90] or an area ratio in (0, 1); std::runtime_error on every rank when no element
// contains the seed or that element is land (no point with H > 0), or when no element is left; what the steps
// throw (coastline, refine_graded, quality_smooth, average, smooth).
OceanMesh build_ocean_mesh(const soma::Comm& comm, const MeshOptions& options);

// Collective: writes `path` as described above. aggregators: soma::SomaWriteOptions::aggregators (0: every rank
// writes; M: only M ranks open the file, for a filesystem such as NFS where many ranks writing one file crawl).
// Throws what soma::write_soma throws.
void write_ocean_mesh(const soma::Comm& comm, const OceanMesh& ocean, const std::filesystem::path& path,
                      int aggregators = 0);

// Collective: reads `path` written by write_ocean_mesh. Throws std::invalid_argument on every rank for order
// < 1 or when the file's GLL fields are missing or not of (order + 1)^2 points per element; what
// soma::read_soma throws. aggregators: soma::SomaReadOptions::aggregators, as write_ocean_mesh's.
OceanMesh read_ocean_mesh(const soma::Comm& comm, const std::filesystem::path& path, int order,
                          int aggregators = 0);

}  // namespace ocean

#endif  // SWE_OCEAN_OCEAN_MESH_HPP_
