// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_ATM_REFERENCE_HPP_
#define SWE_ATM_REFERENCE_HPP_

// Reference solutions on latitude-longitude grids, from NetCDF-4 files (read as
// HDF5: a variable is a dataset, its coordinates the datasets named for them),
// and Williamson's height errors against them on their own grid.

#include <filesystem>
#include <string>
#include <vector>

#include "diagnostics.hpp"
#include "model.hpp"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"

namespace swe {

struct ReferenceSpec {
  std::filesystem::path file;
  std::string variable = "h", lat = "lat", lon = "lon";
  int time_index = -1;  // of a variable (time, lat, lon); -1: the last
};

// Degrees; latitudes ascending, longitudes in [0, 360) ascending; values
// nlat x nlon (row i = latitude i).
struct Reference {
  std::vector<double> lat, lon, values;
};

// Collective: rank 0 reads, every rank receives. The grid is normalised: rows
// reversed when latitudes run north to south, longitudes moved into [0, 360)
// and the columns sorted by them, and a variable of shape (time, lat, lon)
// sliced at time_index. Throws std::runtime_error on every rank, naming the file
// and variable, when the file or a dataset cannot be read, the variable's shape
// is not (lat, lon) or (time, lat, lon) of the coordinates' lengths, or the
// time index is out of range; and when SOMA was built without HDF5
// (SOMA_ENABLE_HDF5).
Reference read_reference(const soma::Comm& comm, const ReferenceSpec& spec);

// Collective: h sampled at the reference's grid points, and the normalised
// l1, l2 and linf errors there, the integrals weighted by cos(lat) times each
// point's share of the latitude and longitude spacing.
Norms compare(const soma::Comm& comm, const Space& space, const soma::HostMesh& mesh, int ne, const ConstValues& h,
              const Reference& reference);

// Collective: compare of the model's free-surface height h + hs, the field
// Williamson's norms (and published reference solutions) measure.
Norms compare_surface(const soma::Comm& comm, const SweModel& model, const Reference& reference);

}  // namespace swe

#endif  // SWE_ATM_REFERENCE_HPP_
