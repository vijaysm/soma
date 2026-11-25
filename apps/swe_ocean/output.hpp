// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_OUTPUT_HPP_
#define SWE_OCEAN_OUTPUT_HPP_

// The ocean's state on disk: SOMA snapshots (the mesh with the state as fields of its GLL space) for restarts and
// analysis, and latitude-longitude fields as NumPy .npy files for a quick look (as swe_atm's --latlon). The .npy
// writer is a copy of swe_atm's write_npy: linking swe_atm_core would put that application's model.hpp, cases.hpp
// and output.hpp, of the same names as the ocean's, on the ocean's include path.

#include <cstddef>
#include <filesystem>
#include <functional>
#include <span>

#include "model.hpp"
#include "sample.hpp"
#include "soma/parallel/comm.hpp"

namespace ocean {

// Collective: runs `write` (a rank's own output, which may do nothing) on every rank, then agrees: when it threw on
// any rank, every rank throws std::runtime_error with the message of the lowest such rank, so no rank is left
// waiting in a later collective.
void collectively(const soma::Comm& comm, const std::function<void()>& write);

// Collective: writes the model's mesh to `path` in SOMA's format (soma::write_soma) with fields of its GLL space:
// eta (m), u (m/s, 3 Cartesian components), H (m), h (m, the K level thicknesses) and, for each tracer t,
// tracer<t> (its K level values T_k = hT_k / h_k). The fields are added to the mesh the first time and
// overwritten later. aggregators: soma::SomaWriteOptions::aggregators, as write_ocean_mesh's. Throws what
// soma::write_soma throws (on every rank), std::runtime_error when SOMA was built without HDF5.
void write_snapshot(const soma::Comm& comm, OceanModel& model, const std::filesystem::path& path,
                    int aggregators = 0);

// Collective: on a model on the sphere, samples eta, H and the eastward and northward velocity on the grid of nlat
// latitudes -90 + 180 i / (nlat - 1) and nlon longitudes 360 j / nlon (degrees; NaN over land, Sampler::sample)
// and writes, on rank 0, stem_eta.npy, stem_H.npy, stem_u_east.npy, stem_u_north.npy (nlat x nlon), stem_lat.npy
// and stem_lon.npy. Throws std::invalid_argument for nlat < 2, nlon < 1 or a space not on the sphere, and on every
// rank std::runtime_error when a file cannot be written.
void write_latlon(const soma::Comm& comm, const OceanModel& model, const Sampler& sampler, int nlat, int nlon,
                  const std::filesystem::path& stem);

// Writes rows x cols doubles (C order) as a NumPy .npy file, format 1.0 ('<f8'). Not collective: call it on one
// rank. Throws std::runtime_error when the file cannot be written and std::invalid_argument when the sizes
// disagree.
void write_npy(const std::filesystem::path& path, std::span<const double> data, std::size_t rows, std::size_t cols);

}  // namespace ocean

#endif  // SWE_OCEAN_OUTPUT_HPP_
