// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_ATM_OUTPUT_HPP_
#define SWE_ATM_OUTPUT_HPP_

// The model's state on disk: legacy VTK for viewing (each element as p x p
// sub-quads between its GLL points), and SOMA's parallel format (the mesh with
// h, hs and u on the GLL space) for restarts and analysis.

#include <filesystem>

#include "model.hpp"
#include "soma/parallel/comm.hpp"

namespace swe {

// Collective in name only (no communication): each rank owning elements writes
// stem.<rank>.vtk, its owned elements as p x p quads each, the points those
// elements' GLL points (repeated per element), with point fields h, hs and u
// (3 components). A rank owning none writes nothing.
void write_vtk_subquads(const soma::Comm& comm, const SweModel& model, const std::filesystem::path& stem);

// Collective: writes the model's mesh to `path` in SOMA's format with fields h,
// hs and u on its GLL space (adding them to the mesh the first time). Throws
// std::runtime_error when SOMA was built without HDF5 (SOMA_ENABLE_HDF5).
void write_soma_state(const soma::Comm& comm, SweModel& model, const std::filesystem::path& path);

}  // namespace swe

#endif  // SWE_ATM_OUTPUT_HPP_
