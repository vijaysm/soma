// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_VTK_AMR_HPP_
#define SOMA_AMR_VTK_AMR_HPP_

/// \file
/// VTK output of a hierarchy as overlapping AMR, which ParaView and VTK's
/// XML readers read: `path` (a .vthb file) lists the boxes of every level,
/// each written as image data (a .vti file) to the directory named like
/// `path` without its extension, as <stem>/<stem>_<level>_<box>.vti. Each box
/// holds the valid points of every field (cell fields as cell data, node
/// fields as point data, components interleaved, ASCII in the shortest form
/// that reads back bitwise equal) and vtkGhostType, 8 (hidden) on the cells a
/// finer level covers. Points are at the geometry's lower corner plus their
/// index times the level's spacing. A 2D box's amr_box is empty in z
/// (hi = lo - 1, an empty dimension of vtkAMRBox): with "0 0" VTK's reader
/// does not hide the cells under finer levels. Checked with ParaView 6.1's
/// vtkXMLUniformGridAMRReader: values, levels, and visible cells adding up to
/// the domain.
///
/// Throws std::invalid_argument, before any file is written, for a path
/// without the .vthb extension, a field that does not match the hierarchy's
/// levels or is not cell- or node-centred, a name that is empty, repeated,
/// vtkGhostType, or not letters, digits, '_', '.' and '-', or a value that is
/// not finite. Each file is written to a temporary and renamed into place,
/// the .vthb last, so a failure never leaves a .vthb naming files that are
/// missing; failures throw std::runtime_error.
///
/// The overload taking a communicator is collective over the hierarchy's
/// ranks: each rank writes the pieces of the boxes it owns, and rank 0 writes
/// the .vthb once every piece is in place. A piece is named by its level and
/// global box id, so the files are those the serial overload writes for the
/// whole hierarchy. Bad input on any rank, or ranks giving different fields,
/// throws std::invalid_argument on every rank before any file is written; a
/// failed write throws std::runtime_error on every rank. The overload without
/// one is the same over Comm::self() and needs a hierarchy of one rank.

#include <filesystem>
#include <string>
#include <vector>

#include "soma/amr/hierarchy.hpp"
#include "soma/parallel/comm.hpp"

namespace soma::amr {

struct VtkAmrField {
  std::string name;
  const MultiLevelField<double>* field = nullptr;
};

void write_vtk_amr(const Comm& comm, const std::filesystem::path& path, const Hierarchy& hierarchy,
                   const std::vector<VtkAmrField>& fields);

void write_vtk_amr(const std::filesystem::path& path, const Hierarchy& hierarchy, const std::vector<VtkAmrField>& fields);

}  // namespace soma::amr

#endif  // SOMA_AMR_VTK_AMR_HPP_
