// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_QUALITY_SMOOTH_HPP_
#define SOMA_ALGORITHMS_QUALITY_SMOOTH_HPP_

/// \file
/// Optimization-based smoothing of a surface mesh of triangles and quads.

#include <cstdint>
#include <vector>

#include "soma/algorithms/laplacian.hpp"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

namespace detail {

/// Per vertex lid (ghosts included), a colour from 0: no two corners of one
/// cell share a colour. Jones and Plassmann's rounds: an uncoloured vertex
/// whose priority, a hash of its gid, beats that of every uncoloured vertex
/// sharing a cell with it takes the least colour none of those has. Colours
/// depend on gids and connectivity only, so they are the same on any number
/// of ranks. `cells` is vertex_cells(mesh). Collective; on more than one rank
/// the mesh needs a ghost layer through vertices. Throws std::logic_error on
/// every rank if a round colours no vertex (ranks disagreeing on the mesh).
std::vector<std::int32_t> vertex_colours(const Comm& comm, const HostMesh& mesh, const VertexCells& cells);

}  // namespace detail

}  // namespace soma

#endif  // SOMA_ALGORITHMS_QUALITY_SMOOTH_HPP_
