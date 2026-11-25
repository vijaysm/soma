// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_CONFORMITY_HPP_
#define SOMA_ALGORITHMS_CONFORMITY_HPP_

/// \file
/// Sides of a mesh that tile without matching: what FvSpace and FemSpace
/// refuse.

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"

namespace soma::detail {

/// Not collective. Counts, over the cells this rank holds, the sides that
/// match no other cell's side yet meet one: a side with another's corner
/// strictly inside one of its edges (a hanging vertex, as refine_marked leaves
/// without MarkedOptions::conforming), and a quad that two triangles of other
/// cells tile (a hex against two tets, no pyramid between). 0 for a
/// conforming mesh. On more than one rank, a ghost layer through vertices
/// holds every cell such a side meets.
LO nonconforming_sides(const Mesh<Kokkos::HostSpace>& mesh);

}  // namespace soma::detail

#endif  // SOMA_ALGORITHMS_CONFORMITY_HPP_
