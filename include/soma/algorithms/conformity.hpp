// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_CONFORMITY_HPP_
#define SOMA_ALGORITHMS_CONFORMITY_HPP_

/// \file
/// Sides of a mesh that tile without matching: FvSpace makes each tile a
/// face; FemSpace constrains hanging vertices and makes a hex whose faces two
/// triangles tile a composite element.

#include <cstdint>
#include <vector>

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

/// The two kinds apart, as nonconforming_sides counts them, and the vertices
/// found strictly inside one-sided sides' edges (hanging), ascending.
struct Nonconforming {
  LO hanging = 0;
  LO split_quads = 0;
  std::vector<LO> vertices;
};
Nonconforming nonconforming(const Mesh<Kokkos::HostSpace>& mesh);

/// Per cell, the sides of a hex that two triangles of other cells tile, as
/// fem::stiffness(c, split, k) takes them: 2 bits a side in canonical order,
/// 0 whole, 1 split by the diagonal through the side's corners 0 and 2, 2 by
/// that through 1 and 3; 0 for other cells. `odd` counts the one-sided quads
/// of owned cells with a one-sided triangle on three of their corners that
/// is not one of two such halves. Not collective: a ghost's code is this
/// rank's view, its owner's taken by a halo update.
struct SplitSides {
  std::vector<std::uint16_t> code;
  LO odd = 0;
};
SplitSides split_sides(const Mesh<Kokkos::HostSpace>& mesh);

/// Per vertex, 1 on the mesh's skin: a corner of a one-sided side (one cell
/// has it) that neither lies in another one-sided side nor has one in it, so
/// that sides between coarse and fine cells are not taken for boundary. A
/// fine side touching none of its coarse side's corners (the middle of a
/// triangle split in four) is not found, but its corners all hang, and a
/// hanging vertex is no boundary vertex.
std::vector<std::uint8_t> conforming_skin(const Mesh<Kokkos::HostSpace>& mesh);

}  // namespace soma::detail

#endif  // SOMA_ALGORITHMS_CONFORMITY_HPP_
