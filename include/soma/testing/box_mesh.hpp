// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTING_BOX_MESH_HPP_
#define SOMA_TESTING_BOX_MESH_HPP_

// Structured box meshes with closed-form counts, coordinates and orientation,
// for tests that must not depend on a file or a mesher.
//
//   vertex (i,j,k)  gid 1 + i + (nx+1) * (j + (ny+1) * k), at (lx*i/nx, ly*j/ny, lz*k/nz);
//                   the far faces sit exactly at lx, ly, lz
//   hex             gid 1 + i + nx * (j + ny * k); corners in canonical order:
//                   v0=(i,j,k) v1=(i+1,j,k) v2=(i+1,j+1,k) v3=(i,j+1,k), v4..v7 the same at k+1
//   tet             hex h (0-based) gives tets gid 6h+1..6h+6 (Kuhn's split): tet o walks
//                   from (i,j,k) to (i+1,j+1,k+1) one axis at a time, in the orders xyz, yzx,
//                   zxy, xzy, zyx, yxz; the last three swap their last two corners, so every
//                   tet has positive volume. Neighbouring hexes split alike: faces match.
//   quad            gid 1 + i + nx * j; v0=(i,j) v1=(i+1,j) v2=(i+1,j+1) v3=(i,j+1)
//   tri             quad q (0-based, i fastest) gives triangle gid 2q+1 = (v0,v1,v2) and
//                   gid 2q+2 = (v0,v2,v3): the quad is split along its v0-v2 diagonal
//
// Quads and triangles lie in the plane z = 0 and make a 2D mesh (nz must be 1).

#include <cstdint>

#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"

namespace soma::testing {

struct BoxSpec {
  int nx = 1, ny = 1, nz = 1;
  Topology element = Topology::hex;
  Real lx = 1, ly = 1, lz = 1;
  Ordering ordering = Ordering::file;
};

// Entity counts of a box, computed in 64-bit arithmetic and checked against
// the widths of LO (vertices, elements) and Offset (connectivity entries).
struct BoxCounts {
  std::int64_t vertices = 0;
  std::int64_t elements = 0;
  std::int64_t connectivity = 0;
};

// Throws std::invalid_argument for an unsupported element, a cell count below
// 1, nz != 1 for a planar element, or a length that is not positive and
// finite or whose product with its cell count is not finite (lz is only read
// for hex and tet); std::overflow_error when a count does not fit its type. Allocates
// nothing.
BoxCounts box_counts(const BoxSpec& spec);

// Same exceptions as box_counts, then the builder's. The mesh is built with
// spec.ordering, so under Ordering::file the vertex and element local ids
// follow the gids.
HostMesh box_mesh(const BoxSpec& spec);

}  // namespace soma::testing

#endif  // SOMA_TESTING_BOX_MESH_HPP_
