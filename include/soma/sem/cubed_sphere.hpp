// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SEM_CUBED_SPHERE_HPP_
#define SOMA_SEM_CUBED_SPHERE_HPP_

/// \file
/// The equiangular cubed sphere, split across the ranks: ne x ne quads on each
/// of the cube's six faces, on the sphere of a given radius, as HOMME's ne
/// grids (ne30 is about 100 km on the Earth).
///
/// ```
///   soma::sem::CubedSphereSpec spec;
///   spec.ne = 30;
///   spec.radius = 6.371e6;
///   soma::HostMesh mesh = soma::sem::cubed_sphere_mesh(comm, spec);   // collective
/// ```
///
/// On cube face f a point of angles (a, b) in [-pi/4, pi/4]^2 is the cube point
/// (tan a, tan b) on that face, projected radially onto the sphere. Elements
/// have gid 1 + f ne^2 + i + ne j (face f in 0..5, i fastest), corners
/// counter-clockwise seen from outside. A vertex's gid comes from its place on
/// the cube's surface lattice, so gids do not depend on the number of ranks;
/// they are unique but not consecutive. Rank r builds the r-th block of
/// consecutive element gids (6 ne^2 / size each, the first ranks one more);
/// ranks past the element count hold none.
///
/// With geometry_order = 2 the elements are Quad9: their mid-edge and centre
/// nodes sit at the parameter midpoints of the equiangular map, on the sphere.

#include "soma/core/mesh.hpp"
#include "soma/core/ordering.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"

namespace soma::sem {

/// A cubed sphere: elements per cube edge, radius, geometry order (1: Quad4,
/// 2: Quad9) and the builder's ordering.
struct CubedSphereSpec {
  int ne = 4;
  Real radius = 1;
  int geometry_order = 1;
  Ordering ordering = default_ordering;
};

/// Collective: this rank's part of the cubed sphere `spec` describes, with its
/// sharing (build_local_part). Throws std::invalid_argument for ne < 1, a
/// radius not positive and finite, or a geometry order other than 1 or 2.
[[nodiscard]] HostMesh cubed_sphere_mesh(const Comm& comm, const CubedSphereSpec& spec);

}  // namespace soma::sem

#endif  // SOMA_SEM_CUBED_SPHERE_HPP_
