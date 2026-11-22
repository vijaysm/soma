// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_ATM_LATLON_HPP_
#define SWE_ATM_LATLON_HPP_

// Spectral-element fields on the cubed sphere of soma::sem::cubed_sphere_mesh
// (geometry order 1), evaluated at any point: the element holding it and its
// reference coordinates, then the element's Lagrange interpolant. A
// latitude-longitude grid is sampled the same, bit for bit, on any number of
// ranks, and written as a NumPy .npy file.

#include <array>
#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

#include "diagnostics.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"

namespace swe {

struct Located {
  soma::GO gid = 0;
  double xi = 0, eta = 0;
};

// The element of the cubed sphere with ne elements along a face's edge that
// holds the direction x (any length), and x's reference coordinates in it under
// the bilinear-then-projected map: equiangular coordinates find the element
// (its edges are equiangular lines), Gauss-Newton the coordinates (at most 20
// iterations, to 1e-14). A point on an element's edge, a cube edge or corner,
// or a pole, goes to the element of smallest gid holding it.
Located locate(int ne, const std::array<double, 3>& x);

// The unit vector at reference coordinates (xi, eta) of element gid: the
// bilinear map of its corners, projected to the sphere.
std::array<double, 3> element_point(int ne, soma::GO gid, double xi, double eta);

// Collective: values' component at each (lat[i], lon[j]) in degrees, as
// nlat x nlon (row i = latitude i), the same on every rank and on any number of
// ranks: the owner of the element holding each point evaluates it there.
// mesh is the space's (a cubed sphere of ne).
std::vector<double> sample(const soma::Comm& comm, const Space& space, const soma::HostMesh& mesh, int ne,
                           const ConstValues& values, int component, std::span<const double> lat,
                           std::span<const double> lon);

// Writes rows x cols doubles (C order) as a NumPy .npy file, format 1.0
// ('<f8'). Not collective: call it on one rank. Throws std::runtime_error when
// the file cannot be written and std::invalid_argument when the sizes disagree.
void write_npy(const std::filesystem::path& path, std::span<const double> data, std::size_t rows, std::size_t cols);

}  // namespace swe

#endif  // SWE_ATM_LATLON_HPP_
