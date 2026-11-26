// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_ALGORITHMS_ELEMENT_MAP_HPP_
#define SOMA_SRC_ALGORITHMS_ELEMENT_MAP_HPP_

// The map of an element from its reference cell (soma/fields/node_tables.hpp)
// to space, and quadrature rules of any degree on the reference cells, for
// soma::sample. Host only.

#include <array>
#include <optional>
#include <vector>

#include "soma/core/mesh_geometry.hpp"
#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"

namespace soma::detail::sample {

using V3 = std::array<Real, 3>;

struct Mapped {
  V3 x{};
  Real measure = 0;  ///< |dx/dxi| (curves), |a_1 x a_2| (surfaces), |det| (volumes)
};

// Linear on simplices, multilinear on quads, hexes and prisms, the usual
// rational map on pyramids; a quad of 8 or 9 nodes (MOAB's order) is
// biquadratic, as SemSpace maps it. On a sphere, a curve or surface is
// projected radially, x = c + R y / |y|, and its measure is the projected
// map's.
class ElementMap {
 public:
  /// Throws std::invalid_argument for a topology or node count it does not map.
  ElementMap(Topology t, std::vector<V3> nodes, const std::optional<MeshGeometry>& surface);

  int reference_dim() const noexcept { return rdim_; }
  Topology topology() const noexcept { return t_; }
  /// x at reference coordinates xi (reference_dim() of them); the measure only
  /// when asked (not at a pyramid's apex).
  Mapped at(const Real* xi, bool measure) const;

 private:
  Topology t_;
  int rdim_ = 0;
  std::vector<V3> nodes_;
  bool sphere_ = false;
  V3 center_{};
  Real radius_ = 0;
};

/// Gauss-Legendre: m points on [-1, 1], ascending, and their weights; exact
/// for polynomials of degree 2m - 1. Throws std::invalid_argument for m < 1.
void gauss_legendre(int m, std::vector<Real>& x, std::vector<Real>& w);

/// A rule on the reference cell of t exact for polynomials of `degree`:
/// tensor Gauss-Legendre on edges, quads and hexes, collapsed (Duffy) rules on
/// triangles, tets and pyramids, their product on prisms. xi holds 3 per
/// point (unused ones 0); the weights add up to the cell's measure. Throws
/// std::invalid_argument for a degree below 0 or above 60, or a topology
/// without a reference cell.
void reference_rule(Topology t, int degree, std::vector<Real>& xi, std::vector<Real>& w);

}  // namespace soma::detail::sample

#endif  // SOMA_SRC_ALGORITHMS_ELEMENT_MAP_HPP_
