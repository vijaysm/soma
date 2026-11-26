// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_SAMPLE_HPP_
#define SWE_OCEAN_SAMPLE_HPP_

// Fields of the ocean's spectral-element space evaluated at any point, and the volume transport through a meridian
// section (Drake Passage).
//
// Sampler: a point is found in an element by Gauss-Newton on the element's map, taken as the degree-p Lagrange
// interpolant X(xi, eta) of the element's GLL point positions (on the sphere, projected to it: the residual is the
// part of X normal to the point's direction). For the bilinear maps of the plane this is the map itself; on the
// sphere it differs from the space's map by the interpolation error of a smooth map, O(h^(p+1)), metres on
// elements of tens of kilometres. The element holds the point when the iteration converges with |xi|, |eta| <= 1 +
// 1e-9; a point on an edge or corner shared by several elements goes to the one of least gid. The value there is
// the element's Lagrange interpolant of the field at its GLL points (the field the model carries); a point no
// element holds (land, outside the mesh) is NaN. Candidate elements come from a soma::GridIndex of the owned
// elements' boxes (each widened by a quarter of its largest extent, more than the sphere's bulge between GLL
// points). Every rank evaluates its own elements; the owner of the least gid is the one value kept, so a sample is
// the same, bit for bit, on any number of ranks.
//
// meridian_transport: the transport (Sv, 1e6 m^3/s) eastward through the meridian `lon` of the integrand
// D u . e_east (D = H + eta the total depth, e_east the unit eastward vector; formed at the GLL points, then
// interpolated), sampled every `spacing` degrees of latitude from lat_south to lat_north and integrated by the
// trapezoidal rule (ds = a dlat) over the run of consecutive wet samples that contains lat_start: from the coast
// to the south to the coast to the north. A sample is dry where no element holds it (land removed from the mesh)
// or, when land flags are given, where their interpolant is at least 1/2 (land points of kept elements, which
// carry the least depth H_min: an island smaller than an element). The coast is found to within one spacing
// (1.1 km for 0.01 degree; the flags' half-way point lies between a land and an ocean GLL point), where D u is
// small; inside the run the samples are the model's own interpolant, so the rule's error is that of the
// trapezoidal rule on a piecewise polynomial with kinks at element edges, O(spacing^2), negligible at 0.01 degree
// beside elements of tens of kilometres.

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/view_types.hpp"
#include "soma/parallel/comm.hpp"
#include "types.hpp"

namespace ocean {

class Sampler {
 public:
  // Not collective: indexes the owned elements of the space over the mesh (the space's mesh, for the gids). The
  // space and mesh must outlive the sampler.
  Sampler(const Space& space, const soma::HostMesh& mesh);
  ~Sampler();
  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;

  // Collective: values' components (a row per point of the space) at each point (m; on the sphere any point along
  // the direction will do, in the plane z is ignored), points.size() rows of values.extent(1) values, row major,
  // NaN where no element holds the point. Throws std::invalid_argument on every rank when values has not a row
  // per point of the space on some rank.
  std::vector<double> sample(const soma::Comm& comm, ConstValues values,
                             std::span<const std::array<double, 3>> points) const;

 private:
  struct Index;
  const Space* space_;
  const soma::HostMesh* mesh_;
  std::unique_ptr<Index> index_;
};

// The section: the meridian lon (degrees east), sampled from lat_south to lat_north every spacing degrees, the
// wet run around lat_start. Defaults: Drake Passage at 67 W.
struct Section {
  double lon = -67, lat_south = -80, lat_north = -45, lat_start = -60, spacing = 0.01;
};

// sv: the transport (Sv, eastward positive); lat_from, lat_to: the wet run's first and last sample (degrees); wet:
// false when lat_start is dry (no run: sv 0).
struct Transport {
  double sv = 0, lat_from = 0, lat_to = 0;
  bool wet = false;
};

// Collective: the transport above, for a space on the sphere of radius a, depth H (1 component), free surface eta
// (1), velocity u (3, Cartesian) and land flags (1, 1 on land; no rows: none). Throws std::invalid_argument on
// every rank for a space not on the sphere or a section with lat_south > lat_start or lat_start > lat_north, a
// latitude outside [-90, 90] or spacing not positive.
Transport meridian_transport(const soma::Comm& comm, const Sampler& sampler, const Space& space, ConstValues H,
                             ConstValues eta, ConstValues u, ConstValues land, const Section& section = {});

}  // namespace ocean

#endif  // SWE_OCEAN_SAMPLE_HPP_
