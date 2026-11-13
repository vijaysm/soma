// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_POINT_MATCH_HPP_
#define SOMA_PARALLEL_POINT_MATCH_HPP_

// Global ids of points from their coordinates alone, over the points of every
// rank (soma/core/hash3d_exact.hpp on one rank):
//
//   exact      two points share an id iff their coordinates are bit for bit
//              equal (-0.0 equals +0.0); a point 1 ulp away has another id.
//   tolerance  points share an id iff they are connected by pairs within
//              `tol` (single linkage: A~B~C share an id even if A and C are
//              further apart). A pair is within tol when, with
//              r_j = |a_j - b_j| / tol, max r_j <= 1 (linf), sum r_j <= 1
//              (l1) or sum r_j^2 <= 1 (l2). Points with an infinite
//              coordinate share an id only with exact copies.
//
// Ids are contiguous in [0, count), number the clusters by their least
// member in x-major order (bit patterns compared as ordered keys), and do not
// depend on the number of ranks, on how the points are spread over them, or
// on their order: they equal spatial_exact::assign_ids_exact and
// assign_ids_tol on all points together.
//
// Exact ids are one global sort of 3-word keys (transport::rank_keys). With a
// tolerance, each distinct point goes to the owner of its grid cell (width a
// power of two in (tol, 2 tol]) and, as a halo, to the owners of the
// neighbouring cells it comes within tol of; owners test the pairs of their
// cells, and the components are joined across owners by FastSV rounds
// (hooking to the smaller parent, then shortcutting), whose number grows
// with the log of the longest chain, not its length. Tolerance mode needs strict
// IEEE arithmetic and throws, on every rank, under -ffast-math (see
// hash3d_exact.hpp).

#include <cstdint>
#include <span>
#include <vector>

#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

enum class PointNorm : std::uint8_t { linf, l1, l2 };

// tol == 0: exact matching.
struct PointTolerance {
  Real tol = 0;
  PointNorm norm = PointNorm::l2;
};

struct PointIds {
  std::vector<GO> ids;    // per point given, in [0, count)
  GO count = 0;           // clusters over all ranks
  std::vector<Real> xyz;  // per point, its cluster's least member (x, y, z)
};

// Collective: the ids of the points `xyz` (x, y, z interleaved) this rank
// holds. Throws std::invalid_argument on every rank if any rank gives a NaN
// coordinate or coordinates that are not whole points, or if the ranks give
// different tolerances, a negative, NaN or infinite one; std::runtime_error
// on every rank when a tolerance is given and some rank's floating point is
// not strict IEEE.
[[nodiscard]] PointIds match_points(const Comm& comm, std::span<const Real> xyz, PointTolerance tol = {});

}  // namespace soma

#endif  // SOMA_PARALLEL_POINT_MATCH_HPP_
