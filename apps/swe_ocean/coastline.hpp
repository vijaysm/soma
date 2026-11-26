// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_COASTLINE_HPP_
#define SWE_OCEAN_COASTLINE_HPP_

// The coastline the mesh is refined toward, and the graded refinement's target "is this cell within g of it".
//
// coastline: the points (unit sphere) where the heights z of a latitude-longitude grid (ETOPO1: m) cross 0
// along the grid's edges. An edge joins two neighbouring samples of a row (and, on a periodic grid, the last
// column to the first) or of a column (rows i and i + 1); it crosses when one end has z <= 0 and the other
// z > 0, at the point linearly interpolated between them in latitude and longitude. Rank r of P scans the rows
// i (ascending latitude) from r N / P to (r + 1) N / P - 1 of the N, reading one row more for the edges to
// the next row, in strips of strip_rows rows, every longitude; it owns the edges from its rows (the row edges
// of row i and the column edges from i to i + 1), so each edge is seen once. The points are then decimated
// to one per bin of h = spacing_km / 2 (on the Earth): latitude bins of h, and in each the circle of latitude
// cut into the least number of equal longitude bins at most h / cos(c) wide (c the bin's central latitude;
// one bin at the poles); a bin keeps the point first in (row, column, edge) order, the row edge before the
// column edge. Every rank gets every kept point, in that order: the list is the same, bit for bit, on any
// number of ranks. Collective on comm. Throws std::invalid_argument for spacing_km not finite and positive or
// strip_rows 0, what open_grid throws, and std::runtime_error on every rank when a read fails on any (with
// read_window's message on the rank it failed on).
//
// coast_target: the target of soma::refine_graded for a mesh on the unit sphere, grow in km on the Earth
// (GradedOptions::length giving km): owned cell e is marked when grow[e] >= 0 and a point lies inside the
// cell's polygon (its corners joined by great-circle arcs) or within grow[e] of one of its sides (the
// great-circle distance to the arc). The points go into a soma::GridIndex once; a cell asks it for the points
// within its circumradius about its centroid plus grow[e], and tests those exactly. A cell's answer depends
// only on its corners, grow[e] and the points: the same wherever the cell lies. Not collective; host only.
// Throws std::invalid_argument, when called, unless grow has one value per owned cell.

#include <array>
#include <cstddef>
#include <vector>

#include "grid_file.hpp"
#include "soma/algorithms/refine_graded.hpp"
#include "soma/parallel/comm.hpp"

namespace ocean {

std::vector<std::array<double, 3>> coastline(const soma::Comm& comm, const GridSpec& spec, double spacing_km,
                                             std::size_t strip_rows = 540);

soma::GradedTarget coast_target(std::vector<std::array<double, 3>> points);

}  // namespace ocean

#endif  // SWE_OCEAN_COASTLINE_HPP_
