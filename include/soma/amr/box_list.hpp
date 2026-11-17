// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_BOX_LIST_HPP_
#define SOMA_AMR_BOX_LIST_HPP_

/// \file
/// Set operations on lists of boxes of one kind (dimension and index type), on
/// the host. Results are lists of pairwise disjoint boxes; inputs to subtract
/// and complement may overlap.

#include <cstdint>
#include <vector>

#include "soma/amr/box.hpp"

namespace soma::amr {

using BoxList = std::vector<Box>;

/// The points of a not in b, as at most 2 * dim boxes.
BoxList subtract(const Box& a, const Box& b);
/// The points of the boxes of a in none of the boxes of b (disjoint when the
/// boxes of a are).
BoxList subtract(const BoxList& a, const BoxList& b);
/// The points of domain in none of the boxes.
BoxList complement(const Box& domain, const BoxList& boxes);
/// The non-empty intersections of the boxes with b.
BoxList intersect(const BoxList& boxes, const Box& b);
/// Each box refined, coarsened or grown.
BoxList refine(const BoxList& boxes, int ratio);
BoxList coarsen(const BoxList& boxes, int ratio);
BoxList grow(const BoxList& boxes, int n);
/// The points of the list, counted once only when the boxes are disjoint.
std::int64_t num_pts(const BoxList& boxes);
bool disjoint(const BoxList& boxes);
/// Whether every point of b lies in some box of the list.
bool covers(const BoxList& boxes, const Box& b);

}  // namespace soma::amr

#endif  // SOMA_AMR_BOX_LIST_HPP_
