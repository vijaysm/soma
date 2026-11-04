// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_CLUSTER_HPP_
#define SOMA_AMR_CLUSTER_HPP_

// Berger–Rigoutsos clustering (M. Berger and I. Rigoutsos, "An algorithm for
// point clustering and grid generation", IEEE Trans. Systems, Man and
// Cybernetics 21(5), 1991). Tagged cells are first coarsened into blocks of
// `block` cells; boxes of blocks are split recursively until each is at
// least `efficiency` full of tagged blocks (or is a single block): at a hole
// in the tag signature closest to the middle, else at the strongest
// inflection of its Laplacian, else at the middle of the longest side. The
// result covers every tag, lies in the domain (a multiple of the block),
// is aligned to the block, and its boxes are disjoint; the same tags give
// the same boxes whatever their order.

#include <vector>

#include "soma/amr/box.hpp"
#include "soma/amr/box_list.hpp"

namespace soma::amr {

struct ClusterOptions {
  double efficiency = 0.7;
  int block = 1;
};

// Throws std::invalid_argument for an efficiency outside (0, 1], a block
// below 1, a domain whose lower corner and lengths are not multiples of the
// block, or a tag outside the domain.
BoxList cluster(const std::vector<IntVect>& tags, const Box& domain, const ClusterOptions& options);

}  // namespace soma::amr

#endif  // SOMA_AMR_CLUSTER_HPP_
