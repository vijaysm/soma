// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/halo.hpp"

#include <stdexcept>

#include "soma/amr/box_list.hpp"

namespace soma::amr {

HaloPlan::HaloPlan(const BoxArray& boxes, IndexType type, int ghosts, Periodicity periodic) : ghosts_(ghosts) {
  if (ghosts < 0) throw std::invalid_argument("HaloPlan: ghosts must not be negative");
  const Box& domain = boxes.domain();
  const int dim = domain.dim();
  const Box domain_t = domain.convert(type);
  // The domain and its periodic images, as far as any ghost reaches.
  Box inside = domain_t;
  for (int d = 0; d < dim; ++d) {
    if (periodic[d]) inside = inside.grow(d, ghosts + domain.length(d));
  }
  std::vector<BoxList> ghost(boxes.size());
  for (std::size_t b = 0; b < boxes.size(); ++b) {
    const Box valid = boxes[b].convert(type);
    const Box grown = valid.grow(ghosts);
    ghost[b] = intersect(subtract(grown, valid), inside);
    // Outside the domain in a direction that is not periodic.
    for (int d = 0; d < dim; ++d) {
      if (periodic[d]) continue;
      if (grown.lo(d) < domain_t.lo(d)) boundary_.push_back({static_cast<int>(b), grown.grow(d, 0, domain_t.lo(d) - 1 - grown.hi(d)), d, 0});
      if (grown.hi(d) > domain_t.hi(d)) boundary_.push_back({static_cast<int>(b), grown.grow(d, grown.lo(d) - (domain_t.hi(d) + 1), 0), d, 1});
    }
  }
  copy_ = CopyPlan(boxes, ghost, type, periodic);
  uncovered_ = copy_.uncovered();
}

}  // namespace soma::amr
