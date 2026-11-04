// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/hierarchy.hpp"

#include <stdexcept>

namespace soma::amr {

namespace {

std::string text(const Box& b) {
  std::string s = "[";
  for (int d = 0; d < b.dim(); ++d) s += (d ? "," : "") + std::to_string(b.lo(d));
  s += "]-[";
  for (int d = 0; d < b.dim(); ++d) s += (d ? "," : "") + std::to_string(b.hi(d));
  return s + "]";
}

bool aligned(const Box& b, int k) {
  for (int d = 0; d < b.dim(); ++d) {
    if (floor_div(b.lo(d), k) * k != b.lo(d) || b.length(d) % k != 0) return false;
  }
  return true;
}

}  // namespace

BoxList detail::wrap(const BoxList& boxes, const Box& domain, Periodicity periodic) {
  BoxList out;
  for (const Box& b : boxes) {
    for (int sz = -1; sz <= 1; ++sz) {
      for (int sy = -1; sy <= 1; ++sy) {
        for (int sx = -1; sx <= 1; ++sx) {
          const int s[3] = {sx, sy, sz};
          IntVect shift;
          bool skip = false;
          for (int d = 0; d < 3; ++d) {
            skip = skip || (s[d] != 0 && (d >= domain.dim() || !periodic[d]));
            shift[d] = s[d] * (d < domain.dim() ? domain.length(d) : 0);
          }
          if (skip) continue;
          const Box part = b.shift(shift) & domain;
          if (!part.empty()) out.push_back(part);
        }
      }
    }
  }
  return out;
}

BoxList detail::nest_region(const BoxList& fine, int ratio, int nesting, const Box& coarse_domain, Periodicity periodic) {
  return wrap(grow(coarsen(fine, ratio), nesting), coarse_domain, periodic);
}

Hierarchy::Hierarchy(const Geometry& geometry, const HierarchyOptions& options, const std::vector<BoxList>& fine)
    : geometry_(geometry), options_(options) {
  const HierarchyOptions& o = options_;
  const Box& domain = geometry_.domain;
  if (o.ratio != 2 && o.ratio != 4) throw std::invalid_argument("Hierarchy: the ratio must be 2 or 4");
  if (o.blocking_factor < o.ratio || o.blocking_factor % o.ratio != 0) {
    throw std::invalid_argument("Hierarchy: the blocking factor must be a multiple of the ratio");
  }
  if (o.max_box < o.blocking_factor || o.max_box % o.blocking_factor != 0) {
    throw std::invalid_argument("Hierarchy: max_box must be a multiple of the blocking factor");
  }
  if (o.max_level < 0 || o.nesting < 0 || !(o.efficiency > 0 && o.efficiency <= 1)) {
    throw std::invalid_argument("Hierarchy: max_level and nesting must not be negative, the efficiency in (0, 1]");
  }
  if (domain.empty() || !(domain.type() == IndexType::cell()) || !aligned(domain, o.blocking_factor)) {
    throw std::invalid_argument("Hierarchy: the domain " + text(domain) + " is not a box of whole blocks of cells");
  }
  if (static_cast<int>(fine.size()) > o.max_level) throw std::invalid_argument("Hierarchy: more levels than max_level");
  levels_.emplace_back(chop({domain}, o.max_box, o.blocking_factor), domain, o.curve);
  for (std::size_t i = 0; i < fine.size(); ++i) {
    const int l = static_cast<int>(i) + 1;
    if (fine[i].empty()) {
      if (i + 1 == fine.size()) break;
      throw std::invalid_argument("Hierarchy: level " + std::to_string(l) + " is empty below a finer level");
    }
    for (const Box& b : fine[i]) {
      if (!aligned(b, o.blocking_factor)) {
        throw std::invalid_argument("Hierarchy: box " + text(b) + " of level " + std::to_string(l) +
                                    " is not aligned to the blocking factor");
      }
    }
    BoxArray boxes(fine[i], this->domain(l), o.curve);
    const BoxList need = detail::nest_region(fine[i], o.ratio, o.nesting, this->domain(l - 1), geometry_.periodic);
    const BoxList missing = subtract(need, levels_.back().boxes());
    if (!missing.empty()) {
      throw std::invalid_argument("Hierarchy: level " + std::to_string(l) + " is not properly nested in level " +
                                  std::to_string(l - 1) + " near " + text(missing.front()));
    }
    levels_.push_back(std::move(boxes));
  }
}

Box Hierarchy::domain(int level) const {
  Box d = geometry_.domain;
  for (int l = 0; l < level; ++l) d = d.refine(options_.ratio);
  return d;
}

double Hierarchy::dx(int level, int d) const {
  return (geometry_.hi[static_cast<std::size_t>(d)] - geometry_.lo[static_cast<std::size_t>(d)]) / domain(level).length(d);
}

const HaloPlan& Hierarchy::halo_plan(int level, IndexType type, int ghosts) const {
  const auto key = std::make_tuple(level, static_cast<int>(type.nodal), ghosts);
  auto it = halo_plans_.find(key);
  if (it == halo_plans_.end()) it = halo_plans_.emplace(key, HaloPlan(boxes(level), type, ghosts, geometry_.periodic)).first;
  return it->second;
}

const CoarseFinePlan& Hierarchy::coarse_fine_plan(int level, IndexType type, int ghosts, Interp interp) const {
  if (level < 1) throw std::invalid_argument("Hierarchy::coarse_fine_plan: level 0 has no level below");
  const auto key = std::make_tuple(level, static_cast<int>(type.nodal), ghosts, static_cast<int>(interp));
  auto it = coarse_fine_plans_.find(key);
  if (it == coarse_fine_plans_.end()) {
    it = coarse_fine_plans_
             .emplace(key, CoarseFinePlan(boxes(level), boxes(level - 1), type, ghosts, options_.ratio, geometry_.periodic,
                                          interp, 1))
             .first;
  }
  return it->second;
}

}  // namespace soma::amr
