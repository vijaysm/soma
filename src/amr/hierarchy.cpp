// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/hierarchy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "soma/amr/cluster.hpp"

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

Hierarchy::Hierarchy(const Comm& comm, const Geometry& geometry, const HierarchyOptions& options,
                     const std::vector<BoxList>& fine)
    : Hierarchy(geometry, options, fine) {
  comm_ = comm;
  distribute(nullptr, nullptr);
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
  distribute(nullptr, nullptr);
}

void Hierarchy::distribute(const Hierarchy* old, const std::vector<std::vector<double>>* measured) {
  const int nranks = comm_ ? comm_->size() : 1;
  owners_.clear();
  for (int l = 0; l < num_levels(); ++l) {
    const BoxArray& level = boxes(l);
    std::vector<double> w(level.size());
    for (std::size_t b = 0; b < level.size(); ++b) {
      w[b] = options_.cost ? options_.cost(l, level[b]) : static_cast<double>(level[b].num_pts());
    }
    const auto lu = static_cast<std::size_t>(l);
    // Measured costs of the old level's boxes, gathered with their ids,
    // spread over the new boxes by overlap. Every rank calls the gather for
    // the levels both hierarchies have, so the collectives match.
    if (old != nullptr && measured != nullptr && !measured->empty() && l < old->num_levels()) {
      const BoxArray& before = old->boxes(l);
      const std::vector<int> mine = old->distribution(l).owned(old->rank());
      struct Cost {
        std::int64_t id;
        double cost;
      };
      std::vector<Cost> given;
      if (lu < measured->size() && !(*measured)[lu].empty()) {
        // A wrong count travels as an id of -1, so every rank throws.
        if ((*measured)[lu].size() != mine.size()) {
          given.push_back({-1, 0.0});
        } else {
          for (std::size_t i = 0; i < mine.size(); ++i) given.push_back({mine[i], (*measured)[lu][i]});
        }
      }
      const auto all = comm_ ? comm_->allgatherv(std::span<const Cost>(given)) : given;
      std::vector<double> old_cost(before.size(), -1.0);
      for (const auto& [id, cost] : all) {
        if (id < 0) {
          throw std::invalid_argument("Hierarchy::regrid: level " + std::to_string(l) +
                                      " needs one measured cost per box of the rank on some rank");
        }
        if (!std::isfinite(cost) || cost < 0) throw std::invalid_argument("Hierarchy::regrid: measured costs must be finite and not negative");
        old_cost[static_cast<std::size_t>(id)] = cost;
      }
      for (std::size_t b = 0; b < level.size(); ++b) {
        double sum = 0;
        bool any = false;
        for (std::size_t o = 0; o < before.size(); ++o) {
          if (old_cost[o] < 0) continue;
          const Box overlap = level[b] & before[o];
          if (overlap.empty()) continue;
          any = true;
          sum += old_cost[o] * static_cast<double>(overlap.num_pts()) / static_cast<double>(before[o].num_pts());
        }
        if (any) w[b] = sum;
      }
    }
    owners_.push_back(Distribution::along_curve(level, nranks, w));
  }
}

FluxRegister Hierarchy::make_flux_register(int level, int ncomp) const {
  if (level < 1) throw std::invalid_argument("Hierarchy::make_flux_register: level 0 has no level below");
  const auto l = static_cast<std::size_t>(level);
  return comm_ ? FluxRegister(*comm_, boxes(level), owners_[l], boxes(level - 1), owners_[l - 1], options_.ratio,
                              geometry_.periodic, ncomp)
               : FluxRegister(boxes(level), boxes(level - 1), options_.ratio, geometry_.periodic, ncomp);
}

const AverageDownPlan& Hierarchy::average_down_plan(int level, IndexType type) const {
  if (level < 1) throw std::invalid_argument("Hierarchy::average_down_plan: level 0 has no level below");
  const auto key = std::make_tuple(level, static_cast<int>(type.nodal));
  auto it = average_down_plans_.find(key);
  if (it == average_down_plans_.end()) {
    const auto l = static_cast<std::size_t>(level);
    it = average_down_plans_
             .emplace(key, comm_ ? AverageDownPlan(*comm_, boxes(level), owners_[l], boxes(level - 1), owners_[l - 1], type,
                                                   options_.ratio)
                                 : AverageDownPlan(boxes(level), boxes(level - 1), type, options_.ratio))
             .first;
  }
  return it->second;
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
  if (it == halo_plans_.end()) {
    it = halo_plans_
             .emplace(key, comm_ ? HaloPlan(*comm_, boxes(level), distribution(level), type, ghosts, geometry_.periodic)
                                 : HaloPlan(boxes(level), type, ghosts, geometry_.periodic))
             .first;
  }
  return it->second;
}

const CoarseFinePlan& Hierarchy::coarse_fine_plan(int level, IndexType type, int ghosts, Interp interp) const {
  if (level < 1) throw std::invalid_argument("Hierarchy::coarse_fine_plan: level 0 has no level below");
  const auto key = std::make_tuple(level, static_cast<int>(type.nodal), ghosts, static_cast<int>(interp));
  auto it = coarse_fine_plans_.find(key);
  if (it == coarse_fine_plans_.end()) {
    const auto l = static_cast<std::size_t>(level);
    it = coarse_fine_plans_
             .emplace(key, comm_ ? CoarseFinePlan(*comm_, boxes(level), owners_[l], boxes(level - 1), owners_[l - 1], type,
                                                  ghosts, options_.ratio, geometry_.periodic, interp, 1)
                                 : CoarseFinePlan(boxes(level), boxes(level - 1), type, ghosts, options_.ratio,
                                                  geometry_.periodic, interp, 1))
             .first;
  }
  return it->second;
}

std::vector<BoxList> Hierarchy::make_grids(const Tagger& tagger) const {
  const HierarchyOptions& o = options_;
  const int top = std::min(finest_level(), o.max_level - 1);
  std::vector<BoxList> out(static_cast<std::size_t>(top + 1));
  for (int l = top; l >= 0; --l) {
    // This rank's tagged blocks (clustering sees no more than the blocks),
    // gathered from every rank; a tagger failing on any rank fails them all
    // before the gather.
    const int block = o.blocking_factor / o.ratio;
    std::vector<IntVect> tags;
    std::exception_ptr failed;
    try {
      const TagField f = make_level_field<std::uint8_t>(l, IndexType::cell(), 1, 0, "amr::Hierarchy.tags");
      tagger(l, f);
      const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f.view().data);
      for (std::size_t b = 0; b < f.num_boxes(); ++b) {
        const Box& v = f.valid(b);
        for (std::int64_t q = 0; q < v.num_pts(); ++q) {
          if (h(static_cast<std::size_t>(f.offset(b) + q)) == 0) continue;
          IntVect t = detail::point_of(v, q);
          for (int d = 0; d < v.dim(); ++d) t[d] = floor_div(t[d], block) * block;
          tags.push_back(t);
        }
      }
      std::sort(tags.begin(), tags.end(), [](const IntVect& a, const IntVect& b) {
        return std::lexicographical_compare(a.v, a.v + 3, b.v, b.v + 3);
      });
      tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    } catch (...) {
      failed = std::current_exception();
    }
    if (comm_) {
      if (comm_->allreduce(static_cast<std::int32_t>(failed ? 1 : 0), Op::max) != 0) {
        if (failed) std::rethrow_exception(failed);
        throw std::runtime_error("Hierarchy::regrid: the tagger failed on another rank");
      }
      tags = comm_->allgatherv(std::span<const IntVect>(tags));
    } else if (failed) {
      std::rethrow_exception(failed);
    }
    // The new level l + 2 must nest in the new level l + 1.
    if (l < top) {
      const BoxList& above = out[static_cast<std::size_t>(l + 1)];
      for (const Box& c : coarsen(detail::nest_region(above, o.ratio, o.nesting, domain(l + 1), geometry_.periodic), o.ratio)) {
        for (std::int64_t q = 0; q < c.num_pts(); ++q) tags.push_back(detail::point_of(c, q));
      }
    }
    const BoxList blocks = cluster(tags, domain(l), {o.efficiency, o.blocking_factor / o.ratio});
    out[static_cast<std::size_t>(l)] = chop(refine(blocks, o.ratio), o.max_box, o.blocking_factor);
  }
  while (!out.empty() && out.back().empty()) out.pop_back();
  return out;
}

}  // namespace soma::amr
