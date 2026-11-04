// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_HIERARCHY_HPP_
#define SOMA_AMR_HIERARCHY_HPP_

// A hierarchy of levels over one domain: level 0 covers it, and each finer
// level, refined by the ratio from the one below, is a union of disjoint
// boxes aligned to the blocking factor and properly nested in the level
// below (by `nesting` cells of that level, or up to the domain boundary).
// Fields over every level are vectors of LevelFields; the hierarchy keeps
// the halo and coarse-fine plans of its levels and fills a field's ghosts
// level by level, coarse first.
//
//   amr::Hierarchy h(amr::Geometry{amr::Box({0, 0, 0}, {63, 63, 63})}, {.max_level = 2});
//   auto u = h.make_field<double>(amr::IndexType::cell(), 1, 2);
//   h.fill_halo(u);
//   h.average_down(u);

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "soma/amr/box.hpp"
#include "soma/amr/box_array.hpp"
#include "soma/amr/box_list.hpp"
#include "soma/amr/copy.hpp"
#include "soma/amr/halo.hpp"
#include "soma/amr/interlevel.hpp"
#include "soma/amr/level_field.hpp"

namespace soma::amr {

// The level-0 cells and the physical box they cover.
struct Geometry {
  Box domain;
  std::array<double, 3> lo{0.0, 0.0, 0.0};
  std::array<double, 3> hi{1.0, 1.0, 1.0};
  Periodicity periodic{};
};

struct HierarchyOptions {
  int max_level = 0;
  int ratio = 2;
  int blocking_factor = 8;
  int max_box = 32;
  double efficiency = 0.7;
  int nesting = 1;
  Curve curve = Curve::morton;
};

namespace detail {

// The parts of `boxes` in the domain, those beyond a periodic side moved
// back into it.
BoxList wrap(const BoxList& boxes, const Box& domain, Periodicity periodic);
// The coarse cells a properly nested level below fine boxes `fine` must
// cover: their coarsening grown by `nesting`, wrapped into the domain.
BoxList nest_region(const BoxList& fine, int ratio, int nesting, const Box& coarse_domain, Periodicity periodic);

}  // namespace detail

// One LevelField per level.
template <class T>
using MultiLevelField = std::vector<LevelField<T>>;

class Hierarchy {
 public:
  // physical(level, box, region, dir, side): ghost points outside the domain.
  using Physical = std::function<void(int level, int box, const Box& region, int dir, int side)>;

  // Level 0 only, the domain chopped to max_box. With `fine`, levels 1 to
  // fine.size() take those boxes (each level's cells). Throws
  // std::invalid_argument for invalid options (ratio 2 or 4, a blocking
  // factor that is a multiple of the ratio, max_box a multiple of the
  // blocking factor, a domain of whole blocks) or boxes outside their
  // domain, overlapping, not aligned to the blocking factor or not properly
  // nested.
  Hierarchy(const Geometry& geometry, const HierarchyOptions& options, const std::vector<BoxList>& fine = {});

  const Geometry& geometry() const { return geometry_; }
  const HierarchyOptions& options() const { return options_; }
  int num_levels() const { return static_cast<int>(levels_.size()); }
  int finest_level() const { return num_levels() - 1; }
  const BoxArray& boxes(int level) const { return levels_.at(static_cast<std::size_t>(level)); }
  Box domain(int level) const;
  double dx(int level, int d) const;

  template <class T>
  LevelField<T> make_level_field(int level, IndexType type, int ncomp, int ghosts,
                                 const std::string& label = "amr::LevelField") const {
    return LevelField<T>(boxes(level), type, ncomp, ghosts, label + "." + std::to_string(level));
  }
  template <class T>
  MultiLevelField<T> make_field(IndexType type, int ncomp, int ghosts, const std::string& label = "amr::LevelField") const {
    MultiLevelField<T> out;
    for (int l = 0; l < num_levels(); ++l) out.push_back(make_level_field<T>(l, type, ncomp, ghosts, label));
    return out;
  }

  // Plans built on first use and kept (not safe to call concurrently).
  const HaloPlan& halo_plan(int level, IndexType type, int ghosts) const;
  // Fills the uncovered ghosts of `level` from level - 1.
  const CoarseFinePlan& coarse_fine_plan(int level, IndexType type, int ghosts, Interp interp) const;

  // Level by level from 0: ghosts from the same level, then (above level 0)
  // the uncovered ones from the level below, then those outside the domain
  // through `physical`.
  template <class T>
  void fill_halo(const MultiLevelField<T>& u, Interp interp = Interp::linear, const Physical& physical = {}) const;
  // From the finest level down: each level's valid points to the level
  // below.
  template <class T>
  void average_down(const MultiLevelField<T>& u) const;

 private:
  template <class T>
  void check(const MultiLevelField<T>& u) const;
  template <class T>
  void fill_level(const LevelField<T>& u, const LevelField<T>* coarse, int level, Interp interp,
                  const Physical& physical) const;

  Geometry geometry_;
  HierarchyOptions options_;
  std::vector<BoxArray> levels_;
  mutable std::map<std::tuple<int, int, int>, HaloPlan> halo_plans_;
  mutable std::map<std::tuple<int, int, int, int>, CoarseFinePlan> coarse_fine_plans_;
};

template <class T>
void Hierarchy::check(const MultiLevelField<T>& u) const {
  if (static_cast<int>(u.size()) != num_levels()) {
    throw std::invalid_argument("Hierarchy: the field has " + std::to_string(u.size()) + " levels, the hierarchy " +
                                std::to_string(num_levels()));
  }
  for (std::size_t l = 0; l < u.size(); ++l) {
    if (u[l].boxes().boxes() != levels_[l].boxes() || !(u[l].type() == u[0].type()) || u[l].ncomp() != u[0].ncomp() ||
        u[l].ghosts() != u[0].ghosts()) {
      throw std::invalid_argument("Hierarchy: level " + std::to_string(l) + " of the field does not match");
    }
  }
}

template <class T>
void Hierarchy::fill_level(const LevelField<T>& u, const LevelField<T>* coarse, int level, Interp interp,
                           const Physical& physical) const {
  if (u.ghosts() == 0) return;
  const HaloPlan& halo = halo_plan(level, u.type(), u.ghosts());
  amr::fill_halo(u, halo);
  if (coarse != nullptr) fill_coarse_fine(u, *coarse, coarse_fine_plan(level, u.type(), u.ghosts(), interp));
  if (physical) {
    for (const BoundaryRegion& r : halo.boundary()) physical(level, r.box, r.region, r.dir, r.side);
  }
}

template <class T>
void Hierarchy::fill_halo(const MultiLevelField<T>& u, Interp interp, const Physical& physical) const {
  check(u);
  for (int l = 0; l < num_levels(); ++l) {
    fill_level(u[static_cast<std::size_t>(l)], l > 0 ? &u[static_cast<std::size_t>(l - 1)] : nullptr, l, interp, physical);
  }
}

template <class T>
void Hierarchy::average_down(const MultiLevelField<T>& u) const {
  check(u);
  for (int l = finest_level(); l > 0; --l) {
    amr::average_down(u[static_cast<std::size_t>(l)], u[static_cast<std::size_t>(l - 1)], options_.ratio);
  }
}

}  // namespace soma::amr

#endif  // SOMA_AMR_HIERARCHY_HPP_
