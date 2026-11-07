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
// Over many ranks (the constructor taking a Comm) every rank holds every
// level's boxes, each level cut along the curve by box cost (point count, or
// options.cost), and fields hold the boxes their rank owns. Fills, restriction
// and regrids are collective; a regrid gathers the tagged blocks, so every
// rank makes the same boxes as one process would, and rebalances.
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
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "soma/amr/box.hpp"
#include "soma/amr/box_array.hpp"
#include "soma/amr/box_list.hpp"
#include "soma/amr/copy.hpp"
#include "soma/amr/flux_register.hpp"
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
  // The cost of a box of a level, which load balancing evens out over the
  // ranks; it must give every rank the same value. Empty: its point count.
  std::function<double(int level, const Box& box)> cost;
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

// A byte per cell of a level, nonzero where the level should be refined.
using TagField = LevelField<std::uint8_t>;

// A physical boundary is a callable
//   physical(const LevelField<T>& field, int level, int box, const Box& region, int dir, int side)
// that fills the ghost points `region` of box `box` outside the domain (a
// generic lambda serves fields of every element type). This one leaves them.
struct NoBoundary {
  template <class T>
  void operator()(const LevelField<T>&, int, int, const Box&, int, int) const {}
};

class Hierarchy {
 public:
  // tagger(level, tags): marks the cells of `level` to refine (tags arrive
  // zeroed, without ghosts).
  using Tagger = std::function<void(int level, const TagField& tags)>;

  // Level 0 only, the domain chopped to max_box. With `fine`, levels 1 to
  // fine.size() take those boxes (each level's cells). Throws
  // std::invalid_argument for invalid options (ratio 2 or 4, a blocking
  // factor that is a multiple of the ratio, max_box a multiple of the
  // blocking factor, a domain of whole blocks) or boxes outside their
  // domain, overlapping, not aligned to the blocking factor or not properly
  // nested.
  Hierarchy(const Geometry& geometry, const HierarchyOptions& options, const std::vector<BoxList>& fine = {});
  // The same over the ranks of comm (collective only in what it calls).
  Hierarchy(const Comm& comm, const Geometry& geometry, const HierarchyOptions& options,
            const std::vector<BoxList>& fine = {});

  const Geometry& geometry() const { return geometry_; }
  const HierarchyOptions& options() const { return options_; }
  int num_levels() const { return static_cast<int>(levels_.size()); }
  int finest_level() const { return num_levels() - 1; }
  const BoxArray& boxes(int level) const { return levels_.at(static_cast<std::size_t>(level)); }
  // Who owns each box of a level, this rank, and the communicator (none for
  // a hierarchy on one process).
  const Distribution& distribution(int level) const { return owners_.at(static_cast<std::size_t>(level)); }
  int rank() const { return comm_ ? comm_->rank() : 0; }
  const std::optional<Comm>& comm() const { return comm_; }
  Box domain(int level) const;
  double dx(int level, int d) const;

  template <class T>
  LevelField<T> make_level_field(int level, IndexType type, int ncomp, int ghosts,
                                 const std::string& label = "amr::LevelField") const {
    return LevelField<T>(boxes(level), distribution(level), rank(), type, ncomp, ghosts, label + "." + std::to_string(level));
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
  // Restricts `level` to level - 1.
  const AverageDownPlan& average_down_plan(int level, IndexType type) const;
  // A flux register between `level` and level - 1.
  FluxRegister make_flux_register(int level, int ncomp) const;

  // Level by level from 0: ghosts from the same level, then (above level 0)
  // the uncovered ones from the level below, then those outside the domain
  // through `physical`.
  template <class T, class P = NoBoundary>
  void fill_halo(const MultiLevelField<T>& u, Interp interp = Interp::linear, const P& physical = {}) const;
  // The same split: begin starts every level's same-level fill (work may go
  // between), end() finishes them coarse first with the coarse-fine fill and
  // the boundary. The field and `physical` must outlive the pending fill.
  template <class T, class P = NoBoundary>
  class PendingFill;
  template <class T, class P = NoBoundary>
  PendingFill<T, P> begin_fill_halo(const MultiLevelField<T>& u, Interp interp = Interp::linear,
                                    const P& physical = {}) const;
  // From the finest level down: each level's valid points to the level
  // below.
  template <class T>
  void average_down(const MultiLevelField<T>& u) const;

  // The boxes of new levels 1, 2, ... from the tags of levels 0 to
  // min(finest, max_level - 1), built from the top down: the tagged cells of
  // level l, with the cells under the new level l + 2 grown by the nesting,
  // are clustered on blocks of blocking_factor / ratio cells, refined, and
  // chopped to max_box; empty levels on top are dropped. Level 0 stays.
  std::vector<BoxList> make_grids(const Tagger& tagger) const;
  // Field u of hierarchy `from` on this hierarchy's boxes (same geometry
  // and ratio): level by level from 0, prolonged from the new level below
  // (its halos filled first), overwritten where the old level had valid
  // points, then restricted from the finest level down. Linear and quadratic
  // prolongation need at least one ghost.
  template <class T, class P = NoBoundary>
  MultiLevelField<T> transfer(const Hierarchy& from, const MultiLevelField<T>& u, Interp interp = Interp::linear,
                              const P& physical = {}) const;
  // Tags, makes the new levels and moves every field to them; on an
  // exception the hierarchy and the fields are unchanged.
  template <class P, class... T>
  void regrid(const Tagger& tagger, Interp interp, const P& physical, MultiLevelField<T>&... fields);
  // The same, balancing measured costs: measured[l][b], the cost of box b
  // of this rank's part of level l (levels missing or empty: options.cost).
  // A new box costs the old boxes' costs in proportion to its overlap with
  // them, and options.cost where it overlaps none.
  template <class P, class... T>
  void regrid(const Tagger& tagger, const std::vector<std::vector<double>>& measured, Interp interp, const P& physical,
              MultiLevelField<T>&... fields);

 private:
  // The levels' owners from options.cost, or from measured costs of the
  // levels of `old` (gathered here).
  void distribute(const Hierarchy* old, const std::vector<std::vector<double>>* measured);
  template <class T>
  void check(const MultiLevelField<T>& u) const;
  template <class T, class P>
  void fill_level(const LevelField<T>& u, const LevelField<T>* coarse, int level, Interp interp, const P& physical) const;

  Geometry geometry_;
  HierarchyOptions options_;
  std::optional<Comm> comm_;
  std::vector<BoxArray> levels_;
  std::vector<Distribution> owners_;
  mutable std::map<std::tuple<int, int, int>, HaloPlan> halo_plans_;
  mutable std::map<std::tuple<int, int, int, int>, CoarseFinePlan> coarse_fine_plans_;
  mutable std::map<std::tuple<int, int>, AverageDownPlan> average_down_plans_;
};

template <class T, class P>
class Hierarchy::PendingFill {
 public:
  void end() {
    if (h_ == nullptr) return;
    const Hierarchy& h = *h_;
    h_ = nullptr;
    for (int l = 0; l < h.num_levels(); ++l) {
      const auto& f = (*u_)[static_cast<std::size_t>(l)];
      levels_[static_cast<std::size_t>(l)].end();
      if (f.ghosts() == 0) continue;
      if (l > 0) fill_coarse_fine(f, (*u_)[static_cast<std::size_t>(l - 1)], h.coarse_fine_plan(l, f.type(), f.ghosts(), interp_));
      for (const BoundaryRegion& r : h.halo_plan(l, f.type(), f.ghosts()).boundary()) (*physical_)(f, l, r.box, r.region, r.dir, r.side);
    }
  }

 private:
  friend class Hierarchy;
  const Hierarchy* h_ = nullptr;
  const MultiLevelField<T>* u_ = nullptr;
  const P* physical_ = nullptr;
  Interp interp_ = Interp::linear;
  std::vector<PendingHalo<T>> levels_;
};

template <class T>
void Hierarchy::check(const MultiLevelField<T>& u) const {
  if (static_cast<int>(u.size()) != num_levels()) {
    throw std::invalid_argument("Hierarchy: the field has " + std::to_string(u.size()) + " levels, the hierarchy " +
                                std::to_string(num_levels()));
  }
  for (std::size_t l = 0; l < u.size(); ++l) {
    if (u[l].boxes().boxes() != levels_[l].boxes() || !(u[l].type() == u[0].type()) || u[l].ncomp() != u[0].ncomp() ||
        u[l].ghosts() != u[0].ghosts() || u[l].distribution().owner != owners_[l].owner || u[l].rank() != rank()) {
      throw std::invalid_argument("Hierarchy: level " + std::to_string(l) + " of the field does not match");
    }
  }
}

template <class T, class P>
void Hierarchy::fill_level(const LevelField<T>& u, const LevelField<T>* coarse, int level, Interp interp,
                           const P& physical) const {
  if (u.ghosts() == 0) return;
  const HaloPlan& halo = halo_plan(level, u.type(), u.ghosts());
  amr::fill_halo(u, halo);
  if (coarse != nullptr) fill_coarse_fine(u, *coarse, coarse_fine_plan(level, u.type(), u.ghosts(), interp));
  for (const BoundaryRegion& r : halo.boundary()) physical(u, level, r.box, r.region, r.dir, r.side);
}

template <class T, class P>
void Hierarchy::fill_halo(const MultiLevelField<T>& u, Interp interp, const P& physical) const {
  begin_fill_halo(u, interp, physical).end();
}

template <class T, class P>
Hierarchy::PendingFill<T, P> Hierarchy::begin_fill_halo(const MultiLevelField<T>& u, Interp interp,
                                                        const P& physical) const {
  check(u);
  PendingFill<T, P> out;
  out.h_ = this;
  out.u_ = &u;
  out.physical_ = &physical;
  out.interp_ = interp;
  for (int l = 0; l < num_levels(); ++l) {
    const auto& f = u[static_cast<std::size_t>(l)];
    out.levels_.push_back(f.ghosts() == 0 ? PendingHalo<T>{} : amr::begin_fill_halo(f, halo_plan(l, f.type(), f.ghosts())));
  }
  return out;
}

template <class T>
void Hierarchy::average_down(const MultiLevelField<T>& u) const {
  check(u);
  for (int l = finest_level(); l > 0; --l) {
    amr::average_down(u[static_cast<std::size_t>(l)], u[static_cast<std::size_t>(l - 1)],
                      average_down_plan(l, u[static_cast<std::size_t>(l)].type()));
  }
}

template <class T, class P>
MultiLevelField<T> Hierarchy::transfer(const Hierarchy& from, const MultiLevelField<T>& u, Interp interp,
                                       const P& physical) const {
  from.check(u);
  if (!(from.geometry_.domain == geometry_.domain) || from.options_.ratio != options_.ratio) {
    throw std::invalid_argument("Hierarchy::transfer: the hierarchies differ in domain or ratio");
  }
  const IndexType type = u[0].type();
  const int ncomp = u[0].ncomp(), ghosts = u[0].ghosts();
  if (interp != Interp::piecewise_constant && ghosts < 1) {
    throw std::invalid_argument("Hierarchy::transfer: linear and quadratic prolongation need a field with ghosts");
  }
  if (comm_.has_value() != from.comm_.has_value() || rank() != from.rank()) {
    throw std::invalid_argument("Hierarchy::transfer: the hierarchies are not over the same ranks");
  }
  MultiLevelField<T> out = make_field<T>(type, ncomp, ghosts, "amr::Hierarchy.transfer");
  for (int l = 0; l < num_levels(); ++l) {
    const auto& f = out[static_cast<std::size_t>(l)];
    const BoxArray& level = boxes(l);
    std::vector<BoxList> valid(level.size());
    for (std::size_t b = 0; b < level.size(); ++b) valid[b] = {level[b].convert(type)};
    const auto lu = static_cast<std::size_t>(l);
    if (l > 0) {
      fill_level(out[lu - 1], l > 1 ? &out[lu - 2] : nullptr, l - 1, interp, physical);
      const int cg = ghosts < 1 ? ghosts : 1;
      const CoarseFinePlan plan =
          comm_ ? CoarseFinePlan(*comm_, level, owners_[lu], valid, boxes(l - 1), owners_[lu - 1], type, 0, options_.ratio,
                                 geometry_.periodic, interp, cg)
                : CoarseFinePlan(level, valid, boxes(l - 1), type, 0, options_.ratio, geometry_.periodic, interp, cg);
      fill_coarse_fine(f, out[lu - 1], plan);
    }
    if (l < from.num_levels()) {
      const CopyPlan plan = comm_ ? CopyPlan(*comm_, from.boxes(l), from.owners_[lu], valid, owners_[lu], type, geometry_.periodic)
                                  : CopyPlan(from.boxes(l), valid, type, geometry_.periodic);
      copy(u[lu], f, plan);
    }
  }
  average_down(out);
  return out;
}

template <class P, class... T>
void Hierarchy::regrid(const Tagger& tagger, Interp interp, const P& physical, MultiLevelField<T>&... fields) {
  regrid(tagger, {}, interp, physical, fields...);
}

template <class P, class... T>
void Hierarchy::regrid(const Tagger& tagger, const std::vector<std::vector<double>>& measured, Interp interp,
                       const P& physical, MultiLevelField<T>&... fields) {
  Hierarchy next(geometry_, options_, make_grids(tagger));
  next.comm_ = comm_;
  next.distribute(this, &measured);
  next.halo_plans_.clear();
  next.coarse_fine_plans_.clear();
  next.average_down_plans_.clear();
  auto moved = std::make_tuple(next.transfer(*this, fields, interp, physical)...);
  std::tie(fields...) = std::move(moved);
  *this = std::move(next);
}

}  // namespace soma::amr

#endif  // SOMA_AMR_HIERARCHY_HPP_
