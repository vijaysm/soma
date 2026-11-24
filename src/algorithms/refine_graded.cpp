// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Graded refinement (refine_graded): passes of refine_quads toward a target,
// each level banded before the next coarser, rerun where a band is thin.

#include "soma/algorithms/refine_graded.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/refine.hpp"
#include "soma/core/view_types.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/parallel/migrate.hpp"

namespace soma {

namespace {

using P3 = std::array<Real, 3>;

P3 sub(const P3& a, const P3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
P3 cross(const P3& a, const P3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Real dot(const P3& a, const P3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Real norm(const P3& a) { return std::sqrt(dot(a, a)); }

std::vector<P3> corners(const HostMesh& m, LO e) {
  const auto mv = m.view();
  const auto x = m.coords();
  std::vector<P3> p;
  for (LO i = 0; i < mv.num_vertices(2, e); ++i) {
    const LO v = mv.vertex(2, e, i);
    p.push_back({x(v, 0), x(v, 1), x(v, 2)});
  }
  return p;
}

// The area of a cell's corners' polygon: twice the area vector about the
// first corner, halved.
Real area(const std::vector<P3>& p) {
  P3 n{0, 0, 0};
  for (std::size_t i = 1; i + 1 < p.size(); ++i) {
    const P3 t = cross(sub(p[i], p[0]), sub(p[i + 1], p[0]));
    for (std::size_t a = 0; a < 3; ++a) n[a] += t[a];
  }
  return norm(n) / 2;
}

Real in_units(const std::function<Real(Real)>& length, Real s) { return length ? length(s) : s; }

// The passes of one run, from `mesh` (refined in place); `widen` per level.
GradedPasses passes(const Comm& comm, HostMesh& mesh, const GradedOptions& o, const GradedTarget& target,
                    const GradedHook& hook, int run, const std::vector<Real>& widen) {
  GradedPasses p;
  // The marks of the owned cells, and their count over every rank.
  const auto mark = [&](std::vector<std::int8_t>& marks) {
    const auto level = refine_levels(mesh);
    const LO owned = mesh.segment_table(2).owned_end;
    std::vector<Real> grow(static_cast<std::size_t>(owned), -1);
    for (LO e = 0; e < owned; ++e) {
      const Real size = in_units(o.length, std::sqrt(area(corners(mesh, e))));
      int k = 0;
      for (Real t = size / (o.np - 1); t > 1.5 * o.spacing; t /= 3) ++k;
      if (k == 0) continue;
      Real g = 0;
      const int cells = o.grow < 0 ? o.transition : o.grow;
      for (int i = 1; i < k; ++i) g += cells * size / std::pow(Real{3}, i);
      for (std::size_t j = static_cast<std::size_t>(level[static_cast<std::size_t>(e)]) + 1; j < widen.size(); ++j) {
        g += widen[j];
      }
      grow[static_cast<std::size_t>(e)] = g;
    }
    const std::vector<std::uint8_t> in = target(mesh, grow);
    const Real bad = in.size() != grow.size() ? 1 : 0;
    if (comm.allreduce(bad, Op::max) > 0) {
      throw std::invalid_argument("refine_graded: the target answered " + std::to_string(in.size()) + " cells of " +
                                  std::to_string(grow.size()) + " on a rank");
    }
    marks.clear();
    for (std::size_t e = 0; e < in.size(); ++e) marks.push_back(grow[e] >= 0 && in[e] != 0 ? 1 : 0);
    Real count = 0;
    for (const auto k : marks) count += k;
    return comm.allreduce(count, Op::sum);
  };
  std::vector<std::int8_t> marks;
  for (;;) {
    p.marked = mark(marks);
    if (p.marked == 0 || p.passes == o.passes) return p;
    // A level about to be refined: the hook first (a smoothing, so the
    // children copy the smoothed shape), then marked again from its corners.
    if (hook && p.passes > 0) {
      hook(mesh, run, p.passes);
      p.marked = mark(marks);
      if (p.marked == 0) return p;
    }
    // Each marked cell becomes 9: at least 8 more cells (transitions add more).
    const Real cells = comm.allreduce(static_cast<Real>(mesh.segment_table(2).owned_end), Op::sum);
    if (o.max_cells > 0 && cells + 8 * p.marked > o.max_cells) {
      throw std::runtime_error("pass " + std::to_string(p.passes + 1) + " would make more than " +
                               std::to_string(static_cast<long long>(cells + 8 * p.marked)) + " cells (max_cells " +
                               std::to_string(static_cast<long long>(o.max_cells)) + ")");
    }
    p.refined += p.marked;
    const auto t0 = std::chrono::steady_clock::now();
    mesh = rebalance(comm, refine_quads(comm, mesh, marks, {.sphere_radius = 1}).mesh);
    p.seconds.push_back(std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count());
    p.cells.push_back(comm.allreduce(static_cast<Real>(mesh.segment_table(2).owned_end), Op::sum));
    ++p.passes;
  }
}

}  // namespace

std::vector<std::int32_t> refine_levels(const HostMesh& mesh) {
  std::vector<std::int32_t> out(static_cast<std::size_t>(mesh.segment_table(2).owned_end), 0);
  if (!mesh.fields().contains("refine:level")) return out;
  const auto l = mesh.fields().get<std::int32_t>("refine:level", 2).view();
  for (std::size_t e = 0; e < out.size(); ++e) out[e] = l(e, 0);
  return out;
}

std::vector<Real> band_widths(const Comm& comm, HostMesh& m, int cap) {
  const auto own = refine_levels(m);
  std::int32_t finest = 0;
  for (const auto l : own) finest = std::max(finest, l);
  finest = static_cast<std::int32_t>(comm.allreduce(static_cast<Real>(finest), Op::max));
  std::vector<Real> out;
  if (finest < 2) return out;
  if (comm.size() > 1) m.add_ghosts(comm, GhostOptions{1, 0});
  build_distributed_adjacencies(comm, m, AdjacencyRequest{}.add(2, 1).add(1, 2));
  const auto down = m.adjacency(2, 1);
  const auto up = m.adjacency(1, 2);
  const auto n = static_cast<std::size_t>(m.count(2));
  const LO owned = m.segment_table(2).owned_end;
  std::optional<HaloExchange<Kokkos::HostSpace>> halo;
  if (comm.size() > 1) halo.emplace(comm, m, 2);
  const View2<Real, Kokkos::HostSpace> lev("levels", n, 1), d("distance", n, 1);
  for (std::size_t e = 0; e < own.size(); ++e) lev(e, 0) = own[e];
  if (halo) halo->update(comm, lev);
  // The cells sharing an edge with e.
  const auto each_neighbour = [&](std::size_t e, auto&& f) {
    for (auto k = down.offsets(e); k < down.offsets(e + 1); ++k) {
      const auto edge = static_cast<std::size_t>(down.targets(k));
      for (auto j = up.offsets(edge); j < up.offsets(edge + 1); ++j) {
        const auto c = static_cast<std::size_t>(up.targets(j));
        if (c != e) f(c);
      }
    }
  };
  const Real far = cap;
  for (std::int32_t l = 1; l < finest; ++l) {
    Real coarser = 0;
    for (std::size_t e = 0; e < n; ++e) {
      d(e, 0) = lev(e, 0) < l ? 0 : far;
      coarser += e < static_cast<std::size_t>(owned) && lev(e, 0) < l ? 1 : 0;
    }
    for (int sweep = 0; sweep < cap; ++sweep) {
      std::vector<Real> next(n);
      for (std::size_t e = 0; e < n; ++e) next[e] = d(e, 0);
      for (LO e = 0; e < owned; ++e) {
        const auto ei = static_cast<std::size_t>(e);
        if (lev(ei, 0) != l) continue;
        each_neighbour(ei, [&](std::size_t c) { next[ei] = std::min(next[ei], d(c, 0) + 1); });
      }
      for (std::size_t e = 0; e < n; ++e) d(e, 0) = next[e];
      if (halo) halo->update(comm, d);
    }
    Real width = far + 1, meets = 0;
    for (LO e = 0; e < owned; ++e) {
      const auto ei = static_cast<std::size_t>(e);
      if (lev(ei, 0) != l) continue;
      bool finer = false;
      each_neighbour(ei, [&](std::size_t c) { finer = finer || lev(c, 0) > l; });
      if (!finer) continue;
      meets = 1;
      width = std::min(width, d(ei, 0));
    }
    width = comm.allreduce(width, Op::min);
    meets = comm.allreduce(meets, Op::max);
    coarser = comm.allreduce(coarser, Op::sum);
    out.push_back(meets == 0 || coarser == 0 ? -1 : std::min(width, far));
  }
  if (comm.size() > 1) m.remove_ghosts(comm);
  return out;
}

std::vector<Real> level_sizes(const Comm& comm, const HostMesh& m, std::size_t count,
                              const std::function<Real(Real)>& length) {
  std::vector<Real> s(2 * count, 0);
  const auto level = refine_levels(m);
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    const auto l = static_cast<std::size_t>(level[static_cast<std::size_t>(e)]);
    if (l >= count) continue;
    s[2 * l] += in_units(length, std::sqrt(area(corners(m, e))));
    s[2 * l + 1] += 1;
  }
  comm.allreduce(std::span<Real>(s), Op::sum);
  std::vector<Real> out(count, 0);
  for (std::size_t l = 0; l < count; ++l) out[l] = s[2 * l + 1] > 0 ? s[2 * l] / s[2 * l + 1] : 0;
  return out;
}

GradedResult refine_graded(const Comm& comm, const HostMesh& start, const GradedOptions& o,
                           const GradedTarget& target, const GradedHook& before_marking) {
  if (o.np < 2 || o.transition < 1 || o.passes < 0 || o.reruns < 0 || !target) {
    throw std::invalid_argument("refine_graded: np " + std::to_string(o.np) + ", transition " +
                                std::to_string(o.transition) + ", passes " + std::to_string(o.passes) + ", reruns " +
                                std::to_string(o.reruns) + (target ? "" : ", no target") +
                                " (np >= 2, transition >= 1, passes and reruns >= 0 and a target)");
  }
  GradedResult g;
  std::vector<Real> widen;
  for (;;) {
    g.mesh = start.clone();
    g.passes = passes(comm, g.mesh, o, target, before_marking, g.reruns, widen);
    // Bands are measured up to this many cells.
    g.widths = band_widths(comm, g.mesh, 4 * o.transition);
    const auto size = level_sizes(comm, g.mesh, g.widths.size() + 1, o.length);
    widen.resize(std::max(widen.size(), g.widths.size() + 1), 0);
    bool thin = false;
    for (std::size_t i = 0; i < g.widths.size(); ++i) {
      if (g.widths[i] < 0 || g.widths[i] >= o.transition) continue;
      widen[i + 1] += (o.transition - g.widths[i]) * size[i + 1];
      thin = true;
    }
    g.settled = !thin;
    if (!thin || g.reruns == o.reruns) return g;
    ++g.reruns;
  }
}

}  // namespace soma
