// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/quality_smooth.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <vector>

#include "soma/algorithms/vertex_graph.hpp"
#include "soma/core/view_types.hpp"
#include "soma/parallel/halo_exchange.hpp"

namespace soma::detail {

namespace {

// splitmix64: a vertex's priority from its gid alone.
std::uint64_t priority(GO gid) {
  std::uint64_t z = static_cast<std::uint64_t>(gid) + 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

}  // namespace

std::vector<std::int32_t> vertex_colours(const Comm& comm, const HostMesh& mesh, const VertexCells& cells) {
  const int dim = mesh.dim();
  const auto m = mesh.view();
  const auto gid = mesh.gid(0);
  const LO n = mesh.count(0);
  const LO owned = mesh.segment_table(0).owned_end;
  const View2<std::int32_t, Kokkos::HostSpace> colour("soma::vertex_colours", static_cast<std::size_t>(n), 1);
  Kokkos::deep_copy(colour, -1);
  std::optional<HaloExchange<Kokkos::HostSpace>> halo;
  if (comm.size() > 1) halo.emplace(comm, mesh, 0);
  // (priority, gid): a total order on the vertices.
  const auto beats = [&](LO a, LO b) {
    const std::uint64_t pa = priority(gid(a)), pb = priority(gid(b));
    return pa != pb ? pa > pb : gid(a) > gid(b);
  };
  std::vector<LO> winners;
  std::vector<char> taken;
  Real before = -1;
  for (;;) {
    Real left = 0;
    winners.clear();
    for (LO v = 0; v < owned; ++v) {
      if (colour(static_cast<std::size_t>(v), 0) >= 0) continue;
      left += 1;
      bool wins = true;
      const auto i0 = static_cast<std::size_t>(v);
      const Offset first = cells.offsets[i0], last = cells.offsets[i0 + 1];
      for (Offset k = first; wins && k < last; ++k) {
        const LO c = cells.cells[static_cast<std::size_t>(k)];
        for (LO i = 0; i < m.num_vertices(dim, c); ++i) {
          const LO u = m.vertex(dim, c, i);
          if (u != v && colour(static_cast<std::size_t>(u), 0) < 0 && beats(u, v)) {
            wins = false;
            break;
          }
        }
      }
      if (wins) winners.push_back(v);
    }
    const Real all = comm.allreduce(left, Op::sum);
    if (all == 0) break;
    // The vertex of highest priority left always wins, unless ranks disagree.
    if (all == before) throw std::logic_error("soma::detail::vertex_colours: a round coloured no vertex");
    before = all;
    // Winners share no cell, so each picks from colours fixed before the round.
    for (const LO v : winners) {
      taken.assign(taken.size(), 0);
      for (Offset k = cells.offsets[static_cast<std::size_t>(v)]; k < cells.offsets[static_cast<std::size_t>(v) + 1];
           ++k) {
        const LO c = cells.cells[static_cast<std::size_t>(k)];
        for (LO i = 0; i < m.num_vertices(dim, c); ++i) {
          const std::int32_t h = colour(static_cast<std::size_t>(m.vertex(dim, c, i)), 0);
          if (h < 0) continue;
          if (static_cast<std::size_t>(h) >= taken.size()) taken.resize(static_cast<std::size_t>(h) + 1, 0);
          taken[static_cast<std::size_t>(h)] = 1;
        }
      }
      std::int32_t least = 0;
      while (static_cast<std::size_t>(least) < taken.size() && taken[static_cast<std::size_t>(least)] != 0) ++least;
      colour(static_cast<std::size_t>(v), 0) = least;
    }
    if (halo) halo->update(comm, colour);
  }
  return {colour.data(), colour.data() + n};
}

}  // namespace soma::detail

namespace soma {

namespace {

using P3 = std::array<Real, 3>;

P3 sub(const P3& a, const P3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
P3 axpy(Real s, const P3& a, const P3& b) { return {s * a[0] + b[0], s * a[1] + b[1], s * a[2] + b[2]}; }
Real dot(const P3& a, const P3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
P3 cross(const P3& a, const P3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Real norm(const P3& a) { return std::sqrt(dot(a, a)); }
P3 unit(const P3& a) {
  const Real l = norm(a);
  return {a[0] / l, a[1] / l, a[2] / l};
}

constexpr Real kDegrees = 180 / std::numbers::pi;
constexpr Real kInfinity = std::numeric_limits<Real>::infinity();
// Relative fall an objective must make for a step to count: rounding is not progress.
constexpr Real kProgress = 1e-12;
// Central differences, in units of the patch's shortest edge.
constexpr Real kDelta = 1e-4;
// First trial step, in the same units, and the halvings tried after it.
constexpr Real kFirstStep = 0.25;
constexpr int kHalvings = 20;
// Steps per vertex visit.
constexpr int kInner = 3;
// Angles within this many degrees of the least are active.
constexpr Real kActive = 1e-3;

// The patch of a vertex measured with the vertex at some point.
struct Measure {
  bool valid = true;      // every corner's det A > 0
  Real kappa = 0;         // summed over the patch's corners
  Real least = 360;       // degrees, over the patch's corners
  Real least_moved = 360; // degrees, over the corners the vertex moves
  std::vector<Real> moved;  // those corners' angles, in patch order
};

class Smoother {
 public:
  Smoother(const HostMesh& mesh, const detail::VertexCells& cells, std::optional<Real> radius)
      : m_(mesh.view()), x_(mesh.coords()), cells_(cells), radius_(radius) {}

  // The patch's normal for det A and the plane's frame, at the vertex's
  // present position.
  void frame(LO v) {
    v_ = v;
    x0_ = point(v);
    P3 area{0, 0, 0};
    shortest_ = kInfinity;
    for (Offset k = begin(v); k < end(v); ++k) {
      const LO c = cells_.cells[static_cast<std::size_t>(k)];
      const LO n = m_.num_vertices(2, c);
      const P3 o = point(m_.vertex(2, c, 0));
      for (LO i = 1; i + 1 < n; ++i) {
        const P3 t = cross(sub(point(m_.vertex(2, c, i)), o), sub(point(m_.vertex(2, c, i + 1)), o));
        for (int q = 0; q < 3; ++q) area[static_cast<std::size_t>(q)] += t[static_cast<std::size_t>(q)];
      }
      for (LO i = 0; i < n; ++i) {
        if (m_.vertex(2, c, i) != v) continue;
        shortest_ = std::min({shortest_, norm(sub(point(m_.vertex(2, c, (i + 1) % n)), x0_)),
                              norm(sub(point(m_.vertex(2, c, (i + n - 1) % n)), x0_))});
      }
    }
    const P3 plane = radius_ ? unit(x0_) : unit(area);
    normal_ = dot(plane, area) < 0 ? P3{-plane[0], -plane[1], -plane[2]} : plane;
    // t1: the axis least along the normal, made orthogonal; t2 = n x t1.
    std::size_t axis = 0;
    for (std::size_t q = 1; q < 3; ++q) axis = std::abs(plane[q]) < std::abs(plane[axis]) ? q : axis;
    P3 e{0, 0, 0};
    e[axis] = 1;
    t1_ = unit(axpy(-dot(e, plane), plane, e));
    t2_ = cross(plane, t1_);
  }

  // The vertex's position at (a, b) in its plane, in units of the shortest edge.
  P3 at(Real a, Real b) const {
    P3 p = axpy(shortest_ * b, t2_, axpy(shortest_ * a, t1_, x0_));
    if (radius_) {
      const Real s = *radius_ / norm(p);
      p = {p[0] * s, p[1] * s, p[2] * s};
    }
    return p;
  }

  Measure measure(const P3& p) {
    const P3 keep = point(v_);
    put(v_, p);
    Measure out;
    for (Offset k = begin(v_); k < end(v_); ++k) {
      const LO c = cells_.cells[static_cast<std::size_t>(k)];
      const LO n = m_.num_vertices(2, c);
      const bool tri = n == 3;
      LO at_v = 0;
      for (LO i = 0; i < n; ++i) at_v = m_.vertex(2, c, i) == v_ ? i : at_v;
      for (LO i = 0; i < n; ++i) {
        const P3 q = point(m_.vertex(2, c, i));
        const P3 e1 = sub(point(m_.vertex(2, c, (i + 1) % n)), q);
        const P3 e2 = sub(point(m_.vertex(2, c, (i + n - 1) % n)), q);
        const Real angle =
            std::acos(std::clamp(dot(e1, e2) / (norm(e1) * norm(e2)), Real{-1}, Real{1})) * kDegrees;
        const P3 f1 = axpy(-dot(e1, normal_), normal_, e1), f2 = axpy(-dot(e2, normal_), normal_, e2);
        Real det = dot(cross(f1, f2), normal_), frob = dot(f1, f1) + dot(f2, f2);
        if (tri) {  // against the equilateral corner: A = J W^-1
          const P3 a2 = axpy(-1 / std::sqrt(Real{3}), f1, P3{2 * f2[0] / std::sqrt(Real{3}),
                                                              2 * f2[1] / std::sqrt(Real{3}),
                                                              2 * f2[2] / std::sqrt(Real{3})});
          frob = dot(f1, f1) + dot(a2, a2);
          det *= 2 / std::sqrt(Real{3});
        }
        if (!(det > 0)) out.valid = false;
        out.kappa += det > 0 ? frob / (2 * det) : kInfinity;
        out.least = std::min(out.least, angle);
        const bool moves = tri || i == at_v || i == (at_v + 1) % n || i == (at_v + n - 1) % n;
        if (moves) {
          out.least_moved = std::min(out.least_moved, angle);
          out.moved.push_back(angle);
        }
      }
    }
    put(v_, keep);
    return out;
  }

  // Lowers the patch's summed condition number; returns the move over the
  // shortest edge.
  Real shape(Real floor) {
    Real a = 0, b = 0;
    Measure now = measure(at(0, 0));
    for (int inner = 0; inner < kInner; ++inner) {
      const Real ga = (measure(at(a + kDelta, b)).kappa - measure(at(a - kDelta, b)).kappa) / (2 * kDelta);
      const Real gb = (measure(at(a, b + kDelta)).kappa - measure(at(a, b - kDelta)).kappa) / (2 * kDelta);
      const Real g = std::hypot(ga, gb);
      if (!std::isfinite(g) || g == 0) break;
      bool stepped = false;
      Real s = kFirstStep;
      for (int h = 0; h <= kHalvings && !stepped; ++h, s /= 2) {
        const Measure t = measure(at(a - s * ga / g, b - s * gb / g));
        if (t.valid && t.kappa < now.kappa * (1 - kProgress) && t.least >= floor) {
          a -= s * ga / g;
          b -= s * gb / g;
          now = t;
          stepped = true;
        }
      }
      if (!stepped) break;
    }
    return finish(a, b);
  }

  // Raises the least angle of the corners the vertex moves (Freitag and
  // Plassmann); returns the move over the shortest edge.
  Real angle(Real threshold) {
    Real a = 0, b = 0;
    Measure now = measure(at(0, 0));
    if (now.least >= threshold) return 0;
    for (int inner = 0; inner < kInner; ++inner) {
      const Measure pa = measure(at(a + kDelta, b)), ma = measure(at(a - kDelta, b));
      const Measure pb = measure(at(a, b + kDelta)), mb = measure(at(a, b - kDelta));
      std::vector<std::array<Real, 2>> g;
      for (std::size_t i = 0; i < now.moved.size(); ++i) {
        if (now.moved[i] > now.least_moved + kActive) continue;
        g.push_back({(pa.moved[i] - ma.moved[i]) / (2 * kDelta), (pb.moved[i] - mb.moved[i]) / (2 * kDelta)});
      }
      const auto d = least_norm(g);
      const Real l = std::hypot(d[0], d[1]);
      if (!std::isfinite(l) || l == 0) break;
      bool stepped = false;
      Real s = kFirstStep;
      for (int h = 0; h <= kHalvings && !stepped; ++h, s /= 2) {
        const Measure t = measure(at(a + s * d[0] / l, b + s * d[1] / l));
        if (t.valid && t.least_moved > now.least_moved * (1 + kProgress)) {
          a += s * d[0] / l;
          b += s * d[1] / l;
          now = t;
          stepped = true;
        }
      }
      if (!stepped) break;
    }
    return finish(a, b);
  }

  P3 point(LO v) const { return {x_(v, 0), x_(v, 1), x_(v, 2)}; }
  [[nodiscard]] Real shortest() const { return shortest_; }

 private:
  Offset begin(LO v) const { return cells_.offsets[static_cast<std::size_t>(v)]; }
  Offset end(LO v) const { return cells_.offsets[static_cast<std::size_t>(v) + 1]; }
  void put(LO v, const P3& p) const {
    for (int q = 0; q < 3; ++q) x_(v, q) = p[static_cast<std::size_t>(q)];
  }

  Real finish(Real a, Real b) {
    if (a == 0 && b == 0) return 0;
    const P3 p = at(a, b);
    const Real moved = norm(sub(p, x0_)) / shortest_;
    put(v_, p);
    return moved;
  }

  // The point of least norm in the convex hull of the gradients: the
  // steepest ascent of their minimum; zero when the origin is in the hull.
  static std::array<Real, 2> least_norm(const std::vector<std::array<Real, 2>>& g) {
    std::array<Real, 2> best{0, 0};
    Real best_norm = kInfinity, scale = 0;
    for (const auto& p : g) scale = std::max(scale, std::hypot(p[0], p[1]));
    const auto consider = [&](const std::array<Real, 2>& c) {
      const Real cc = c[0] * c[0] + c[1] * c[1];
      for (const auto& p : g) {
        if (c[0] * p[0] + c[1] * p[1] < cc - 1e-12 * scale * scale) return;
      }
      if (std::sqrt(cc) < best_norm) {
        best_norm = std::sqrt(cc);
        best = c;
      }
    };
    for (std::size_t i = 0; i < g.size(); ++i) {
      consider(g[i]);
      for (std::size_t j = i + 1; j < g.size(); ++j) {
        const Real ex = g[j][0] - g[i][0], ey = g[j][1] - g[i][1], ee = ex * ex + ey * ey;
        if (ee == 0) continue;
        const Real t = std::clamp(-(g[i][0] * ex + g[i][1] * ey) / ee, Real{0}, Real{1});
        consider({g[i][0] + t * ex, g[i][1] + t * ey});
      }
    }
    return best_norm <= 1e-12 * scale ? std::array<Real, 2>{0, 0} : best;
  }

  MeshView<Kokkos::HostSpace> m_;
  CoordView<Kokkos::HostSpace> x_;
  const detail::VertexCells& cells_;
  std::optional<Real> radius_;
  LO v_ = 0;
  P3 x0_{}, normal_{}, t1_{}, t2_{};
  Real shortest_ = 0;
};

// The least corner angle (degrees) over the owned cells of every rank.
Real least_angle(const Comm& comm, const HostMesh& mesh) {
  const auto m = mesh.view();
  const auto x = mesh.coords();
  Real least = 360;
  for (LO c = 0; c < mesh.segment_table(2).owned_end; ++c) {
    const LO n = m.num_vertices(2, c);
    for (LO i = 0; i < n; ++i) {
      const LO q = m.vertex(2, c, i), a = m.vertex(2, c, (i + n - 1) % n), b = m.vertex(2, c, (i + 1) % n);
      const P3 e1{x(b, 0) - x(q, 0), x(b, 1) - x(q, 1), x(b, 2) - x(q, 2)};
      const P3 e2{x(a, 0) - x(q, 0), x(a, 1) - x(q, 1), x(a, 2) - x(q, 2)};
      const Real cosine = std::clamp(dot(e1, e2) / (norm(e1) * norm(e2)), Real{-1}, Real{1});
      least = std::min(least, std::acos(cosine) * kDegrees);
    }
  }
  return comm.allreduce(least, Op::min);
}

}  // namespace

QualitySmoothReport quality_smooth(const Comm& comm, HostMesh& mesh, const QualitySmoothOptions& options) {
  const int dim = mesh.dim();
  const LO n = mesh.count(0);
  const auto m = mesh.view();
  const bool no_ghosts = comm.size() > 1 && (mesh.ghost_layers() < 1 || mesh.ghost_bridge_dim() != 0);
  bool bad = dim != 2 || mesh.edited() || no_ghosts || options.shape_sweeps < 0 || options.angle_sweeps < 0 ||
             !(options.angle_threshold > 0) || !(options.angle_threshold <= 90) ||
             !std::isfinite(options.tolerance) || options.tolerance < 0 ||
             (!options.fixed.empty() && options.fixed.size() != static_cast<std::size_t>(n));
  const Real radius = options.sphere_radius.value_or(1);
  bad = bad || !std::isfinite(radius) || radius <= 0;
  if (dim == 2) {
    for (LO c = 0; c < mesh.count(2) && !bad; ++c) {
      bad = m.topology(2, c) != Topology::tri && m.topology(2, c) != Topology::quad;
    }
  }
  // Everything but the corners checked before the cells are measured.
  const auto agree = [&](bool flag) {
    const auto num = [](Real r) { return std::isfinite(r) ? r : 0; };
    const Real s = options.shape_sweeps, a = options.angle_sweeps, t = num(options.angle_threshold),
               tol = num(options.tolerance), given = options.sphere_radius ? 1 : 0, r = num(radius);
    std::array<Real, 13> f{flag ? Real{1} : Real{0}, s, -s, a, -a, t, -t, tol, -tol, given, -given, r, -r};
    comm.allreduce(std::span<Real>(f), Op::max);
    for (std::size_t i = 1; i < f.size(); i += 2) {
      if (f[i] != -f[i + 1]) return false;
    }
    return f[0] == 0;
  };
  const auto refuse = [] {
    throw std::invalid_argument(
        "quality_smooth: a rank passes a mesh not of dimension 2, with a cell not a triangle or quad, with an "
        "inverted corner or with edits, or, on more than one rank, without a ghost layer through vertices; "
        "negative sweeps, an angle threshold outside (0, 90], a negative or non-finite tolerance, a fixed mask "
        "not one per vertex or a sphere radius not positive and finite; or ranks differ in their options");
  };
  if (!agree(bad)) refuse();

  const detail::VertexCells cells = detail::vertex_cells(mesh);
  const std::vector<std::uint8_t> skin = options.fixed.empty() ? detail::skin(mesh) : std::vector<std::uint8_t>{};
  const LO owned = mesh.segment_table(0).owned_end;
  std::vector<std::uint8_t> moving(static_cast<std::size_t>(owned), 0);
  Smoother smoother(mesh, cells, options.sphere_radius);
  bool inverted = false;
  for (LO v = 0; v < owned; ++v) {
    const auto i = static_cast<std::size_t>(v);
    if (cells.offsets[i + 1] == cells.offsets[i]) continue;
    smoother.frame(v);
    inverted = inverted || !smoother.measure(smoother.point(v)).valid;
    const bool fixed = options.fixed.empty() ? skin[i] != 0 : options.fixed[i] != 0;
    moving[i] = fixed ? 0 : 1;
  }
  if (comm.allreduce(inverted ? Real{1} : Real{0}, Op::max) != 0) refuse();

  const std::vector<std::int32_t> colour = detail::vertex_colours(comm, mesh, cells);
  std::int32_t colours = 0;
  for (const auto h : colour) colours = std::max(colours, h + 1);
  colours = static_cast<std::int32_t>(comm.allreduce(static_cast<Real>(colours), Op::max));
  std::vector<std::vector<LO>> by_colour(static_cast<std::size_t>(colours));
  for (LO v = 0; v < owned; ++v) {
    const auto i = static_cast<std::size_t>(v);
    if (moving[i] != 0) by_colour[static_cast<std::size_t>(colour[i])].push_back(v);
  }
  std::optional<HaloExchange<Kokkos::HostSpace>> halo;
  if (comm.size() > 1) halo.emplace(comm, mesh, 0);
  const auto x = mesh.coords();
  const View2<Real, Kokkos::HostSpace> values(x.data(), static_cast<std::size_t>(n), 3);

  QualitySmoothReport report;
  report.least_angle_before = least_angle(comm, mesh);
  const Real floor = report.least_angle_before;
  // One phase: sweeps of `step` over the colours in turn.
  const auto phase = [&](int sweeps, int& made, auto&& step) {
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      Real most = 0;
      for (const auto& vertices : by_colour) {
        for (const LO v : vertices) {
          smoother.frame(v);
          most = std::max(most, step());
        }
        if (halo) halo->update(comm, values);
      }
      ++made;
      if (options.tolerance > 0 && comm.allreduce(most, Op::max) <= options.tolerance) break;
    }
  };
  phase(options.shape_sweeps, report.shape_sweeps, [&] { return smoother.shape(floor); });
  phase(options.angle_sweeps, report.angle_sweeps, [&] { return Real{0}; });
  report.least_angle_after = least_angle(comm, mesh);
  return report;
}

}  // namespace soma
