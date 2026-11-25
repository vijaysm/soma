// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// All-quad conforming refinement (refine_quads): marks on corners, promotion
// to a fixpoint, Schneiders' vertex-based templates with edges split in
// three, and the shared assembly of refine.cpp.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "refine_internal.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/core/mesh_geometry.hpp"
#include "soma/fields/registry.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/topology/sides.hpp"

namespace soma {

namespace {

using detail::refine::Host;
using detail::refine::NewVertex;
using detail::refine::Numbers;
using detail::refine::P3;
using detail::refine::find_lid;
using detail::refine::lid_table;

constexpr const char* who = "refine_quads";
constexpr const char* level_field = "refine:level";

// A node of a template, in thirds of the reference square: (a/3, b/3).
struct Node {
  int a = 0, b = 0;
};
using Quad = std::array<Node, 4>;

// The templates by case (whole, one corner, two adjacent corners, all four),
// for marked corners v0 (and v1), children in the documented order.
enum Case : std::uint8_t { whole = 0, corner = 1, side = 2, full = 3 };

const std::vector<Quad>& templates(Case c) {
  static const std::array<std::vector<Quad>, 4> all = [] {
    const Node v0{0, 0}, v1{3, 0}, v2{3, 3}, v3{0, 3};
    const Node a{1, 0}, b{2, 0}, d{3, 1}, c{0, 1}, p{1, 1}, q{2, 1}, r{1, 2}, s{2, 2};
    std::array<std::vector<Quad>, 4> t;
    t[whole] = {{v0, v1, v2, v3}};
    t[corner] = {{v0, a, p, c}, {a, v1, v2, p}, {c, p, v2, v3}};
    t[side] = {{v0, a, p, c}, {a, b, q, p}, {b, v1, d, q}, {c, p, r, v3}, {p, q, s, r}, {q, d, v2, s}, {r, s, v2, v3}};
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) t[full].push_back({Node{i, j}, Node{i + 1, j}, Node{i + 1, j + 1}, Node{i, j + 1}});
    }
    return t;
  }();
  return all[c];
}

// Turned a quarter k times: canonical corner i becomes corner (i + k) mod 4.
Node rotated(Node n, int k) {
  for (int i = 0; i < k; ++i) n = {3 - n.b, n.a};
  return n;
}

// The case and rotation of a cell from its marked corners (bit i: corner i);
// cases other than these are promoted first.
std::pair<Case, int> case_of(unsigned m) {
  const int count = std::popcount(m);
  if (count == 0) return {whole, 0};
  if (count == 4) return {full, 0};
  if (count == 1) return {corner, std::countr_zero(m)};
  for (int k = 0; k < 4; ++k) {
    if (m == ((1u << k) | (1u << ((k + 1) % 4)))) return {side, k};
  }
  throw std::logic_error("refine_quads: a cell is left to promote");
}

P3 cross(const P3& a, const P3& b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Real dot(const P3& a, const P3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Whether a quad turns about n at every corner: a bilinear quad that does
// has a positive Jacobian everywhere (it is affine in each reference
// coordinate), at its GLL points too.
bool turns_about(const std::array<P3, 4>& x, const P3& n) {
  for (std::size_t i = 0; i < 4; ++i) {
    if (dot(cross(x[(i + 1) % 4] - x[i], x[(i + 2) % 4] - x[(i + 1) % 4]), n) <= 0) return false;
  }
  return true;
}

// Whether a cell with these marked corners is promoted to all four.
bool promoted(unsigned m) { return std::popcount(m) == 3 || m == 0b0101 || m == 0b1010; }

}  // namespace

RefineResult refine_quads(const Comm& comm, const HostMesh& mesh, std::span<const std::int8_t> marks,
                          QuadRefineOptions options) {
  detail::refine::check(comm, mesh, 1, who, false);
  const int dim = mesh.dim();
  const LO owned = mesh.segment_table(dim).owned_end;
  {
    // Somewhere: marks not one per owned cell; a cell other than a quad; a
    // radius not positive and finite.
    const Real radius = options.sphere_radius.value_or(1);
    std::int32_t flags[3] = {static_cast<LO>(marks.size()) == owned ? 0 : 1, dim == 2 ? 0 : 1,
                             std::isfinite(radius) && radius > 0 ? 0 : 1};
    const MeshView<Host> m = mesh.view();
    for (LO e = 0; e < mesh.count(dim) && flags[1] == 0; ++e) flags[1] = m.topology(dim, e) == Topology::quad ? 0 : 1;
    comm.allreduce(std::span<std::int32_t>(flags), Op::max);
    if (flags[1] != 0) throw std::invalid_argument(std::string(who) + ": a rank's mesh holds a cell other than a quad");
    if (flags[0] != 0) throw std::invalid_argument(std::string(who) + ": a rank's marks are not one per owned cell");
    if (flags[2] != 0) {
      throw std::invalid_argument(std::string(who) + ": a rank's sphere radius is not positive and finite");
    }
    // Every rank the same radius, or none: (given, radius) as max and min.
    std::array<Real, 4> agree{options.sphere_radius ? 1.0 : 0.0, radius, -(options.sphere_radius ? 1.0 : 0.0),
                              -radius};
    comm.allreduce(std::span<Real>(agree), Op::max);
    if (agree[0] != -agree[2] || agree[1] != -agree[3]) {
      throw std::invalid_argument(std::string(who) + ": the ranks differ in sphere radius");
    }
  }

  const HostMesh work = detail::refine::work_part(comm, mesh);
  const MeshView<Host> w = work.view();
  const LO cells = work.count(2);
  const auto wv = work.gid(0);
  const auto we = work.gid(1);
  const auto wc = work.gid(2);
  const auto in_cell = lid_table(mesh, 2, owned);
  std::vector<LO> input(static_cast<std::size_t>(cells));  // the input's owned cell of each work cell
  for (LO e = 0; e < cells; ++e) input[static_cast<std::size_t>(e)] = find_lid(in_cell, wc(e));

  // Marks on corners, agreed by every rank holding the corner; then
  // promotion until no cell on any rank changes. Marks only grow and a
  // superset of a promoted cell's marks promotes it too (or marks it fully),
  // so the fixpoint does not depend on the order: nor on the ranks.
  const HaloExchange<Host> halo(comm, work, 0);
  View2<std::int32_t, Host> vmark("soma::refine_quads.marks", static_cast<std::size_t>(work.count(0)), 1);
  const auto corners_marked = [&](LO e) {
    unsigned m = 0;
    for (LO i = 0; i < 4; ++i) m |= vmark(static_cast<std::size_t>(w.vertex(2, e, i)), 0) != 0 ? 1u << i : 0u;
    return m;
  };
  for (LO e = 0; e < cells; ++e) {
    if (marks[static_cast<std::size_t>(input[static_cast<std::size_t>(e)])] == 0) continue;
    for (LO i = 0; i < 4; ++i) vmark(static_cast<std::size_t>(w.vertex(2, e, i)), 0) = 1;
  }
  halo.reduce(comm, vmark, Op::max);
  for (;;) {
    std::int32_t changed = 0;
    for (LO e = 0; e < cells; ++e) {
      if (!promoted(corners_marked(e))) continue;
      for (LO i = 0; i < 4; ++i) vmark(static_cast<std::size_t>(w.vertex(2, e, i)), 0) = 1;
      changed = 1;
    }
    halo.reduce(comm, vmark, Op::max);
    if (comm.allreduce(changed, Op::max) == 0) break;
  }

  // Gids: edge vertices from V, inside vertices from V + 2 E, children 9 per cell.
  const Numbers nums = detail::refine::numbers(comm, work, [](const MeshView<Host>&, LO) { return GO{9}; }, who);
  {
    GO top = 0;
    for (LO e = 0; e < cells; ++e) top = std::max(top, nums.number(wc(e)));
    top = comm.allreduce(top, Op::max);
    GO twice = 0, base = 0, inside = 0, end = 0;
    if (__builtin_mul_overflow(nums.emax, GO{2}, &twice) || __builtin_add_overflow(nums.vmax, twice, &base) ||
        __builtin_mul_overflow(top, GO{4}, &inside) || __builtin_add_overflow(base, inside, &end)) {
      throw std::invalid_argument(std::string(who) + ": the refined mesh's gids would not fit a GO");
    }
  }
  const GO inside_base = nums.vmax + 2 * nums.emax;

  std::array<std::vector<GO>, 6> kid_gids, kid_conn;
  std::vector<NewVertex> added;
  const auto wx = work.coords();
  GO against = std::numeric_limits<GO>::max();  // the least split cell turned against, by gid
  std::vector<Case> case_by_input(static_cast<std::size_t>(owned), whole);
  const auto edges = work.adjacency(2, 1);
  for (LO e = 0; e < cells; ++e) {
    const auto ei = static_cast<std::size_t>(e);
    std::array<GO, 4> g{};
    std::array<LO, 4> lid{};
    for (LO i = 0; i < 4; ++i) {
      lid[static_cast<std::size_t>(i)] = w.vertex(2, e, i);
      g[static_cast<std::size_t>(i)] = wv(lid[static_cast<std::size_t>(i)]);
    }
    // The gid of the edge from corner i to corner i + 1.
    std::array<GO, 4> edge_gid{};
    const Offset e0 = edges.offsets(ei);
    for (int s = 0; s < 4; ++s) {
      const topology::Side sd = topology::side_of(w, 2, e, 1, s);
      const auto at = [&](LO v) { return static_cast<int>(std::find(lid.begin(), lid.end(), v) - lid.begin()); };
      const int i = at(sd.v[0]), j = at(sd.v[1]);
      const int from = (j == (i + 1) % 4) ? i : j;
      edge_gid[static_cast<std::size_t>(from)] = we(edges.targets(static_cast<std::size_t>(e0 + s)));
    }
    const auto [kind, k] = case_of(corners_marked(e));
    case_by_input[static_cast<std::size_t>(input[ei])] = kind;
    const GO n = nums.number(wc(e));
    std::array<P3, 4> x{};
    for (std::size_t i = 0; i < 4; ++i) x[i] = {wx(lid[i], 0), wx(lid[i], 1), wx(lid[i], 2)};
    const auto corner_at = [&](GO cg) {
      return x[static_cast<std::size_t>(std::find(g.begin(), g.end(), cg) - g.begin())];
    };
    // A node's gid and position; new ones go to `added` with their corners
    // and weights.
    const auto node = [&](Node p) -> std::pair<GO, P3> {
      const auto made = [&](NewVertex nv) {
        const std::pair<GO, P3> out{nv.gid, detail::refine::place(nv, corner_at, options.sphere_radius)};
        added.push_back(std::move(nv));
        return out;
      };
      const bool ea = p.a == 0 || p.a == 3, eb = p.b == 0 || p.b == 3;
      if (ea && eb) {
        const auto i = static_cast<std::size_t>(p.b == 0 ? (p.a == 0 ? 0 : 1) : (p.a == 3 ? 2 : 3));
        return {g[i], x[i]};
      }
      if (ea || eb) {
        // On the edge from corner `from`, t thirds along it.
        const int from = p.b == 0 ? 0 : p.a == 3 ? 1 : p.b == 3 ? 2 : 3;
        const int t = from == 0 ? p.a : from == 1 ? p.b : from == 2 ? 3 - p.a : 3 - p.b;
        const GO a = g[static_cast<std::size_t>(from)], b = g[static_cast<std::size_t>((from + 1) % 4)];
        const GO lo = std::min(a, b), hi = std::max(a, b);
        const int t_lo = a < b ? t : 3 - t;  // thirds from the lower-gid end
        const GO gid = nums.vmax + 2 * (edge_gid[static_cast<std::size_t>(from)] - 1) + (t_lo - 1) + 1;
        return made({gid, {lo, hi}, {static_cast<Real>(3 - t_lo) / 3, static_cast<Real>(t_lo) / 3}});
      }
      // Inside: numbered by the corner it lies nearest; bilinear weights.
      const int j = p.b == 1 ? (p.a == 1 ? 0 : 1) : (p.a == 2 ? 2 : 3);
      const GO gid = inside_base + 4 * (n - 1) + j + 1;
      const std::array<int, 4> ninths{(3 - p.a) * (3 - p.b), p.a * (3 - p.b), p.a * p.b, (3 - p.a) * p.b};
      std::array<std::pair<GO, Real>, 4> by_gid{};
      for (std::size_t i = 0; i < 4; ++i) by_gid[i] = {g[i], static_cast<Real>(ninths[i]) / 9};
      std::sort(by_gid.begin(), by_gid.end());
      NewVertex nv{gid, {}, {}};
      for (const auto& [cg, cw] : by_gid) {
        nv.corners.push_back(cg);
        nv.weights.push_back(cw);
      }
      return made(std::move(nv));
    };
    // The cell's area vector: a split cell and its children must turn about it.
    P3 up;
    for (std::size_t i = 0; i < 4; ++i) up = up + cross(x[i], x[(i + 1) % 4]);
    bool valid = kind == whole || turns_about(x, up);
    const std::vector<Quad>& kids = templates(kind);
    for (std::size_t c = 0; c < kids.size(); ++c) {
      kid_gids[1].push_back(nums.child(wc(e), static_cast<GO>(c)));
      std::array<P3, 4> y{};
      for (std::size_t i = 0; i < 4; ++i) {
        const auto [gid, at] = node(rotated(kids[c][i], k));
        kid_conn[1].push_back(gid);
        y[i] = at;
      }
      valid = valid && (kind == whole || turns_about(y, up));
    }
    if (!valid) against = std::min(against, wc(e));
  }
  against = comm.allreduce(against, Op::min);
  if (against != std::numeric_limits<GO>::max()) {
    throw std::invalid_argument(std::string(who) + ": cell " + std::to_string(against) +
                                " or one of its children turns against it");
  }

  detail::refine::Level fine =
      detail::refine::assemble(comm, mesh, work, kid_gids, kid_conn, added, nums, detail::refine::Polygons{},
                               options.sphere_radius);
  RefineResult out;
  out.mesh = std::move(fine.mesh);
  out.parent = std::move(fine.parent);
  out.dropped = std::move(fine.dropped);

  // The new vertices are on the mesh's surface where this put them there.
  if (const auto g = geometry(mesh.values())) {
    const MeshGeometry made =
        options.sphere_radius ? MeshGeometry::sphere(*options.sphere_radius) : MeshGeometry::affine();
    if (g->kind == made.kind && g->center == made.center && g->radius == made.radius) {
      set_geometry(out.mesh.values(), *g);
    }
  }

  // Levels: a full split's children one up.
  std::vector<std::int32_t> level_of(static_cast<std::size_t>(owned), 0);
  if (mesh.fields().contains(level_field)) {
    const auto l = mesh.fields().get<std::int32_t>(level_field, dim).view();
    for (LO e = 0; e < owned; ++e) level_of[static_cast<std::size_t>(e)] = l(static_cast<std::size_t>(e), 0);
  }
  auto lv = out.mesh.fields().add_tag<std::int32_t>(level_field, dim, 1).view();
  for (LO c = 0; c < out.mesh.count(dim); ++c) {
    const auto p = static_cast<std::size_t>(out.parent[static_cast<std::size_t>(c)]);
    lv(static_cast<std::size_t>(c), 0) = level_of[p] + (case_by_input[p] == full ? 1 : 0);
  }

  out.child_offsets.assign(static_cast<std::size_t>(owned) + 1, 0);
  for (const LO p : out.parent) ++out.child_offsets[static_cast<std::size_t>(p) + 1];
  for (std::size_t p = 0; p < static_cast<std::size_t>(owned); ++p) out.child_offsets[p + 1] += out.child_offsets[p];
  out.children.resize(out.parent.size());
  std::vector<Offset> next(out.child_offsets.begin(), out.child_offsets.end() - 1);
  for (std::size_t c = 0; c < out.parent.size(); ++c) {
    out.children[static_cast<std::size_t>(next[static_cast<std::size_t>(out.parent[c])]++)] = static_cast<LO>(c);
  }
  return out;
}

}  // namespace soma
