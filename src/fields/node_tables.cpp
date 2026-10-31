// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/fields/node_tables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include "soma/topology/canonical.hpp"

namespace soma {
namespace {

constexpr int newton_iteration_cap = 100;
constexpr long double newton_tolerance = 1e-15L;

// Optimized blending parameters, Hesthaven and Warburton (2008), section 6.1
// (Nodes2D alpopt, Nodes3D alphastore); orders above the tables use their defaults.
constexpr Real alpha_triangle_table[max_optimized_simplex_order] = {
    0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 1.2832,
    1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258};
constexpr Real alpha_tet_table[max_optimized_simplex_order] = {
    0.0, 0.0, 0.0, 0.1002, 1.1332, 1.5608, 1.3413, 1.2577,
    1.1603, 1.10153, 0.6080, 0.4523, 0.8856, 0.8717, 0.9655};

Real alpha_triangle(int p) {
  return p <= max_optimized_simplex_order ? alpha_triangle_table[p - 1] : 5.0 / 3.0;
}
Real alpha_tet(int p) { return p <= max_optimized_simplex_order ? alpha_tet_table[p - 1] : 1.0; }

using Point = std::array<Real, 3>;
using Index = std::array<int, 3>;

const char* name_of(NodeFamily f) {
  switch (f) {
    case NodeFamily::lagrange_equispaced: return "lagrange_equispaced";
    case NodeFamily::gll: return "gll";
    case NodeFamily::gauss: return "gauss";
  }
  return "unknown";
}

const char* name_of(NodeOrdering o) {
  switch (o) {
    case NodeOrdering::moab_native: return "moab_native";
    case NodeOrdering::tensor_lexicographic: return "tensor_lexicographic";
    case NodeOrdering::hierarchical: return "hierarchical";
  }
  return "unknown";
}

[[noreturn]] void unsupported(const NodeTableKey& k, const char* reason) {
  std::ostringstream os;
  os << "unsupported node table: topology=" << name(k.topology)
     << " family=" << name_of(k.family) << " order=" << k.order
     << " ordering=" << name_of(k.ordering) << " serendipity=" << (k.serendipity ? 1 : 0)
     << " (" << reason << ")";
  throw std::invalid_argument(os.str());
}

void validate(const NodeTableKey& k) {
  switch (k.topology) {
    case Topology::edge:
    case Topology::tri:
    case Topology::quad:
    case Topology::tet:
    case Topology::prism:
    case Topology::hex:
    case Topology::pyramid: break;
    default: unsupported(k, "no node table for this topology");
  }
  if (k.family == NodeFamily::gauss) {
    unsupported(k, "gauss nodes are not implemented");
  }
  if (k.order < 1 || k.order > max_node_order) {
    unsupported(k, "order must be in [1, max_node_order]");
  }
  if (k.topology == Topology::pyramid) {
    // GLL and equispaced nodes coincide for the pyramid at order 1 and, for the
    // serendipity set (corners and mid-edge nodes), at order 2. The 14-node
    // pyramid has a mid-region node and exists for equispaced nodes only.
    const bool families_coincide = k.order == 1 || (k.serendipity && k.order == 2);
    if (k.family != NodeFamily::lagrange_equispaced && !families_coincide) {
      unsupported(k, "pyramids support lagrange_equispaced only, except where GLL and "
                     "equispaced nodes coincide (order 1, serendipity order 2)");
    }
    if (k.order > 2) {
      unsupported(k, "pyramids support order <= 2");
    }
    if (k.ordering == NodeOrdering::tensor_lexicographic) {
      unsupported(k, "pyramids have no tensor ordering; use hierarchical or moab_native");
    }
  }
  if (k.serendipity) {
    const bool shape_ok = k.topology == Topology::quad || k.topology == Topology::hex ||
                          k.topology == Topology::prism || k.topology == Topology::pyramid;
    if (!shape_ok || k.order != 2) {
      unsupported(k, "serendipity exists for quad, hex, prism and pyramid at order 2 only");
    }
  }
  if (k.ordering == NodeOrdering::moab_native && k.order > 2) {
    unsupported(k, "moab_native is defined for order <= 2");
  }
}

// Orders on a mapped integer lattice. A node is (i, j, k) with the corners at
// p * corner(v); simplices additionally satisfy the barycentric bound.
Index corner(Topology t, int v) {
  switch (t) {
    case Topology::edge: return {v, 0, 0};
    case Topology::tri: {
      constexpr Index c[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
      return c[v];
    }
    case Topology::quad: {
      constexpr Index c[] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
      return c[v];
    }
    case Topology::tet: {
      constexpr Index c[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
      return c[v];
    }
    case Topology::prism: {
      constexpr Index c[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
      return c[v];
    }
    case Topology::hex: {
      constexpr Index c[] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                             {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
      return c[v];
    }
    default: return {0, 0, 0};
  }
}

struct Lattice {
  int p = 1;
  std::vector<Index> nodes;  // tensor_lexicographic order
  std::vector<int> position; // encoded (i, j, k) -> index in nodes, or -1

  int encode(const Index& n) const { return n[0] + (p + 1) * (n[1] + (p + 1) * n[2]); }
  int find(const Index& n) const {
    for (std::size_t d = 0; d < 3; ++d) {
      if (n[d] < 0 || n[d] > p) {
        return -1;
      }
    }
    return position[static_cast<std::size_t>(encode(n))];
  }
};

Lattice make_lattice(Topology t, int p) {
  Lattice L;
  L.p = p;
  switch (t) {
    case Topology::edge:
      for (int i = 0; i <= p; ++i) L.nodes.push_back({i, 0, 0});
      break;
    case Topology::tri:
      for (int j = 0; j <= p; ++j)
        for (int i = 0; i <= p - j; ++i) L.nodes.push_back({i, j, 0});
      break;
    case Topology::quad:
      for (int j = 0; j <= p; ++j)
        for (int i = 0; i <= p; ++i) L.nodes.push_back({i, j, 0});
      break;
    case Topology::tet:
      for (int k = 0; k <= p; ++k)
        for (int j = 0; j <= p - k; ++j)
          for (int i = 0; i <= p - j - k; ++i) L.nodes.push_back({i, j, k});
      break;
    case Topology::prism:
      for (int k = 0; k <= p; ++k)
        for (int j = 0; j <= p; ++j)
          for (int i = 0; i <= p - j; ++i) L.nodes.push_back({i, j, k});
      break;
    case Topology::hex:
      for (int k = 0; k <= p; ++k)
        for (int j = 0; j <= p; ++j)
          for (int i = 0; i <= p; ++i) L.nodes.push_back({i, j, k});
      break;
    default: break;
  }
  const auto stride = static_cast<std::size_t>(p + 1);
  L.position.assign(stride * stride * stride, -1);
  for (std::size_t q = 0; q < L.nodes.size(); ++q) {
    L.position[static_cast<std::size_t>(L.encode(L.nodes[q]))] = static_cast<int>(q);
  }
  return L;
}

// Vertices, then edge interiors in canonical edge order (first to second canonical
// vertex), then face interiors in canonical face order (i along the face's first
// to second vertex, j along first to last; faces of cells of dimension 3 only),
// then the cell interior in tensor order. `prefix` receives the vertex + edge count.
std::vector<int> hierarchical_order(Topology t, const Lattice& L, int& prefix) {
  const int p = L.p;
  std::vector<char> used(L.nodes.size(), 0);
  std::vector<int> out;
  out.reserve(L.nodes.size());
  const auto take = [&](const Index& n) {
    const int q = L.find(n);
    if (q < 0) {
      throw std::logic_error("hierarchical_order: lattice node outside the element");
    }
    if (!used[static_cast<std::size_t>(q)]) {
      used[static_cast<std::size_t>(q)] = 1;
      out.push_back(q);
    }
  };
  const auto at = [&](const Index& c0, const Index& u1, int i, const Index& u2, int j) {
    Index n{};
    for (int d = 0; d < 3; ++d) {
      n[static_cast<std::size_t>(d)] = p * c0[static_cast<std::size_t>(d)] +
                                       i * u1[static_cast<std::size_t>(d)] +
                                       j * u2[static_cast<std::size_t>(d)];
    }
    return n;
  };
  const auto diff = [](const Index& a, const Index& b) {
    return Index{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  };
  const Index zero{0, 0, 0};

  for (int v = 0; v < num_vertices(t); ++v) {
    take(at(corner(t, v), zero, 0, zero, 0));
  }
  for (int e = 0; e < canonical::num_sides(t, 1); ++e) {
    const auto s = canonical::side(t, 1, e);
    const Index a = corner(t, s.v[0]);
    const Index u = diff(corner(t, s.v[1]), a);
    for (int i = 1; i < p; ++i) take(at(a, u, i, zero, 0));
  }
  prefix = static_cast<int>(out.size());
  if (dimension(t) == 3) {
    for (int f = 0; f < canonical::num_sides(t, 2); ++f) {
      const auto s = canonical::side(t, 2, f);
      const Index c0 = corner(t, s.v[0]);
      const Index u1 = diff(corner(t, s.v[1]), c0);
      if (s.num_vertices == 3) {
        const Index u2 = diff(corner(t, s.v[2]), c0);
        for (int j = 1; j <= p - 2; ++j)
          for (int i = 1; i <= p - 1 - j; ++i) take(at(c0, u1, i, u2, j));
      } else {
        const Index u2 = diff(corner(t, s.v[3]), c0);
        for (int j = 1; j < p; ++j)
          for (int i = 1; i < p; ++i) take(at(c0, u1, i, u2, j));
      }
    }
  }
  for (std::size_t q = 0; q < L.nodes.size(); ++q) {
    if (!used[q]) {
      out.push_back(static_cast<int>(q));
    }
  }
  return out;
}

std::vector<int> ordering_positions(const NodeTableKey& key, const Lattice& L) {
  std::vector<int> lex(L.nodes.size());
  std::iota(lex.begin(), lex.end(), 0);
  if (key.ordering == NodeOrdering::tensor_lexicographic && !key.serendipity) {
    return lex;
  }
  int prefix = 0;
  std::vector<int> hier = hierarchical_order(key.topology, L, prefix);
  if (key.serendipity) {
    hier.resize(static_cast<std::size_t>(prefix));
  }
  if (key.ordering != NodeOrdering::tensor_lexicographic) {
    return hier;
  }
  std::vector<char> keep(L.nodes.size(), 0);
  for (const int q : hier) keep[static_cast<std::size_t>(q)] = 1;
  std::vector<int> out;
  for (const int q : lex) {
    if (keep[static_cast<std::size_t>(q)]) out.push_back(q);
  }
  return out;
}

std::vector<Real> equispaced_1d(int p) {
  std::vector<Real> x(static_cast<std::size_t>(p) + 1);
  for (int i = 0; i <= p; ++i) {
    x[static_cast<std::size_t>(i)] = static_cast<Real>(2 * i - p) / static_cast<Real>(p);
  }
  return x;
}

// w(x) / (1 - x^2) at x = s / p, where w interpolates (GLL - equispaced) on the
// equispaced nodes. Written as a rational function of the integer s so that an
// equispaced argument makes all but one Lagrange term vanish exactly.
struct Warp {
  int p = 1;
  std::vector<Real> displacement;  // GLL_i - equispaced_i

  explicit Warp(int order) : p(order), displacement(static_cast<std::size_t>(order) + 1) {
    const std::vector<Real> g = gll_points_1d(order);
    const std::vector<Real> e = equispaced_1d(order);
    for (std::size_t i = 0; i < g.size(); ++i) displacement[i] = g[i] - e[i];
  }

  Real operator()(int s) const {
    Real sum = 0;
    for (int i = 1; i < p; ++i) {
      Real term = -displacement[static_cast<std::size_t>(i)] * static_cast<Real>(p) *
                  static_cast<Real>(p) / (4.0 * static_cast<Real>(i) * static_cast<Real>(i - p));
      for (int j = 1; j < p; ++j) {
        if (j != i) {
          term *= static_cast<Real>(s - (2 * j - p)) / static_cast<Real>(2 * (i - j));
        }
      }
      sum += term;
    }
    return sum;
  }
};

// Triangle shift for three barycentric weights a[0..2] (integers over p), applied
// along the edges between the vertex positions v. Symmetric in the three vertices.
Point triangle_shift(int p, const Warp& warp, Real alpha, const std::array<int, 3>& a,
                     const std::array<Point, 3>& v) {
  Point out{0, 0, 0};
  const Real rp = static_cast<Real>(p);
  for (int opp = 0; opp < 3; ++opp) {
    const int b = (opp + 1) % 3;
    const int c = (opp + 2) % 3;
    const Real lb = a[static_cast<std::size_t>(b)] / rp;
    const Real lc = a[static_cast<std::size_t>(c)] / rp;
    const Real la = a[static_cast<std::size_t>(opp)] / rp;
    const Real f = 4.0 * lb * lc * warp(a[static_cast<std::size_t>(c)] - a[static_cast<std::size_t>(b)]) *
                   (1.0 + (alpha * la) * (alpha * la));
    for (std::size_t d = 0; d < 3; ++d) {
      out[d] += 0.5 * f * (v[static_cast<std::size_t>(c)][d] - v[static_cast<std::size_t>(b)][d]);
    }
  }
  return out;
}

const std::array<Point, 3> tri_vertices = {Point{-1, -1, 0}, Point{1, -1, 0}, Point{-1, 1, 0}};
const std::array<Point, 4> tet_vertices = {Point{-1, -1, -1}, Point{1, -1, -1}, Point{-1, 1, -1},
                                           Point{-1, -1, 1}};

Point tri_point(int p, const Warp* warp, int i, int j) {
  const std::vector<Real> e = equispaced_1d(p);
  Point x{e[static_cast<std::size_t>(i)], e[static_cast<std::size_t>(j)], 0};
  if (warp != nullptr && p > 2) {
    const Point s = triangle_shift(p, *warp, alpha_triangle(p), {p - i - j, i, j}, tri_vertices);
    for (std::size_t d = 0; d < 3; ++d) x[d] += s[d];
  }
  return x;
}

Point tet_point(int p, const Warp* warp, int i, int j, int k) {
  const std::vector<Real> e = equispaced_1d(p);
  Point x{e[static_cast<std::size_t>(i)], e[static_cast<std::size_t>(j)],
          e[static_cast<std::size_t>(k)]};
  if (warp == nullptr || p <= 2) {
    return x;
  }
  const std::array<int, 4> a = {p - i - j - k, i, j, k};
  const Real rp = static_cast<Real>(p);
  const auto face_shift = [&](int f) {
    std::array<int, 3> w{};
    std::array<Point, 3> v{};
    int n = 0;
    for (int q = 0; q < 4; ++q) {
      if (q != f) {
        w[static_cast<std::size_t>(n)] = a[static_cast<std::size_t>(q)];
        v[static_cast<std::size_t>(n)] = tet_vertices[static_cast<std::size_t>(q)];
        ++n;
      }
    }
    return triangle_shift(p, *warp, alpha_triangle(p), w, v);
  };
  Point shift{0, 0, 0};
  for (int f = 0; f < 4; ++f) {
    if (a[static_cast<std::size_t>(f)] == 0) {
      shift = face_shift(f);
      for (std::size_t d = 0; d < 3; ++d) x[d] += shift[d];
      return x;
    }
  }
  const Real alpha = alpha_tet(p);
  for (int f = 0; f < 4; ++f) {
    const Real la = a[static_cast<std::size_t>(f)] / rp;
    Real num = 1.0;
    Real den = 1.0;
    for (int q = 0; q < 4; ++q) {
      if (q != f) {
        const Real lq = a[static_cast<std::size_t>(q)] / rp;
        num *= lq;
        den *= lq + 0.5 * la;
      }
    }
    const Real blend = (1.0 + (alpha * la) * (alpha * la)) * num / den;
    const Point s = face_shift(f);
    for (std::size_t d = 0; d < 3; ++d) shift[d] += blend * s[d];
  }
  for (std::size_t d = 0; d < 3; ++d) x[d] += shift[d];
  return x;
}

Point lattice_point(const NodeTableKey& key, const std::vector<Real>& line, const Warp* warp,
                    const Index& n) {
  const int p = key.order;
  const auto at = [&](int i) { return line[static_cast<std::size_t>(i)]; };
  switch (key.topology) {
    case Topology::edge: return {at(n[0]), 0, 0};
    case Topology::quad: return {at(n[0]), at(n[1]), 0};
    case Topology::hex: return {at(n[0]), at(n[1]), at(n[2])};
    case Topology::tri: return tri_point(p, warp, n[0], n[1]);
    case Topology::tet: return tet_point(p, warp, n[0], n[1], n[2]);
    case Topology::prism: {
      Point x = tri_point(p, warp, n[0], n[1]);
      x[2] = at(n[2]);
      return x;
    }
    default: return {0, 0, 0};
  }
}

std::vector<Point> pyramid_points(const NodeTableKey& key) {
  std::vector<Point> pts = {Point{-1, -1, 0}, Point{1, -1, 0}, Point{1, 1, 0}, Point{-1, 1, 0},
                            Point{0, 0, 1}};
  if (key.order == 2) {
    const auto average = [&](const canonical::SideDef& s) {
      Point m{0, 0, 0};
      for (int q = 0; q < s.num_vertices; ++q) {
        for (std::size_t d = 0; d < 3; ++d) m[d] += pts[s.v[q]][d];
      }
      for (std::size_t d = 0; d < 3; ++d) m[d] /= static_cast<Real>(s.num_vertices);
      return m;
    };
    for (int e = 0; e < canonical::num_sides(Topology::pyramid, 1); ++e) {
      pts.push_back(average(canonical::side(Topology::pyramid, 1, e)));
    }
    if (!key.serendipity) {
      Point m{0, 0, 0};
      for (int v = 0; v < 5; ++v) {
        for (std::size_t d = 0; d < 3; ++d) m[d] += pts[static_cast<std::size_t>(v)][d] / 5.0;
      }
      pts.push_back(m);  // MOAB Pyramid14: the extra node is a mid-region node
    }
  }
  return pts;
}

}  // namespace

std::vector<Real> gll_points_1d(int p) {
  if (p < 1 || p > max_node_order) {
    throw std::invalid_argument("gll_points_1d: order " + std::to_string(p) +
                                " is outside [1, " + std::to_string(max_node_order) + "]");
  }
  std::vector<Real> x(static_cast<std::size_t>(p) + 1);
  x.front() = -1;
  x.back() = 1;
  const long double n = static_cast<long double>(p);
  for (int i = 1; i <= (p - 1) / 2; ++i) {
    long double t = -std::cos(std::numbers::pi_v<long double> * static_cast<long double>(i) / n);
    bool converged = false;
    for (int it = 0; it < newton_iteration_cap && !converged; ++it) {
      long double p0 = 1, p1 = t, d0 = 0, d1 = 1;
      for (int k = 1; k < p; ++k) {
        const long double kk = static_cast<long double>(k);
        const long double p2 = ((2 * kk + 1) * t * p1 - kk * p0) / (kk + 1);
        const long double d2 = d0 + (2 * kk + 1) * p1;
        p0 = p1;
        p1 = p2;
        d0 = d1;
        d1 = d2;
      }
      // (1 - t^2) P'' = 2 t P' - n (n + 1) P
      const long double second = (2 * t * d1 - n * (n + 1) * p1) / (1 - t * t);
      const long double step = d1 / second;
      t -= step;
      converged = std::fabs(step) < newton_tolerance;
    }
    if (!converged) {
      throw std::runtime_error("gll_points_1d: Newton iteration did not converge for order " +
                               std::to_string(p));
    }
    x[static_cast<std::size_t>(i)] = static_cast<Real>(t);
    x[static_cast<std::size_t>(p - i)] = static_cast<Real>(-t);
  }
  if (p % 2 == 0) {
    x[static_cast<std::size_t>(p / 2)] = 0;
  }
  return x;
}

NodeTableHost make_node_table(const NodeTableKey& key) {
  validate(key);
  NodeTableHost table;
  table.dim = dimension(key.topology);
  if (key.topology == Topology::pyramid) {
    const std::vector<Point> pts = pyramid_points(key);
    table.count = static_cast<int>(pts.size());
    table.x.resize(static_cast<std::size_t>(table.count) * 3);
    for (int q = 0; q < table.count; ++q) {
      for (int d = 0; d < 3; ++d) {
        table.x[static_cast<std::size_t>(d * table.count + q)] =
            pts[static_cast<std::size_t>(q)][static_cast<std::size_t>(d)];
      }
    }
    return table;
  }
  const Lattice L = make_lattice(key.topology, key.order);
  const std::vector<int> ids = ordering_positions(key, L);
  const bool gll = key.family == NodeFamily::gll;
  const std::vector<Real> line = gll ? gll_points_1d(key.order) : equispaced_1d(key.order);
  const Warp warp(key.order);
  table.count = static_cast<int>(ids.size());
  table.x.assign(static_cast<std::size_t>(table.count) * static_cast<std::size_t>(table.dim), 0);
  for (int q = 0; q < table.count; ++q) {
    const Point x = lattice_point(key, line, gll ? &warp : nullptr,
                                  L.nodes[static_cast<std::size_t>(ids[static_cast<std::size_t>(q)])]);
    for (int d = 0; d < table.dim; ++d) {
      table.x[static_cast<std::size_t>(d * table.count + q)] = x[static_cast<std::size_t>(d)];
    }
  }
  return table;
}

std::vector<int> make_node_permutation(const NodeTableKey& from, NodeOrdering to) {
  validate(from);
  NodeTableKey target = from;
  target.ordering = to;
  validate(target);
  if (from.topology == Topology::pyramid) {
    std::vector<int> id(static_cast<std::size_t>(make_node_table(from).count));
    std::iota(id.begin(), id.end(), 0);
    return id;
  }
  const Lattice L = make_lattice(from.topology, from.order);
  const std::vector<int> src = ordering_positions(from, L);
  const std::vector<int> dst = ordering_positions(target, L);
  std::vector<int> where(L.nodes.size(), -1);
  for (std::size_t i = 0; i < src.size(); ++i) {
    where[static_cast<std::size_t>(src[i])] = static_cast<int>(i);
  }
  std::vector<int> perm(dst.size());
  for (std::size_t i = 0; i < dst.size(); ++i) {
    perm[i] = where[static_cast<std::size_t>(dst[i])];
  }
  return perm;
}

}  // namespace soma
