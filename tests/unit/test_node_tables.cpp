// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/topology.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/node_tables.hpp"
#include "soma/topology/canonical.hpp"

using soma::NodeFamily;
using soma::NodeOrdering;
using soma::NodeTableHost;
using soma::NodeTableKey;
using soma::Topology;
namespace canonical = soma::canonical;

namespace soma_test {

using Point = std::array<double, 3>;
using Wide = long double;

constexpr Topology shapes[] = {Topology::edge, Topology::tri,   Topology::quad,
                               Topology::tet,  Topology::prism, Topology::hex};

std::size_t sz(int i) { return static_cast<std::size_t>(i); }

NodeTableKey key(Topology t, int p, NodeFamily f = NodeFamily::gll,
                 NodeOrdering o = NodeOrdering::tensor_lexicographic, bool serendipity = false) {
  return NodeTableKey{t, f, p, o, serendipity};
}

int expected_count(Topology t, int p) {
  switch (t) {
    case Topology::edge: return p + 1;
    case Topology::tri: return (p + 1) * (p + 2) / 2;
    case Topology::quad: return (p + 1) * (p + 1);
    case Topology::tet: return (p + 1) * (p + 2) * (p + 3) / 6;
    case Topology::prism: return (p + 1) * (p + 1) * (p + 2) / 2;
    case Topology::hex: return (p + 1) * (p + 1) * (p + 1);
    default: return -1;
  }
}

int interior_count(Topology t, int p) {
  switch (t) {
    case Topology::edge: return p - 1;
    case Topology::tri: return (p - 1) * (p - 2) / 2;
    case Topology::quad: return (p - 1) * (p - 1);
    case Topology::tet: return (p - 1) * (p - 2) * (p - 3) / 6;
    case Topology::prism: return (p - 1) * (p - 1) * (p - 2) / 2;
    case Topology::hex: return (p - 1) * (p - 1) * (p - 1);
    default: return -1;
  }
}

Point point(const NodeTableHost& t, int i) {
  Point x{0, 0, 0};
  for (int d = 0; d < t.dim; ++d) x[sz(d)] = t.coord(i, d);
  return x;
}

Point operator+(const Point& a, const Point& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Point operator-(const Point& a, const Point& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Point operator*(double s, const Point& a) { return {s * a[0], s * a[1], s * a[2]}; }
double dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Point cross(const Point& a, const Point& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dist(const Point& a, const Point& b) {
  const Point d = a - b;
  return std::sqrt(dot(d, d));
}

// Reference corners written out independently of the implementation.
std::vector<Point> reference_corners(Topology t) {
  switch (t) {
    case Topology::edge: return {{-1, 0, 0}, {1, 0, 0}};
    case Topology::tri: return {{-1, -1, 0}, {1, -1, 0}, {-1, 1, 0}};
    case Topology::quad: return {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
    case Topology::tet: return {{-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};
    case Topology::pyramid: return {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {0, 0, 1}};
    case Topology::prism:
      return {{-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {-1, 1, 1}};
    case Topology::hex:
      return {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
              {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
    default: return {};
  }
}

Point average(const std::vector<Point>& corners, const canonical::SideDef& s) {
  Point m{0, 0, 0};
  for (int q = 0; q < s.num_vertices; ++q) m = m + corners[s.v[q]];
  return (1.0 / s.num_vertices) * m;
}

Point centroid(const std::vector<Point>& c) {
  Point m{0, 0, 0};
  for (const Point& x : c) m = m + x;
  return (1.0 / static_cast<double>(c.size())) * m;
}

bool contains(const std::vector<Point>& set, const Point& x, double tol) {
  return std::any_of(set.begin(), set.end(), [&](const Point& y) { return dist(x, y) <= tol; });
}

// One-to-one match of two point sets within tol.
::testing::AssertionResult same_set(const std::vector<Point>& a, const std::vector<Point>& b,
                                    double tol) {
  if (a.size() != b.size()) {
    return ::testing::AssertionFailure() << "sizes " << a.size() << " and " << b.size();
  }
  std::vector<bool> used(b.size(), false);
  for (std::size_t i = 0; i < a.size(); ++i) {
    bool found = false;
    for (std::size_t j = 0; j < b.size() && !found; ++j) {
      if (!used[j] && dist(a[i], b[j]) <= tol) {
        used[j] = true;
        found = true;
      }
    }
    if (!found) {
      return ::testing::AssertionFailure()
             << "no match for (" << a[i][0] << ", " << a[i][1] << ", " << a[i][2] << ")";
    }
  }
  return ::testing::AssertionSuccess();
}

std::vector<Point> all_points(const NodeTableHost& t) {
  std::vector<Point> out;
  for (int i = 0; i < t.count; ++i) out.push_back(point(t, i));
  return out;
}

std::vector<Point> table_points(const NodeTableKey& k) { return all_points(soma::make_node_table(k)); }

// Position of x along the segment a-b in [-1,1], or a value > 1e9 if x is off the segment.
double along(const Point& a, const Point& b, const Point& x, double tol) {
  const Point u = b - a;
  const double s = dot(x - a, u) / dot(u, u);
  if (dist(x, a + s * u) > tol || s < -tol || s > 1 + tol) return 1e10;
  return 2 * s - 1;
}

// Coordinates (s, t) of x in the plane of a face with corner positions f0, f1, f2:
// x = f0 + s (f1 - f0) + t (f2 - f0). Returns false when x is off the plane.
bool plane_coords(const Point& f0, const Point& f1, const Point& f2, const Point& x, double tol,
                  double& s, double& t) {
  const Point u = f1 - f0;
  const Point v = f2 - f0;
  const Point w = x - f0;
  const double uu = dot(u, u), uv = dot(u, v), vv = dot(v, v);
  const double det = uu * vv - uv * uv;
  s = (dot(w, u) * vv - dot(w, v) * uv) / det;
  t = (dot(w, v) * uu - dot(w, u) * uv) / det;
  return dist(x, f0 + s * u + t * v) <= tol;
}

// Barycentric weights of a point in the reference simplex (both use -1 + 2 lambda).
std::array<double, 4> simplex_bary(Topology t, const Point& x) {
  const int d = soma::dimension(t);
  std::array<double, 4> l{0, 0, 0, 0};
  double sum = 0;
  for (int i = 0; i < d; ++i) {
    l[sz(i + 1)] = (1 + x[sz(i)]) / 2;
    sum += l[sz(i + 1)];
  }
  l[0] = 1 - sum;
  return l;
}

Point from_bary(Topology t, const std::array<double, 4>& l, const std::array<int, 4>& perm) {
  const std::vector<Point> v = reference_corners(t);
  Point x{0, 0, 0};
  for (int i = 0; i <= soma::dimension(t); ++i) x = x + l[sz(i)] * v[sz(perm[sz(i)])];
  return x;
}

// ---- Lebesgue constants (orthogonal Jacobi basis, linear solve, sampling) ----

Wide jacobi(int n, int alpha, Wide x) {  // P_n^{(alpha,0)}, orthonormalized on [-1,1]
  Wide p0 = 1;
  if (n == 0) return p0 / std::sqrt(static_cast<Wide>(2) / (alpha + 1));
  const Wide a = alpha;
  Wide p1 = (a + 2) * x / 2 + a / 2;
  for (int k = 1; k < n; ++k) {
    const Wide kk = k;
    const Wide c = 2 * kk + a;
    const Wide a1 = 2 * (kk + 1) * (kk + a + 1) * c;
    const Wide a2 = (c + 1) * a * a;
    const Wide a3 = c * (c + 1) * (c + 2);
    const Wide a4 = 2 * (kk + a) * kk * (c + 2);
    const Wide p2 = ((a2 + a3 * x) * p1 - a4 * p0) / a1;
    p0 = p1;
    p1 = p2;
  }
  return p1 / std::sqrt(static_cast<Wide>(2) / (2 * n + a + 1));
}

std::vector<Wide> basis_values(Topology t, int p, const Point& x) {
  constexpr Wide eps = 1e-12L;
  std::vector<Wide> out;
  if (t == Topology::tri) {
    const Wide r = x[0], s = x[1];
    const Wide a = std::fabs(1 - s) < eps ? -1 : 2 * (1 + r) / (1 - s) - 1;
    for (int i = 0; i <= p; ++i)
      for (int j = 0; j <= p - i; ++j)
        out.push_back(jacobi(i, 0, a) * jacobi(j, 2 * i + 1, s) * std::pow(1 - s, i));
  } else {
    const Wide r = x[0], s = x[1], u = x[2];
    const Wide a = std::fabs(s + u) < eps ? -1 : -2 * (1 + r) / (s + u) - 1;
    const Wide b = std::fabs(1 - u) < eps ? -1 : 2 * (1 + s) / (1 - u) - 1;
    for (int i = 0; i <= p; ++i)
      for (int j = 0; j <= p - i; ++j)
        for (int k = 0; k <= p - i - j; ++k)
          out.push_back(jacobi(i, 0, a) * jacobi(j, 2 * i + 1, b) * jacobi(k, 2 * i + 2 * j + 2, u) *
                        std::pow(1 - b, i) * std::pow(1 - u, i + j));
  }
  return out;
}

struct LuSolver {
  int n = 0;
  std::vector<Wide> lu;  // V^T, factored
  std::vector<int> piv;

  explicit LuSolver(const std::vector<std::vector<Wide>>& v) : n(static_cast<int>(v.size())) {
    lu.assign(sz(n) * sz(n), 0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) lu[sz(j) * sz(n) + sz(i)] = v[sz(i)][sz(j)];  // transpose
    piv.resize(sz(n));
    for (int c = 0; c < n; ++c) {
      int best = c;
      for (int r = c + 1; r < n; ++r)
        if (std::fabs(lu[sz(r) * sz(n) + sz(c)]) > std::fabs(lu[sz(best) * sz(n) + sz(c)])) best = r;
      piv[sz(c)] = best;
      if (best != c)
        for (int j = 0; j < n; ++j) std::swap(lu[sz(c) * sz(n) + sz(j)], lu[sz(best) * sz(n) + sz(j)]);
      for (int r = c + 1; r < n; ++r) {
        const Wide f = lu[sz(r) * sz(n) + sz(c)] / lu[sz(c) * sz(n) + sz(c)];
        lu[sz(r) * sz(n) + sz(c)] = f;
        for (int j = c + 1; j < n; ++j) lu[sz(r) * sz(n) + sz(j)] -= f * lu[sz(c) * sz(n) + sz(j)];
      }
    }
  }

  void solve(std::vector<Wide>& b) const {
    for (int c = 0; c < n; ++c) std::swap(b[sz(c)], b[sz(piv[sz(c)])]);
    for (int c = 0; c < n; ++c) {
      for (int r = c + 1; r < n; ++r) b[sz(r)] -= lu[sz(r) * sz(n) + sz(c)] * b[sz(c)];
    }
    for (int r = n - 1; r >= 0; --r) {
      for (int j = r + 1; j < n; ++j) b[sz(r)] -= lu[sz(r) * sz(n) + sz(j)] * b[sz(j)];
      b[sz(r)] /= lu[sz(r) * sz(n) + sz(r)];
    }
  }
};

// Sum of |Lagrange basis values| at x. A function, not a lambda inside
// lebesgue(): nvcc inspects lambda bodies in its device pass and warns that
// long double is treated as double there (#20208-D), host-only or not.
Wide lebesgue_sum(const LuSolver& solver, Topology t, int p, const Point& x) {
  std::vector<Wide> psi = basis_values(t, p, x);
  solver.solve(psi);
  Wide sum = 0;
  for (const Wide c : psi) sum += std::fabs(c);
  return sum;
}

double lebesgue(Topology t, int p, const std::vector<Point>& nodes) {
  std::vector<std::vector<Wide>> v;
  for (const Point& x : nodes) v.push_back(basis_values(t, p, x));
  const LuSolver solver(v);
  const int res = (t == Topology::tri ? 12 : 4) * p;
  Wide best = 0;
  const double h = 2.0 / res;
  if (t == Topology::tri) {
    for (int j = 0; j <= res; ++j)
      for (int i = 0; i <= res - j; ++i)
        best = std::max(best, lebesgue_sum(solver, t, p, {-1 + h * i, -1 + h * j, 0}));
  } else {
    for (int k = 0; k <= res; ++k)
      for (int j = 0; j <= res - k; ++j)
        for (int i = 0; i <= res - j - k; ++i)
          best = std::max(best, lebesgue_sum(solver, t, p, {-1 + h * i, -1 + h * j, -1 + h * k}));
  }
  return static_cast<double>(best);
}

// ---- device proof ----

using Mem = Kokkos::DefaultExecutionSpace::memory_space;

// Explicit return type: CUDA forbids extended lambdas in functions with a deduced return type.
Kokkos::View<double*, Kokkos::HostSpace> ends_from_default_space(
    const soma::NodeTableView<Mem>& table, const soma::NodeTables<Mem>::Permutation& perm) {
  Kokkos::View<double*> out("soma::test::node_table_ends", 8);
  const soma::NodeTableView<Mem> t = table;
  const soma::NodeTables<Mem>::Permutation q = perm;
  Kokkos::parallel_for(
      "soma::test::node_table_ends", 1, KOKKOS_LAMBDA(const int) {
        const int last = t.count - 1;
        out(0) = t(0, 0);
        out(1) = t(0, 1);
        out(2) = t(0, 2);
        out(3) = t(last, 0);
        out(4) = t(last, 1);
        out(5) = t(last, 2);
        out(6) = static_cast<double>(q(0));
        out(7) = static_cast<double>(q(last));
      });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
}

}  // namespace soma_test

using namespace soma_test;

TEST(NodeTables, Gll1DIsSymmetricAscendingAndKnown) {
  for (int p = 1; p <= soma::max_node_order; ++p) {
    const std::vector<double> x = soma::gll_points_1d(p);
    ASSERT_EQ(static_cast<int>(x.size()), p + 1);
    EXPECT_EQ(x.front(), -1.0);
    EXPECT_EQ(x.back(), 1.0);
    for (int i = 0; i < p; ++i) EXPECT_LT(x[sz(i)], x[sz(i) + 1]) << "p=" << p;
    for (int i = 0; i <= p; ++i) EXPECT_EQ(x[sz(i)], -x[sz(p - i)]) << "p=" << p;
  }
  const auto x3 = soma::gll_points_1d(3);
  EXPECT_NEAR(x3[1], -1 / std::sqrt(5.0), 1e-15);
  const auto x4 = soma::gll_points_1d(4);
  EXPECT_NEAR(x4[1], -std::sqrt(3.0 / 7.0), 1e-15);
  const auto x5 = soma::gll_points_1d(5);
  EXPECT_NEAR(x5[1], -std::sqrt(1.0 / 3.0 + 2 * std::sqrt(7.0) / 21.0), 1e-15);
}

TEST(NodeTables, Gll1DAreRootsOfLegendreDerivative) {
  for (int p = 2; p <= soma::max_node_order; ++p) {
    const std::vector<double> x = soma::gll_points_1d(p);
    for (int i = 1; i < p; ++i) {
      // P'_p(x) from the three-term recurrence, in long double
      const Wide t = x[sz(i)];
      Wide p0 = 1, p1 = t, d0 = 0, d1 = 1;
      for (int k = 1; k < p; ++k) {
        const Wide kk = k;
        const Wide p2 = ((2 * kk + 1) * t * p1 - kk * p0) / (kk + 1);
        const Wide d2 = d0 + (2 * kk + 1) * p1;
        p0 = p1;
        p1 = p2;
        d0 = d1;
        d1 = d2;
      }
      EXPECT_LT(std::fabs(static_cast<double>(d1)) / (p * (p + 1) / 2.0), 1e-13)
          << "p=" << p << " i=" << i;
    }
  }
}

TEST(NodeTables, Gll1DRejectsOrdersOutsideTheRange) {
  EXPECT_THROW(soma::gll_points_1d(0), std::invalid_argument);
  EXPECT_THROW(soma::gll_points_1d(soma::max_node_order + 1), std::invalid_argument);
}

TEST(NodeTables, NodeCountsMatchTheFormulas) {
  for (const Topology t : shapes) {
    for (int p = 1; p <= 12; ++p) {
      for (const NodeFamily f : {NodeFamily::lagrange_equispaced, NodeFamily::gll}) {
        for (const NodeOrdering o : {NodeOrdering::tensor_lexicographic, NodeOrdering::hierarchical}) {
          const NodeTableHost table = soma::make_node_table(key(t, p, f, o));
          EXPECT_EQ(table.count, expected_count(t, p)) << soma::name(t) << " p=" << p;
          EXPECT_EQ(table.dim, soma::dimension(t));
          EXPECT_EQ(static_cast<int>(table.x.size()), table.count * table.dim);
        }
      }
    }
  }
}

TEST(NodeTables, SerendipityAndPyramidCounts) {
  const auto count = [](Topology t, int p, bool ser, NodeOrdering o) {
    return soma::make_node_table(key(t, p, NodeFamily::lagrange_equispaced, o, ser)).count;
  };
  EXPECT_EQ(count(Topology::quad, 2, true, NodeOrdering::moab_native), 8);
  EXPECT_EQ(count(Topology::hex, 2, true, NodeOrdering::moab_native), 20);
  EXPECT_EQ(count(Topology::prism, 2, true, NodeOrdering::moab_native), 15);
  EXPECT_EQ(count(Topology::pyramid, 2, true, NodeOrdering::moab_native), 13);
  EXPECT_EQ(count(Topology::pyramid, 2, false, NodeOrdering::moab_native), 14);
  EXPECT_EQ(count(Topology::pyramid, 1, false, NodeOrdering::hierarchical), 5);
  EXPECT_EQ(count(Topology::quad, 2, true, NodeOrdering::tensor_lexicographic), 8);
  EXPECT_EQ(count(Topology::hex, 2, true, NodeOrdering::hierarchical), 20);
}

TEST(NodeTables, PyramidGllCoincidesWithEquispacedWhereTheNodesCoincide) {
  const auto table = [](int p, NodeFamily f, NodeOrdering o, bool ser) {
    return soma::make_node_table(key(Topology::pyramid, p, f, o, ser));
  };
  for (const NodeOrdering o : {NodeOrdering::hierarchical, NodeOrdering::moab_native}) {
    const NodeTableHost a = table(2, NodeFamily::lagrange_equispaced, o, true);
    const NodeTableHost b = table(2, NodeFamily::gll, o, true);
    ASSERT_EQ(a.count, 13);
    ASSERT_EQ(b.count, 13);
    EXPECT_EQ(a.x, b.x);
    const NodeTableHost c = table(1, NodeFamily::lagrange_equispaced, o, false);
    const NodeTableHost d = table(1, NodeFamily::gll, o, false);
    ASSERT_EQ(d.count, 5);
    EXPECT_EQ(c.x, d.x);
  }
  EXPECT_THROW(table(2, NodeFamily::gll, NodeOrdering::moab_native, false), std::invalid_argument);
  EXPECT_THROW(table(3, NodeFamily::gll, NodeOrdering::hierarchical, false), std::invalid_argument);
}

TEST(NodeTables, UnsupportedCombinationsThrowAndNameTheCombination) {
  const auto message = [](const NodeTableKey& k) {
    try {
      (void)soma::make_node_table(k);
    } catch (const std::invalid_argument& e) {
      return std::string(e.what());
    }
    return std::string("no exception");
  };
  const std::string gauss = message(key(Topology::quad, 3, NodeFamily::gauss));
  EXPECT_NE(gauss.find("gauss"), std::string::npos) << gauss;
  EXPECT_NE(gauss.find("quad"), std::string::npos) << gauss;
  EXPECT_NE(message(key(Topology::pyramid, 2, NodeFamily::gll, NodeOrdering::hierarchical)).find("pyramid"),
            std::string::npos);
  EXPECT_NE(message(key(Topology::pyramid, 3, NodeFamily::lagrange_equispaced,
                        NodeOrdering::hierarchical)).find("pyramid"),
            std::string::npos);
  EXPECT_NE(message(key(Topology::pyramid, 2, NodeFamily::lagrange_equispaced)).find("pyramid"),
            std::string::npos);
  EXPECT_NE(message(key(Topology::hex, 3, NodeFamily::gll, NodeOrdering::moab_native)).find("moab_native"),
            std::string::npos);
  EXPECT_NE(message(key(Topology::polygon, 1)).find("polygon"), std::string::npos);
  EXPECT_NE(message(key(Topology::knife, 1)).find("knife"), std::string::npos);
  EXPECT_NE(message(key(Topology::tri, 2, NodeFamily::gll, NodeOrdering::hierarchical, true))
                .find("serendipity"),
            std::string::npos);
  EXPECT_NE(message(key(Topology::hex, 3, NodeFamily::gll, NodeOrdering::hierarchical, true))
                .find("serendipity"),
            std::string::npos);
  EXPECT_NE(message(key(Topology::tri, 0)).find("order"), std::string::npos);
  EXPECT_NE(message(key(Topology::tri, soma::max_node_order + 1)).find("order"), std::string::npos);
}

TEST(NodeTables, ReferenceCornersFollowTheCanonicalVertexOrder) {
  for (const Topology t : {Topology::edge, Topology::tri, Topology::quad, Topology::tet,
                           Topology::prism, Topology::hex}) {
    for (const NodeOrdering o : {NodeOrdering::hierarchical, NodeOrdering::moab_native}) {
      const NodeTableHost table = soma::make_node_table(key(t, 2, NodeFamily::gll, o));
      const std::vector<Point> c = reference_corners(t);
      for (int v = 0; v < soma::num_vertices(t); ++v) {
        EXPECT_LE(dist(point(table, v), c[sz(v)]), 1e-15) << soma::name(t) << " vertex " << v;
      }
    }
  }
  for (const bool ser : {false, true}) {
    const NodeTableHost table = soma::make_node_table(
        key(Topology::pyramid, 2, NodeFamily::lagrange_equispaced, NodeOrdering::moab_native, ser));
    const std::vector<Point> c = reference_corners(Topology::pyramid);
    for (int v = 0; v < 5; ++v) EXPECT_LE(dist(point(table, v), c[sz(v)]), 1e-15);
  }
}

// The corner convention and the canonical side tables must agree: every canonical
// edge joins two corners that are an element edge, and every canonical face is
// wound so that its normal points out of the element.
TEST(NodeTables, ReferenceCornersAreConsistentWithTheCanonicalSides) {
  for (const Topology t : {Topology::tet, Topology::pyramid, Topology::prism, Topology::hex}) {
    const std::vector<Point> c = reference_corners(t);
    const Point mid = centroid(c);
    for (int f = 0; f < canonical::num_sides(t, 2); ++f) {
      const canonical::SideDef s = canonical::side(t, 2, f);
      const Point n = cross(c[s.v[1]] - c[s.v[0]], c[s.v[2]] - c[s.v[0]]);
      EXPECT_GT(dot(n, average(c, s) - mid), 1e-12) << soma::name(t) << " face " << f;
      for (int q = 3; q < s.num_vertices; ++q) {
        EXPECT_NEAR(dot(n, c[s.v[q]] - c[s.v[0]]), 0.0, 1e-14) << soma::name(t) << " face " << f;
      }
    }
  }
  for (const Topology t : {Topology::tri, Topology::quad}) {
    const std::vector<Point> c = reference_corners(t);
    for (int i = 0; i < soma::num_vertices(t); ++i) {
      const Point a = c[sz(i)], b = c[sz((i + 1) % soma::num_vertices(t))];
      const canonical::SideDef e = canonical::side(t, 1, i);
      EXPECT_EQ(dist(c[e.v[0]], a), 0.0);
      EXPECT_EQ(dist(c[e.v[1]], b), 0.0);
      const Point n = cross(b - a, {0, 0, 1});
      EXPECT_GT(dot(n, 0.5 * (a + b) - centroid(c)), 0.0) << soma::name(t);
    }
  }
}

TEST(NodeTables, EquispacedNodesAreTheUniformLattice) {
  const NodeTableHost tri = soma::make_node_table(
      key(Topology::tri, 3, NodeFamily::lagrange_equispaced));
  const std::array<std::array<double, 2>, 10> expect = {{{-1, -1}, {-1.0 / 3, -1}, {1.0 / 3, -1},
                                                         {1, -1}, {-1, -1.0 / 3}, {-1.0 / 3, -1.0 / 3},
                                                         {1.0 / 3, -1.0 / 3}, {-1, 1.0 / 3},
                                                         {-1.0 / 3, 1.0 / 3}, {-1, 1}}};
  ASSERT_EQ(tri.count, 10);
  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR(tri.coord(i, 0), expect[sz(i)][0], 1e-15);
    EXPECT_NEAR(tri.coord(i, 1), expect[sz(i)][1], 1e-15);
  }
}

TEST(NodeTables, SecondOrderPositionsAreSubEntityMidpoints) {
  for (const Topology t : shapes) {
    for (const NodeFamily f : {NodeFamily::lagrange_equispaced, NodeFamily::gll}) {
      const NodeTableHost table =
          soma::make_node_table(key(t, 2, f, NodeOrdering::hierarchical));
      const std::vector<Point> c = reference_corners(t);
      std::vector<Point> expect = c;
      for (int e = 0; e < canonical::num_sides(t, 1); ++e) expect.push_back(average(c, canonical::side(t, 1, e)));
      if (soma::dimension(t) == 3) {
        for (int fa = 0; fa < canonical::num_sides(t, 2); ++fa) {
          const canonical::SideDef s = canonical::side(t, 2, fa);
          if (s.num_vertices == 4) expect.push_back(average(c, s));
        }
        if (t == Topology::hex) expect.push_back(centroid(c));
      }
      if (t == Topology::quad) expect.push_back(centroid(c));
      ASSERT_EQ(static_cast<int>(expect.size()), table.count) << soma::name(t);
      for (int i = 0; i < table.count; ++i) {
        EXPECT_LE(dist(point(table, i), expect[sz(i)]), 1e-15) << soma::name(t) << " node " << i;
      }
    }
  }
}

TEST(NodeTables, TensorLexicographicRunsWithIFastest) {
  const NodeTableHost quad = soma::make_node_table(key(Topology::quad, 2, NodeFamily::gll));
  EXPECT_EQ(quad.coord(1, 0), 0.0);
  EXPECT_EQ(quad.coord(1, 1), -1.0);
  EXPECT_EQ(quad.coord(3, 0), -1.0);
  EXPECT_EQ(quad.coord(3, 1), 0.0);
  const NodeTableHost tet = soma::make_node_table(
      key(Topology::tet, 2, NodeFamily::lagrange_equispaced));
  ASSERT_EQ(tet.count, 10);
  EXPECT_LE(dist(point(tet, 1), {0, -1, -1}), 1e-15);   // i along v0 -> v1
  EXPECT_LE(dist(point(tet, 3), {-1, 0, -1}), 1e-15);   // then j along v0 -> v2
  EXPECT_LE(dist(point(tet, 6), {-1, -1, 0}), 1e-15);   // then k along v0 -> v3
  const NodeTableHost prism = soma::make_node_table(
      key(Topology::prism, 1, NodeFamily::lagrange_equispaced));
  ASSERT_EQ(prism.count, 6);
  for (int i = 0; i < 3; ++i) EXPECT_EQ(prism.coord(i, 2), -1.0);   // triangle index fastest
  for (int i = 3; i < 6; ++i) EXPECT_EQ(prism.coord(i, 2), 1.0);
}

TEST(NodeTables, GllTriangleAndTetRestrictToTheEdgeNodes) {
  for (const Topology t : {Topology::tri, Topology::tet, Topology::prism}) {
    for (int p = 1; p <= 20; ++p) {
      const std::vector<Point> nodes = table_points(key(t, p));
      const std::vector<Point> c = reference_corners(t);
      const std::vector<double> line = soma::gll_points_1d(p);
      for (int e = 0; e < canonical::num_sides(t, 1); ++e) {
        const canonical::SideDef s = canonical::side(t, 1, e);
        std::vector<double> found;
        for (const Point& x : nodes) {
          const double a = along(c[s.v[0]], c[s.v[1]], x, 1e-12);
          if (a < 1e9) found.push_back(a);
        }
        std::sort(found.begin(), found.end());
        ASSERT_EQ(static_cast<int>(found.size()), p + 1) << soma::name(t) << " p=" << p << " edge " << e;
        for (int i = 0; i <= p; ++i) {
          EXPECT_NEAR(found[sz(i)], line[sz(i)], 1e-14)
              << soma::name(t) << " p=" << p << " edge " << e << " node " << i;
        }
      }
    }
  }
}

TEST(NodeTables, GllFacesCarryTheTriangleAndQuadSets) {
  for (const Topology t : {Topology::tet, Topology::prism, Topology::hex}) {
    for (int p = 1; p <= 15; ++p) {
      const std::vector<Point> nodes = table_points(key(t, p));
      const std::vector<Point> c = reference_corners(t);
      const std::vector<Point> tri = table_points(key(Topology::tri, p));
      const std::vector<Point> quad = table_points(key(Topology::quad, p));
      for (int f = 0; f < canonical::num_sides(t, 2); ++f) {
        const canonical::SideDef s = canonical::side(t, 2, f);
        const bool is_tri = s.num_vertices == 3;
        const Point f0 = c[s.v[0]], f1 = c[s.v[1]], f2 = c[s.v[is_tri ? 2 : 3]];
        std::vector<Point> on_face;
        for (const Point& x : nodes) {
          double a = 0, b = 0;
          if (plane_coords(f0, f1, f2, x, 1e-12, a, b)) {
            on_face.push_back(is_tri ? Point{-1, -1, 0} + 2 * a * Point{1, 0, 0} + 2 * b * Point{0, 1, 0}
                                     : Point{-1 + 2 * a, -1 + 2 * b, 0});
          }
        }
        EXPECT_TRUE(same_set(on_face, is_tri ? tri : quad, 1e-14))
            << soma::name(t) << " p=" << p << " face " << f;
      }
    }
  }
}

TEST(NodeTables, SimplexNodeSetsAreInvariantUnderVertexPermutations) {
  for (const Topology t : {Topology::tri, Topology::tet}) {
    const int nv = soma::num_vertices(t);
    for (int p = 1; p <= 15; ++p) {
      const std::vector<Point> nodes = table_points(key(t, p));
      std::array<int, 4> perm{0, 1, 2, 3};
      std::vector<int> head(perm.begin(), perm.begin() + nv);
      do {
        std::array<int, 4> pm{0, 1, 2, 3};
        for (int i = 0; i < nv; ++i) pm[sz(i)] = head[sz(i)];
        std::vector<Point> mapped;
        for (const Point& x : nodes) mapped.push_back(from_bary(t, simplex_bary(t, x), pm));
        EXPECT_TRUE(same_set(mapped, nodes, 1e-14)) << soma::name(t) << " p=" << p;
      } while (std::next_permutation(head.begin(), head.end()));
    }
  }
}

TEST(NodeTables, PrismAndTensorNodeSetsAreInvariantUnderTheirSymmetries) {
  for (int p = 1; p <= 10; ++p) {
    const std::vector<Point> prism = table_points(key(Topology::prism, p));
    std::array<int, 4> pm{0, 1, 2, 3};
    do {
      for (const double flip : {1.0, -1.0}) {
        std::vector<Point> mapped;
        for (const Point& x : prism) {
          Point q = from_bary(Topology::tri, simplex_bary(Topology::tri, {x[0], x[1], 0}), pm);
          q[2] = flip * x[2];
          mapped.push_back(q);
        }
        EXPECT_TRUE(same_set(mapped, prism, 1e-14)) << "prism p=" << p;
      }
    } while (std::next_permutation(pm.begin(), pm.begin() + 3));
    const std::vector<Point> hex = table_points(key(Topology::hex, p));
    std::array<int, 3> ax{0, 1, 2};
    do {
      for (int signs = 0; signs < 8; ++signs) {
        std::vector<Point> mapped;
        for (const Point& x : hex) {
          Point q{0, 0, 0};
          for (int d = 0; d < 3; ++d) q[sz(d)] = ((signs >> d) & 1 ? -1.0 : 1.0) * x[sz(ax[sz(d)])];
          mapped.push_back(q);
        }
        EXPECT_TRUE(same_set(mapped, hex, 1e-15)) << "hex p=" << p;
      }
    } while (std::next_permutation(ax.begin(), ax.end()));
  }
}

TEST(NodeTables, SerendipityNodesAreTheCornersAndMidEdgeNodes) {
  for (const Topology t : {Topology::quad, Topology::hex, Topology::prism}) {
    for (const NodeOrdering o : {NodeOrdering::tensor_lexicographic, NodeOrdering::hierarchical,
                                 NodeOrdering::moab_native}) {
      const std::vector<Point> ser = table_points(key(t, 2, NodeFamily::gll, o, true));
      const std::vector<Point> c = reference_corners(t);
      std::vector<Point> expect = c;
      for (int e = 0; e < canonical::num_sides(t, 1); ++e) expect.push_back(average(c, canonical::side(t, 1, e)));
      EXPECT_TRUE(same_set(ser, expect, 1e-15)) << soma::name(t);
    }
  }
}

TEST(NodeTables, OrderingsArePermutationsOfEachOther) {
  struct Case {
    Topology t;
    int p;
    bool ser;
  };
  std::vector<Case> cases;
  for (const Topology t : shapes) {
    for (int p = 1; p <= 6; ++p) cases.push_back({t, p, false});
  }
  for (const Topology t : {Topology::quad, Topology::hex, Topology::prism}) cases.push_back({t, 2, true});
  const NodeOrdering all[] = {NodeOrdering::tensor_lexicographic, NodeOrdering::hierarchical,
                              NodeOrdering::moab_native};
  for (const Case& cs : cases) {
    for (const NodeFamily f : {NodeFamily::lagrange_equispaced, NodeFamily::gll}) {
      for (const NodeOrdering from : all) {
        for (const NodeOrdering to : all) {
          if (cs.p > 2 && (from == NodeOrdering::moab_native || to == NodeOrdering::moab_native)) continue;
          const NodeTableKey kf = key(cs.t, cs.p, f, from, cs.ser);
          const NodeTableKey kt = key(cs.t, cs.p, f, to, cs.ser);
          const NodeTableHost a = soma::make_node_table(kf);
          const NodeTableHost b = soma::make_node_table(kt);
          const std::vector<int> perm = soma::make_node_permutation(kf, to);
          ASSERT_EQ(static_cast<int>(perm.size()), b.count);
          std::vector<int> sorted = perm;
          std::sort(sorted.begin(), sorted.end());
          for (int i = 0; i < a.count; ++i) ASSERT_EQ(sorted[sz(i)], i) << "not a bijection";
          for (int i = 0; i < b.count; ++i) {
            for (int d = 0; d < b.dim; ++d) {
              ASSERT_EQ(b.coord(i, d), a.coord(perm[sz(i)], d))
                  << soma::name(cs.t) << " p=" << cs.p << " node " << i;
            }
          }
        }
      }
    }
  }
}

TEST(NodeTables, HierarchicalInteriorCountsAndPlacement) {
  for (const Topology t : shapes) {
    if (soma::dimension(t) == 1) continue;
    for (int p = 2; p <= 6; ++p) {
      for (const NodeFamily f : {NodeFamily::lagrange_equispaced, NodeFamily::gll}) {
        const NodeTableHost table = soma::make_node_table(key(t, p, f, NodeOrdering::hierarchical));
        const std::vector<Point> c = reference_corners(t);
        const std::vector<double> line = soma::gll_points_1d(p);
        int next = soma::num_vertices(t);
        for (int e = 0; e < canonical::num_sides(t, 1); ++e) {
          const canonical::SideDef s = canonical::side(t, 1, e);
          for (int i = 1; i < p; ++i, ++next) {
            const double u = f == NodeFamily::gll ? line[sz(i)] : -1.0 + 2.0 * i / p;
            const Point expect = c[s.v[0]] + 0.5 * (1 + u) * (c[s.v[1]] - c[s.v[0]]);
            ASSERT_LE(dist(point(table, next), expect), 1e-14)
                << soma::name(t) << " p=" << p << " edge " << e << " node " << i;
          }
        }
        if (soma::dimension(t) == 3) {
          const std::vector<Point> tri = table_points(key(Topology::tri, p, f));
          const std::vector<Point> quad = table_points(key(Topology::quad, p, f));
          for (int fa = 0; fa < canonical::num_sides(t, 2); ++fa) {
            const canonical::SideDef s = canonical::side(t, 2, fa);
            const bool is_tri = s.num_vertices == 3;
            const Point f0 = c[s.v[0]], f1 = c[s.v[1]], f2 = c[s.v[is_tri ? 2 : 3]];
            const std::vector<Point>& face = is_tri ? tri : quad;
            int interior = 0;
            for (const Point& r : face) {
              const double a = (1 + r[0]) / 2, b = (1 + r[1]) / 2;
              const bool boundary = a < 1e-12 || b < 1e-12 ||
                                    (is_tri ? (1 - a - b) < 1e-12 : (a > 1 - 1e-12 || b > 1 - 1e-12));
              if (boundary) continue;
              const Point expect = f0 + a * (f1 - f0) + b * (f2 - f0);
              ASSERT_LE(dist(point(table, next), expect), 1e-14)
                  << soma::name(t) << " p=" << p << " face " << fa << " node " << interior;
              ++next;
              ++interior;
            }
            EXPECT_EQ(interior, is_tri ? (p - 1) * (p - 2) / 2 : (p - 1) * (p - 1));
          }
        }
        const int cell = interior_count(t, p);
        EXPECT_EQ(table.count - next, cell) << soma::name(t) << " p=" << p;
        if (t == Topology::hex || t == Topology::quad) {
          // interior block is the tensor interior in lexicographic order
          const std::vector<Point> lex = table_points(key(t, p, f));
          std::vector<Point> inner;
          for (const Point& x : lex) {
            if (std::fabs(x[0]) < 1 - 1e-12 && std::fabs(x[1]) < 1 - 1e-12 &&
                (t == Topology::quad || std::fabs(x[2]) < 1 - 1e-12)) {
              inner.push_back(x);
            }
          }
          for (int i = 0; i < cell; ++i) {
            ASSERT_LE(dist(point(table, next + i), inner[sz(i)]), 1e-15)
                << soma::name(t) << " p=" << p << " interior " << i;
          }
        }
      }
    }
  }
}

TEST(NodeTables, MoabNativeFollowsTheSecondOrderLayout) {
  struct Case {
    Topology t;
    bool ser;
    int count;
  };
  const Case cases[] = {{Topology::edge, false, 3},  {Topology::tri, false, 6},
                        {Topology::quad, true, 8},   {Topology::quad, false, 9},
                        {Topology::tet, false, 10},  {Topology::pyramid, true, 13},
                        {Topology::pyramid, false, 14}, {Topology::prism, true, 15},
                        {Topology::prism, false, 18}, {Topology::hex, true, 20},
                        {Topology::hex, false, 27}};
  for (const Case& cs : cases) {
    const NodeTableKey k = key(cs.t, 2, NodeFamily::lagrange_equispaced, NodeOrdering::moab_native, cs.ser);
    const NodeTableHost table = soma::make_node_table(k);
    ASSERT_EQ(table.count, cs.count) << soma::name(cs.t);
    const std::vector<Point> c = reference_corners(cs.t);
    std::vector<Point> expect = c;
    for (int e = 0; e < canonical::num_sides(cs.t, 1); ++e) expect.push_back(average(c, canonical::side(cs.t, 1, e)));
    if (!cs.ser && cs.t == Topology::pyramid) {
      expect.push_back(centroid(c));  // mid-region node, as in MOAB's Pyramid14
    } else if (!cs.ser) {
      for (int f = 0; f < canonical::num_sides(cs.t, 2); ++f) {
        const canonical::SideDef s = canonical::side(cs.t, 2, f);
        if (s.num_vertices == 4) expect.push_back(average(c, s));
      }
      if (cs.t == Topology::hex) expect.push_back(centroid(c));
    }
    ASSERT_EQ(static_cast<int>(expect.size()), cs.count) << soma::name(cs.t);
    for (int i = 0; i < cs.count; ++i) {
      EXPECT_LE(dist(point(table, i), expect[sz(i)]), 1e-15) << soma::name(cs.t) << " node " << i;
    }
  }
  const NodeTableHost hex27 = soma::make_node_table(
      key(Topology::hex, 2, NodeFamily::lagrange_equispaced, NodeOrdering::moab_native));
  EXPECT_LE(dist(point(hex27, 8), {0, -1, -1}), 1e-15);
  EXPECT_LE(dist(point(hex27, 20), {0, -1, 0}), 1e-15);
  EXPECT_LE(dist(point(hex27, 25), {0, 0, 1}), 1e-15);
  EXPECT_LE(dist(point(hex27, 26), {0, 0, 0}), 1e-15);
}

TEST(NodeTables, LebesgueConstantsOfTheEquispacedTriangleMatchThePublishedValues) {
  // Pasquetti and Rapetti, Numer. Algor. 55 (2010), Table 1, "Uniform" column.
  const struct {
    int p;
    double published;
  } rows[] = {{3, 2.27}, {6, 8.75}, {9, 40.92}, {12, 221.41}, {15, 1315.89}};
  for (const auto& r : rows) {
    const double got =
        lebesgue(Topology::tri, r.p, table_points(key(Topology::tri, r.p, NodeFamily::lagrange_equispaced)));
    EXPECT_NEAR(got, r.published, 0.02 * r.published) << "p=" << r.p;
  }
}

TEST(NodeTables, LebesgueConstantsOfTheGllTriangleMatchThePublishedValues) {
  // Same table, "warp & blend" column (Warburton's nodes with the optimized alpha).
  const struct {
    int p;
    double published;
  } rows[] = {{3, 2.11}, {6, 3.70}, {9, 5.74}, {12, 9.36}, {15, 17.65}};
  for (const auto& r : rows) {
    const double got = lebesgue(Topology::tri, r.p, table_points(key(Topology::tri, r.p)));
    EXPECT_NEAR(got, r.published, 0.005 * r.published) << "p=" << r.p;
  }
}

// The published table covers only some orders, so every order is pinned to this
// implementation's own values (which agree with the published rows above). A
// wrong optimized parameter for any single order moves its constant.
TEST(NodeTables, LebesgueConstantsOfTheGllTriangleAreAPinnedSnapshot) {
  constexpr double snapshot[] = {1.0000, 1.6667, 2.1110, 2.6622, 3.1211, 3.6940, 4.2724, 4.9629,
                                 5.7321, 6.6630, 7.9029, 9.3530, 11.4298, 13.9698, 17.6230};
  for (int p = 1; p <= soma::max_optimized_simplex_order; ++p) {
    const double got = lebesgue(Topology::tri, p, table_points(key(Topology::tri, p)));
    EXPECT_NEAR(got, snapshot[p - 1], 0.003 * snapshot[p - 1]) << "p=" << p;
  }
}

// No published tetrahedron table could be retrieved, so these values are a
// snapshot of this implementation (tetrahedron faces carry the triangle node
// set), not a literature comparison. The monotone and bounded checks are the
// independent sanity guard.
TEST(NodeTables, LebesgueConstantsOfTheTetAreAPinnedSnapshotWithSanityBounds) {
  constexpr double snapshot[] = {1.0000, 2.0000, 2.9317,  4.0213,  5.1815,
                                 6.7706, 8.7367, 12.1084, 16.8764, 24.2455};
  double previous = 0;
  for (int p = 1; p <= 10; ++p) {
    const double gll = lebesgue(Topology::tet, p, table_points(key(Topology::tet, p)));
    EXPECT_NEAR(gll, snapshot[p - 1], 0.0005 * snapshot[p - 1]) << "p=" << p;
    if (p >= 3) {
      const double equi = lebesgue(Topology::tet, p,
                                   table_points(key(Topology::tet, p, NodeFamily::lagrange_equispaced)));
      EXPECT_LT(gll, equi) << "p=" << p;
    }
    EXPECT_LE(gll, 0.25 * p * p + 3.0) << "p=" << p;
    EXPECT_GE(gll, previous) << "p=" << p;
    previous = gll;
  }
}

// Sum of |x|^2 over the tetrahedron nodes, pinned for every order the optimized
// parameter table covers: it moves with every interior node, so a wrong
// parameter for any single order shows here even where the sampled Lebesgue
// constant barely changes.
TEST(NodeTables, TetSecondMomentIsAPinnedSnapshot) {
  constexpr double snapshot[] = {12.0,       21.0,       37.6,       62.036475,  95.606582,
                                 139.945027, 196.048410, 265.707600, 349.992032, 450.715919,
                                 568.343566, 705.622772, 863.004125, 1042.510787, 1244.809785};
  for (int p = 1; p <= soma::max_optimized_simplex_order; ++p) {
    double moment = 0;
    for (const Point& x : table_points(key(Topology::tet, p))) moment += dot(x, x);
    EXPECT_NEAR(moment, snapshot[p - 1], 2e-6 * snapshot[p - 1]) << "p=" << p;
  }
}

TEST(NodeTables, TablesAreDeterministic) {
  for (const Topology t : shapes) {
    for (int p : {1, 2, 5, 9}) {
      const NodeTableKey k = key(t, p);
      const NodeTableHost a = soma::make_node_table(k);
      const NodeTableHost b = soma::make_node_table(k);
      ASSERT_EQ(a.x.size(), b.x.size());
      EXPECT_EQ(std::memcmp(a.x.data(), b.x.data(), a.x.size() * sizeof(double)), 0)
          << soma::name(t) << " p=" << p;
    }
  }
  const auto g1 = soma::gll_points_1d(17);
  const auto g2 = soma::gll_points_1d(17);
  EXPECT_EQ(std::memcmp(g1.data(), g2.data(), g1.size() * sizeof(double)), 0);
}

TEST(NodeTables, DeviceCacheReturnsTheHostTableOnTheDefaultSpace) {
  const NodeTableKey k = key(Topology::hex, 3);
  const NodeTableHost host = soma::make_node_table(k);
  const auto table = soma::NodeTables<Mem>::get(k);
  const auto again = soma::NodeTables<Mem>::get(k);
  EXPECT_EQ(table.x.data(), again.x.data());
  EXPECT_EQ(table.count, host.count);
  EXPECT_EQ(table.dim, 3);
  const auto perm = soma::NodeTables<Mem>::permutation(k, NodeOrdering::hierarchical);
  const auto ends = soma_test::ends_from_default_space(table, perm);
  const std::vector<int> hperm = soma::make_node_permutation(k, NodeOrdering::hierarchical);
  EXPECT_EQ(ends(0), host.coord(0, 0));
  EXPECT_EQ(ends(1), host.coord(0, 1));
  EXPECT_EQ(ends(2), host.coord(0, 2));
  EXPECT_EQ(ends(3), 1.0);
  EXPECT_EQ(ends(4), 1.0);
  EXPECT_EQ(ends(5), 1.0);
  EXPECT_EQ(ends(0), -1.0);
  EXPECT_EQ(static_cast<int>(ends(6)), hperm.front());
  EXPECT_EQ(static_cast<int>(ends(7)), hperm.back());
}
