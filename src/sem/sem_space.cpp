// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/sem/sem_space.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "soma/core/topology.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/node_tables.hpp"
#include "soma/sem/gll.hpp"

namespace soma::sem {

SemGeometry SemGeometry::sphere(Real radius) {
  if (!(radius > 0) || !std::isfinite(radius)) {
    throw std::invalid_argument("SemGeometry::sphere: radius " + std::to_string(radius) +
                                " is not positive and finite");
  }
  return SemGeometry{Kind::sphere, radius};
}

namespace detail {

namespace {

using Vec = std::array<Real, 3>;

// The reference coordinates of a quad's nodes in MOAB's order: corners, the
// midpoints of edges 0-1, 1-2, 2-3, 3-0, the centre.
constexpr Real node_xi[9] = {-1, 1, 1, -1, 0, 1, 0, -1, 0};
constexpr Real node_eta[9] = {-1, -1, 1, 1, -1, 0, 1, 0, 0};

// The 1D quadratic Lagrange basis at -1, 0, 1, and its derivative.
Real quadratic(Real t, Real at) { return at < 0 ? t * (t - 1) / 2 : at > 0 ? t * (t + 1) / 2 : 1 - t * t; }
Real quadratic_d(Real t, Real at) { return at < 0 ? t - Real{0.5} : at > 0 ? t + Real{0.5} : -2 * t; }

// Shape functions of a quad of n nodes (4 bilinear, 8 serendipity, 9
// Lagrange) and their derivatives in xi and eta at (s, t).
void shape(int n, Real s, Real t, Real* f, Real* fs, Real* ft) {
  for (int a = 0; a < n; ++a) {
    const Real p = node_xi[a], q = node_eta[a];
    if (n == 4) {
      f[a] = (1 + s * p) * (1 + t * q) / 4;
      fs[a] = p * (1 + t * q) / 4;
      ft[a] = q * (1 + s * p) / 4;
    } else if (n == 9) {
      f[a] = quadratic(s, p) * quadratic(t, q);
      fs[a] = quadratic_d(s, p) * quadratic(t, q);
      ft[a] = quadratic(s, p) * quadratic_d(t, q);
    } else if (a < 4) {
      f[a] = (1 + s * p) * (1 + t * q) * (s * p + t * q - 1) / 4;
      fs[a] = p * (1 + t * q) * (2 * s * p + t * q) / 4;
      ft[a] = q * (1 + s * p) * (s * p + 2 * t * q) / 4;
    } else if (p == 0) {
      f[a] = (1 - s * s) * (1 + t * q) / 2;
      fs[a] = -s * (1 + t * q);
      ft[a] = q * (1 - s * s) / 2;
    } else {
      f[a] = (1 + s * p) * (1 - t * t) / 2;
      fs[a] = p * (1 - t * t) / 2;
      ft[a] = -t * (1 + s * p);
    }
  }
}

Vec cross(const Vec& a, const Vec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Real dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

}  // namespace

SemGeometryData sem_geometry(const Mesh<Kokkos::HostSpace>& mesh, int order, const SemGeometry& geometry) {
  SemGeometryData out;
  if (mesh.dim() != 2) {
    out.error = SemGeometryData::Error::dimension;
    return out;
  }
  const bool sphere = geometry.kind == SemGeometry::Kind::sphere;
  if (order < 1 || order > max_node_order || (sphere && (!(geometry.radius > 0) || !std::isfinite(geometry.radius)))) {
    out.error = SemGeometryData::Error::order;
    return out;
  }
  const auto view = mesh.view();
  const auto topology = mesh.topology(2);
  const LO elements = mesh.count(2);
  const int np = order + 1, ppe = np * np;
  bool radius_off = false;
  GO radius_gid = 0;
  for (LO e = 0; e < elements; ++e) {
    const LO n = view.num_nodes(2, e);
    if (sphere && n >= 4) {
      // The geometry's sphere must be the mesh's: corners within 1e-8 of R.
      for (LO q = 0; q < 4; ++q) {
        const LO v = view.vertex(2, e, q);
        const Real r = std::hypot(view.x(v, 0), view.x(v, 1), view.x(v, 2));
        const bool off = !(std::abs(r - geometry.radius) <= 1e-8 * geometry.radius);
        if (off && (!radius_off || view.gid(2, e) < radius_gid)) {
          radius_off = true;
          radius_gid = view.gid(2, e);
        }
      }
    }
    if (topology(e) != static_cast<std::uint8_t>(Topology::quad) || (n != 4 && n != 8 && n != 9)) {
      if (out.error != SemGeometryData::Error::topology || view.gid(2, e) < out.gid) out.gid = view.gid(2, e);
      out.error = SemGeometryData::Error::topology;
    }
  }
  if (out.error != SemGeometryData::Error::none) return out;
  if (radius_off) {
    out.error = SemGeometryData::Error::radius;
    out.gid = radius_gid;
    return out;
  }

  const std::vector<Real> xi = gll_points_1d(order), w = gll_weights_1d(order);
  const auto rows = static_cast<std::size_t>(elements) * static_cast<std::size_t>(ppe);
  using H = View2<Real, Kokkos::HostSpace>;
  out.x = H("soma::sem::SemSpace::x", rows, 3);
  out.covariant = H("soma::sem::SemSpace::covariant", rows, 6);
  out.contravariant = H("soma::sem::SemSpace::contravariant", rows, 6);
  out.jacobian = H("soma::sem::SemSpace::jacobian", rows, 1);
  out.local_mass = H("soma::sem::SemSpace::local_mass", rows, 1);
  for (LO e = 0; e < elements; ++e) {
    const int n = static_cast<int>(view.num_nodes(2, e));
    Vec node[9];
    for (int a = 0; a < n; ++a) {
      const LO v = view.node(2, e, a);
      node[a] = {view.x(v, 0), view.x(v, 1), view.x(v, 2)};
    }
    bool flipped = false;
    for (int j = 0; j < np; ++j) {
      for (int i = 0; i < np; ++i) {
        Real f[9], fs[9], ft[9];
        shape(n, xi[static_cast<std::size_t>(i)], xi[static_cast<std::size_t>(j)], f, fs, ft);
        Vec y{}, a1{}, a2{};
        for (int a = 0; a < n; ++a) {
          for (int k = 0; k < 3; ++k) {
            y[k] += f[a] * node[a][k];
            a1[k] += fs[a] * node[a][k];
            a2[k] += ft[a] * node[a][k];
          }
        }
        Vec x = y, up{0, 0, 1};
        if (sphere) {
          // The derivative of R y / |y|: (R / |y|) (I - u u^T) dy, u = y / |y|.
          const Real l = std::sqrt(dot(y, y)), s = geometry.radius / l;
          const Vec u{y[0] / l, y[1] / l, y[2] / l};
          const Real u1 = dot(u, a1), u2 = dot(u, a2);
          for (int k = 0; k < 3; ++k) {
            x[k] = s * y[k];
            a1[k] = s * (a1[k] - u1 * u[k]);
            a2[k] = s * (a2[k] - u2 * u[k]);
          }
          up = u;
        }
        const Vec normal = cross(a1, a2);
        const Real jac2 = dot(normal, normal), jac = std::sqrt(jac2);
        if (!(dot(normal, up) > 0)) flipped = true;
        const Vec b1 = cross(a2, normal), b2 = cross(normal, a1);
        const auto r = static_cast<std::size_t>(e) * static_cast<std::size_t>(ppe) +
                       static_cast<std::size_t>(i + np * j);
        for (int k = 0; k < 3; ++k) {
          out.x(r, k) = x[k];
          out.covariant(r, k) = a1[k];
          out.covariant(r, 3 + k) = a2[k];
          out.contravariant(r, k) = b1[k] / jac2;
          out.contravariant(r, 3 + k) = b2[k] / jac2;
        }
        out.jacobian(r, 0) = jac;
        out.local_mass(r, 0) = w[static_cast<std::size_t>(i)] * w[static_cast<std::size_t>(j)] * jac;
      }
    }
    if (flipped && (out.error == SemGeometryData::Error::none || view.gid(2, e) < out.gid)) {
      out.error = SemGeometryData::Error::orientation;
      out.gid = view.gid(2, e);
    }
  }
  return out;
}

SemDssPlan sem_dss_plan(const Mesh<Kokkos::HostSpace>& mesh, int order) {
  SemDssPlan plan;
  const auto view = mesh.view();
  const auto gid = mesh.gid(2);
  const auto vgid = mesh.gid(0);
  const LO elements = mesh.count(2);
  const LO p = order, np = order + 1, ppe = np * np;
  const LO owned_vertices = mesh.segment_table(0).owned_end, owned_edges = mesh.segment_table(1).owned_end;
  // Side q runs from corner q to corner q + 1: its points in the element are
  // first[q] + along[q] t for t in 0..p, from that corner.
  const LO first[4] = {0, p, ppe - 1, p * np};
  const LO along[4] = {1, np, -1, -np};
  std::vector<std::vector<std::pair<GO, LO>>> at_vertex(static_cast<std::size_t>(owned_vertices));
  std::vector<std::vector<std::tuple<GO, LO, LO>>> at_edge(static_cast<std::size_t>(owned_edges));
  const auto sides = static_cast<std::size_t>(4) * static_cast<std::size_t>(elements);
  plan.element_vertex.resize(sides);
  plan.element_edge.resize(sides);
  plan.element_base.resize(sides);
  plan.element_step.resize(sides);
  for (LO e = 0; e < elements; ++e) {
    const auto edges = view.adj(2, 1, e);
    if (edges.size != 4) {
      plan.complete = false;
      continue;
    }
    for (int q = 0; q < 4; ++q) {
      const LO c0 = view.vertex(2, e, q), c1 = view.vertex(2, e, (q + 1) % 4);
      LO edge = -1;
      for (LO s = 0; s < 4; ++s) {
        const LO a = view.vertex(1, edges[s], 0), b = view.vertex(1, edges[s], 1);
        if ((a == c0 && b == c1) || (a == c1 && b == c0)) edge = edges[s];
      }
      if (edge < 0) {
        plan.complete = false;
        continue;
      }
      // Points along an edge count from its vertex of smaller gid, which every
      // rank holding the edge agrees on (the stored vertex order may differ).
      const LO a = view.vertex(1, edge, 0), b = view.vertex(1, edge, 1);
      const bool reversed = (vgid(a) < vgid(b) ? a : b) != c0;
      const LO base = e * ppe + (reversed ? first[q] + along[q] * p : first[q]);
      const LO step = reversed ? -along[q] : along[q];
      const auto i = static_cast<std::size_t>(4 * e + q);
      plan.element_vertex[i] = c0;
      plan.element_edge[i] = edge;
      plan.element_base[i] = base;
      plan.element_step[i] = step;
      if (c0 < owned_vertices) at_vertex[static_cast<std::size_t>(c0)].emplace_back(gid(e), e * ppe + first[q]);
      if (edge < owned_edges) at_edge[static_cast<std::size_t>(edge)].emplace_back(gid(e), base, step);
    }
  }
  for (auto& list : at_vertex) {
    std::sort(list.begin(), list.end());
    for (const auto& [g, row] : list) plan.vertex_rows.push_back(row);
    plan.vertex_offsets.push_back(static_cast<Offset>(plan.vertex_rows.size()));
  }
  for (auto& list : at_edge) {
    std::sort(list.begin(), list.end());
    for (const auto& [g, base, step] : list) {
      plan.edge_base.push_back(base);
      plan.edge_step.push_back(step);
    }
    plan.edge_offsets.push_back(static_cast<Offset>(plan.edge_base.size()));
  }
  return plan;
}

void check_sem_geometry(const Comm& comm, const SemGeometryData& data, int dim, int order) {
  const auto worst = comm.allreduce(static_cast<std::int32_t>(data.error), Op::max);
  if (worst == 0) return;
  const GO none = std::numeric_limits<GO>::max();
  const bool mine = static_cast<std::int32_t>(data.error) == worst;
  const auto gid = comm.allreduce(static_cast<std::int64_t>(mine ? data.gid : none), Op::min);
  using E = SemGeometryData::Error;
  switch (static_cast<E>(worst)) {
    case E::dimension:
      throw std::invalid_argument("SemSpace: a rank's mesh is not of dimension 2 (here " + std::to_string(dim) + ")");
    case E::order:
      throw std::invalid_argument("SemSpace: order " + std::to_string(order) + " is outside [1, " +
                                  std::to_string(max_node_order) + "] or the sphere's radius is not positive");
    case E::topology:
      throw std::invalid_argument("SemSpace: element gid " + std::to_string(gid) +
                                  " is not a quad of 4, 8 or 9 nodes");
    case E::edited:
      throw std::invalid_argument("SemSpace: a rank's mesh has removed entities; call compress() first");
    case E::radius:
      throw std::invalid_argument("SemSpace: element gid " + std::to_string(gid) +
                                  " has a corner off the sphere of the geometry's radius: the radius must be the "
                                  "mesh's");
    case E::ghosts:
      throw std::invalid_argument(
          "SemSpace: on more than one rank the mesh needs a ghost layer through vertices and distributed edges: "
          "call mesh.add_ghosts(comm, GhostOptions{1, 0}), then build_distributed_adjacencies(comm, mesh, "
          "AdjacencyRequest{}.add(2, 1))");
    default:
      throw std::invalid_argument("SemSpace: element gid " + std::to_string(gid) +
                                  " is inverted or degenerate: its normal does not point along +z (plane) or "
                                  "outward (sphere) everywhere");
  }
}

}  // namespace detail

}  // namespace soma::sem
