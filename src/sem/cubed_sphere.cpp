// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/sem/cubed_sphere.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "soma/core/mesh_builder.hpp"
#include "soma/core/topology.hpp"
#include "soma/parallel/distributed.hpp"

namespace soma::sem {

namespace {

using Point = std::array<int, 3>;  // a point of the cube's surface lattice, each index in [0, m]

// The rank holding element e (0-based) of n: consecutive blocks, the first n % size ranks one more.
int owner(std::int64_t e, std::int64_t n, int size) {
  const std::int64_t base = n / size, rem = n % size;
  if (e < rem * (base + 1)) return static_cast<int>(e / (base + 1));
  return static_cast<int>(rem + (e - rem * (base + 1)) / base);
}

}  // namespace

HostMesh cubed_sphere_mesh(const Comm& comm, const CubedSphereSpec& spec) {
  const int ne = spec.ne;
  if (ne < 1) throw std::invalid_argument("cubed_sphere_mesh: ne " + std::to_string(ne) + " < 1");
  if (!(spec.radius > 0) || !std::isfinite(spec.radius)) {
    throw std::invalid_argument("cubed_sphere_mesh: radius " + std::to_string(spec.radius) +
                                " is not positive and finite");
  }
  if (spec.geometry_order != 1 && spec.geometry_order != 2) {
    throw std::invalid_argument("cubed_sphere_mesh: geometry order " + std::to_string(spec.geometry_order) +
                                " is not 1 or 2");
  }
  const int step = spec.geometry_order;  // lattice points per element edge
  const int m = ne * step;
  const auto gid_of = [&](const Point& t) {
    return static_cast<GO>((static_cast<std::int64_t>(t[2]) * (m + 1) + t[1]) * (m + 1) + t[0] + 1);
  };
  const std::int64_t n = 6LL * ne * ne;

  std::vector<GO> elements, nodes;
  std::vector<Point> used;
  for (int k = 0; k < 3; ++k) {
    for (const int s : {-1, 1}) {
      const int f = 2 * k + (s > 0 ? 1 : 0), a = (k + 1) % 3, b = (k + 2) % 3;
      const auto lattice = [&](int u, int w) {
        Point t{};
        t[k] = s > 0 ? m : 0;
        t[a] = u;
        t[b] = w;
        return t;
      };
      for (int j = 0; j < ne; ++j) {
        for (int i = 0; i < ne; ++i) {
          const std::int64_t e = static_cast<std::int64_t>(f) * ne * ne + i + static_cast<std::int64_t>(ne) * j;
          if (owner(e, n, comm.size()) != comm.rank()) continue;
          std::array<Point, 4> corner = {lattice(i * step, j * step), lattice((i + 1) * step, j * step),
                                         lattice((i + 1) * step, (j + 1) * step), lattice(i * step, (j + 1) * step)};
          if (s < 0) std::reverse(corner.begin() + 1, corner.end());  // counter-clockwise from outside
          std::vector<Point> element(corner.begin(), corner.end());
          if (step == 2) {  // MOAB's Quad9: the four mid-edge nodes, then the centre
            const auto mid = [](const Point& p, const Point& q) {
              return Point{(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2};
            };
            for (int q = 0; q < 4; ++q) element.push_back(mid(corner[q], corner[(q + 1) % 4]));
            element.push_back(mid(corner[0], corner[2]));
          }
          elements.push_back(static_cast<GO>(e + 1));
          for (const Point& p : element) {
            nodes.push_back(gid_of(p));
            used.push_back(p);
          }
        }
      }
    }
  }

  std::sort(used.begin(), used.end(), [&](const Point& p, const Point& q) { return gid_of(p) < gid_of(q); });
  used.erase(std::unique(used.begin(), used.end()), used.end());
  const auto at = [&](int t) { return std::tan(std::numbers::pi / 4 * (2.0 * t / m - 1)); };
  std::vector<GO> vertex_gids;
  std::vector<Real> xyz;
  vertex_gids.reserve(used.size());
  xyz.reserve(3 * used.size());
  for (const Point& t : used) {
    const Real p[3] = {at(t[0]), at(t[1]), at(t[2])};
    const Real l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    vertex_gids.push_back(gid_of(t));
    for (const Real c : p) xyz.push_back(c * spec.radius / l);
  }

  MeshBuilder builder(2);
  builder.add_vertices(vertex_gids, xyz);
  if (step == 1) {
    builder.add_elements(Topology::quad, elements, nodes);
  } else {
    builder.add_elements(Topology::quad, elements, nodes, 9);
  }
  return build_local_part(comm, std::move(builder), spec.ordering);
}

}  // namespace soma::sem
