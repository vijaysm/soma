// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/testing/box_mesh.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace soma::testing {

namespace {

std::int64_t checked_product(std::int64_t a, std::int64_t b, std::int64_t limit, const char* what,
                             const char* type) {
  if (a != 0 && b > limit / a) {
    throw std::overflow_error(std::string("box_mesh: the number of ") + what + " exceeds the " +
                              type + " limit " + std::to_string(limit));
  }
  return a * b;
}

void check_cells(const char* axis, int n) {
  if (n < 1) {
    throw std::invalid_argument(std::string("box_mesh: ") + axis + " must be at least 1 (got " +
                                std::to_string(n) + ")");
  }
}

void check_length(const char* axis, Real length) {
  if (!(length > 0) || !std::isfinite(length)) {
    throw std::invalid_argument(std::string("box_mesh: ") + axis +
                                " must be positive and finite (got " + std::to_string(length) + ")");
  }
}

Real coordinate(Real length, std::int64_t i, std::int64_t n) {
  if (i == n) return length;
  return length * static_cast<Real>(i) / static_cast<Real>(n);
}

}  // namespace

BoxCounts box_counts(const BoxSpec& spec) {
  const Topology t = spec.element;
  if (t != Topology::hex && t != Topology::quad && t != Topology::tri) {
    throw std::invalid_argument(std::string("box_mesh: unsupported element topology '") + name(t) +
                                "' (supported: hex, quad, tri)");
  }
  check_cells("nx", spec.nx);
  check_cells("ny", spec.ny);
  check_cells("nz", spec.nz);
  const bool solid = t == Topology::hex;
  if (!solid && spec.nz != 1) {
    throw std::invalid_argument(std::string("box_mesh: nz must be 1 for ") + name(t) +
                                " elements (got " + std::to_string(spec.nz) + ")");
  }
  check_length("lx", spec.lx);
  check_length("ly", spec.ly);
  if (solid) check_length("lz", spec.lz);

  const auto lo_max = static_cast<std::int64_t>(std::numeric_limits<LO>::max());
  const auto offset_max = static_cast<std::int64_t>(std::numeric_limits<Offset>::max());
  const std::int64_t nx = spec.nx;
  const std::int64_t ny = spec.ny;
  const std::int64_t nz = spec.nz;

  BoxCounts c;
  c.vertices = checked_product(checked_product(nx + 1, ny + 1, lo_max, "vertices", "LO"),
                               solid ? nz + 1 : 1, lo_max, "vertices", "LO");
  const std::int64_t cells = checked_product(checked_product(nx, ny, lo_max, "elements", "LO"), nz,
                                             lo_max, "elements", "LO");
  c.elements = checked_product(cells, t == Topology::tri ? 2 : 1, lo_max, "elements", "LO");
  c.connectivity =
      checked_product(c.elements, num_vertices(t), offset_max, "connectivity entries", "Offset");
  return c;
}

HostMesh box_mesh(const BoxSpec& spec) {
  const BoxCounts counts = box_counts(spec);
  const Topology t = spec.element;
  const bool solid = t == Topology::hex;
  const std::int64_t nx = spec.nx;
  const std::int64_t ny = spec.ny;
  const std::int64_t nz = spec.nz;
  const std::int64_t px = nx + 1;
  const std::int64_t py = ny + 1;
  const std::int64_t pz = solid ? nz + 1 : 1;
  const auto vertex = [&](std::int64_t i, std::int64_t j, std::int64_t k) -> GO {
    return static_cast<GO>(1 + i + px * (j + py * k));
  };

  const auto nv = static_cast<std::size_t>(counts.vertices);
  std::vector<GO> vertex_gids;
  std::vector<Real> xyz;
  vertex_gids.reserve(nv);
  xyz.reserve(3 * nv);
  for (std::int64_t k = 0; k < pz; ++k) {
    for (std::int64_t j = 0; j < py; ++j) {
      for (std::int64_t i = 0; i < px; ++i) {
        vertex_gids.push_back(vertex(i, j, k));
        xyz.push_back(coordinate(spec.lx, i, nx));
        xyz.push_back(coordinate(spec.ly, j, ny));
        xyz.push_back(solid ? coordinate(spec.lz, k, nz) : Real{0});
      }
    }
  }

  std::vector<GO> element_gids;
  std::vector<GO> conn;
  element_gids.reserve(static_cast<std::size_t>(counts.elements));
  conn.reserve(static_cast<std::size_t>(counts.connectivity));
  for (std::int64_t k = 0; k < (solid ? nz : 1); ++k) {
    for (std::int64_t j = 0; j < ny; ++j) {
      for (std::int64_t i = 0; i < nx; ++i) {
        const std::int64_t cell = i + nx * (j + ny * k);
        const GO v0 = vertex(i, j, k);
        const GO v1 = vertex(i + 1, j, k);
        const GO v2 = vertex(i + 1, j + 1, k);
        const GO v3 = vertex(i, j + 1, k);
        if (solid) {
          element_gids.push_back(static_cast<GO>(1 + cell));
          conn.insert(conn.end(), {v0, v1, v2, v3, vertex(i, j, k + 1), vertex(i + 1, j, k + 1),
                                   vertex(i + 1, j + 1, k + 1), vertex(i, j + 1, k + 1)});
        } else if (t == Topology::quad) {
          element_gids.push_back(static_cast<GO>(1 + cell));
          conn.insert(conn.end(), {v0, v1, v2, v3});
        } else {
          element_gids.push_back(static_cast<GO>(2 * cell + 1));
          element_gids.push_back(static_cast<GO>(2 * cell + 2));
          conn.insert(conn.end(), {v0, v1, v2, v0, v2, v3});
        }
      }
    }
  }

  MeshBuilder builder(solid ? 3 : 2);
  builder.add_vertices(vertex_gids, xyz);
  builder.add_elements(t, element_gids, conn);
  return std::move(builder).build(spec.ordering);
}

}  // namespace soma::testing
