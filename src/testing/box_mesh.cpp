// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/testing/box_mesh.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "soma/parallel/distributed.hpp"

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

// Coordinates are length * i / n, so length * n must stay finite too.
void check_length(const char* axis, Real length, int n) {
  if (!(length > 0) || !std::isfinite(length)) {
    throw std::invalid_argument(std::string("box_mesh: ") + axis +
                                " must be positive and finite (got " + std::to_string(length) +
                                ")");
  }
  if (!std::isfinite(length * static_cast<Real>(n))) {
    throw std::invalid_argument(std::string("box_mesh: ") + axis + " times the cell count " +
                                std::to_string(n) + " is not finite (got " +
                                std::to_string(length) + ")");
  }
}

Real coordinate(Real length, std::int64_t i, std::int64_t n) {
  if (i == n) return length;
  return length * static_cast<Real>(i) / static_cast<Real>(n);
}

}  // namespace

BoxCounts box_counts(const BoxSpec& spec) {
  const Topology t = spec.element;
  if (t != Topology::hex && t != Topology::tet && t != Topology::quad && t != Topology::tri) {
    throw std::invalid_argument(std::string("box_mesh: unsupported element topology '") + name(t) +
                                "' (supported: hex, tet, quad, tri)");
  }
  check_cells("nx", spec.nx);
  check_cells("ny", spec.ny);
  check_cells("nz", spec.nz);
  const bool solid = t == Topology::hex || t == Topology::tet;
  if (!solid && spec.nz != 1) {
    throw std::invalid_argument(std::string("box_mesh: nz must be 1 for ") + name(t) +
                                " elements (got " + std::to_string(spec.nz) + ")");
  }
  check_length("lx", spec.lx, spec.nx);
  check_length("ly", spec.ly, spec.ny);
  if (solid) check_length("lz", spec.lz, spec.nz);

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
  const std::int64_t split = t == Topology::tri ? 2 : t == Topology::tet ? 6 : 1;
  c.elements = checked_product(cells, split, lo_max, "elements", "LO");
  c.connectivity =
      checked_product(c.elements, num_vertices(t), offset_max, "connectivity entries", "Offset");
  return c;
}

namespace {

// The builder of the elements in layers [first, last) along the box's last
// axis (k for hex and tet, j for quad and tri) and of the vertices of layers
// [first, last]; none when first == last.
MeshBuilder box_builder(const BoxSpec& spec, std::int64_t first, std::int64_t last) {
  const BoxCounts counts = box_counts(spec);
  const Topology t = spec.element;
  const bool solid = t == Topology::hex || t == Topology::tet;
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
      const std::int64_t layer = solid ? k : j;
      if (first == last || layer < first || layer > last) continue;
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
      const std::int64_t layer = solid ? k : j;
      if (layer < first || layer >= last) continue;
      for (std::int64_t i = 0; i < nx; ++i) {
        const std::int64_t cell = i + nx * (j + ny * k);
        const GO v0 = vertex(i, j, k);
        const GO v1 = vertex(i + 1, j, k);
        const GO v2 = vertex(i + 1, j + 1, k);
        const GO v3 = vertex(i, j + 1, k);
        if (t == Topology::tet) {
          // Kuhn's split: one tet per order of the axes, along the path from the
          // cube's first corner to its last; odd orders swap the last two corners
          // so every tet has positive volume. Every cube splits alike, so faces
          // match.
          static constexpr int order[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1},
                                              {0, 2, 1}, {2, 1, 0}, {1, 0, 2}};
          for (int o = 0; o < 6; ++o) {
            std::int64_t at[3] = {i, j, k};
            GO c[4];
            c[0] = vertex(at[0], at[1], at[2]);
            for (int s = 0; s < 3; ++s) {
              ++at[order[o][s]];
              c[s + 1] = vertex(at[0], at[1], at[2]);
            }
            if (o >= 3) std::swap(c[2], c[3]);
            element_gids.push_back(static_cast<GO>(6 * cell + o + 1));
            conn.insert(conn.end(), {c[0], c[1], c[2], c[3]});
          }
        } else if (solid) {
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
  return builder;
}

}  // namespace

HostMesh box_mesh(const BoxSpec& spec) {
  const bool solid = spec.element == Topology::hex || spec.element == Topology::tet;
  return box_builder(spec, 0, solid ? spec.nz : spec.ny).build(spec.ordering);
}

HostMesh box_mesh(const Comm& comm, const BoxSpec& spec) {
  const bool solid = spec.element == Topology::hex || spec.element == Topology::tet;
  const std::int64_t n = solid ? spec.nz : spec.ny;
  const std::int64_t first = n * comm.rank() / comm.size();
  const std::int64_t last = n * (comm.rank() + 1) / comm.size();
  std::exception_ptr failure;
  std::optional<MeshBuilder> builder;
  try {
    builder.emplace(box_builder(spec, first, last));
  } catch (...) {
    failure = std::current_exception();
  }
  // A bad spec throws on every rank alike; agreeing keeps any rank from
  // entering build_local_part alone.
  if (comm.allreduce(static_cast<std::int32_t>(failure != nullptr), Op::max) != 0) {
    if (failure) std::rethrow_exception(failure);
    throw std::invalid_argument("box_mesh: the box specification failed on another rank");
  }
  return build_local_part(comm, std::move(*builder), spec.ordering);
}

}  // namespace soma::testing
