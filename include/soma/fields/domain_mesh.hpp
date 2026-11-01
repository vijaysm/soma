// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_DOMAIN_MESH_HPP_
#define SOMA_FIELDS_DOMAIN_MESH_HPP_

// Mesh-aware construction of domains and spaces. The mesh supplies the entity
// counts, the global ids that feed the fingerprint, the topology of each
// entity, and its epoch.

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/segments.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/space.hpp"

namespace soma {

// Which segments of one dimension a domain takes. Unset members match anything.
struct SegmentFilter {
  enum class Own : std::uint8_t { any, owned, interior, ghost };
  std::optional<Topology> topology{};
  std::optional<std::uint16_t> arity{};
  Own ownership = Own::any;

  bool matches(const Segment& s) const noexcept {
    if (topology && *topology != s.topology) return false;
    if (arity && *arity != s.arity) return false;
    switch (ownership) {
      case Own::any: return true;
      case Own::owned: return s.ownership != Ownership::ghost;
      case Own::interior: return s.ownership == Ownership::interior;
      case Own::ghost: return s.ownership == Ownership::ghost;
    }
    return false;
  }
};

template <class MemSpace>
Domain<MemSpace> Domain<MemSpace>::all(const Mesh<MemSpace>& mesh, int dim) {
  check_dim(dim, 0);
  return all(dim, mesh.count(dim), mesh.epoch(), mesh.gid(dim));
}

template <class MemSpace>
Domain<MemSpace> Domain<MemSpace>::segments(const Mesh<MemSpace>& mesh, int dim,
                                            const SegmentFilter& filter) {
  check_dim(dim, 0);
  std::vector<Interval> runs;
  for (const Segment& s : mesh.segments(dim)) {
    if (filter.matches(s)) runs.push_back(Interval{s.begin, s.end});
  }
  return from_intervals(dim, mesh.count(dim), mesh.epoch(), runs, mesh.gid(dim));
}

template <class MemSpace>
Domain<MemSpace> Domain<MemSpace>::from_list(const Mesh<MemSpace>& mesh, int dim,
                                             std::span<const LO> sorted_lids) {
  check_dim(dim, 0);
  return from_list(dim, mesh.count(dim), mesh.epoch(), sorted_lids, mesh.gid(dim));
}

// Per-dimension topology and polygon connectivity offsets of the mesh, the
// input of element_nodes spaces.
template <class MemSpace>
ElementInfo<MemSpace> element_info(const Mesh<MemSpace>& mesh) {
  ElementInfo<MemSpace> info;
  for (int d = 1; d <= 3; ++d) {
    info.topology[static_cast<std::size_t>(d)] = mesh.topology(d);
    info.conn_offsets[static_cast<std::size_t>(d)] = mesh.conn_offsets(d);
  }
  return info;
}

template <class MemSpace>
Space<MemSpace> make_space(const Mesh<MemSpace>& mesh, const Domain<MemSpace>& domain,
                           const Layout& layout) {
  return Space<MemSpace>(domain, layout, element_info(mesh));
}

}  // namespace soma

#endif  // SOMA_FIELDS_DOMAIN_MESH_HPP_
