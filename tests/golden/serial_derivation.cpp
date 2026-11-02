// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "serial_derivation.hpp"

#include <algorithm>
#include <array>
#include <unordered_map>

#include "soma/topology/canonical.hpp"

namespace soma_golden {

namespace {

constexpr std::size_t max_side_vertices = 4;

// Order-independent identity of a sub-entity: its sorted vertex ids, unused
// slots -1.
using Key = std::array<LO, max_side_vertices>;

struct KeyHash {
  std::size_t operator()(const Key& key) const noexcept {
    std::uint64_t hash = 1469598103934665603ULL;  // FNV-1a
    for (const LO value : key) {
      hash ^= static_cast<std::uint64_t>(value);
      hash *= 1099511628211ULL;
    }
    return static_cast<std::size_t>(hash);
  }
};

Key make_key(std::span<const LO> vertices) noexcept {
  Key key;
  key.fill(-1);
  const std::size_t n = std::min(vertices.size(), max_side_vertices);
  for (std::size_t i = 0; i < n; ++i) key[i] = vertices[i];
  // Insertion sort of at most four values. std::sort here makes GCC 13 report
  // an out-of-bounds subscript from its inlined 16-element threshold path.
  for (std::size_t i = 1; i < n; ++i) {
    const LO v = key[i];
    std::size_t j = i;
    for (; j > 0 && key[j - 1] > v; --j) key[j] = key[j - 1];
    key[j] = v;
  }
  return key;
}

// +1 if candidate is a rotation of canonical, -1 if a rotation of its
// reverse, 0 otherwise. Two vertices agree only in the same order.
std::int8_t relative_sense(std::span<const LO> canonical, std::span<const LO> candidate) noexcept {
  const std::size_t n = canonical.size();
  if (n == 0 || candidate.size() != n) return 0;
  if (n == 1) return canonical[0] == candidate[0] ? 1 : 0;
  if (n == 2) {
    if (canonical[0] == candidate[0] && canonical[1] == candidate[1]) return 1;
    if (canonical[0] == candidate[1] && canonical[1] == candidate[0]) return -1;
    return 0;
  }
  std::size_t start = n;
  for (std::size_t i = 0; i < n; ++i) {
    if (candidate[i] == canonical[0]) {
      start = i;
      break;
    }
  }
  if (start == n) return 0;
  bool forward = true;
  for (std::size_t i = 0; i < n && forward; ++i) {
    forward = candidate[(start + i) % n] == canonical[i];
  }
  if (forward) return 1;
  bool backward = true;
  for (std::size_t i = 0; i < n && backward; ++i) {
    backward = candidate[(start + n - i) % n] == canonical[i];
  }
  return backward ? -1 : 0;
}

}  // namespace

Derived derive(const Elements& elements, int sub_dimension) {
  Derived out;
  if (sub_dimension != 1 && sub_dimension != 2) return out;
  const std::size_t element_count = elements.count();
  if (element_count == 0) return out;

  std::unordered_map<Key, LO, KeyHash> lookup;
  lookup.reserve(element_count * (sub_dimension == 1 ? 3 : 2));

  // Accumulate flat, then counting-sort into CSR by sub-entity id.
  std::vector<LO> incidence_owner;
  std::vector<Incidence> incidence_flat;
  std::vector<LO> scratch;
  scratch.reserve(max_side_vertices);

  auto register_side = [&](std::span<const LO> verts, LO element, std::uint8_t local_index) {
    const auto [it, inserted] = lookup.try_emplace(make_key(verts), static_cast<LO>(out.count()));
    const LO id = it->second;
    if (inserted) {
      out.connectivity.insert(out.connectivity.end(), verts.begin(), verts.end());
      out.offsets.push_back(static_cast<LO>(out.connectivity.size()));
    }
    incidence_owner.push_back(id);
    incidence_flat.push_back(Incidence{element, local_index, relative_sense(out.vertices(id), verts)});
  };

  for (LO element = 0; element < static_cast<LO>(element_count); ++element) {
    const auto conn = elements.connectivity(element);
    const soma::Topology topology = elements.topologies[static_cast<std::size_t>(element)];

    // A polygon's edges are its vertex cycle; it has no faces to derive.
    if (topology == soma::Topology::polygon) {
      if (sub_dimension != 1) continue;
      const std::size_t n = conn.size();
      for (std::size_t i = 0; i < n; ++i) {
        const LO pair[2] = {conn[i], conn[(i + 1) % n]};
        register_side(std::span<const LO>(pair, 2), element, static_cast<std::uint8_t>(i));
      }
      continue;
    }
    if (!soma::has_fixed_topology(topology)) {
      out.unsupported_elements.push_back(element);
      continue;
    }
    const int sides = soma::canonical::num_sides(topology, sub_dimension);
    for (int i = 0; i < sides; ++i) {
      const auto def = soma::canonical::side(topology, sub_dimension, i);
      scratch.clear();
      bool valid = true;
      for (std::uint8_t k = 0; k < def.num_vertices; ++k) {
        if (def.v[k] >= conn.size()) {
          valid = false;
          break;
        }
        scratch.push_back(conn[def.v[k]]);
      }
      if (!valid) {
        out.unsupported_elements.push_back(element);
        break;
      }
      register_side(scratch, element, static_cast<std::uint8_t>(i));
    }
  }

  const std::size_t sub_count = out.count();
  out.incidence_offsets.assign(sub_count + 1, 0);
  for (const LO owner : incidence_owner) ++out.incidence_offsets[static_cast<std::size_t>(owner) + 1];
  for (std::size_t i = 1; i < out.incidence_offsets.size(); ++i) {
    out.incidence_offsets[i] += out.incidence_offsets[i - 1];
  }
  out.incidences.resize(incidence_flat.size());
  std::vector<LO> cursor = out.incidence_offsets;
  for (std::size_t i = 0; i < incidence_flat.size(); ++i) {
    const auto owner = static_cast<std::size_t>(incidence_owner[i]);
    out.incidences[static_cast<std::size_t>(cursor[owner]++)] = incidence_flat[i];
  }
  return out;
}

}  // namespace soma_golden
