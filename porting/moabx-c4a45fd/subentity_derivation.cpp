#include "moabx/mesh/subentity_derivation.hpp"

#include <algorithm>
#include <array>

namespace moabx::mesh {

namespace {

// Order-independent identity for a sub-entity: its sorted vertex ids.
// Sub-entities have at most 4 corner vertices across all fixed topologies,
// and polygon-derived edges have 2, so a small fixed buffer suffices.
using sorted_key = std::array<index_type, max_subentity_vertices>;

struct key_hash {
  std::size_t operator()(const sorted_key& key) const noexcept {
    // FNV-1a over the fixed-width key. Unused slots are zero-filled and
    // participate consistently, so distinct arities cannot collide by accident.
    std::uint64_t hash = 1469598103934665603ULL;
    for (const index_type value : key) {
      hash ^= static_cast<std::uint64_t>(value);
      hash *= 1099511628211ULL;
    }
    return static_cast<std::size_t>(hash);
  }
};

sorted_key make_key(std::span<const index_type> vertices) noexcept {
  sorted_key key{};
  const std::size_t n = std::min(vertices.size(), max_subentity_vertices);
  for (std::size_t i = 0; i < n; ++i) {
    key[i] = vertices[i];
  }
  std::sort(key.begin(), key.begin() + static_cast<std::ptrdiff_t>(n));
  return key;
}

// Relative orientation of `candidate` against the stored canonical ordering.
//
// Both describe the same vertex set. For an edge, agreement means identical
// order. For a polygonal face, agreement means the candidate is a rotation of
// the canonical cycle; reversal means it is a rotation of the reversed cycle.
// Returns +1, -1, or 0 when neither (a non-conforming or degenerate case).
std::int8_t relative_sense(
    std::span<const index_type> canonical,
    std::span<const index_type> candidate) noexcept {
  const std::size_t n = canonical.size();
  if (n == 0 || candidate.size() != n) {
    return 0;
  }
  if (n == 1) {
    return canonical[0] == candidate[0] ? std::int8_t{1} : std::int8_t{0};
  }
  if (n == 2) {
    if (canonical[0] == candidate[0] && canonical[1] == candidate[1]) {
      return 1;
    }
    if (canonical[0] == candidate[1] && canonical[1] == candidate[0]) {
      return -1;
    }
    return 0;
  }

  // Locate canonical[0] in the candidate, then compare both directions.
  std::size_t start = n;
  for (std::size_t i = 0; i < n; ++i) {
    if (candidate[i] == canonical[0]) {
      start = i;
      break;
    }
  }
  if (start == n) {
    return 0;
  }

  bool forward = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (candidate[(start + i) % n] != canonical[i]) {
      forward = false;
      break;
    }
  }
  if (forward) {
    return 1;
  }

  bool backward = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (candidate[(start + n - i) % n] != canonical[i]) {
      backward = false;
      break;
    }
  }
  return backward ? std::int8_t{-1} : std::int8_t{0};
}

}  // namespace

derived_subentities derive_subentities(
    const element_topology_view& elements,
    int sub_dimension) {
  derived_subentities out;
  if (sub_dimension != 1 && sub_dimension != 2) {
    return out;
  }

  const std::size_t element_count = elements.element_count();
  if (element_count == 0) {
    return out;
  }

  phmap::flat_hash_map<sorted_key, index_type, key_hash> lookup;
  // Rough upper bound; over-reserving costs memory, under-reserving costs
  // rehashes on multi-million-element meshes.
  lookup.reserve(element_count * (sub_dimension == 1 ? 3 : 2));

  // Incidences are gathered per sub-entity id, then flattened. A vector of
  // vectors would allocate per sub-entity, so accumulate flat and counting-sort
  // into CSR at the end.
  std::vector<index_type> incidence_owner;
  std::vector<subentity_incidence> incidence_flat;
  incidence_owner.reserve(element_count * 4);
  incidence_flat.reserve(element_count * 4);

  std::vector<index_type> scratch;
  scratch.reserve(max_subentity_vertices);

  auto register_subentity =
      [&](std::span<const index_type> verts, index_type element, std::uint8_t local_index) {
        const sorted_key key = make_key(verts);
        const auto [it, inserted] = lookup.try_emplace(key, static_cast<index_type>(out.count()));
        const index_type sub_id = it->second;

        if (inserted) {
          out.connectivity.insert(out.connectivity.end(), verts.begin(), verts.end());
          out.offsets.push_back(static_cast<index_type>(out.connectivity.size()));
        }

        const std::int8_t sense = relative_sense(out.vertices(sub_id), verts);
        incidence_owner.push_back(sub_id);
        incidence_flat.push_back(subentity_incidence{element, local_index, sense});
      };

  for (index_type element = 0; element < element_count; ++element) {
    const auto conn = elements.connectivity(element);
    const element_topology topology = elements.topologies[element];

    // Polygons have no canonical table: CN::NumSubEntities(MBPOLYGON, 1) is 0.
    // Their edges are the vertex cycle. Their single 2D "face" is the element
    // itself, so there is nothing to derive at sub_dimension 2.
    if (topology == element_topology::polygon) {
      if (sub_dimension != 1) {
        continue;
      }
      const std::size_t n = conn.size();
      for (std::size_t i = 0; i < n; ++i) {
        const index_type pair[2] = {conn[i], conn[(i + 1) % n]};
        register_subentity(std::span<const index_type>(pair, 2), element, static_cast<std::uint8_t>(i));
      }
      continue;
    }

    if (!has_fixed_topology(topology)) {
      // Polyhedron faces are stored explicitly rather than derived; report so
      // the caller can fall back for these elements.
      out.unsupported_elements.push_back(element);
      continue;
    }

    const auto defs = subentities(topology, sub_dimension);
    for (std::size_t i = 0; i < defs.size(); ++i) {
      const auto& def = defs[i];
      scratch.clear();
      bool valid = true;
      for (std::uint8_t k = 0; k < def.vertex_count; ++k) {
        const std::size_t vi = def.vertices[k];
        if (vi >= conn.size()) {
          valid = false;
          break;
        }
        scratch.push_back(conn[vi]);
      }
      if (!valid) {
        // Connectivity shorter than the topology requires (for example a
        // higher-order element read as corner-only). Skip rather than index
        // out of bounds.
        out.unsupported_elements.push_back(element);
        break;
      }
      register_subentity(scratch, element, static_cast<std::uint8_t>(i));
    }
  }

  // Flatten incidences into CSR ordered by sub-entity id.
  const std::size_t sub_count = out.count();
  out.incidence_offsets.assign(sub_count + 1, 0);
  for (const index_type owner : incidence_owner) {
    ++out.incidence_offsets[owner + 1];
  }
  for (std::size_t i = 1; i < out.incidence_offsets.size(); ++i) {
    out.incidence_offsets[i] += out.incidence_offsets[i - 1];
  }

  out.incidences.resize(incidence_flat.size());
  std::vector<index_type> cursor = out.incidence_offsets;
  for (std::size_t i = 0; i < incidence_flat.size(); ++i) {
    out.incidences[cursor[incidence_owner[i]]++] = incidence_flat[i];
  }

  return out;
}

}  // namespace moabx::mesh
