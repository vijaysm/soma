#ifndef MOABX_MESH_NATIVE_REFERENCE_HPP_
#define MOABX_MESH_NATIVE_REFERENCE_HPP_

// Reference neighborhood traversals implemented directly against the MOAB API.
//
// These are the ground truth that MOABX adjacency results are validated against
// and benchmarked against. They previously existed as two near-verbatim copies,
// one in src/native_validation.cpp and one in src/moabx_graph_bench.cpp,
// differing only in whether `create_if_missing` was hardcoded to true. That was
// a correctness hazard: a fix applied to one copy would silently not apply to
// the other, and the validator would then certify MOABX against a different
// algorithm than the one being timed.
//
// Single definition, parameterized on `create_if_missing`.
//
// Note on `create_if_missing`: passing true lets MOAB materialize adjacency
// tables on demand, which mutates native-side state and therefore changes the
// cost of any subsequent native query. Validation passes true. Benchmarks pass
// whatever the caller selected so that "precreate" and "no-precreate" native
// modes remain distinguishable.

#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "moab/Interface.hpp"
#include "moab/Range.hpp"

#include "moabx/mesh/entity_kind.hpp"

namespace moabx::mesh::native_reference {

inline void check_moab(moab::ErrorCode code, const char* what) {
  if (code != moab::MB_SUCCESS) {
    throw std::runtime_error(
        std::string("MOAB failure in ") + what + " error=" + std::to_string(static_cast<int>(code)));
  }
}

// Topological dimension of the kind treated as the mesh's primary element.
// Returns 0 for a vertex-only mesh, which callers use as a "skip" signal.
inline int element_dimension(entity_kind kind) noexcept {
  switch (kind) {
    case entity_kind::region:
      return 3;
    case entity_kind::face:
      return 2;
    case entity_kind::edge:
      return 1;
    case entity_kind::vertex:
    default:
      return 0;
  }
}

inline moab::Range make_range(std::span<const moab::EntityHandle> handles) {
  moab::Range range;
  for (const auto h : handles) {
    range.insert(h);
  }
  return range;
}

// frontier vertices -> incident elements -> connectivity vertices
inline std::vector<moab::EntityHandle> vertex_neighborhood_handles(
    moab::Interface& native,
    moab::EntityHandle seed_vertex,
    int elem_dim,
    int rings,
    bool create_if_missing = true) {
  if (rings < 0 || elem_dim <= 0) {
    return {};
  }

  std::unordered_set<moab::EntityHandle> visited_vertices{seed_vertex};
  std::vector<moab::EntityHandle> frontier_vertices{seed_vertex};
  std::vector<moab::EntityHandle> neighborhood{seed_vertex};

  for (int hop = 0; hop < rings; ++hop) {
    const moab::Range frontier_range =
        make_range(std::span<const moab::EntityHandle>(frontier_vertices.data(), frontier_vertices.size()));

    moab::Range elements;
    check_moab(
        native.get_adjacencies(frontier_range, elem_dim, create_if_missing, elements, moab::Interface::UNION),
        "native neighborhood(vertex->elem)");

    moab::Range next_vertices_range;
    check_moab(
        native.get_adjacencies(elements, 0, create_if_missing, next_vertices_range, moab::Interface::UNION),
        "native neighborhood(elem->vertex)");

    std::vector<moab::EntityHandle> next_vertices;
    for (const auto vh : next_vertices_range) {
      if (visited_vertices.insert(vh).second) {
        next_vertices.push_back(vh);
        neighborhood.push_back(vh);
      }
    }

    if (next_vertices.empty()) {
      break;
    }
    frontier_vertices = std::move(next_vertices);
  }

  return neighborhood;
}

// frontier edges -> edge vertices -> incident elements -> connectivity vertices -> incident edges
inline std::vector<moab::EntityHandle> edge_neighborhood_handles(
    moab::Interface& native,
    moab::EntityHandle seed_edge,
    int elem_dim,
    int rings,
    bool create_if_missing = true) {
  if (rings < 0 || elem_dim <= 0) {
    return {};
  }

  std::unordered_set<moab::EntityHandle> visited_edges{seed_edge};
  std::vector<moab::EntityHandle> frontier_edges{seed_edge};
  std::vector<moab::EntityHandle> neighborhood{seed_edge};

  for (int hop = 0; hop < rings; ++hop) {
    const moab::Range frontier_range =
        make_range(std::span<const moab::EntityHandle>(frontier_edges.data(), frontier_edges.size()));

    moab::Range vertices_from_edges;
    check_moab(
        native.get_adjacencies(frontier_range, 0, create_if_missing, vertices_from_edges, moab::Interface::UNION),
        "native neighborhood(edge->vertex)");

    moab::Range elements;
    check_moab(
        native.get_adjacencies(vertices_from_edges, elem_dim, create_if_missing, elements, moab::Interface::UNION),
        "native neighborhood(vertex->elem)");

    moab::Range vertices_from_elements;
    check_moab(
        native.get_adjacencies(elements, 0, create_if_missing, vertices_from_elements, moab::Interface::UNION),
        "native neighborhood(elem->vertex)");

    moab::Range next_edges_range;
    check_moab(
        native.get_adjacencies(vertices_from_elements, 1, create_if_missing, next_edges_range, moab::Interface::UNION),
        "native neighborhood(vertex->edge)");

    std::vector<moab::EntityHandle> next_edges;
    for (const auto eh : next_edges_range) {
      if (visited_edges.insert(eh).second) {
        next_edges.push_back(eh);
        neighborhood.push_back(eh);
      }
    }

    if (next_edges.empty()) {
      break;
    }
    frontier_edges = std::move(next_edges);
  }

  return neighborhood;
}

// frontier elements -> connectivity vertices -> incident elements
inline std::vector<moab::EntityHandle> element_neighborhood_handles(
    moab::Interface& native,
    moab::EntityHandle seed_element,
    int elem_dim,
    int rings,
    bool create_if_missing = true) {
  if (rings < 0 || elem_dim <= 0) {
    return {};
  }

  std::unordered_set<moab::EntityHandle> visited_elements{seed_element};
  std::vector<moab::EntityHandle> frontier_elements{seed_element};
  std::vector<moab::EntityHandle> neighborhood{seed_element};

  for (int hop = 0; hop < rings; ++hop) {
    const moab::Range frontier_range =
        make_range(std::span<const moab::EntityHandle>(frontier_elements.data(), frontier_elements.size()));

    moab::Range vertices;
    check_moab(
        native.get_adjacencies(frontier_range, 0, create_if_missing, vertices, moab::Interface::UNION),
        "native neighborhood(elem->vertex)");

    moab::Range next_elements_range;
    check_moab(
        native.get_adjacencies(vertices, elem_dim, create_if_missing, next_elements_range, moab::Interface::UNION),
        "native neighborhood(vertex->elem)");

    std::vector<moab::EntityHandle> next_elements;
    for (const auto eh : next_elements_range) {
      if (visited_elements.insert(eh).second) {
        next_elements.push_back(eh);
        neighborhood.push_back(eh);
      }
    }

    if (next_elements.empty()) {
      break;
    }
    frontier_elements = std::move(next_elements);
  }

  return neighborhood;
}

}  // namespace moabx::mesh::native_reference

#endif  // MOABX_MESH_NATIVE_REFERENCE_HPP_
