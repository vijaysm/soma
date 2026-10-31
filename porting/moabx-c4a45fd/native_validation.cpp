#include "moabx/mesh/native_validation.hpp"

#include <algorithm>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "moab/MeshTopoUtil.hpp"
#include "moab/Range.hpp"

#include "moabx/mesh/native_reference.hpp"

namespace moabx::mesh {

namespace {

using native_reference::check_moab;
using native_reference::element_dimension;
using native_reference::make_range;

std::vector<index_type> sorted_unique_span(std::span<const index_type> row) {
  std::vector<index_type> out(row.begin(), row.end());
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<index_type> map_native_handles_to_local_ids(
    const std::vector<moab::EntityHandle>& native_handles,
    entity_kind kind,
    const mesh_graph& graph,
    adjacency_validation_result& report) {
  std::vector<index_type> out;
  out.reserve(native_handles.size());
  for (const auto h : native_handles) {
    const auto lid = graph.local_id(kind, static_cast<handle_type>(h));
    if (!lid.has_value()) {
      ++report.missing_handles;
      continue;
    }
    out.push_back(*lid);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<index_type> map_native_range_to_local_ids(
    const moab::Range& native_range,
    entity_kind kind,
    const mesh_graph& graph,
    adjacency_validation_result& report,
    handle_type self_to_skip = 0) {
  std::vector<moab::EntityHandle> handles;
  handles.reserve(native_range.size());
  for (const auto h : native_range) {
    if (self_to_skip != 0 && static_cast<handle_type>(h) == self_to_skip) {
      continue;
    }
    handles.push_back(h);
  }
  return map_native_handles_to_local_ids(handles, kind, graph, report);
}

template <class T>
void print_vector_preview(
    std::ostream& os,
    std::string_view label,
    const std::vector<T>& values,
    std::size_t max_items = 12) {
  os << label << "=[";
  const std::size_t n = std::min(values.size(), max_items);
  for (std::size_t i = 0; i < n; ++i) {
    if (i != 0) {
      os << ",";
    }
    os << values[i];
  }
  if (values.size() > n) {
    os << ",...";
  }
  os << "]";
}

std::vector<index_type> map_native_connectivity_to_vertex_local_ids(
    moab::Interface& native,
    handle_type entity_handle,
    const mesh_graph& graph,
    adjacency_validation_result& report) {
  const auto native_handle = static_cast<moab::EntityHandle>(entity_handle);
  const moab::EntityHandle* conn = nullptr;
  int nnodes = 0;
  check_moab(native.get_connectivity(native_handle, conn, nnodes, true), "get_connectivity(entity)");

  std::vector<moab::EntityHandle> conn_handles;
  conn_handles.reserve(static_cast<std::size_t>(nnodes));
  for (int i = 0; i < nnodes; ++i) {
    conn_handles.push_back(conn[i]);
  }
  return map_native_handles_to_local_ids(conn_handles, entity_kind::vertex, graph, report);
}

void compare_row(
    const std::string& query_name,
    handle_type entity_handle,
    std::span<const index_type> actual_row,
    const std::vector<index_type>& expected_sorted,
    adjacency_validation_result& report,
    std::ostream* log,
    std::int64_t max_logged_mismatches);

void validate_vertex_coordinates(
    const mesh_graph& graph,
    moab::Interface& native,
    adjacency_validation_result& report,
    std::ostream* log,
    std::int64_t max_logged_mismatches) {
  const std::size_t vertex_count = graph.entity_count(entity_kind::vertex);
  std::int64_t logged = 0;

  for (index_type v = 0; v < vertex_count; ++v) {
    const auto handle = static_cast<moab::EntityHandle>(graph.handle(entity_kind::vertex, v));
    double native_xyz[3] = {0.0, 0.0, 0.0};
    if (native.get_coords(&handle, 1, native_xyz) != moab::MB_SUCCESS) {
      ++report.missing_handles;
      continue;
    }

    const auto stored = graph.vertex_coordinates(v);
    ++report.checks;
    if (stored.size() < 3) {
      ++report.mismatches;
      continue;
    }

    const bool equal = stored[0] == native_xyz[0] &&
                       stored[1] == native_xyz[1] &&
                       stored[2] == native_xyz[2];
    if (!equal) {
      ++report.mismatches;
      if (log != nullptr && logged < max_logged_mismatches) {
        ++logged;
        *log << "[validate] vertex_coordinates local_id=" << v
             << " moabx=(" << stored[0] << ", " << stored[1] << ", " << stored[2] << ")"
             << " native=(" << native_xyz[0] << ", " << native_xyz[1] << ", " << native_xyz[2]
             << ")\n";
      }
    }
  }
}

void validate_connectivity_kind(
    entity_kind kind,
    const char* query_name,
    const mesh_graph& graph,
    moab::Interface& native,
    adjacency_validation_result& report,
    std::ostream* log,
    std::int64_t max_logged_mismatches) {
  for (index_type i = 0; i < graph.entity_count(kind); ++i) {
    const handle_type eh = graph.handle(kind, i);
    const auto expected = map_native_connectivity_to_vertex_local_ids(native, eh, graph, report);
    compare_row(query_name, eh, graph.connectivity(kind, i), expected, report, log, max_logged_mismatches);
  }
}

void compare_row(
    const std::string& query_name,
    handle_type entity_handle,
    std::span<const index_type> actual_row,
    const std::vector<index_type>& expected_sorted,
    adjacency_validation_result& report,
    std::ostream* log,
    std::int64_t max_logged_mismatches) {
  ++report.checks;
  const auto actual_sorted = sorted_unique_span(actual_row);
  if (actual_sorted == expected_sorted) {
    return;
  }

  ++report.mismatches;
  if (log != nullptr && report.mismatches <= max_logged_mismatches) {
    *log << "[validate] mismatch query=" << query_name
         << " entity_handle=" << entity_handle
         << " actual_size=" << actual_sorted.size()
         << " expected_size=" << expected_sorted.size();
    print_vector_preview(*log, " actual_ids", actual_sorted);
    print_vector_preview(*log, " expected_ids", expected_sorted);
    *log << "\n";
  }
}

void compare_vec(
    const std::string& query_name,
    handle_type entity_handle,
    std::vector<index_type> actual,
    const std::vector<index_type>& expected_sorted,
    adjacency_validation_result& report,
    std::ostream* log,
    std::int64_t max_logged_mismatches) {
  std::sort(actual.begin(), actual.end());
  actual.erase(std::unique(actual.begin(), actual.end()), actual.end());

  ++report.checks;
  if (actual == expected_sorted) {
    return;
  }

  ++report.mismatches;
  if (log != nullptr && report.mismatches <= max_logged_mismatches) {
    *log << "[validate] mismatch query=" << query_name
         << " entity_handle=" << entity_handle
         << " actual_size=" << actual.size()
         << " expected_size=" << expected_sorted.size();
    print_vector_preview(*log, " actual_ids", actual);
    print_vector_preview(*log, " expected_ids", expected_sorted);
    *log << "\n";
  }
}

}  // namespace

adjacency_validation_result validate_graph_against_moab(
    const mesh_graph& graph,
    moab::Interface& native,
    bool validate_rings,
    std::ostream* log,
    std::int64_t max_logged_mismatches) {
  adjacency_validation_result report{};

  // Coordinates were never validated: only topology was. A loader that mapped
  // the right connectivity onto the wrong vertex positions would have passed
  // every previous check. Coordinates are copied verbatim from MOAB, so exact
  // equality is the correct expectation -- a tolerance here would mask a bug
  // rather than accommodate rounding.
  validate_vertex_coordinates(graph, native, report, log, max_logged_mismatches);

  const auto elem_kind = graph.primary_element_kind();
  const int elem_dim = element_dimension(elem_kind);
  const bool has_edges = graph.entity_count(entity_kind::edge) > 0;
  const bool has_faces = graph.entity_count(entity_kind::face) > 0;
  const bool has_regions = graph.entity_count(entity_kind::region) > 0;
  moab::MeshTopoUtil mtu(&native);

  if (has_edges) {
    validate_connectivity_kind(
        entity_kind::edge,
        "connectivity_edge",
        graph,
        native,
        report,
        log,
        max_logged_mismatches);
  }
  if (has_faces) {
    validate_connectivity_kind(
        entity_kind::face,
        "connectivity_face",
        graph,
        native,
        report,
        log,
        max_logged_mismatches);
  }
  if (has_regions) {
    validate_connectivity_kind(
        entity_kind::region,
        "connectivity_region",
        graph,
        native,
        report,
        log,
        max_logged_mismatches);
  }

  for (index_type v = 0; v < graph.entity_count(entity_kind::vertex); ++v) {
    const handle_type vh = graph.handle(entity_kind::vertex, v);
    const auto native_vh = static_cast<moab::EntityHandle>(vh);

    if (elem_dim > 0) {
      std::vector<moab::EntityHandle> native_adj;
      check_moab(
          native.get_adjacencies(
              &native_vh,
              1,
              elem_dim,
              true,
              native_adj,
              moab::Interface::UNION),
          "get_adjacencies(vertex->element)");
      const auto expected = map_native_handles_to_local_ids(native_adj, elem_kind, graph, report);
      compare_row("vertex_to_element", vh, graph.vertex_to_elements(v), expected, report, log, max_logged_mismatches);
    }

    if (has_edges) {
      std::vector<moab::EntityHandle> native_adj;
      check_moab(
          native.get_adjacencies(
              &native_vh,
              1,
              1,
              true,
              native_adj,
              moab::Interface::UNION),
          "get_adjacencies(vertex->edge)");
      const auto expected = map_native_handles_to_local_ids(native_adj, entity_kind::edge, graph, report);
      compare_row("vertex_to_edge", vh, graph.vertex_to_edges(v), expected, report, log, max_logged_mismatches);
    }

    if (has_faces) {
      std::vector<moab::EntityHandle> native_adj;
      check_moab(
          native.get_adjacencies(
              &native_vh,
              1,
              2,
              true,
              native_adj,
              moab::Interface::UNION),
          "get_adjacencies(vertex->face)");
      const auto expected = map_native_handles_to_local_ids(native_adj, entity_kind::face, graph, report);
      compare_row("vertex_to_face", vh, graph.vertex_to_faces(v), expected, report, log, max_logged_mismatches);
    }

    if (has_regions) {
      std::vector<moab::EntityHandle> native_adj;
      check_moab(
          native.get_adjacencies(
              &native_vh,
              1,
              3,
              true,
              native_adj,
              moab::Interface::UNION),
          "get_adjacencies(vertex->region)");
      const auto expected = map_native_handles_to_local_ids(native_adj, entity_kind::region, graph, report);
      compare_row("vertex_to_region", vh, graph.vertex_to_regions(v), expected, report, log, max_logged_mismatches);
    }

    if (validate_rings && has_edges) {
      moab::Range native_ring;
      check_moab(mtu.get_bridge_adjacencies(native_vh, 1, 0, native_ring), "mtu.get_bridge_adjacencies(vertex ring)");
      const auto expected = map_native_range_to_local_ids(native_ring, entity_kind::vertex, graph, report, vh);
      compare_row("vertex_ring", vh, graph.vertex_ring(v), expected, report, log, max_logged_mismatches);
    }

    if (validate_rings && elem_dim > 0) {
      {
        const auto expected = map_native_handles_to_local_ids(
            native_reference::vertex_neighborhood_handles(native, native_vh, elem_dim, 1),
            entity_kind::vertex,
            graph,
            report);
        compare_vec("vertex_neighborhood_1", vh, graph.vertex_neighborhood(v, 1), expected, report, log, max_logged_mismatches);
      }
      {
        const auto expected = map_native_handles_to_local_ids(
            native_reference::vertex_neighborhood_handles(native, native_vh, elem_dim, 2),
            entity_kind::vertex,
            graph,
            report);
        compare_vec("vertex_neighborhood_2", vh, graph.vertex_neighborhood(v, 2), expected, report, log, max_logged_mismatches);
      }
    }
  }

  if (has_edges) {
    for (index_type e = 0; e < graph.entity_count(entity_kind::edge); ++e) {
      const handle_type eh = graph.handle(entity_kind::edge, e);
      const auto native_eh = static_cast<moab::EntityHandle>(eh);

      if (has_faces) {
        std::vector<moab::EntityHandle> native_adj;
        check_moab(
            native.get_adjacencies(
                &native_eh,
                1,
                2,
                true,
                native_adj,
                moab::Interface::UNION),
            "get_adjacencies(edge->face)");
        const auto expected = map_native_handles_to_local_ids(native_adj, entity_kind::face, graph, report);
        compare_row("edge_to_face", eh, graph.edge_to_faces(e), expected, report, log, max_logged_mismatches);
      }

      if (validate_rings) {
        moab::Range native_ring;
        check_moab(mtu.get_bridge_adjacencies(native_eh, 0, 1, native_ring), "mtu.get_bridge_adjacencies(edge ring)");
        const auto expected = map_native_range_to_local_ids(native_ring, entity_kind::edge, graph, report, eh);
        compare_vec("edge_ring", eh, graph.edge_ring(e, 1), expected, report, log, max_logged_mismatches);
      }

      if (validate_rings && elem_dim > 0) {
        {
          const auto expected = map_native_handles_to_local_ids(
              native_reference::edge_neighborhood_handles(native, native_eh, elem_dim, 1),
              entity_kind::edge,
              graph,
              report);
          compare_vec("edge_neighborhood_1", eh, graph.edge_neighborhood(e, 1), expected, report, log, max_logged_mismatches);
        }
        {
          const auto expected = map_native_handles_to_local_ids(
              native_reference::edge_neighborhood_handles(native, native_eh, elem_dim, 2),
              entity_kind::edge,
              graph,
              report);
          compare_vec("edge_neighborhood_2", eh, graph.edge_neighborhood(e, 2), expected, report, log, max_logged_mismatches);
        }
      }
    }
  }

  if (validate_rings && elem_dim > 0) {
    for (index_type e = 0; e < graph.entity_count(elem_kind); ++e) {
      const handle_type eh = graph.handle(elem_kind, e);
      const auto native_eh = static_cast<moab::EntityHandle>(eh);
      {
        const auto expected = map_native_handles_to_local_ids(
            native_reference::element_neighborhood_handles(native, native_eh, elem_dim, 1),
            elem_kind,
            graph,
            report);
        compare_vec("element_neighborhood_1", eh, graph.element_neighborhood(e, 1), expected, report, log, max_logged_mismatches);
      }
      {
        const auto expected = map_native_handles_to_local_ids(
            native_reference::element_neighborhood_handles(native, native_eh, elem_dim, 2),
            elem_kind,
            graph,
            report);
        compare_vec("element_neighborhood_2", eh, graph.element_neighborhood(e, 2), expected, report, log, max_logged_mismatches);
      }
    }
  }

  if (validate_rings && has_faces && has_edges) {
    for (index_type f = 0; f < graph.entity_count(entity_kind::face); ++f) {
      const handle_type fh = graph.handle(entity_kind::face, f);
      const auto native_fh = static_cast<moab::EntityHandle>(fh);

      moab::Range native_ring;
      check_moab(mtu.get_bridge_adjacencies(native_fh, 1, 2, native_ring), "mtu.get_bridge_adjacencies(face ring)");
      const auto expected = map_native_range_to_local_ids(native_ring, entity_kind::face, graph, report, fh);
      compare_vec("face_ring", fh, graph.face_ring(f, 1), expected, report, log, max_logged_mismatches);
    }
  }

  return report;
}

}  // namespace moabx::mesh
