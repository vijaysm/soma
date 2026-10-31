#ifndef MOABX_MESH_SUBENTITY_TOPOLOGY_HPP_
#define MOABX_MESH_SUBENTITY_TOPOLOGY_HPP_

// Canonical sub-entity tables for fixed-topology element types.
//
// These mirror MOAB's moab::CN tables but carry no MOAB dependency, so the
// derivation path is a pure function of connectivity plus constant data. That
// keeps it parallelizable and, once PP_HOST_DEVICE is applied project-wide,
// device-callable.
//
// The tables were generated from moab::CN::SubEntityVertexIndices and are
// verified against it by moabx_subentity_topology_tests, which fails if MOAB's
// canonical ordering ever diverges from this copy. Do not hand-edit: regenerate
// and re-run the parity test.
//
// Variable-arity types (polygon, polyhedron) are deliberately absent. A polygon
// has no fixed table -- its edges are its vertex cycle -- and a polyhedron's
// faces are stored explicitly rather than derived. Callers must handle both;
// `has_fixed_topology()` reports which.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace moabx::mesh {

// Element topologies with a fixed canonical sub-entity table.
// Values are independent of moab::EntityType; conversion lives in the loader.
enum class element_topology : std::uint8_t {
  unknown = 0,
  edge,
  triangle,
  quadrilateral,
  tetrahedron,
  pyramid,
  prism,
  knife,
  hexahedron,
  polygon,     // variable arity: edges are the vertex cycle
  polyhedron,  // faces stored explicitly; not derivable
};

constexpr const char* element_topology_name(element_topology topology) noexcept {
  switch (topology) {
    case element_topology::edge: return "edge";
    case element_topology::triangle: return "triangle";
    case element_topology::quadrilateral: return "quadrilateral";
    case element_topology::tetrahedron: return "tetrahedron";
    case element_topology::pyramid: return "pyramid";
    case element_topology::prism: return "prism";
    case element_topology::knife: return "knife";
    case element_topology::hexahedron: return "hexahedron";
    case element_topology::polygon: return "polygon";
    case element_topology::polyhedron: return "polyhedron";
    case element_topology::unknown:
    default: return "unknown";
  }
}

// True when a constant sub-entity table exists for this topology.
constexpr bool has_fixed_topology(element_topology topology) noexcept {
  switch (topology) {
    case element_topology::polygon:
    case element_topology::polyhedron:
    case element_topology::unknown:
      return false;
    default:
      return true;
  }
}

// Maximum vertices in any single sub-entity across all fixed topologies (quad).
constexpr std::size_t max_subentity_vertices = 4;

struct subentity_def {
  element_topology topology = element_topology::unknown;
  std::uint8_t vertex_count = 0;
  std::array<std::uint8_t, max_subentity_vertices> vertices{};
};

namespace detail {

constexpr subentity_def make_edge(std::uint8_t a, std::uint8_t b) noexcept {
  return subentity_def{element_topology::edge, 2, {a, b, 0, 0}};
}
constexpr subentity_def make_tri(std::uint8_t a, std::uint8_t b, std::uint8_t c) noexcept {
  return subentity_def{element_topology::triangle, 3, {a, b, c, 0}};
}
constexpr subentity_def make_quad(std::uint8_t a, std::uint8_t b, std::uint8_t c, std::uint8_t d) noexcept {
  return subentity_def{element_topology::quadrilateral, 4, {a, b, c, d}};
}

// --- edges (dimension 1) -------------------------------------------------

inline constexpr std::array<subentity_def, 1> edge_edges{make_edge(0, 1)};

inline constexpr std::array<subentity_def, 3> tri_edges{
    make_edge(0, 1), make_edge(1, 2), make_edge(2, 0)};

inline constexpr std::array<subentity_def, 4> quad_edges{
    make_edge(0, 1), make_edge(1, 2), make_edge(2, 3), make_edge(3, 0)};

inline constexpr std::array<subentity_def, 6> tet_edges{
    make_edge(0, 1), make_edge(1, 2), make_edge(2, 0),
    make_edge(0, 3), make_edge(1, 3), make_edge(2, 3)};

inline constexpr std::array<subentity_def, 8> pyramid_edges{
    make_edge(0, 1), make_edge(1, 2), make_edge(2, 3), make_edge(3, 0),
    make_edge(0, 4), make_edge(1, 4), make_edge(2, 4), make_edge(3, 4)};

inline constexpr std::array<subentity_def, 9> prism_edges{
    make_edge(0, 1), make_edge(1, 2), make_edge(2, 0),
    make_edge(0, 3), make_edge(1, 4), make_edge(2, 5),
    make_edge(3, 4), make_edge(4, 5), make_edge(5, 3)};

inline constexpr std::array<subentity_def, 10> knife_edges{
    make_edge(0, 1), make_edge(1, 2), make_edge(2, 3), make_edge(3, 0),
    make_edge(0, 4), make_edge(1, 5), make_edge(2, 6), make_edge(3, 5),
    make_edge(4, 5), make_edge(5, 6)};

inline constexpr std::array<subentity_def, 12> hex_edges{
    make_edge(0, 1), make_edge(1, 2), make_edge(2, 3), make_edge(3, 0),
    make_edge(0, 4), make_edge(1, 5), make_edge(2, 6), make_edge(3, 7),
    make_edge(4, 5), make_edge(5, 6), make_edge(6, 7), make_edge(7, 4)};

// --- faces (dimension 2) -------------------------------------------------
//
// Vertex order is the canonical order relative to the parent element and
// therefore encodes the face sense. Preserve it exactly; sorting here would
// destroy the orientation information the normal computation depends on.

inline constexpr std::array<subentity_def, 1> tri_faces{make_tri(0, 1, 2)};

inline constexpr std::array<subentity_def, 1> quad_faces{make_quad(0, 1, 2, 3)};

inline constexpr std::array<subentity_def, 4> tet_faces{
    make_tri(0, 1, 3), make_tri(1, 2, 3), make_tri(0, 3, 2), make_tri(0, 2, 1)};

inline constexpr std::array<subentity_def, 5> pyramid_faces{
    make_tri(0, 1, 4), make_tri(1, 2, 4), make_tri(2, 3, 4), make_tri(3, 0, 4),
    make_quad(0, 3, 2, 1)};

inline constexpr std::array<subentity_def, 5> prism_faces{
    make_quad(0, 1, 4, 3), make_quad(1, 2, 5, 4), make_quad(0, 3, 5, 2),
    make_tri(0, 2, 1), make_tri(3, 4, 5)};

inline constexpr std::array<subentity_def, 5> knife_faces{
    make_quad(0, 1, 5, 4), make_quad(1, 2, 6, 5), make_quad(2, 3, 5, 6),
    make_quad(3, 0, 4, 5), make_quad(0, 3, 2, 1)};

inline constexpr std::array<subentity_def, 6> hex_faces{
    make_quad(0, 1, 5, 4), make_quad(1, 2, 6, 5), make_quad(2, 3, 7, 6),
    make_quad(3, 0, 4, 7), make_quad(0, 3, 2, 1), make_quad(4, 5, 6, 7)};

}  // namespace detail

// Corner vertex count, or 0 for variable-arity topologies.
constexpr std::size_t vertices_per_element(element_topology topology) noexcept {
  switch (topology) {
    case element_topology::edge: return 2;
    case element_topology::triangle: return 3;
    case element_topology::quadrilateral: return 4;
    case element_topology::tetrahedron: return 4;
    case element_topology::pyramid: return 5;
    case element_topology::prism: return 6;
    case element_topology::knife: return 7;
    case element_topology::hexahedron: return 8;
    default: return 0;
  }
}

constexpr int topological_dimension_of(element_topology topology) noexcept {
  switch (topology) {
    case element_topology::edge: return 1;
    case element_topology::triangle:
    case element_topology::quadrilateral:
    case element_topology::polygon: return 2;
    case element_topology::tetrahedron:
    case element_topology::pyramid:
    case element_topology::prism:
    case element_topology::knife:
    case element_topology::hexahedron:
    case element_topology::polyhedron: return 3;
    default: return 0;
  }
}

// Canonical sub-entities of `topology` at `sub_dimension` (1 = edges,
// 2 = faces). Empty for variable-arity topologies or unsupported combinations.
constexpr std::span<const subentity_def> subentities(
    element_topology topology,
    int sub_dimension) noexcept {
  if (sub_dimension == 1) {
    switch (topology) {
      case element_topology::edge: return detail::edge_edges;
      case element_topology::triangle: return detail::tri_edges;
      case element_topology::quadrilateral: return detail::quad_edges;
      case element_topology::tetrahedron: return detail::tet_edges;
      case element_topology::pyramid: return detail::pyramid_edges;
      case element_topology::prism: return detail::prism_edges;
      case element_topology::knife: return detail::knife_edges;
      case element_topology::hexahedron: return detail::hex_edges;
      default: return {};
    }
  }
  if (sub_dimension == 2) {
    switch (topology) {
      case element_topology::triangle: return detail::tri_faces;
      case element_topology::quadrilateral: return detail::quad_faces;
      case element_topology::tetrahedron: return detail::tet_faces;
      case element_topology::pyramid: return detail::pyramid_faces;
      case element_topology::prism: return detail::prism_faces;
      case element_topology::knife: return detail::knife_faces;
      case element_topology::hexahedron: return detail::hex_faces;
      default: return {};
    }
  }
  return {};
}

constexpr std::size_t subentity_count(element_topology topology, int sub_dimension) noexcept {
  return subentities(topology, sub_dimension).size();
}

}  // namespace moabx::mesh

#endif  // MOABX_MESH_SUBENTITY_TOPOLOGY_HPP_
