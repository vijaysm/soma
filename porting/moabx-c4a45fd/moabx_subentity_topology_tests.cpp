// Verifies that the mirrored constexpr sub-entity tables in
// subentity_topology.hpp agree exactly with moab::CN, including vertex order.
//
// Order matters: canonical order encodes sub-entity sense, which the face
// orientation logic depends on. A test that only compared sets would pass while
// silently destroying orientation.

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "moab/CN.hpp"

#include "moabx/mesh/subentity_topology.hpp"

namespace {

int failures = 0;

void fail(const std::string& message) {
  std::cerr << "[FAIL] " << message << '\n';
  ++failures;
}

void expect(bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

moab::EntityType to_moab_type(moabx::mesh::element_topology topology) {
  using moabx::mesh::element_topology;
  switch (topology) {
    case element_topology::edge: return moab::MBEDGE;
    case element_topology::triangle: return moab::MBTRI;
    case element_topology::quadrilateral: return moab::MBQUAD;
    case element_topology::tetrahedron: return moab::MBTET;
    case element_topology::pyramid: return moab::MBPYRAMID;
    case element_topology::prism: return moab::MBPRISM;
    case element_topology::knife: return moab::MBKNIFE;
    case element_topology::hexahedron: return moab::MBHEX;
    default: return moab::MBMAXTYPE;
  }
}

moabx::mesh::element_topology from_moab_sub_type(moab::EntityType type) {
  using moabx::mesh::element_topology;
  switch (type) {
    case moab::MBEDGE: return element_topology::edge;
    case moab::MBTRI: return element_topology::triangle;
    case moab::MBQUAD: return element_topology::quadrilateral;
    default: return element_topology::unknown;
  }
}

void check_topology(moabx::mesh::element_topology topology) {
  const moab::EntityType moab_type = to_moab_type(topology);
  const std::string name = moabx::mesh::element_topology_name(topology);

  expect(
      moabx::mesh::vertices_per_element(topology) ==
          static_cast<std::size_t>(moab::CN::VerticesPerEntity(moab_type)),
      name + ": vertices_per_element disagrees with CN::VerticesPerEntity");

  expect(
      moabx::mesh::topological_dimension_of(topology) ==
          static_cast<int>(moab::CN::Dimension(moab_type)),
      name + ": topological_dimension_of disagrees with CN::Dimension");

  for (int dim = 1; dim <= 2; ++dim) {
    const auto mirrored = moabx::mesh::subentities(topology, dim);
    const int expected_count = moab::CN::NumSubEntities(moab_type, dim);

    if (expected_count <= 0) {
      expect(
          mirrored.empty(),
          name + " d=" + std::to_string(dim) + ": CN reports no sub-entities but table is non-empty");
      continue;
    }

    if (mirrored.size() != static_cast<std::size_t>(expected_count)) {
      fail(
          name + " d=" + std::to_string(dim) + ": sub-entity count " +
          std::to_string(mirrored.size()) + " != CN " + std::to_string(expected_count));
      continue;
    }

    for (int i = 0; i < expected_count; ++i) {
      moab::EntityType sub_type = moab::MBMAXTYPE;
      int sub_vertex_count = 0;
      const short* indices =
          moab::CN::SubEntityVertexIndices(moab_type, dim, i, sub_type, sub_vertex_count);

      const auto& def = mirrored[static_cast<std::size_t>(i)];
      const std::string where = name + " d=" + std::to_string(dim) + " sub=" + std::to_string(i);

      if (def.vertex_count != static_cast<std::uint8_t>(sub_vertex_count)) {
        fail(
            where + ": vertex count " + std::to_string(def.vertex_count) + " != CN " +
            std::to_string(sub_vertex_count));
        continue;
      }

      expect(
          def.topology == from_moab_sub_type(sub_type),
          where + ": sub-entity topology disagrees with CN");

      for (int k = 0; k < sub_vertex_count; ++k) {
        // Exact positional comparison: canonical order carries the sense.
        if (def.vertices[static_cast<std::size_t>(k)] != static_cast<std::uint8_t>(indices[k])) {
          fail(
              where + " vertex " + std::to_string(k) + ": " +
              std::to_string(def.vertices[static_cast<std::size_t>(k)]) + " != CN " +
              std::to_string(indices[k]) + " (canonical order encodes sense)");
        }
      }
    }
  }
}

void test_variable_arity_reported_correctly() {
  using moabx::mesh::element_topology;
  expect(!moabx::mesh::has_fixed_topology(element_topology::polygon),
         "polygon must not report fixed topology");
  expect(!moabx::mesh::has_fixed_topology(element_topology::polyhedron),
         "polyhedron must not report fixed topology");
  expect(!moabx::mesh::has_fixed_topology(element_topology::unknown),
         "unknown must not report fixed topology");
  expect(moabx::mesh::has_fixed_topology(element_topology::hexahedron),
         "hexahedron must report fixed topology");

  // CN itself reports zero sub-entities for polygons; this is the trap that
  // silently produced zero derived edges during feasibility work.
  expect(moab::CN::NumSubEntities(moab::MBPOLYGON, 1) == 0,
         "expected CN::NumSubEntities(MBPOLYGON, 1) == 0");
  expect(moabx::mesh::subentities(element_topology::polygon, 1).empty(),
         "polygon sub-entity table must be empty");
  expect(moabx::mesh::subentities(element_topology::polyhedron, 2).empty(),
         "polyhedron sub-entity table must be empty");
}

void test_subentity_indices_in_range() {
  using moabx::mesh::element_topology;
  for (const auto topology : {
           element_topology::edge, element_topology::triangle,
           element_topology::quadrilateral, element_topology::tetrahedron,
           element_topology::pyramid, element_topology::prism,
           element_topology::knife, element_topology::hexahedron}) {
    const std::size_t nverts = moabx::mesh::vertices_per_element(topology);
    const std::string name = moabx::mesh::element_topology_name(topology);
    for (int dim = 1; dim <= 2; ++dim) {
      for (const auto& def : moabx::mesh::subentities(topology, dim)) {
        for (std::uint8_t k = 0; k < def.vertex_count; ++k) {
          expect(
              def.vertices[k] < nverts,
              name + ": sub-entity vertex index " + std::to_string(def.vertices[k]) +
                  " out of range for " + std::to_string(nverts) + " vertices");
        }
      }
    }
  }
}

}  // namespace

int main() {
  using moabx::mesh::element_topology;

  for (const auto topology : {
           element_topology::edge, element_topology::triangle,
           element_topology::quadrilateral, element_topology::tetrahedron,
           element_topology::pyramid, element_topology::prism,
           element_topology::knife, element_topology::hexahedron}) {
    check_topology(topology);
  }

  test_variable_arity_reported_correctly();
  test_subentity_indices_in_range();

  if (failures != 0) {
    std::cerr << "[FAIL] " << failures << " subentity topology check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "[PASS] subentity topology tables match moab::CN\n";
  return EXIT_SUCCESS;
}
