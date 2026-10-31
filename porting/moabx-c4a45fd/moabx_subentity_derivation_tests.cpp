// Phase 1.5c/1.5d: validate implicitly derived sub-entities against MOAB.
//
// 1.5c parity:      derived edge/face set == get_adjacencies(create_if_missing)
// 1.5d orientation: every interior face carries exactly one +1 and one -1
//                   incidence; every boundary face exactly one incidence.
//
// Parity alone is insufficient: it compares sets and is blind to sense. The
// orientation check is what establishes that consistent outward normals are
// recoverable without storing them.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "moab/CN.hpp"
#include "moab/Core.hpp"
#include "moab/Range.hpp"

#include "moabx/mesh/subentity_derivation.hpp"
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

moabx::mesh::element_topology from_moab_type(moab::EntityType type) {
  using moabx::mesh::element_topology;
  switch (type) {
    case moab::MBEDGE: return element_topology::edge;
    case moab::MBTRI: return element_topology::triangle;
    case moab::MBQUAD: return element_topology::quadrilateral;
    case moab::MBPOLYGON: return element_topology::polygon;
    case moab::MBTET: return element_topology::tetrahedron;
    case moab::MBPYRAMID: return element_topology::pyramid;
    case moab::MBPRISM: return element_topology::prism;
    case moab::MBKNIFE: return element_topology::knife;
    case moab::MBHEX: return element_topology::hexahedron;
    case moab::MBPOLYHEDRON: return element_topology::polyhedron;
    default: return element_topology::unknown;
  }
}

using vertex_key = std::vector<moabx::mesh::index_type>;

vertex_key sorted_of(std::span<const moabx::mesh::index_type> vertices) {
  vertex_key key(vertices.begin(), vertices.end());
  std::sort(key.begin(), key.end());
  return key;
}

struct loaded_mesh {
  moab::Core core;
  moab::Range elements;
  int element_dimension = 0;
  std::vector<moabx::mesh::index_type> offsets{0};
  std::vector<moabx::mesh::index_type> indices;
  std::vector<moabx::mesh::element_topology> topologies;
  std::map<moab::EntityHandle, moabx::mesh::index_type> vertex_ids;
};

bool load(loaded_mesh& mesh, const std::string& path) {
  if (mesh.core.load_file(path.c_str()) != moab::MB_SUCCESS) {
    fail("could not load " + path);
    return false;
  }

  const moab::EntityHandle root = 0;
  moab::Range vertices;
  mesh.core.get_entities_by_type(root, moab::MBVERTEX, vertices, true);
  moabx::mesh::index_type next = 0;
  for (const moab::EntityHandle v : vertices) {
    mesh.vertex_ids.emplace(v, next++);
  }

  moab::Range regions;
  moab::Range faces;
  mesh.core.get_entities_by_dimension(root, 3, regions, true);
  mesh.core.get_entities_by_dimension(root, 2, faces, true);
  mesh.element_dimension = regions.empty() ? 2 : 3;
  mesh.elements = regions.empty() ? faces : regions;

  for (const moab::EntityHandle e : mesh.elements) {
    const moab::EntityHandle* conn = nullptr;
    int nnodes = 0;
    if (mesh.core.get_connectivity(e, conn, nnodes, true) != moab::MB_SUCCESS) {
      fail("get_connectivity failed");
      return false;
    }
    for (int i = 0; i < nnodes; ++i) {
      mesh.indices.push_back(mesh.vertex_ids.at(conn[i]));
    }
    mesh.offsets.push_back(static_cast<moabx::mesh::index_type>(mesh.indices.size()));
    mesh.topologies.push_back(from_moab_type(mesh.core.type_from_handle(e)));
  }
  return true;
}

// 1.5c: compare the derived sub-entity set against MOAB's, as sorted
// local-vertex-id tuples.
void check_parity(loaded_mesh& mesh, const std::string& label, int sub_dimension) {
  moabx::mesh::element_topology_view view{
      mesh.offsets, mesh.indices, mesh.topologies};
  const auto derived = moabx::mesh::derive_subentities(view, sub_dimension);

  const std::string what = label + " d=" + std::to_string(sub_dimension);

  expect(
      derived.unsupported_elements.empty(),
      what + ": " + std::to_string(derived.unsupported_elements.size()) +
          " element(s) had no derivation path");

  moab::Range reference;
  mesh.core.get_adjacencies(mesh.elements, sub_dimension, true, reference, moab::Interface::UNION);

  std::vector<vertex_key> reference_keys;
  reference_keys.reserve(reference.size());
  for (const moab::EntityHandle h : reference) {
    const moab::EntityHandle* conn = nullptr;
    int nnodes = 0;
    mesh.core.get_connectivity(h, conn, nnodes, true);
    vertex_key key;
    key.reserve(static_cast<std::size_t>(nnodes));
    for (int i = 0; i < nnodes; ++i) {
      key.push_back(mesh.vertex_ids.at(conn[i]));
    }
    std::sort(key.begin(), key.end());
    reference_keys.push_back(std::move(key));
  }

  std::vector<vertex_key> derived_keys;
  derived_keys.reserve(derived.count());
  for (moabx::mesh::index_type s = 0; s < derived.count(); ++s) {
    derived_keys.push_back(sorted_of(derived.vertices(s)));
  }

  std::sort(reference_keys.begin(), reference_keys.end());
  std::sort(derived_keys.begin(), derived_keys.end());

  if (derived_keys.size() != reference_keys.size()) {
    fail(
        what + ": derived " + std::to_string(derived_keys.size()) + " != MOAB " +
        std::to_string(reference_keys.size()));
    return;
  }
  if (derived_keys != reference_keys) {
    fail(what + ": same count but differing sub-entity sets");
    return;
  }

  std::cout << "  [ok] " << what << " parity: " << derived_keys.size() << " sub-entities\n";
}

// 1.5d: orientation.
//
// The strict "one +1 and one -1" rule applies only to codimension-1
// sub-entities -- faces of a 3D mesh, edges of a 2D mesh -- because only those
// separate exactly two elements and therefore admit a well-defined outward
// normal. An edge of a 3D mesh is shared by arbitrarily many elements and has
// no such constraint, so for that case we verify only that every sense is
// determinate (+1 or -1) and that the derived ordering is self-consistent.
void check_orientation(loaded_mesh& mesh, const std::string& label, int sub_dimension) {
  const bool codimension_one = (sub_dimension == mesh.element_dimension - 1);
  moabx::mesh::element_topology_view view{
      mesh.offsets, mesh.indices, mesh.topologies};
  const auto derived = moabx::mesh::derive_subentities(view, sub_dimension);

  const std::string what = label + " d=" + std::to_string(sub_dimension);

  std::size_t boundary = 0;
  std::size_t interior = 0;
  std::size_t bad_sense = 0;
  std::size_t non_conforming = 0;
  std::size_t inconsistent = 0;

  for (moabx::mesh::index_type s = 0; s < derived.count(); ++s) {
    const auto incidences = derived.incidences_of(s);

    int positive = 0;
    int negative = 0;
    for (const auto& incidence : incidences) {
      if (incidence.sense == 1) {
        ++positive;
      } else if (incidence.sense == -1) {
        ++negative;
      } else {
        ++bad_sense;
      }
    }

    if (incidences.size() == 1) {
      ++boundary;
      if (codimension_one && positive != 1) {
        ++inconsistent;
      }
    } else if (incidences.size() == 2) {
      ++interior;
      // Opposing senses are what make a consistent outward normal definable.
      // Only meaningful at codimension 1.
      if (codimension_one && (positive != 1 || negative != 1)) {
        ++inconsistent;
      }
    } else {
      ++non_conforming;
    }
  }

  expect(bad_sense == 0, what + ": " + std::to_string(bad_sense) + " incidence(s) with undetermined sense");
  expect(
      inconsistent == 0,
      what + ": " + std::to_string(inconsistent) + " sub-entity(ies) without consistent opposing senses");

  if (codimension_one) {
    // A closed conforming mesh with no boundary would have zero; any real mesh
    // here has one. What must not happen is a codim-1 entity shared by 3+
    // elements, which would mean a non-manifold surface.
    expect(
        non_conforming == 0,
        what + ": " + std::to_string(non_conforming) +
            " codimension-1 sub-entity(ies) shared by 3+ elements (non-manifold)");
  }

  std::cout << "  [ok] " << what << " orientation: interior=" << interior
            << " boundary=" << boundary;
  if (non_conforming != 0) {
    std::cout << " shared_by_3plus=" << non_conforming;
  }
  if (!codimension_one) {
    std::cout << " (codim>1: sense determinacy only)";
  }
  std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <mesh> [mesh...]\n";
    return EXIT_FAILURE;
  }

  for (int i = 1; i < argc; ++i) {
    const std::string path = argv[i];
    std::cout << "[mesh] " << path << '\n';

    loaded_mesh mesh;
    if (!load(mesh, path)) {
      continue;
    }

    check_parity(mesh, path, 1);
    check_orientation(mesh, path, 1);

    if (mesh.element_dimension == 3) {
      check_parity(mesh, path, 2);
      check_orientation(mesh, path, 2);
    }
  }

  if (failures != 0) {
    std::cerr << "[FAIL] " << failures << " check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "[PASS] derived sub-entities match MOAB with consistent orientation\n";
  return EXIT_SUCCESS;
}
