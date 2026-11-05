// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// read_h5m against MOAB's load_file on the same files: entity counts per
// type, coordinates and connectivity bitwise, and every tag MOAB holds on
// vertices and elements, value by value, with its default and global value.
// The "written:" files are those files read by read_h5m and written again by
// write_h5m, so MOAB is checked to read what SOMA writes; the "compressed:"
// ones are written so at deflate level 6; the "converted:" ones are converted
// to SOMA files and back by convert_mesh_file (soma-convert).
//
// MOAB numbers a freshly loaded file's entities of one type in file id
// order, so the i-th handle of a type is the entity with the i-th smallest
// file id of that type. Nothing here trusts that blindly: a different order
// shows up as coordinates, connectivity or tag values that disagree.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "h5m_writer.hpp"
#include "mesh_text.hpp"
#include "moab/CN.hpp"
#include "moab/Core.hpp"
#include "moab/Range.hpp"
#include "soma/core/handle_ref.hpp"
#include "soma/core/mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/fields/var_field.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/topology/build.hpp"

namespace fs = std::filesystem;

namespace soma_test {

void check(moab::ErrorCode code, const std::string& what) {
  if (code != moab::MB_SUCCESS) {
    throw std::runtime_error("MOAB failure in " + what + ": error " + std::to_string(static_cast<int>(code)));
  }
}

constexpr moab::EntityType moab_types[] = {moab::MBVERTEX, moab::MBEDGE,    moab::MBTRI,
                                           moab::MBQUAD,   moab::MBPOLYGON, moab::MBTET,
                                           moab::MBPYRAMID, moab::MBPRISM,  moab::MBHEX};

// One scratch directory per run, removed when the program ends.
fs::path scratch_dir() {
  struct Scratch {
    fs::path dir;
    Scratch() {
      const char* t = std::getenv("TMPDIR");
      std::string pattern = ((t != nullptr && *t != '\0') ? fs::path(t) : fs::path("/tmp")) / "soma_h5mm_XXXXXX";
      if (mkdtemp(pattern.data()) == nullptr) throw std::runtime_error("cannot create " + pattern);
      dir = pattern;
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    ~Scratch() {
      std::error_code ignored;
      fs::remove_all(dir, ignored);
    }
  };
  static const Scratch s;
  return s.dir;
}

// A file MOAB writes itself, with a tag of every fixed-length kind on entities.
fs::path tag_kinds_file() {
  const fs::path p = scratch_dir() / "tag_kinds.h5m";
  if (fs::exists(p)) return p;
  moab::Core mb;
  const double xyz[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0, 2, 1, 0};
  moab::Range verts;
  check(mb.create_vertices(xyz, 6, verts), "create_vertices");
  const std::vector<moab::EntityHandle> v(verts.begin(), verts.end());
  moab::EntityHandle q = 0, t1 = 0, t2 = 0, poly = 0;
  check(mb.create_element(moab::MBQUAD, v.data(), 4, q), "create quad");
  const moab::EntityHandle tc1[] = {v[1], v[4], v[2]};
  check(mb.create_element(moab::MBTRI, tc1, 3, t1), "create tri");
  const moab::EntityHandle tc2[] = {v[4], v[5], v[2]};
  check(mb.create_element(moab::MBTRI, tc2, 3, t2), "create tri");
  const moab::EntityHandle pc[] = {v[0], v[1], v[4], v[5], v[3]};
  check(mb.create_element(moab::MBPOLYGON, pc, 5, poly), "create polygon");

  moab::Tag tag = nullptr;
  const double ddef[] = {9, 8};
  check(mb.tag_get_handle("DENSE_D2", 2, moab::MB_TYPE_DOUBLE, tag, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, ddef),
        "DENSE_D2");
  const double dv[] = {1.25, -2.5};
  check(mb.tag_set_data(tag, &v[0], 1, dv), "set DENSE_D2");
  check(mb.tag_set_data(tag, &t2, 1, dv), "set DENSE_D2");
  const int idef = -7;
  check(mb.tag_get_handle("SPARSE_I", 1, moab::MB_TYPE_INTEGER, tag, moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT,
                          &idef),
        "SPARSE_I");
  const int iv[] = {3, 4};
  check(mb.tag_set_data(tag, &v[3], 1, &iv[0]), "set SPARSE_I");
  check(mb.tag_set_data(tag, &t1, 1, &iv[1]), "set SPARSE_I");
  // A sparse value equal to the default is still a value.
  check(mb.tag_set_data(tag, &v[5], 1, &idef), "set SPARSE_I");
  const int pair[] = {5, 6};
  check(mb.tag_get_handle("PAIR", 2, moab::MB_TYPE_INTEGER, tag, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT),
        "PAIR");
  check(mb.tag_set_data(tag, &poly, 1, pair), "set PAIR");
  check(mb.tag_get_handle("OPAQUE7", 7, moab::MB_TYPE_OPAQUE, tag,
                          moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT | moab::MB_TAG_BYTES),
        "OPAQUE7");
  const char text[7] = {'a', 'b', 'c', 'd', 'e', 'f', 'g'};
  check(mb.tag_set_data(tag, &q, 1, text), "set OPAQUE7");
  const unsigned char bdef = 2;
  check(mb.tag_get_handle("BITS3", 3, moab::MB_TYPE_BIT, tag, moab::MB_TAG_BIT | moab::MB_TAG_CREAT, &bdef),
        "BITS3");
  const unsigned char bits = 5;
  check(mb.tag_set_data(tag, &v[1], 1, &bits), "set BITS3");
  const int gval = 11;
  check(mb.tag_get_handle("WITH_GLOBAL", 1, moab::MB_TYPE_INTEGER, tag, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT,
                          &idef),
        "WITH_GLOBAL");
  const moab::EntityHandle root = 0;
  check(mb.tag_set_data(tag, &root, 1, &gval), "set WITH_GLOBAL on the mesh");
  check(mb.tag_set_data(tag, &q, 1, &iv[0]), "set WITH_GLOBAL");
  check(mb.write_file(p.c_str()), "write_file");
  return p;
}

// A file of variable-arity polygons (connectivity plus poly_indices), a
// layout MOAB reads but no longer writes.
fs::path poly_indices_file() {
  const fs::path p = scratch_dir() / "poly_indices.h5m";
  if (fs::exists(p)) return p;
  h5m::Writer w(p);
  w.nodes(1, 3, {0, 0, 0, 1, 0, 0, 2, 0, 0, 2, 1, 0, 1, 1, 0, 0, 1, 0, 1, 2, 0});
  w.poly_elements("Polygon", "Polygon", 8, {1, 2, 5, 6, 2, 3, 4, 7, 5}, {3, 8});
  w.no_sets();
  const std::int32_t def = -1;
  w.tag("GLOBAL_ID", 2, H5T_NATIVE_INT32, &def, &def);
  const std::int32_t gid[] = {10, 20};
  w.dense("elements/Polygon", "GLOBAL_ID", gid, 2);
  return p;
}

// A file MOAB writes itself with every kind of set and every tag kind that is
// not of fixed length on entities: ordered sets with repeats, sets inside
// sets, a parent-child cycle, an empty set, tags on sets, variable-length,
// handle and mesh tags.
fs::path sets_kinds_file() {
  const fs::path p = scratch_dir() / "sets_kinds.h5m";
  if (fs::exists(p)) return p;
  moab::Core mb;
  std::vector<double> xyz;  // a 3 x 2 grid of quads
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 4; ++i) xyz.insert(xyz.end(), {double(i), double(j), 0});
  }
  moab::Range verts;
  check(mb.create_vertices(xyz.data(), 12, verts), "create_vertices");
  const std::vector<moab::EntityHandle> v(verts.begin(), verts.end());
  std::vector<moab::EntityHandle> q;
  for (std::size_t j = 0; j < 2; ++j) {
    for (std::size_t i = 0; i < 3; ++i) {
      const moab::EntityHandle c[4] = {v[i + 4 * j], v[i + 1 + 4 * j], v[i + 1 + 4 * (j + 1)], v[i + 4 * (j + 1)]};
      moab::EntityHandle h = 0;
      check(mb.create_element(moab::MBQUAD, c, 4, h), "create quad");
      q.push_back(h);
    }
  }
  moab::EntityHandle ordered = 0, unordered = 0, contig = 0, empty = 0, parent = 0, child = 0, runs = 0;
  check(mb.create_meshset(moab::MESHSET_ORDERED, ordered), "ordered");
  const std::vector<moab::EntityHandle> oc{q[2], v[0], v[1], v[2], v[3], q[2], v[5]};
  check(mb.add_entities(ordered, oc.data(), static_cast<int>(oc.size())), "fill ordered");
  check(mb.create_meshset(moab::MESHSET_SET, unordered), "unordered");
  check(mb.add_entities(unordered, &q[0], 6), "fill unordered");
  check(mb.add_entities(unordered, &v[0], 12), "fill unordered");
  check(mb.create_meshset(moab::MESHSET_SET | moab::MESHSET_TRACK_OWNER, contig), "contig");
  check(mb.add_entities(contig, &v[4], 4), "fill contig");
  check(mb.create_meshset(moab::MESHSET_SET, empty), "empty");
  check(mb.create_meshset(moab::MESHSET_ORDERED, parent), "parent");
  check(mb.create_meshset(moab::MESHSET_SET, child), "child");
  check(mb.add_entities(parent, &unordered, 1), "fill parent");
  check(mb.add_entities(parent, &contig, 1), "fill parent");
  check(mb.add_entities(parent, &q[1], 1), "fill parent");
  check(mb.add_parent_child(parent, child), "link");
  check(mb.add_parent_child(child, parent), "link back");  // a cycle
  check(mb.add_parent_child(parent, contig), "link");
  check(mb.create_meshset(moab::MESHSET_ORDERED, runs), "runs");
  check(mb.add_entities(runs, &v[0], 12), "fill runs");

  moab::Tag tag = nullptr;
  const double vdef[3] = {-1, -2, -3};
  check(mb.tag_get_handle("VARF", 3, moab::MB_TYPE_DOUBLE, tag,
                          moab::MB_TAG_VARLEN | moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT, vdef),
        "VARF");
  for (int i = 0; i < 4; ++i) {
    const std::vector<double> val(static_cast<std::size_t>(i + 1), 10.0 * i);
    const void* ptr = val.data();
    const int n = i + 1;
    check(mb.tag_set_by_ptr(tag, &v[static_cast<std::size_t>(i)], 1, &ptr, &n), "set VARF");
  }
  check(mb.tag_get_handle("VAROPQ", 0, moab::MB_TYPE_OPAQUE, tag,
                          moab::MB_TAG_VARLEN | moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT),
        "VAROPQ");
  {
    const char text[] = "hello";
    const void* ptr = text;
    const int n = 5;
    check(mb.tag_set_by_ptr(tag, &ordered, 1, &ptr, &n), "set VAROPQ");
  }
  check(mb.tag_get_handle("QV", 1, moab::MB_TYPE_HANDLE, tag, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT), "QV");
  for (const moab::EntityHandle e : q) {
    const moab::EntityHandle* c = nullptr;
    int n = 0;
    check(mb.get_connectivity(e, c, n), "get_connectivity");
    check(mb.tag_set_data(tag, &e, 1, &c[0]), "set QV");
  }
  check(mb.tag_get_handle("SS", 1, moab::MB_TYPE_HANDLE, tag, moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT), "SS");
  const moab::EntityHandle none = 0;
  check(mb.tag_set_data(tag, &parent, 1, &child), "set SS");
  check(mb.tag_set_data(tag, &unordered, 1, &none), "set SS");
  check(mb.tag_get_handle("VH", 0, moab::MB_TYPE_HANDLE, tag,
                          moab::MB_TAG_VARLEN | moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT),
        "VH");
  {
    const moab::EntityHandle hs[3] = {q[0], q[1], v[3]};
    const void* ptr = hs;
    const int n = 3;
    check(mb.tag_set_by_ptr(tag, &v[0], 1, &ptr, &n), "set VH");
  }
  check(mb.tag_get_handle("SETI", 1, moab::MB_TYPE_INTEGER, tag, moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT), "SETI");
  const int seti[] = {5, 6, 7};
  check(mb.tag_set_data(tag, &contig, 1, &seti[0]), "set SETI");
  check(mb.tag_set_data(tag, &empty, 1, &seti[1]), "set SETI");
  check(mb.tag_set_data(tag, &v[7], 1, &seti[2]), "set SETI");
  const moab::EntityHandle root = 0;
  check(mb.tag_get_handle("MESHVAL", 2, moab::MB_TYPE_INTEGER, tag, moab::MB_TAG_MESH | moab::MB_TAG_CREAT),
        "MESHVAL");
  const int mv[2] = {42, 43};
  check(mb.tag_set_data(tag, &root, 1, mv), "set MESHVAL");
  check(mb.tag_get_handle("VARG", 0, moab::MB_TYPE_INTEGER, tag,
                          moab::MB_TAG_VARLEN | moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT),
        "VARG");
  {
    const int gv[4] = {1, 2, 3, 4};
    const void* ptr = gv;
    const int n = 4;
    check(mb.tag_set_by_ptr(tag, &root, 1, &ptr, &n), "set VARG");
  }
  const double ddef = 0.5;
  check(mb.tag_get_handle("DECLARED", 1, moab::MB_TYPE_DOUBLE, tag, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &ddef),
        "DECLARED");
  check(mb.write_file(p.c_str()), "write_file");
  return p;
}

fs::path path_of(const std::string& name);

// The file `name` names, read by read_h5m and written again by write_h5m at
// the compression level.
fs::path path_of(const std::string& name);

// One element of each 3D type, apart, with a quad side of the hex, a tri side
// of the prism and an edge of the tet as explicit entities.
fs::path zoo_file() {
  const fs::path p = scratch_dir() / "zoo.h5m";
  if (fs::exists(p)) return p;
  moab::Core mb;
  const auto add = [&](moab::EntityType t, std::vector<std::array<double, 3>> xyz, double shift) {
    std::vector<moab::EntityHandle> v(xyz.size());
    for (std::size_t i = 0; i < xyz.size(); ++i) {
      xyz[i][0] += shift;
      check(mb.create_vertex(xyz[i].data(), v[i]), "create_vertex");
    }
    moab::EntityHandle e = 0;
    check(mb.create_element(t, v.data(), static_cast<int>(v.size()), e), "create_element");
    return v;
  };
  const auto side = [&](moab::EntityType t, std::vector<moab::EntityHandle> v) {
    moab::EntityHandle e = 0;
    check(mb.create_element(t, v.data(), static_cast<int>(v.size()), e), "create_element");
  };
  const auto hex = add(moab::MBHEX, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}, 0);
  const auto prism = add(moab::MBPRISM, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}}, 2);
  add(moab::MBPYRAMID, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1}}, 4);
  const auto tet = add(moab::MBTET, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, 6);
  side(moab::MBQUAD, {hex[0], hex[1], hex[5], hex[4]});
  side(moab::MBTRI, {prism[3], prism[4], prism[5]});
  side(moab::MBEDGE, {tet[0], tet[1]});
  check(mb.write_file(p.c_str()), "write_file");
  return p;
}

// "ho:<file>:<bits>": the file with the mid-edge (2), mid-face (4) and
// mid-region (8) nodes MOAB's convert_entities adds to every entity, or with
// bit 1 to every other element of the highest dimension only, which leaves
// neighbours that disagree on a shared side.
fs::path converted_file(const std::string& spec) {
  const auto colon = spec.rfind(':');
  const std::string base = spec.substr(0, colon);
  const int bits = std::stoi(spec.substr(colon + 1));
  std::string flat = base;
  std::replace(flat.begin(), flat.end(), ':', '_');
  if (!flat.ends_with(".h5m")) flat += ".h5m";
  const fs::path p = scratch_dir() / ("ho" + std::to_string(bits) + "_" + flat);
  if (fs::exists(p)) return p;
  moab::Core mb;
  check(mb.load_file(path_of(base).c_str()), "load_file " + base);
  // convert_entities takes a set; the root set it does not.
  moab::Range chosen;
  for (int dim = 3; dim > 0; --dim) {
    moab::Range r;
    check(mb.get_entities_by_dimension(0, dim, r), "entities");
    if ((bits & 1) == 0) {
      chosen.merge(r);
    } else if (chosen.empty()) {
      std::size_t i = 0;
      for (const moab::EntityHandle e : r) {
        if (i++ % 2 == 0) chosen.insert(e);
      }
    }
  }
  moab::EntityHandle set = 0;
  check(mb.create_meshset(moab::MESHSET_SET, set), "create_meshset");
  check(mb.add_entities(set, chosen), "add_entities");
  check(mb.convert_entities(set, (bits & 2) != 0, (bits & 4) != 0, (bits & 8) != 0), "convert_entities");
  check(mb.delete_entities(&set, 1), "delete the set");
  check(mb.write_file(p.c_str()), "write_file");
  return p;
}

fs::path written_file(const std::string& name, int compression) {
  std::string flat = name;
  std::replace(flat.begin(), flat.end(), ':', '_');
  const fs::path p = scratch_dir() / ((compression > 0 ? "compressed_" : "written_") + flat);
  if (!fs::exists(p)) soma::write_h5m(soma::read_h5m(path_of(name)), p, {.compression = compression});
  return p;
}

// The file converted to a SOMA file and back.
fs::path round_trip_file(const std::string& name) {
  std::string flat = name;
  std::replace(flat.begin(), flat.end(), ':', '_');
  const fs::path p = scratch_dir() / ("converted_" + flat);
  if (!fs::exists(p)) {
    const fs::path soma = scratch_dir() / ("converted_" + flat + ".soma");
    soma::convert_mesh_file(soma::Comm::self(), path_of(name), soma);
    soma::convert_mesh_file(soma::Comm::self(), soma, p);
  }
  return p;
}

// The file a test parameter names without its "written:", "compressed:" or
// "converted:".
std::string original_of(const std::string& name) {
  for (const std::string prefix : {"written:", "compressed:", "converted:"}) {
    if (name.rfind(prefix, 0) == 0) return name.substr(prefix.size());
  }
  return name;
}

fs::path path_of(const std::string& name) {
  if (name.rfind("written:", 0) == 0) return written_file(name.substr(8), 0);
  if (name.rfind("compressed:", 0) == 0) return written_file(name.substr(11), 6);
  if (name.rfind("converted:", 0) == 0) return round_trip_file(name.substr(10));
  if (name == "tag_kinds") return tag_kinds_file();
  if (name == "poly_indices") return poly_indices_file();
  if (name == "sets_kinds") return sets_kinds_file();
  if (name == "zoo") return zoo_file();
  if (name.rfind("ho:", 0) == 0) return converted_file(name.substr(3));
  return fs::path(SOMA_TEST_DATA_DIR) / name;
}

constexpr int slots[] = {0, 1, 2, 3, soma::set_dim};

soma::LO slot_count(const soma::HostMesh& m, int slot) {
  if (slot == soma::set_dim) return m.sets().count();
  return slot <= m.dim() ? m.count(slot) : 0;
}

std::size_t value_size(moab::DataType type) {
  switch (type) {
    case moab::MB_TYPE_INTEGER: return sizeof(int);
    case moab::MB_TYPE_DOUBLE: return sizeof(double);
    case moab::MB_TYPE_HANDLE: return sizeof(moab::EntityHandle);
    default: return 1;
  }
}

struct Loaded {
  moab::Core mb;
  soma::HostMesh mesh;
  soma::H5mReport report;
  // Per MOAB type: handles in handle order, and the SOMA lid of each.
  std::map<moab::EntityType, std::vector<moab::EntityHandle>> handles;
  std::map<moab::EntityType, std::vector<soma::LO>> lids;
  // MOAB's sets in handle order, which is the file's order and SOMA's.
  std::vector<moab::EntityHandle> sets;
  std::map<moab::EntityHandle, std::pair<int, soma::LO>> where;  // SOMA slot and lid

  explicit Loaded(const fs::path& p) {
    check(mb.load_file(p.c_str()), "load_file " + p.string());
    mesh = soma::read_h5m(p, {}, report);
    for (const moab::EntityType t : moab_types) {
      moab::Range r;
      check(mb.get_entities_by_type(0, t, r), "get_entities_by_type");
      // A file's groups of one type (Hex8, Hex27) load in an order of their
      // own, each in file order: pair the two by node count, then by order.
      std::vector<std::pair<int, moab::EntityHandle>> theirs;
      for (const moab::EntityHandle h : r) {
        const moab::EntityHandle* c = nullptr;
        int n = 1;
        if (t != moab::MBVERTEX) check(mb.get_connectivity(h, c, n), "get_connectivity");
        theirs.emplace_back(n, h);
      }
      std::sort(theirs.begin(), theirs.end());
      for (const auto& [n, h] : theirs) handles[t].push_back(h);
      const int d = moab::CN::Dimension(t);
      std::vector<std::tuple<int, soma::GO, soma::LO>> mine;
      if (d <= mesh.dim()) {
        const auto v = mesh.view();
        const auto topo = mesh.topology(d);
        const auto gid = mesh.gid(d);
        for (std::size_t l = 0; l < gid.extent(0); ++l) {
          const auto lid = static_cast<soma::LO>(l);
          // Vertices carry no topology array.
          if (d == 0 || topo(l) == static_cast<std::uint8_t>(t)) mine.emplace_back(v.num_nodes(d, lid), gid(l), lid);
        }
      }
      std::sort(mine.begin(), mine.end());
      for (const auto& [n, g, l] : mine) lids[t].push_back(l);
      for (std::size_t i = 0; i < handles[t].size() && i < lids[t].size(); ++i) where[handles[t][i]] = {d, lids[t][i]};
    }
    moab::Range r;
    check(mb.get_entities_by_type(0, moab::MBENTITYSET, r), "get_entities_by_type");
    sets.assign(r.begin(), r.end());
    for (std::size_t i = 0; i < sets.size(); ++i) where[sets[i]] = {soma::set_dim, static_cast<soma::LO>(i)};
  }

  // SOMA's gid of what a MOAB handle names, which is its file id; 0 for none.
  soma::GO file_id(moab::EntityHandle h) const {
    if (h == 0) return 0;
    const auto [slot, lid] = where.at(h);
    const auto l = static_cast<std::size_t>(lid);
    return slot == soma::set_dim ? mesh.sets().gid()(l) : mesh.gid(slot)(l);
  }

  // SOMA's handle of what a MOAB handle names: its slot and gid; 0 for none.
  soma::GO soma_handle(moab::EntityHandle h) const {
    return h == 0 ? 0 : soma::make_handle(where.at(h).first, file_id(h));
  }

  // MOAB's holders of the tag on a slot, by SOMA lid.
  std::map<soma::LO, moab::EntityHandle> holders(moab::Tag tag, int slot) {
    std::map<soma::LO, moab::EntityHandle> out;
    for (const moab::EntityType t : moab_types) {
      if (slot == soma::set_dim || moab::CN::Dimension(t) != slot) continue;
      moab::Range r;
      check(mb.get_entities_by_type_and_tag(0, t, &tag, nullptr, 1, r), "get_entities_by_type_and_tag");
      for (const moab::EntityHandle e : r) out[where.at(e).second] = e;
    }
    if (slot == soma::set_dim) {
      moab::Range r;
      check(mb.get_entities_by_type_and_tag(0, moab::MBENTITYSET, &tag, nullptr, 1, r), "sets with the tag");
      for (const moab::EntityHandle e : r) out[where.at(e).second] = e;
    }
    return out;
  }

  // n bytes of MOAB values as SOMA holds them: handles become SOMA's.
  std::vector<unsigned char> as_soma(moab::DataType type, const void* p, std::size_t n) const {
    const auto* c = static_cast<const unsigned char*>(p);
    if (type != moab::MB_TYPE_HANDLE) return {c, c + n};
    std::vector<unsigned char> out(n);
    for (std::size_t i = 0; i < n; i += sizeof(moab::EntityHandle)) {
      moab::EntityHandle h = 0;
      std::memcpy(&h, c + i, sizeof h);
      const soma::GO g = soma_handle(h);
      std::memcpy(out.data() + i, &g, sizeof g);
    }
    return out;
  }
};

template <class T>
std::vector<unsigned char> raw(std::span<const T> x) {
  std::vector<unsigned char> b(x.size_bytes());
  if (!x.empty()) std::memcpy(b.data(), x.data(), b.size());
  return b;
}

std::vector<unsigned char> raw(const std::vector<std::byte>& b) {
  std::vector<unsigned char> out(b.size());
  if (!b.empty()) std::memcpy(out.data(), b.data(), b.size());
  return out;
}

// SOMA's values of `name` on a slot, by lid (none for a non-member), with the
// default and global; present is false without a field there.
struct SomaValues {
  bool present = false;
  bool handle = false;
  std::size_t bytes = 0;  // per entity, fixed length only
  std::vector<std::optional<std::vector<unsigned char>>> by_lid;
  std::optional<std::vector<unsigned char>> def, global;
};

template <class T>
SomaValues values_of(const soma::HostMesh& m, const std::string& name, int slot, bool variable) {
  SomaValues out;
  const soma::LO n = slot_count(m, slot);
  out.by_lid.resize(static_cast<std::size_t>(n));
  soma::FieldId id = soma::invalid_field;
  try {
    if (variable) {
      const auto f = m.fields().get_var<T>(name, slot);
      id = f.id();
      for (soma::LO l = 0; l < n; ++l) {
        if (f.space().entity_rows(slot, l).valid()) out.by_lid[static_cast<std::size_t>(l)] = raw(f.value_or_default(slot, l));
      }
      if (f.default_value()) out.def = raw(std::span<const T>(*f.default_value()));
      if (f.global_value()) out.global = raw(std::span<const T>(*f.global_value()));
    } else {
      const auto f = m.fields().get<T>(name, slot);
      id = f.id();
      out.bytes = static_cast<std::size_t>(f.ncomp()) * sizeof(T);
      for (soma::LO l = 0; l < n; ++l) {
        const auto r = f.space().entity_rows(slot, l);
        if (!r.valid()) continue;
        std::vector<T> row(static_cast<std::size_t>(f.ncomp()));
        for (int c = 0; c < f.ncomp(); ++c) row[static_cast<std::size_t>(c)] = f.view()(r.begin, c);
        out.by_lid[static_cast<std::size_t>(l)] = raw(std::span<const T>(row));
      }
      if (f.has_default()) out.def = raw(std::span<const T>(f.default_value()));
      if (f.has_global()) out.global = raw(std::span<const T>(f.global_value()));
    }
  } catch (const std::out_of_range&) {
    return SomaValues{};
  }
  out.present = true;
  out.handle = m.fields().info(id).handle;
  return out;
}

SomaValues values_of(const soma::HostMesh& m, moab::DataType type, const std::string& name, int slot, bool variable) {
  switch (type) {
    case moab::MB_TYPE_INTEGER: return values_of<std::int32_t>(m, name, slot, variable);
    case moab::MB_TYPE_DOUBLE: return values_of<double>(m, name, slot, variable);
    case moab::MB_TYPE_HANDLE: return values_of<soma::GO>(m, name, slot, variable);
    default: return values_of<std::uint8_t>(m, name, slot, variable);  // opaque and bit
  }
}

// True for a tag a MOAB instance holds before it reads anything. MOAB keeps
// its own default for such a tag (-1 for GLOBAL_ID), not the file's, and
// reports it on the mesh and on every dense holder without a value; only the
// stored values of these tags can be compared.
bool builtin(const std::string& name) {
  static moab::Core fresh;
  moab::Tag t = nullptr;
  return fresh.tag_get_handle(name.c_str(), t) == moab::MB_SUCCESS;
}

class H5mMoab : public ::testing::TestWithParam<std::string> {};

TEST_P(H5mMoab, EntityCountsPerTypeMatch) {
  Loaded f(path_of(GetParam()));
  for (const moab::EntityType t : moab_types) {
    EXPECT_EQ(f.lids[t].size(), f.handles[t].size()) << moab::CN::EntityTypeName(t);
  }
  EXPECT_EQ(f.mesh.sets().count(), static_cast<soma::LO>(f.sets.size()));
}

TEST_P(H5mMoab, CoordinatesMatchBitwise) {
  Loaded f(path_of(GetParam()));
  const auto& h = f.handles[moab::MBVERTEX];
  const auto& l = f.lids[moab::MBVERTEX];
  ASSERT_EQ(h.size(), l.size());
  std::vector<double> xyz(3 * h.size());
  check(f.mb.get_coords(h.data(), static_cast<int>(h.size()), xyz.data()), "get_coords");
  const auto c = f.mesh.coords();
  for (std::size_t i = 0; i < h.size(); ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      const double s = c(static_cast<std::size_t>(l[i]), k);
      ASSERT_EQ(std::memcmp(&s, &xyz[3 * i + k], sizeof(double)), 0) << "vertex " << i << " axis " << k;
    }
  }
}

TEST_P(H5mMoab, ConnectivityMatches) {
  Loaded f(path_of(GetParam()));
  for (const moab::EntityType t : moab_types) {
    if (t == moab::MBVERTEX) continue;
    const auto& h = f.handles[t];
    const auto& l = f.lids[t];
    ASSERT_EQ(h.size(), l.size()) << moab::CN::EntityTypeName(t);
    const int d = moab::CN::Dimension(t);
    const auto off = f.mesh.conn_offsets(d);
    const auto conn = f.mesh.conn(d);
    const auto g0 = f.mesh.gid(0);
    for (std::size_t i = 0; i < h.size(); ++i) {
      const moab::EntityHandle* mc = nullptr;
      int n = 0;
      check(f.mb.get_connectivity(h[i], mc, n, true), "get_connectivity");
      std::vector<soma::GO> want, got;
      for (int j = 0; j < n; ++j) want.push_back(f.file_id(mc[j]));
      const auto e = static_cast<std::size_t>(l[i]);
      for (auto j = off(e); j < off(e + 1); ++j) got.push_back(g0(static_cast<std::size_t>(conn(static_cast<std::size_t>(j)))));
      ASSERT_EQ(got, want) << moab::CN::EntityTypeName(t) << " " << i;
    }
  }
}

// Every element's nodes, corners then the others, as MOAB lists them.
TEST_P(H5mMoab, NodesMatch) {
  Loaded f(path_of(GetParam()));
  const auto v = f.mesh.view();
  for (const moab::EntityType t : moab_types) {
    if (t == moab::MBVERTEX) continue;
    const auto& h = f.handles[t];
    const auto& l = f.lids[t];
    ASSERT_EQ(h.size(), l.size()) << moab::CN::EntityTypeName(t);
    const int d = moab::CN::Dimension(t);
    for (std::size_t i = 0; i < h.size(); ++i) {
      const moab::EntityHandle* mc = nullptr;
      int n = 0;
      check(f.mb.get_connectivity(h[i], mc, n, false), "get_connectivity");
      std::vector<soma::GO> want, got;
      for (int j = 0; j < n; ++j) want.push_back(f.file_id(mc[j]));
      for (soma::LO k = 0; k < v.num_nodes(d, l[i]); ++k) got.push_back(v.gid(0, v.node(d, l[i], k)));
      ASSERT_EQ(got, want) << moab::CN::EntityTypeName(t) << " " << i;
    }
  }
}

// Every set's flags, members (in order for an ordered set, repeats
// included), children and parents, by file id.
TEST_P(H5mMoab, SetsMatch) {
  Loaded f(path_of(GetParam()));
  const auto& sets = f.mesh.sets();
  ASSERT_EQ(sets.count(), static_cast<soma::LO>(f.sets.size()));
  const auto gids = [&](const std::vector<moab::EntityHandle>& hs) {
    std::vector<soma::GO> out;
    for (const moab::EntityHandle h : hs) out.push_back(f.file_id(h));
    return out;
  };
  const auto set_gids = [&](std::span<const soma::LO> ls) {
    std::vector<soma::GO> out;
    for (const soma::LO l : ls) out.push_back(sets.gid()(static_cast<std::size_t>(l)));
    return out;
  };
  for (soma::LO s = 0; s < sets.count(); ++s) {
    const moab::EntityHandle h = f.sets[static_cast<std::size_t>(s)];
    SCOPED_TRACE("set " + std::to_string(sets.gid()(static_cast<std::size_t>(s))));
    unsigned options = 0;
    check(f.mb.get_meshset_options(h, options), "get_meshset_options");
    EXPECT_EQ(sets.flags(s), options & soma::set_flag_bits);
    std::vector<moab::EntityHandle> contents;
    check(f.mb.get_entities_by_handle(h, contents), "get_entities_by_handle");
    if (sets.ordered(s)) {
      std::vector<soma::GO> got;
      for (const soma::SetEntry& e : sets.sequence(s)) {
        const auto l = static_cast<std::size_t>(e.lid);
        got.push_back(e.slot == soma::set_dim ? sets.gid()(l) : f.mesh.gid(e.slot)(l));
      }
      EXPECT_EQ(got, gids(contents));
    }
    std::vector<soma::GO> want = gids(contents);
    std::sort(want.begin(), want.end());
    want.erase(std::unique(want.begin(), want.end()), want.end());
    std::vector<soma::GO> got;
    for (const int slot : slots) {
      if (slot != soma::set_dim && slot > f.mesh.dim()) continue;
      for (const soma::LO l : sets.members_of(s, slot)) {
        const auto i = static_cast<std::size_t>(l);
        got.push_back(slot == soma::set_dim ? sets.gid()(i) : f.mesh.gid(slot)(i));
      }
    }
    std::sort(got.begin(), got.end());
    EXPECT_EQ(got, want);
    std::vector<moab::EntityHandle> links;
    check(f.mb.get_child_meshsets(h, links), "get_child_meshsets");
    EXPECT_EQ(set_gids(sets.children(s)), gids(links));
    links.clear();
    check(f.mb.get_parent_meshsets(h, links), "get_parent_meshsets");
    EXPECT_EQ(set_gids(sets.parents(s)), gids(links));
  }
  if (original_of(GetParam()) == "sets_kinds") {
    EXPECT_EQ(sets.count(), 7);
    for (soma::LO s = 0; s < sets.count(); ++s) {
      if (!sets.children(s).empty() && sets.ordered(s)) return;  // the parent of the cycle
    }
    ADD_FAILURE() << "no ordered set with children";
  }
}

TEST_P(H5mMoab, EveryTagMatchesValueByValue) {
  Loaded f(path_of(GetParam()));
  for (const auto& sk : f.report.skipped) ADD_FAILURE() << sk.name << ": " << sk.reason;
  std::vector<moab::Tag> tags;
  check(f.mb.tag_get_tags(tags), "tag_get_tags");
  std::set<std::pair<std::string, int>> compared;
  std::set<std::string> mesh_values;
  for (moab::Tag tag : tags) {
    std::string name;
    check(f.mb.tag_get_name(tag, name), "tag_get_name");
    moab::DataType type{};
    check(f.mb.tag_get_data_type(tag, type), "tag_get_data_type");
    int bytes = 0;
    const bool variable = f.mb.tag_get_bytes(tag, bytes) == moab::MB_VARIABLE_DATA_LENGTH;
    const std::size_t size = value_size(type);
    moab::TagType storage{};
    check(f.mb.tag_get_type(tag, storage), "tag_get_type");
    // MOAB's default, and its value on the mesh: the global or, without one,
    // the default.
    std::optional<std::vector<unsigned char>> mdef, mroot;
    const moab::EntityHandle root = 0;
    if (variable) {
      const void* ptr = nullptr;
      int len = 0;
      if (f.mb.tag_get_default_value(tag, ptr, len) == moab::MB_SUCCESS) {
        mdef = f.as_soma(type, ptr, static_cast<std::size_t>(len) * size);
      }
      if (f.mb.tag_get_by_ptr(tag, &root, 1, &ptr, &len) == moab::MB_SUCCESS) {
        mroot = f.as_soma(type, ptr, static_cast<std::size_t>(len) * size);
      }
    } else {
      std::vector<unsigned char> b(static_cast<std::size_t>(bytes));
      if (f.mb.tag_get_default_value(tag, b.data()) == moab::MB_SUCCESS) mdef = f.as_soma(type, b.data(), b.size());
      if (f.mb.tag_get_data(tag, &root, 1, b.data()) == moab::MB_SUCCESS) mroot = f.as_soma(type, b.data(), b.size());
    }
    bool held = false;
    for (const int slot : slots) {
      SCOPED_TRACE(name + " on slot " + std::to_string(slot));
      std::map<soma::LO, std::vector<unsigned char>> want;
      for (const auto& [l, e] : f.holders(tag, slot)) {
        if (variable) {
          const void* ptr = nullptr;
          int len = 0;
          check(f.mb.tag_get_by_ptr(tag, &e, 1, &ptr, &len), "tag_get_by_ptr " + name);
          want[l] = f.as_soma(type, ptr, static_cast<std::size_t>(len) * size);
        } else {
          std::vector<unsigned char> b(static_cast<std::size_t>(bytes));
          check(f.mb.tag_get_data(tag, &e, 1, b.data()), "tag_get_data " + name);
          want[l] = f.as_soma(type, b.data(), b.size());
        }
      }
      const SomaValues s = values_of(f.mesh, type, name, slot, variable);
      if (want.empty()) {
        EXPECT_FALSE(s.present);
        continue;
      }
      ASSERT_TRUE(s.present);
      held = true;
      compared.insert({name, slot});
      EXPECT_EQ(s.handle, type == moab::MB_TYPE_HANDLE);
      if (!variable) ASSERT_EQ(s.bytes, static_cast<std::size_t>(bytes));
      std::size_t members = 0;
      for (soma::LO l = 0; l < slot_count(f.mesh, slot); ++l) {
        const auto& got = s.by_lid[static_cast<std::size_t>(l)];
        const auto it = want.find(l);
        if (it == want.end()) {
          EXPECT_FALSE(got.has_value()) << "SOMA holds a value MOAB lacks at lid " << l;
          continue;
        }
        ++members;
        if (!got) {
          // MOAB's dense storage reports the default on every entity of a
          // sequence it allocated; SOMA keeps the file's members and reports
          // the default for the others. Only dense tags can do that.
          ASSERT_EQ(storage, moab::MB_TAG_DENSE) << "lid " << l << " has a value MOAB holds and SOMA does not";
          if (!builtin(name)) {
            ASSERT_EQ(std::optional(it->second), s.def) << "lid " << l << " has a value MOAB holds and SOMA does not";
          }
          continue;
        }
        ASSERT_EQ(*got, it->second) << "lid " << l;
      }
      EXPECT_EQ(members, want.size());
      if (builtin(name)) continue;
      EXPECT_EQ(s.def, mdef);
      EXPECT_EQ(s.global ? s.global : s.def, mroot);
    }
    if (held) continue;
    // A tag on no entity or set is a mesh value. MOAB's own tags need not be
    // in the file at all.
    if (builtin(name) && !f.mesh.values().contains(name)) continue;
    if (!builtin(name)) mesh_values.insert(name);
    SCOPED_TRACE(name + " on the mesh");
    ASSERT_TRUE(f.mesh.values().contains(name));
    const soma::MeshValue& mv = f.mesh.values().at(name);
    EXPECT_EQ(mv.handle, type == moab::MB_TYPE_HANDLE);
    EXPECT_EQ(mv.ncomp == 0, variable);
    if (builtin(name)) continue;
    const auto def = mv.default_value ? std::optional(raw(*mv.default_value)) : std::nullopt;
    const auto global = mv.global_value ? std::optional(raw(*mv.global_value)) : std::nullopt;
    EXPECT_EQ(def, mdef);
    EXPECT_EQ(global ? global : def, mroot);
  }
  // Not vacuous: every file but hex_2048 carries entity tags, and the files
  // MOAB wrote carry every kind.
  const auto has = [&](const char* n) {
    return std::any_of(compared.begin(), compared.end(), [&](const auto& c) { return c.first == n; });
  };
  const std::string original = original_of(GetParam());
  if (original != "hex_2048.h5m") EXPECT_FALSE(compared.empty());
  if (original == "tag_kinds") {
    for (const char* n : {"DENSE_D2", "SPARSE_I", "PAIR", "OPAQUE7", "BITS3", "WITH_GLOBAL"}) EXPECT_TRUE(has(n)) << n;
  }
  if (original == "sets_kinds") {
    for (const char* n : {"VARF", "VAROPQ", "QV", "SS", "VH"}) EXPECT_TRUE(has(n)) << n;
    EXPECT_EQ(compared.count({"SETI", soma::set_dim}) + compared.count({"SETI", 0}), 2u);
    EXPECT_EQ(mesh_values, (std::set<std::string>{"DECLARED", "MESHVAL", "VARG"}));
  }
  // SOMA holds no field MOAB does not.
  for (const soma::FieldInfo& i : f.mesh.fields().list()) {
    for (const int d : i.dims) EXPECT_TRUE(compared.count({i.name, d}) == 1) << i.name << " on " << d;
  }
}

// The legacy writer stores a tag dense on every group the tag covers,
// whatever its storage class. MOAB reads a sparse-class tag stored that way
// (S: dense on the vertices), also when it is dense on one group and listed
// on another (M: dense on the vertices, id_list for the quad); the tag stays
// sparse and only the holders are tagged.
TEST(H5mMoabFormat, ReadsASparseTagStoredDense) {
  const fs::path p = scratch_dir() / "sparse_dense.h5m";
  {
    h5m::Writer w(p);
    w.nodes(1, 3, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0});
    w.elements("Quad4", "Quad", 5, 4, {1, 2, 3, 4});
    w.sets(6, 1);
    const int def = -7;
    w.tag("S", 1, H5T_NATIVE_INT32, &def);
    const int sv[] = {10, 20, 30, 40};
    w.dense("nodes", "S", sv, 4);
    w.tag("M", 1, H5T_NATIVE_INT32);
    const int mv[] = {1, 2, 3, 4};
    w.dense("nodes", "M", mv, 4);
    const int qv[] = {9};
    w.sparse("M", {5}, qv);
  }
  moab::Core mb;
  check(mb.load_file(p.c_str()), "load_file");
  moab::Range verts;
  moab::Range quads;
  check(mb.get_entities_by_type(0, moab::MBVERTEX, verts), "vertices");
  check(mb.get_entities_by_type(0, moab::MBQUAD, quads), "quads");
  ASSERT_EQ(verts.size(), 4u);
  ASSERT_EQ(quads.size(), 1u);
  const auto values = [&](const char* name, const moab::Range& r) {
    moab::Tag t;
    check(mb.tag_get_handle(name, t), name);
    moab::TagType type;
    check(mb.tag_get_type(t, type), "tag_get_type");
    EXPECT_EQ(type, moab::MB_TAG_SPARSE) << name;
    std::vector<int> v(r.size());
    check(mb.tag_get_data(t, r, v.data()), name);
    return v;
  };
  EXPECT_EQ(values("S", verts), (std::vector<int>{10, 20, 30, 40}));
  EXPECT_EQ(values("S", quads), std::vector<int>{-7});  // the default
  EXPECT_EQ(values("M", verts), (std::vector<int>{1, 2, 3, 4}));
  EXPECT_EQ(values("M", quads), std::vector<int>{9});
  for (const auto& [name, held] : {std::pair<const char*, std::size_t>{"S", 4}, {"M", 5}}) {
    moab::Tag t;
    check(mb.tag_get_handle(name, t), name);
    moab::Range tagged;
    check(mb.get_entities_by_type_and_tag(0, moab::MBMAXTYPE, &t, nullptr, 1, tagged), name);
    EXPECT_EQ(tagged.size(), held) << name;
  }
}

// A file read by read_h5m and written by write_h5m reads back as the same
// mesh: every MOAB-written kind of tag, set and entity survives.
class H5mMoabWritten : public ::testing::TestWithParam<std::string> {};

TEST_P(H5mMoabWritten, ReadsAsTheOriginal) {
  const soma::HostMesh original = soma::read_h5m(path_of(GetParam()));
  for (const std::string prefix : {"written:", "compressed:", "converted:"}) {
    SCOPED_TRACE(prefix);
    soma::HostMesh written = soma::read_h5m(path_of(prefix + GetParam()));
    drop_added_global_id(written, original);
    EXPECT_EQ(mesh_difference(written, original), "");
  }
}

// MOAB loads a file with cached adjacencies (under /tstt/soma, which it
// ignores) as the same mesh: the entities of every dimension, including the
// faces the cached graphs made write_h5m write.
TEST(H5mMoabCached, LoadsAFileWithCachedAdjacencies) {
  soma::HostMesh mesh = soma::read_h5m(path_of("64bricks_1khex.h5m"), {.tags = false});
  const auto request = soma::AdjacencyRequest{}.add(0, 3).add(3, 2).add(2, 3).add_bridge(3, 2);
  soma::build_adjacencies(mesh, request, Kokkos::DefaultHostExecutionSpace{});
  const fs::path out = scratch_dir() / "cached_64bricks.h5m";
  soma::write_h5m(mesh, out, {.adjacencies = request});
  moab::Core mb;
  check(mb.load_file(out.c_str()), "load_file");
  for (int d = 0; d <= 3; ++d) {
    int n = 0;
    check(mb.get_number_entities_by_dimension(0, d, n), "count");
    EXPECT_EQ(n, mesh.count(d)) << d;
  }
  EXPECT_GT(mesh.count(2), 0);
  EXPECT_EQ(soma::read_h5m(out).adjacencies().bits(), request.bits());
}

// Every configuration of a file's elements, and the mix of bit 1.
#define HIGH_ORDER(prefix, file)                                                                         \
  prefix "ho:" file ":2", prefix "ho:" file ":4", prefix "ho:" file ":6", prefix "ho:" file ":8",      \
      prefix "ho:" file ":10", prefix "ho:" file ":12", prefix "ho:" file ":14", prefix "ho:" file ":15"

INSTANTIATE_TEST_SUITE_P(Files, H5mMoab,
                         ::testing::Values("64bricks_1khex.h5m", "hex_2048.h5m", "hexes_mixed.h5m",
                                           "hybrid_3quad_1tri.h5m", "moab_i.h5m", "recMeshOcn128.h5m",
                                           "tag_kinds", "poly_indices", "sets_kinds",
                                           "written:64bricks_1khex.h5m", "written:hex_2048.h5m",
                                           "written:hexes_mixed.h5m", "written:hybrid_3quad_1tri.h5m",
                                           "written:moab_i.h5m", "written:recMeshOcn128.h5m",
                                           "written:poly_indices", "written:tag_kinds", "written:sets_kinds",
                                           "compressed:64bricks_1khex.h5m", "compressed:hex_2048.h5m",
                                           "compressed:hexes_mixed.h5m", "compressed:hybrid_3quad_1tri.h5m",
                                           "compressed:moab_i.h5m", "compressed:recMeshOcn128.h5m",
                                           "compressed:poly_indices", "compressed:tag_kinds",
                                           "compressed:sets_kinds", "zoo", HIGH_ORDER("", "zoo"),
                                           HIGH_ORDER("", "hexes_mixed.h5m"),
                                           HIGH_ORDER("", "hybrid_3quad_1tri.h5m"),
                                           HIGH_ORDER("written:", "zoo"),
                                           HIGH_ORDER("written:", "hexes_mixed.h5m"),
                                           HIGH_ORDER("written:", "hybrid_3quad_1tri.h5m"),
                                           "converted:64bricks_1khex.h5m", "converted:hex_2048.h5m",
                                           "converted:hexes_mixed.h5m", "converted:hybrid_3quad_1tri.h5m",
                                           "converted:moab_i.h5m", "converted:recMeshOcn128.h5m",
                                           "converted:poly_indices", "converted:tag_kinds",
                                           "converted:sets_kinds", "converted:zoo", HIGH_ORDER("converted:", "zoo")),
                         [](const auto& info) {
                           std::string n = info.param;
                           std::replace_if(n.begin(), n.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
                           return n;
                         });

INSTANTIATE_TEST_SUITE_P(Files, H5mMoabWritten,
                         ::testing::Values("64bricks_1khex.h5m", "hex_2048.h5m", "hexes_mixed.h5m",
                                           "hybrid_3quad_1tri.h5m", "moab_i.h5m", "recMeshOcn128.h5m",
                                           "tag_kinds", "poly_indices", "sets_kinds", "zoo",
                                           HIGH_ORDER("", "zoo"), HIGH_ORDER("", "hexes_mixed.h5m"),
                                           HIGH_ORDER("", "hybrid_3quad_1tri.h5m")),
                         [](const auto& info) {
                           std::string n = info.param;
                           std::replace_if(n.begin(), n.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
                           return n;
                         });

#undef HIGH_ORDER

}  // namespace soma_test
