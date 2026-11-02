// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// read_h5m against MOAB's load_file on the same files: entity counts per
// type, coordinates and connectivity bitwise, and every tag MOAB holds on
// vertices and elements, value by value, with its default and global value.
//
// MOAB numbers a freshly loaded file's entities of one type in file id
// order, so the i-th handle of a type is the entity with the i-th smallest
// file id of that type. Nothing here trusts that blindly: a different order
// shows up as coordinates, connectivity or tag values that disagree.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "h5m_writer.hpp"
#include "moab/CN.hpp"
#include "moab/Core.hpp"
#include "moab/Range.hpp"
#include "soma/core/mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/io/h5m.hpp"

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

fs::path scratch_dir() {
  static const fs::path dir = [] {
    const char* t = std::getenv("TMPDIR");
    std::string pattern = ((t != nullptr && *t != '\0') ? fs::path(t) : fs::path("/tmp")) / "soma_h5mm_XXXXXX";
    if (mkdtemp(pattern.data()) == nullptr) throw std::runtime_error("cannot create " + pattern);
    return fs::path(pattern);
  }();
  return dir;
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

fs::path path_of(const std::string& name) {
  if (name == "tag_kinds") return tag_kinds_file();
  if (name == "poly_indices") return poly_indices_file();
  return fs::path(SOMA_TEST_DATA_DIR) / name;
}

struct Loaded {
  moab::Core mb;
  soma::HostMesh mesh;
  soma::H5mReport report;
  // Per MOAB type: handles in handle order, and the SOMA (dim, lid) of each.
  std::map<moab::EntityType, std::vector<moab::EntityHandle>> handles;
  std::map<moab::EntityType, std::vector<soma::LO>> lids;
  std::map<moab::EntityHandle, soma::GO> file_id;  // vertices only

  explicit Loaded(const fs::path& p) {
    check(mb.load_file(p.c_str()), "load_file " + p.string());
    mesh = soma::read_h5m(p, {}, report);
    for (const moab::EntityType t : moab_types) {
      moab::Range r;
      check(mb.get_entities_by_type(0, t, r), "get_entities_by_type");
      handles[t].assign(r.begin(), r.end());
      const int d = moab::CN::Dimension(t);
      std::vector<std::pair<soma::GO, soma::LO>> mine;
      if (d <= mesh.dim()) {
        const auto topo = mesh.topology(d);
        const auto gid = mesh.gid(d);
        for (std::size_t l = 0; l < gid.extent(0); ++l) {
          // Vertices carry no topology array.
          if (d == 0 || topo(l) == static_cast<std::uint8_t>(t)) mine.emplace_back(gid(l), static_cast<soma::LO>(l));
        }
      }
      std::sort(mine.begin(), mine.end());
      for (const auto& [g, l] : mine) lids[t].push_back(l);
    }
    const auto& v = handles[moab::MBVERTEX];
    const auto& vl = lids[moab::MBVERTEX];
    for (std::size_t i = 0; i < v.size() && i < vl.size(); ++i) {
      file_id[v[i]] = mesh.gid(0)(static_cast<std::size_t>(vl[i]));
    }
  }

  // True if SOMA reports leaving out the tag's values on vertices and
  // elements (not just those on sets).
  bool skipped(const std::string& name) const {
    return std::any_of(report.skipped.begin(), report.skipped.end(), [&](const auto& s) {
      return s.name == name && s.reason.find("entity sets") == std::string::npos;
    });
  }
};

// The bytes of every value of `name` on dimension d, by lid; empty when
// SOMA has no field of that name on d. Also the default and global bytes.
struct SomaValues {
  bool present = false;
  std::size_t bytes = 0;  // per entity
  std::vector<std::vector<unsigned char>> by_lid;  // empty vector: no value
  std::vector<unsigned char> def, global;
};

template <class T>
SomaValues values_of(const soma::HostMesh& m, const std::string& name, int d) {
  SomaValues out;
  soma::Field<T, Kokkos::HostSpace> f;
  try {
    f = m.fields().get<T>(name, d);
  } catch (const std::out_of_range&) {
    return out;
  }
  out.present = true;
  const auto nc = static_cast<std::size_t>(f.ncomp());
  out.bytes = nc * sizeof(T);
  out.by_lid.resize(static_cast<std::size_t>(m.count(d)));
  const auto v = f.view();
  for (soma::LO l = 0; l < m.count(d); ++l) {
    const auto r = f.space().entity_rows(d, l);
    if (!r.valid()) continue;
    auto& b = out.by_lid[static_cast<std::size_t>(l)];
    b.resize(out.bytes);
    for (std::size_t c = 0; c < nc; ++c) {
      const T x = v(static_cast<std::size_t>(r.begin), c);
      std::memcpy(b.data() + c * sizeof(T), &x, sizeof(T));
    }
  }
  const auto bytes_of = [](const std::vector<T>& x) {
    std::vector<unsigned char> b(x.size() * sizeof(T));
    if (!x.empty()) std::memcpy(b.data(), x.data(), b.size());
    return b;
  };
  out.def = bytes_of(f.default_value());
  out.global = bytes_of(f.global_value());
  return out;
}

SomaValues values_of(const soma::HostMesh& m, moab::DataType type, const std::string& name, int d) {
  switch (type) {
    case moab::MB_TYPE_INTEGER: return values_of<std::int32_t>(m, name, d);
    case moab::MB_TYPE_DOUBLE: return values_of<double>(m, name, d);
    default: return values_of<std::uint8_t>(m, name, d);  // opaque and bit
  }
}

// True for a tag a MOAB instance holds before it reads anything.
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
      for (int j = 0; j < n; ++j) want.push_back(f.file_id.at(mc[j]));
      const auto e = static_cast<std::size_t>(l[i]);
      for (auto j = off(e); j < off(e + 1); ++j) got.push_back(g0(static_cast<std::size_t>(conn(static_cast<std::size_t>(j)))));
      ASSERT_EQ(got, want) << moab::CN::EntityTypeName(t) << " " << i;
    }
  }
}

TEST_P(H5mMoab, EveryEntityTagMatchesValueByValue) {
  Loaded f(path_of(GetParam()));
  std::vector<moab::Tag> tags;
  check(f.mb.tag_get_tags(tags), "tag_get_tags");
  std::set<std::pair<std::string, int>> compared;
  for (moab::Tag tag : tags) {
    std::string name;
    check(f.mb.tag_get_name(tag, name), "tag_get_name");
    moab::DataType type{};
    check(f.mb.tag_get_data_type(tag, type), "tag_get_data_type");
    int bytes = 0;
    if (f.mb.tag_get_bytes(tag, bytes) == moab::MB_VARIABLE_DATA_LENGTH) {
      EXPECT_TRUE(f.skipped(name)) << name << " is variable-length";
      continue;
    }
    check(f.mb.tag_get_bytes(tag, bytes), "tag_get_bytes");
    moab::TagType storage{};
    check(f.mb.tag_get_type(tag, storage), "tag_get_type");
    std::vector<unsigned char> mdef(static_cast<std::size_t>(bytes));
    const bool has_def = type != moab::MB_TYPE_HANDLE &&
                         f.mb.tag_get_default_value(tag, mdef.data()) == moab::MB_SUCCESS;
    for (int d = 0; d <= 3; ++d) {
      // Entities of dimension d carrying the tag, by SOMA lid.
      std::map<soma::LO, std::vector<unsigned char>> want;
      for (const moab::EntityType t : moab_types) {
        if (moab::CN::Dimension(t) != d) continue;
        moab::Range r;
        check(f.mb.get_entities_by_type_and_tag(0, t, &tag, nullptr, 1, r), "get_entities_by_type_and_tag");
        const auto& h = f.handles[t];
        for (const moab::EntityHandle e : r) {
          const auto at = std::lower_bound(h.begin(), h.end(), e);
          ASSERT_TRUE(at != h.end() && *at == e);
          std::vector<unsigned char> b(static_cast<std::size_t>(bytes));
          check(f.mb.tag_get_data(tag, &e, 1, b.data()), "tag_get_data " + name);
          want[f.lids[t][static_cast<std::size_t>(at - h.begin())]] = std::move(b);
        }
      }
      if (want.empty() && f.skipped(name)) continue;
      SCOPED_TRACE(name + " on dimension " + std::to_string(d));
      ASSERT_FALSE(f.skipped(name) && !want.empty()) << "MOAB holds values SOMA reported as not read";
      const SomaValues s = values_of(f.mesh, type, name, d);
      if (want.empty()) {
        EXPECT_FALSE(s.present);
        continue;
      }
      ASSERT_TRUE(s.present);
      compared.insert({name, d});
      ASSERT_EQ(s.bytes, static_cast<std::size_t>(bytes));
      std::size_t members = 0;
      for (soma::LO l = 0; l < f.mesh.count(d); ++l) {
        const auto& got = s.by_lid[static_cast<std::size_t>(l)];
        const auto it = want.find(l);
        if (it == want.end()) {
          EXPECT_TRUE(got.empty()) << "SOMA holds a value MOAB lacks at lid " << l;
          continue;
        }
        ++members;
        if (got.empty()) {
          // MOAB's dense storage reports the default on every entity of a
          // sequence it allocated; SOMA keeps the file's members and reports
          // the default for the others. Only dense tags can do that.
          ASSERT_EQ(storage, moab::MB_TAG_DENSE) << "lid " << l << " has a value MOAB holds and SOMA does not";
          ASSERT_EQ(it->second, s.def) << "lid " << l << " has a value MOAB holds and SOMA does not";
          continue;
        }
        ASSERT_EQ(got, it->second) << "lid " << l;
      }
      EXPECT_EQ(members, want.size());
      // MOAB defines some tags (GLOBAL_ID) before reading, with a default of
      // its own; the file's default, when it has one, is what SOMA keeps.
      if (builtin(name) && s.def.empty()) continue;
      EXPECT_EQ(s.def, has_def ? mdef : std::vector<unsigned char>{});
      // On the mesh MOAB reports the global value or, without one, the default.
      std::vector<unsigned char> mglobal(static_cast<std::size_t>(bytes));
      const moab::EntityHandle root = 0;
      const bool on_mesh = f.mb.tag_get_data(tag, &root, 1, mglobal.data()) == moab::MB_SUCCESS;
      EXPECT_EQ(s.global.empty() ? s.def : s.global, on_mesh ? mglobal : std::vector<unsigned char>{});
    }
  }
  // Not vacuous: every file but hex_2048 carries entity tags, and the file
  // MOAB wrote carries one of every fixed-length kind.
  if (GetParam() != "hex_2048.h5m") EXPECT_FALSE(compared.empty());
  if (GetParam() == "tag_kinds") {
    for (const char* n : {"DENSE_D2", "SPARSE_I", "PAIR", "OPAQUE7", "BITS3", "WITH_GLOBAL"}) {
      EXPECT_TRUE(std::any_of(compared.begin(), compared.end(), [&](const auto& c) { return c.first == n; }))
          << n;
    }
  }
  // SOMA holds no field MOAB does not.
  for (const soma::FieldInfo& i : f.mesh.fields().list()) {
    for (const int d : i.dims) EXPECT_TRUE(compared.count({i.name, d}) == 1) << i.name << " on " << d;
  }
}

INSTANTIATE_TEST_SUITE_P(Files, H5mMoab,
                         ::testing::Values("64bricks_1khex.h5m", "hex_2048.h5m", "hexes_mixed.h5m",
                                           "hybrid_3quad_1tri.h5m", "moab_i.h5m", "recMeshOcn128.h5m",
                                           "tag_kinds", "poly_indices"),
                         [](const auto& info) {
                           std::string n = info.param;
                           std::replace_if(n.begin(), n.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
                           return n;
                         });

}  // namespace soma_test
