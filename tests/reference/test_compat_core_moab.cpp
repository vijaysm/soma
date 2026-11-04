// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// soma::compat::Core and moab::Core side by side: the same calls on the same
// files give the same handles, values and return codes.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>
#include <moab/Core.hpp>
#include <moab/ScdInterface.hpp>

#include "soma/compat/core.hpp"
#include "soma/compat/scd.hpp"

namespace sc = soma::compat;
using H = std::uint64_t;

namespace {

const std::string data = SOMA_TEST_DATA_DIR;

template <class R>
std::vector<H> vec(const R& r) {
  return {r.begin(), r.end()};
}

// Every entity of the instance: handles by type, coordinates, connectivity.
void expect_same_mesh(sc::Core& a, moab::Core& b) {
  for (int t = 0; t <= 11; ++t) {
    sc::Range ra;
    moab::Range rb;
    ASSERT_EQ(a.get_entities_by_type(0, static_cast<sc::EntityType>(t), ra), sc::MB_SUCCESS);
    ASSERT_EQ(b.get_entities_by_type(0, static_cast<moab::EntityType>(t), rb), moab::MB_SUCCESS);
    ASSERT_EQ(vec(ra), vec(rb)) << "type " << t;
    if (t == 0) {
      std::vector<double> xa(3 * ra.size()), xb(3 * rb.size());
      a.get_coords(ra, xa.data());
      b.get_coords(rb, xb.data());
      EXPECT_EQ(xa, xb);
    } else if (t < 11) {
      for (const H h : ra) {
        std::vector<H> ca, cb;
        ASSERT_EQ(a.get_connectivity(&h, 1, ca), sc::MB_SUCCESS);
        b.get_connectivity(&h, 1, cb);
        ASSERT_EQ(ca, cb) << "entity " << h;
      }
    }
  }
  for (int d = 0; d <= 4; ++d) {
    int na = 0, nb = 0;
    a.get_number_entities_by_dimension(0, d, na);
    b.get_number_entities_by_dimension(0, d, nb);
    EXPECT_EQ(na, nb) << "dimension " << d;
  }
}

struct TempDir {
  std::filesystem::path path;
  TempDir() {
    std::string templ = (std::filesystem::temp_directory_path() / "soma_compat_XXXXXX").string();
    path = mkdtemp(templ.data());
  }
  ~TempDir() { std::filesystem::remove_all(path); }
};

}  // namespace

// Files read here and by MOAB give the same handles, coordinates and
// connectivity, sets included.
TEST(CompatCore, LoadsAsMoabDoes) {
  for (const char* f : {"64bricks_1khex.h5m", "hexes_mixed.h5m", "hybrid_3quad_1tri.h5m", "recMeshOcn128.h5m",
                        "hexes_mixed.vtk", "sphere_quads_20.vtk"}) {
    SCOPED_TRACE(f);
    sc::Core a;
    moab::Core b;
    ASSERT_EQ(a.load_file((data + "/" + f).c_str()), sc::MB_SUCCESS);
    ASSERT_EQ(b.load_file((data + "/" + f).c_str()), moab::MB_SUCCESS);
    expect_same_mesh(a, b);
  }
}

TEST(CompatCore, HandlesAndErrors) {
  sc::Core a;
  moab::Core b;
  ASSERT_EQ(a.load_file((data + "/hexes_mixed.h5m").c_str()), sc::MB_SUCCESS);
  b.load_file((data + "/hexes_mixed.h5m").c_str());
  sc::Range hexes;
  a.get_entities_by_type(0, sc::MBHEX, hexes);
  const H h = hexes.front();
  EXPECT_EQ(a.type_from_handle(h), sc::MBHEX);
  EXPECT_EQ(a.id_from_handle(h), b.id_from_handle(h));
  EXPECT_EQ(a.dimension_from_handle(h), b.dimension_from_handle(h));
  H x = 0, y = 0;
  EXPECT_EQ(a.handle_from_id(sc::MBTET, 3, x), sc::MB_SUCCESS);
  b.handle_from_id(moab::MBTET, 3, y);
  EXPECT_EQ(x, y);
  // A second file, file options and unknown files are refused.
  EXPECT_EQ(a.load_file((data + "/hexes_mixed.h5m").c_str()), sc::MB_NOT_IMPLEMENTED);
  sc::Core c;
  EXPECT_EQ(c.load_file((data + "/hexes_mixed.h5m").c_str(), nullptr, "PARALLEL=READ_PART"), sc::MB_UNHANDLED_OPTION);
  EXPECT_EQ(c.load_file("/nonexistent/x.h5m"), sc::MB_FILE_DOES_NOT_EXIST);
  const sc::EntityHandle* conn = nullptr;
  int n = 0;
  EXPECT_EQ(a.get_connectivity(h + 100000, conn, n), sc::MB_ENTITY_NOT_FOUND);
  // An element's coordinates are its corners' mean.
  double ca[3], cb[3];
  ASSERT_EQ(a.get_coords(&h, 1, ca), sc::MB_SUCCESS);
  b.get_coords(&h, 1, cb);
  for (int q = 0; q < 3; ++q) EXPECT_DOUBLE_EQ(ca[q], cb[q]);
}

// Adjacencies among existing entities: vertex to elements, element to
// vertices, sides, and up, single and intersected, as MOAB answers them.
TEST(CompatCore, AdjacenciesMatchMoab) {
  for (const char* f : {"64bricks_1khex.h5m", "hexes_mixed.h5m", "hybrid_3quad_1tri.h5m"}) {
    SCOPED_TRACE(f);
    sc::Core a;
    moab::Core b;
    ASSERT_EQ(a.load_file((data + "/" + f).c_str()), sc::MB_SUCCESS);
    b.load_file((data + "/" + f).c_str());
    sc::Range all;
    a.get_entities_by_handle(0, all);
    int checked = 0;
    for (const H h : all) {
      if (a.type_from_handle(h) == sc::MBENTITYSET || (checked++ % 7) != 0) continue;
      for (int d = 0; d <= 3; ++d) {
        sc::Range ra;
        moab::Range rb;
        ASSERT_EQ(a.get_adjacencies(&h, 1, d, false, ra), sc::MB_SUCCESS);
        b.get_adjacencies(&h, 1, d, false, rb);
        ASSERT_EQ(vec(ra), vec(rb)) << "entity " << h << " to dimension " << d;
      }
    }
    // Two vertices of one element, intersected and united.
    sc::Range elems;
    a.get_entities_by_dimension(0, 3, elems);
    if (elems.empty()) a.get_entities_by_dimension(0, 2, elems);
    std::vector<H> conn;
    a.get_connectivity(&elems.front(), 1, conn);
    for (const int op : {0, 1}) {
      sc::Range ra;
      moab::Range rb;
      a.get_adjacencies(conn.data(), 2, a.dimension_from_handle(elems.front()), false, ra, op);
      b.get_adjacencies(conn.data(), 2, b.dimension_from_handle(elems.front()), false, rb, op);
      EXPECT_EQ(vec(ra), vec(rb)) << "op " << op;
    }
  }
}

// Created sides take the next ids of their type, as MOAB creates them.
TEST(CompatCore, CreatedSidesMatchMoab) {
  sc::Core a;
  moab::Core b;
  ASSERT_EQ(a.load_file((data + "/hex_2048.h5m").c_str()), sc::MB_SUCCESS);
  b.load_file((data + "/hex_2048.h5m").c_str());
  sc::Range hexes;
  a.get_entities_by_type(0, sc::MBHEX, hexes);
  for (const int d : {1, 2}) {
    for (auto it = hexes.begin(); it != hexes.end(); it += 97) {
      const H h = *it;
      std::vector<H> va, vb;
      ASSERT_EQ(a.get_adjacencies(&h, 1, d, true, va), sc::MB_SUCCESS);
      b.get_adjacencies(&h, 1, d, true, vb);
      std::sort(va.begin(), va.end());
      std::sort(vb.begin(), vb.end());
      ASSERT_EQ(va, vb) << "hex " << h << " dimension " << d;
      if (hexes.end() - it <= 97) break;
    }
  }
  expect_same_mesh(a, b);
  int side = 0, sense = 0, offset = 0, mside = 0, msense = 0, moffset = 0;
  std::vector<H> faces;
  a.get_adjacencies(&hexes.front(), 1, 2, false, faces);
  ASSERT_FALSE(faces.empty());
  EXPECT_EQ(a.side_number(hexes.front(), faces[0], side, sense, offset), sc::MB_SUCCESS);
  b.side_number(hexes.front(), faces[0], mside, msense, moffset);
  EXPECT_EQ(side, mside);
  EXPECT_EQ(sense, msense);
  EXPECT_EQ(offset, moffset);
}

// Tags: declaration, defaults, dense and sparse values, variable length,
// pointers, iteration and queries by value, on entities, sets and the root.
TEST(CompatCore, TagsMatchMoab) {
  sc::Core a;
  moab::Core b;
  ASSERT_EQ(a.load_file((data + "/64bricks_1khex.h5m").c_str()), sc::MB_SUCCESS);
  b.load_file((data + "/64bricks_1khex.h5m").c_str());
  // Tags read from the file, by name, type, length and values.
  for (const char* name : {"MATERIAL_SET", "NEUMANN_SET", "PARALLEL_PARTITION", "vertex_field", "element_field",
                           "GLOBAL_ID", "CATEGORY"}) {
    SCOPED_TRACE(name);
    sc::Tag ta = nullptr;
    moab::Tag tb = nullptr;
    ASSERT_EQ(a.tag_get_handle(name, ta), sc::MB_SUCCESS);
    ASSERT_EQ(b.tag_get_handle(name, tb), moab::MB_SUCCESS);
    sc::DataType da;
    moab::DataType db;
    a.tag_get_data_type(ta, da);
    b.tag_get_data_type(tb, db);
    EXPECT_EQ(static_cast<int>(da), static_cast<int>(db));
    int la = 0, lb = 0;
    a.tag_get_bytes(ta, la);
    b.tag_get_bytes(tb, lb);
    EXPECT_EQ(la, lb);
    for (int t = 0; t <= 11; ++t) {
      sc::Range ra;
      moab::Range rb;
      a.get_entities_by_type_and_tag(0, static_cast<sc::EntityType>(t), &ta, nullptr, 1, ra);
      b.get_entities_by_type_and_tag(0, static_cast<moab::EntityType>(t), &tb, nullptr, 1, rb);
      ASSERT_EQ(ra.size(), rb.size()) << "type " << t;
      ASSERT_EQ(vec(ra), vec(rb)) << "type " << t;
      std::vector<unsigned char> va(ra.size() * static_cast<std::size_t>(la)), vb(va.size());
      if (ra.empty()) continue;
      ASSERT_EQ(a.tag_get_data(ta, ra, va.data()), sc::MB_SUCCESS);
      b.tag_get_data(tb, rb, vb.data());
      EXPECT_EQ(va, vb) << "type " << t;
    }
  }
  // A dense tag with a default, set on part of the vertices.
  const double def[2] = {-1, -2};
  sc::Tag ta = nullptr;
  moab::Tag tb = nullptr;
  ASSERT_EQ(a.tag_get_handle("vel", 2, sc::MB_TYPE_DOUBLE, ta, sc::MB_TAG_DENSE | sc::MB_TAG_CREAT, def), sc::MB_SUCCESS);
  b.tag_get_handle("vel", 2, moab::MB_TYPE_DOUBLE, tb, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, def);
  EXPECT_EQ(a.tag_get_handle("vel", 2, sc::MB_TYPE_DOUBLE, ta, sc::MB_TAG_DENSE | sc::MB_TAG_CREAT | sc::MB_TAG_EXCL, def),
            sc::MB_ALREADY_ALLOCATED);
  EXPECT_EQ(a.tag_get_handle("vel", 3, sc::MB_TYPE_DOUBLE, ta), sc::MB_INVALID_SIZE);
  EXPECT_EQ(a.tag_get_handle("vel", 2, sc::MB_TYPE_INTEGER, ta), sc::MB_TYPE_OUT_OF_RANGE);
  EXPECT_EQ(a.tag_get_handle("nope", 1, sc::MB_TYPE_INTEGER, ta), sc::MB_TAG_NOT_FOUND);
  a.tag_get_handle("vel", 2, sc::MB_TYPE_DOUBLE, ta);
  sc::Range verts;
  a.get_entities_by_type(0, sc::MBVERTEX, verts);
  const sc::Range part(verts[10], verts[40]);
  std::vector<double> vals(2 * part.size());
  for (std::size_t i = 0; i < vals.size(); ++i) vals[i] = 0.5 * static_cast<double>(i);
  ASSERT_EQ(a.tag_set_data(ta, part, vals.data()), sc::MB_SUCCESS);
  moab::Range mpart(verts[10], verts[40]);
  b.tag_set_data(tb, mpart, vals.data());
  std::vector<double> ga(2 * 60), gb(2 * 60);
  const sc::Range first(verts[0], verts[59]);
  moab::Range mfirst(verts[0], verts[59]);
  EXPECT_EQ(a.tag_get_data(ta, first, ga.data()), sc::MB_SUCCESS);
  b.tag_get_data(tb, mfirst, gb.data());
  EXPECT_EQ(ga, gb);
  // Iterating a dense run.
  int count = 0, mcount = 0;
  void* pa = nullptr;
  void* pb = nullptr;
  ASSERT_EQ(a.tag_iterate(ta, verts.begin(), verts.end(), count, pa), sc::MB_SUCCESS);
  b.tag_iterate(tb, mfirst.begin(), mfirst.end(), mcount, pb);
  EXPECT_EQ(count, static_cast<int>(verts.size()));
  EXPECT_EQ(static_cast<double*>(pa)[2 * 20 + 1], vals[2 * 10 + 1]);
  // A sparse integer tag without a default: no value is MB_TAG_NOT_FOUND.
  sc::Tag sa = nullptr;
  moab::Tag sb = nullptr;
  a.tag_get_handle("mark", 1, sc::MB_TYPE_INTEGER, sa, sc::MB_TAG_SPARSE | sc::MB_TAG_CREAT);
  b.tag_get_handle("mark", 1, moab::MB_TYPE_INTEGER, sb, moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT);
  int v = 0;
  EXPECT_EQ(static_cast<int>(a.tag_get_data(sa, &verts.front(), 1, &v)),
            static_cast<int>(b.tag_get_data(sb, &verts.front(), 1, &v)));
  const int seven = 7;
  a.tag_set_data(sa, &verts.back(), 1, &seven);
  b.tag_set_data(sb, &verts.back(), 1, &seven);
  const void* want[] = {&seven};
  sc::Range ra;
  moab::Range rb;
  a.get_entities_by_type_and_tag(0, sc::MBVERTEX, &sa, want, 1, ra);
  b.get_entities_by_type_and_tag(0, moab::MBVERTEX, &sb, want, 1, rb);
  EXPECT_EQ(vec(ra), vec(rb));
  // Variable-length values through pointers, and a value on the root set.
  sc::Tag va = nullptr;
  moab::Tag vb = nullptr;
  a.tag_get_handle("list", 0, sc::MB_TYPE_INTEGER, va, sc::MB_TAG_VARLEN | sc::MB_TAG_SPARSE | sc::MB_TAG_CREAT);
  b.tag_get_handle("list", 0, moab::MB_TYPE_INTEGER, vb, moab::MB_TAG_VARLEN | moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT);
  const int three[3] = {1, 2, 3};
  const void* p3 = three;
  const int n3 = 3;
  const H root = 0;
  EXPECT_EQ(a.tag_set_by_ptr(va, &root, 1, &p3, &n3), sc::MB_SUCCESS);
  b.tag_set_by_ptr(vb, &root, 1, &p3, &n3);
  const void* got = nullptr;
  int size = 0;
  EXPECT_EQ(a.tag_get_by_ptr(va, &root, 1, &got, &size), sc::MB_SUCCESS);
  EXPECT_EQ(size, 3);
  EXPECT_EQ(static_cast<const int*>(got)[2], 3);
  int len = 0;
  EXPECT_EQ(static_cast<int>(a.tag_get_length(va, len)), static_cast<int>(b.tag_get_length(vb, len)));
  std::vector<sc::Tag> on;
  a.tag_get_tags_on_entity(verts.back(), on);
  EXPECT_NE(std::find(on.begin(), on.end(), sa), on.end());
  EXPECT_EQ(a.tag_delete(sa), sc::MB_SUCCESS);
  EXPECT_EQ(a.tag_get_handle("mark", sa), sc::MB_TAG_NOT_FOUND);
}

// Sets: creation, contents (ordered and not), recursion, links, as MOAB
// keeps them.
TEST(CompatCore, SetsMatchMoab) {
  sc::Core a;
  moab::Core b;
  ASSERT_EQ(a.load_file((data + "/64bricks_1khex.h5m").c_str()), sc::MB_SUCCESS);
  b.load_file((data + "/64bricks_1khex.h5m").c_str());
  sc::Range sets;
  a.get_entities_by_type(0, sc::MBENTITYSET, sets);
  for (auto it = sets.begin(); it != sets.end(); it += 37) {
    const H s = *it;
    for (const bool rec : {false, true}) {
      std::vector<H> ca, cb;
      a.get_entities_by_handle(s, ca, rec);
      b.get_entities_by_handle(s, cb, rec);
      ASSERT_EQ(ca, cb) << "set " << s << " recursive " << rec;
    }
    std::vector<H> pa, pb, ka, kb;
    a.get_parent_meshsets(s, pa);
    b.get_parent_meshsets(s, pb);
    a.get_child_meshsets(s, ka, 0);
    b.get_child_meshsets(s, kb, 0);
    EXPECT_EQ(pa, pb);
    EXPECT_EQ(ka, kb);
    unsigned fa = 0, fb = 0;
    a.get_meshset_options(s, fa);
    b.get_meshset_options(s, fb);
    EXPECT_EQ(fa & 0x6u, fb & 0x6u);
    if (sets.end() - it <= 37) break;
  }
  // New sets: an ordered one keeps order and repeats, a unique one sorts.
  H oa = 0, ob = 0, ua = 0, ub = 0;
  a.create_meshset(sc::MESHSET_ORDERED, oa);
  b.create_meshset(moab::MESHSET_ORDERED, ob);
  a.create_meshset(sc::MESHSET_SET, ua);
  b.create_meshset(moab::MESHSET_SET, ub);
  EXPECT_EQ(oa, ob);
  EXPECT_EQ(ua, ub);
  sc::Range verts;
  a.get_entities_by_type(0, sc::MBVERTEX, verts);
  const H some[] = {verts[5], verts[2], verts[5], ua};
  a.add_entities(oa, some, 4);
  b.add_entities(ob, some, 4);
  a.add_entities(ua, some, 3);
  b.add_entities(ub, some, 3);
  for (const H s : {oa, ua}) {
    std::vector<H> ca, cb;
    a.get_entities_by_handle(s, ca);
    b.get_entities_by_handle(s, cb);
    EXPECT_EQ(ca, cb) << "set " << s;
    int na = 0, nb = 0;
    a.get_number_entities_by_handle(s, na);
    b.get_number_entities_by_handle(s, nb);
    EXPECT_EQ(na, nb);
  }
  const H v5 = verts[5], v2 = verts[2], v3 = verts[3];
  a.remove_entities(oa, &v5, 1);
  b.remove_entities(ob, &v5, 1);
  std::vector<H> ca, cb;
  a.get_entities_by_handle(oa, ca);
  b.get_entities_by_handle(ob, cb);
  EXPECT_EQ(ca, cb);
  a.add_parent_child(oa, ua);
  b.add_parent_child(ob, ub);
  std::vector<H> pa, pb;
  a.get_parent_meshsets(ua, pa);
  b.get_parent_meshsets(ub, pb);
  EXPECT_EQ(pa, pb);
  std::vector<H> xa, xb;
  a.get_contained_meshsets(oa, xa);
  b.get_contained_meshsets(ob, xb);
  EXPECT_EQ(xa, xb);
  EXPECT_TRUE(a.contains_entities(ua, &v2, 1));
  EXPECT_FALSE(a.contains_entities(ua, &v3, 1));
}

// Creating and deleting entities: handles follow MOAB's, deleted ones stay
// gone (and leave sets), the others keep theirs.
TEST(CompatCore, CreateAndDeleteMatchMoab) {
  sc::Core a;
  moab::Core b;
  const double xyz[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 2, 0, 0};
  sc::Range va;
  moab::Range vb;
  ASSERT_EQ(a.create_vertices(xyz, 9, va), sc::MB_SUCCESS);
  b.create_vertices(xyz, 9, vb);
  EXPECT_EQ(vec(va), vec(vb));
  const H hex[] = {va[0], va[1], va[2], va[3], va[4], va[5], va[6], va[7]};
  const H tri[] = {va[1], va[8], va[2]};
  H ha = 0, hb = 0, ta = 0, tb = 0;
  ASSERT_EQ(a.create_element(sc::MBHEX, hex, 8, ha), sc::MB_SUCCESS);
  b.create_element(moab::MBHEX, hex, 8, hb);
  ASSERT_EQ(a.create_element(sc::MBTRI, tri, 3, ta), sc::MB_SUCCESS);
  b.create_element(moab::MBTRI, tri, 3, tb);
  EXPECT_EQ(ha, hb);
  EXPECT_EQ(ta, tb);
  H sa = 0, sb = 0;
  a.create_meshset(sc::MESHSET_SET, sa);
  b.create_meshset(moab::MESHSET_SET, sb);
  a.add_entities(sa, &ta, 1);
  b.add_entities(sb, &tb, 1);
  expect_same_mesh(a, b);
  ASSERT_EQ(a.delete_entities(&ta, 1), sc::MB_SUCCESS);
  b.delete_entities(&tb, 1);
  expect_same_mesh(a, b);
  std::vector<H> ca;
  a.get_entities_by_handle(sa, ca);
  EXPECT_TRUE(ca.empty());
  // A vertex a live element uses cannot go.
  const H v0 = va[0];
  EXPECT_EQ(a.delete_entities(&v0, 1), sc::MB_FAILURE);
  // A created vertex after a deletion takes the next id.
  H na = 0, nb = 0;
  a.create_vertex(xyz, na);
  b.create_vertex(xyz, nb);
  EXPECT_EQ(na, nb);
  expect_same_mesh(a, b);
}

// What compat writes, MOAB reads back as compat holds it, and the reverse.
TEST(CompatCore, WritesWhatMoabReads) {
  TempDir dir;
  sc::Core a;
  ASSERT_EQ(a.load_file((data + "/64bricks_1khex.h5m").c_str()), sc::MB_SUCCESS);
  sc::Tag t = nullptr;
  const int zero = 0;
  a.tag_get_handle("flag", 1, sc::MB_TYPE_INTEGER, t, sc::MB_TAG_SPARSE | sc::MB_TAG_CREAT, &zero);
  sc::Range hexes;
  a.get_entities_by_type(0, sc::MBHEX, hexes);
  std::vector<int> ones(10, 1);
  a.tag_set_data(t, sc::Range(hexes[0], hexes[9]), ones.data());
  const std::string out = (dir.path / "out.h5m").string();
  ASSERT_EQ(a.write_file(out.c_str()), sc::MB_SUCCESS);
  moab::Core b;
  ASSERT_EQ(b.load_file(out.c_str()), moab::MB_SUCCESS);
  expect_same_mesh(a, b);
  moab::Tag tb = nullptr;
  ASSERT_EQ(b.tag_get_handle("flag", tb), moab::MB_SUCCESS);
  moab::Range tagged;
  b.get_entities_by_type_and_tag(0, moab::MBHEX, &tb, nullptr, 1, tagged);
  EXPECT_EQ(tagged.size(), 10u);
  EXPECT_EQ(a.write_file(out.c_str(), nullptr, "PARALLEL=WRITE_PART"), sc::MB_UNHANDLED_OPTION);
  EXPECT_EQ(a.write_file((dir.path / "out.stl").string().c_str()), sc::MB_NOT_IMPLEMENTED);
}

// A structured box: MOAB's handles for (i, j, k), coordinates and box set.
TEST(CompatCore, BoxesMatchMoab) {
  sc::Core a;
  moab::Core b;
  sc::ScdInterface* sa = nullptr;
  moab::ScdInterface* sb = nullptr;
  ASSERT_EQ(a.query_interface(sa), sc::MB_SUCCESS);
  b.query_interface(sb);
  sc::ScdBox* ba = nullptr;
  moab::ScdBox* bb = nullptr;
  ASSERT_EQ(sa->construct_box(sc::HomCoord(1, 2, 3), sc::HomCoord(4, 4, 5), nullptr, 0, ba, nullptr, nullptr, true),
            sc::MB_SUCCESS);
  sb->construct_box(moab::HomCoord(1, 2, 3), moab::HomCoord(4, 4, 5), nullptr, 0, bb, nullptr, nullptr, true);
  expect_same_mesh(a, b);
  for (int k = 3; k <= 5; ++k) {
    for (int j = 2; j <= 4; ++j) {
      for (int i = 1; i <= 4; ++i) {
        EXPECT_EQ(ba->get_vertex(i, j, k), bb->get_vertex(i, j, k));
        if (i < 4 && j < 4 && k < 5) EXPECT_EQ(ba->get_element(i, j, k), bb->get_element(i, j, k));
      }
    }
  }
  sc::Tag ga = nullptr;
  moab::Tag gb = nullptr;
  a.tag_get_handle("GLOBAL_ID", ga);
  b.tag_get_handle("GLOBAL_ID", gb);
  sc::Range verts;
  a.get_entities_by_type(0, sc::MBVERTEX, verts);
  moab::Range mverts;
  b.get_entities_by_type(0, moab::MBVERTEX, mverts);
  std::vector<int> ia(verts.size()), ib(verts.size());
  a.tag_get_data(ga, verts, ia.data());
  b.tag_get_data(gb, mverts, ib.data());
  EXPECT_EQ(ia, ib);
  // A flat k extent gives quads.
  sc::Core c;
  c.query_interface(sa);
  ASSERT_EQ(sa->construct_box(sc::HomCoord(0, 0, 0), sc::HomCoord(3, 2, 0), nullptr, 0, ba), sc::MB_SUCCESS);
  EXPECT_EQ(c.type_from_handle(ba->get_element(2, 1)), sc::MBQUAD);
  EXPECT_EQ(sa->construct_box(sc::HomCoord(0, 0, 0), sc::HomCoord(3, 2, 0), nullptr, 0, ba), sc::MB_NOT_IMPLEMENTED);
}

// Ids as MOAB hands them out: freed ids are taken again, another node count
// starts a new block (16384 ids for elements made one at a time, 16384 / n
// for n-gons), and a run created after a file continues its type.
TEST(CompatCore, IdsFollowMoabAllocation) {
  sc::Core a;
  moab::Core b;
  std::vector<double> xyz(3 * 40);
  for (std::size_t i = 0; i < xyz.size(); ++i) xyz[i] = static_cast<double>(i);
  sc::Range va;
  moab::Range vb;
  a.create_vertices(xyz.data(), 40, va);
  b.create_vertices(xyz.data(), 40, vb);
  const std::vector<H> c(va.begin(), va.end());
  const auto both = [&](sc::EntityType t, std::size_t first, int n) {
    H x = 0, y = 0;
    EXPECT_EQ(a.create_element(t, c.data() + first, n, x), sc::MB_SUCCESS);
    b.create_element(static_cast<moab::EntityType>(t), c.data() + first, n, y);
    EXPECT_EQ(x, y) << "type " << t << " with " << n << " nodes";
    return x;
  };
  const H h1 = both(sc::MBHEX, 0, 8);
  const H h2 = both(sc::MBHEX, 1, 8);
  a.delete_entities(&h2, 1);
  b.delete_entities(&h2, 1);
  both(sc::MBHEX, 2, 8);  // takes h2's id
  a.delete_entities(&h1, 1);
  b.delete_entities(&h1, 1);
  both(sc::MBHEX, 3, 8);   // takes h1's id
  both(sc::MBHEX, 0, 27);  // a new block
  both(sc::MBPOLYGON, 0, 5);
  both(sc::MBPOLYGON, 0, 6);  // after the pentagons' 3276
  both(sc::MBPOLYGON, 1, 5);
  H s1 = 0, s2 = 0;
  a.create_meshset(sc::MESHSET_SET, s1);
  b.create_meshset(moab::MESHSET_SET, s2);
  EXPECT_EQ(s1, s2);
  const H last = va.back();
  a.delete_entities(&last, 1);
  b.delete_entities(&last, 1);
  H x = 0, y = 0;
  a.create_vertex(xyz.data(), x);
  b.create_vertex(xyz.data(), y);
  EXPECT_EQ(x, y);
  expect_same_mesh(a, b);

  sc::Core f;
  moab::Core g;
  ASSERT_EQ(f.load_file((data + "/64bricks_1khex.h5m").c_str()), sc::MB_SUCCESS);
  g.load_file((data + "/64bricks_1khex.h5m").c_str());
  sc::Range hexes;
  f.get_entities_by_type(0, sc::MBHEX, hexes);
  std::vector<H> hc;
  f.get_connectivity(&hexes.front(), 1, hc);
  f.create_element(sc::MBHEX, hc.data(), 8, x);
  g.create_element(moab::MBHEX, hc.data(), 8, y);
  EXPECT_EQ(x, y);
  f.create_meshset(sc::MESHSET_SET, x);
  g.create_meshset(moab::MESHSET_SET, y);
  EXPECT_EQ(x, y);
}
