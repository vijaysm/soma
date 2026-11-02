// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_UNIT_H5M_WRITER_HPP_
#define SOMA_TESTS_UNIT_H5M_WRITER_HPP_

// Writes small files in MOAB's .h5m layout with the HDF5 C API, so reader
// tests can build exactly the structure they need, malformed ones included.
// Every call throws std::runtime_error when HDF5 refuses it.

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

namespace soma_test::h5m {

inline hid_t ok(hid_t id, const char* what) {
  if (id < 0) throw std::runtime_error(std::string("h5m writer: HDF5 could not ") + what);
  return id;
}

inline void check(herr_t rc, const char* what) {
  if (rc < 0) throw std::runtime_error(std::string("h5m writer: HDF5 could not ") + what);
}

class Writer {
 public:
  explicit Writer(const std::filesystem::path& path) {
    file_ = ok(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), "create the file");
    tstt_ = ok(H5Gcreate2(file_, "tstt", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "create /tstt");
    types_ = ok(H5Tenum_create(H5T_NATIVE_UINT8), "create the element type enum");
    const char* names[] = {"Edge", "Tri", "Quad", "Polygon", "Tet", "Pyramid", "Prism", "Knife", "Hex",
                           "Polyhedron", "Septahedron"};
    for (std::uint8_t v = 1; v <= 11; ++v) check(H5Tenum_insert(types_, names[v - 1], &v), "insert an enum value");
    check(H5Tcommit2(tstt_, "elemtypes", types_, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "commit elemtypes");
  }
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;
  ~Writer() { close(); }

  // Writes max_id (the largest file id written, unless set) and closes.
  void close() {
    if (file_ < 0) return;
    if (write_max_id_) scalar_attribute(tstt_, "max_id", H5T_NATIVE_UINT64, max_id_set_ ? &max_id_ : &largest_);
    H5Tclose(types_);
    H5Gclose(tstt_);
    H5Fclose(file_);
    file_ = -1;
  }

  void max_id(std::uint64_t v) {
    max_id_ = v;
    max_id_set_ = true;
  }
  void omit_max_id() { write_max_id_ = false; }

  hid_t tstt() const { return tstt_; }

  // xyz holds cols values per vertex.
  void nodes(std::int64_t start, int cols, const std::vector<double>& xyz) {
    const hid_t g = group("nodes");
    const hsize_t dims[2] = {xyz.size() / static_cast<std::size_t>(cols), static_cast<hsize_t>(cols)};
    const hid_t ds = dataset(g, "coordinates", H5T_IEEE_F64LE, 2, dims);
    write(ds, H5T_NATIVE_DOUBLE, xyz.data(), xyz.size());
    scalar_attribute(ds, "start_id", H5T_NATIVE_INT64, &start);
    note(start, dims[0]);
    H5Dclose(ds);
    H5Gclose(g);
  }

  // Fixed arity: conn holds arity vertex file ids per element.
  void elements(const std::string& name, const std::string& type, std::int64_t start, int arity,
                const std::vector<std::uint64_t>& conn) {
    const hid_t g = element_group(name, type);
    const hsize_t dims[2] = {conn.size() / static_cast<std::size_t>(arity), static_cast<hsize_t>(arity)};
    const hid_t ds = dataset(g, "connectivity", H5T_STD_U64LE, 2, dims);
    write(ds, H5T_NATIVE_UINT64, conn.data(), conn.size());
    scalar_attribute(ds, "start_id", H5T_NATIVE_INT64, &start);
    note(start, dims[0]);
    H5Dclose(ds);
    H5Gclose(g);
  }

  // Variable arity: last[i] is the index in conn of polygon i's last vertex.
  void poly_elements(const std::string& name, const std::string& type, std::int64_t start,
                     const std::vector<std::uint64_t>& conn, const std::vector<std::int64_t>& last) {
    const hid_t g = element_group(name, type);
    const hsize_t n = conn.size();
    const hid_t ds = dataset(g, "connectivity", H5T_STD_U64LE, 1, &n);
    write(ds, H5T_NATIVE_UINT64, conn.data(), conn.size());
    scalar_attribute(ds, "start_id", H5T_NATIVE_INT64, &start);
    const hsize_t m = last.size();
    const hid_t pi = dataset(g, "poly_indices", H5T_STD_I64LE, 1, &m);
    write(pi, H5T_NATIVE_INT64, last.data(), last.size());
    note(start, m);
    H5Dclose(pi);
    H5Dclose(ds);
    H5Gclose(g);
  }

  // An empty /tstt/sets, which MOAB requires.
  void no_sets() { H5Gclose(group("sets/tags")); }

  // n sets with no contents, children or parents.
  void sets(std::int64_t start, std::size_t n) {
    std::vector<std::int64_t> rows;
    for (std::size_t i = 0; i < n; ++i) rows.insert(rows.end(), {-1, -1, -1, 2});
    set_list(start, rows);
  }

  // One set: its flags and its runs of contents, children and parents.
  struct Set {
    std::int64_t flags = 2;
    std::vector<std::uint64_t> contents;
    std::vector<std::uint64_t> children;
    std::vector<std::uint64_t> parents;
  };

  // Sets with file ids from start on, the list's ends taken from their runs.
  // A column no set uses gets no dataset.
  void sets(std::int64_t start, const std::vector<Set>& s) {
    std::vector<std::int64_t> rows;
    std::array<std::vector<std::uint64_t>, 3> columns;
    for (const Set& x : s) {
      columns[0].insert(columns[0].end(), x.contents.begin(), x.contents.end());
      columns[1].insert(columns[1].end(), x.children.begin(), x.children.end());
      columns[2].insert(columns[2].end(), x.parents.begin(), x.parents.end());
      for (const auto& c : columns) rows.push_back(static_cast<std::int64_t>(c.size()) - 1);
      rows.push_back(x.flags);
    }
    set_list(start, rows);
    const char* names[] = {"contents", "children", "parents"};
    for (std::size_t c = 0; c < columns.size(); ++c) {
      if (!columns[c].empty()) set_column(names[c], columns[c]);
    }
  }

  // /tstt/sets/list as given, four values per set.
  void set_list(std::int64_t start, const std::vector<std::int64_t>& rows) {
    const hid_t g = group("sets");
    const hsize_t dims[2] = {rows.size() / 4, 4};
    const hid_t ds = dataset(g, "list", H5T_STD_I64LE, 2, dims);
    write(ds, H5T_NATIVE_INT64, rows.data(), rows.size());
    scalar_attribute(ds, "start_id", H5T_NATIVE_INT64, &start);
    note(start, dims[0]);
    H5Dclose(ds);
    H5Gclose(g);
  }

  // /tstt/sets/<name> ("contents", "children", "parents") as given.
  void set_column(const std::string& name, const std::vector<std::uint64_t>& ids) {
    const hid_t g = group("sets");
    const hsize_t n = ids.size();
    const hid_t ds = dataset(g, name.c_str(), H5T_STD_U64LE, 1, &n);
    write(ds, H5T_NATIVE_UINT64, ids.data(), ids.size());
    H5Dclose(ds);
    H5Gclose(g);
  }

  // Defines tag `name` with value type `type` (a memory type, committed as is)
  // and MOAB storage class cls. def and global point at one value, if given.
  void tag(const std::string& name, int cls, hid_t type, const void* def = nullptr,
           const void* global = nullptr) {
    const hid_t g = group("tags/" + name);
    const hid_t t = ok(H5Tcopy(type), "copy a tag type");
    check(H5Tcommit2(g, "type", t, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "commit a tag type");
    std::int32_t c = cls;
    scalar_attribute(g, "class", H5T_NATIVE_INT32, &c);
    if (def != nullptr) scalar_attribute(g, "default", t, def);
    if (global != nullptr) scalar_attribute(g, "global", t, global);
    H5Tclose(t);
    H5Gclose(g);
  }

  void int_attribute(const std::string& path, const char* name, std::int32_t v) {
    const hid_t o = ok(H5Oopen(tstt_, path.c_str(), H5P_DEFAULT), "open an object");
    scalar_attribute(o, name, H5T_NATIVE_INT32, &v);
    H5Oclose(o);
  }

  // One value per entity of base ("nodes", "elements/Quad4", "sets").
  void dense(const std::string& base, const std::string& name, const void* values, std::size_t n) {
    const hid_t t = tag_type(name);
    const hid_t g = group(base + "/tags");
    const hsize_t dims = n;
    const hid_t ds = dataset(g, name.c_str(), t, 1, &dims);
    if (n > 0) check(H5Dwrite(ds, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, values), "write dense values");
    H5Dclose(ds);
    H5Gclose(g);
    H5Tclose(t);
  }

  void sparse(const std::string& name, const std::vector<std::uint64_t>& ids, const void* values) {
    const hid_t t = tag_type(name);
    const hid_t g = group("tags/" + name);
    const hsize_t n = ids.size();
    const hid_t il = dataset(g, "id_list", H5T_STD_U64LE, 1, &n);
    write(il, H5T_NATIVE_UINT64, ids.data(), ids.size());
    const hid_t vs = dataset(g, "values", t, 1, &n);
    if (n > 0) check(H5Dwrite(vs, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, values), "write sparse values");
    H5Dclose(vs);
    H5Dclose(il);
    H5Gclose(g);
    H5Tclose(t);
  }

  // A variable-length tag's values: file id ids[i] holds the run of values
  // that ends at index last[i] of `values` (n of them, of the tag's type).
  void var_sparse(const std::string& name, const std::vector<std::uint64_t>& ids,
                  const std::vector<std::int64_t>& last, const void* values, std::size_t n) {
    const hid_t t = tag_type(name);
    const hid_t g = group("tags/" + name);
    const hsize_t ni = ids.size();
    const hid_t il = dataset(g, "id_list", H5T_STD_U64LE, 1, &ni);
    write(il, H5T_NATIVE_UINT64, ids.data(), ids.size());
    const hsize_t nx = last.size();
    const hid_t vi = dataset(g, "var_indices", H5T_STD_I64LE, 1, &nx);
    write(vi, H5T_NATIVE_INT64, last.data(), last.size());
    const hsize_t nv = n;
    const hid_t vs = dataset(g, "values", t, 1, &nv);
    if (n > 0) check(H5Dwrite(vs, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, values), "write variable-length values");
    H5Dclose(vs);
    H5Dclose(vi);
    H5Dclose(il);
    H5Gclose(g);
    H5Tclose(t);
  }

  // Attribute `name` of tag `tag`: one value of an array of n of the tag's type.
  void array_attribute(const std::string& tag, const char* name, const void* values, std::size_t n) {
    const hid_t t = tag_type(tag);
    const hsize_t dims = n;
    const hid_t a = ok(H5Tarray_create2(t, 1, &dims), "make an array type");
    const hid_t g = group("tags/" + tag);
    scalar_attribute(g, name, a, values);
    H5Gclose(g);
    H5Tclose(a);
    H5Tclose(t);
  }

 private:
  // Opens path below /tstt, creating the groups on the way.
  hid_t group(const std::string& path) {
    std::size_t at = 0;
    while (true) {
      const std::size_t slash = path.find('/', at);
      const std::string prefix = path.substr(0, slash);
      if (H5Lexists(tstt_, prefix.c_str(), H5P_DEFAULT) <= 0) {
        H5Gclose(ok(H5Gcreate2(tstt_, prefix.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "create a group"));
      }
      if (slash == std::string::npos) break;
      at = slash + 1;
    }
    return ok(H5Gopen2(tstt_, path.c_str(), H5P_DEFAULT), "open a group");
  }

  hid_t element_group(const std::string& name, const std::string& type) {
    const hid_t g = group("elements/" + name);
    std::uint8_t v = 0;
    check(H5Tenum_valueof(types_, type.c_str(), &v), "find an element type");
    scalar_attribute(g, "element_type", types_, &v);
    return g;
  }

  hid_t tag_type(const std::string& name) {
    return ok(H5Topen2(tstt_, ("tags/" + name + "/type").c_str(), H5P_DEFAULT), "open a tag type");
  }

  static hid_t dataset(hid_t loc, const char* name, hid_t type, int rank, const hsize_t* dims) {
    const hid_t space = ok(H5Screate_simple(rank, dims, nullptr), "create a dataspace");
    const hid_t ds = ok(H5Dcreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                        "create a dataset");
    H5Sclose(space);
    return ds;
  }

  static void write(hid_t ds, hid_t mem, const void* data, std::size_t n) {
    if (n > 0) check(H5Dwrite(ds, mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), "write a dataset");
  }

  static void scalar_attribute(hid_t obj, const char* name, hid_t type, const void* value) {
    const hid_t space = ok(H5Screate(H5S_SCALAR), "create a scalar dataspace");
    const hid_t a = ok(H5Acreate2(obj, name, type, space, H5P_DEFAULT, H5P_DEFAULT), "create an attribute");
    check(H5Awrite(a, type, value), "write an attribute");
    H5Aclose(a);
    H5Sclose(space);
  }

  void note(std::int64_t start, std::uint64_t n) {
    if (n > 0) largest_ = std::max(largest_, static_cast<std::uint64_t>(start) + n - 1);
  }

  hid_t file_ = -1;
  hid_t tstt_ = -1;
  hid_t types_ = -1;
  std::uint64_t largest_ = 0;
  std::uint64_t max_id_ = 0;
  bool max_id_set_ = false;
  bool write_max_id_ = true;
};

}  // namespace soma_test::h5m

#endif  // SOMA_TESTS_UNIT_H5M_WRITER_HPP_
