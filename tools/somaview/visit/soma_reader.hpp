// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.
// soma_reader.hpp -- header-only C++17 reader for SOMA mesh and AMR checkpoint files.
//
// Mirrors somaview's Python core (format.py, cells.py, mesh.py, amr.py) over the HDF5 C API,
// with no dependency on VTK or VisIt. It is meant to be driven by a VisIt (or other) database
// plugin, which maps the VTK cell type codes and point/cell arrays this header produces onto
// its own data model.
//
// Layout recap (see somaview's format.py docstring for the full story): a SOMA mesh file keeps
// everything under the group /soma. Datasets that every "part" (what SOMA calls its domain
// pieces) writes rows of are concatenated in part order; /soma/parts/offsets (nparts + 1 rows,
// one column per such dataset) says where each part's rows start. A part owns its own vertices;
// an element may reference vertices owned by other parts, discovered through the sharing table
// (dim0/shared/index, counts, parts).
//
// This header only reads; it never writes a SOMA file.

#ifndef SOMA_READER_HPP
#define SOMA_READER_HPP

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace soma_reader {

using GO = std::int64_t;  // global id
using LO = std::int64_t;  // local row / index

// ---------------------------------------------------------------------------------------------
// soma::Topology codes, as stored in dimD/topology (soma/topology/canonical.hpp).
enum class Topology : int {
  vertex = 0, edge = 1, tri = 2, quad = 3, polygon = 4, tet = 5, pyramid = 6, prism = 7,
  knife = 8, hex = 9, polyhedron = 10
};

// soma::Layout::Kind: where a field's rows sit on its entities.
enum class LayoutKind : int { per_entity = 0, element_nodes = 1, shared_nodes = 2, quadrature = 3 };

// soma::NodeFamily::gll, soma::NodeOrdering::tensor_lexicographic and soma::max_node_order.
constexpr int FAMILY_GLL = 1, ORDERING_TENSOR = 1, MAX_NODE_ORDER = 32;

enum class FileKind { mesh, amr, unknown };

// soma::DataType codes, in enum order (format.py DTYPES).
enum class DType : int { i8 = 0, u8 = 1, i32 = 2, u32 = 3, i64 = 4, u64 = 5, f32 = 6, f64 = 7 };

inline std::size_t dtype_size(DType t) {
  switch (t) {
    case DType::i8: case DType::u8: return 1;
    case DType::i32: case DType::u32: case DType::f32: return 4;
    default: return 8;
  }
}

// Slot of the entity sets in a field's dimension list (soma::set_dim).
constexpr int SET_DIM = 4;
constexpr int FORMAT_VERSION = 1;

// -- VTK cell type codes (vtkCellType.h), kept here so this header has no VTK dependency --------
namespace vtk_type {
constexpr int VERTEX = 1, LINE = 3, TRIANGLE = 5, POLYGON = 7, QUAD = 9;
constexpr int TETRA = 10, HEXAHEDRON = 12, WEDGE = 13, PYRAMID = 14;
constexpr int QUADRATIC_EDGE = 21, QUADRATIC_TRIANGLE = 22, QUADRATIC_QUAD = 23;
constexpr int QUADRATIC_TETRA = 24, QUADRATIC_HEXAHEDRON = 25, QUADRATIC_WEDGE = 26, QUADRATIC_PYRAMID = 27;
constexpr int BIQUADRATIC_QUAD = 28, TRIQUADRATIC_HEXAHEDRON = 29, BIQUADRATIC_TRIANGLE = 34;
constexpr int CONVEX_POINT_SET = 41, POLYHEDRON = 42;
}  // namespace vtk_type

// The knife's faces in canonical order (soma/topology/canonical.hpp), as positions in its
// 7-vertex corner list. A knife is written as a VTK_POLYHEDRON with these faces.
inline const std::array<std::array<int, 4>, 5>& knife_faces() {
  static const std::array<std::array<int, 4>, 5> f{{
      {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 5, 6}, {3, 0, 4, 5}, {0, 3, 2, 1}}};
  return f;
}

inline std::runtime_error io_error(const std::string& what) { return std::runtime_error(what); }

// -------------------------------------------------------------------------------------------
// Minimal HDF5 C API wrappers: open file/group/dataset/attribute, read with automatic type
// conversion to int64 or double (HDF5 converts between any of the stored numeric types and the
// requested native type), with or without a row-range hyperslab selection.
// -------------------------------------------------------------------------------------------
namespace h5detail {

inline void check(hid_t id, const std::string& what) {
  if (id < 0) throw io_error("soma_reader: " + what);
}

struct File {
  hid_t id = -1;
  explicit File(const std::string& path) {
    id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    check(id, "cannot open '" + path + "' as HDF5");
  }
  ~File() { if (id >= 0) H5Fclose(id); }
  File(const File&) = delete;
  File& operator=(const File&) = delete;
};

// True when `path` (relative to /soma) names a dataset or group.
inline bool exists(hid_t file, const std::string& soma_relpath) {
  std::string full = "/soma/" + soma_relpath;
  // H5Lexists on each path component in turn: a missing intermediate group must not raise.
  std::string cur;
  std::istringstream ss(full.substr(1));
  std::string tok;
  while (std::getline(ss, tok, '/')) {
    cur += "/" + tok;
    htri_t e = H5Lexists(file, cur.c_str(), H5P_DEFAULT);
    if (e <= 0) return false;
  }
  return true;
}

// Dataset extents: rank 1 (n) or rank 2 (n, ncomp). Returns (n, ncomp) with ncomp = 1 for rank 1.
inline std::pair<hsize_t, hsize_t> dataset_extent(hid_t file, const std::string& path) {
  hid_t ds = H5Dopen2(file, ("/soma/" + path).c_str(), H5P_DEFAULT);
  check(ds, "no dataset /soma/" + path);
  hid_t sp = H5Dget_space(ds);
  int rank = H5Sget_simple_extent_ndims(sp);
  hsize_t dims[2] = {0, 1};
  H5Sget_simple_extent_dims(sp, dims, nullptr);
  H5Sclose(sp);
  H5Dclose(ds);
  hsize_t ncomp = rank == 2 ? dims[1] : 1;
  return {dims[0], ncomp};
}

// Reads rows [start, start + count) of `path`, converted to T (T = int64_t or double), with
// `ncomp` set to the dataset's row width. An empty read (count == 0) still reports ncomp.
template <class T>
inline std::vector<T> read_rows(hid_t file, const std::string& path, hsize_t start, hsize_t count,
                                 hsize_t* ncomp_out = nullptr) {
  hid_t ds = H5Dopen2(file, ("/soma/" + path).c_str(), H5P_DEFAULT);
  check(ds, "no dataset /soma/" + path);
  hid_t sp = H5Dget_space(ds);
  int rank = H5Sget_simple_extent_ndims(sp);
  hsize_t dims[2] = {0, 1};
  H5Sget_simple_extent_dims(sp, dims, nullptr);
  hsize_t ncomp = rank == 2 ? dims[1] : 1;
  if (ncomp_out) *ncomp_out = ncomp;
  std::vector<T> out(static_cast<std::size_t>(count * ncomp));
  if (count > 0) {
    hsize_t off[2] = {start, 0};
    hsize_t cnt[2] = {count, ncomp};
    H5Sselect_hyperslab(sp, H5S_SELECT_SET, off, nullptr, cnt, nullptr);
    hid_t msp = H5Screate_simple(rank, cnt, nullptr);
    hid_t mtype = std::is_same<T, double>::value ? H5T_NATIVE_DOUBLE
                  : std::is_same<T, std::int64_t>::value ? H5T_NATIVE_INT64
                  : std::is_same<T, std::int32_t>::value ? H5T_NATIVE_INT32
                  : std::is_same<T, std::uint8_t>::value ? H5T_NATIVE_UINT8
                                                          : H5T_NATIVE_INT64;
    herr_t e = H5Dread(ds, mtype, msp, sp, H5P_DEFAULT, out.data());
    H5Sclose(msp);
    if (e < 0) {
      H5Sclose(sp);
      H5Dclose(ds);
      throw io_error("read failed: /soma/" + path + " rows [" + std::to_string(start) + ", " +
                     std::to_string(start + count) + ") of " + std::to_string(dims[0]));
    }
  }
  H5Sclose(sp);
  H5Dclose(ds);
  return out;
}

template <class T>
inline std::vector<T> read_all(hid_t file, const std::string& path, hsize_t* ncomp_out = nullptr) {
  auto [n, ncomp] = dataset_extent(file, path);
  (void)ncomp;
  return read_rows<T>(file, path, 0, n, ncomp_out);
}

// Reads an attribute of /soma/<group> (group == "" for /soma itself) as int64, with automatic
// conversion from whatever integer type it is stored as.
inline std::vector<std::int64_t> read_int_attr(hid_t file, const std::string& group,
                                               const std::string& name) {
  std::string path = group.empty() ? "/soma" : "/soma/" + group;
  hid_t obj = H5Oopen(file, path.c_str(), H5P_DEFAULT);
  check(obj, "no group " + path);
  hid_t at = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
  if (at < 0) { H5Oclose(obj); throw io_error("no attribute " + name + " on " + path); }
  hid_t sp = H5Aget_space(at);
  hssize_t n = H5Sget_simple_extent_npoints(sp);
  H5Sclose(sp);
  std::vector<std::int64_t> out(static_cast<std::size_t>(std::max<hssize_t>(n, 0)));
  if (n > 0) H5Aread(at, H5T_NATIVE_INT64, out.data());
  H5Aclose(at);
  H5Oclose(obj);
  return out;
}

// Reads a (fixed- or variable-length) string attribute.
inline std::string read_string_attr(hid_t file, const std::string& group, const std::string& name) {
  std::string path = group.empty() ? "/soma" : "/soma/" + group;
  hid_t obj = H5Oopen(file, path.c_str(), H5P_DEFAULT);
  check(obj, "no group " + path);
  hid_t at = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
  if (at < 0) { H5Oclose(obj); throw io_error("no attribute " + name + " on " + path); }
  hid_t ty = H5Aget_type(at);
  std::string out;
  if (H5Tis_variable_str(ty) > 0) {
    char* p = nullptr;
    hid_t mty = H5Tcopy(H5T_C_S1);
    H5Tset_size(mty, H5T_VARIABLE);
    H5Aread(at, mty, &p);
    if (p) { out = p; free(p); }
    H5Tclose(mty);
  } else {
    std::size_t sz = H5Tget_size(ty);
    std::vector<char> buf(sz + 1, 0);
    H5Aread(at, ty, buf.data());
    out.assign(buf.data(), strnlen(buf.data(), sz));
  }
  H5Tclose(ty);
  H5Aclose(at);
  H5Oclose(obj);
  return out;
}

// Raw bytes of an attribute (for the "default"/"global" byte-blob attributes field and value
// groups store, reinterpreted per the field's dtype -- see `typed_attr` below).
inline std::optional<std::vector<std::uint8_t>> read_raw_attr(hid_t file, const std::string& group,
                                                               const std::string& name) {
  std::string path = group.empty() ? "/soma" : "/soma/" + group;
  hid_t obj = H5Oopen(file, path.c_str(), H5P_DEFAULT);
  check(obj, "no group " + path);
  hid_t at = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
  if (at < 0) { H5Oclose(obj); return std::nullopt; }
  hid_t sp = H5Aget_space(at);
  hssize_t n = H5Sget_simple_extent_npoints(sp);
  H5Sclose(sp);
  std::vector<std::uint8_t> out(static_cast<std::size_t>(std::max<hssize_t>(n, 0)));
  if (n > 0) H5Aread(at, H5T_NATIVE_UINT8, out.data());
  H5Aclose(at);
  H5Oclose(obj);
  return out;
}

inline bool has_attr(hid_t file, const std::string& group, const std::string& name) {
  std::string path = group.empty() ? "/soma" : "/soma/" + group;
  hid_t obj = H5Oopen(file, path.c_str(), H5P_DEFAULT);
  if (obj < 0) return false;
  htri_t e = H5Aexists(obj, name.c_str());
  H5Oclose(obj);
  return e > 0;
}

}  // namespace h5detail

// -------------------------------------------------------------------------------------------
// Field and root metadata (format.py FieldInfo / RootMeta / MeshValue).
// -------------------------------------------------------------------------------------------

struct FieldInfo {
  int index = 0;
  std::string name;
  DType dtype = DType::f64;
  int ncomp = 1;
  bool variable = false;
  bool handle = false;
  LayoutKind layout = LayoutKind::per_entity;
  int order = 0;
  int family = 0, ordering = 0;
  bool serendipity = false;
  int full_bits = 0;
  std::vector<int> dims;

  std::string group() const { return "fields/f" + std::to_string(index); }

  // True for fields with one fixed-size row per entity: the ones a renderer can show.
  bool drawable() const { return layout == LayoutKind::per_entity && !variable && !handle; }

  // True for a spectral-element field: GLL values on quads, drawn on sub-quads (see subcells).
  bool nodal_on_quads() const {
    return layout == LayoutKind::element_nodes && !variable && !handle && family == FAMILY_GLL &&
           ordering == ORDERING_TENSOR && !serendipity && order >= 1 && order <= MAX_NODE_ORDER && has_dim(2);
  }

  bool full_on(int dim) const { return ((full_bits >> dim) & 1) != 0; }

  bool has_dim(int d) const { return std::find(dims.begin(), dims.end(), d) != dims.end(); }
};

struct MeshValue {
  std::string name;
  DType dtype = DType::f64;
  int ncomp = 1;
  bool has_value = false;
  std::vector<double> value;  // reinterpreted per dtype, promoted to double
};

struct RootMeta {
  int format_version = 0, dim = 0, nparts = 0;
  std::vector<int> written, input, nodes;
  int graphs = 0, nsets = 0, nfields = 0, nvalues = 0, ordering = 0;
};

namespace detail {
// Throws unless the attribute `what` holds at least `n` words (a truncated or foreign file).
inline void need_words(const std::vector<std::int64_t>& w, std::size_t n, const std::string& what) {
  if (w.size() < n)
    throw io_error(what + " has " + std::to_string(w.size()) + " words, fewer than " + std::to_string(n));
}

inline std::vector<int> mask_bits(std::int64_t mask) {
  std::vector<int> out;
  for (int d = 0; d < 4; ++d)
    if ((mask >> d) & 1) out.push_back(d);
  return out;
}

// Reinterprets a raw byte attribute as `ncomp` little-endian values of `dtype`, promoted to
// double (format.py's _typed, for the field/value "default"/"global" attributes).
inline std::vector<double> typed_attr(const std::optional<std::vector<std::uint8_t>>& raw, DType dtype,
                                      int ncomp) {
  std::vector<double> out;
  if (!raw) return out;
  if (ncomp < 0 || raw->size() < static_cast<std::size_t>(ncomp) * dtype_size(dtype))
    throw io_error("an attribute of " + std::to_string(raw->size()) + " bytes is shorter than its " +
                   std::to_string(ncomp) + " components");
  const std::uint8_t* p = raw->data();
  out.resize(static_cast<std::size_t>(ncomp));
  for (int c = 0; c < ncomp; ++c) {
    switch (dtype) {
      case DType::i8: out[c] = static_cast<double>(*reinterpret_cast<const std::int8_t*>(p)); break;
      case DType::u8: out[c] = static_cast<double>(*reinterpret_cast<const std::uint8_t*>(p)); break;
      case DType::i32: out[c] = static_cast<double>(*reinterpret_cast<const std::int32_t*>(p)); break;
      case DType::u32: out[c] = static_cast<double>(*reinterpret_cast<const std::uint32_t*>(p)); break;
      case DType::i64: out[c] = static_cast<double>(*reinterpret_cast<const std::int64_t*>(p)); break;
      case DType::u64: out[c] = static_cast<double>(*reinterpret_cast<const std::uint64_t*>(p)); break;
      case DType::f32: out[c] = static_cast<double>(*reinterpret_cast<const float*>(p)); break;
      case DType::f64: out[c] = *reinterpret_cast<const double*>(p); break;
    }
    p += dtype_size(dtype);
  }
  return out;
}

// Maximal runs [a, b) of consecutive values in a sorted vector of distinct integers (format.py
// runs()); lets a multi-part read become one hyperslab per contiguous group of parts.
inline std::vector<std::pair<LO, LO>> runs(const std::vector<LO>& sorted_distinct) {
  std::vector<std::pair<LO, LO>> out;
  std::size_t i = 0, n = sorted_distinct.size();
  while (i < n) {
    std::size_t j = i + 1;
    while (j < n && sorted_distinct[j] == sorted_distinct[j - 1] + 1) ++j;
    out.emplace_back(sorted_distinct[i], sorted_distinct[j - 1] + 1);
    i = j;
  }
  return out;
}
}  // namespace detail

// -------------------------------------------------------------------------------------------
// SomaFile: an open SOMA file (mesh or AMR checkpoint), with the part table, field and value
// metadata parsed. Reading of entity data is done through `read_i64`/`read_f64`, which take the
// same "parts" selector as somaview's SomaFile.read.
// -------------------------------------------------------------------------------------------
class SomaFile {
 public:
  explicit SomaFile(const std::string& path) : path_(path), file_(path) {
    if (!h5detail::exists(file_.id, "")) throw io_error(path + ": not a SOMA file (no /soma group)");
    if (h5detail::has_attr(file_.id, "", "amr")) {
      kind_ = FileKind::amr;
      auto w = h5detail::read_int_attr(file_.id, "", "amr");
      detail::need_words(w, 3, path + ": /soma attribute amr");
      format_version_ = static_cast<int>(w[0]);
      nparts_ = static_cast<int>(w[2]);
    } else if (h5detail::has_attr(file_.id, "", "meta")) {
      kind_ = FileKind::mesh;
      auto w = h5detail::read_int_attr(file_.id, "", "meta");
      detail::need_words(w, 10, path + ": /soma attribute meta");
      meta_.format_version = static_cast<int>(w[0]);
      meta_.dim = static_cast<int>(w[1]);
      meta_.nparts = static_cast<int>(w[2]);
      meta_.written = detail::mask_bits(w[3]);
      meta_.input = detail::mask_bits(w[4]);
      meta_.nodes = detail::mask_bits(w[5]);
      meta_.graphs = static_cast<int>(w[6]);
      meta_.nsets = static_cast<int>(w[7]);
      meta_.nfields = static_cast<int>(w[8]);
      meta_.nvalues = static_cast<int>(w[9]);
      meta_.ordering = w.size() > 10 ? static_cast<int>(w[10]) : 0;
      format_version_ = meta_.format_version;
      nparts_ = meta_.nparts;
    } else {
      throw io_error(path + ": /soma has neither the attribute meta nor amr");
    }
    if (format_version_ > FORMAT_VERSION)
      throw io_error(path + ": format_version " + std::to_string(format_version_) +
                     " is newer than this reader (" + std::to_string(FORMAT_VERSION) + ")");
    if (h5detail::exists(file_.id, "parts/offsets")) {
      auto [n, ncols] = h5detail::dataset_extent(file_.id, "parts/offsets");
      table_ncols_ = static_cast<std::size_t>(ncols);
      table_ = h5detail::read_all<std::int64_t>(file_.id, "parts/offsets");  // uint64 -> int64, safe here
      std::string cols = h5detail::read_string_attr(file_.id, "parts/offsets", "columns");
      std::istringstream ss(cols);
      std::string tok;
      int k = 0;
      while (std::getline(ss, tok, '\n')) {
        if (!tok.empty()) columns_[tok] = k++;
      }
      if (nparts_ < 0 || n != static_cast<decltype(n)>(nparts_) + 1 || static_cast<std::size_t>(k) > table_ncols_)
        throw io_error(path + ": parts/offsets is " + std::to_string(n) + " x " + std::to_string(ncols) + " for " +
                       std::to_string(nparts_) + " parts and " + std::to_string(k) + " columns");
    }
    if (kind_ == FileKind::mesh) { read_fields(); read_values(); }
  }

  const std::string& path() const { return path_; }
  FileKind kind() const { return kind_; }
  int format_version() const { return format_version_; }
  int nparts() const { return nparts_; }
  const RootMeta& meta() const { return meta_; }
  int dim() const { return meta_.dim; }
  const std::vector<int>& written_dims() const { return meta_.written; }
  const std::vector<FieldInfo>& fields() const { return fields_; }
  const std::map<std::string, MeshValue>& values() const { return values_; }
  hid_t raw() const { return file_.id; }

  bool has(const std::string& relpath) const { return h5detail::exists(file_.id, relpath); }
  bool is_column(const std::string& relpath) const { return columns_.count(relpath) != 0; }

  // Rows [start, stop) of `path` that `part` wrote.
  std::pair<LO, LO> part_rows(const std::string& path, int part) const {
    auto it = columns_.find(path);
    if (it == columns_.end()) throw io_error(path_ + ": '" + path + "' is not a per-part column");
    if (part < 0 || part >= nparts_)
      throw io_error(path_ + ": part " + std::to_string(part) + " of " + std::to_string(nparts_));
    LO lo = table_[static_cast<std::size_t>(part) * table_ncols_ + static_cast<std::size_t>(it->second)];
    LO hi = table_[static_cast<std::size_t>(part + 1) * table_ncols_ + static_cast<std::size_t>(it->second)];
    return {lo, hi};
  }

  // The N + 1 row starts of `path` (one per part, plus the total).
  std::vector<LO> part_starts(const std::string& path) const {
    auto it = columns_.find(path);
    if (it == columns_.end()) throw io_error(path_ + ": '" + path + "' is not a per-part column");
    std::vector<LO> out(static_cast<std::size_t>(nparts_) + 1);
    for (std::size_t p = 0; p < out.size(); ++p) out[p] = table_[p * table_ncols_ + static_cast<std::size_t>(it->second)];
    return out;
  }

  LO count(int dim, const std::vector<int>* parts = nullptr) const {
    std::string path = "dim" + std::to_string(dim) + "/gid";
    if (!is_column(path)) return 0;
    auto starts = part_starts(path);
    if (!parts) return starts.back();
    LO n = 0;
    for (int p : *parts) n += starts[static_cast<std::size_t>(p) + 1] - starts[static_cast<std::size_t>(p)];
    return n;
  }

  std::vector<LO> counts_by_part(int dim) const {
    std::string path = "dim" + std::to_string(dim) + "/gid";
    std::vector<LO> out(static_cast<std::size_t>(nparts_), 0);
    if (!is_column(path)) return out;
    auto starts = part_starts(path);
    for (int p = 0; p < nparts_; ++p) out[static_cast<std::size_t>(p)] = starts[p + 1] - starts[p];
    return out;
  }

  // The rows `parts` wrote of a per-part dataset, in part order (every row with parts==nullptr).
  // `ncomp_out`, if given, receives the dataset's row width.
  template <class T>
  std::vector<T> read(const std::string& path, const std::vector<int>* parts = nullptr,
                      hsize_t* ncomp_out = nullptr) const {
    if (!parts) return h5detail::read_all<T>(file_.id, path, ncomp_out);
    auto starts = part_starts(path);
    std::vector<LO> uniq(parts->begin(), parts->end());
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    std::vector<T> out;
    hsize_t ncomp = 1;
    bool have_ncomp = false;
    for (auto [a, b] : detail::runs(uniq)) {
      LO lo = starts[static_cast<std::size_t>(a)], hi = starts[static_cast<std::size_t>(b)];
      if (hi <= lo) continue;
      hsize_t nc;
      auto chunk = h5detail::read_rows<T>(file_.id, path, static_cast<hsize_t>(lo),
                                          static_cast<hsize_t>(hi - lo), &nc);
      if (!have_ncomp) { ncomp = nc; have_ncomp = true; }
      out.insert(out.end(), chunk.begin(), chunk.end());
    }
    if (!have_ncomp) { auto ext = h5detail::dataset_extent(file_.id, path); ncomp = ext.second; }
    if (ncomp_out) *ncomp_out = ncomp;
    return out;
  }

  // Rows at arbitrary global row indices (any order, repeats allowed), coalesced into hyperslabs
  // merging runs closer than `gap` rows apart (format.py's read_rows).
  template <class T>
  std::vector<T> read_rows_at(const std::string& path, const std::vector<LO>& rows, hsize_t* ncomp_out,
                              LO gap = 4096) const {
    hsize_t ncomp;
    auto ext = h5detail::dataset_extent(file_.id, path);
    ncomp = ext.second;
    if (ncomp_out) *ncomp_out = ncomp;
    if (rows.empty()) return {};
    std::vector<std::size_t> order(rows.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return rows[a] < rows[b]; });
    std::vector<T> out(rows.size() * static_cast<std::size_t>(ncomp));
    std::size_t i = 0;
    while (i < order.size()) {
      std::size_t j = i;
      LO lo = rows[order[i]], hi = rows[order[i]] + 1;
      while (j + 1 < order.size() && rows[order[j + 1]] - hi <= gap) {
        ++j;
        hi = std::max(hi, rows[order[j]] + 1);
      }
      auto block = h5detail::read_rows<T>(file_.id, path, static_cast<hsize_t>(lo),
                                          static_cast<hsize_t>(hi - lo));
      for (std::size_t k = i; k <= j; ++k) {
        LO r = rows[order[k]] - lo;
        std::copy(block.begin() + static_cast<std::ptrdiff_t>(r * static_cast<LO>(ncomp)),
                 block.begin() + static_cast<std::ptrdiff_t>((r + 1) * static_cast<LO>(ncomp)),
                 out.begin() + static_cast<std::ptrdiff_t>(order[k] * static_cast<std::size_t>(ncomp)));
      }
      i = j + 1;
    }
    return out;
  }

  const FieldInfo* field(const std::string& name) const {
    for (auto& f : fields_) if (f.name == name) return &f;
    return nullptr;
  }

  std::optional<double> time() const {
    for (auto& [name, v] : values_) {
      std::string lower = name;
      std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
      if (lower == "time" && v.has_value && !v.value.empty()) return v.value[0];
    }
    return std::nullopt;
  }

 private:
  void read_fields() {
    fields_.reserve(static_cast<std::size_t>(meta_.nfields));
    for (int i = 0; i < meta_.nfields; ++i) {
      std::string g = "fields/f" + std::to_string(i);
      auto m = h5detail::read_int_attr(file_.id, g, "meta");
      detail::need_words(m, 17, path_ + ": " + g + " attribute meta");
      if (m[16] < 0 || static_cast<std::size_t>(m[16]) > m.size() - 17)
        throw io_error(path_ + ": " + g + " attribute meta gives " + std::to_string(m[16]) + " dimensions in " +
                       std::to_string(m.size()) + " words");
      FieldInfo f;
      f.index = i;
      f.name = h5detail::read_string_attr(file_.id, g, "name");
      f.dtype = static_cast<DType>(m[0]);
      f.ncomp = static_cast<int>(m[1]);
      f.variable = m[2] != 0;
      f.handle = m[3] != 0;
      f.layout = static_cast<LayoutKind>(m[4]);
      f.family = static_cast<int>(m[5]);
      f.order = static_cast<int>(m[6]);
      f.ordering = static_cast<int>(m[7]);
      f.serendipity = m[8] != 0;
      f.full_bits = static_cast<int>(m[15]);
      int ndims = static_cast<int>(m[16]);
      for (int d = 0; d < ndims; ++d) f.dims.push_back(static_cast<int>(m[17 + d]));
      fields_.push_back(std::move(f));
    }
  }

  void read_values() {
    for (int i = 0; i < meta_.nvalues; ++i) {
      std::string g = "values/v" + std::to_string(i);
      auto m = h5detail::read_int_attr(file_.id, g, "meta");
      detail::need_words(m, 2, path_ + ": " + g + " attribute meta");
      MeshValue v;
      v.dtype = static_cast<DType>(m[0]);
      v.ncomp = static_cast<int>(m[1]);
      v.name = h5detail::read_string_attr(file_.id, g, "name");
      bool has_global = m.size() > 7 && m[7] != 0;
      bool has_default = m.size() > 6 && m[6] != 0;
      std::optional<std::vector<std::uint8_t>> raw;
      if (has_global) raw = h5detail::read_raw_attr(file_.id, g, "global");
      else if (has_default) raw = h5detail::read_raw_attr(file_.id, g, "default");
      if (raw) { v.value = detail::typed_attr(raw, v.dtype, v.ncomp); v.has_value = true; }
      values_[v.name] = std::move(v);
    }
  }

  std::string path_;
  h5detail::File file_;
  FileKind kind_ = FileKind::unknown;
  int format_version_ = 0;
  int nparts_ = 0;
  RootMeta meta_;
  std::vector<std::int64_t> table_;
  std::size_t table_ncols_ = 0;
  std::map<std::string, int> columns_;
  std::vector<FieldInfo> fields_;
  std::map<std::string, MeshValue> values_;
};

// -------------------------------------------------------------------------------------------
// GidIndex: positions of global ids in a set of distinct gids (mesh.py's GidIndex). A dense
// lookup table when the gid range is not much larger than the count, else binary search over a
// sorted copy -- the same trade-off the Python core makes.
// -------------------------------------------------------------------------------------------
class GidIndex {
 public:
  explicit GidIndex(const std::vector<GO>& gids) : n_(static_cast<LO>(gids.size())) {
    if (n_ == 0) return;
    GO lo = *std::min_element(gids.begin(), gids.end());
    GO hi = *std::max_element(gids.begin(), gids.end());
    if (hi - lo + 1 <= 4 * n_ + 1024) {
      dense_ = true;
      lo_ = lo;
      table_.assign(static_cast<std::size_t>(hi - lo + 1), -1);
      for (LO i = 0; i < n_; ++i) table_[static_cast<std::size_t>(gids[static_cast<std::size_t>(i)] - lo)] = i;
    } else {
      order_.resize(static_cast<std::size_t>(n_));
      for (LO i = 0; i < n_; ++i) order_[static_cast<std::size_t>(i)] = i;
      std::sort(order_.begin(), order_.end(),
               [&](LO a, LO b) { return gids[static_cast<std::size_t>(a)] < gids[static_cast<std::size_t>(b)]; });
      sorted_.resize(static_cast<std::size_t>(n_));
      for (LO i = 0; i < n_; ++i) sorted_[static_cast<std::size_t>(i)] = gids[static_cast<std::size_t>(order_[static_cast<std::size_t>(i)])];
    }
  }

  // Position of `g`, or -1 if `g` is not in the index.
  LO lookup_one(GO g) const {
    if (n_ == 0) return -1;
    if (dense_) {
      GO k = g - lo_;
      if (k < 0 || static_cast<std::size_t>(k) >= table_.size()) return -1;
      return table_[static_cast<std::size_t>(k)];
    }
    auto it = std::lower_bound(sorted_.begin(), sorted_.end(), g);
    if (it == sorted_.end() || *it != g) return -1;
    return order_[static_cast<std::size_t>(it - sorted_.begin())];
  }

  std::vector<LO> lookup(const std::vector<GO>& q) const {
    std::vector<LO> out(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) out[i] = lookup_one(q[i]);
    return out;
  }

 private:
  LO n_ = 0;
  bool dense_ = false;
  GO lo_ = 0;
  std::vector<LO> table_;
  std::vector<LO> sorted_;
  std::vector<LO> order_;
};

// -------------------------------------------------------------------------------------------
// VTK cell construction from SOMA's per-element topology, corner counts and connectivity
// (cells.py vtk_cells). `conn` and `nodes` must already be point-local indices (the caller maps
// gids through a GidIndex first, as mesh.py does before calling this).
// -------------------------------------------------------------------------------------------
struct VtkCells {
  std::vector<std::uint8_t> types;   // VTK cell type code per element
  std::vector<LO> offsets;           // size n + 1, into connectivity
  std::vector<LO> connectivity;      // point-local indices, offsets[c]..offsets[c+1) per cell
};

namespace detail {

inline int linear_vtk_type(Topology t) {
  switch (t) {
    case Topology::vertex: return vtk_type::VERTEX;
    case Topology::edge: return vtk_type::LINE;
    case Topology::tri: return vtk_type::TRIANGLE;
    case Topology::quad: return vtk_type::QUAD;
    case Topology::polygon: return vtk_type::POLYGON;
    case Topology::tet: return vtk_type::TETRA;
    case Topology::pyramid: return vtk_type::PYRAMID;
    case Topology::prism: return vtk_type::WEDGE;
    case Topology::knife: return vtk_type::POLYHEDRON;
    case Topology::hex: return vtk_type::HEXAHEDRON;
    case Topology::polyhedron: return vtk_type::CONVEX_POINT_SET;
  }
  return vtk_type::VERTEX;
}

// Corner permutation for topologies whose corner order differs from VTK's (cells.py
// corner_orders). VTK 9.7 changed the wedge's point ordering to match its parametric
// coordinates: a SOMA prism (base triangle (0, 1, 2) counterclockwise seen from the top one) is
// a positive wedge in VTK >= 9.7 as it is, and needs its two triangles reversed for VTK <= 9.6
// (`legacy_wedge`).
inline const std::vector<int>* corner_perm(Topology t, bool legacy_wedge) {
  static const std::vector<int> prism_perm{0, 2, 1, 3, 5, 4};
  if (t == Topology::prism && legacy_wedge) return &prism_perm;
  return nullptr;
}

struct HighOrder { int vtk_type; std::vector<int> perm; };

inline std::vector<int> ho_perm(int corners, std::vector<int> edges, std::vector<int> faces = {},
                                bool region = false) {
  std::vector<int> out;
  for (int i = 0; i < corners; ++i) out.push_back(i);
  int ne = static_cast<int>(edges.size());
  for (int e : edges) out.push_back(corners + e);
  for (int f : faces) out.push_back(corners + ne + f);
  if (region) out.push_back(corners + ne + static_cast<int>(faces.size()));
  return out;
}

// (topology, node count) -> (VTK type, SOMA position of each VTK node) (cells.py HIGH_ORDER).
inline const std::map<std::pair<Topology, int>, HighOrder>& high_order_table(bool legacy_wedge) {
  static const auto make = [](bool legacy) {
    std::map<std::pair<Topology, int>, HighOrder> m;
    m[{Topology::edge, 3}] = {vtk_type::QUADRATIC_EDGE, ho_perm(2, {0})};
    m[{Topology::tri, 6}] = {vtk_type::QUADRATIC_TRIANGLE, ho_perm(3, {0, 1, 2})};
    m[{Topology::tri, 7}] = {vtk_type::BIQUADRATIC_TRIANGLE, ho_perm(3, {0, 1, 2}, {0})};
    m[{Topology::quad, 8}] = {vtk_type::QUADRATIC_QUAD, ho_perm(4, {0, 1, 2, 3})};
    m[{Topology::quad, 9}] = {vtk_type::BIQUADRATIC_QUAD, ho_perm(4, {0, 1, 2, 3}, {0})};
    m[{Topology::tet, 10}] = {vtk_type::QUADRATIC_TETRA, ho_perm(4, {0, 1, 2, 3, 4, 5})};
    m[{Topology::pyramid, 13}] = {vtk_type::QUADRATIC_PYRAMID, ho_perm(5, {0, 1, 2, 3, 4, 5, 6, 7})};
    if (legacy) {
      // VTK <= 9.6: corners reversed as for the linear wedge; MOAB prism edges (0-1,1-2,2-0,
      // 0-3,1-4,2-5,3-4,4-5,5-3) in VTK's order: base, top, vertical, of the reversed corners.
      std::vector<int> perm{0, 2, 1, 3, 5, 4};
      for (int e : {2, 1, 0, 8, 7, 6, 3, 5, 4}) perm.push_back(6 + e);
      m[{Topology::prism, 15}] = {vtk_type::QUADRATIC_WEDGE, perm};
    } else {
      // VTK >= 9.7: SOMA's corners; edges base, top, vertical.
      m[{Topology::prism, 15}] = {vtk_type::QUADRATIC_WEDGE, ho_perm(6, {0, 1, 2, 6, 7, 8, 3, 4, 5})};
    }
    // MOAB hex edges: 4 bottom, 4 vertical, 4 top; VTK: bottom, top, vertical.
    m[{Topology::hex, 20}] = {vtk_type::QUADRATIC_HEXAHEDRON, ho_perm(8, {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7})};
    // MOAB hex faces: y-, x+, y+, x-, z-, z+; VTK: x-, x+, y-, y+, z-, z+.
    m[{Topology::hex, 27}] = {vtk_type::TRIQUADRATIC_HEXAHEDRON,
                             ho_perm(8, {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7}, {3, 1, 0, 2, 4, 5}, true)};
    return m;
  };
  static const std::map<std::pair<Topology, int>, HighOrder> current = make(false), legacy = make(true);
  return legacy_wedge ? legacy : current;
}

}  // namespace detail

// `topology[e]`, `counts[e]` (corner count) and `conn` (flattened, point-local ids, `counts[e]`
// per element) describe the elements of one dimension; `node_counts`/`nodes` give each
// element's further (edge/face/region) node ids when the file has high-order nodes and
// `high_order` is set. Elements VTK has no quadratic cell for fall back to their linear corners.
inline VtkCells vtk_cells(const std::vector<std::uint8_t>& topology, const std::vector<LO>& counts,
                         const std::vector<LO>& conn, const std::vector<LO>* node_counts,
                         const std::vector<LO>* nodes, bool high_order, bool legacy_wedge = false) {
  VtkCells out;
  std::size_t n = topology.size();
  out.types.resize(n);
  out.offsets.resize(n + 1, 0);
  std::vector<LO> cstart(n + 1, 0), nstart(n + 1, 0);
  if (counts.size() != n) throw io_error("soma_reader: " + std::to_string(n) + " elements with " +
                                         std::to_string(counts.size()) + " corner counts");
  for (std::size_t e = 0; e < n; ++e) {
    if (counts[e] < 0) throw io_error("soma_reader: an element with " + std::to_string(counts[e]) + " corners");
    cstart[e + 1] = cstart[e] + counts[e];
  }
  if (static_cast<std::size_t>(cstart[n]) != conn.size())
    throw io_error("soma_reader: corner counts sum to " + std::to_string(cstart[n]) + ", the connectivity has " +
                   std::to_string(conn.size()) + " entries");
  bool use_nodes = high_order && nodes != nullptr && node_counts != nullptr && !nodes->empty();
  if (use_nodes) {
    if (node_counts->size() != n) throw io_error("soma_reader: " + std::to_string(n) + " elements with " +
                                                 std::to_string(node_counts->size()) + " node counts");
    for (std::size_t e = 0; e < n; ++e) {
      if ((*node_counts)[e] < 0) throw io_error("soma_reader: an element with a negative node count");
      nstart[e + 1] = nstart[e] + (*node_counts)[e];
    }
    if (static_cast<std::size_t>(nstart[n]) != nodes->size())
      throw io_error("soma_reader: node counts sum to " + std::to_string(nstart[n]) + ", the nodes have " +
                     std::to_string(nodes->size()) + " entries");
  }
  // An element's node ids, permuted; throws if the permutation needs more nodes than it has.
  auto take = [](const std::vector<LO>& from, int p) {
    if (p < 0 || static_cast<std::size_t>(p) >= from.size())
      throw io_error("soma_reader: an element of " + std::to_string(from.size()) + " nodes needs node " +
                     std::to_string(p));
    return from[static_cast<std::size_t>(p)];
  };

  std::vector<LO> full;  // scratch: this element's SOMA-ordered node ids (corners then extra)
  for (std::size_t e = 0; e < n; ++e) {
    Topology t = static_cast<Topology>(topology[e]);
    LO c = counts[e];
    LO extra = use_nodes ? (*node_counts)[e] : 0;
    full.assign(conn.begin() + cstart[e], conn.begin() + cstart[e + 1]);
    if (extra > 0) full.insert(full.end(), nodes->begin() + nstart[e], nodes->begin() + nstart[e + 1]);
    const detail::HighOrder* ho = nullptr;
    if (extra > 0) {
      auto it = detail::high_order_table(legacy_wedge).find({t, static_cast<int>(c + extra)});
      if (it != detail::high_order_table(legacy_wedge).end()) ho = &it->second;
    }
    if (ho) {
      out.types[e] = static_cast<std::uint8_t>(ho->vtk_type);
      for (int p : ho->perm) out.connectivity.push_back(take(full, p));
      out.offsets[e + 1] = out.offsets[e] + static_cast<LO>(ho->perm.size());
    } else {
      out.types[e] = static_cast<std::uint8_t>(detail::linear_vtk_type(t));
      const std::vector<int>* perm = detail::corner_perm(t, legacy_wedge);
      if (perm) {
        for (int p : *perm) out.connectivity.push_back(take(full, p));
        out.offsets[e + 1] = out.offsets[e] + static_cast<LO>(perm->size());
      } else {
        out.connectivity.insert(out.connectivity.end(), full.begin(), full.begin() + c);
        out.offsets[e + 1] = out.offsets[e] + c;
      }
    }
  }
  return out;
}

// The VTK polyhedron face stream of a knife with (point-local) corner ids `ids` (7 of them, in
// SOMA's corner order): a flat [nfaces, (npts, id...) per face] list, as vtkCellArray /
// VTK_POLYHEDRON legacy streams expect (vtkconv.py's knife_face_stream).
inline std::vector<LO> knife_face_stream(const std::vector<LO>& ids) {
  std::vector<LO> stream;
  stream.push_back(static_cast<LO>(knife_faces().size()));
  for (auto& f : knife_faces()) {
    stream.push_back(static_cast<LO>(f.size()));
    for (int k : f) stream.push_back(ids[static_cast<std::size_t>(k)]);
  }
  return stream;
}

// -------------------------------------------------------------------------------------------
// A field's values as a dense row-major array (NaN where an entity is off a partial domain, as
// mesh.py's _float_fill promotes to): the common currency this header hands field data back in.
// -------------------------------------------------------------------------------------------
struct Array {
  int ncomp = 1;
  std::vector<double> data;  // size n * ncomp
};

struct CellBlock {
  int dim = 0;
  std::vector<GO> gids;
  std::vector<int> part;
  std::vector<std::uint8_t> types;       // VTK cell type code per cell
  std::vector<LO> offsets;               // size n + 1
  std::vector<LO> connectivity;          // point-local indices
  std::map<std::string, Array> cell_data;
  LO size() const { return static_cast<LO>(types.size()); }
};

struct MeshPiece {
  std::string path;
  int dim = 0;
  std::vector<int> parts;
  std::vector<double> points;   // n * 3
  std::vector<GO> point_gids;
  std::vector<int> point_part;
  std::map<std::string, Array> point_data;
  std::map<int, CellBlock> blocks;
  LO halo_points = 0;
  std::vector<std::string> skipped;
  std::optional<double> time;
};

struct ReadOptions {
  std::optional<std::set<std::string>> point_fields;  // nullopt: every drawable point field
  std::optional<std::set<std::string>> cell_fields;    // nullopt: every drawable cell field
  bool high_order = true;
  // Prism corners for VTK <= 9.6 (true) or VTK >= 9.7 (false, default); see corner_perm.
  bool legacy_wedge = false;
  // 0 = no set fields, 1 (default) = every set-valued field, 2 = only `set_field_names`.
  int set_fields_mode = 1;
  std::set<std::string> set_field_names;
};

// Per-(field, dim) cache of the "full" bit array (one bit per part: does the part's domain cover
// every owned entity of this dim), read once no matter how many times a part's data is needed.
struct FieldCache {
  std::map<std::pair<int, int>, std::vector<bool>> full_bits;
};

namespace detail {

inline const std::vector<bool>& full_flags(const SomaFile& sf, const FieldInfo& info, int dim,
                                           FieldCache& cache) {
  auto key = std::make_pair(info.index, dim);
  auto it = cache.full_bits.find(key);
  if (it != cache.full_bits.end()) return it->second;
  auto raw = sf.read<std::uint8_t>(info.group() + "/dim" + std::to_string(dim) + "/full");
  std::vector<bool> bits(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) bits[i] = raw[i] != 0;
  return cache.full_bits.emplace(key, std::move(bits)).first->second;
}

// 0, 1, ... within each part, for parts of the given sizes laid end to end (mesh.py _local_rows).
inline std::vector<LO> local_rows(const std::vector<LO>& per_part) {
  LO total = 0;
  for (LO c : per_part) total += c;
  std::vector<LO> out;
  out.reserve(static_cast<std::size_t>(total));
  for (LO c : per_part) for (LO i = 0; i < c; ++i) out.push_back(i);
  return out;
}

}  // namespace detail

// Values of `info` on the entities (owner part, local row) given by parallel arrays `owner` and
// `local`, of dimension `dim`; NaN where a row is off a partial domain (mesh.py's _values_at).
inline Array values_at(const SomaFile& sf, const FieldInfo& info, int dim, const std::vector<int>& owner,
                       const std::vector<LO>& local, FieldCache& cache) {
  std::string base = info.group() + "/dim" + std::to_string(dim);
  LO n = static_cast<LO>(owner.size());
  Array out;
  out.ncomp = info.ncomp;
  out.data.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(std::max(info.ncomp, 1)),
                 std::numeric_limits<double>::quiet_NaN());
  if (n == 0) return out;
  std::string values_path = base + "/values";
  auto vstart = sf.part_starts(values_path);
  std::vector<LO> rows(static_cast<std::size_t>(n));
  std::vector<char> present(static_cast<std::size_t>(n), 1);
  if (info.full_on(dim)) {
    for (LO i = 0; i < n; ++i)
      rows[static_cast<std::size_t>(i)] =
          vstart[static_cast<std::size_t>(owner[static_cast<std::size_t>(i)])] + local[static_cast<std::size_t>(i)];
  } else {
    const auto& full = detail::full_flags(sf, info, dim, cache);
    std::map<int, std::vector<LO>> groups;
    for (LO i = 0; i < n; ++i) groups[owner[static_cast<std::size_t>(i)]].push_back(i);
    for (auto& [q, idxs] : groups) {
      bool q_full = static_cast<std::size_t>(q) < full.size() && full[static_cast<std::size_t>(q)];
      if (q_full) {
        for (LO i : idxs)
          rows[static_cast<std::size_t>(i)] = vstart[static_cast<std::size_t>(q)] + local[static_cast<std::size_t>(i)];
        continue;
      }
      std::vector<int> qp{q};
      auto dom = sf.read<std::int64_t>(base + "/domain", &qp);
      for (LO i : idxs) {
        LO loc = local[static_cast<std::size_t>(i)];
        auto it = std::lower_bound(dom.begin(), dom.end(), loc);
        if (it != dom.end() && *it == loc) {
          LO pos = static_cast<LO>(it - dom.begin());
          rows[static_cast<std::size_t>(i)] = vstart[static_cast<std::size_t>(q)] + pos;
        } else {
          present[static_cast<std::size_t>(i)] = 0;
          rows[static_cast<std::size_t>(i)] = vstart[static_cast<std::size_t>(q)];
        }
      }
    }
  }
  std::vector<LO> present_rows;
  std::vector<LO> present_idx;
  for (LO i = 0; i < n; ++i)
    if (present[static_cast<std::size_t>(i)]) {
      present_rows.push_back(rows[static_cast<std::size_t>(i)]);
      present_idx.push_back(i);
    }
  if (!present_rows.empty()) {
    hsize_t ncomp;
    auto got = sf.read_rows_at<double>(values_path, present_rows, &ncomp);
    for (std::size_t k = 0; k < present_idx.size(); ++k)
      for (int c = 0; c < info.ncomp; ++c)
        out.data[static_cast<std::size_t>(present_idx[k]) * static_cast<std::size_t>(info.ncomp) +
                static_cast<std::size_t>(c)] = got[k * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(c)];
  }
  return out;
}

// Values of `info` on the owned entities of `parts`: a direct read when the field is full on
// every one of them (one hyperslab per run of parts), else `values_at` (mesh.py _values_owned).
inline Array values_owned(const SomaFile& sf, const FieldInfo& info, int dim, const std::vector<int>& parts,
                          const std::vector<int>& owner, const std::vector<LO>& local, FieldCache& cache) {
  bool all_full = true;
  for (int p : parts) {
    const auto& full = detail::full_flags(sf, info, dim, cache);
    if (!(static_cast<std::size_t>(p) < full.size() && full[static_cast<std::size_t>(p)])) { all_full = false; break; }
  }
  if (all_full) {
    hsize_t ncomp;
    auto v = sf.read<double>(info.group() + "/dim" + std::to_string(dim) + "/values", &parts, &ncomp);
    Array a;
    a.ncomp = static_cast<int>(ncomp);
    a.data = std::move(v);
    return a;
  }
  return values_at(sf, info, dim, owner, local, cache);
}

// -------------------------------------------------------------------------------------------
// Halo vertices: the vertices elements of the requested parts use but do not own, found by
// scanning the other parts' sharing tables (dim0/shared/index, counts, parts), with a fallback
// that scans the other parts' gids directly for anything the sharing table did not resolve
// (mesh.py's _halo_by_scan / _scan_gids).
// -------------------------------------------------------------------------------------------
struct HaloResult {
  std::vector<GO> gid;
  std::vector<double> xyz;  // n * 3
  std::vector<int> part;
  std::map<std::string, Array> data;
};

inline HaloResult halo_by_scan(const SomaFile& sf, const std::vector<int>& mine_parts,
                               const std::vector<GO>& missing_sorted,
                               const std::vector<const FieldInfo*>& point_fields, FieldCache& cache,
                               LO chunk_rows = (1LL << 22)) {
  HaloResult res;
  if (missing_sorted.empty()) return res;
  std::set<int> mine_set(mine_parts.begin(), mine_parts.end());
  std::vector<int> others;
  for (int p = 0; p < sf.nparts(); ++p)
    if (!mine_set.count(p)) others.push_back(p);

  auto idx_rows = sf.part_starts("dim0/shared/index");
  GidIndex missing_index(missing_sorted);

  std::vector<int> found_owner;
  std::vector<LO> found_local;

  std::size_t i0 = 0;
  while (i0 < others.size()) {
    std::vector<int> group;
    LO rows = 0;
    while (i0 < others.size()) {
      LO n = idx_rows[static_cast<std::size_t>(others[i0]) + 1] - idx_rows[static_cast<std::size_t>(others[i0])];
      if (!group.empty() && rows + n > chunk_rows) break;
      group.push_back(others[i0]);
      rows += n;
      ++i0;
    }
    if (group.empty()) break;

    auto index = sf.read<std::int64_t>("dim0/shared/index", &group);
    if (index.empty()) continue;
    auto counts = sf.read<std::int64_t>("dim0/shared/counts", &group);
    auto dest = sf.read<std::int64_t>("dim0/shared/parts", &group);
    std::vector<LO> row_of;
    row_of.reserve(dest.size());
    for (std::size_t k = 0; k < counts.size(); ++k)
      for (LO c = 0; c < counts[k]; ++c) row_of.push_back(static_cast<LO>(k));
    std::vector<char> hit(index.size(), 0);
    for (std::size_t k = 0; k < dest.size(); ++k)
      if (mine_set.count(static_cast<int>(dest[k]))) hit[static_cast<std::size_t>(row_of[k])] = 1;
    std::vector<int> owner_of_row;
    owner_of_row.reserve(index.size());
    for (int p : group) {
      LO c = idx_rows[static_cast<std::size_t>(p) + 1] - idx_rows[static_cast<std::size_t>(p)];
      for (LO k = 0; k < c; ++k) owner_of_row.push_back(p);
    }
    for (std::size_t r = 0; r < index.size(); ++r) {
      if (!hit[r]) continue;
      found_owner.push_back(owner_of_row[r]);
      found_local.push_back(index[r]);
    }
  }

  auto gstart = sf.part_starts("dim0/gid");
  std::vector<LO> rows(found_owner.size());
  for (std::size_t k = 0; k < rows.size(); ++k)
    rows[k] = gstart[static_cast<std::size_t>(found_owner[k])] + found_local[k];
  std::vector<GO> gid_candidates;
  if (!rows.empty()) gid_candidates = sf.read_rows_at<std::int64_t>("dim0/gid", rows, nullptr);

  std::vector<int> owner_final;
  std::vector<LO> local_final;
  std::vector<GO> gid_final;
  std::set<GO> seen;
  for (std::size_t k = 0; k < gid_candidates.size(); ++k) {
    GO g = gid_candidates[k];
    if (missing_index.lookup_one(g) < 0 || seen.count(g)) continue;
    seen.insert(g);
    owner_final.push_back(found_owner[k]);
    local_final.push_back(found_local[k]);
    gid_final.push_back(g);
  }

  if (gid_final.size() != missing_sorted.size()) {
    std::vector<GO> rest;
    for (GO g : missing_sorted)
      if (!seen.count(g)) rest.push_back(g);
    GidIndex rest_index(rest);
    std::vector<int> ro(rest.size(), -1);
    std::vector<LO> rl(rest.size(), -1);
    for (int p : others) {
      std::vector<int> pp{p};
      auto g = sf.read<std::int64_t>("dim0/gid", &pp);
      for (LO loc = 0; loc < static_cast<LO>(g.size()); ++loc) {
        LO pos = rest_index.lookup_one(g[static_cast<std::size_t>(loc)]);
        if (pos >= 0) {
          ro[static_cast<std::size_t>(pos)] = p;
          rl[static_cast<std::size_t>(pos)] = loc;
        }
      }
    }
    for (std::size_t k = 0; k < rest.size(); ++k) {
      if (ro[k] < 0)
        throw io_error(sf.path() + ": a vertex used by the requested parts is in no part");
      owner_final.push_back(ro[k]);
      local_final.push_back(rl[k]);
      gid_final.push_back(rest[k]);
    }
  }

  res.gid = gid_final;
  res.part.assign(owner_final.begin(), owner_final.end());
  std::vector<LO> final_rows(owner_final.size());
  for (std::size_t k = 0; k < final_rows.size(); ++k)
    final_rows[k] = gstart[static_cast<std::size_t>(owner_final[k])] + local_final[k];
  if (!final_rows.empty()) res.xyz = sf.read_rows_at<double>("dim0/coords", final_rows, nullptr);
  for (auto* f : point_fields) res.data[f->name] = values_at(sf, *f, 0, owner_final, local_final, cache);
  return res;
}

// -------------------------------------------------------------------------------------------
// One requested dimension's elements, as read from the file before gids are mapped to the
// piece's local point indices (mesh.py PieceReader.raw_cells).
// -------------------------------------------------------------------------------------------
struct RawCells {
  std::vector<GO> gids;
  std::vector<std::uint8_t> topology;
  std::vector<LO> counts;
  std::vector<GO> conn;
  std::vector<int> part;
  std::vector<LO> local;
  std::vector<LO> node_counts;
  std::vector<GO> nodes;
  bool has_nodes = false;
};

// -------------------------------------------------------------------------------------------
// Entity sets as cell labels (mesh.py _sets_info / _set_labels / _set_cell_data): sets contain
// mesh entities (sets/members: set, dim, local row) and, through sets/set_members (unordered,
// by gid) and sets/entries (ordered, (set, position, gid)), other sets. A set-valued field's
// value on a domain set is inherited by every set reachable from it through that containment.
// -------------------------------------------------------------------------------------------
struct SetsInfo {
  LO nsets = 0;
  std::vector<LO> contains_starts;    // CSR: set -> contained (child) set indices
  std::vector<LO> contains_members;
  std::vector<LO> members_set;        // sets/members columns, rows of the requested parts only
  std::vector<int> members_dim;
  std::vector<LO> members_local;
  std::vector<int> members_part;
  std::map<std::string, std::vector<double>> labels;  // field name -> per-set value, NaN if none
};

inline SetsInfo build_sets_info(const SomaFile& sf, const std::vector<int>& parts,
                                const std::vector<const FieldInfo*>& set_fields) {
  SetsInfo info;
  if (!sf.has("sets/gid")) return info;
  auto set_gid = sf.read<std::int64_t>("sets/gid");
  info.nsets = static_cast<LO>(set_gid.size());
  if (info.nsets == 0) return info;
  GidIndex sidx(set_gid);

  std::vector<LO> owner;
  std::vector<LO> child;
  if (sf.has("sets/set_members_counts")) {
    auto mc = sf.read<std::int64_t>("sets/set_members_counts");
    auto set_members = sf.has("sets/set_members") ? sf.read<std::int64_t>("sets/set_members")
                                                   : std::vector<GO>{};
    for (LO s = 0; s < info.nsets; ++s)
      for (LO k = 0; k < mc[static_cast<std::size_t>(s)]; ++k) owner.push_back(s);
    child = sidx.lookup(set_members);
  }
  if (sf.has("sets/entries")) {
    hsize_t ncomp;
    auto entries = sf.read<std::int64_t>("sets/entries", nullptr, &ncomp);
    if (ncomp == 3) {
      LO ne = static_cast<LO>(entries.size() / 3);
      std::vector<GO> echild(static_cast<std::size_t>(ne));
      for (LO e = 0; e < ne; ++e) {
        owner.push_back(entries[static_cast<std::size_t>(e) * 3]);
        echild[static_cast<std::size_t>(e)] = entries[static_cast<std::size_t>(e) * 3 + 2];
      }
      auto echild_local = sidx.lookup(echild);
      child.insert(child.end(), echild_local.begin(), echild_local.end());
    }
  }
  std::vector<std::pair<LO, LO>> pairs;
  for (std::size_t i = 0; i < owner.size(); ++i)
    if (child[i] >= 0) pairs.emplace_back(owner[i], child[i]);
  info.contains_starts.assign(static_cast<std::size_t>(info.nsets) + 1, 0);
  for (auto& pr : pairs) info.contains_starts[static_cast<std::size_t>(pr.first) + 1]++;
  for (std::size_t i = 1; i < info.contains_starts.size(); ++i) info.contains_starts[i] += info.contains_starts[i - 1];
  info.contains_members.resize(pairs.size());
  {
    std::vector<LO> cursor(info.contains_starts.begin(), info.contains_starts.end() - 1);
    for (auto& pr : pairs)
      info.contains_members[static_cast<std::size_t>(cursor[static_cast<std::size_t>(pr.first)]++)] = pr.second;
  }

  if (sf.has("sets/members")) {
    auto members = sf.read<std::int64_t>("sets/members", &parts);
    LO nm = static_cast<LO>(members.size() / 4);
    auto mstarts = sf.part_starts("sets/members");
    for (int p : parts) {
      LO cnt = mstarts[static_cast<std::size_t>(p) + 1] - mstarts[static_cast<std::size_t>(p)];
      for (LO i = 0; i < cnt; ++i) info.members_part.push_back(p);
    }
    info.members_set.resize(static_cast<std::size_t>(nm));
    info.members_dim.resize(static_cast<std::size_t>(nm));
    info.members_local.resize(static_cast<std::size_t>(nm));
    for (LO i = 0; i < nm; ++i) {
      info.members_set[static_cast<std::size_t>(i)] = members[static_cast<std::size_t>(i) * 4];
      info.members_dim[static_cast<std::size_t>(i)] = static_cast<int>(members[static_cast<std::size_t>(i) * 4 + 1]);
      info.members_local[static_cast<std::size_t>(i)] = members[static_cast<std::size_t>(i) * 4 + 2];
    }
  }

  for (auto* f : set_fields) {
    std::vector<double> label(static_cast<std::size_t>(info.nsets), std::numeric_limits<double>::quiet_NaN());
    auto dom = sf.read<std::int64_t>(f->group() + "/sets/domain");
    auto vals = sf.read<double>(f->group() + "/sets/values");
    for (std::size_t k = 0; k < dom.size(); ++k) {
      LO s = dom[k];
      double v = vals[k];
      std::vector<char> seen(static_cast<std::size_t>(info.nsets), 0);
      std::vector<LO> stack{s};
      seen[static_cast<std::size_t>(s)] = 1;
      while (!stack.empty()) {
        LO t = stack.back();
        stack.pop_back();
        for (LO m = info.contains_starts[static_cast<std::size_t>(t)]; m < info.contains_starts[static_cast<std::size_t>(t) + 1]; ++m) {
          LO c = info.contains_members[static_cast<std::size_t>(m)];
          if (!seen[static_cast<std::size_t>(c)]) { seen[static_cast<std::size_t>(c)] = 1; stack.push_back(c); }
        }
      }
      for (LO i = 0; i < info.nsets; ++i)
        if (seen[static_cast<std::size_t>(i)]) label[static_cast<std::size_t>(i)] = v;
    }
    info.labels[f->name] = std::move(label);
  }
  return info;
}

inline void apply_set_labels(const SetsInfo& info, int dim, const RawCells& c,
                             const std::vector<int>& parts, const SomaFile& sf, CellBlock& block) {
  if (info.nsets == 0) return;
  auto counts_d = sf.counts_by_part(dim);
  std::map<int, LO> first;
  LO running = 0;
  for (int p : parts) {
    first[p] = running;
    running += counts_d[static_cast<std::size_t>(p)];
  }
  LO ncell = static_cast<LO>(c.gids.size());
  for (auto& [name, label] : info.labels) {
    std::vector<double> out(static_cast<std::size_t>(ncell), std::numeric_limits<double>::quiet_NaN());
    bool any = false;
    for (std::size_t i = 0; i < info.members_set.size(); ++i) {
      if (info.members_dim[i] != dim) continue;
      auto itp = first.find(info.members_part[i]);
      if (itp == first.end()) continue;
      LO cell = itp->second + info.members_local[i];
      double v = label[static_cast<std::size_t>(info.members_set[i])];
      if (!std::isnan(v)) { out[static_cast<std::size_t>(cell)] = v; any = true; }
    }
    if (any) {
      Array a;
      a.ncomp = 1;
      a.data = std::move(out);
      block.cell_data["set:" + name] = std::move(a);
    }
  }
}

// -------------------------------------------------------------------------------------------
// Field selection (mesh.py PieceReader._choose / _choose_sets).
// -------------------------------------------------------------------------------------------
inline const char* layout_name(LayoutKind k) {
  switch (k) {
    case LayoutKind::per_entity: return "per_entity";
    case LayoutKind::element_nodes: return "element_nodes";
    case LayoutKind::shared_nodes: return "shared_nodes";
    case LayoutKind::quadrature: return "quadrature";
  }
  return "?";
}

inline std::vector<const FieldInfo*> choose_fields(const SomaFile& sf, int dim,
                                                   const std::optional<std::set<std::string>>& names,
                                                   std::vector<std::string>& skipped) {
  std::vector<const FieldInfo*> out;
  for (auto& f : sf.fields()) {
    if (!f.has_dim(dim)) continue;
    if (names && !names->count(f.name)) continue;
    if (!f.drawable()) {
      std::string why = f.variable ? "variable-length" : f.handle ? "handle" : layout_name(f.layout);
      skipped.push_back(f.name + " (dim " + std::to_string(dim) + ": " + why + ")");
      continue;
    }
    out.push_back(&f);
  }
  return out;
}

inline std::vector<const FieldInfo*> choose_set_fields(const SomaFile& sf, const ReadOptions& opt) {
  std::vector<const FieldInfo*> out;
  if (opt.set_fields_mode == 0) return out;
  for (auto& f : sf.fields()) {
    if (!f.has_dim(SET_DIM) || f.handle || f.variable || f.ncomp != 1) continue;
    if (opt.set_fields_mode == 2 && !opt.set_field_names.count(f.name)) continue;
    out.push_back(&f);
  }
  return out;
}

// -------------------------------------------------------------------------------------------
// Reads `parts` of a SOMA mesh file's elements of `dims` as a MeshPiece (mesh.py read_piece,
// scanning for the halo rather than the collective MPI exchange -- see HaloCache below for the
// per-process-cached version a database plugin wants for repeated per-domain requests).
// -------------------------------------------------------------------------------------------
inline MeshPiece read_piece(const SomaFile& sf, std::vector<int> parts, std::vector<int> dims,
                           const ReadOptions& opt = ReadOptions{}) {
  if (sf.kind() != FileKind::mesh) throw io_error(sf.path() + ": an AMR checkpoint, not a mesh");
  if (parts.empty()) {
    parts.resize(static_cast<std::size_t>(sf.nparts()));
    for (int p = 0; p < sf.nparts(); ++p) parts[static_cast<std::size_t>(p)] = p;
  }
  std::sort(parts.begin(), parts.end());
  parts.erase(std::unique(parts.begin(), parts.end()), parts.end());
  if (parts.front() < 0 || parts.back() >= sf.nparts())
    throw io_error(sf.path() + ": parts must be in [0, " + std::to_string(sf.nparts()) + ")");
  for (int p : parts)
    if (p < 0 || p >= sf.nparts()) throw io_error(sf.path() + ": parts must be in [0, nparts)");
  if (dims.empty()) dims = {sf.dim()};
  std::sort(dims.begin(), dims.end());
  dims.erase(std::unique(dims.begin(), dims.end()), dims.end());

  FieldCache cache;
  std::vector<std::string> skipped;
  auto point_fields = choose_fields(sf, 0, opt.point_fields, skipped);
  std::map<int, std::vector<const FieldInfo*>> cell_fields;
  for (int d : dims) cell_fields[d] = choose_fields(sf, d, opt.cell_fields, skipped);
  auto set_fields = choose_set_fields(sf, opt);

  auto per0 = sf.counts_by_part(0);
  std::vector<LO> per0_sel;
  for (int p : parts) per0_sel.push_back(per0[static_cast<std::size_t>(p)]);
  std::vector<GO> owned_gids = sf.read<std::int64_t>("dim0/gid", &parts);
  std::vector<double> owned_xyz = sf.read<double>("dim0/coords", &parts);
  std::vector<int> owned_part;
  for (std::size_t k = 0; k < parts.size(); ++k)
    for (LO i = 0; i < per0_sel[k]; ++i) owned_part.push_back(parts[k]);
  std::vector<LO> owned_local = detail::local_rows(per0_sel);
  GidIndex owned_index(owned_gids);
  std::map<std::string, Array> owned_data;
  for (auto* f : point_fields) owned_data[f->name] = values_owned(sf, *f, 0, parts, owned_part, owned_local, cache);

  std::map<int, RawCells> raw;
  std::vector<GO> needed_all;
  for (int d : dims) {
    RawCells c;
    std::string g = "dim" + std::to_string(d);
    c.gids = sf.read<std::int64_t>(g + "/gid", &parts);
    c.topology = sf.read<std::uint8_t>(g + "/topology", &parts);
    c.counts = sf.read<std::int64_t>(g + "/counts", &parts);
    c.conn = sf.read<std::int64_t>(g + "/conn", &parts);
    auto perd = sf.counts_by_part(d);
    std::vector<LO> perd_sel;
    for (int p : parts) perd_sel.push_back(perd[static_cast<std::size_t>(p)]);
    for (std::size_t k = 0; k < parts.size(); ++k)
      for (LO i = 0; i < perd_sel[k]; ++i) c.part.push_back(parts[k]);
    c.local = detail::local_rows(perd_sel);
    if (opt.high_order && sf.has(g + "/nodes")) {
      c.node_counts = sf.read<std::int64_t>(g + "/node_counts", &parts);
      c.nodes = sf.read<std::int64_t>(g + "/nodes", &parts);
      c.has_nodes = true;
      needed_all.insert(needed_all.end(), c.nodes.begin(), c.nodes.end());
    }
    needed_all.insert(needed_all.end(), c.conn.begin(), c.conn.end());
    raw[d] = std::move(c);
  }
  std::sort(needed_all.begin(), needed_all.end());
  needed_all.erase(std::unique(needed_all.begin(), needed_all.end()), needed_all.end());
  std::vector<GO> missing;
  for (GO g : needed_all)
    if (owned_index.lookup_one(g) < 0) missing.push_back(g);
  std::sort(missing.begin(), missing.end());

  HaloResult halo = halo_by_scan(sf, parts, missing, point_fields, cache);

  MeshPiece piece;
  piece.path = sf.path();
  piece.dim = sf.dim();
  piece.parts = parts;
  piece.halo_points = static_cast<LO>(halo.gid.size());
  piece.skipped = std::move(skipped);
  piece.time = sf.time();

  piece.point_gids = owned_gids;
  piece.point_gids.insert(piece.point_gids.end(), halo.gid.begin(), halo.gid.end());
  piece.points = owned_xyz;
  piece.points.insert(piece.points.end(), halo.xyz.begin(), halo.xyz.end());
  piece.point_part = owned_part;
  piece.point_part.insert(piece.point_part.end(), halo.part.begin(), halo.part.end());
  for (auto* f : point_fields) {
    Array a = owned_data[f->name];
    auto hit = halo.data.find(f->name);
    if (hit != halo.data.end()) a.data.insert(a.data.end(), hit->second.data.begin(), hit->second.data.end());
    piece.point_data[f->name] = std::move(a);
  }

  GidIndex full_index = halo.gid.empty() ? std::move(owned_index) : GidIndex(piece.point_gids);

  bool have_sets = !set_fields.empty();
  SetsInfo sets_info = have_sets ? build_sets_info(sf, parts, set_fields) : SetsInfo{};

  for (int d : dims) {
    auto& c = raw[d];
    auto conn_local = full_index.lookup(c.conn);
    for (auto v : conn_local)
      if (v < 0)
        throw io_error(sf.path() + ": elements of dimension " + std::to_string(d) +
                       " use vertices no part shares with the requested parts");
    std::vector<LO> nodes_local;
    std::vector<LO>* nodes_local_ptr = nullptr;
    if (c.has_nodes) {
      nodes_local = full_index.lookup(c.nodes);
      for (auto v : nodes_local)
        if (v < 0)
          throw io_error(sf.path() + ": elements of dimension " + std::to_string(d) +
                         " use vertices no part shares with the requested parts");
      nodes_local_ptr = &nodes_local;
    }
    VtkCells vc = vtk_cells(c.topology, c.counts, conn_local, c.has_nodes ? &c.node_counts : nullptr,
                           nodes_local_ptr, opt.high_order, opt.legacy_wedge);
    CellBlock block;
    block.dim = d;
    block.gids = c.gids;
    block.part = c.part;
    block.types = std::move(vc.types);
    block.offsets = std::move(vc.offsets);
    block.connectivity = std::move(vc.connectivity);
    for (auto* f : cell_fields[d]) block.cell_data[f->name] = values_owned(sf, *f, d, parts, c.part, c.local, cache);
    if (have_sets) apply_set_labels(sets_info, d, c, parts, sf, block);
    piece.blocks[d] = std::move(block);
  }
  return piece;
}

// Convenience: the parts process `rank` of `nranks` reads under SOMA's own assignment, part q
// going to floor(q * nranks / nparts) (mesh.py parts_for_rank).
inline std::vector<int> parts_for_rank(int nparts, int nranks, int rank) {
  std::vector<int> out;
  for (int q = 0; q < nparts; ++q)
    if ((static_cast<long long>(q) * nranks) / nparts == rank) out.push_back(q);
  return out;
}

// -------------------------------------------------------------------------------------------
// AMR checkpoints (amr.py AmrFile): the hierarchy's boxes and the fields' valid points, level by
// level. Under /soma/amr a checkpoint holds every level's boxes (amr/levels/lL: rows lo, hi of
// cell indices, inclusive) and per field (amr/fields/fJ) its value type, index type (nodal
// bits), components and ghost width, with the valid points of every box of every level in
// dataset lL: box after box, each with i fastest, then j, k, then the component.
// -------------------------------------------------------------------------------------------
struct AmrField {
  int index = 0;
  std::string name;
  DType dtype = DType::f64;
  int nodal_bits = 0;
  int ncomp = 1;
  int ghosts = 0;

  bool is_nodal(int d) const { return ((nodal_bits >> d) & 1) != 0; }

  // "cell", "node", or "mixed" (face- or edge-centred: nodal in some directions only).
  std::string centering(int dim) const {
    int nodal_count = 0;
    for (int d = 0; d < dim; ++d) if (is_nodal(d)) ++nodal_count;
    if (nodal_count == 0) return "cell";
    if (nodal_count == dim) return "node";
    return "mixed";
  }
};

// A cell box, inclusive: lo[3], hi[3] cell indices.
struct AmrBox { std::array<LO, 3> lo, hi; };

class AmrFile {
 public:
  explicit AmrFile(const SomaFile& sf) : sf_(sf) {
    if (sf.kind() != FileKind::amr) throw io_error(sf.path() + ": a SOMA mesh file, not an AMR checkpoint");
    auto w = h5detail::read_int_attr(sf.raw(), "", "amr");
    detail::need_words(w, 11, sf.path() + ": /soma attribute amr");
    format_version_ = static_cast<int>(w[0]);
    dim_ = static_cast<int>(w[1]);
    nparts_ = static_cast<int>(w[2]);
    nlevels_ = static_cast<int>(w[3]);
    int nfields = static_cast<int>(w[4]);
    ratio_ = static_cast<int>(w[5]);
    blocking_factor_ = static_cast<int>(w[6]);
    max_box_ = static_cast<int>(w[7]);
    max_level_ = static_cast<int>(w[8]);
    nesting_ = static_cast<int>(w[9]);
    curve_ = static_cast<int>(w[10]);
    int periodic_mask = w.size() > 11 ? static_cast<int>(w[11]) : 0;
    for (int d = 0; d < 3; ++d) periodic_[static_cast<std::size_t>(d)] = ((periodic_mask >> d) & 1) != 0;

    auto dom = h5detail::read_int_attr(sf.raw(), "amr", "domain");
    for (int d = 0; d < 3; ++d) { domain_lo_[static_cast<std::size_t>(d)] = dom[static_cast<std::size_t>(d)];
                                 domain_hi_[static_cast<std::size_t>(d)] = dom[static_cast<std::size_t>(d) + 3]; }
    auto geo = sf.read<double>("amr/geometry");
    for (int d = 0; d < 3; ++d) { lo_[static_cast<std::size_t>(d)] = geo[static_cast<std::size_t>(d)];
                                 hi_[static_cast<std::size_t>(d)] = geo[static_cast<std::size_t>(d) + 3]; }
    efficiency_ = geo.size() > 6 ? geo[6] : 0.0;

    boxes_.resize(static_cast<std::size_t>(nlevels_));
    for (int l = 0; l < nlevels_; ++l) {
      auto raw = sf.read<std::int64_t>("amr/levels/l" + std::to_string(l));
      LO nb = static_cast<LO>(raw.size() / 6);
      auto& v = boxes_[static_cast<std::size_t>(l)];
      v.resize(static_cast<std::size_t>(nb));
      for (LO b = 0; b < nb; ++b) {
        for (int d = 0; d < 3; ++d) {
          v[static_cast<std::size_t>(b)].lo[static_cast<std::size_t>(d)] = raw[static_cast<std::size_t>(b) * 6 + static_cast<std::size_t>(d)];
          v[static_cast<std::size_t>(b)].hi[static_cast<std::size_t>(d)] = raw[static_cast<std::size_t>(b) * 6 + static_cast<std::size_t>(d) + 3];
        }
      }
    }
    for (int j = 0; j < nfields; ++j) {
      std::string g = "amr/fields/f" + std::to_string(j);
      auto m = h5detail::read_int_attr(sf.raw(), g, "meta");
      AmrField f;
      f.index = j;
      f.name = h5detail::read_string_attr(sf.raw(), g, "name");
      f.dtype = static_cast<DType>(m[0]);
      f.nodal_bits = static_cast<int>(m[1]);
      f.ncomp = static_cast<int>(m[2]);
      f.ghosts = m.size() > 3 ? static_cast<int>(m[3]) : 0;
      fields_.push_back(std::move(f));
    }
  }

  int dim() const { return dim_; }
  int nparts() const { return nparts_; }
  int nlevels() const { return nlevels_; }
  int ratio() const { return ratio_; }
  int blocking_factor() const { return blocking_factor_; }
  int max_box() const { return max_box_; }
  int max_level() const { return max_level_; }
  const std::array<double, 3>& lo() const { return lo_; }
  const std::array<double, 3>& hi() const { return hi_; }
  const std::array<LO, 3>& domain_lo() const { return domain_lo_; }
  const std::array<LO, 3>& domain_hi() const { return domain_hi_; }
  const std::array<bool, 3>& periodic() const { return periodic_; }
  const std::vector<std::vector<AmrBox>>& boxes() const { return boxes_; }
  const std::vector<AmrField>& fields() const { return fields_; }

  const AmrField& field(const std::string& name) const {
    for (auto& f : fields_) if (f.name == name) return f;
    throw io_error(sf_.path() + ": no AMR field '" + name + "'");
  }

  // Cell size at `level` (third component 1 in 2D).
  std::array<double, 3> spacing(int level) const {
    std::array<double, 3> dx{};
    for (int d = 0; d < 3; ++d) {
      double n = static_cast<double>(domain_hi_[static_cast<std::size_t>(d)] - domain_lo_[static_cast<std::size_t>(d)] + 1);
      dx[static_cast<std::size_t>(d)] = (hi_[static_cast<std::size_t>(d)] - lo_[static_cast<std::size_t>(d)]) / n /
                                       std::pow(static_cast<double>(ratio_), level);
    }
    if (dim_ == 2) dx[2] = 1.0;
    return dx;
  }

  // (ni, nj, nk) valid points of field `f` on a cell box.
  std::array<LO, 3> point_shape(const AmrField& f, const AmrBox& box) const {
    std::array<LO, 3> n{};
    for (int d = 0; d < 3; ++d) {
      LO base = box.hi[static_cast<std::size_t>(d)] - box.lo[static_cast<std::size_t>(d)] + 1;
      n[static_cast<std::size_t>(d)] = base + ((d < dim_ && f.is_nodal(d)) ? 1 : 0);
    }
    if (dim_ == 2) n[2] = 1;
    return n;
  }

  // Valid points of field `name` on `boxes` of `level` (default: all), each a flattened
  // (ncomp, nk, nj, ni) array, i fastest, as stored.
  std::vector<std::vector<double>> read(const std::string& name, int level,
                                        const std::vector<LO>* which = nullptr) const {
    const AmrField& f = field(name);
    std::string path = "amr/fields/f" + std::to_string(f.index) + "/l" + std::to_string(level);
    auto starts = box_starts(f, level);
    std::vector<LO> idx;
    if (which) idx = *which;
    else { idx.resize(boxes_[static_cast<std::size_t>(level)].size()); std::iota(idx.begin(), idx.end(), 0); }
    std::vector<std::vector<double>> out;
    out.reserve(idx.size());
    for (LO b : idx) {
      LO lo = starts[static_cast<std::size_t>(b)], hi = starts[static_cast<std::size_t>(b) + 1];
      out.push_back(h5detail::read_rows<double>(sf_.raw(), path, static_cast<hsize_t>(lo), static_cast<hsize_t>(hi - lo)));
    }
    return out;
  }

  // Per level, the process of each box when `nranks` processes read: contiguous runs in box
  // order, balanced by cell count (amr.py box_owner).
  std::vector<std::vector<int>> box_owner(int nranks) const {
    std::vector<std::vector<int>> out;
    for (auto& level : boxes_) {
      std::vector<double> cells(level.size());
      double total = 0;
      for (std::size_t b = 0; b < level.size(); ++b) {
        double c = 1;
        for (int d = 0; d < 3; ++d) c *= static_cast<double>(level[b].hi[static_cast<std::size_t>(d)] - level[b].lo[static_cast<std::size_t>(d)] + 1);
        cells[b] = c;
        total += c;
      }
      std::vector<int> owner(level.size());
      double running = 0;
      for (std::size_t b = 0; b < level.size(); ++b) {
        double mid = running + cells[b] / 2;
        running += cells[b];
        int r = static_cast<int>((mid / std::max(total, 1.0)) * nranks);
        owner[b] = std::min(r, nranks - 1);
      }
      out.push_back(std::move(owner));
    }
    return out;
  }

 private:
  std::vector<LO> box_starts(const AmrField& f, int level) const {
    auto& lvl = boxes_[static_cast<std::size_t>(level)];
    std::vector<LO> s(lvl.size() + 1, 0);
    for (std::size_t b = 0; b < lvl.size(); ++b) {
      auto shp = point_shape(f, lvl[b]);
      LO sz = shp[0] * shp[1] * shp[2] * f.ncomp;
      s[b + 1] = s[b] + sz;
    }
    return s;
  }

  const SomaFile& sf_;
  int format_version_ = 0, dim_ = 0, nparts_ = 0, nlevels_ = 0;
  int ratio_ = 1, blocking_factor_ = 0, max_box_ = 0, max_level_ = 0, nesting_ = 0, curve_ = 0;
  std::array<bool, 3> periodic_{};
  std::array<LO, 3> domain_lo_{}, domain_hi_{};
  std::array<double, 3> lo_{}, hi_{};
  double efficiency_ = 0.0;
  std::vector<std::vector<AmrBox>> boxes_;
  std::vector<AmrField> fields_;
};

// -------------------------------------------------------------------------------------------
// HaloCache: the sharing tables read and inverted once per process, so that reading the halo
// of one part (one VisIt domain) afterwards costs only that part's own interface, not a rescan
// of every other part. Built lazily on first use; reused across every GetMesh call a database
// plugin makes in one process (VisIt's parallel engine gives each process a distinct subset of
// domains, so each process builds its own cache once).
// -------------------------------------------------------------------------------------------
class HaloCache {
 public:
  explicit HaloCache(const SomaFile& sf) : sf_(sf) {}

  void ensure_built() const {
    if (built_) return;
    built_ = true;
    if (!sf_.has("dim0/shared/index")) return;
    auto index = sf_.read<std::int64_t>("dim0/shared/index");
    auto counts = sf_.read<std::int64_t>("dim0/shared/counts");
    auto dest = sf_.read<std::int64_t>("dim0/shared/parts");
    auto idx_rows = sf_.part_starts("dim0/shared/index");
    std::vector<int> owner_of_row(index.size());
    std::size_t r = 0;
    for (int p = 0; p < sf_.nparts(); ++p) {
      LO c = idx_rows[static_cast<std::size_t>(p) + 1] - idx_rows[static_cast<std::size_t>(p)];
      for (LO k = 0; k < c; ++k) owner_of_row[r++] = p;
    }
    std::size_t k = 0;
    for (std::size_t row = 0; row < index.size(); ++row) {
      LO cnt = counts[row];
      for (LO c = 0; c < cnt; ++c, ++k) {
        int d = static_cast<int>(dest[k]);
        dest_to_sources_[d].emplace_back(owner_of_row[row], index[row]);
      }
    }
  }

  // The halo vertices that elements of `parts` need, resolved from the cached inverted index in
  // time proportional to `parts`' own interface; falls back to a full scan (halo_by_scan) for
  // anything the sharing table did not resolve (should not happen on a well-formed file).
  HaloResult halo_for(const std::vector<int>& parts, const std::vector<GO>& missing_sorted,
                      const std::vector<const FieldInfo*>& point_fields, FieldCache& cache) const {
    HaloResult res;
    if (missing_sorted.empty()) return res;
    ensure_built();
    GidIndex missing_index(missing_sorted);
    std::vector<std::pair<int, LO>> sources;
    for (int p : parts) {
      auto it = dest_to_sources_.find(p);
      if (it == dest_to_sources_.end()) continue;
      sources.insert(sources.end(), it->second.begin(), it->second.end());
    }
    std::sort(sources.begin(), sources.end());
    sources.erase(std::unique(sources.begin(), sources.end()), sources.end());

    auto gstart = sf_.part_starts("dim0/gid");
    std::vector<LO> rows(sources.size());
    for (std::size_t k = 0; k < rows.size(); ++k)
      rows[k] = gstart[static_cast<std::size_t>(sources[k].first)] + sources[k].second;
    auto gid_candidates = rows.empty() ? std::vector<GO>{} : sf_.read_rows_at<std::int64_t>("dim0/gid", rows, nullptr);

    std::vector<int> owner_final;
    std::vector<LO> local_final;
    std::vector<GO> gid_final;
    std::set<GO> seen;
    for (std::size_t k = 0; k < gid_candidates.size(); ++k) {
      GO g = gid_candidates[k];
      if (missing_index.lookup_one(g) < 0 || seen.count(g)) continue;
      seen.insert(g);
      owner_final.push_back(sources[k].first);
      local_final.push_back(sources[k].second);
      gid_final.push_back(g);
    }

    res.gid = gid_final;
    res.part.assign(owner_final.begin(), owner_final.end());
    std::vector<LO> final_rows(owner_final.size());
    for (std::size_t k = 0; k < final_rows.size(); ++k)
      final_rows[k] = gstart[static_cast<std::size_t>(owner_final[k])] + local_final[k];
    if (!final_rows.empty()) res.xyz = sf_.read_rows_at<double>("dim0/coords", final_rows, nullptr);
    for (auto* f : point_fields) res.data[f->name] = values_at(sf_, *f, 0, owner_final, local_final, cache);

    if (gid_final.size() != missing_sorted.size()) {
      std::vector<GO> rest;
      for (GO g : missing_sorted)
        if (!seen.count(g)) rest.push_back(g);
      HaloResult fb = halo_by_scan(sf_, parts, rest, point_fields, cache);
      res.gid.insert(res.gid.end(), fb.gid.begin(), fb.gid.end());
      res.xyz.insert(res.xyz.end(), fb.xyz.begin(), fb.xyz.end());
      res.part.insert(res.part.end(), fb.part.begin(), fb.part.end());
      for (auto* f : point_fields) {
        auto& a = res.data[f->name];
        auto& b = fb.data[f->name];
        a.data.insert(a.data.end(), b.data.begin(), b.data.end());
      }
    }
    return res;
  }

 private:
  const SomaFile& sf_;
  mutable bool built_ = false;
  mutable std::map<int, std::vector<std::pair<int, LO>>> dest_to_sources_;
};

// Like read_piece, but resolving the halo from a HaloCache (O(part + its halo) per call once the
// cache is built) rather than rescanning the sharing tables -- the path a database plugin should
// use for its repeated per-domain GetMesh calls.
inline MeshPiece read_piece_cached(const SomaFile& sf, const HaloCache& halo_cache, std::vector<int> parts,
                                  std::vector<int> dims, const ReadOptions& opt = ReadOptions{}) {
  if (sf.kind() != FileKind::mesh) throw io_error(sf.path() + ": an AMR checkpoint, not a mesh");
  if (parts.empty()) {
    parts.resize(static_cast<std::size_t>(sf.nparts()));
    for (int p = 0; p < sf.nparts(); ++p) parts[static_cast<std::size_t>(p)] = p;
  }
  std::sort(parts.begin(), parts.end());
  parts.erase(std::unique(parts.begin(), parts.end()), parts.end());
  if (parts.front() < 0 || parts.back() >= sf.nparts())
    throw io_error(sf.path() + ": parts must be in [0, " + std::to_string(sf.nparts()) + ")");
  if (dims.empty()) dims = {sf.dim()};
  std::sort(dims.begin(), dims.end());
  dims.erase(std::unique(dims.begin(), dims.end()), dims.end());

  FieldCache cache;
  std::vector<std::string> skipped;
  auto point_fields = choose_fields(sf, 0, opt.point_fields, skipped);
  std::map<int, std::vector<const FieldInfo*>> cell_fields;
  for (int d : dims) cell_fields[d] = choose_fields(sf, d, opt.cell_fields, skipped);
  auto set_fields = choose_set_fields(sf, opt);

  auto per0 = sf.counts_by_part(0);
  std::vector<LO> per0_sel;
  for (int p : parts) per0_sel.push_back(per0[static_cast<std::size_t>(p)]);
  std::vector<GO> owned_gids = sf.read<std::int64_t>("dim0/gid", &parts);
  std::vector<double> owned_xyz = sf.read<double>("dim0/coords", &parts);
  std::vector<int> owned_part;
  for (std::size_t k = 0; k < parts.size(); ++k)
    for (LO i = 0; i < per0_sel[k]; ++i) owned_part.push_back(parts[k]);
  std::vector<LO> owned_local = detail::local_rows(per0_sel);
  GidIndex owned_index(owned_gids);
  std::map<std::string, Array> owned_data;
  for (auto* f : point_fields) owned_data[f->name] = values_owned(sf, *f, 0, parts, owned_part, owned_local, cache);

  std::map<int, RawCells> raw;
  std::vector<GO> needed_all;
  for (int d : dims) {
    RawCells c;
    std::string g = "dim" + std::to_string(d);
    c.gids = sf.read<std::int64_t>(g + "/gid", &parts);
    c.topology = sf.read<std::uint8_t>(g + "/topology", &parts);
    c.counts = sf.read<std::int64_t>(g + "/counts", &parts);
    c.conn = sf.read<std::int64_t>(g + "/conn", &parts);
    auto perd = sf.counts_by_part(d);
    std::vector<LO> perd_sel;
    for (int p : parts) perd_sel.push_back(perd[static_cast<std::size_t>(p)]);
    for (std::size_t k = 0; k < parts.size(); ++k)
      for (LO i = 0; i < perd_sel[k]; ++i) c.part.push_back(parts[k]);
    c.local = detail::local_rows(perd_sel);
    if (opt.high_order && sf.has(g + "/nodes")) {
      c.node_counts = sf.read<std::int64_t>(g + "/node_counts", &parts);
      c.nodes = sf.read<std::int64_t>(g + "/nodes", &parts);
      c.has_nodes = true;
      needed_all.insert(needed_all.end(), c.nodes.begin(), c.nodes.end());
    }
    needed_all.insert(needed_all.end(), c.conn.begin(), c.conn.end());
    raw[d] = std::move(c);
  }
  std::sort(needed_all.begin(), needed_all.end());
  needed_all.erase(std::unique(needed_all.begin(), needed_all.end()), needed_all.end());
  std::vector<GO> missing;
  for (GO g : needed_all)
    if (owned_index.lookup_one(g) < 0) missing.push_back(g);
  std::sort(missing.begin(), missing.end());

  HaloResult halo = halo_cache.halo_for(parts, missing, point_fields, cache);

  MeshPiece piece;
  piece.path = sf.path();
  piece.dim = sf.dim();
  piece.parts = parts;
  piece.halo_points = static_cast<LO>(halo.gid.size());
  piece.skipped = std::move(skipped);
  piece.time = sf.time();
  piece.point_gids = owned_gids;
  piece.point_gids.insert(piece.point_gids.end(), halo.gid.begin(), halo.gid.end());
  piece.points = owned_xyz;
  piece.points.insert(piece.points.end(), halo.xyz.begin(), halo.xyz.end());
  piece.point_part = owned_part;
  piece.point_part.insert(piece.point_part.end(), halo.part.begin(), halo.part.end());
  for (auto* f : point_fields) {
    Array a = owned_data[f->name];
    auto hit = halo.data.find(f->name);
    if (hit != halo.data.end()) a.data.insert(a.data.end(), hit->second.data.begin(), hit->second.data.end());
    piece.point_data[f->name] = std::move(a);
  }

  GidIndex full_index = halo.gid.empty() ? std::move(owned_index) : GidIndex(piece.point_gids);
  bool have_sets = !set_fields.empty();
  SetsInfo sets_info = have_sets ? build_sets_info(sf, parts, set_fields) : SetsInfo{};

  for (int d : dims) {
    auto& c = raw[d];
    auto conn_local = full_index.lookup(c.conn);
    for (auto v : conn_local)
      if (v < 0)
        throw io_error(sf.path() + ": elements of dimension " + std::to_string(d) +
                       " use vertices no part shares with the requested parts");
    std::vector<LO> nodes_local;
    std::vector<LO>* nodes_local_ptr = nullptr;
    if (c.has_nodes) {
      nodes_local = full_index.lookup(c.nodes);
      for (auto v : nodes_local)
        if (v < 0)
          throw io_error(sf.path() + ": elements of dimension " + std::to_string(d) +
                         " use vertices no part shares with the requested parts");
      nodes_local_ptr = &nodes_local;
    }
    VtkCells vc = vtk_cells(c.topology, c.counts, conn_local, c.has_nodes ? &c.node_counts : nullptr,
                           nodes_local_ptr, opt.high_order, opt.legacy_wedge);
    CellBlock block;
    block.dim = d;
    block.gids = c.gids;
    block.part = c.part;
    block.types = std::move(vc.types);
    block.offsets = std::move(vc.offsets);
    block.connectivity = std::move(vc.connectivity);
    for (auto* f : cell_fields[d]) block.cell_data[f->name] = values_owned(sf, *f, d, parts, c.part, c.local, cache);
    if (have_sets) apply_set_labels(sets_info, d, c, parts, sf, block);
    piece.blocks[d] = std::move(block);
  }
  return piece;
}

// -------------------------------------------------------------------------------------------
// Spectral-element fields on sub-quads (nodal.py): a field of soma::sem::SemSpace holds (p + 1)^2
// values per quad, at its Gauss-Lobatto-Legendre (GLL) points, value k = i + (p + 1) j at
// (xi_i, eta_j). Every quad becomes p x p linear sub-quads on those points and the stored values
// point data, nothing averaged or interpolated; points are not shared between elements, so a field
// that jumps from one to the next is drawn as it is. The points are where SemSpace puts them:
// bilinear in the corners (biquadratic with 8 or 9 nodes), on a sphere projected to it.
// -------------------------------------------------------------------------------------------

// SemSpace takes a corner to be on its sphere within this relative distance; two corners may
// therefore differ by twice as much.
constexpr double SPHERE_TOLERANCE = 1e-8;
// Vertices looked at to tell a sphere from a plane.
constexpr hsize_t SPHERE_SAMPLE = 1 << 16;

// The p + 1 GLL points on [-1, 1]: the ends and the zeros of P_p'.
inline std::vector<double> gll_points(int p) {
  std::vector<double> x(static_cast<std::size_t>(p) + 1, 0.0);
  x.front() = -1;
  x.back() = 1;
  const double pi = std::acos(-1.0);
  for (int i = 1; i <= (p - 1) / 2; ++i) {
    double t = -std::cos(pi * i / p);
    for (int it = 0; it < 100; ++it) {
      double p0 = 1, p1 = t, d0 = 0, d1 = 1;
      for (int k = 1; k < p; ++k) {
        const double p2 = ((2 * k + 1) * t * p1 - k * p0) / (k + 1);
        const double d2 = d0 + (2 * k + 1) * p1;
        p0 = p1; p1 = p2; d0 = d1; d1 = d2;
      }
      const double step = d1 / ((2 * t * d1 - p * (p + 1.0) * p1) / (1 - t * t));
      t -= step;
      if (std::fabs(step) < 1e-15) break;
    }
    x[static_cast<std::size_t>(i)] = t;
    x[static_cast<std::size_t>(p - i)] = -t;
  }
  return x;
}

// Shape functions of a quad of n = 4, 8 or 9 nodes (MOAB's order) at (s, t).
inline void quad_shape(int n, double s, double t, double* f) {
  static const int ref[9][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}, {0, -1}, {1, 0}, {0, 1}, {-1, 0}, {0, 0}};
  auto line = [](int a, double r) { return a == 0 ? 1 - r * r : r * (r + a) / 2; };
  for (int k = 0; k < n; ++k) {
    const int a = ref[k][0], b = ref[k][1];
    if (n == 4) f[k] = (1 + a * s) * (1 + b * t) / 4;
    else if (n == 9) f[k] = line(a, s) * line(b, t);
    else if (k < 4) f[k] = (1 + a * s) * (1 + b * t) * (a * s + b * t - 1) / 4;
    else if (a == 0) f[k] = (1 - s * s) * (1 + b * t) / 2;
    else f[k] = (1 + a * s) * (1 - t * t) / 2;
  }
}

// The radius of the sphere about the origin the mesh is on, 0 for any other mesh: a file does not
// say. Decided from the file's first vertices, so the same for every part of it: all at one distance
// from the origin, within SemSpace's tolerance, and not all in the plane z = 0.
inline double sphere_radius(const SomaFile& sf) {
  const hsize_t n = std::min(h5detail::dataset_extent(sf.raw(), "dim0/coords").first, SPHERE_SAMPLE);
  if (n == 0) return 0;
  hsize_t width = 0;
  const auto x = h5detail::read_rows<double>(sf.raw(), "dim0/coords", 0, n, &width);
  if (width != 3) throw io_error(sf.path() + ": dim0/coords has " + std::to_string(width) + " columns");
  auto radius = [&](hsize_t i) { return std::sqrt(x[3 * i] * x[3 * i] + x[3 * i + 1] * x[3 * i + 1] + x[3 * i + 2] * x[3 * i + 2]); };
  const double r = radius(0), tol = 2 * SPHERE_TOLERANCE * r;
  double z = 0;
  for (hsize_t i = 0; i < n; ++i) {
    if (!(std::fabs(radius(i) - r) <= tol)) return 0;
    z = std::max(z, std::fabs(x[3 * i + 2]));
  }
  return z > tol ? r : 0;
}

struct Subcells {
  int order = 0;
  std::vector<LO> cells;         // the block's cell each group of order^2 sub-quads is of
  std::vector<double> points;    // (order + 1)^2 per cell, n * 3
  std::vector<LO> connectivity;  // 4 per sub-quad
  LO size() const { return static_cast<LO>(connectivity.size() / 4); }
  LO num_points() const { return static_cast<LO>(points.size() / 3); }
};

// The sub-quads of order `p` of the quads of `block`, on the sphere of `radius` (0: none).
inline Subcells subcells(const MeshPiece& piece, const CellBlock& block, int p, double radius) {
  Subcells out;
  out.order = p;
  const int np = p + 1;
  const auto xi = gll_points(p);
  std::vector<double> shape4, shape8, shape9;
  for (auto [n, table] : {std::make_pair(4, &shape4), std::make_pair(8, &shape8), std::make_pair(9, &shape9)}) {
    table->resize(static_cast<std::size_t>(np * np * n));
    for (int k = 0; k < np * np; ++k)
      quad_shape(n, xi[static_cast<std::size_t>(k % np)], xi[static_cast<std::size_t>(k / np)], table->data() + k * n);
  }
  for (LO c = 0; c < block.size(); ++c) {
    const auto uc = static_cast<std::size_t>(c);
    const int t = block.types[uc];
    const int n = t == vtk_type::QUAD ? 4 : t == vtk_type::QUADRATIC_QUAD ? 8 : t == vtk_type::BIQUADRATIC_QUAD ? 9 : 0;
    if (n == 0) continue;
    const auto& shape = n == 4 ? shape4 : n == 8 ? shape8 : shape9;
    double node[9][3];
    bool ball = radius > 0;
    for (int a = 0; a < n; ++a) {
      const auto v = static_cast<std::size_t>(block.connectivity[static_cast<std::size_t>(block.offsets[uc] + a)]);
      for (int d = 0; d < 3; ++d) node[a][d] = piece.points[3 * v + static_cast<std::size_t>(d)];
      const double r = std::sqrt(node[a][0] * node[a][0] + node[a][1] * node[a][1] + node[a][2] * node[a][2]);
      if (a < 4 && !(std::fabs(r - radius) <= 2 * SPHERE_TOLERANCE * radius)) ball = false;
    }
    const LO first = out.num_points();
    for (int k = 0; k < np * np; ++k) {
      double y[3] = {0, 0, 0};
      for (int a = 0; a < n; ++a)
        for (int d = 0; d < 3; ++d) y[d] += shape[static_cast<std::size_t>(k * n + a)] * node[a][d];
      const double s = ball ? radius / std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]) : 1;
      for (double v : y) out.points.push_back(s * v);
    }
    for (int j = 0; j < p; ++j)
      for (int i = 0; i < p; ++i)
        for (int k : {i + np * j, i + 1 + np * j, i + 1 + np * (j + 1), i + np * (j + 1)})
          out.connectivity.push_back(first + k);
    out.cells.push_back(c);
  }
  return out;
}

// Values of `info` at the points of `sub`, NaN on elements off its domain. `piece` holds the
// elements of dimension 2 of whole parts, in file order, as `sub` was made from it.
inline Array subcell_values(const SomaFile& sf, const FieldInfo& info, const MeshPiece& piece, const Subcells& sub,
                            FieldCache& cache) {
  const std::string base = info.group() + "/dim2";
  const LO npts = static_cast<LO>(sub.order + 1) * (sub.order + 1);
  const auto& full = detail::full_flags(sf, info, 2, cache);
  std::vector<LO> first, size;  // per cell: its first row among the parts' values (-1: none), its rows
  LO start = 0;
  for (int q : piece.parts) {
    const std::vector<int> qp{q};
    const LO n = sf.count(2, &qp);
    const auto counts = sf.read<std::int64_t>(base + "/counts", &qp);
    const bool whole = static_cast<std::size_t>(q) < full.size() && full[static_cast<std::size_t>(q)];
    const auto domain = whole ? std::vector<std::int64_t>{} : sf.read<std::int64_t>(base + "/domain", &qp);
    const LO members = whole ? n : static_cast<LO>(domain.size());
    const auto [lo, hi] = sf.part_rows(base + "/values", q);
    const std::string where = " of part " + std::to_string(q);
    for (std::size_t m = 0; m < domain.size(); ++m)
      if (domain[m] < 0 || domain[m] >= n || (m > 0 && domain[m] <= domain[m - 1]))
        throw io_error(sf.path() + ": " + base + "/domain" + where + " is not increasing rows of its " +
                       std::to_string(n) + " elements");
    if (static_cast<LO>(counts.size()) != members)
      throw io_error(sf.path() + ": " + base + "/counts" + where + " has " + std::to_string(counts.size()) +
                     " rows for " + std::to_string(members) + " elements");
    const std::size_t offset = first.size();
    first.resize(offset + static_cast<std::size_t>(n), -1);
    size.resize(offset + static_cast<std::size_t>(n), npts);
    LO at = 0;
    for (LO m = 0; m < members; ++m) {
      const LO c = counts[static_cast<std::size_t>(m)];
      if (c < 0 || c > hi - lo - at)
        throw io_error(sf.path() + ": " + base + "/counts" + where + " does not add up to its " +
                       std::to_string(hi - lo) + " values");
      const std::size_t e = offset + static_cast<std::size_t>(whole ? m : domain[static_cast<std::size_t>(m)]);
      first[e] = start + at;
      size[e] = c;
      at += c;
    }
    if (at != hi - lo)
      throw io_error(sf.path() + ": " + base + "/counts" + where + " does not add up to its " +
                     std::to_string(hi - lo) + " values");
    start += hi - lo;
  }
  hsize_t width = 0;
  const auto values = sf.read<double>(base + "/values", &piece.parts, &width);
  if (static_cast<int>(width) != info.ncomp || values.size() != static_cast<std::size_t>(start) * width)
    throw io_error(sf.path() + ": " + base + "/values does not hold " + std::to_string(start) + " rows of " +
                   std::to_string(info.ncomp) + " components");
  Array out;
  out.ncomp = info.ncomp;
  const std::size_t block = static_cast<std::size_t>(npts) * width;
  out.data.assign(sub.cells.size() * block, std::numeric_limits<double>::quiet_NaN());
  for (std::size_t k = 0; k < sub.cells.size(); ++k) {
    const auto e = static_cast<std::size_t>(sub.cells[k]);
    if (e >= first.size()) throw io_error(sf.path() + ": " + base + " has fewer elements than its mesh");
    if (size[e] != npts)
      throw io_error(sf.path() + ": " + base + "/counts gives a quad " + std::to_string(size[e]) + " values, not " +
                     std::to_string(npts));
    if (first[e] < 0) continue;
    const auto from = values.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(first[e]) * width);
    std::copy(from, from + static_cast<std::ptrdiff_t>(block), out.data.begin() + static_cast<std::ptrdiff_t>(k * block));
  }
  return out;
}

}  // namespace soma_reader

#endif  // SOMA_READER_HPP
