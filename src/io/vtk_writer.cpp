// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/io/vtk_writer.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "soma/core/topology.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/fields/layout.hpp"

namespace soma {
namespace {

namespace fs = std::filesystem;

std::string cannot_write(const fs::path& path, const std::string& why) {
  return "write_vtk: cannot write " + path.string() + ": " + why;
}

// The text of an errno value; unlike strerror, safe to call from any thread.
std::string error_text(int e) { return std::system_category().message(e); }

// Buffered writes to a file descriptor; a failed write throws naming `path`.
class Sink {
 public:
  Sink(int fd, const fs::path& path) : fd_(fd), path_(path) { buf_.reserve(flush_at + 512); }

  void text(std::string_view s) {
    buf_.append(s);
    drain();
  }
  void put(char c) {
    buf_.push_back(c);
    drain();
  }
  void end_line() {
    buf_.push_back('\n');
    drain();
  }

  template <class T>
  void integer(T v) {
    char tmp[24];
    const auto r = std::to_chars(tmp, tmp + sizeof tmp, v);
    buf_.append(tmp, r.ptr);
    drain();
  }

  // The shortest text that reads back as the same value.
  template <class F>
  void real(F v) {
    char tmp[40];
    const auto r = std::to_chars(tmp, tmp + sizeof tmp, v);
    buf_.append(tmp, r.ptr);
    drain();
  }

  void finish() { flush(); }

 private:
  static constexpr std::size_t flush_at = std::size_t{1} << 16;

  void drain() {
    if (buf_.size() >= flush_at) flush();
  }
  void flush() {
    const char* p = buf_.data();
    std::size_t left = buf_.size();
    while (left > 0) {
      const ssize_t n = ::write(fd_, p, left);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) {
        const int e = n < 0 ? errno : EIO;
        throw std::runtime_error(cannot_write(path_, "the write failed: " + error_text(e)));
      }
      p += n;
      left -= static_cast<std::size_t>(n);
    }
    buf_.clear();
  }

  int fd_;
  const fs::path& path_;
  std::string buf_;
};

int cell_type_code(Topology t) {
  switch (t) {
    case Topology::edge: return 3;
    case Topology::tri: return 5;
    case Topology::polygon: return 7;
    case Topology::quad: return 9;
    case Topology::tet: return 10;
    case Topology::hex: return 12;
    case Topology::prism: return 13;
    case Topology::pyramid: return 14;
    case Topology::vertex:
    case Topology::knife:
    case Topology::polyhedron: break;
  }
  return -1;
}

const char* value_type_name(DataType t) {
  switch (t) {
    case DataType::f64: return "double";
    case DataType::f32: return "float";
    case DataType::i32: return "int";
    case DataType::i64: return "long";
    case DataType::u8: return "unsigned_char";
    case DataType::u32: return "unsigned_int";
    case DataType::u64: return "unsigned_long";
    case DataType::i8: break;
  }
  return nullptr;
}

std::string in_quotes(const std::string& s) { return "'" + s + "'"; }

std::string type_text(const FieldInfo& i) {
  return std::string(name(i.type)) + "x" + std::to_string(i.ncomp);
}

std::string dims_text(const FieldInfo& i) {
  std::string s;
  for (const int d : i.dims) s += (s.empty() ? "" : ",") + std::to_string(d);
  return s;
}

// Why a name cannot be a legacy array name, or nullptr when it can.
const char* name_problem(const std::string& n) {
  if (n.empty()) return "it is empty";
  if (n.size() > 255) return "it is longer than 255 characters";
  if (n == "NULL_ARRAY") return "NULL_ARRAY is reserved";
  for (const char c : n) {
    const auto u = static_cast<unsigned char>(c);
    if (u == ' ' || (u >= '\t' && u <= '\r')) return "it holds whitespace";
    if (u < 0x20 || u == 0x7f) return "it holds a control character";
    if (c == '%') return "it holds '%'";
  }
  return nullptr;
}

void check_name(const std::string& n) {
  if (const char* why = name_problem(n)) {
    throw std::invalid_argument("write_vtk: field name " + in_quotes(n) + " cannot be written: " +
                                why);
  }
}

struct Array {
  FieldInfo info;
  int dim = 0;
};

struct Contents {
  int dim = 0;
  std::vector<std::uint8_t> cell_types;
  std::vector<Array> point_arrays;
  std::vector<Array> cell_arrays;
};

// Fetching the field is also the currency check: it throws StaleField.
DomainKind domain_kind(const HostMesh& mesh, const FieldInfo& info) {
  DomainKind kind = DomainKind::full;
  dispatch_data_type(info.type, [&]<class T>() {
    kind = mesh.fields().template get<T>(info.id).domain().info(0).kind;
  });
  return kind;
}

Array resolve(const HostMesh& mesh, const VtkField& want) {
  check_name(want.name);
  const std::vector<FieldInfo> all = mesh.fields().list();
  std::vector<const FieldInfo*> hits;
  for (const FieldInfo& i : all) {
    if (i.name != want.name) continue;
    if (want.dim && std::find(i.dims.begin(), i.dims.end(), *want.dim) == i.dims.end()) continue;
    hits.push_back(&i);
  }
  const std::string what = "field " + in_quotes(want.name) +
                           (want.dim ? " on dimension " + std::to_string(*want.dim) : "");
  if (hits.empty()) throw std::invalid_argument("write_vtk: no " + what);
  if (hits.size() > 1) {
    std::string list;
    bool dims_differ = false;
    bool types_differ = false;
    for (const FieldInfo* h : hits) {
      list += (list.empty() ? "" : "; ") + type_text(*h) + " " + describe(h->layout) +
              " on dimension " + dims_text(*h);
      dims_differ = dims_differ || h->dims != hits.front()->dims;
      types_differ = types_differ || type_text(*h) != type_text(*hits.front());
    }
    const char* advice =
        !want.dim && dims_differ ? "select it with a dimension, or rename one"
        : types_differ ? "the candidates differ by value type, which a selection cannot "
                         "express; rename one"
                       : "the candidates differ by layout or domain, which a selection cannot "
                         "express yet; rename one";
    throw std::invalid_argument("write_vtk: " + what + " is ambiguous (" + list + "); " + advice);
  }
  const FieldInfo& info = *hits.front();
  const DomainKind kind = domain_kind(mesh, info);
  if (info.dims.size() != 1) {
    throw std::invalid_argument("write_vtk: field " + in_quotes(info.name) +
                                " spans dimensions " + dims_text(info) + "; it needs exactly one");
  }
  const int dim = info.dims.front();
  if (dim != 0 && dim != mesh.dim()) {
    throw std::invalid_argument("write_vtk: field " + in_quotes(info.name) +
                                " lives on dimension " + std::to_string(dim) +
                                "; only dimension 0 (points) and " + std::to_string(mesh.dim()) +
                                " (cells) can be written");
  }
  if (info.layout.kind != Layout::Kind::per_entity) {
    throw std::invalid_argument("write_vtk: field " + in_quotes(info.name) + " has layout " +
                                describe(info.layout) + "; only per_entity can be written");
  }
  if (value_type_name(info.type) == nullptr) {
    throw std::invalid_argument("write_vtk: field " + in_quotes(info.name) +
                                " has value type " + name(info.type) +
                                ", which has no portable legacy VTK name");
  }
  const std::size_t pending = mesh.fields().pending_count(info.id);
  if (pending != 0) {
    throw std::invalid_argument("write_vtk: field " + in_quotes(info.name) + " has " +
                                std::to_string(pending) +
                                " uncommitted writes; commit it before writing");
  }
  if (kind != DomainKind::full) {
    throw std::invalid_argument("write_vtk: field " + in_quotes(info.name) +
                                " is defined on a " + name(kind) +
                                " subset of the entities; only a full domain can be written");
  }
  return Array{info, dim};
}

// A mirror that left out the coordinates or a dimension holds Views that do
// not match its counts; reading them would run past their ends.
void check_mesh_extents(const HostMesh& mesh, int dim) {
  const auto nv = static_cast<std::size_t>(mesh.count(0));
  const auto xyz = mesh.coords();
  detail::check_vtk_extent("the coordinates", xyz.extent(0), xyz.extent(1), nv, 3);
  const auto nc = static_cast<std::size_t>(mesh.count(dim));
  if (nc == 0) return;
  const std::string of = " of dimension " + std::to_string(dim);
  const auto entries = [](std::size_t k) { return std::to_string(k) + " entries"; };
  const auto offsets = mesh.conn_offsets(dim);
  const auto conn = mesh.conn(dim);
  if (offsets.extent(0) != nc + 1) {
    throw std::invalid_argument("write_vtk: the connectivity offsets" + of + " have " +
                                entries(offsets.extent(0)) + " for " + std::to_string(nc) +
                                " cells");
  }
  if (mesh.topology(dim).extent(0) != nc || mesh.gid(dim).extent(0) != nc) {
    throw std::invalid_argument("write_vtk: the cell types or global ids" + of + " do not have " +
                                entries(nc));
  }
  for (std::size_t e = 0; e < nc; ++e) {
    if (offsets(e) < 0 || offsets(e + 1) < offsets(e)) {
      throw std::invalid_argument("write_vtk: the connectivity offsets" + of +
                                  " decrease at cell row " + std::to_string(e));
    }
  }
  const auto used = static_cast<std::size_t>(offsets(nc));
  if (conn.extent(0) < used) {
    throw std::invalid_argument("write_vtk: the connectivity" + of + " has " +
                                entries(conn.extent(0)) + "; its offsets need " +
                                std::to_string(used));
  }
  for (std::size_t e = 0; e < nc; ++e) {
    for (auto k = static_cast<std::size_t>(offsets(e)); k < static_cast<std::size_t>(offsets(e + 1));
         ++k) {
      if (conn(k) < 0 || static_cast<std::size_t>(conn(k)) >= nv) {
        throw std::invalid_argument("write_vtk: the connectivity of cell row " +
                                    std::to_string(e) + " refers to vertex " +
                                    std::to_string(conn(k)) + " of " + std::to_string(nv));
      }
    }
  }
}

void check_field_extents(const HostMesh& mesh, const Array& a) {
  dispatch_data_type(a.info.type, [&]<class T>() {
    const auto f = mesh.fields().template get<T>(a.info.id);
    const auto v = f.view();
    const std::string what = "field " + in_quotes(a.info.name);
    const auto ncomp = static_cast<std::size_t>(a.info.ncomp);
    detail::check_vtk_extent(what, v.extent(0), v.extent(1),
                             static_cast<std::size_t>(mesh.count(a.dim)), ncomp);
    detail::check_vtk_extent(what, v.extent(0), v.extent(1),
                             static_cast<std::size_t>(f.space().rows()), ncomp);
  });
}

Contents collect_contents(const HostMesh& mesh, std::span<const VtkField> fields) {
  Contents contents;
  contents.dim = mesh.dim();
  if (contents.dim < 1) {
    throw std::invalid_argument("write_vtk: the mesh has no elements (default or moved-from)");
  }
  check_mesh_extents(mesh, contents.dim);
  const auto topology = mesh.topology(contents.dim);
  const auto gid = mesh.gid(contents.dim);
  const auto n = static_cast<std::size_t>(mesh.count(contents.dim));
  contents.cell_types.resize(n);
  for (std::size_t e = 0; e < n; ++e) {
    const auto t = static_cast<Topology>(topology(e));
    const int code = cell_type_code(t);
    if (code < 0) {
      throw std::invalid_argument(std::string("write_vtk: element gid ") +
                                  std::to_string(gid(e)) + " is a " + name(t) +
                                  ", which legacy VTK cannot represent");
    }
    contents.cell_types[e] = static_cast<std::uint8_t>(code);
  }

  std::vector<FieldId> seen;
  for (const VtkField& want : fields) {
    Array a = resolve(mesh, want);
    if (std::find(seen.begin(), seen.end(), a.info.id) != seen.end()) {
      throw std::invalid_argument("write_vtk: field " + in_quotes(a.info.name) +
                                  " is selected twice");
    }
    seen.push_back(a.info.id);
    check_field_extents(mesh, a);
    (a.dim == 0 ? contents.point_arrays : contents.cell_arrays).push_back(std::move(a));
  }
  return contents;
}

// Legacy VTK readers stop at a NaN or an infinity and lose the rest of the
// file; their strtod rejects subnormal values.
template <class F>
const char* unreadable(F v) {
  switch (std::fpclassify(v)) {
    case FP_NAN: return "is NaN";
    case FP_INFINITE: return "is infinite";
    case FP_SUBNORMAL: return "is subnormal";
    default: return nullptr;
  }
}

template <class V>
void check_reals(const V& v, std::size_t rows, std::size_t ncomp, const std::string& what) {
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t c = 0; c < ncomp; ++c) {
      if (const char* why = unreadable(v(i, c))) {
        throw std::invalid_argument("write_vtk: " + what + " at row " + std::to_string(i) +
                                    ", component " + std::to_string(c) + " " + why +
                                    ", which legacy VTK cannot read");
      }
    }
  }
}

void check_ids(const HostMesh& mesh, const Contents& contents) {
  const auto nc = static_cast<std::size_t>(mesh.count(contents.dim));
  std::int64_t largest = -1;
  std::int64_t size = 0;
  if (nc > 0) {
    const auto offsets = mesh.conn_offsets(contents.dim);
    const auto conn = mesh.conn(contents.dim);
    const auto end = static_cast<std::size_t>(offsets(nc));
    for (std::size_t k = 0; k < end; ++k) largest = std::max<std::int64_t>(largest, conn(k));
    size = static_cast<std::int64_t>(nc) + static_cast<std::int64_t>(offsets(nc));
  }
  detail::check_vtk_int32(mesh.count(0), largest, size);
}

void check_values(const HostMesh& mesh, const Contents& contents) {
  check_reals(mesh.coords(), static_cast<std::size_t>(mesh.count(0)), 3, "coordinates");
  for (const auto* arrays : {&contents.point_arrays, &contents.cell_arrays}) {
    for (const Array& a : *arrays) {
      dispatch_data_type(a.info.type, [&]<class T>() {
        if constexpr (std::is_floating_point_v<T>) {
          check_reals(mesh.fields().template get<T>(a.info.id).view(),
                      static_cast<std::size_t>(mesh.count(a.dim)),
                      static_cast<std::size_t>(a.info.ncomp), "field " + in_quotes(a.info.name));
        }
      });
    }
  }
}

template <class T>
void emit_values(Sink& out, const HostMesh& mesh, const Array& a) {
  const auto v = mesh.fields().template get<T>(a.info.id).view();
  const auto n = static_cast<std::size_t>(mesh.count(a.dim));
  const auto ncomp = static_cast<std::size_t>(a.info.ncomp);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t c = 0; c < ncomp; ++c) {
      if (c != 0) out.put(' ');
      if constexpr (std::is_floating_point_v<T>) {
        out.real(v(i, c));
      } else {
        out.integer(v(i, c));
      }
    }
    out.end_line();
  }
}

void emit_attributes(Sink& out, const HostMesh& mesh, const char* section, LO count,
                     const std::vector<Array>& arrays) {
  if (arrays.empty()) return;
  out.text(section);
  out.put(' ');
  out.integer(count);
  out.end_line();
  out.text("FIELD FieldData ");
  out.integer(arrays.size());
  out.end_line();
  for (const Array& a : arrays) {
    out.text(a.info.name);
    out.put(' ');
    out.integer(a.info.ncomp);
    out.put(' ');
    out.integer(count);
    out.put(' ');
    out.text(value_type_name(a.info.type));
    out.end_line();
    dispatch_data_type(a.info.type, [&]<class T>() { emit_values<T>(out, mesh, a); });
  }
}

void emit(Sink& out, const HostMesh& mesh, const Contents& contents) {
  out.text("# vtk DataFile Version 3.0");
  out.end_line();
  out.text("SOMA mesh");
  out.end_line();
  out.text("ASCII");
  out.end_line();
  out.text("DATASET UNSTRUCTURED_GRID");
  out.end_line();

  const LO nv = mesh.count(0);
  out.text("POINTS ");
  out.integer(nv);
  out.text(" double");
  out.end_line();
  const auto xyz = mesh.coords();
  for (std::size_t i = 0; i < static_cast<std::size_t>(nv); ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      if (k != 0) out.put(' ');
      out.real(xyz(i, k));
    }
    out.end_line();
  }

  const LO nc = mesh.count(contents.dim);
  const auto offsets = mesh.conn_offsets(contents.dim);
  const auto conn = mesh.conn(contents.dim);
  const auto nce = static_cast<std::size_t>(nc);
  out.text("CELLS ");
  out.integer(nc);
  out.put(' ');
  out.integer(nce == 0 ? std::int64_t{0}
                      : static_cast<std::int64_t>(nce) + static_cast<std::int64_t>(offsets(nce)));
  out.end_line();
  for (std::size_t e = 0; e < nce; ++e) {
    const auto first = static_cast<std::size_t>(offsets(e));
    const auto last = static_cast<std::size_t>(offsets(e + 1));
    out.integer(last - first);
    for (std::size_t k = first; k < last; ++k) {
      out.put(' ');
      out.integer(conn(k));
    }
    out.end_line();
  }

  out.text("CELL_TYPES ");
  out.integer(nc);
  out.end_line();
  for (const std::uint8_t t : contents.cell_types) {
    out.integer(static_cast<int>(t));
    out.end_line();
  }

  emit_attributes(out, mesh, "POINT_DATA", nv, contents.point_arrays);
  emit_attributes(out, mesh, "CELL_DATA", nc, contents.cell_arrays);
  out.finish();
}

// A new file next to `path`, named `path` + "." + six random characters and
// created exclusively (O_EXCL, O_NOFOLLOW), so that it never opens an existing
// file or follows a symlink. Unlike mkstemp, which forces mode 0600, it is
// created with 0666 so that the umask decides the mode of the result.
class Temporary {
 public:
  explicit Temporary(const fs::path& path) : path_(path) {
    static constexpr char letters[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    std::random_device random;
    for (int attempt = 0; attempt < 100; ++attempt) {
      std::string candidate = path.string() + ".";
      for (int k = 0; k < 6; ++k) candidate += letters[random() % (sizeof letters - 1)];
      fd_ = ::open(candidate.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0666);
      if (fd_ >= 0) {
        name_ = std::move(candidate);
        return;
      }
      const int e = errno;
      if (e != EEXIST && e != EINTR) {
        throw std::runtime_error(
            cannot_write(path, "cannot create a temporary file: " + error_text(e)));
      }
    }
    throw std::runtime_error(cannot_write(path, "no unused temporary file name was found"));
  }
  Temporary(const Temporary&) = delete;
  Temporary& operator=(const Temporary&) = delete;
  ~Temporary() {
    if (fd_ >= 0) ::close(fd_);
    if (!name_.empty()) ::unlink(name_.c_str());
  }

  int fd() const noexcept { return fd_; }

  void close() {
    const int fd = fd_;
    fd_ = -1;
    if (::close(fd) != 0) {
      const int e = errno;
      throw std::runtime_error(cannot_write(path_, "closing the file failed: " + error_text(e)));
    }
  }

  // Moves the file over `path`; afterwards there is nothing left to remove.
  void commit() {
    if (::rename(name_.c_str(), path_.c_str()) != 0) {
      const int e = errno;
      throw std::runtime_error(cannot_write(path_, error_text(e)));
    }
    name_.clear();
  }

 private:
  const fs::path& path_;
  std::string name_;
  int fd_ = -1;
};

}  // namespace

void write_vtk(const HostMesh& mesh, const fs::path& path, std::span<const VtkField> fields) {
  const Contents contents = collect_contents(mesh, fields);
  check_ids(mesh, contents);
  check_values(mesh, contents);

  Temporary file(path);
  Sink out(file.fd(), path);
  emit(out, mesh, contents);
  file.close();
  file.commit();
}

void detail::check_vtk_int32(std::int64_t npoints, std::int64_t largest_id,
                             std::int64_t cells_size) {
  constexpr std::int64_t most = std::numeric_limits<std::int32_t>::max();
  const auto check = [](std::int64_t v, const char* what) {
    if (v > most) {
      throw std::length_error("write_vtk: " + std::string(what) + " is " + std::to_string(v) +
                              ", more than the 32-bit int (" + std::to_string(most) +
                              ") a legacy VTK reader uses");
    }
  };
  check(npoints, "the number of points");
  check(largest_id, "the largest connectivity id");
  check(cells_size, "the CELLS size (cells plus vertex references)");
}

void detail::check_vtk_extent(const std::string& what, std::size_t rows, std::size_t cols,
                              std::size_t want_rows, std::size_t want_cols) {
  if (rows != want_rows || cols != want_cols) {
    throw std::invalid_argument("write_vtk: the extents of " + what + " are " +
                                std::to_string(rows) + " x " + std::to_string(cols) +
                                " where the mesh needs " +
                                std::to_string(want_rows) + " x " + std::to_string(want_cols));
  }
}

void write_vtk(const HostMesh& mesh, const fs::path& path, std::span<const std::string> fields) {
  std::vector<VtkField> selection;
  selection.reserve(fields.size());
  for (const std::string& n : fields) selection.push_back(VtkField{n, std::nullopt});
  write_vtk(mesh, path, std::span<const VtkField>(selection));
}

}  // namespace soma
