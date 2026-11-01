// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/io/vtk_writer.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#include "soma/core/topology.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/fields/layout.hpp"

namespace soma {
namespace {

namespace fs = std::filesystem;

class Sink {
 public:
  explicit Sink(std::ostream& os) : os_(os) { buf_.reserve(flush_at + 512); }

  void text(std::string_view s) {
    buf_.append(s);
    drain();
  }
  void put(char c) { buf_.push_back(c); }
  void end_line() {
    buf_.push_back('\n');
    drain();
  }

  template <class T>
  void integer(T v) {
    char tmp[24];
    const auto r = std::to_chars(tmp, tmp + sizeof tmp, v);
    buf_.append(tmp, r.ptr);
  }

  // The shortest text that reads back as the same value; NaN carries no sign
  // or payload in the file.
  template <class F>
  void real(F v) {
    if (std::isnan(v)) {
      buf_ += "nan";
      return;
    }
    char tmp[40];
    const auto r = std::to_chars(tmp, tmp + sizeof tmp, v);
    buf_.append(tmp, r.ptr);
  }

  void finish() {
    flush();
    os_.flush();
  }

 private:
  static constexpr std::size_t flush_at = std::size_t{1} << 16;

  void drain() {
    if (buf_.size() >= flush_at) flush();
  }
  void flush() {
    os_.write(buf_.data(), static_cast<std::streamsize>(buf_.size()));
    buf_.clear();
  }

  std::ostream& os_;
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

void check_name(const std::string& n) {
  const bool plain = !n.empty() && std::none_of(n.begin(), n.end(), [](char c) {
    return static_cast<unsigned char>(c) <= 0x20 || static_cast<unsigned char>(c) == 0x7f ||
           c == '%';
  });
  if (!plain || n.size() > 255 || n == "NULL_ARRAY") {
    throw std::invalid_argument("write_vtk: field name " + in_quotes(n) +
                                " cannot be written (it is empty, longer than 255 characters, "
                                "holds whitespace, control characters or '%', or is NULL_ARRAY)");
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
    for (const FieldInfo* h : hits) {
      list += (list.empty() ? "" : "; ") + type_text(*h) + " on dimension " + dims_text(*h);
    }
    throw std::invalid_argument("write_vtk: " + what + " is ambiguous (" + list +
                                "); select it with a dimension, or rename one");
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

Contents collect_contents(const HostMesh& mesh, std::span<const VtkField> fields) {
  Contents contents;
  contents.dim = mesh.dim();
  if (contents.dim < 1) {
    throw std::invalid_argument("write_vtk: the mesh has no elements (default or moved-from)");
  }
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
    (a.dim == 0 ? contents.point_arrays : contents.cell_arrays).push_back(std::move(a));
  }
  return contents;
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
  out.integer(static_cast<std::int64_t>(nce) + static_cast<std::int64_t>(offsets(nce)));
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

std::string cannot_write(const fs::path& path, const std::string& why) {
  return "write_vtk: cannot write " + path.string() + ": " + why;
}

}  // namespace

void write_vtk(const HostMesh& mesh, const fs::path& path, std::span<const VtkField> fields) {
  const Contents contents = collect_contents(mesh, fields);

  fs::path tmp = path;
  tmp += ".tmp";
  std::error_code ignored;
  {
    errno = 0;
    std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
    if (!file) {
      const int err = errno;
      throw std::runtime_error(
          cannot_write(path, err != 0 ? std::strerror(err) : "the file cannot be opened"));
    }
    try {
      Sink out(file);
      emit(out, mesh, contents);
      file.close();
      if (file.fail()) throw std::runtime_error(cannot_write(path, "the write failed"));
    } catch (...) {
      file.close();
      fs::remove(tmp, ignored);
      throw;
    }
  }
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) {
    fs::remove(tmp, ignored);
    throw std::runtime_error(cannot_write(path, ec.message()));
  }
}

void write_vtk(const HostMesh& mesh, const fs::path& path, std::span<const std::string> fields) {
  std::vector<VtkField> selection;
  selection.reserve(fields.size());
  for (const std::string& n : fields) selection.push_back(VtkField{n, std::nullopt});
  write_vtk(mesh, path, std::span<const VtkField>(selection));
}

}  // namespace soma
