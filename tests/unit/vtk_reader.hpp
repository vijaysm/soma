// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_VTK_READER_HPP_
#define SOMA_TESTS_VTK_READER_HPP_

// A strict reader of the legacy ASCII subset that the writer produces
// (UNSTRUCTURED_GRID with POINT_DATA / CELL_DATA FIELD arrays, in the version
// 3.0 or the version 5.1 layout). It follows the
// published format and shares no code with the writer: every count that the
// format ties together is checked, and anything unexpected throws.

#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

namespace soma_test {

// Reals must be finite and normal (or zero): VTK stops reading an array at a
// nan or inf token, and a subnormal one is out of range for its strtod. The
// token is checked as text as well, so that the result does not depend on
// what from_chars accepts.
template <class T>
T parse_number(const std::string& token) {
  if constexpr (std::is_floating_point_v<T>) {
    for (const char c : token) {
      if (c != '-' && c != '+' && c != '.' && c != 'e' && c != 'E' && (c < '0' || c > '9')) {
        throw std::runtime_error("vtk: '" + token + "' is not a finite number");
      }
    }
  }
  T value{};
  const char* first = token.data();
  const char* last = first + token.size();
  const auto r = std::from_chars(first, last, value);
  if (r.ec != std::errc{} || r.ptr != last) {
    throw std::runtime_error("vtk: cannot parse '" + token + "'");
  }
  if constexpr (std::is_floating_point_v<T>) {
    if (!std::isfinite(value) || std::fpclassify(value) == FP_SUBNORMAL) {
      throw std::runtime_error("vtk: '" + token + "' is not a finite normal number");
    }
  }
  return value;
}

// Vertices per cell for the fixed legacy types; 0 for the polygon (any count
// from 3) and -1 for a type the writer never produces.
inline int vtk_vertex_count(int type) {
  switch (type) {
    case 3: return 2;
    case 5: return 3;
    case 7: return 0;
    case 9: return 4;
    case 10: return 4;
    case 12: return 8;
    case 13: return 6;
    case 14: return 5;
    default: return -1;
  }
}

struct VtkArray {
  std::string name;
  int ncomp = 0;
  std::size_t tuples = 0;
  std::string type;
  std::vector<std::string> tokens;

  template <class T>
  std::vector<T> values() const {
    std::vector<T> out;
    out.reserve(tokens.size());
    for (const std::string& t : tokens) out.push_back(parse_number<T>(t));
    return out;
  }
};

struct VtkAttributes {
  bool present = false;
  std::size_t count = 0;
  std::vector<VtkArray> arrays;

  const VtkArray* find(const std::string& name) const {
    for (const VtkArray& a : arrays) {
      if (a.name == name) return &a;
    }
    return nullptr;
  }
};

struct VtkFile {
  std::string text;
  std::string version_line;
  std::string version;  // "3.0" or "5.1"
  std::string title;
  std::string format;
  std::size_t npoints = 0;
  std::string point_type;
  std::vector<double> points;
  std::size_t cells_declared = 0;
  std::size_t cells_size_declared = 0;
  std::vector<std::vector<std::int64_t>> cells;
  std::vector<int> cell_types;
  VtkAttributes point_data;
  VtkAttributes cell_data;
};

inline std::string slurp(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

class VtkTokens {
 public:
  explicit VtkTokens(std::istringstream& in) : in_(in) {}

  std::string next() {
    std::string t;
    if (!(in_ >> t)) throw std::runtime_error("vtk: unexpected end of file");
    return t;
  }
  bool done() {
    in_ >> std::ws;
    return in_.peek() == std::char_traits<char>::eof();
  }
  void expect(const std::string& keyword) {
    const std::string t = next();
    if (t != keyword) throw std::runtime_error("vtk: expected " + keyword + ", found " + t);
  }
  template <class T>
  T number() {
    return parse_number<T>(next());
  }
  // A count or a size, which must not be negative.
  std::size_t count(const std::string& what) {
    const std::string t = next();
    if (parse_number<std::int64_t>(t) < 0) {
      throw std::runtime_error("vtk: " + what + " is negative (" + t + ")");
    }
    return parse_number<std::size_t>(t);
  }

 private:
  std::istringstream& in_;
};

// Every value must parse as its declared type. The 64-bit names differ by version.
inline void check_value(const std::string& type, const std::string& token,
                        const std::string& version) {
  const bool v51 = version == "5.1";
  if (type == (v51 ? "long" : "vtktypeint64") ||
      type == (v51 ? "unsigned_long" : "vtktypeuint64")) {
    throw std::runtime_error("vtk: value type " + type + " in a version " + version + " file");
  }
  if (type == "double") {
    parse_number<double>(token);
  } else if (type == "float") {
    parse_number<float>(token);
  } else if (type == "int") {
    parse_number<std::int32_t>(token);
  } else if (type == "long" || type == "vtktypeint64") {
    parse_number<std::int64_t>(token);
  } else if (type == "unsigned_char") {
    parse_number<std::uint8_t>(token);
  } else if (type == "unsigned_int") {
    parse_number<std::uint32_t>(token);
  } else if (type == "unsigned_long" || type == "vtktypeuint64") {
    parse_number<std::uint64_t>(token);
  } else {
    throw std::runtime_error("vtk: unknown value type " + type);
  }
}

inline VtkAttributes read_attributes(VtkTokens& tok, std::size_t count,
                                     const std::string& version) {
  VtkAttributes out;
  out.present = true;
  out.count = count;
  tok.expect("FIELD");
  tok.expect("FieldData");
  const auto narrays = tok.count("the FieldData array count");
  for (std::size_t a = 0; a < narrays; ++a) {
    VtkArray arr;
    arr.name = tok.next();
    arr.ncomp = tok.number<int>();
    if (arr.ncomp < 1) {
      throw std::runtime_error("vtk: array " + arr.name + " has " + std::to_string(arr.ncomp) +
                               " components");
    }
    arr.tuples = tok.count("the tuple count of array " + arr.name);
    arr.type = tok.next();
    if (arr.tuples != count) {
      throw std::runtime_error("vtk: array " + arr.name + " has " +
                               std::to_string(arr.tuples) + " tuples for " +
                               std::to_string(count) + " entities");
    }
    const std::size_t n = arr.tuples * static_cast<std::size_t>(arr.ncomp);
    for (std::size_t i = 0; i < n; ++i) {
      arr.tokens.push_back(tok.next());
      check_value(arr.type, arr.tokens.back(), version);
    }
    out.arrays.push_back(std::move(arr));
  }
  return out;
}

inline void check_point_id(const VtkFile& f, std::size_t cell, std::int64_t id) {
  if (id < 0 || static_cast<std::size_t>(id) >= f.npoints) {
    throw std::runtime_error("vtk: cell " + std::to_string(cell) + " refers to point " +
                             std::to_string(id) + " of " + std::to_string(f.npoints));
  }
}

// Version 3.0: `count ids...` per cell; CELLS declares the cells and the integers.
inline void read_cell_list(VtkTokens& tok, VtkFile& f) {
  std::size_t used = 0;
  for (std::size_t c = 0; c < f.cells_declared; ++c) {
    const auto k = tok.count("the vertex count of cell " + std::to_string(c));
    std::vector<std::int64_t> cell;
    for (std::size_t i = 0; i < k; ++i) {
      const auto id = tok.number<std::int64_t>();
      check_point_id(f, c, id);
      cell.push_back(id);
    }
    used += 1 + k;
    f.cells.push_back(std::move(cell));
  }
  if (used != f.cells_size_declared) {
    throw std::runtime_error("vtk: CELLS declares " + std::to_string(f.cells_size_declared) +
                             " integers and holds " + std::to_string(used));
  }
}

// Version 5.1: CELLS declares ncells + 1 offsets and the connectivity size.
inline void read_offsets_and_connectivity(VtkTokens& tok, VtkFile& f) {
  if (f.cells_declared < 1) throw std::runtime_error("vtk: CELLS declares no offsets");
  tok.expect("OFFSETS");
  tok.expect("vtktypeint64");
  std::vector<std::size_t> offsets;
  for (std::size_t i = 0; i < f.cells_declared; ++i) {
    offsets.push_back(tok.count("offset " + std::to_string(i)));
    if (i == 0 && offsets[0] != 0) throw std::runtime_error("vtk: the first offset is not 0");
    if (i > 0 && offsets[i] < offsets[i - 1]) {
      throw std::runtime_error("vtk: offsets decrease at " + std::to_string(i));
    }
  }
  if (offsets.back() != f.cells_size_declared) {
    throw std::runtime_error("vtk: the last offset is " + std::to_string(offsets.back()) +
                             " and CELLS declares " + std::to_string(f.cells_size_declared));
  }
  tok.expect("CONNECTIVITY");
  tok.expect("vtktypeint64");
  for (std::size_t c = 0; c + 1 < offsets.size(); ++c) {
    std::vector<std::int64_t> cell;
    for (std::size_t k = offsets[c]; k < offsets[c + 1]; ++k) {
      const auto id = tok.number<std::int64_t>();
      check_point_id(f, c, id);
      cell.push_back(id);
    }
    f.cells.push_back(std::move(cell));
  }
}

inline VtkFile read_vtk_file(const std::filesystem::path& path) {
  VtkFile f;
  f.text = slurp(path);
  if (f.text.empty() || f.text.back() != '\n') {
    throw std::runtime_error("vtk: missing final newline");
  }
  std::istringstream in(f.text);
  std::getline(in, f.version_line);
  std::getline(in, f.title);
  std::getline(in, f.format);
  if (f.version_line == "# vtk DataFile Version 3.0") {
    f.version = "3.0";
  } else if (f.version_line == "# vtk DataFile Version 5.1") {
    f.version = "5.1";
  } else {
    throw std::runtime_error("vtk: unexpected version line '" + f.version_line + "'");
  }
  if (f.title.size() > 256) throw std::runtime_error("vtk: title is longer than 256 characters");
  if (f.format != "ASCII") throw std::runtime_error("vtk: unexpected format '" + f.format + "'");

  VtkTokens tok(in);
  tok.expect("DATASET");
  tok.expect("UNSTRUCTURED_GRID");
  tok.expect("POINTS");
  f.npoints = tok.count("the POINTS count");
  f.point_type = tok.next();
  if (f.point_type != "double") {
    throw std::runtime_error("vtk: unexpected POINTS type '" + f.point_type + "'");
  }
  for (std::size_t i = 0; i < 3 * f.npoints; ++i) f.points.push_back(tok.number<double>());

  tok.expect("CELLS");
  f.cells_declared = tok.count("the CELLS count");
  f.cells_size_declared = tok.count("the CELLS size");
  if (f.version == "5.1") {
    read_offsets_and_connectivity(tok, f);
  } else {
    read_cell_list(tok, f);
  }

  tok.expect("CELL_TYPES");
  const auto ntypes = tok.count("the CELL_TYPES count");
  if (ntypes != f.cells.size()) {
    throw std::runtime_error("vtk: CELL_TYPES count differs from CELLS");
  }
  for (std::size_t c = 0; c < ntypes; ++c) {
    const int type = tok.number<int>();
    const int want = vtk_vertex_count(type);
    const std::size_t have = f.cells[c].size();
    if (want < 0) throw std::runtime_error("vtk: unknown cell type " + std::to_string(type));
    if (want == 0 ? have < 3 : have != static_cast<std::size_t>(want)) {
      throw std::runtime_error("vtk: cell " + std::to_string(c) + " of type " +
                               std::to_string(type) + " has " + std::to_string(have) +
                               " vertices");
    }
    f.cell_types.push_back(type);
  }

  while (!tok.done()) {
    const std::string section = tok.next();
    const auto count = tok.count("the " + section + " count");
    if (section == "POINT_DATA") {
      if (f.point_data.present) throw std::runtime_error("vtk: POINT_DATA appears twice");
      if (count != f.npoints) {
        throw std::runtime_error("vtk: POINT_DATA count differs from POINTS");
      }
      f.point_data = read_attributes(tok, count, f.version);
    } else if (section == "CELL_DATA") {
      if (f.cell_data.present) throw std::runtime_error("vtk: CELL_DATA appears twice");
      if (count != f.cells.size()) {
        throw std::runtime_error("vtk: CELL_DATA count differs from CELLS");
      }
      f.cell_data = read_attributes(tok, count, f.version);
    } else {
      throw std::runtime_error("vtk: unexpected section " + section);
    }
  }
  return f;
}

}  // namespace soma_test

#endif  // SOMA_TESTS_VTK_READER_HPP_
