// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_VTK_READER_HPP_
#define SOMA_TESTS_VTK_READER_HPP_

// A strict reader of the legacy ASCII subset that the writer produces
// (UNSTRUCTURED_GRID with POINT_DATA / CELL_DATA FIELD arrays). It follows the
// published format and shares no code with the writer: every count that the
// format ties together is checked, and anything unexpected throws.

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace soma_test {

template <class T>
T parse_number(const std::string& token) {
  T value{};
  const char* first = token.data();
  const char* last = first + token.size();
  const auto r = std::from_chars(first, last, value);
  if (r.ec != std::errc{} || r.ptr != last) {
    throw std::runtime_error("vtk: cannot parse '" + token + "'");
  }
  return value;
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

 private:
  std::istringstream& in_;
};

inline VtkAttributes read_attributes(VtkTokens& tok, std::size_t count) {
  VtkAttributes out;
  out.present = true;
  out.count = count;
  tok.expect("FIELD");
  tok.next();
  const auto narrays = tok.number<std::size_t>();
  for (std::size_t a = 0; a < narrays; ++a) {
    VtkArray arr;
    arr.name = tok.next();
    arr.ncomp = tok.number<int>();
    arr.tuples = tok.number<std::size_t>();
    arr.type = tok.next();
    if (arr.tuples != count) {
      throw std::runtime_error("vtk: array " + arr.name + " has " + std::to_string(arr.tuples) +
                               " tuples for " + std::to_string(count) + " entities");
    }
    const std::size_t n = arr.tuples * static_cast<std::size_t>(arr.ncomp);
    for (std::size_t i = 0; i < n; ++i) arr.tokens.push_back(tok.next());
    out.arrays.push_back(std::move(arr));
  }
  return out;
}

inline VtkFile read_vtk_file(const std::filesystem::path& path) {
  VtkFile f;
  f.text = slurp(path);
  if (f.text.empty() || f.text.back() != '\n') throw std::runtime_error("vtk: missing final newline");
  std::istringstream in(f.text);
  std::getline(in, f.version_line);
  std::getline(in, f.title);
  std::getline(in, f.format);
  if (f.title.size() > 256) throw std::runtime_error("vtk: title is longer than 256 characters");

  VtkTokens tok(in);
  tok.expect("DATASET");
  tok.expect("UNSTRUCTURED_GRID");
  tok.expect("POINTS");
  f.npoints = tok.number<std::size_t>();
  f.point_type = tok.next();
  for (std::size_t i = 0; i < 3 * f.npoints; ++i) f.points.push_back(tok.number<double>());

  tok.expect("CELLS");
  f.cells_declared = tok.number<std::size_t>();
  f.cells_size_declared = tok.number<std::size_t>();
  std::size_t used = 0;
  for (std::size_t c = 0; c < f.cells_declared; ++c) {
    const auto k = tok.number<std::size_t>();
    std::vector<std::int64_t> cell;
    for (std::size_t i = 0; i < k; ++i) cell.push_back(tok.number<std::int64_t>());
    used += 1 + k;
    f.cells.push_back(std::move(cell));
  }
  if (used != f.cells_size_declared) {
    throw std::runtime_error("vtk: CELLS declares " + std::to_string(f.cells_size_declared) +
                             " integers and holds " + std::to_string(used));
  }

  tok.expect("CELL_TYPES");
  const auto ntypes = tok.number<std::size_t>();
  if (ntypes != f.cells_declared) throw std::runtime_error("vtk: CELL_TYPES count differs from CELLS");
  for (std::size_t c = 0; c < ntypes; ++c) f.cell_types.push_back(tok.number<int>());

  while (!tok.done()) {
    const std::string section = tok.next();
    const auto count = tok.number<std::size_t>();
    if (section == "POINT_DATA") {
      if (f.point_data.present) throw std::runtime_error("vtk: POINT_DATA appears twice");
      if (count != f.npoints) throw std::runtime_error("vtk: POINT_DATA count differs from POINTS");
      f.point_data = read_attributes(tok, count);
    } else if (section == "CELL_DATA") {
      if (f.cell_data.present) throw std::runtime_error("vtk: CELL_DATA appears twice");
      if (count != f.cells_declared) throw std::runtime_error("vtk: CELL_DATA count differs from CELLS");
      f.cell_data = read_attributes(tok, count);
    } else {
      throw std::runtime_error("vtk: unexpected section " + section);
    }
  }
  return f;
}

}  // namespace soma_test

#endif  // SOMA_TESTS_VTK_READER_HPP_
