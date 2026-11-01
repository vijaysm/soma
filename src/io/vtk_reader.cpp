// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/io/vtk_reader.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "soma/core/mesh_builder.hpp"
#include "soma/core/topology.hpp"
#include "soma/fields/data_type.hpp"

namespace soma {
namespace {

namespace fs = std::filesystem;

std::string lower(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return out;
}

std::string upper(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  }
  return out;
}

bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// A token quoted in a message, cut to its first 64 characters.
std::string in_quotes(std::string_view s) {
  constexpr std::size_t most = 64;
  if (s.size() <= most) return "'" + std::string(s) + "'";
  return "'" + std::string(s.substr(0, most)) + "...'";
}

// Whitespace-separated tokens with the line each one starts on.
class Lexer {
 public:
  Lexer(std::string text, const fs::path& path) : text_(std::move(text)), path_(path) {}

  [[noreturn]] void fail(const std::string& what) const { fail_at(line_, what); }
  [[noreturn]] void fail_at(std::size_t line, const std::string& what) const {
    throw std::runtime_error("read_vtk: " + path_.string() + ", line " + std::to_string(line) +
                             ": " + what);
  }

  // The rest of the current line, without its end of line; for the header.
  std::string_view line_text(const char* what) {
    if (pos_ >= text_.size()) fail_at(cur_line_, std::string("the file ends before ") + what);
    line_ = cur_line_;
    const std::size_t end = std::min(text_.find('\n', pos_), text_.size());
    std::string_view s(text_.data() + pos_, end - pos_);
    if (!s.empty() && s.back() == '\r') s.remove_suffix(1);
    pos_ = end;
    if (pos_ < text_.size()) {
      ++pos_;
      ++cur_line_;
    }
    return s;
  }

  bool at_end() {
    skip_space();
    return pos_ >= text_.size();
  }

  std::string_view peek() {
    skip_space();
    std::size_t end = pos_;
    while (end < text_.size() && !is_space(text_[end])) ++end;
    return {text_.data() + pos_, end - pos_};
  }

  std::string_view next(const std::string& what) {
    skip_space();
    if (pos_ >= text_.size()) fail_at(cur_line_, "the file ends where " + what + " was expected");
    line_ = cur_line_;
    const std::size_t start = pos_;
    while (pos_ < text_.size() && !is_space(text_[pos_])) ++pos_;
    return {text_.data() + start, pos_ - start};
  }

  void expect(std::string_view keyword) {
    const std::string_view t = next(std::string(keyword));
    if (lower(t) != lower(keyword)) {
      fail("expected " + std::string(keyword) + ", found " + in_quotes(t));
    }
  }

  // A count or a size: a non-negative integer that `per` values of at least
  // two bytes each could follow in what is left of the file.
  std::size_t count(const std::string& what, std::size_t per = 0) {
    const std::string_view t = next(what);
    std::int64_t v = 0;
    const auto r = std::from_chars(t.data(), t.data() + t.size(), v);
    if (r.ec == std::errc::result_out_of_range) fail(what + " is out of range: " + in_quotes(t));
    if (r.ec != std::errc{} || r.ptr != t.data() + t.size()) {
      fail(what + " is not an integer: " + in_quotes(t));
    }
    if (v < 0) fail(what + " is negative: " + in_quotes(t));
    const auto n = static_cast<std::size_t>(v);
    if (per != 0) check_room(n, per, what + " (" + std::string(t) + ")");
    return n;
  }

  // Throws unless `n` x `per` values, of two bytes each but the last, could
  // follow in the rest of the file: no count makes the reader allocate more.
  void check_room(std::size_t n, std::size_t per, const std::string& what) const {
    const std::size_t room = (text_.size() - pos_ + 1) / 2;
    if (n > room / per) fail(what + " needs more values than the rest of the file holds");
  }

  // After METADATA: every line up to and including the next blank one.
  void skip_metadata() {
    std::size_t end = text_.find('\n', pos_);
    pos_ = end == std::string::npos ? text_.size() : end + 1;
    if (end != std::string::npos) ++cur_line_;
    while (pos_ < text_.size()) {
      end = std::min(text_.find('\n', pos_), text_.size());
      const bool blank = std::all_of(text_.begin() + static_cast<std::ptrdiff_t>(pos_),
                                     text_.begin() + static_cast<std::ptrdiff_t>(end), is_space);
      pos_ = end < text_.size() ? end + 1 : end;
      if (end < text_.size()) ++cur_line_;
      if (blank) return;
    }
  }

  void skip_metadata_if_present() {
    if (!at_end() && lower(peek()) == "metadata") {
      next("METADATA");
      skip_metadata();
    }
  }

  std::size_t line() const { return line_; }

 private:
  void skip_space() {
    while (pos_ < text_.size() && is_space(text_[pos_])) {
      if (text_[pos_] == '\n') ++cur_line_;
      ++pos_;
    }
  }

  std::string text_;
  const fs::path& path_;
  std::size_t pos_ = 0;
  std::size_t cur_line_ = 1;
  std::size_t line_ = 1;
};

// A VTK value type, the SOMA type it becomes and the range of its integers.
struct ValueType {
  DataType type = DataType::f64;
  std::int64_t min = 0;
  std::uint64_t max = 0;
};

std::optional<ValueType> value_type(std::string_view vtk) {
  using I8 = std::numeric_limits<std::int8_t>;
  using I16 = std::numeric_limits<std::int16_t>;
  using I32 = std::numeric_limits<std::int32_t>;
  using I64 = std::numeric_limits<std::int64_t>;
  using U8 = std::numeric_limits<std::uint8_t>;
  using U16 = std::numeric_limits<std::uint16_t>;
  using U32 = std::numeric_limits<std::uint32_t>;
  using U64 = std::numeric_limits<std::uint64_t>;
  const auto u = [](auto v) { return static_cast<std::uint64_t>(v); };
  const std::string t = lower(vtk);
  if (t == "double") return ValueType{DataType::f64, 0, 0};
  if (t == "float") return ValueType{DataType::f32, 0, 0};
  if (t == "int") return ValueType{DataType::i32, I32::min(), u(I32::max())};
  if (t == "short") return ValueType{DataType::i32, I16::min(), u(I16::max())};
  if (t == "unsigned_int") return ValueType{DataType::u32, 0, U32::max()};
  if (t == "unsigned_short") return ValueType{DataType::u32, 0, U16::max()};
  if (t == "long" || t == "vtktypeint64" || t == "vtkidtype") {
    return ValueType{DataType::i64, I64::min(), u(I64::max())};
  }
  if (t == "unsigned_long" || t == "vtktypeuint64") return ValueType{DataType::u64, 0, U64::max()};
  if (t == "unsigned_char") return ValueType{DataType::u8, 0, U8::max()};
  // VTK writes a char array, signed where char is, as 0..255: 128 is -128.
  if (t == "char") return ValueType{DataType::i8, I8::min(), U8::max()};
  if (t == "signed_char") return ValueType{DataType::i8, I8::min(), u(I8::max())};
  return std::nullopt;
}

ValueType value_type_or_fail(Lexer& lex, std::string_view vtk, const std::string& of) {
  const auto t = value_type(vtk);
  if (!t) lex.fail("the value type " + in_quotes(vtk) + " of " + of + " is not supported");
  return *t;
}

bool is_integer(DataType t) { return t != DataType::f32 && t != DataType::f64; }

// The value of a real token: nan, inf and -inf are accepted, as VTK writes them.
// The token without a leading '+', which from_chars does not take; empty when a
// second sign follows it.
std::string_view without_plus(std::string_view t) {
  if (t.empty() || t.front() != '+') return t;
  t.remove_prefix(1);
  if (!t.empty() && (t.front() == '-' || t.front() == '+')) return {};
  return t;
}

template <class F>
F parse_real(Lexer& lex, std::string_view t, const std::string& of) {
  const std::string_view s = without_plus(t);
  F v{};
  const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
  if (r.ec == std::errc::result_out_of_range) {
    lex.fail(in_quotes(t) + " in " + of + " is out of range for " +
             (std::is_same_v<F, float> ? "float" : "double"));
  }
  if (r.ec != std::errc{} || r.ptr != s.data() + s.size() || s.empty()) {
    lex.fail("expected a number in " + of + ", found " + in_quotes(t));
  }
  return v;
}

template <class T>
T parse_integer(Lexer& lex, std::string_view t, const ValueType& vt, const std::string& of) {
  const std::string_view s = without_plus(t);
  bool ok = false;
  T out{};
  if (!s.empty() && s.front() == '-') {
    std::int64_t v = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec == std::errc{} && r.ptr == s.data() + s.size()) {
      ok = true;
      if (v < vt.min) lex.fail(in_quotes(t) + " in " + of + " is out of range for its type");
      out = static_cast<T>(v);
    } else if (r.ec == std::errc::result_out_of_range) {
      lex.fail(in_quotes(t) + " in " + of + " is out of range for its type");
    }
  } else {
    std::uint64_t v = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec == std::errc{} && r.ptr == s.data() + s.size()) {
      ok = true;
      if (v > vt.max) lex.fail(in_quotes(t) + " in " + of + " is out of range for its type");
      out = static_cast<T>(v);
    } else if (r.ec == std::errc::result_out_of_range) {
      lex.fail(in_quotes(t) + " in " + of + " is out of range for its type");
    }
  }
  if (!ok) lex.fail("expected an integer in " + of + ", found " + in_quotes(t));
  return out;
}

using Values =
    std::variant<std::vector<std::int8_t>, std::vector<std::uint8_t>, std::vector<std::int32_t>,
                 std::vector<std::uint32_t>, std::vector<std::int64_t>, std::vector<std::uint64_t>,
                 std::vector<float>, std::vector<double>>;

struct Array {
  std::string name;
  DataType type = DataType::f64;
  int ncomp = 1;
  Values values;
  std::size_t line = 0;
};

Array read_values(Lexer& lex, std::string name, const ValueType& vt, int ncomp, std::size_t rows,
                  const std::string& of) {
  Array a;
  a.line = lex.line();
  a.name = std::move(name);
  a.type = vt.type;
  a.ncomp = ncomp;
  lex.check_room(rows, static_cast<std::size_t>(ncomp), "the size of " + of);
  const std::size_t n = rows * static_cast<std::size_t>(ncomp);
  dispatch_data_type(vt.type, [&]<class T>() {
    std::vector<T> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      const std::string_view t = lex.next("a value of " + of);
      if constexpr (std::is_floating_point_v<T>) {
        v.push_back(parse_real<T>(lex, t, of));
      } else {
        v.push_back(parse_integer<T>(lex, t, vt, of));
      }
    }
    a.values = std::move(v);
  });
  lex.skip_metadata_if_present();
  return a;
}

// VTK escapes characters of array names as %XX.
std::string decode_name(Lexer& lex, std::string_view s) {
  std::string out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '%') {
      out.push_back(s[i]);
      continue;
    }
    unsigned v = 0;
    const auto r = i + 2 < s.size() ? std::from_chars(s.data() + i + 1, s.data() + i + 3, v, 16)
                                    : std::from_chars_result{nullptr, std::errc::invalid_argument};
    if (r.ec != std::errc{} || r.ptr != s.data() + i + 3) {
      lex.fail("the array name " + in_quotes(s) + " holds a malformed % escape");
    }
    if (v == 0) lex.fail("the array name " + in_quotes(s) + " holds the escape %00");
    out.push_back(static_cast<char>(v));
    i += 2;
  }
  return out;
}

int positive_int(Lexer& lex, const std::string& what) {
  const std::size_t n = lex.count(what);
  if (n < 1 || n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    lex.fail(what + " must be positive, found " + std::to_string(n));
  }
  return static_cast<int>(n);
}

// FIELD name n, then n arrays `name ncomp ntuples type` and their values.
// `rows` is the tuple count every array must have, or nullopt for dataset field data.
std::vector<Array> read_field(Lexer& lex, std::optional<std::size_t> rows, const char* section) {
  lex.next("the FIELD name");
  const std::size_t narrays = lex.count("the FIELD array count");
  std::vector<Array> out;
  for (std::size_t k = 0; k < narrays; ++k) {
    const std::string_view raw = lex.next("an array name");
    if (raw == "NULL_ARRAY") continue;
    const std::string name = decode_name(lex, raw);
    const std::string of = "array " + in_quotes(name) + " of " + section;
    const int ncomp = positive_int(lex, "the component count of " + of);
    const std::size_t line = lex.line();
    const std::size_t tuples =
        lex.count("the tuple count of " + of, static_cast<std::size_t>(ncomp));
    const std::string_view type = lex.next("the value type of " + of);
    if (rows && tuples != *rows) {
      lex.fail_at(line, of + " has " + std::to_string(tuples) + " tuples for " +
                            std::to_string(*rows) + " entities");
    }
    const ValueType vt = value_type_or_fail(lex, type, of);
    out.push_back(read_values(lex, name, vt, ncomp, tuples, of));
  }
  return out;
}

struct Cell {
  int code = 0;
  std::size_t first = 0;  // into the connectivity
  std::size_t size = 0;
};

struct Grid {
  int major = 0;
  std::size_t npoints = 0;
  bool have_points = false;
  std::vector<Real> xyz;
  bool have_cells = false;
  bool have_types = false;
  std::vector<std::size_t> offsets;  // ncells + 1
  std::vector<std::int64_t> conn;
  std::vector<int> codes;
  std::vector<Array> point_arrays;
  std::vector<Array> cell_arrays;
  std::size_t cell_data_line = 0;
};

void read_points(Lexer& lex, Grid& g) {
  if (g.have_points) lex.fail("POINTS appears twice");
  g.npoints = lex.count("the POINTS count", 3);
  const std::string_view type = lex.next("the POINTS value type");
  const std::string t = lower(type);
  if (t != "double" && t != "float") {
    lex.fail("POINTS of type " + in_quotes(type) + " are not supported; only float and double");
  }
  g.xyz.reserve(3 * g.npoints);
  for (std::size_t i = 0; i < 3 * g.npoints; ++i) {
    const std::string_view tok = lex.next("a coordinate of POINTS");
    const double v = t == "double" ? parse_real<double>(lex, tok, "POINTS")
                                   : static_cast<double>(parse_real<float>(lex, tok, "POINTS"));
    if (!std::isfinite(v)) {
      lex.fail("point " + std::to_string(i / 3) + " has the non-finite coordinate " +
               in_quotes(tok));
    }
    g.xyz.push_back(v);
  }
  g.have_points = true;
  lex.skip_metadata_if_present();
}

const ValueType any_id{DataType::i64, std::numeric_limits<std::int64_t>::min(),
                      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())};

std::int64_t read_id(Lexer& lex, const std::string& of, const ValueType& vt = any_id) {
  return parse_integer<std::int64_t>(lex, lex.next("a value of " + of), vt, of);
}

// The range of the integer type of OFFSETS or CONNECTIVITY, which may also be
// one of VTK's fixed-width names; values beyond int64 are out of range anyway.
ValueType integer_type(Lexer& lex, std::string_view type, const char* of) {
  using I64 = std::numeric_limits<std::int64_t>;
  const auto fixed = [](std::int64_t min, std::uint64_t max) {
    return ValueType{DataType::i64, min, std::min(max, static_cast<std::uint64_t>(I64::max()))};
  };
  const std::string name = lower(type);
  if (name == "vtktypeint8") return fixed(-128, 127);
  if (name == "vtktypeint16") return fixed(-32768, 32767);
  if (name == "vtktypeint32") return fixed(-2147483647 - 1, 2147483647);
  if (name == "vtktypeint64") return fixed(I64::min(), static_cast<std::uint64_t>(I64::max()));
  if (name == "vtktypeuint8") return fixed(0, 255);
  if (name == "vtktypeuint16") return fixed(0, 65535);
  if (name == "vtktypeuint32") return fixed(0, 4294967295u);
  if (name == "vtktypeuint64") return fixed(0, std::numeric_limits<std::uint64_t>::max());
  const auto t = value_type(type);
  if (!t || !is_integer(t->type)) {
    lex.fail(std::string(of) + " has the value type " + in_quotes(type) +
             "; it needs an integer type");
  }
  return fixed(t->min, t->max);
}

void read_cells(Lexer& lex, Grid& g) {
  if (g.have_cells) lex.fail("CELLS appears twice");
  g.have_cells = true;
  if (g.major >= 5) {
    const std::size_t noffsets = lex.count("the CELLS offset count", 1);
    const std::size_t nconn = lex.count("the CELLS connectivity size", 1);
    lex.expect("OFFSETS");
    const ValueType offset_type = integer_type(lex, lex.next("the OFFSETS value type"), "OFFSETS");
    g.offsets.reserve(noffsets);
    for (std::size_t i = 0; i < noffsets; ++i) {
      const std::int64_t v = read_id(lex, "OFFSETS", offset_type);
      if (v < 0) lex.fail("offset " + std::to_string(i) + " is negative: " + std::to_string(v));
      const auto o = static_cast<std::size_t>(v);
      if (i == 0 && o != 0) lex.fail("the first offset is " + std::to_string(o) + ", not 0");
      if (i > 0 && o < g.offsets.back()) lex.fail("offset " + std::to_string(i) + " decreases");
      g.offsets.push_back(o);
    }
    lex.skip_metadata_if_present();
    if (g.offsets.empty()) g.offsets.push_back(0);
    if (g.offsets.back() != nconn) {
      lex.fail("the last offset is " + std::to_string(g.offsets.back()) +
               " but CELLS declares a connectivity size of " + std::to_string(nconn));
    }
    lex.expect("CONNECTIVITY");
    const ValueType conn_type =
        integer_type(lex, lex.next("the CONNECTIVITY value type"), "CONNECTIVITY");
    g.conn.reserve(nconn);
    for (std::size_t k = 0; k < nconn; ++k) {
      g.conn.push_back(read_id(lex, "CONNECTIVITY", conn_type));
    }
    lex.skip_metadata_if_present();
  } else {
    const std::size_t ncells = lex.count("the CELLS count", 2);
    const std::size_t size = lex.count("the CELLS size", 1);
    g.offsets.reserve(ncells + 1);
    g.offsets.push_back(0);
    std::size_t used = 0;
    for (std::size_t c = 0; c < ncells; ++c) {
      const std::size_t line = lex.line();
      const std::size_t k = lex.count("the vertex count of cell " + std::to_string(c));
      if (used + 1 + k > size) {
        lex.fail_at(line, "cell " + std::to_string(c) + " runs past the CELLS size " +
                              std::to_string(size));
      }
      for (std::size_t i = 0; i < k; ++i) g.conn.push_back(read_id(lex, "CELLS"));
      used += 1 + k;
      g.offsets.push_back(g.conn.size());
    }
    if (used != size) {
      lex.fail("CELLS declares a size of " + std::to_string(size) + " and holds " +
               std::to_string(used));
    }
    lex.skip_metadata_if_present();
  }
  for (std::size_t k = 0; k < g.conn.size(); ++k) {
    if (g.conn[k] < 0 || static_cast<std::uint64_t>(g.conn[k]) >= g.npoints) {
      lex.fail("a cell refers to point " + std::to_string(g.conn[k]) + " of " +
               std::to_string(g.npoints) + (g.have_points ? "" : " (CELLS before POINTS)"));
    }
  }
}

std::optional<Topology> topology_of(int code) {
  switch (code) {
    case 3: return Topology::edge;
    case 5: return Topology::tri;
    case 7: return Topology::polygon;
    case 9: return Topology::quad;
    case 10: return Topology::tet;
    case 12: return Topology::hex;
    case 13: return Topology::prism;
    case 14: return Topology::pyramid;
    default: return std::nullopt;
  }
}

void read_cell_types(Lexer& lex, Grid& g) {
  if (!g.have_cells) lex.fail("CELL_TYPES comes before CELLS");
  if (g.have_types) lex.fail("CELL_TYPES appears twice");
  g.have_types = true;
  const std::size_t ncells = g.offsets.size() - 1;
  const std::size_t n = lex.count("the CELL_TYPES count", 1);
  if (n != ncells) {
    lex.fail("CELL_TYPES declares " + std::to_string(n) + " cells and CELLS holds " +
             std::to_string(ncells));
  }
  g.codes.reserve(n);
  for (std::size_t c = 0; c < n; ++c) {
    const std::int64_t v = read_id(lex, "CELL_TYPES");
    const auto t = v >= 0 && v <= 255 ? topology_of(static_cast<int>(v)) : std::nullopt;
    if (!t) {
      lex.fail("cell " + std::to_string(c) + " has the cell type code " + std::to_string(v) +
               ", which is unknown or not supported (supported: 3, 5, 7, 9, 10, 12, 13, 14)");
    }
    const std::size_t have = g.offsets[c + 1] - g.offsets[c];
    const int want = num_vertices(*t);
    if (want == 0 ? have < 3 : have != static_cast<std::size_t>(want)) {
      lex.fail("cell " + std::to_string(c) + " of cell type code " + std::to_string(v) + " has " +
               std::to_string(have) + " vertices");
    }
    g.codes.push_back(static_cast<int>(v));
  }
  lex.skip_metadata_if_present();
}

// The attribute arrays of one POINT_DATA or CELL_DATA section.
std::vector<Array> read_attributes(Lexer& lex, std::size_t rows, const char* section) {
  std::vector<Array> out;
  while (!lex.at_end()) {
    const std::string key = lower(lex.peek());
    if (key == "point_data" || key == "cell_data") break;
    lex.next("a keyword");
    const std::size_t key_line = lex.line();
    if (key == "field") {
      for (Array& a : read_field(lex, rows, section)) out.push_back(std::move(a));
    } else if (key == "metadata") {
      lex.skip_metadata();
    } else if (key == "lookup_table") {
      lex.next("the LOOKUP_TABLE name");
      const std::size_t n = lex.count("the LOOKUP_TABLE size", 4);
      for (std::size_t i = 0; i < 4 * n; ++i) lex.next("a LOOKUP_TABLE value");
    } else if (key == "color_scalars") {
      lex.next("the COLOR_SCALARS name");
      const std::size_t n = lex.count("the COLOR_SCALARS component count");
      if (n != 0) lex.check_room(rows, n, "the size of COLOR_SCALARS");
      for (std::size_t i = 0; i < rows * n; ++i) lex.next("a COLOR_SCALARS value");
    } else if (key == "scalars" || key == "vectors" || key == "normals" || key == "tensors" ||
               key == "tensors6" || key == "texture_coordinates") {
      const std::string name = decode_name(lex, lex.next(upper(key) + " name"));
      const std::string of = upper(key) + " " + in_quotes(name) + " of " + section;
      int ncomp = key == "vectors" || key == "normals" ? 3
                  : key == "tensors"                   ? 9
                  : key == "tensors6"                  ? 6
                                                       : 1;
      if (key == "texture_coordinates") ncomp = positive_int(lex, "the dimension of " + of);
      const std::string_view type = lex.next("the value type of " + of);
      const ValueType vt = value_type_or_fail(lex, type, of);
      if (key == "scalars") {
        if (lower(lex.peek()) != "lookup_table") {
          ncomp = positive_int(lex, "the component count of " + of);
        }
        lex.expect("LOOKUP_TABLE");
        lex.next("the LOOKUP_TABLE name of " + of);
      }
      out.push_back(read_values(lex, name, vt, ncomp, rows, of));
    } else {
      lex.fail_at(key_line,
                  "the section " + in_quotes(key) + " in " + section + " is not supported");
    }
  }
  for (std::size_t i = 0; i < out.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      if (out[i].name == out[j].name) {
        lex.fail_at(out[i].line,
                    "the array name " + in_quotes(out[i].name) + " appears twice in " + section);
      }
    }
  }
  return out;
}

Grid parse(Lexer& lex) {
  Grid g;
  const std::string_view head = lex.line_text("the version line");
  constexpr std::string_view prefix = "# vtk DataFile Version ";
  if (head.substr(0, prefix.size()) != prefix) {
    lex.fail("the first line is not '# vtk DataFile Version x.y': " + in_quotes(head));
  }
  std::string_view version = head.substr(prefix.size());
  while (!version.empty() && is_space(version.back())) version.remove_suffix(1);
  if (version == "5.1") {
    g.major = 5;
  } else if (version.size() == 3 && version[1] == '.' && version[0] >= '2' && version[0] <= '4' &&
             version[2] >= '0' && version[2] <= '9') {
    g.major = version[0] - '0';
  } else {
    lex.fail("version " + in_quotes(version) + " is not supported; 2.0, 3.0, 4.x and 5.1 are");
  }
  lex.line_text("the title line");
  std::string_view format = lex.line_text("the format line");
  while (!format.empty() && is_space(format.back())) format.remove_suffix(1);
  while (!format.empty() && is_space(format.front())) format.remove_prefix(1);
  const std::string f = lower(format);
  if (f == "binary") lex.fail("BINARY files are not supported; only ASCII");
  if (f != "ascii") lex.fail("the format line is " + in_quotes(format) + ", not ASCII");

  lex.expect("DATASET");
  const std::string_view dataset = lex.next("the DATASET type");
  if (lower(dataset) != "unstructured_grid") {
    lex.fail("DATASET " + std::string(dataset) + " is not supported; only UNSTRUCTURED_GRID");
  }

  bool have_point_data = false;
  bool have_cell_data = false;
  while (!lex.at_end()) {
    const std::string key = lower(lex.next("a keyword"));
    if (key == "field" && !have_point_data && !have_cell_data) {
      read_field(lex, std::nullopt, "the dataset");
    } else if (key == "points" && !have_point_data && !have_cell_data) {
      read_points(lex, g);
    } else if (key == "cells" && !have_point_data && !have_cell_data) {
      read_cells(lex, g);
    } else if (key == "cell_types" && !have_point_data && !have_cell_data) {
      read_cell_types(lex, g);
    } else if (key == "metadata") {
      lex.skip_metadata();
    } else if (key == "point_data" || key == "cell_data") {
      const bool points = key == "point_data";
      const char* section = points ? "POINT_DATA" : "CELL_DATA";
      if (!g.have_points) lex.fail(std::string(section) + " comes before POINTS");
      if (!points && !g.have_types) lex.fail("CELL_DATA comes before CELL_TYPES");
      bool& seen = points ? have_point_data : have_cell_data;
      if (seen) lex.fail(std::string(section) + " appears twice");
      seen = true;
      if (!points) g.cell_data_line = lex.line();
      const std::size_t want = points ? g.npoints : g.codes.size();
      const std::size_t n = lex.count(std::string("the ") + section + " count");
      if (n != want) {
        lex.fail(std::string(section) + " declares " + std::to_string(n) +
                 " entities and the file has " + std::to_string(want) +
                 (points ? " points" : " cells"));
      }
      (points ? g.point_arrays : g.cell_arrays) = read_attributes(lex, n, section);
    } else {
      lex.fail("unexpected keyword " + in_quotes(key));
    }
  }
  if (!g.have_points) lex.fail("the file has no POINTS");
  if (g.have_cells && !g.have_types) lex.fail("the file has CELLS but no CELL_TYPES");
  if (g.codes.empty()) lex.fail("the file has no cells; a mesh needs at least one");
  return g;
}

template <class T>
void fill_tag(HostMesh& mesh, const Array& a, int dim, const std::vector<std::size_t>& row_of) {
  const auto& src = std::get<std::vector<T>>(a.values);
  auto f = mesh.fields().add_tag<T>(a.name, dim, a.ncomp);
  const auto v = f.view();
  const auto ncomp = static_cast<std::size_t>(a.ncomp);
  for (std::size_t r = 0; r < row_of.size(); ++r) {
    for (std::size_t c = 0; c < ncomp; ++c) v(row_of[r], c) = src[r * ncomp + c];
  }
}

// row_of[i] is the local row of the entity with gid i + 1.
std::vector<std::size_t> rows_by_gid(const HostMesh& mesh, int dim) {
  const auto gid = mesh.gid(dim);
  std::vector<std::size_t> row_of(gid.extent(0));
  for (std::size_t r = 0; r < gid.extent(0); ++r) row_of[static_cast<std::size_t>(gid(r) - 1)] = r;
  return row_of;
}

}  // namespace

HostMesh read_vtk(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("read_vtk: cannot open " + path.string());
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (in.bad()) throw std::runtime_error("read_vtk: cannot read " + path.string());
  Lexer lex(std::move(text), path);
  const Grid g = parse(lex);

  const std::size_t ncells = g.codes.size();
  std::vector<int> cell_dim(ncells);
  int dim = 0;
  for (std::size_t c = 0; c < ncells; ++c) {
    cell_dim[c] = dimension(*topology_of(g.codes[c]));
    dim = std::max(dim, cell_dim[c]);
  }
  if (g.npoints > static_cast<std::size_t>(std::numeric_limits<LO>::max()) ||
      g.conn.size() > static_cast<std::size_t>(std::numeric_limits<Offset>::max())) {
    throw std::runtime_error("read_vtk: " + path.string() +
                             ": the mesh is too large for the LO and Offset types of this build");
  }
  if (!g.cell_arrays.empty() &&
      std::any_of(cell_dim.begin(), cell_dim.end(), [&](int d) { return d != dim; })) {
    throw std::runtime_error("read_vtk: " + path.string() + ", line " +
                             std::to_string(g.cell_data_line) +
                             ": CELL_DATA on cells of several dimensions is not supported; the "
                             "cells span dimensions up to " +
                             std::to_string(dim));
  }

  MeshBuilder b(dim);
  {
    std::vector<GO> gids(g.npoints);
    for (std::size_t i = 0; i < g.npoints; ++i) gids[i] = static_cast<GO>(i + 1);
    b.add_vertices(gids, g.xyz);
  }
  struct Block {
    std::vector<GO> gids;
    std::vector<Offset> offsets{0};
    std::vector<GO> vertices;
  };
  std::array<Block, 16> blocks;  // by cell type code
  std::array<GO, 4> next{1, 1, 1, 1};
  for (std::size_t c = 0; c < ncells; ++c) {
    Block& k = blocks[static_cast<std::size_t>(g.codes[c])];
    k.gids.push_back(next[static_cast<std::size_t>(cell_dim[c])]++);
    for (std::size_t i = g.offsets[c]; i < g.offsets[c + 1]; ++i)
      k.vertices.push_back(g.conn[i] + 1);
    k.offsets.push_back(static_cast<Offset>(k.vertices.size()));
  }
  for (std::size_t code = 0; code < blocks.size(); ++code) {
    const Block& k = blocks[code];
    if (k.gids.empty()) continue;
    const Topology t = *topology_of(static_cast<int>(code));
    if (t == Topology::polygon) {
      b.add_polygons(k.gids, k.offsets, k.vertices);
    } else {
      b.add_elements(t, k.gids, k.vertices);
    }
  }
  HostMesh mesh = std::move(b).build(Ordering::file);

  const auto attach = [&](const std::vector<Array>& arrays, int d) {
    if (arrays.empty()) return;
    const std::vector<std::size_t> row_of = rows_by_gid(mesh, d);
    for (const Array& a : arrays) {
      dispatch_data_type(a.type, [&]<class T>() { fill_tag<T>(mesh, a, d, row_of); });
    }
  };
  attach(g.point_arrays, 0);
  attach(g.cell_arrays, dim);
  return mesh;
}

}  // namespace soma
