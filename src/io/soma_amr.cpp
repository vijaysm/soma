// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// AMR checkpoints in the SOMA format: the hierarchy's metadata written once,
// each field's valid points as one column per level, each rank's boxes being
// a run of the curve; read back by runs of rows on any number of ranks.

#include "soma/io/soma_amr.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "hdf5.hpp"
#include "soma/fields/data_type.hpp"
#include "soma_image.hpp"
#include "soma_layout.hpp"

namespace soma {

namespace {

namespace fs = std::filesystem;
using h5::Id;

constexpr const char* writer = "write_soma";
constexpr const char* reader = "read_soma_amr";

// The words of the root attribute `amr`.
enum Word : std::size_t {
  w_version,
  w_dimension,
  w_parts,
  w_levels,
  w_fields,
  w_ratio,
  w_blocking_factor,
  w_max_box,
  w_max_level,
  w_nesting,
  w_curve,
  w_periodic,
  w_num_words
};

std::string level_path(int l) { return "amr/levels/l" + std::to_string(l); }
std::string field_path(std::size_t j) { return "amr/fields/f" + std::to_string(j); }

// Where point x of component c of a box lives in its grown storage.
std::int64_t place(const amr::Box& grown, const amr::IntVect& x, int c) {
  const std::int64_t ni = grown.length(0), nj = grown.length(1);
  return (x[0] - grown.lo(0)) + ni * ((x[1] - grown.lo(1)) + nj * (x[2] - grown.lo(2))) + grown.num_pts() * c;
}

// The valid points of the field's boxes, box after box, each i fastest, then
// j, k, then the component.
template <class T>
std::vector<std::byte> valid_points(const amr::LevelField<T>& f) {
  const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f.view().data);
  std::vector<T> out;
  for (std::size_t b = 0; b < f.num_boxes(); ++b) {
    const amr::Box& v = f.valid(b);
    for (int c = 0; c < f.ncomp(); ++c) {
      for (std::int64_t q = 0; q < v.num_pts(); ++q) {
        out.push_back(h(static_cast<std::size_t>(f.offset(b) + place(f.grown(b), amr::detail::point_of(v, q), c))));
      }
    }
  }
  std::vector<std::byte> bytes(out.size() * sizeof(T));
  if (!out.empty()) std::memcpy(bytes.data(), out.data(), bytes.size());
  return bytes;
}

std::uint64_t mix(std::uint64_t h, std::uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}

std::uint64_t mix(std::uint64_t h, const std::string& s) {
  for (const char ch : s) h = mix(h, static_cast<unsigned char>(ch));
  return mix(h, s.size());
}

// Why this rank cannot write the hierarchy and fields, or nothing; adds what
// every rank must agree on to `hash`.
std::string check(const Comm& comm, const amr::Hierarchy& h, const std::vector<AmrFieldRef>& fields,
                  const SomaWriteOptions& options, std::uint64_t& hash) {
  if (options.compression < 0 || options.compression > 9) {
    return "compression is a deflate level in [0, 9], not " + std::to_string(options.compression);
  }
  if (options.compression > 0 && H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0) return "this HDF5 has no deflate filter";
  if (std::string e = soma_file::access_error(options.hints, options.aggregators, comm.size()); !e.empty()) return e;
  if (h.rank() != comm.rank() || (h.comm() ? h.comm()->size() : 1) != comm.size()) {
    return "the hierarchy is not over the ranks of the communicator";
  }
  for (int l = 0; l < h.num_levels(); ++l) {
    const auto& owner = h.distribution(l).owner;
    if (!std::is_sorted(owner.begin(), owner.end())) return "level " + std::to_string(l) + " is not cut along the curve";
  }
  std::set<std::string> names;
  for (const AmrFieldRef& f : fields) {
    if (f.name.empty() || !names.insert(f.name).second) return "field names must be distinct and not empty";
    hash = mix(mix(hash, f.name), f.levels.index());
    std::string error;
    std::visit(
        [&](const auto* u) {
          if (u == nullptr || static_cast<int>(u->size()) != h.num_levels()) {
            error = "field " + f.name + " does not have a level per level of the hierarchy";
            return;
          }
          for (std::size_t l = 0; l < u->size(); ++l) {
            const auto& lf = (*u)[l];
            if (lf.boxes().boxes() != h.boxes(static_cast<int>(l)).boxes() ||
                lf.distribution().owner != h.distribution(static_cast<int>(l)).owner || lf.rank() != h.rank() ||
                !(lf.type() == (*u)[0].type()) || lf.ncomp() != (*u)[0].ncomp() || lf.ghosts() != (*u)[0].ghosts()) {
              error = "level " + std::to_string(l) + " of field " + f.name + " does not match the hierarchy";
              return;
            }
          }
          hash = mix(mix(mix(hash, (*u)[0].type().nodal), static_cast<std::uint64_t>((*u)[0].ncomp())),
                     static_cast<std::uint64_t>((*u)[0].ghosts()));
        },
        f.levels);
    if (!error.empty()) return error;
  }
  return "";
}

soma_file::Image lay_out(const Comm& comm, const amr::Hierarchy& h, const std::vector<AmrFieldRef>& fields) {
  soma_file::Image img;
  const amr::Geometry& g = h.geometry();
  const amr::HierarchyOptions& o = h.options();
  img.groups = {"amr", "amr/levels", "amr/fields"};
  std::int64_t per = 0;
  for (int d = 0; d < 3; ++d) per |= g.periodic[d] ? std::int64_t{1} << d : 0;
  img.ints("", "amr",
           {soma_file::format_version, g.domain.dim(), comm.size(), h.num_levels(), static_cast<std::int64_t>(fields.size()),
            o.ratio, o.blocking_factor, o.max_box, o.max_level, o.nesting, static_cast<std::int64_t>(o.curve), per});
  const amr::Box& d = g.domain;
  img.ints("amr", "domain", {d.lo(0), d.lo(1), d.lo(2), d.hi(0), d.hi(1), d.hi(2)});
  {
    const std::array<double, 7> x{g.lo[0], g.lo[1], g.lo[2], g.hi[0], g.hi[1], g.hi[2], o.efficiency};
    auto& ds = img.single("amr/geometry", DataType::f64, 7);
    ds.data.resize(sizeof x);
    std::memcpy(ds.data.data(), x.data(), sizeof x);
  }
  for (int l = 0; l < h.num_levels(); ++l) {
    std::vector<std::int32_t> boxes;
    for (const amr::Box& b : h.boxes(l).boxes()) {
      for (const int v : {b.lo(0), b.lo(1), b.lo(2), b.hi(0), b.hi(1), b.hi(2)}) boxes.push_back(v);
    }
    auto& ds = img.single(level_path(l), DataType::i32, 6);
    ds.data.resize(boxes.size() * sizeof(std::int32_t));
    if (!boxes.empty()) std::memcpy(ds.data.data(), boxes.data(), ds.data.size());
  }
  for (std::size_t j = 0; j < fields.size(); ++j) {
    const AmrFieldRef& f = fields[j];
    img.groups.push_back(field_path(j));
    img.text(field_path(j), "name", f.name);
    std::visit(
        [&]<class T>(const amr::MultiLevelField<T>* u) {
          const auto& first = (*u)[0];
          img.ints(field_path(j), "meta",
                   {static_cast<std::int64_t>(data_type_of<T>()), first.type().nodal, first.ncomp(), first.ghosts()});
          for (std::size_t l = 0; l < u->size(); ++l) {
            img.column(field_path(j) + "/l" + std::to_string(l), data_type_of<T>()).data = valid_points((*u)[l]);
          }
        },
        f.levels);
  }
  return img;
}

// Runs f; a failure on any rank throws std::runtime_error on every rank,
// naming the path, the failing ones with their own cause.
template <class F>
void agreed(const Comm& comm, const fs::path& path, F&& f) {
  std::string cause;
  try {
    f();
  } catch (const std::bad_alloc&) {
    cause = "out of memory";
  } catch (const std::exception& e) {
    cause = e.what();
  }
  if (comm.allreduce(static_cast<std::int32_t>(cause.empty() ? 0 : 1), Op::max) != 0) {
    throw std::runtime_error(std::string(reader) + ": " + path.string() + ": " +
                             (cause.empty() ? "failed on another rank" : cause));
  }
}

std::vector<std::int64_t> ints(hid_t obj, const char* name, const std::string& where) {
  if (H5Aexists(obj, name) <= 0) throw std::runtime_error(where + " has no attribute " + name);
  const Id a = h5::check(H5Aopen(obj, name, H5P_DEFAULT), std::string("open the attribute ") + name);
  const Id space = h5::check(H5Aget_space(a.get()), "get the dataspace of an attribute");
  std::vector<std::int64_t> out(static_cast<std::size_t>(H5Sget_simple_extent_npoints(space.get())));
  if (!out.empty()) h5::check(H5Aread(a.get(), H5T_NATIVE_INT64, out.data()), std::string("read ") + name);
  return out;
}

std::string text(hid_t obj, const char* name, const std::string& where) {
  if (H5Aexists(obj, name) <= 0) throw std::runtime_error(where + " has no attribute " + name);
  const Id a = h5::check(H5Aopen(obj, name, H5P_DEFAULT), std::string("open the attribute ") + name);
  const Id type = h5::check(H5Aget_type(a.get()), "get the type of an attribute");
  std::string out(H5Tget_size(type.get()), '\0');
  h5::check(H5Aread(a.get(), type.get(), out.data()), std::string("read ") + name);
  out.resize(std::strlen(out.c_str()));
  return out;
}

// Rows [first, first + count) of a dataset of `width` columns, which must
// hold `rows` rows.
template <class T>
std::vector<T> rows(hid_t root, const std::string& path, std::uint64_t first, std::uint64_t count, std::uint64_t rows,
                    std::size_t width) {
  if (!h5::exists(root, path)) throw std::runtime_error("the dataset /soma/" + path + " is missing");
  const Id ds = h5::open_dataset(root, path);
  const auto e = h5::extent(ds.get(), path);
  if (e.empty() || e[0] != rows || (e.size() > 1 ? e[1] : 1) != width) {
    throw std::runtime_error("the dataset /soma/" + path + " does not have the rows its boxes need");
  }
  std::vector<T> out(static_cast<std::size_t>(count) * width);
  h5::read_rows(ds.get(), soma_file::memory_type(data_type_of<T>()), first, count, out.data(), "/soma/" + path);
  return out;
}

// A field's column of a level: whether the file has it, its extent and
// where its rows are.
struct Column {
  bool exists = false;
  std::vector<std::uint64_t> extent;
  soma_file::Location at;
};

struct Header {
  std::vector<std::int64_t> words, domain;
  std::vector<double> geometry;
  std::vector<amr::BoxList> boxes;
  struct Field {
    std::string name;
    DataType type = DataType::f64;
    amr::IndexType index{};
    int ncomp = 1, ghosts = 0;
  };
  std::vector<Field> fields;
  std::vector<std::vector<Column>> columns;  // by field, then level
};

template <class T>
void put(std::vector<std::byte>& out, const std::vector<T>& v) {
  const std::uint64_t n = v.size();
  const std::size_t at = out.size();
  out.resize(at + sizeof n + n * sizeof(T));
  std::memcpy(out.data() + at, &n, sizeof n);
  if (n > 0) std::memcpy(out.data() + at + sizeof n, v.data(), n * sizeof(T));
}

template <class T>
std::vector<T> take(std::span<const std::byte> in, std::size_t& at) {
  std::uint64_t n = 0;
  if (at + sizeof n > in.size()) throw std::logic_error("a header was cut short");
  std::memcpy(&n, in.data() + at, sizeof n);
  at += sizeof n;
  if (n > (in.size() - at) / sizeof(T)) throw std::logic_error("a header was cut short");
  std::vector<T> v(static_cast<std::size_t>(n));
  if (n > 0) std::memcpy(v.data(), in.data() + at, v.size() * sizeof(T));
  at += v.size() * sizeof(T);
  return v;
}

std::vector<std::byte> pack(const Header& x) {
  std::vector<std::byte> out;
  put(out, x.words);
  put(out, x.domain);
  put(out, x.geometry);
  put(out, std::vector<std::uint64_t>{x.boxes.size()});
  for (const amr::BoxList& list : x.boxes) {
    std::vector<std::int32_t> w;
    for (const amr::Box& b : list) {
      for (const int v : {b.lo(0), b.lo(1), b.lo(2), b.hi(0), b.hi(1), b.hi(2)}) w.push_back(v);
    }
    put(out, w);
  }
  put(out, std::vector<std::uint64_t>{x.fields.size()});
  for (std::size_t j = 0; j < x.fields.size(); ++j) {
    const Header::Field& f = x.fields[j];
    put(out, std::vector<char>(f.name.begin(), f.name.end()));
    put(out, std::vector<std::int64_t>{static_cast<std::int64_t>(f.type), f.index.nodal, f.ncomp, f.ghosts});
    for (const Column& c : x.columns[j]) {
      put(out, std::vector<std::uint64_t>{c.exists ? 1U : 0U, c.at.raw ? 1U : 0U, c.at.at});
      put(out, c.extent);
    }
  }
  return out;
}

Header unpack(std::span<const std::byte> in) {
  Header x;
  std::size_t at = 0;
  x.words = take<std::int64_t>(in, at);
  x.domain = take<std::int64_t>(in, at);
  x.geometry = take<double>(in, at);
  if (x.words.size() != w_num_words) throw std::logic_error("a header was cut short");
  const int dim = static_cast<int>(x.words[w_dimension]);
  const auto levels = take<std::uint64_t>(in, at).at(0);
  for (std::uint64_t l = 0; l < levels; ++l) {
    const auto w = take<std::int32_t>(in, at);
    amr::BoxList list;
    for (std::size_t b = 0; b + 5 < w.size(); b += 6) {
      list.push_back(
          amr::Box(amr::IntVect{{w[b], w[b + 1], w[b + 2]}}, amr::IntVect{{w[b + 3], w[b + 4], w[b + 5]}}, dim));
    }
    x.boxes.push_back(std::move(list));
  }
  const auto nfields = take<std::uint64_t>(in, at).at(0);
  for (std::uint64_t j = 0; j < nfields; ++j) {
    const auto name = take<char>(in, at);
    const auto m = take<std::int64_t>(in, at);
    if (m.size() != 4) throw std::logic_error("a header was cut short");
    x.fields.push_back({std::string(name.begin(), name.end()), static_cast<DataType>(m[0]),
                        amr::IndexType{static_cast<std::uint8_t>(m[1])}, static_cast<int>(m[2]),
                        static_cast<int>(m[3])});
    std::vector<Column>& columns = x.columns.emplace_back();
    for (std::uint64_t l = 0; l < levels; ++l) {
      const auto c = take<std::uint64_t>(in, at);
      if (c.size() != 3) throw std::logic_error("a header was cut short");
      columns.push_back({c[0] != 0, take<std::uint64_t>(in, at), {c[1] != 0, c[2]}});
    }
  }
  return x;
}

Header read_header(hid_t root, const std::string& where) {
  Header x;
  if (H5Aexists(root, "amr") <= 0) {
    if (H5Aexists(root, "meta") > 0) throw std::runtime_error("it is a SOMA mesh file, not an AMR checkpoint (read_soma reads it)");
    throw std::runtime_error("it is not an AMR checkpoint");
  }
  x.words = ints(root, "amr", where);
  if (x.words.size() != w_num_words) throw std::runtime_error("its attribute amr does not have " + std::to_string(w_num_words) + " words");
  if (x.words[w_version] < 1 || x.words[w_version] > soma_file::format_version) {
    throw std::runtime_error("format_version " + std::to_string(x.words[w_version]) + " is not one this build reads");
  }
  const std::int64_t dim = x.words[w_dimension], nlevels = x.words[w_levels], nfields = x.words[w_fields];
  if ((dim != 2 && dim != 3) || nlevels < 1 || nfields < 0) throw std::runtime_error("its attribute amr is not valid");
  const Id amr_group = h5::open_group(root, "amr");
  x.domain = ints(amr_group.get(), "domain", where);
  if (x.domain.size() != 6) throw std::runtime_error("its domain does not have six words");
  x.geometry = rows<double>(root, "amr/geometry", 0, 1, 1, 7);
  std::vector<std::vector<std::int32_t>> raw;
  for (int l = 0; l < nlevels; ++l) {
    const std::string path = level_path(l);
    if (!h5::exists(root, path)) throw std::runtime_error("the dataset /soma/" + path + " is missing");
    const Id ds = h5::open_dataset(root, path);
    const auto e = h5::extent(ds.get(), path);
    if (e.size() != 2 || e[1] != 6) throw std::runtime_error("the dataset /soma/" + path + " does not hold boxes");
    const auto v = rows<std::int32_t>(root, path, 0, e[0], e[0], 6);
    amr::BoxList list;
    for (std::size_t b = 0; b < e[0]; ++b) {
      const std::int32_t* w = v.data() + 6 * b;
      list.push_back(amr::Box(amr::IntVect{{w[0], w[1], w[2]}}, amr::IntVect{{w[3], w[4], w[5]}}, static_cast<int>(dim)));
    }
    x.boxes.push_back(std::move(list));
  }
  for (std::int64_t j = 0; j < nfields; ++j) {
    const std::string path = field_path(static_cast<std::size_t>(j));
    if (!h5::exists(root, path)) throw std::runtime_error("the group /soma/" + path + " is missing");
    const Id g = h5::open_group(root, path);
    Header::Field f;
    f.name = text(g.get(), "name", where);
    const auto m = ints(g.get(), "meta", where);
    if (m.size() != 4 || m[0] < 0 || m[0] > static_cast<std::int64_t>(DataType::f64) || m[1] < 0 || m[1] > 7 || m[2] < 1 ||
        m[3] < 0) {
      throw std::runtime_error("the field " + f.name + " has an invalid meta attribute");
    }
    f.type = static_cast<DataType>(m[0]);
    f.index = amr::IndexType{static_cast<std::uint8_t>(m[1])};
    f.ncomp = static_cast<int>(m[2]);
    f.ghosts = static_cast<int>(m[3]);
    x.fields.push_back(f);
    std::vector<Column>& columns = x.columns.emplace_back();
    for (int l = 0; l < nlevels; ++l) {
      const std::string column = path + "/l" + std::to_string(l);
      Column c;
      c.exists = h5::exists(root, column);
      if (c.exists) {
        const Id ds = h5::open_dataset(root, column);
        for (const hsize_t n : h5::extent(ds.get(), column)) c.extent.push_back(n);
        c.at = soma_file::location(ds.get(), soma_file::memory_type(f.type), "/soma/" + column);
      }
      columns.push_back(std::move(c));
    }
  }
  return x;
}

// Rows of every box of a level before box g, for a field of `ncomp`
// components of index type t.
std::vector<std::uint64_t> box_rows(const amr::BoxArray& boxes, amr::IndexType t, int ncomp) {
  std::vector<std::uint64_t> start{0};
  for (const amr::Box& b : boxes.boxes()) start.push_back(start.back() + static_cast<std::uint64_t>(b.convert(t).num_pts() * ncomp));
  return start;
}

// The rows of a level's column this rank reads, [first, last), and the
// column's row count: those of the boxes it holds, a run of the curve.
struct Run {
  std::uint64_t first = 0, last = 0, rows = 0;
};

Run run_of(const amr::Hierarchy& h, int l, const Header::Field& f) {
  const auto start = box_rows(h.boxes(l), f.index, f.ncomp);
  const std::vector<int>& owner = h.distribution(l).owner;
  const auto lo = static_cast<std::size_t>(std::lower_bound(owner.begin(), owner.end(), h.rank()) - owner.begin());
  const auto hi = static_cast<std::size_t>(std::upper_bound(owner.begin(), owner.end(), h.rank()) - owner.begin());
  return {start[lo], start[hi], start.back()};
}

// The rows of every field's columns this rank reads, by field and level, as
// bytes: through HDF5 (opening /soma with `root` when needed), and with
// `raw` the columns stored as read in one collective MPI-IO read.
std::vector<std::vector<std::vector<std::byte>>> field_rows(hid_t file, const std::function<hid_t()>& root,
                                                           const amr::Hierarchy& h, const Header& x, bool raw) {
  std::vector<std::vector<std::vector<std::byte>>> out(x.fields.size());
  std::vector<soma_file::Piece> pieces;
  bool any = false;
  for (std::size_t j = 0; j < x.fields.size(); ++j) {
    const Header::Field& f = x.fields[j];
    const std::size_t size = soma_file::value_size(f.type);
    for (int l = 0; l < h.num_levels(); ++l) {
      const std::string path = field_path(j) + "/l" + std::to_string(l);
      const Column& c = x.columns.at(j).at(static_cast<std::size_t>(l));
      const Run r = run_of(h, l, f);
      if (!c.exists) throw std::runtime_error("the dataset /soma/" + path + " is missing");
      if (c.extent.empty() || c.extent[0] != r.rows || (c.extent.size() > 1 ? c.extent[1] : 1) != 1) {
        throw std::runtime_error("the dataset /soma/" + path + " does not have the rows its boxes need");
      }
      std::vector<std::byte>& bytes = out[j].emplace_back((r.last - r.first) * size);
      if (raw && c.at.raw) {
        any = true;
        if (!bytes.empty()) pieces.push_back({c.at.at + r.first * size, bytes.size(), bytes.data()});
        continue;
      }
      const Id ds = h5::open_dataset(root(), path);
      h5::read_rows(ds.get(), soma_file::memory_type(f.type), r.first, r.last - r.first,
                    bytes.empty() ? nullptr : bytes.data(), "/soma/" + path);
    }
  }
  if (any) soma_file::read_raw(file, std::move(pieces));
  return out;
}

template <class T>
amr::MultiLevelField<T> read_field(const amr::Hierarchy& h, const Header::Field& f,
                                   const std::vector<std::vector<std::byte>>& rows) {
  amr::MultiLevelField<T> u = h.make_field<T>(f.index, f.ncomp, f.ghosts, f.name);
  for (int l = 0; l < h.num_levels(); ++l) {
    amr::LevelField<T>& lf = u[static_cast<std::size_t>(l)];
    const std::vector<std::byte>& bytes = rows[static_cast<std::size_t>(l)];
    std::vector<T> v(bytes.size() / sizeof(T));
    if (!v.empty()) std::memcpy(v.data(), bytes.data(), v.size() * sizeof(T));
    auto host = Kokkos::create_mirror_view(lf.view().data);
    Kokkos::deep_copy(host, T{});
    std::size_t at = 0;
    for (std::size_t b = 0; b < lf.num_boxes(); ++b) {
      const amr::Box& valid = lf.valid(b);
      for (int c = 0; c < lf.ncomp(); ++c) {
        for (std::int64_t q = 0; q < valid.num_pts(); ++q) {
          host(static_cast<std::size_t>(lf.offset(b) + place(lf.grown(b), amr::detail::point_of(valid, q), c))) =
              v.at(at++);
        }
      }
    }
    Kokkos::deep_copy(lf.view().data, host);
  }
  return u;
}

}  // namespace

void write_soma(const Comm& comm, const amr::Hierarchy& hierarchy, const std::vector<AmrFieldRef>& fields,
                const fs::path& path, const SomaWriteOptions& options) {
  std::uint64_t hash = 0;
  std::string error;
  try {
    error = check(comm, hierarchy, fields, options, hash);
  } catch (const std::exception& e) {
    error = e.what();
  }
  // Failures, the fields' names and kinds, and the access options, agreed in
  // one reduction.
  const auto h63 = static_cast<std::int64_t>(hash >> 1);
  const auto a63 = static_cast<std::int64_t>(soma_file::access_hash(options.hints, options.aggregators) >> 1);
  std::array<std::int64_t, 5> agree{error.empty() ? 0 : 1, h63, -h63, a63, -a63};
  comm.allreduce(std::span<std::int64_t>(agree), Op::max);
  if (agree[0] != 0) {
    throw std::invalid_argument(std::string(writer) + ": " + (error.empty() ? "another rank cannot write its part" : error));
  }
  if (agree[1] != -agree[2]) throw std::invalid_argument(std::string(writer) + ": the ranks give different fields");
  if (agree[3] != -agree[4]) {
    throw std::invalid_argument(std::string(writer) + ": the ranks' aggregators or MPI-IO hints differ");
  }
  soma_file::Image img = lay_out(comm, hierarchy, fields);
  soma_file::write_image(comm, path, img, options, writer);
}

AmrCheckpoint read_soma_amr(const Comm& comm, const fs::path& path, const SomaAmrReadOptions& options) {
  const h5::Session quiet;
  // Over more than one rank with MPI-IO the file is opened once, collectively,
  // and rank 0 alone reads the header and where the columns are, given to
  // every rank in one broadcast; the rows are read in one MPI-IO read.
  const bool collective = soma_file::parallel_hdf5() && comm.size() > 1;
  std::optional<Id> file, root;
  std::optional<Id> dxpl;
  Header header;
  const auto open_root = [&] {
    if (H5Lexists(file->get(), soma_file::root, H5P_DEFAULT) <= 0) throw std::runtime_error("it is not a SOMA file");
    root = h5::open_group(file->get(), soma_file::root);
  };
  agreed(comm, path, [&] {
    if (!collective && H5Fis_accessible(path.c_str(), H5P_DEFAULT) <= 0) {
      throw std::runtime_error("cannot be opened as an HDF5 file");
    }
    const Id fapl = soma_file::file_access(comm, collective, collective);
    const hid_t id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl.get());
    if (id < 0) throw std::runtime_error("cannot be opened as an HDF5 file");
    file.emplace(id);
    dxpl = soma_file::transfer(collective);
    if (!collective) {
      open_root();
      header = read_header(root->get(), path.string());
    }
  });
  if (collective) {
    std::vector<std::byte> message{std::byte{0}};
    if (comm.rank() == 0) {
      try {
        open_root();
        const std::vector<std::byte> packed = pack(read_header(root->get(), path.string()));
        message.insert(message.end(), packed.begin(), packed.end());
      } catch (const std::exception& e) {
        const std::string why = e.what();
        message.assign(1, std::byte{1});
        for (const char ch : why) message.push_back(static_cast<std::byte>(ch));
      }
    }
    message = soma_file::broadcast(comm, std::move(message));
    const auto rest = std::span<const std::byte>(message).subspan(1);
    if (message[0] != std::byte{0}) {
      throw std::runtime_error(std::string(reader) + ": " + path.string() + ": " +
                               std::string(reinterpret_cast<const char*>(rest.data()), rest.size()));
    }
    header = unpack(rest);
  }
  // The hierarchy over this communicator; a file whose boxes are not a valid
  // hierarchy is a bad file on every rank alike.
  amr::Geometry geometry;
  geometry.domain = amr::Box(amr::IntVect{{static_cast<int>(header.domain[0]), static_cast<int>(header.domain[1]),
                                           static_cast<int>(header.domain[2])}},
                             amr::IntVect{{static_cast<int>(header.domain[3]), static_cast<int>(header.domain[4]),
                                           static_cast<int>(header.domain[5])}},
                             static_cast<int>(header.words[w_dimension]));
  for (std::size_t d = 0; d < 3; ++d) {
    geometry.lo[d] = header.geometry[d];
    geometry.hi[d] = header.geometry[3 + d];
    geometry.periodic.periodic[d] = ((header.words[w_periodic] >> d) & 1) != 0;
  }
  amr::HierarchyOptions o;
  o.ratio = static_cast<int>(header.words[w_ratio]);
  o.blocking_factor = static_cast<int>(header.words[w_blocking_factor]);
  o.max_box = static_cast<int>(header.words[w_max_box]);
  o.max_level = static_cast<int>(header.words[w_max_level]);
  o.nesting = static_cast<int>(header.words[w_nesting]);
  o.curve = static_cast<amr::Curve>(header.words[w_curve]);
  o.efficiency = header.geometry[6];
  o.cost = options.cost;
  std::vector<amr::BoxList> fine(header.boxes.begin() + 1, header.boxes.end());
  std::optional<amr::Hierarchy> h;
  try {
    h.emplace(comm, geometry, o, fine);
    if (h->boxes(0).boxes() != amr::BoxArray(header.boxes[0], geometry.domain, o.curve).boxes()) {
      throw std::invalid_argument("level 0 is not the domain cut to max_box");
    }
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error(std::string(reader) + ": " + path.string() + ": its hierarchy is not valid: " + e.what());
  }
  std::vector<AmrField> out;
  agreed(comm, path, [&] {
    const h5::CollectiveReads reads(dxpl->get());
    const auto root_id = [&] {
      if (!root) open_root();
      return root->get();
    };
    const auto data = field_rows(file->get(), root_id, *h, header, collective);
    for (std::size_t j = 0; j < header.fields.size(); ++j) {
      const Header::Field& f = header.fields[j];
      AmrField field;
      field.name = f.name;
      dispatch_data_type(f.type, [&]<class T>() { field.levels = read_field<T>(*h, f, data[j]); });
      out.push_back(std::move(field));
    }
  });
  return AmrCheckpoint{std::move(*h), std::move(out)};
}

}  // namespace soma
