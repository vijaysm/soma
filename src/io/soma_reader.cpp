// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The SOMA format's reader (design §22.4): each rank reads a contiguous
// block of parts, owners send their shared entities to the ranks holding
// copies, and each rank builds its part with the sharing the file gives.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../parallel/records.hpp"
#include "hdf5.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_values.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/parallel/partition.hpp"
#include "soma/topology/build.hpp"
#include "soma_layout.hpp"

namespace soma {

namespace {

namespace fs = std::filesystem;
using detail::Batch;
using h5::Id;
using soma_file::dim_group;
using soma_file::field_group;
using soma_file::value_group;

constexpr const char* who = "read_soma";

[[noreturn]] void bad_file(const std::string& why) { throw std::runtime_error(std::string(who) + ": " + why); }

// Reads values from a byte stream, checking that they are there.
struct Cursor {
  std::span<const std::byte> in;
  std::size_t at = 0;
  template <class T>
  T take() {
    if (at + sizeof(T) > in.size()) throw std::logic_error("read_soma: a record was cut short");
    T v;
    std::memcpy(&v, in.data() + at, sizeof(T));
    at += sizeof(T);
    return v;
  }
  std::span<const std::byte> bytes(std::size_t n) {
    if (at + n > in.size()) throw std::logic_error("read_soma: a record was cut short");
    const auto out = in.subspan(at, n);
    at += n;
    return out;
  }
};

template <class T>
void append(std::vector<std::byte>& out, const T& v) {
  const std::size_t at = out.size();
  out.resize(at + sizeof(T));
  std::memcpy(out.data() + at, &v, sizeof(T));
}

void append(std::vector<std::byte>& out, std::span<const std::byte> v) { out.insert(out.end(), v.begin(), v.end()); }

// The parts rank r reads: q with floor(qP/N) = r, [first, last).
struct Block {
  std::uint64_t first = 0, last = 0;
};

Block block_of(int r, int p, std::uint64_t n) {
  const auto ceil_div = [](std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; };
  const auto rr = static_cast<std::uint64_t>(r), pp = static_cast<std::uint64_t>(p);
  return {ceil_div(rr * n, pp), ceil_div((rr + 1) * n, pp)};
}

int rank_of(std::uint64_t q, int p, std::uint64_t n) {
  return static_cast<int>(q * static_cast<std::uint64_t>(p) / n);
}

// The file and its metadata, read alike on every rank (collectively with
// MPI-IO).
class File {
 public:
  File(const Comm& comm, const fs::path& path, bool collective) {
    const Id fapl = soma_file::file_access(comm, collective);
    file_ = h5::check(H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl.get()), "open " + path.string());
    if (H5Lexists(file_.get(), soma_file::root, H5P_DEFAULT) <= 0) bad_file(path.string() + " is not a SOMA file");
    root_ = h5::open_group(file_.get(), soma_file::root);
    dxpl_ = soma_file::transfer(collective);
  }

  [[nodiscard]] hid_t root() const { return root_.get(); }
  [[nodiscard]] hid_t dxpl() const { return dxpl_.get(); }
  [[nodiscard]] bool has(const std::string& path) const { return h5::exists(root_.get(), path); }

  [[nodiscard]] std::vector<std::int64_t> ints(const std::string& object, const char* name) const {
    const Id a = attribute(object, name);
    const Id space = h5::check(H5Aget_space(a.get()), "get the dataspace of an attribute");
    std::vector<std::int64_t> out(static_cast<std::size_t>(H5Sget_simple_extent_npoints(space.get())));
    if (!out.empty()) h5::check(H5Aread(a.get(), H5T_NATIVE_INT64, out.data()), std::string("read ") + name);
    return out;
  }

  [[nodiscard]] std::optional<std::vector<std::byte>> bytes(const std::string& object, const char* name) const {
    const Id obj = open(object);
    if (H5Aexists(obj.get(), name) <= 0) return std::nullopt;
    const Id a = h5::check(H5Aopen(obj.get(), name, H5P_DEFAULT), std::string("open the attribute ") + name);
    const Id space = h5::check(H5Aget_space(a.get()), "get the dataspace of an attribute");
    std::vector<std::byte> out(static_cast<std::size_t>(H5Sget_simple_extent_npoints(space.get())));
    if (!out.empty()) h5::check(H5Aread(a.get(), H5T_NATIVE_UINT8, out.data()), std::string("read ") + name);
    return out;
  }

  [[nodiscard]] std::string text(const std::string& object, const char* name) const {
    const Id a = attribute(object, name);
    const Id type = h5::check(H5Aget_type(a.get()), "get the type of an attribute");
    std::string out(H5Tget_size(type.get()), '\0');
    h5::check(H5Aread(a.get(), type.get(), out.data()), std::string("read ") + name);
    out.resize(std::strlen(out.c_str()));
    return out;
  }

 private:
  [[nodiscard]] Id open(const std::string& object) const {
    return h5::check(H5Oopen(root_.get(), object.empty() ? "." : object.c_str(), H5P_DEFAULT), "open /soma/" + object);
  }
  [[nodiscard]] Id attribute(const std::string& object, const char* name) const {
    const Id obj = open(object);
    return h5::check(H5Aopen(obj.get(), name, H5P_DEFAULT),
                     std::string("open the attribute ") + name + " of /soma/" + object);
  }

  Id file_, root_, dxpl_;
};

// Rows [first, first + rows) of a dataset of `width` columns into `out`.
struct Read {
  std::string path;
  DataType type = DataType::u8;
  std::size_t width = 1;
  std::uint64_t first = 0, rows = 0;
  std::vector<std::byte>* out = nullptr;
};

// Every read in one transfer: the datasets are opened and their extents
// checked on every rank first, so that a bad file throws on every rank
// before any data moves.
void read_all(const Comm& comm, const File& f, std::vector<Read>& reads) {
  std::vector<Id> ds;
  std::string error;
  for (Read& r : reads) {
    ds.push_back(h5::open_dataset(f.root(), r.path));
    const auto dims = h5::extent(ds.back().get(), r.path);
    const bool fits = dims.size() == 2 && dims[1] == r.width && r.first <= dims[0] && r.rows <= dims[0] - r.first;
    if (!fits && error.empty()) error = "/soma/" + r.path + " does not hold the rows its part table gives";
  }
  if (comm.allreduce(static_cast<std::int32_t>(error.empty() ? 0 : 1), Op::max) != 0) {
    bad_file(error.empty() ? "another rank's datasets do not match the part table" : error);
  }
  std::vector<Id> spaces, memories;
  std::vector<hid_t> ids, types, mem, file;
  std::vector<void*> buffers;
  for (std::size_t k = 0; k < reads.size(); ++k) {
    Read& r = reads[k];
    r.out->resize(r.rows * r.width * soma_file::value_size(r.type));
    Id space = h5::check(H5Dget_space(ds[k].get()), "get a dataspace");
    Id memory;
    if (r.rows == 0) {
      h5::check(H5Sselect_none(space.get()), "select nothing");
      memory = h5::check(H5Scopy(space.get()), "copy a dataspace");
      h5::check(H5Sselect_none(memory.get()), "select nothing");
    } else {
      const std::array<hsize_t, 2> start{r.first, 0}, count{r.rows, r.width};
      h5::check(H5Sselect_hyperslab(space.get(), H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr),
                "select rows");
      memory = h5::check(H5Screate_simple(2, count.data(), nullptr), "create a memory dataspace");
    }
    ids.push_back(ds[k].get());
    types.push_back(soma_file::memory_type(r.type));
    mem.push_back(memory.get());
    file.push_back(space.get());
    buffers.push_back(r.out->empty() ? static_cast<void*>(&r) : r.out->data());
    spaces.push_back(std::move(space));
    memories.push_back(std::move(memory));
  }
  if (ids.empty()) return;
#if H5_VERSION_GE(1, 14, 0)
  h5::check(H5Dread_multi(ids.size(), ids.data(), types.data(), mem.data(), file.data(), f.dxpl(), buffers.data()),
            "read the datasets");
#else
  for (std::size_t k = 0; k < ids.size(); ++k) {
    h5::check(H5Dread(ids[k], types[k], mem[k], file[k], f.dxpl(), buffers[k]), "read a dataset");
  }
#endif
}

template <class T>
std::vector<T> as(const std::vector<std::byte>& b) {
  std::vector<T> out(b.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), b.data(), out.size() * sizeof(T));
  return out;
}

// What the root says.
struct Meta {
  int dim = 0;
  std::uint64_t parts = 0;
  std::array<bool, 4> written{}, input{}, nodes{};
  std::uint32_t graphs = 0;
  std::int64_t sets = 0, fields = 0, values = 0;
};

Meta read_meta(const File& f) {
  const auto m = f.ints("", "meta");
  if (m.size() < 11) bad_file("/soma has no complete meta attribute");
  if (m[0] > soma_file::format_version || m[0] < 1) {
    bad_file("format_version " + std::to_string(m[0]) + " is not one this build reads (up to " +
             std::to_string(soma_file::format_version) + ")");
  }
  Meta x;
  x.dim = static_cast<int>(m[1]);
  if (x.dim < 0 || x.dim > 3 || m[2] < 1) bad_file("/soma's meta attribute is invalid");
  x.parts = static_cast<std::uint64_t>(m[2]);
  for (std::size_t d = 0; d < 4; ++d) {
    x.written[d] = ((m[3] >> d) & 1) != 0;
    x.input[d] = ((m[4] >> d) & 1) != 0;
    x.nodes[d] = ((m[5] >> d) & 1) != 0;
  }
  x.graphs = static_cast<std::uint32_t>(m[6]);
  x.sets = m[7];
  x.fields = m[8];
  x.values = m[9];
  return x;
}

// A field's options, as its group's attributes give them.
struct FieldMeta;
bool on(const FieldMeta& f, int d);

struct FieldMeta {
  std::string name;
  DataType type = DataType::u8;
  int ncomp = 1;
  bool variable = false, handle = false;
  Layout layout{};
  TagFormat format{};
  std::optional<std::vector<std::byte>> def, glob;
  std::array<bool, 4> full{};
  std::vector<int> dims;
  bool counted = false;
  std::size_t unit = 0;  // bytes of a row (fixed) or of a value (variable length)
};

bool on(const FieldMeta& f, int d) { return std::find(f.dims.begin(), f.dims.end(), d) != f.dims.end(); }

FieldMeta read_field_meta(const File& f, std::size_t i) {
  const std::string g = field_group(i);
  FieldMeta x;
  x.name = f.text(g, "name");
  const auto m = f.ints(g, "meta");
  if (m.size() < 17) bad_file("/soma/" + g + " has no complete meta attribute");
  x.type = static_cast<DataType>(m[0]);
  x.ncomp = static_cast<int>(m[1]);
  x.variable = m[2] != 0;
  x.handle = m[3] != 0;
  x.layout.kind = static_cast<Layout::Kind>(m[4]);
  x.layout.family = static_cast<NodeFamily>(m[5]);
  x.layout.order = static_cast<int>(m[6]);
  x.layout.ordering = static_cast<NodeOrdering>(m[7]);
  x.layout.serendipity = m[8] != 0;
  x.layout.quad_rule = static_cast<int>(m[9]);
  x.format.storage_class = static_cast<int>(m[10]);
  x.format.kind = static_cast<decltype(x.format.kind)>(m[11]);
  x.format.nbits = static_cast<int>(m[12]);
  if (m[13] != 0) x.def = f.bytes(g, "default").value_or(std::vector<std::byte>{});
  if (m[14] != 0) x.glob = f.bytes(g, "global").value_or(std::vector<std::byte>{});
  for (std::size_t d = 0; d < 4; ++d) x.full[d] = ((m[15] >> d) & 1) != 0;
  const auto n = static_cast<std::size_t>(m[16]);
  if (m.size() != 17 + n) bad_file("/soma/" + g + " has an invalid meta attribute");
  for (std::size_t k = 0; k < n; ++k) x.dims.push_back(static_cast<int>(m[17 + k]));
  x.counted = x.variable || x.layout.kind != Layout::Kind::per_entity;
  x.unit = soma_file::value_size(x.type) * static_cast<std::size_t>(x.variable ? 1 : x.ncomp);
  return x;
}

// The values of one field on one dimension, by row of `Rows`: the number of
// units (rows or values) and where they start in `bytes`; -1 without one.
struct FieldRows {
  std::vector<std::int64_t> count, at;
  std::vector<std::byte> bytes;

  void resize(std::size_t n) {
    count.resize(n, -1);
    at.resize(n, -1);
  }
  void set(std::size_t row, std::int64_t units, std::span<const std::byte> v) {
    count[row] = units;
    at[row] = static_cast<std::int64_t>(bytes.size());
    bytes.insert(bytes.end(), v.begin(), v.end());
  }
};

// A set membership: set, entity row, the part that owned the entity, and
// the position in an ordered set (-1 for others).
struct Membership {
  std::int64_t set;
  std::size_t row;
  std::uint64_t part;
  std::int64_t pos;
};

// The entities of one dimension this rank holds: its parts' owned rows,
// then the copies their owners sent. `others`: the other ranks holding
// each.
struct Rows {
  std::vector<GO> gid;
  std::vector<double> xyz;
  std::vector<std::uint8_t> topo;
  std::vector<std::uint32_t> ncorner, nnode;
  std::vector<std::size_t> corner_at, node_at;
  std::vector<GO> corners, nodes;
  std::vector<std::vector<int>> others;
  std::vector<std::uint64_t> part;
  std::vector<FieldRows> fields;  // per field; empty where the field has no values here
  std::vector<Membership> members;
  std::size_t owned = 0;

  [[nodiscard]] std::size_t size() const { return gid.size(); }
};

// Everything this rank read.
struct Parts {
  Meta meta;
  Block block;
  std::vector<FieldMeta> fields;
  std::array<Rows, 4> rows;
  // Sets, the same on every rank.
  std::vector<GO> set_gid;
  std::vector<std::uint8_t> set_flags;
  std::vector<std::vector<GO>> children, parents, set_members;
  std::vector<std::tuple<std::int64_t, std::int64_t, GO>> entries;  // (set, position, gid), from part 0
  // Values on sets per field: (set, units, bytes).
  std::vector<std::vector<std::tuple<LO, std::int64_t, std::vector<std::byte>>>> on_sets;
};

std::vector<std::vector<GO>> csr(const std::vector<std::int64_t>& counts, const std::vector<std::int64_t>& values,
                                 const std::string& what) {
  std::vector<std::vector<GO>> out;
  std::size_t at = 0;
  for (const std::int64_t n : counts) {
    if (n < 0 || at + static_cast<std::size_t>(n) > values.size()) bad_file(what + " is inconsistent");
    out.emplace_back(values.begin() + static_cast<std::ptrdiff_t>(at),
                     values.begin() + static_cast<std::ptrdiff_t>(at + static_cast<std::size_t>(n)));
    at += static_cast<std::size_t>(n);
  }
  return out;
}

Parts read_file(const Comm& comm, const File& f) {
  Parts in;
  in.meta = read_meta(f);
  const Meta& m = in.meta;
  in.block = block_of(comm.rank(), comm.size(), m.parts);
  const Block& b = in.block;

  // The part table rows of this block's parts and the one after.
  const auto columns_text = f.text("parts/offsets", "columns");
  std::vector<std::string> columns;
  for (std::size_t at = 0; at < columns_text.size();) {
    const std::size_t nl = columns_text.find('\n', at);
    columns.push_back(columns_text.substr(at, nl - at));
    at = nl == std::string::npos ? columns_text.size() : nl + 1;
  }
  const std::size_t k_count = columns.size();
  std::vector<std::byte> table_bytes;
  {
    std::vector<Read> one{{"parts/offsets", DataType::u64, k_count, b.first, b.last - b.first + 1, &table_bytes}};
    read_all(comm, f, one);
  }
  const auto table = as<std::uint64_t>(table_bytes);
  std::map<std::string, std::size_t> column_of;
  for (std::size_t k = 0; k < k_count; ++k) column_of[columns[k]] = k;
  const auto start = [&](std::uint64_t q, const std::string& c) {
    const auto it = column_of.find(c);
    if (it == column_of.end()) bad_file("the part table names no column " + c);
    return table[(q - b.first) * k_count + it->second];
  };

  for (std::int64_t i = 0; i < m.fields; ++i) in.fields.push_back(read_field_meta(f, static_cast<std::size_t>(i)));

  // Every dataset this rank reads, in one transfer.
  std::vector<Read> reads;
  std::map<std::string, std::vector<std::byte>> data;
  const auto column = [&](const std::string& path, DataType t, std::size_t width) {
    reads.push_back({path, t, width, start(b.first, path), start(b.last, path) - start(b.first, path), &data[path]});
  };
  const auto whole = [&](const std::string& path, DataType t, std::size_t width) {
    const Id ds = h5::open_dataset(f.root(), path);
    const auto dims = h5::extent(ds.get(), path);
    reads.push_back({path, t, width, 0, dims.empty() ? 0 : dims[0], &data[path]});
  };
  for (int d = 0; d <= m.dim; ++d) {
    if (!m.written[static_cast<std::size_t>(d)]) continue;
    const std::string g = dim_group(d);
    column(g + "/gid", DataType::i64, 1);
    if (d == 0) {
      column(g + "/coords", DataType::f64, 3);
    } else {
      column(g + "/topology", DataType::u8, 1);
      column(g + "/counts", DataType::u32, 1);
      column(g + "/conn", DataType::i64, 1);
      if (m.nodes[static_cast<std::size_t>(d)]) {
        column(g + "/node_counts", DataType::u32, 1);
        column(g + "/nodes", DataType::i64, 1);
      }
    }
    column(g + "/shared/index", DataType::i64, 1);
    column(g + "/shared/counts", DataType::u32, 1);
    column(g + "/shared/parts", DataType::i32, 1);
  }
  for (std::size_t i = 0; i < in.fields.size(); ++i) {
    const FieldMeta& fm = in.fields[i];
    for (const int d : fm.dims) {
      const std::string dg = field_group(i) + (d == set_dim ? "/sets" : "/" + dim_group(d));
      const std::size_t width = fm.variable ? 1 : static_cast<std::size_t>(fm.ncomp);
      if (d == set_dim) {
        whole(dg + "/domain", DataType::i64, 1);
        whole(dg + "/values", fm.type, width);
        if (fm.counted) whole(dg + "/counts", DataType::i64, 1);
        continue;
      }
      column(dg + "/full", DataType::u8, 1);
      column(dg + "/domain", DataType::i64, 1);
      column(dg + "/values", fm.type, width);
      if (fm.counted) column(dg + "/counts", DataType::i64, 1);
    }
  }
  for (const char* s : {"sets/gid", "sets/children_counts", "sets/children", "sets/parents_counts", "sets/parents",
                        "sets/set_members_counts", "sets/set_members"}) {
    whole(s, DataType::i64, 1);
  }
  whole("sets/flags", DataType::u8, 1);
  whole("sets/entries", DataType::i64, 3);
  column("sets/members", DataType::i64, 4);
  read_all(comm, f, reads);

  // Entities, by dimension.
  for (int d = 0; d <= m.dim; ++d) {
    const auto dd = static_cast<std::size_t>(d);
    if (!m.written[dd]) continue;
    const std::string g = dim_group(d);
    Rows& r = in.rows[dd];
    const auto gid = as<std::int64_t>(data[g + "/gid"]);
    r.gid.assign(gid.begin(), gid.end());
    r.owned = r.gid.size();
    r.others.resize(r.size());
    r.part.resize(r.size());
    for (std::uint64_t q = b.first; q < b.last; ++q) {
      const auto lo = start(q, g + "/gid") - start(b.first, g + "/gid");
      const auto hi = start(q + 1, g + "/gid") - start(b.first, g + "/gid");
      for (auto i = lo; i < hi; ++i) r.part[i] = q;
    }
    if (d == 0) {
      r.xyz = as<double>(data[g + "/coords"]);
    } else {
      r.topo = as<std::uint8_t>(data[g + "/topology"]);
      r.ncorner = as<std::uint32_t>(data[g + "/counts"]);
      const auto conn = as<std::int64_t>(data[g + "/conn"]);
      r.corners.assign(conn.begin(), conn.end());
      std::size_t at = 0;
      for (const std::uint32_t n : r.ncorner) {
        r.corner_at.push_back(at);
        at += n;
      }
      if (at != r.corners.size()) bad_file("/soma/" + g + "/conn does not match its counts");
      if (m.nodes[dd]) {
        r.nnode = as<std::uint32_t>(data[g + "/node_counts"]);
        const auto nodes = as<std::int64_t>(data[g + "/nodes"]);
        r.nodes.assign(nodes.begin(), nodes.end());
        at = 0;
        for (const std::uint32_t n : r.nnode) {
          r.node_at.push_back(at);
          at += n;
        }
        if (at != r.nodes.size()) bad_file("/soma/" + g + "/nodes does not match its counts");
      } else {
        r.nnode.assign(r.size(), 0);
        r.node_at.assign(r.size(), 0);
      }
    }
    // Sharing: each owned shared entity's other parts, as ranks other than
    // this one.
    const auto index = as<std::int64_t>(data[g + "/shared/index"]);
    const auto counts = as<std::uint32_t>(data[g + "/shared/counts"]);
    const auto parts = as<std::int32_t>(data[g + "/shared/parts"]);
    std::size_t at = 0, rec = 0;
    for (std::uint64_t q = b.first; q < b.last; ++q) {
      const auto base = start(q, g + "/gid") - start(b.first, g + "/gid");
      const auto n = start(q + 1, g + "/shared/index") - start(q, g + "/shared/index");
      for (std::uint64_t k = 0; k < n; ++k, ++rec) {
        const auto row = base + static_cast<std::uint64_t>(index[rec]);
        if (row >= r.size() || at + counts[rec] > parts.size()) bad_file("/soma/" + g + "/shared is inconsistent");
        auto& o = r.others[row];
        for (std::uint32_t j = 0; j < counts[rec]; ++j, ++at) {
          const auto s = parts[at];
          if (s < 0 || static_cast<std::uint64_t>(s) >= m.parts) bad_file("/soma/" + g + "/shared names no part");
          const int rr = rank_of(static_cast<std::uint64_t>(s), comm.size(), m.parts);
          if (rr != comm.rank()) o.push_back(rr);
        }
        std::sort(o.begin(), o.end());
        o.erase(std::unique(o.begin(), o.end()), o.end());
      }
    }
    r.fields.resize(in.fields.size());
  }

  // Field values on entities, by row.
  for (std::size_t i = 0; i < in.fields.size(); ++i) {
    const FieldMeta& fm = in.fields[i];
    in.on_sets.emplace_back();
    for (const int d : fm.dims) {
      const std::string dg = field_group(i) + (d == set_dim ? "/sets" : "/" + dim_group(d));
      const auto values = data[dg + "/values"];
      const auto counts = fm.counted ? as<std::int64_t>(data[dg + "/counts"]) : std::vector<std::int64_t>{};
      std::size_t vat = 0, cat = 0;
      const auto next = [&](std::int64_t& units) {
        units = fm.counted ? (cat < counts.size() ? counts[cat++] : -1) : 1;
        const auto n = static_cast<std::size_t>(units) * fm.unit;
        if (units < 0 || vat + n > values.size()) bad_file("/soma/" + dg + " is inconsistent");
        const auto out = std::span<const std::byte>(values).subspan(vat, n);
        vat += n;
        return out;
      };
      if (d == set_dim) {
        for (const std::int64_t s : as<std::int64_t>(data[dg + "/domain"])) {
          std::int64_t units = 0;
          const auto v = next(units);
          in.on_sets[i].emplace_back(static_cast<LO>(s), units, std::vector<std::byte>(v.begin(), v.end()));
        }
        continue;
      }
      if (d > m.dim || !m.written[static_cast<std::size_t>(d)]) bad_file("field values on a dimension not written");
      Rows& r = in.rows[static_cast<std::size_t>(d)];
      FieldRows& fr = r.fields[i];
      fr.resize(r.size());
      const auto full = as<std::uint8_t>(data[dg + "/full"]);
      const auto domain = as<std::int64_t>(data[dg + "/domain"]);
      std::size_t dat = 0;
      const std::string g = dim_group(d);
      for (std::uint64_t q = b.first; q < b.last; ++q) {
        const auto base = start(q, g + "/gid") - start(b.first, g + "/gid");
        const auto end = start(q + 1, g + "/gid") - start(b.first, g + "/gid");
        const auto put = [&](std::uint64_t row) {
          std::int64_t units = 0;
          const auto v = next(units);
          if (row >= end) bad_file("/soma/" + dg + "/domain names no entity of its part");
          fr.set(row, units, v);
        };
        if (full[q - b.first] != 0) {
          for (auto row = base; row < end; ++row) put(row);
        } else {
          const auto n = start(q + 1, dg + "/domain") - start(q, dg + "/domain");
          for (std::uint64_t k = 0; k < n; ++k) put(base + static_cast<std::uint64_t>(domain[dat++]));
        }
      }
    }
  }

  // Sets.
  const auto sg = as<std::int64_t>(data["sets/gid"]);
  in.set_gid.assign(sg.begin(), sg.end());
  in.set_flags = as<std::uint8_t>(data["sets/flags"]);
  in.children = csr(as<std::int64_t>(data["sets/children_counts"]), as<std::int64_t>(data["sets/children"]),
                    "/soma/sets/children");
  in.parents = csr(as<std::int64_t>(data["sets/parents_counts"]), as<std::int64_t>(data["sets/parents"]),
                   "/soma/sets/parents");
  in.set_members = csr(as<std::int64_t>(data["sets/set_members_counts"]), as<std::int64_t>(data["sets/set_members"]),
                       "/soma/sets/set_members");
  const auto nsets = in.set_gid.size();
  if (in.set_flags.size() != nsets || in.children.size() != nsets || in.parents.size() != nsets ||
      in.set_members.size() != nsets) {
    bad_file("/soma/sets is inconsistent");
  }
  const auto entries = as<std::int64_t>(data["sets/entries"]);
  for (std::size_t k = 0; k + 2 < entries.size(); k += 3) in.entries.emplace_back(entries[k], entries[k + 1], entries[k + 2]);
  const auto members = as<std::int64_t>(data["sets/members"]);
  std::size_t at = 0;
  for (std::uint64_t q = b.first; q < b.last; ++q) {
    const auto n = start(q + 1, "sets/members") - start(q, "sets/members");
    for (std::uint64_t k = 0; k < n; ++k, at += 4) {
      const std::int64_t set = members[at], d = members[at + 1], index = members[at + 2], pos = members[at + 3];
      if (set < 0 || static_cast<std::size_t>(set) >= nsets || d < 0 || d > m.dim ||
          !m.written[static_cast<std::size_t>(d)]) {
        bad_file("/soma/sets/members is inconsistent");
      }
      const std::string g = dim_group(static_cast<int>(d));
      const auto row = start(q, g + "/gid") - start(b.first, g + "/gid") + static_cast<std::uint64_t>(index);
      Rows& r = in.rows[static_cast<std::size_t>(d)];
      if (index < 0 || row >= r.size()) bad_file("/soma/sets/members names no entity of its part");
      r.members.push_back({set, row, q, pos});
    }
  }
  return in;
}

// Owners send each shared entity to the other ranks holding it: its row,
// the ranks holding it, its field values and set memberships. Records go as
// one byte stream per destination, cut into rows of `chunk` bytes, in one
// exchange; route keeps each sender's rows in order.
constexpr std::size_t chunk = 256;

void send_copies(const Comm& comm, Parts& in) {
  std::map<int, std::vector<std::byte>> streams;
  for (int d = 0; d <= in.meta.dim; ++d) {
    Rows& r = in.rows[static_cast<std::size_t>(d)];
    // Memberships by row.
    std::unordered_map<std::size_t, std::vector<const Membership*>> member_of;
    for (const Membership& x : r.members) member_of[x.row].push_back(&x);
    for (std::size_t i = 0; i < r.owned; ++i) {
      if (r.others[i].empty()) continue;
      std::vector<std::byte> rec;
      append(rec, static_cast<std::int32_t>(d));
      append(rec, static_cast<std::int64_t>(r.gid[i]));
      append(rec, r.part[i]);
      std::vector<int> holders = r.others[i];
      holders.push_back(comm.rank());
      std::sort(holders.begin(), holders.end());
      append(rec, static_cast<std::int32_t>(holders.size()));
      for (const int h : holders) append(rec, static_cast<std::int32_t>(h));
      if (d == 0) {
        for (std::size_t c = 0; c < 3; ++c) append(rec, r.xyz[3 * i + c]);
      } else {
        append(rec, r.topo[i]);
        append(rec, r.ncorner[i]);
        for (std::uint32_t k = 0; k < r.ncorner[i]; ++k) append(rec, static_cast<std::int64_t>(r.corners[r.corner_at[i] + k]));
        append(rec, r.nnode[i]);
        for (std::uint32_t k = 0; k < r.nnode[i]; ++k) append(rec, static_cast<std::int64_t>(r.nodes[r.node_at[i] + k]));
      }
      for (std::size_t f = 0; f < in.fields.size(); ++f) {
        if (!on(in.fields[f], d)) continue;
        const FieldRows& fr = r.fields[f];
        append(rec, fr.count[i]);
        if (fr.count[i] >= 0) {
          const auto n = static_cast<std::size_t>(fr.count[i]) * in.fields[f].unit;
          append(rec, std::span<const std::byte>(fr.bytes).subspan(static_cast<std::size_t>(fr.at[i]), n));
        }
      }
      const auto it = member_of.find(i);
      const auto& ms = it == member_of.end() ? std::vector<const Membership*>{} : it->second;
      append(rec, static_cast<std::int32_t>(ms.size()));
      for (const Membership* x : ms) {
        append(rec, x->set);
        append(rec, x->pos);
      }
      for (const int dest : r.others[i]) {
        auto& s = streams[dest];
        append(s, static_cast<std::uint64_t>(rec.size()));
        append(s, std::span<const std::byte>(rec));
      }
    }
  }
  std::vector<int> dest;
  std::vector<std::span<const std::byte>> pieces;
  for (auto& [to, s] : streams) {
    // A stream is its length then its records, padded to whole rows.
    std::vector<std::byte> framed;
    append(framed, static_cast<std::uint64_t>(s.size()));
    append(framed, std::span<const std::byte>(s));
    framed.resize((framed.size() + chunk - 1) / chunk * chunk);
    s = std::move(framed);
    for (std::size_t at = 0; at < s.size(); at += chunk) {
      dest.push_back(to);
      pieces.push_back(std::span<const std::byte>(s).subspan(at, chunk));
    }
  }
  // To neighbours only: NBX, with no P-sized arrays.
  Batch out(dest.size(), chunk, "read_soma.copies");
  for (std::size_t k = 0; k < dest.size(); ++k) {
    out.dest(k) = dest[k];
    std::memcpy(&out.payload(k, 0), pieces[k].data(), chunk);
  }
  const Batch got = transport::route(detail::HostExec{}, comm, out, transport::Strategy::nbx);

  // Each sender's rows, in order, are its stream to this rank.
  std::vector<std::byte> stream;
  const auto parse = [&](std::span<const std::byte> s) {
    Cursor c{s};
    const auto length = c.take<std::uint64_t>();
    Cursor body{c.bytes(static_cast<std::size_t>(length))};
    while (body.at < body.in.size()) {
      const auto n = body.take<std::uint64_t>();
      Cursor rec{body.bytes(static_cast<std::size_t>(n))};
      const auto d = rec.take<std::int32_t>();
      if (d < 0 || d > in.meta.dim) throw std::logic_error("read_soma: a copy names no dimension");
      Rows& r = in.rows[static_cast<std::size_t>(d)];
      const std::size_t row = r.size();
      r.gid.push_back(rec.take<std::int64_t>());
      r.part.push_back(rec.take<std::uint64_t>());
      const auto nh = rec.take<std::int32_t>();
      std::vector<int> others;
      for (std::int32_t k = 0; k < nh; ++k) {
        const auto h = rec.take<std::int32_t>();
        if (h != comm.rank()) others.push_back(h);
      }
      r.others.push_back(std::move(others));
      if (d == 0) {
        for (int k = 0; k < 3; ++k) r.xyz.push_back(rec.take<double>());
      } else {
        r.topo.push_back(rec.take<std::uint8_t>());
        const auto nc = rec.take<std::uint32_t>();
        r.ncorner.push_back(nc);
        r.corner_at.push_back(r.corners.size());
        for (std::uint32_t k = 0; k < nc; ++k) r.corners.push_back(rec.take<std::int64_t>());
        const auto nn = rec.take<std::uint32_t>();
        r.nnode.push_back(nn);
        r.node_at.push_back(r.nodes.size());
        for (std::uint32_t k = 0; k < nn; ++k) r.nodes.push_back(rec.take<std::int64_t>());
      }
      for (std::size_t f = 0; f < in.fields.size(); ++f) {
        if (!on(in.fields[f], d)) continue;
        FieldRows& fr = r.fields[f];
        fr.resize(row + 1);
        const auto units = rec.take<std::int64_t>();
        if (units >= 0) fr.set(row, units, rec.bytes(static_cast<std::size_t>(units) * in.fields[f].unit));
      }
      const auto nm = rec.take<std::int32_t>();
      for (std::int32_t k = 0; k < nm; ++k) {
        const auto set = rec.take<std::int64_t>();
        const auto pos = rec.take<std::int64_t>();
        r.members.push_back({set, row, r.part.back(), pos});
      }
    }
  };
  for (std::size_t k = 0; k < got.size(); ++k) {
    stream.insert(stream.end(), &got.payload(k, 0), &got.payload(k, 0) + chunk);
    const bool last = k + 1 == got.size() || got.source(k + 1) != got.source(k);
    if (!last) continue;
    parse(stream);
    stream.clear();
  }
  // Fields without values on a dimension have no rows there; the others one
  // per entity.
  for (int d = 0; d <= in.meta.dim; ++d) {
    Rows& r = in.rows[static_cast<std::size_t>(d)];
    for (std::size_t f = 0; f < r.fields.size(); ++f) {
      if (on(in.fields[f], d)) r.fields[f].resize(r.size());
    }
  }
}

// Runs f; a failure on any rank throws on every rank, the failing ones
// rethrowing their own.
template <class F>
void collectively(const Comm& comm, F&& f) {
  std::exception_ptr failure;
  try {
    f();
  } catch (...) {
    failure = std::current_exception();
  }
  if (comm.allreduce(static_cast<std::int32_t>(failure != nullptr), Op::max) != 0) {
    if (failure) std::rethrow_exception(failure);
    bad_file("failed on another rank");
  }
}

// The builder's gids of dimension d, in its order.
std::vector<GO> builder_gids(const MeshBuilder& b, int d) {
  if (d == 0) return {b.vertex_gids().begin(), b.vertex_gids().end()};
  return b.element_gids(d);
}

// The builder of this rank's entities and their sharing in its order.
struct Prepared {
  MeshBuilder builder;
  std::array<std::optional<Sharing>, 4> sharing;
};

Prepared prepare(const Parts& in) {
  const Meta& m = in.meta;
  Prepared p{MeshBuilder(m.dim), {}};
  MeshBuilder& b = p.builder;
  {
    const Rows& r = in.rows[0];
    std::vector<Real> xyz(r.xyz.begin(), r.xyz.end());
    b.add_vertices(r.gid, xyz);
  }
  for (int d = 1; d <= m.dim; ++d) {
    const Rows& r = in.rows[static_cast<std::size_t>(d)];
    if (!m.written[static_cast<std::size_t>(d)]) continue;
    // Groups of one topology and node count, in row order.
    std::map<std::pair<std::uint8_t, std::uint32_t>, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < r.size(); ++i) groups[{r.topo[i], r.nnode[i]}].push_back(i);
    for (const auto& [key, rows] : groups) {
      const auto t = static_cast<Topology>(key.first);
      std::vector<GO> gids, verts;
      if (t == Topology::polygon) {
        std::vector<Offset> offsets{0};
        for (const std::size_t i : rows) {
          gids.push_back(r.gid[i]);
          verts.insert(verts.end(), r.corners.begin() + static_cast<std::ptrdiff_t>(r.corner_at[i]),
                       r.corners.begin() + static_cast<std::ptrdiff_t>(r.corner_at[i] + r.ncorner[i]));
          offsets.push_back(static_cast<Offset>(verts.size()));
        }
        b.add_polygons(gids, offsets, verts);
        continue;
      }
      for (const std::size_t i : rows) {
        gids.push_back(r.gid[i]);
        verts.insert(verts.end(), r.corners.begin() + static_cast<std::ptrdiff_t>(r.corner_at[i]),
                     r.corners.begin() + static_cast<std::ptrdiff_t>(r.corner_at[i] + r.ncorner[i]));
        verts.insert(verts.end(), r.nodes.begin() + static_cast<std::ptrdiff_t>(r.node_at[i]),
                     r.nodes.begin() + static_cast<std::ptrdiff_t>(r.node_at[i] + r.nnode[i]));
      }
      if (key.second == 0) {
        b.add_elements(t, gids, verts);
      } else {
        const auto per = static_cast<int>(r.ncorner[rows.front()] + key.second);
        b.add_elements(t, gids, verts, per);
      }
    }
  }
  // The sharing of every dimension, in the builder's order.
  for (int d = 0; d <= m.dim; ++d) {
    const auto dd = static_cast<std::size_t>(d);
    if (!m.written[dd]) continue;
    const Rows& r = in.rows[dd];
    std::unordered_map<GO, std::size_t> row_of;
    row_of.reserve(r.size());
    for (std::size_t i = 0; i < r.size(); ++i) {
      if (!row_of.emplace(r.gid[i], i).second) bad_file("dimension " + std::to_string(d) + " holds gid " + std::to_string(r.gid[i]) + " twice");
    }
    Sharing s;
    s.offsets.push_back(0);
    for (const GO g : builder_gids(b, d)) {
      const auto& o = r.others[row_of.at(g)];
      s.ranks.insert(s.ranks.end(), o.begin(), o.end());
      s.offsets.push_back(static_cast<std::uint32_t>(s.ranks.size()));
    }
    p.sharing[dd] = std::move(s);
  }
  return p;
}

HostMesh build(const Comm& comm, const Parts& in, Ordering ordering) {
  const Meta& m = in.meta;
  std::optional<Prepared> p;
  collectively(comm, [&] { p.emplace(prepare(in)); });
  HostMesh mesh = build_local_part(comm, std::move(p->builder), p->sharing, ordering);
  // Dimensions written for their data but derived in the mesh written.
  for (int d = 1; d <= m.dim; ++d) {
    if (m.written[static_cast<std::size_t>(d)] && !m.input[static_cast<std::size_t>(d)]) {
      detail::TopologyAccess::block(mesh, d).input = false;
    }
  }
  return mesh;
}

std::vector<std::pair<GO, LO>> by_gid(const HostMesh& mesh, int d) {
  std::vector<std::pair<GO, LO>> out(static_cast<std::size_t>(mesh.count(d)));
  const auto g = mesh.gid(d);
  for (LO e = 0; e < mesh.count(d); ++e) out[static_cast<std::size_t>(e)] = {g(e), e};
  std::sort(out.begin(), out.end());
  return out;
}

LO lid_of(const std::vector<std::pair<GO, LO>>& sorted, GO g) {
  const auto it = std::lower_bound(sorted.begin(), sorted.end(), std::pair<GO, LO>{g, 0});
  if (it == sorted.end() || it->first != g) throw std::logic_error("read_soma: an entity read is not in the part");
  return it->second;
}

template <class T>
std::optional<std::vector<T>> typed(const std::optional<std::vector<std::byte>>& b) {
  if (!b) return std::nullopt;
  std::vector<T> out(b->size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), b->data(), out.size() * sizeof(T));
  return out;
}

template <class T>
void install_field(HostMesh& mesh, const Parts& in, std::size_t i, const std::array<std::vector<std::pair<GO, LO>>, 4>& lids) {
  const FieldMeta& fm = in.fields[i];
  using HostDomain = Domain<Kokkos::HostSpace>;
  // Per dimension the lids with values and where they are.
  struct At {
    LO lid;
    std::size_t row;
  };
  std::array<std::vector<At>, 4> at;
  HostDomain dom;
  bool first = true;
  for (const int d : fm.dims) {
    HostDomain blk;
    if (d == set_dim) {
      std::vector<LO> s;
      for (const auto& x : in.on_sets[i]) s.push_back(std::get<0>(x));
      blk = HostDomain::from_list(mesh, d, s);
    } else {
      const auto dd = static_cast<std::size_t>(d);
      const Rows& r = in.rows[dd];
      const FieldRows& fr = r.fields[i];
      for (std::size_t k = 0; k < fr.count.size(); ++k) {
        if (fr.count[k] >= 0) at[dd].push_back({lid_of(lids[dd], r.gid[k]), k});
      }
      std::sort(at[dd].begin(), at[dd].end(), [](const At& x, const At& y) { return x.lid < y.lid; });
      if (fm.full[dd]) {
        blk = HostDomain::all(mesh, d);
      } else {
        std::vector<LO> l;
        for (const At& x : at[dd]) l.push_back(x.lid);
        blk = HostDomain::from_list(mesh, d, l);
      }
    }
    dom = first ? std::move(blk) : HostDomain::combine(dom, blk);
    first = false;
  }
  const auto value = [&](int d, std::size_t row) -> std::pair<std::int64_t, std::span<const std::byte>> {
    const FieldRows& fr = in.rows[static_cast<std::size_t>(d)].fields[i];
    const auto units = fr.count[row];
    return {units, std::span<const std::byte>(fr.bytes).subspan(static_cast<std::size_t>(fr.at[row]),
                                                                static_cast<std::size_t>(units) * fm.unit)};
  };
  FieldOptions<T> options;
  options.default_value = typed<T>(fm.def);
  options.global_value = typed<T>(fm.glob);
  options.handle = fm.handle;
  auto& reg = mesh.fields();
  if (fm.variable) {
    std::vector<Offset> offsets{0};
    std::vector<T> values;
    for (int bi = 0; bi < dom.num_blocks(); ++bi) {
      const int d = dom.info(bi).dim;
      std::size_t k = 0;
      for (const LO l : dom.members(bi)) {
        std::span<const std::byte> v;
        if (d == set_dim) {
          for (const auto& x : in.on_sets[i]) {
            if (std::get<0>(x) == l) v = std::get<2>(x);
          }
        } else {
          const auto& a = at[static_cast<std::size_t>(d)];
          while (k < a.size() && a[k].lid < l) ++k;
          if (k < a.size() && a[k].lid == l) v = value(d, a[k].row).second;
        }
        const std::size_t n = v.size() / sizeof(T);
        const std::size_t base = values.size();
        values.resize(base + n);
        if (n > 0) std::memcpy(values.data() + base, v.data(), v.size());
        offsets.push_back(static_cast<Offset>(values.size()));
      }
    }
    (void)reg.template add_var<T>(fm.name, Space<Kokkos::HostSpace>(dom, fm.layout), offsets, values, options);
  } else {
    const auto g = reg.template add<T>(fm.name, make_space(mesh, dom, fm.layout), fm.ncomp, options);
    const auto nc = static_cast<std::size_t>(fm.ncomp);
    const auto fill = [&](int d, LO l, std::int64_t units, std::span<const std::byte> v) {
      const RowRange rr = g.space().entity_rows(d, l);
      if (rr.end - rr.begin != units) bad_file("field " + fm.name + " has the wrong number of values for an entity");
      for (LO k = 0; k < rr.end - rr.begin; ++k) {
        for (std::size_t c = 0; c < nc; ++c) {
          T x;
          std::memcpy(&x, v.data() + (static_cast<std::size_t>(k) * nc + c) * sizeof(T), sizeof(T));
          g.view()(rr.begin + k, static_cast<int>(c)) = x;
        }
      }
    };
    for (const int d : fm.dims) {
      if (d == set_dim) {
        for (const auto& x : in.on_sets[i]) fill(d, std::get<0>(x), std::get<1>(x), std::get<2>(x));
        continue;
      }
      for (const At& x : at[static_cast<std::size_t>(d)]) {
        const auto [units, v] = value(d, x.row);
        fill(d, x.lid, units, v);
      }
    }
  }
  if (fm.format != TagFormat{}) {
    reg.set_tag_format(reg.template family<T>(fm.name, fm.ncomp, fm.layout, fm.handle), fm.format);
  }
}

void install_sets(HostMesh& mesh, const Parts& in, const std::array<std::vector<std::pair<GO, LO>>, 4>& lids) {
  const std::size_t n = in.set_gid.size();
  if (n == 0) return;
  // (set, part, position, item): ordered sets list each part's entries in
  // order, parts in order, the set entries with part 0's.
  std::vector<std::tuple<std::int64_t, std::uint64_t, std::int64_t, SetItem>> got;
  for (int d = 0; d <= in.meta.dim; ++d) {
    const auto dd = static_cast<std::size_t>(d);
    const Rows& r = in.rows[dd];
    for (const Membership& x : r.members) {
      got.emplace_back(x.set, x.part, x.pos, SetItem{d, GO{lid_of(lids[dd], r.gid[x.row])}});
    }
  }
  for (const auto& [set, pos, gid] : in.entries) got.emplace_back(set, 0, pos, SetItem{set_dim, gid});
  std::sort(got.begin(), got.end(), [](const auto& x, const auto& y) {
    return std::tuple(std::get<0>(x), std::get<1>(x), std::get<2>(x)) <
           std::tuple(std::get<0>(y), std::get<1>(y), std::get<2>(y));
  });
  SetBuilder b;
  std::size_t k = 0;
  for (std::size_t i = 0; i < n; ++i) {
    SetSpec spec;
    spec.gid = in.set_gid[i];
    spec.flags = in.set_flags[i];
    spec.children = in.children[i];
    spec.parents = in.parents[i];
    for (; k < got.size() && std::get<0>(got[k]) == static_cast<std::int64_t>(i); ++k) {
      spec.contents.push_back(std::get<3>(got[k]));
    }
    for (const GO g : in.set_members[i]) spec.contents.push_back({set_dim, g});
    b.add(std::move(spec));
  }
  std::move(b).install(mesh);
}

void install_values(HostMesh& mesh, const File& f, const Meta& m) {
  for (std::int64_t i = 0; i < m.values; ++i) {
    const std::string g = value_group(static_cast<std::size_t>(i));
    const auto meta = f.ints(g, "meta");
    if (meta.size() < 8) bad_file("/soma/" + g + " has no complete meta attribute");
    MeshValue v;
    v.type = static_cast<DataType>(meta[0]);
    v.ncomp = static_cast<int>(meta[1]);
    v.handle = meta[2] != 0;
    v.format.storage_class = static_cast<int>(meta[3]);
    v.format.kind = static_cast<decltype(v.format.kind)>(meta[4]);
    v.format.nbits = static_cast<int>(meta[5]);
    if (meta[6] != 0) v.default_value = f.bytes(g, "default").value_or(std::vector<std::byte>{});
    if (meta[7] != 0) v.global_value = f.bytes(g, "global").value_or(std::vector<std::byte>{});
    mesh.values().set(f.text(g, "name"), std::move(v));
  }
}


}  // namespace

HostMesh read_soma(const Comm& comm, const fs::path& path, const SomaReadOptions& options) {
  const bool collective = soma_file::parallel_hdf5() && comm.size() > 1;
  std::optional<File> f;
  Parts in;
  collectively(comm, [&] {
    const h5::Session quiet;
    try {
      f.emplace(comm, path, collective);
      in = read_file(comm, *f);
    } catch (const h5::Error& e) {
      bad_file("cannot read " + path.string() + ": " + e.what());
    }
  });
  collectively(comm, [&] { send_copies(comm, in); });
  HostMesh mesh = build(comm, in, options.ordering);
  if (options.adjacencies && in.meta.graphs != 0) {
    build_distributed_adjacencies(comm, mesh, AdjacencyRequest::from_bits(in.meta.graphs));
  }
  collectively(comm, [&] {
    const h5::Session quiet;
    std::array<std::vector<std::pair<GO, LO>>, 4> lids;
    for (int d = 0; d <= mesh.dim(); ++d) lids[static_cast<std::size_t>(d)] = by_gid(mesh, d);
    install_sets(mesh, in, lids);
    for (std::size_t i = 0; i < in.fields.size(); ++i) {
      dispatch_data_type(in.fields[i].type, [&]<class T>() { install_field<T>(mesh, in, i, lids); });
    }
    install_values(mesh, *f, in.meta);
  });
  f.reset();
  if (options.rebalance && in.meta.parts < static_cast<std::uint64_t>(comm.size())) {
    const int top = mesh.dim();
    const auto view = mesh.view();
    std::vector<Real> centroids;
    std::vector<GO> gids;
    for (LO e = 0; e < mesh.count(top); ++e) {
      std::array<Real, 3> c{};
      const LO k = view.num_vertices(top, e);
      for (LO j = 0; j < k; ++j) {
        for (int x = 0; x < 3; ++x) c[static_cast<std::size_t>(x)] += view.x(view.vertex(top, e, j), x);
      }
      for (const Real x : c) centroids.push_back(x / static_cast<Real>(std::max<LO>(k, 1)));
      gids.push_back(view.gid(top, e));
    }
    const std::vector<int> dest = partition_by_curve(comm, centroids, gids);
    mesh = migrate(comm, mesh, dest, options.ordering);
  }
  return mesh;
}

HostMesh read_soma(const fs::path& path, const SomaReadOptions& options) {
  return read_soma(Comm::self(), path, options);
}

MeshFormat detect_format(const fs::path& path) {
  const h5::Session quiet;
  if (H5Fis_accessible(path.c_str(), H5P_DEFAULT) <= 0) return MeshFormat::unknown;
  const hid_t id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (id < 0) return MeshFormat::unknown;
  const Id file(id);
  if (H5Lexists(file.get(), soma_file::root, H5P_DEFAULT) > 0) return MeshFormat::soma;
  if (H5Lexists(file.get(), "tstt", H5P_DEFAULT) > 0) return MeshFormat::legacy_h5m;
  return MeshFormat::unknown;
}

}  // namespace soma
