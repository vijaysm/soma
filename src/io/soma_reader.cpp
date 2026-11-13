// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The SOMA format's reader: each rank reads a contiguous
// block of parts (or, with aggregators, the first rank of its group reads the
// group's and sends it its own), owners send their shared entities to the
// ranks holding copies, and each rank builds its part with the sharing the
// file gives.

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
#include <utility>
#include <vector>

#include "../core/dense_ids.hpp"
#include "../parallel/records.hpp"
#include "hdf5.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_values.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/io/vtk_reader.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/topology/build.hpp"
#include "soma_layout.hpp"

namespace soma {

namespace {

namespace fs = std::filesystem;
using detail::Batch;
using detail::GidIndex;
using h5::Id;
using soma_file::dim_group;
using soma_file::field_group;
using soma_file::value_group;

constexpr const char* who = "read_soma";

[[noreturn]] void bad_file(const std::string& why) { throw std::runtime_error(std::string(who) + ": " + why); }

// A file whose datasets do not hold the rows its part table gives this rank.
struct PartTableError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

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

template <class T>
std::vector<T> as(const std::vector<std::byte>& b) {
  std::vector<T> out(b.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), b.data(), out.size() * sizeof(T));
  return out;
}

template <class T>
std::vector<std::byte> bytes_of(const std::vector<T>& v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), v.data(), out.size());
  return out;
}

// The lines of a text.
std::vector<std::string> lines(const std::string& text) {
  std::vector<std::string> out;
  for (std::size_t at = 0; at < text.size();) {
    const std::size_t nl = text.find('\n', at);
    out.push_back(text.substr(at, nl - at));
    at = nl == std::string::npos ? text.size() : nl + 1;
  }
  return out;
}

// Rows [first, first + rows) of a dataset of `width` columns into `out`;
// all of them, the same on every rank, when `whole`.
struct Read {
  std::string path;
  DataType type = DataType::u8;
  std::size_t width = 1;
  std::uint64_t first = 0, rows = 0;
  std::vector<std::byte>* out = nullptr;
  bool whole = false;
};

// Rows of a dataset read, from row `first`, of `row_bytes` each.
struct RowsRead {
  std::uint64_t first = 0, row_bytes = 0;
  std::vector<std::byte> data;
  bool whole = false;
};

// What a file answered: the attributes and dataset extents asked of it, by
// query, and the rows read of each dataset. With aggregators, the first rank
// of a group records one for its group's parts, and each rank of the group
// reads its part from its slice of it as from the file.
struct Replica {
  std::map<std::string, std::vector<std::byte>> answers;
  std::map<std::string, RowsRead> rows;
};

// The file and its metadata, read alike on every rank, or a replica of what a
// file answered. Over more than one rank with MPI-IO the metadata is shared:
// the file is opened once, collectively, and the first rank alone asks it
// every question (see share_answers); the others take its answers, and every
// rank reads the rows straight from the file in one collective MPI-IO read
// where a dataset is stored as the rows it holds.
class File {
 public:
  // Opens the file; what it answers is added to `record` when given.
  File(const Comm& comm, const fs::path& path, bool collective, const MpiIoHints& hints = {},
       Replica* record = nullptr)
      : path_(path), record_(record), shared_(collective && comm.size() > 1) {
    const Id fapl = soma_file::file_access(comm, collective, shared_, hints);
    file_ = h5::check(H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl.get()), "open " + path.string());
    if (!shared_) open_root();
    dxpl_ = soma_file::transfer(collective);
  }

  // Answers as `replica` recorded them, without a file.
  explicit File(Replica replica) : replica_(std::move(replica)) {}

  // Whether the metadata is shared, to be asked of the first rank alone.
  [[nodiscard]] bool shared() const { return shared_; }

  // On the first rank of a shared file: from now on the file is asked and its
  // answers kept, and reads only note where the rows are, filling zeros.
  void survey() {
    open_root();
    answers_.emplace();
    surveying_ = true;
  }

  // The answers kept since survey().
  std::map<std::string, std::vector<std::byte>> surveyed() {
    surveying_ = false;
    return std::exchange(answers_, std::nullopt).value_or(std::map<std::string, std::vector<std::byte>>{});
  }

  // Answers the first rank surveyed, given instead of the file's.
  void answer_from(std::map<std::string, std::vector<std::byte>> answers) { answers_ = std::move(answers); }

  [[nodiscard]] bool has_amr() const {
    return !must("amr", answer("amr", [&] { return std::vector<std::byte>(H5Aexists(root_.get(), "amr") > 0); }))
                .empty();
  }

  [[nodiscard]] std::vector<std::uint64_t> extent(const std::string& path) const {
    return as<std::uint64_t>(must(path, answer("x|" + path, [&] { return file_extent(path); })));
  }

  [[nodiscard]] std::vector<std::int64_t> ints(const std::string& object, const char* name) const {
    return as<std::int64_t>(must(name, answer("i|" + object + "|" + name, [&] { return file_ints(object, name); })));
  }

  [[nodiscard]] std::optional<std::vector<std::byte>> bytes(const std::string& object, const char* name) const {
    return answer("b|" + object + "|" + name, [&] { return file_bytes(object, name); });
  }

  [[nodiscard]] std::string text(const std::string& object, const char* name) const {
    const auto b = must(name, answer("t|" + object + "|" + name, [&] { return file_text(object, name); }));
    const auto chars = as<char>(b);
    return {chars.begin(), chars.end()};
  }

  // Every read in one transfer. Rows a dataset does not hold make this rank
  // read nothing, still joining the transfer, and are returned as the cause,
  // for the caller to report with its next check.
  [[nodiscard]] std::string read(std::vector<Read>& reads) const {
    if (surveying_) {
      for (Read& r : reads) {
        (void)extent(r.path);
        (void)answer("o|" + r.path, [&] { return file_location(r.path, r.type); });
        r.out->assign(r.rows * r.width * soma_file::value_size(r.type), std::byte{0});
      }
      return {};
    }
    std::vector<bool> empty;  // no rows on any rank: not read
    std::string error;
    for (const Read& r : reads) {
      const auto dims = extent(r.path);
      empty.push_back(dims.size() == 2 && dims[0] == 0);
      const bool fits = dims.size() == 2 && dims[1] == r.width && r.first <= dims[0] && r.rows <= dims[0] - r.first;
      if (!fits && error.empty()) error = "/soma/" + r.path + " does not hold the rows its part table gives";
    }
    if (!error.empty()) {
      for (Read& r : reads) r.rows = 0;
    }
    for (Read& r : reads) r.out->resize(r.rows * r.width * soma_file::value_size(r.type));
    if (replica_) {
      for (Read& r : reads) {
        if (r.rows == 0) continue;
        const auto it = replica_->rows.find(r.path);
        const std::size_t n = r.out->size();
        if (it == replica_->rows.end() || r.first < it->second.first ||
            (r.first - it->second.first) * it->second.row_bytes + n > it->second.data.size()) {
          bad_file("the rows of /soma/" + r.path + " read for this rank's group do not hold its part's");
        }
        const auto at = static_cast<std::ptrdiff_t>((r.first - it->second.first) * it->second.row_bytes);
        std::copy(it->second.data.begin() + at, it->second.data.begin() + at + static_cast<std::ptrdiff_t>(n),
                  r.out->begin());
      }
      return error;
    }
    // Rows of datasets stored as they are read, in one MPI-IO read; the
    // others through HDF5. Where each is, and whether it is, is the same on
    // every rank.
    std::vector<bool> raw(reads.size(), false);
    if (answers_) {
      std::vector<soma_file::Piece> pieces;
      bool any = false;
      for (std::size_t k = 0; k < reads.size(); ++k) {
        const Read& r = reads[k];
        if (empty[k]) continue;
        const auto at = as<std::uint64_t>(must(r.path, answer("o|" + r.path, [] { return std::nullopt; })));
        if (at.size() != 2 || at[0] == 0) continue;
        raw[k] = any = true;
        const std::uint64_t row = r.width * soma_file::value_size(r.type);
        if (r.rows > 0) pieces.push_back({at[1] + r.first * row, r.rows * row, r.out->data()});
      }
      if (any) {
        try {
          soma_file::read_raw(file_.get(), std::move(pieces));
        } catch (const h5::Error&) {
          throw;
        } catch (const std::exception& e) {
          bad_file(path_.string() + ": " + e.what());
        }
      }
    }
    std::vector<Id> ds, spaces, memories;
    std::vector<hid_t> ids, types, mem, file;
    std::vector<void*> buffers;
    for (std::size_t k = 0; k < reads.size(); ++k) {
      Read& r = reads[k];
      if (empty[k] || raw[k]) continue;
      open_root();
      ds.push_back(h5::open_dataset(root_.get(), r.path));
      Id space = h5::check(H5Dget_space(ds.back().get()), "get a dataspace");
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
      ids.push_back(ds.back().get());
      types.push_back(soma_file::memory_type(r.type));
      mem.push_back(memory.get());
      file.push_back(space.get());
      buffers.push_back(r.out->empty() ? static_cast<void*>(&r) : r.out->data());
      spaces.push_back(std::move(space));
      memories.push_back(std::move(memory));
    }
    if (!ids.empty()) {
#if H5_VERSION_GE(1, 14, 0)
      h5::check(H5Dread_multi(ids.size(), ids.data(), types.data(), mem.data(), file.data(), dxpl_.get(),
                              buffers.data()),
                "read the datasets");
#else
      for (std::size_t k = 0; k < ids.size(); ++k) {
        h5::check(H5Dread(ids[k], types[k], mem[k], file[k], dxpl_.get(), buffers[k]), "read a dataset");
      }
#endif
    }
    if (record_ != nullptr) {
      for (const Read& r : reads) {
        record_->rows[r.path] = {r.first, r.width * soma_file::value_size(r.type), *r.out, r.whole};
      }
    }
    return error;
  }

 private:
  // The answer to a query: the file's (recorded when recording), or the
  // replica's; none for an attribute the file does not have.
  template <class F>
  std::optional<std::vector<std::byte>> answer(const std::string& key, F&& ask) const {
    const auto given = replica_ ? &replica_->answers : surveying_ ? nullptr : answers_ ? &*answers_ : nullptr;
    if (given != nullptr) {
      const auto it = given->find(key);
      if (it == given->end()) return std::nullopt;
      if (record_ != nullptr) record_->answers[key] = it->second;
      return it->second;
    }
    std::optional<std::vector<std::byte>> a = ask();
    if (surveying_ && a) (*answers_)[key] = *a;
    if (record_ != nullptr && !surveying_ && a) record_->answers[key] = *a;
    return a;
  }
  // Opened on the first rank's survey, and on the others when a dataset is
  // read through HDF5.
  void open_root() const {
    if (root_.get() >= 0) return;
    if (H5Lexists(file_.get(), soma_file::root, H5P_DEFAULT) <= 0) bad_file(path_.string() + " is not a SOMA file");
    root_ = h5::open_group(file_.get(), soma_file::root);
  }
  // Where a dataset's rows start in the file, after a flag set when they are
  // stored there as they are read.
  [[nodiscard]] std::vector<std::byte> file_location(const std::string& path, DataType t) const {
    const Id ds = h5::open_dataset(root_.get(), path);
    const soma_file::Location at = soma_file::location(ds.get(), soma_file::memory_type(t), "/soma/" + path);
    return bytes_of(std::vector<std::uint64_t>{at.raw ? 1U : 0U, at.at});
  }
  [[nodiscard]] std::vector<std::byte> file_extent(const std::string& path) const {
    const Id ds = h5::open_dataset(root_.get(), path);
    std::vector<std::uint64_t> dims;
    for (const hsize_t n : h5::extent(ds.get(), path)) dims.push_back(n);
    return bytes_of(dims);
  }
  [[nodiscard]] std::vector<std::byte> file_ints(const std::string& object, const char* name) const {
    const Id a = attribute(object, name);
    const Id space = h5::check(H5Aget_space(a.get()), "get the dataspace of an attribute");
    std::vector<std::int64_t> out(static_cast<std::size_t>(H5Sget_simple_extent_npoints(space.get())));
    if (!out.empty()) h5::check(H5Aread(a.get(), H5T_NATIVE_INT64, out.data()), std::string("read ") + name);
    return bytes_of(out);
  }
  [[nodiscard]] std::optional<std::vector<std::byte>> file_bytes(const std::string& object, const char* name) const {
    const Id obj = open(object);
    if (H5Aexists(obj.get(), name) <= 0) return std::nullopt;
    const Id a = h5::check(H5Aopen(obj.get(), name, H5P_DEFAULT), std::string("open the attribute ") + name);
    const Id space = h5::check(H5Aget_space(a.get()), "get the dataspace of an attribute");
    std::vector<std::byte> out(static_cast<std::size_t>(H5Sget_simple_extent_npoints(space.get())));
    if (!out.empty()) h5::check(H5Aread(a.get(), H5T_NATIVE_UINT8, out.data()), std::string("read ") + name);
    return out;
  }
  [[nodiscard]] std::vector<std::byte> file_text(const std::string& object, const char* name) const {
    const Id a = attribute(object, name);
    const Id type = h5::check(H5Aget_type(a.get()), "get the type of an attribute");
    std::string out(H5Tget_size(type.get()), '\0');
    h5::check(H5Aread(a.get(), type.get(), out.data()), std::string("read ") + name);
    out.resize(std::strlen(out.c_str()));
    return bytes_of(std::vector<char>(out.begin(), out.end()));
  }
  [[nodiscard]] static std::vector<std::byte> must(const std::string& what, std::optional<std::vector<std::byte>> a) {
    if (!a) bad_file("the group's first rank read no " + what + " for this rank");
    return std::move(*a);
  }
  [[nodiscard]] Id open(const std::string& object) const {
    return h5::check(H5Oopen(root_.get(), object.empty() ? "." : object.c_str(), H5P_DEFAULT), "open /soma/" + object);
  }
  [[nodiscard]] Id attribute(const std::string& object, const char* name) const {
    const Id obj = open(object);
    return h5::check(H5Aopen(obj.get(), name, H5P_DEFAULT),
                     std::string("open the attribute ") + name + " of /soma/" + object);
  }

  fs::path path_;
  Id file_, dxpl_;
  mutable Id root_;
  Replica* record_ = nullptr;
  std::optional<Replica> replica_;
  bool shared_ = false, surveying_ = false;
  mutable std::optional<std::map<std::string, std::vector<std::byte>>> answers_;  // shared: the first rank's
};

// What the root says.
struct Meta {
  int dim = 0;
  std::uint64_t parts = 0;
  std::array<bool, 4> written{}, input{}, nodes{};
  std::uint32_t graphs = 0;
  std::int64_t sets = 0, fields = 0, values = 0;
};

Meta read_meta(const File& f) {
  if (f.has_amr()) bad_file("it is an AMR checkpoint, not a mesh (read_soma_amr reads it)");
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

// The ranks [first, last) of `size` whose parts are read together.
struct Ranks {
  int first = 0, last = 0, size = 0;
};

// Reads the parts of `ranks` (collectively over comm) and, when `parse`,
// lays them out as Parts; without, only the metadata and block are set.
Parts read_file(const Comm& comm, const File& f, Ranks ranks, bool parse = true) {
  Parts in;
  in.meta = read_meta(f);
  const Meta& m = in.meta;
  in.block = {block_of(ranks.first, ranks.size, m.parts).first, block_of(ranks.last - 1, ranks.size, m.parts).last};
  const Block& b = in.block;

  // The part table rows of this block's parts and the one after.
  const std::vector<std::string> columns = lines(f.text("parts/offsets", "columns"));
  const std::size_t k_count = columns.size();
  std::vector<std::byte> table_bytes;
  std::string bad;  // rows the datasets do not hold, reported after the transfers
  {
    std::vector<Read> one{{"parts/offsets", DataType::u64, k_count, b.first, b.last - b.first + 1, &table_bytes}};
    bad = f.read(one);
  }
  // Without its table rows, the rank reads nothing more.
  const auto table = bad.empty() ? as<std::uint64_t>(table_bytes)
                                 : std::vector<std::uint64_t>((b.last - b.first + 1) * k_count, 0);
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
    const auto dims = f.extent(path);
    reads.push_back({path, t, width, 0, dims.empty() ? 0 : dims[0], &data[path], true});
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
  if (std::string more = f.read(reads); bad.empty()) bad = std::move(more);
  if (!bad.empty()) throw PartTableError(std::string(who) + ": " + bad);
  if (!parse) return in;

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
    // The part's first row of each dimension written.
    std::array<std::uint64_t, 4> base{};
    for (int d = 0; d <= m.dim; ++d) {
      const std::string g = dim_group(d);
      if (m.written[static_cast<std::size_t>(d)]) {
        base[static_cast<std::size_t>(d)] = start(q, g + "/gid") - start(b.first, g + "/gid");
      }
    }
    for (std::uint64_t k = 0; k < n; ++k, at += 4) {
      const std::int64_t set = members[at], d = members[at + 1], index = members[at + 2], pos = members[at + 3];
      if (set < 0 || static_cast<std::size_t>(set) >= nsets || d < 0 || d > m.dim ||
          !m.written[static_cast<std::size_t>(d)]) {
        bad_file("/soma/sets/members is inconsistent");
      }
      const auto row = base[static_cast<std::size_t>(d)] + static_cast<std::uint64_t>(index);
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
    if (std::all_of(r.others.begin(), r.others.begin() + static_cast<std::ptrdiff_t>(r.owned),
                    [](const std::vector<int>& o) { return o.empty(); })) {
      continue;
    }
    // Memberships by row.
    std::unordered_map<std::size_t, std::vector<const Membership*>> member_of;
    for (const Membership& x : r.members) {
      if (!r.others[x.row].empty()) member_of[x.row].push_back(&x);
    }
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
// rethrowing their own, the others naming a part table that does not match
// its datasets when that is a cause.
template <class F>
void collectively(const Comm& comm, F&& f) {
  std::exception_ptr failure;
  std::int32_t code = 0;
  try {
    f();
  } catch (const PartTableError&) {
    failure = std::current_exception();
    code = 2;
  } catch (...) {
    failure = std::current_exception();
    code = 1;
  }
  code = comm.allreduce(code, Op::max);
  if (code != 0) {
    if (failure) std::rethrow_exception(failure);
    bad_file(code == 2 ? "another rank's datasets do not match the part table" : "failed on another rank");
  }
}

// The local ids of a dimension's entities by gid.
class Lids {
 public:
  Lids() = default;
  Lids(const HostMesh& mesh, int d) : index_(static_cast<std::size_t>(mesh.count(d)), [g = mesh.gid(d)](std::size_t i) {
    return g(static_cast<LO>(i));
  }) {}

  // Throws std::logic_error when no entity has gid g.
  LO operator()(GO g) const {
    const std::size_t at = index_.find(g);
    if (at == GidIndex::none) throw std::logic_error("read_soma: an entity read is not in the part");
    return static_cast<LO>(at);
  }

 private:
  GidIndex index_;
};

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
    const GidIndex row_of(r.size(), [&](std::size_t i) { return r.gid[i]; });
    if (row_of.repeated) {
      bad_file("dimension " + std::to_string(d) + " holds gid " + std::to_string(*row_of.repeated) + " twice");
    }
    Sharing s;
    s.offsets.push_back(0);
    for (const GO g : builder_gids(b, d)) {
      const std::size_t row = row_of.find(g);
      if (row == GidIndex::none) throw std::logic_error("read_soma: the builder holds a gid not read");
      const auto& o = r.others[row];
      s.ranks.insert(s.ranks.end(), o.begin(), o.end());
      s.offsets.push_back(static_cast<std::uint32_t>(s.ranks.size()));
    }
    p.sharing[dd] = std::move(s);
  }
  return p;
}

// The part, with `before` (this rank's failure so far) reported with the
// build's checks; a failure of the build itself is left in `failure`.
HostMesh build(const Comm& comm, const Parts& in, Ordering ordering, std::exception_ptr before,
               std::exception_ptr& failure) {
  const Meta& m = in.meta;
  std::optional<Prepared> p;
  if (!before) {
    try {
      p.emplace(prepare(in));
    } catch (...) {
      before = std::current_exception();
    }
  }
  HostMesh mesh = detail::build_local_part_deferred(comm, p ? std::move(p->builder) : MeshBuilder(m.dim),
                                                    p ? p->sharing : std::array<std::optional<Sharing>, 4>{},
                                                    ordering, before, failure, who);
  if (failure) return mesh;
  // Dimensions written (for their data, or all of them) but derived in the
  // mesh written: complete on every rank, since each holds its owned
  // entities and copies of the rest its elements use, so building a graph
  // keeps them, with their order and sharing, instead of deriving them again.
  for (int d = 1; d <= m.dim; ++d) {
    if (m.written[static_cast<std::size_t>(d)] && !m.input[static_cast<std::size_t>(d)]) {
      detail::TopologyAccess::block(mesh, d).input = false;
      detail::TopologyAccess::block(mesh, d).derived = true;
    }
  }
  return mesh;
}

template <class T>
std::optional<std::vector<T>> typed(const std::optional<std::vector<std::byte>>& b) {
  if (!b) return std::nullopt;
  std::vector<T> out(b->size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), b->data(), out.size() * sizeof(T));
  return out;
}

template <class T>
void install_field(HostMesh& mesh, const Parts& in, std::size_t i, const std::array<Lids, 4>& lids) {
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
        if (fr.count[k] >= 0) at[dd].push_back({lids[dd](r.gid[k]), k});
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

void install_sets(HostMesh& mesh, const Parts& in, const std::array<Lids, 4>& lids) {
  const std::size_t n = in.set_gid.size();
  if (n == 0) return;
  // (set, part, position, item): ordered sets list each part's entries in
  // order, parts in order, the set entries with part 0's.
  std::vector<std::tuple<std::int64_t, std::uint64_t, std::int64_t, SetItem>> got;
  for (int d = 0; d <= in.meta.dim; ++d) {
    const auto dd = static_cast<std::size_t>(d);
    const Rows& r = in.rows[dd];
    for (const Membership& x : r.members) {
      got.emplace_back(x.set, x.part, x.pos, SetItem{d, GO{lids[dd](r.gid[x.row])}});
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

// The mesh values the file holds, by name.
std::vector<std::pair<std::string, MeshValue>> values_of(const File& f, const Meta& m) {
  std::vector<std::pair<std::string, MeshValue>> out;
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
    out.emplace_back(f.text(g, "name"), std::move(v));
  }
  return out;
}

void install_values(HostMesh& mesh, const File& f, const Meta& m) {
  for (auto& [name, v] : values_of(f, m)) mesh.values().set(name, std::move(v));
}


// The part of a group's replica that a rank of it reads, for its block of
// parts: the datasets read whole, and those parts' rows of the part table
// and of each other column.
Replica slice(const Replica& group, Block b) {
  Replica out;
  out.answers = group.answers;
  const auto names = group.answers.find("t|parts/offsets|columns");
  const auto offsets = group.rows.find("parts/offsets");
  if (names == group.answers.end() || offsets == group.rows.end()) bad_file("the part table was not read");
  const auto chars = as<char>(names->second);
  const std::vector<std::string> columns = lines(std::string(chars.begin(), chars.end()));
  std::map<std::string, std::size_t> column_of;
  for (std::size_t k = 0; k < columns.size(); ++k) column_of[columns[k]] = k;
  const RowsRead& t = offsets->second;
  const auto table = as<std::uint64_t>(t.data);
  const auto start = [&](std::uint64_t q, std::size_t k) {
    const std::size_t at = (q - t.first) * columns.size() + k;
    if (q < t.first || at >= table.size()) bad_file("the part table rows read do not cover a rank's parts");
    return table[at];
  };
  for (const auto& [path, rows] : group.rows) {
    std::uint64_t lo = 0, hi = 0;
    if (rows.whole) {
      out.rows[path] = rows;
      continue;
    }
    if (path == "parts/offsets") {
      lo = b.first;
      hi = b.last + 1;
    } else if (const auto k = column_of.find(path); k != column_of.end()) {
      lo = start(b.first, k->second);
      hi = start(b.last, k->second);
    } else {
      bad_file("/soma/" + path + " is read by part but is no column of the part table");
    }
    // Rows the group did not read (a bad part table) are left out: the rank's
    // own read finds the table wrong, as without aggregators.
    if (lo < rows.first || hi < lo || (hi - rows.first) * rows.row_bytes > rows.data.size()) continue;
    const auto begin = rows.data.begin() + static_cast<std::ptrdiff_t>((lo - rows.first) * rows.row_bytes);
    out.rows[path] = {lo, rows.row_bytes,
                      std::vector<std::byte>(begin, begin + static_cast<std::ptrdiff_t>((hi - lo) * rows.row_bytes)),
                      false};
  }
  return out;
}

void append_text(std::vector<std::byte>& out, const std::string& s) {
  append(out, static_cast<std::uint64_t>(s.size()));
  append(out, std::as_bytes(std::span<const char>(s)));
}

std::string take_text(Cursor& c) {
  const auto n = static_cast<std::size_t>(c.take<std::uint64_t>());
  const auto b = c.bytes(n);
  return {reinterpret_cast<const char*>(b.data()), n};
}

std::vector<std::byte> pack(const Replica& r) {
  std::vector<std::byte> out;
  append(out, static_cast<std::uint64_t>(r.answers.size()));
  for (const auto& [key, value] : r.answers) {
    append_text(out, key);
    append(out, static_cast<std::uint64_t>(value.size()));
    append(out, std::span<const std::byte>(value));
  }
  append(out, static_cast<std::uint64_t>(r.rows.size()));
  for (const auto& [path, rows] : r.rows) {
    append_text(out, path);
    append(out, rows.first);
    append(out, rows.row_bytes);
    append(out, static_cast<std::uint8_t>(rows.whole ? 1 : 0));
    append(out, static_cast<std::uint64_t>(rows.data.size()));
    append(out, std::span<const std::byte>(rows.data));
  }
  return out;
}

Replica unpack(std::span<const std::byte> in) {
  Cursor c{in};
  Replica r;
  for (auto n = c.take<std::uint64_t>(); n > 0; --n) {
    std::string key = take_text(c);
    const auto b = c.bytes(static_cast<std::size_t>(c.take<std::uint64_t>()));
    r.answers[std::move(key)] = {b.begin(), b.end()};
  }
  for (auto n = c.take<std::uint64_t>(); n > 0; --n) {
    std::string path = take_text(c);
    RowsRead rows;
    rows.first = c.take<std::uint64_t>();
    rows.row_bytes = c.take<std::uint64_t>();
    rows.whole = c.take<std::uint8_t>() != 0;
    const auto b = c.bytes(static_cast<std::size_t>(c.take<std::uint64_t>()));
    rows.data.assign(b.begin(), b.end());
    r.rows[std::move(path)] = std::move(rows);
  }
  return r;
}

// For a shared file: the first rank of comm asks the file every question
// `ask` asks (and where each dataset read is), alone, and its answers, or its
// failure, go to every rank in one broadcast; the file then answers from
// them. Otherwise nothing.
template <class F>
void share_answers(const Comm& comm, File& f, F&& ask) {
  if (!f.shared()) return;
  std::vector<std::byte> message;
  if (comm.rank() == 0) {
    message.push_back(std::byte{0});
    try {
      f.survey();
      ask();
      Replica r;
      r.answers = f.surveyed();
      append(message, std::span<const std::byte>(pack(r)));
    } catch (const std::exception& e) {
      (void)f.surveyed();
      message.assign(1, dynamic_cast<const h5::Error*>(&e) != nullptr ? std::byte{2} : std::byte{1});
      append(message, std::as_bytes(std::span<const char>(e.what(), std::strlen(e.what()))));
    }
  }
  message = soma_file::broadcast(comm, std::move(message));
  if (message.empty()) bad_file("the first rank sent no answers");
  const auto rest = std::span<const std::byte>(message).subspan(1);
  const std::string text(reinterpret_cast<const char*>(rest.data()), message[0] == std::byte{0} ? 0 : rest.size());
  if (message[0] == std::byte{2}) throw h5::Error(text);
  if (message[0] != std::byte{0}) throw std::runtime_error(text);
  f.answer_from(unpack(rest).answers);
}

constexpr int tag_replica_size = 31010;
constexpr int tag_replica = 31011;

// With aggregators: the first rank of each group alone opens the file
// (collectively with the other first ranks), reads its group's parts, and
// sends each rank of the group its slice, a failure's cause instead when it
// fails; it keeps its own.
Replica gather_reads(const Comm& comm, const fs::path& path, const SomaReadOptions& options) {
  const int p = comm.size(), me = comm.rank(), m = options.aggregators;
  const int leader = soma_file::leader_of(me, m, p);
  const SplitComm leaders = comm.split(leader == me ? 0 : -1, me);
  if (leader != me) {
    std::uint64_t n = 0;
    Request size = comm.irecv(std::span<std::uint64_t>(&n, 1), leader, tag_replica_size);
    size.wait();
    std::vector<std::byte> message(n);
    Request data = comm.irecv(std::span<std::byte>(message), leader, tag_replica);
    data.wait();
    if (message.empty()) bad_file("the group's first rank sent nothing");
    const auto rest = std::span<const std::byte>(message).subspan(1);
    if (message[0] == std::byte{0}) {
      throw std::runtime_error(std::string(reinterpret_cast<const char*>(rest.data()), rest.size()));
    }
    return unpack(rest);
  }
  int end = me + 1;
  while (end < p && soma_file::leader_of(end, m, p) == me) ++end;
  std::string error;
  Replica group;
  std::uint64_t parts = 0;
  try {
    const h5::Session quiet;
    try {
      File f(leaders.comm(), path, soma_file::parallel_hdf5(), options.hints, &group);
      share_answers(leaders.comm(), f, [&] {
        const Parts in = read_file(leaders.comm(), f, Ranks{me, end, p}, false);
        (void)values_of(f, in.meta);
      });
      const Parts in = read_file(leaders.comm(), f, Ranks{me, end, p}, false);
      (void)values_of(f, in.meta);  // asked now for the group's ranks
      parts = in.meta.parts;
    } catch (const h5::Error& e) {
      bad_file("cannot read " + path.string() + ": " + e.what());
    }
  } catch (const std::exception& e) {
    error = e.what();
  }
  const auto message = [&](int r) {
    std::vector<std::byte> out{std::byte{1}};
    if (error.empty()) {
      try {
        append(out, std::span<const std::byte>(pack(slice(group, block_of(r, p, parts)))));
        return out;
      } catch (const std::exception& e) {
        error = e.what();
      }
    }
    out.assign(1, std::byte{0});
    append(out, std::as_bytes(std::span<const char>(error)));
    return out;
  };
  for (int r = me + 1; r < end; ++r) {
    const std::vector<std::byte> out = message(r);
    const std::uint64_t n = out.size();
    Request size = comm.issend(std::span<const std::uint64_t>(&n, 1), r, tag_replica_size);
    size.wait();
    Request data = comm.issend(std::span<const std::byte>(out), r, tag_replica);
    data.wait();
  }
  Replica mine;
  if (error.empty()) mine = slice(group, block_of(me, p, parts));
  if (!error.empty()) throw std::runtime_error(error);
  return mine;
}

}  // namespace

HostMesh read_soma(const Comm& comm, const fs::path& path, const SomaReadOptions& options) {
  {
    // The access options, checked and agreed in one reduction.
    const std::string error = soma_file::access_error(options.hints, options.aggregators, comm.size());
    const auto h63 = static_cast<std::int64_t>(soma_file::access_hash(options.hints, options.aggregators) >> 1);
    std::array<std::int64_t, 3> agree{error.empty() ? 0 : 1, h63, -h63};
    comm.allreduce(std::span<std::int64_t>(agree), Op::max);
    if (agree[0] != 0) {
      throw std::invalid_argument(std::string(who) + ": " +
                                  (error.empty() ? "another rank's options are invalid" : error));
    }
    if (agree[1] != -agree[2]) {
      throw std::invalid_argument(std::string(who) + ": the ranks' aggregators or MPI-IO hints differ");
    }
  }
  const bool collective = soma_file::parallel_hdf5() && comm.size() > 1;
  const Ranks mine{comm.rank(), comm.rank() + 1, comm.size()};
  std::optional<File> f;
  Parts in;
  if (options.aggregators > 0) {
    Replica replica;
    collectively(comm, [&] { replica = gather_reads(comm, path, options); });
    collectively(comm, [&] {
      f.emplace(std::move(replica));
      in = read_file(comm, *f, mine);
    });
  } else {
    collectively(comm, [&] {
      const h5::Session quiet;
      try {
        f.emplace(comm, path, collective, options.hints);
        share_answers(comm, *f, [&] { (void)values_of(*f, read_file(comm, *f, mine, false).meta); });
        in = read_file(comm, *f, mine);
      } catch (const h5::Error& e) {
        bad_file("cannot read " + path.string() + ": " + e.what());
      }
    });
  }
  // A failure sending copies or building is reported with the next check.
  std::exception_ptr before, built;
  try {
    send_copies(comm, in);
  } catch (...) {
    before = std::current_exception();
  }
  HostMesh mesh = build(comm, in, options.ordering, before, built);
  if (options.adjacencies && in.meta.graphs != 0) {
    collectively(comm, [&] {
      if (built) std::rethrow_exception(std::exchange(built, nullptr));
    });
    build_distributed_adjacencies(comm, mesh, AdjacencyRequest::from_bits(in.meta.graphs));
  }
  collectively(comm, [&] {
    if (built) std::rethrow_exception(built);
    const h5::Session quiet;
    std::array<Lids, 4> lids;
    for (int d = 0; d <= mesh.dim(); ++d) lids[static_cast<std::size_t>(d)] = Lids(mesh, d);
    install_sets(mesh, in, lids);
    for (std::size_t i = 0; i < in.fields.size(); ++i) {
      dispatch_data_type(in.fields[i].type, [&]<class T>() { install_field<T>(mesh, in, i, lids); });
    }
    install_values(mesh, *f, in.meta);
  });
  f.reset();
  if (options.rebalance && in.meta.parts < static_cast<std::uint64_t>(comm.size())) {
    mesh = rebalance(comm, mesh, options.ordering);
  }

  return mesh;
}

HostMesh read_soma(const fs::path& path, const SomaReadOptions& options) {
  return read_soma(Comm::self(), path, options);
}

MeshFormat detect_format(const fs::path& path) {
  if (detail::legacy_vtk_header(path)) return MeshFormat::legacy_vtk;
  const h5::Session quiet;
  if (H5Fis_accessible(path.c_str(), H5P_DEFAULT) <= 0) return MeshFormat::unknown;
  const hid_t id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (id < 0) return MeshFormat::unknown;
  const Id file(id);
  if (H5Lexists(file.get(), soma_file::root, H5P_DEFAULT) > 0) {
    const hid_t root = H5Gopen2(file.get(), soma_file::root, H5P_DEFAULT);
    if (root < 0) return MeshFormat::unknown;
    const Id group(root);
    return H5Aexists(group.get(), "amr") > 0 ? MeshFormat::soma_amr : MeshFormat::soma;
  }
  if (H5Lexists(file.get(), "tstt", H5P_DEFAULT) > 0) return MeshFormat::legacy_h5m;
  return MeshFormat::unknown;
}

}  // namespace soma
