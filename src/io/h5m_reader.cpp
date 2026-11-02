// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hdf5.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/io/h5m.hpp"

namespace soma {

namespace {

namespace fs = std::filesystem;
using h5::Id;

class Failure : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& what) { throw Failure(what); }

std::string str(std::uint64_t v) { return std::to_string(v); }
std::string str(std::int64_t v) { return std::to_string(v); }

GO start_id(hid_t dataset, const std::string& where) {
  if (!h5::has_attribute(dataset, "start_id")) fail(where + " has no start_id attribute");
  const auto s = h5::read_attribute<std::int64_t>(dataset, "start_id", where);
  if (s < 1) fail(where + " has start_id " + str(s) + "; file ids start at 1");
  return s;
}

// File ids first .. first + count - 1, held by one dataset.
struct IdRange {
  GO first = 1;
  GO count = 0;
  int dim = -1;  // -1: entity sets
  std::string where;
};

struct Nodes {
  GO start = 1;
  GO count = 0;
  std::vector<Real> xyz;
};

Nodes read_nodes(hid_t tstt) {
  const std::string where = "/tstt/nodes/coordinates";
  if (!h5::exists(tstt, "nodes/coordinates")) fail("it has no " + where);
  const Id ds = h5::open_dataset(tstt, "nodes/coordinates");
  const auto ext = h5::extent(ds.get(), where);
  if (ext.size() != 2 || ext[1] < 1 || ext[1] > 3) {
    fail(where + " must have one row per vertex and 1 to 3 columns");
  }
  if (ext[0] == 0) fail(where + " holds no vertices");
  if (ext[0] > static_cast<hsize_t>(std::numeric_limits<LO>::max())) {
    fail(where + " holds " + str(std::uint64_t{ext[0]}) +
         " vertices, more than the LO type of this build can number");
  }
  if (ext[0] > std::numeric_limits<std::size_t>::max() / 3) fail(where + " holds more values than memory can index");
  Nodes n;
  n.count = static_cast<GO>(ext[0]);
  const auto rows = static_cast<std::size_t>(ext[0]);
  const auto cols = static_cast<std::size_t>(ext[1]);
  const auto raw = h5::read_all<double>(ds.get(), rows * cols, where);
  n.xyz.assign(3 * rows, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t c = 0; c < cols; ++c) n.xyz[3 * i + c] = raw[cols * i + c];
  }
  n.start = start_id(ds.get(), where);
  return n;
}

std::optional<Topology> topology_named(std::string_view name) {
  static constexpr std::array<std::pair<std::string_view, Topology>, 8> table{{
      {"Edge", Topology::edge},
      {"Tri", Topology::tri},
      {"Quad", Topology::quad},
      {"Polygon", Topology::polygon},
      {"Tet", Topology::tet},
      {"Pyramid", Topology::pyramid},
      {"Prism", Topology::prism},
      {"Hex", Topology::hex},
  }};
  for (const auto& [n, t] : table) {
    if (n == name) return t;
  }
  return std::nullopt;
}

// The name the file's element type enumeration gives the group's value.
std::string element_type(hid_t group, const std::string& where) {
  if (!h5::has_attribute(group, "element_type")) fail(where + " has no element_type attribute");
  const Id a = h5::check(H5Aopen(group, "element_type", H5P_DEFAULT),
                         "open the element_type of " + where);
  const Id t = h5::check(H5Aget_type(a.get()), "get the type of the element_type of " + where);
  if (H5Tget_class(t.get()) != H5T_ENUM) fail(where + ": element_type is not an enumeration");
  h5::require_one_value(a.get(), "the element_type of " + where);
  std::vector<unsigned char> value(H5Tget_size(t.get()));
  h5::check(H5Aread(a.get(), t.get(), value.data()), "read the element_type of " + where);
  std::array<char, 256> name{};
  if (H5Tenum_nameof(t.get(), value.data(), name.data(), name.size()) < 0) {
    fail(where + ": element_type holds a value its enumeration does not name");
  }
  return name.data();
}

struct Group {
  std::string where;
  Topology topology = Topology::edge;
  GO start = 1;
  GO count = 0;
  std::vector<Offset> offsets;  // polygons: count + 1
  std::vector<GO> conn;         // vertex file ids
};

Offset to_offset(std::uint64_t n, const std::string& where) {
  if (n > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    fail(where + " has " + str(n) + " entries, more than the Offset type of this build can index");
  }
  return static_cast<Offset>(n);
}

Group read_group(hid_t elements, const std::string& name, const Nodes& nodes) {
  Group g;
  g.where = "/tstt/elements/" + name;
  const Id grp = h5::open_group(elements, name);
  const std::string type = element_type(grp.get(), g.where);
  const auto t = topology_named(type);
  if (!t) fail(g.where + " holds " + type + " elements, which SOMA does not read");
  g.topology = *t;
  if (!h5::exists(grp.get(), "connectivity")) fail(g.where + " has no connectivity");
  const std::string cw = g.where + "/connectivity";
  const Id ds = h5::open_dataset(grp.get(), "connectivity");
  const auto ext = h5::extent(ds.get(), cw);

  std::uint64_t nnz = 0;
  std::uint64_t arity = 0;  // 0: variable
  if (h5::exists(grp.get(), "poly_indices")) {
    if (g.topology != Topology::polygon) {
      fail(g.where + " has poly_indices, but its elements are " + type + ", not polygons");
    }
    if (ext.size() != 1) fail(cw + " must be one-dimensional beside poly_indices");
    nnz = ext[0];
    const std::string pw = g.where + "/poly_indices";
    const Id pi = h5::open_dataset(grp.get(), "poly_indices");
    const auto pext = h5::extent(pi.get(), pw);
    if (pext.size() != 1) fail(pw + " must be one-dimensional");
    if (pext[0] > static_cast<hsize_t>(std::numeric_limits<LO>::max())) {
      fail(g.where + " holds more elements than the LO type of this build can number");
    }
    to_offset(nnz, cw);
    const auto last = h5::read_all<std::int64_t>(pi.get(), static_cast<std::size_t>(pext[0]), pw);
    g.count = static_cast<GO>(last.size());
    g.offsets.assign(last.size() + 1, 0);
    std::int64_t prev = -1;
    for (std::size_t i = 0; i < last.size(); ++i) {
      // prev is in [-1, nnz), so prev + 3 cannot overflow.
      if (last[i] < prev + 3) {
        fail(pw + ": polygon " + str(std::uint64_t{i}) + " ends at index " + str(last[i]) +
             ", leaving it fewer than 3 vertices");
      }
      if (static_cast<std::uint64_t>(last[i]) >= nnz) {
        fail(pw + ": polygon " + str(std::uint64_t{i}) + " ends at index " + str(last[i]) + ", but " + cw +
             " has " + str(nnz) + " entries");
      }
      prev = last[i];
      g.offsets[i + 1] = to_offset(static_cast<std::uint64_t>(prev) + 1, cw);
    }
    if (static_cast<std::uint64_t>(prev + 1) != nnz) {
      fail(pw + " ends at index " + str(prev) + ", but " + cw + " has " + str(nnz) + " entries");
    }
  } else {
    if (ext.size() != 2) fail(cw + " must be two-dimensional (elements by vertices)");
    if (ext[0] > static_cast<hsize_t>(std::numeric_limits<LO>::max())) {
      fail(g.where + " holds more elements than the LO type of this build can number");
    }
    if (ext[1] != 0 && ext[0] > std::numeric_limits<std::uint64_t>::max() / ext[1]) {
      fail(cw + " has more entries than an index can count");
    }
    to_offset(ext[0] * ext[1], cw);  // so that i * arity below fits too
    g.count = static_cast<GO>(ext[0]);
    arity = ext[1];
    if (g.topology == Topology::polygon) {
      if (arity < 3) fail(cw + " has " + str(arity) + " vertices per polygon; polygons need 3");
      g.offsets.assign(static_cast<std::size_t>(g.count) + 1, 0);
      for (std::size_t i = 0; i <= static_cast<std::size_t>(g.count); ++i) {
        g.offsets[i] = to_offset(i * arity, cw);
      }
    } else if (arity != static_cast<std::uint64_t>(num_vertices(g.topology))) {
      fail(cw + " has " + str(arity) + " vertices per " + type + "; SOMA reads " +
           std::to_string(num_vertices(g.topology)) + "-vertex " + type +
           " elements (high-order connectivity is not supported)");
    }
    nnz = ext[0] * ext[1];
  }
  if (g.count == 0) return g;
  g.start = start_id(ds.get(), cw);

  const auto ids = h5::read_all<std::uint64_t>(ds.get(), static_cast<std::size_t>(nnz), cw);
  const auto first = static_cast<std::uint64_t>(nodes.start);
  const auto end = first + static_cast<std::uint64_t>(nodes.count);
  g.conn.resize(ids.size());
  for (std::size_t j = 0; j < ids.size(); ++j) {
    if (ids[j] < first || ids[j] >= end) {
      const auto row = arity != 0 ? j / arity
                                  : static_cast<std::size_t>(std::upper_bound(g.offsets.begin(), g.offsets.end(),
                                                                              static_cast<Offset>(j)) -
                                                             g.offsets.begin() - 1);
      fail(cw + ": element " + str(static_cast<std::uint64_t>(g.start) + row) + " refers to file id " + str(ids[j]) +
           ", which is not a vertex");
    }
    g.conn[j] = static_cast<GO>(ids[j]);
  }
  return g;
}

void check_ranges(std::vector<IdRange> ranges, std::uint64_t max_id) {
  std::erase_if(ranges, [](const IdRange& r) { return r.count == 0; });
  std::sort(ranges.begin(), ranges.end(),
            [](const IdRange& a, const IdRange& b) { return a.first < b.first; });
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const IdRange& r = ranges[i];
    if (r.first > std::numeric_limits<GO>::max() - r.count) fail(r.where + " has file ids beyond the gid type");
    const auto last = static_cast<std::uint64_t>(r.first + r.count - 1);
    if (last > max_id) {
      fail(r.where + " has file ids up to " + str(last) + ", beyond the max_id " + str(max_id) + " of /tstt");
    }
    if (i > 0) {
      const IdRange& p = ranges[i - 1];
      if (r.first <= p.first + p.count - 1) {
        fail(p.where + " and " + r.where + " both claim file id " + str(r.first));
      }
    }
  }
}

// File id -> (dimension, local id) through each dimension's (gid, lid) pairs.
class Locator {
 public:
  Locator(const HostMesh& mesh, std::vector<IdRange> ranges) : ranges_(std::move(ranges)) {
    std::erase_if(ranges_, [](const IdRange& r) { return r.count == 0; });
    std::sort(ranges_.begin(), ranges_.end(),
              [](const IdRange& a, const IdRange& b) { return a.first < b.first; });
    for (int d = 0; d <= mesh.dim(); ++d) {
      const auto gid = mesh.gid(d);
      auto& p = pairs_[static_cast<std::size_t>(d)];
      p.resize(gid.extent(0));
      for (std::size_t l = 0; l < p.size(); ++l) p[l] = {gid(l), static_cast<LO>(l)};
      if (!std::is_sorted(p.begin(), p.end())) std::sort(p.begin(), p.end());
    }
  }

  // The range holding id, or nullptr.
  const IdRange* range_of(GO id) const {
    auto it = std::upper_bound(ranges_.begin(), ranges_.end(), id,
                               [](GO v, const IdRange& r) { return v < r.first; });
    if (it == ranges_.begin()) return nullptr;
    --it;
    return id < it->first + it->count ? &*it : nullptr;
  }

  LO lid(int dim, GO id) const {
    const auto& p = pairs_[static_cast<std::size_t>(dim)];
    const auto it = std::lower_bound(p.begin(), p.end(), std::pair<GO, LO>{id, 0});
    return it->second;
  }

  // Local ids of the count entities of dimension dim from file id first on.
  std::vector<LO> lids(int dim, GO first, GO count) const {
    const auto& p = pairs_[static_cast<std::size_t>(dim)];
    auto it = std::lower_bound(p.begin(), p.end(), std::pair<GO, LO>{first, 0});
    std::vector<LO> out(static_cast<std::size_t>(count));
    for (auto& l : out) l = (it++)->second;
    return out;
  }

 private:
  std::vector<IdRange> ranges_;
  std::array<std::vector<std::pair<GO, LO>>, 4> pairs_;
};

// How a tag's values are held in memory.
struct TagKind {
  DataType type = DataType::u8;
  int ncomp = 1;
  Id mem;
};

std::optional<DataType> scalar_type(hid_t t, std::string& why) {
  const std::size_t size = H5Tget_size(t);
  switch (H5Tget_class(t)) {
    case H5T_INTEGER: {
      const bool sign = H5Tget_sign(t) == H5T_SGN_2;
      if (size == 1) return sign ? DataType::i8 : DataType::u8;
      if (size == 4) return sign ? DataType::i32 : DataType::u32;
      if (size == 8) return sign ? DataType::i64 : DataType::u64;
      why = std::to_string(8 * size) + "-bit integer values are not read";
      return std::nullopt;
    }
    case H5T_FLOAT:
      if (size == 4) return DataType::f32;
      if (size == 8) return DataType::f64;
      why = std::to_string(8 * size) + "-bit real values are not read";
      return std::nullopt;
    default:
      why = "values of this HDF5 type class are not read";
      return std::nullopt;
  }
}

hid_t native_of(DataType t) {
  hid_t out = H5I_INVALID_HID;
  dispatch_data_type(t, [&]<class T>() { out = h5::native<T>(); });
  return out;
}

std::optional<TagKind> tag_kind(hid_t type, std::string& why) {
  TagKind k;
  const std::size_t size = H5Tget_size(type);
  switch (H5Tget_class(type)) {
    case H5T_INTEGER:
    case H5T_FLOAT: {
      const auto s = scalar_type(type, why);
      if (!s) return std::nullopt;
      k.type = *s;
      k.mem = h5::check(H5Tcopy(native_of(*s)), "copy a native type");
      return k;
    }
    case H5T_ARRAY: {
      const Id base = h5::check(H5Tget_super(type), "get the base of an array type");
      const auto s = scalar_type(base.get(), why);
      if (!s) return std::nullopt;
      const int rank = H5Tget_array_ndims(type);
      if (rank < 1) fail("an array tag type has no dimensions");
      std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
      h5::check(H5Tget_array_dims2(type, dims.data()), "get the dimensions of an array type");
      const hsize_t n = std::accumulate(dims.begin(), dims.end(), hsize_t{1}, std::multiplies<>());
      if (n < 1 || n > static_cast<hsize_t>(std::numeric_limits<int>::max())) {
        why = "the array size is out of range";
        return std::nullopt;
      }
      k.type = *s;
      k.ncomp = static_cast<int>(n);
      // HDF5 converts between array types of the same shape only.
      k.mem = h5::check(H5Tarray_create2(native_of(*s), static_cast<unsigned>(rank), dims.data()),
                        "create an array type");
      return k;
    }
    case H5T_OPAQUE:
      if (size < 1 || size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        why = "the opaque size is out of range";
        return std::nullopt;
      }
      k.type = DataType::u8;
      k.ncomp = static_cast<int>(size);
      k.mem = h5::check(H5Tcopy(type), "copy an opaque type");
      return k;
    case H5T_BITFIELD:
      if (size != 1) {
        why = "bit tags wider than 8 bits are not read";
        return std::nullopt;
      }
      k.type = DataType::u8;
      k.mem = h5::check(H5Tcopy(H5T_NATIVE_B8), "copy the bitfield type");
      return k;
    default:
      why = "values of this HDF5 type class are not read";
      return std::nullopt;
  }
}

// A dense tag dataset: one row per entity of the group whose file ids start at first.
struct Dense {
  std::string path;  // below /tstt
  int dim = 0;
  GO first = 1;
  GO count = 0;
};

template <class T>
std::vector<T> read_rows(hid_t loc, const std::string& path, const TagKind& k, GO rows) {
  const std::string where = "/tstt/" + path;
  const Id ds = h5::open_dataset(loc, path);
  const auto ext = h5::extent(ds.get(), where);
  if (ext.size() != 1 || ext[0] != static_cast<hsize_t>(rows)) {
    fail(where + " must hold one value per entity (" + str(rows) + ")");
  }
  std::vector<T> out(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k.ncomp));
  if (!out.empty()) h5::read(ds.get(), k.mem.get(), out.data(), where);
  return out;
}

template <class T>
std::optional<std::vector<T>> read_value_attribute(hid_t tag, const char* name, const TagKind& k,
                                                   const std::string& where) {
  if (!h5::has_attribute(tag, name)) return std::nullopt;
  const Id a = h5::check(H5Aopen(tag, name, H5P_DEFAULT), std::string("open the ") + name + " of " + where);
  h5::require_one_value(a.get(), std::string("the ") + name + " of " + where);
  std::vector<T> v(static_cast<std::size_t>(k.ncomp));
  h5::check(H5Aread(a.get(), k.mem.get(), v.data()), std::string("read the ") + name + " of " + where);
  return v;
}

template <class T>
void import_tag(HostMesh& mesh, hid_t tstt, const std::string& name, const TagKind& k,
                const std::vector<Dense>& dense, const Locator& at, H5mReport& report) {
  const std::string where = "/tstt/tags/" + name;
  const auto nc = static_cast<std::size_t>(k.ncomp);
  std::array<std::vector<LO>, 4> lids;
  std::array<std::vector<T>, 4> values;
  std::uint64_t on_sets = 0;

  for (const Dense& s : dense) {
    if (s.dim < 0) {
      on_sets += static_cast<std::uint64_t>(s.count);
      continue;
    }
    std::vector<T> v = read_rows<T>(tstt, s.path, k, s.count);
    const auto d = static_cast<std::size_t>(s.dim);
    const auto l = at.lids(s.dim, s.first, s.count);
    lids[d].insert(lids[d].end(), l.begin(), l.end());
    values[d].insert(values[d].end(), v.begin(), v.end());
  }

  const Id tag = h5::open_group(tstt, "tags/" + name);
  if (h5::exists(tag.get(), "id_list")) {
    const std::string iw = where + "/id_list";
    const Id ids_ds = h5::open_dataset(tag.get(), "id_list");
    const auto ext = h5::extent(ids_ds.get(), iw);
    if (ext.size() != 1) fail(iw + " must be one-dimensional");
    const auto n = static_cast<GO>(ext[0]);
    const auto ids = h5::read_all<std::uint64_t>(ids_ds.get(), static_cast<std::size_t>(n), iw);
    if (!h5::exists(tag.get(), "values")) fail(where + " has an id_list but no values");
    const std::vector<T> v = read_rows<T>(tstt, "tags/" + name + "/values", k, n);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      const IdRange* r = ids[i] <= static_cast<std::uint64_t>(std::numeric_limits<GO>::max())
                             ? at.range_of(static_cast<GO>(ids[i]))
                             : nullptr;
      if (r == nullptr) fail(iw + " lists file id " + str(ids[i]) + ", which names no entity");
      if (r->dim < 0) {
        ++on_sets;
        continue;
      }
      const auto d = static_cast<std::size_t>(r->dim);
      lids[d].push_back(at.lid(r->dim, static_cast<GO>(ids[i])));
      values[d].insert(values[d].end(), v.begin() + static_cast<std::ptrdiff_t>(i * nc),
                       v.begin() + static_cast<std::ptrdiff_t>((i + 1) * nc));
    }
  }

  FieldOptions<T> options;
  options.default_value = read_value_attribute<T>(tag.get(), "default", k, where);
  options.global_value = read_value_attribute<T>(tag.get(), "global", k, where);

  bool any = false;
  for (int d = 0; d <= mesh.dim(); ++d) {
    auto& l = lids[static_cast<std::size_t>(d)];
    if (l.empty()) continue;
    any = true;
    std::vector<std::size_t> order(l.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    if (!std::is_sorted(l.begin(), l.end())) {
      std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return l[a] < l[b]; });
    }
    std::vector<LO> sorted(l.size());
    for (std::size_t i = 0; i < order.size(); ++i) sorted[i] = l[order[i]];
    for (std::size_t i = 1; i < sorted.size(); ++i) {
      if (sorted[i] == sorted[i - 1]) {
        fail(where + " has two values on file id " + str(mesh.gid(d)(static_cast<std::size_t>(sorted[i]))));
      }
    }
    // A list of every entity of d is a full domain.
    const auto domain = Domain<Kokkos::HostSpace>::from_list(mesh, d, sorted);
    const auto f = mesh.fields().add<T>(name, Space<Kokkos::HostSpace>(domain, Layout::per_entity()),
                                        k.ncomp, options);
    // A per_entity field has one row per member, in member order.
    const auto view = f.view();
    const auto& src = values[static_cast<std::size_t>(d)];
    for (std::size_t i = 0; i < order.size(); ++i) {
      for (std::size_t c = 0; c < nc; ++c) view(i, c) = src[order[i] * nc + c];
    }
  }
  if (on_sets > 0) {
    report.skipped.push_back({name, str(on_sets) + " values on entity sets are not read yet"});
  } else if (!any) {
    report.skipped.push_back({name, "it has no values on vertices or elements"});
  }
}

void import_tags(HostMesh& mesh, hid_t tstt, const Nodes& nodes, const std::vector<Group>& groups,
                 const std::vector<std::string>& group_names, const IdRange& sets,
                 const Locator& at, H5mReport& report) {
  std::map<std::string, std::vector<Dense>> dense;
  const auto collect = [&](const std::string& base, int dim, GO first, GO count) {
    if (!h5::exists(tstt, base + "/tags")) return;
    const Id g = h5::open_group(tstt, base + "/tags");
    for (const std::string& n : h5::children(g.get(), "/tstt/" + base + "/tags")) {
      dense[n].push_back({base + "/tags/" + n, dim, first, count});
    }
  };
  collect("nodes", 0, nodes.start, nodes.count);
  for (std::size_t i = 0; i < groups.size(); ++i) {
    collect("elements/" + group_names[i], dimension(groups[i].topology), groups[i].start, groups[i].count);
  }
  collect("sets", -1, sets.first, sets.count);

  std::vector<std::string> names;
  if (h5::exists(tstt, "tags")) {
    const Id tags = h5::open_group(tstt, "tags");
    names = h5::children(tags.get(), "/tstt/tags");
  }
  for (const auto& [n, sources] : dense) {
    if (!std::binary_search(names.begin(), names.end(), n)) {
      fail("/tstt/" + sources.front().path + " holds values of a tag /tstt/tags does not define");
    }
  }

  for (const std::string& name : names) {
    const std::string where = "/tstt/tags/" + name;
    const Id tag = h5::open_group(tstt, "tags/" + name);
    if (h5::has_attribute(tag.get(), "variable_length") &&
        h5::read_attribute<std::int32_t>(tag.get(), "variable_length", where) != 0) {
      report.skipped.push_back({name, "variable-length tags are not read yet"});
      continue;
    }
    if (h5::has_attribute(tag.get(), "is_handle") &&
        h5::read_attribute<std::int32_t>(tag.get(), "is_handle", where) != 0) {
      report.skipped.push_back({name, "handle-valued tags are not read yet"});
      continue;
    }
    if (!h5::exists(tag.get(), "type")) fail(where + " has no type");
    const Id type = h5::check(H5Topen2(tag.get(), "type", H5P_DEFAULT), "open the type of " + where);
    std::string why;
    const auto kind = tag_kind(type.get(), why);
    if (!kind) {
      report.skipped.push_back({name, why});
      continue;
    }
    const auto it = dense.find(name);
    const std::vector<Dense> none;
    const std::vector<Dense>& sources = it != dense.end() ? it->second : none;
    dispatch_data_type(kind->type, [&]<class T>() {
      import_tag<T>(mesh, tstt, name, *kind, sources, at, report);
    });
  }
}

HostMesh read(const fs::path& path, const H5mReadOptions& options, H5mReport& report) {
  const h5::Session session;
  if (H5Fis_accessible(path.c_str(), H5P_DEFAULT) <= 0) fail("it is not an HDF5 file, or cannot be opened");
  const Id file = h5::check(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), "open the file");
  if (!h5::exists(file.get(), "tstt")) fail("it has no /tstt group, so it is not a MOAB .h5m file");
  const Id tstt = h5::open_group(file.get(), "tstt");
  if (!h5::has_attribute(tstt.get(), "max_id")) fail("/tstt has no max_id attribute");
  const auto max_id = h5::read_attribute<std::uint64_t>(tstt.get(), "max_id", "/tstt");

  const Nodes nodes = read_nodes(tstt.get());
  std::vector<Group> groups;
  std::vector<std::string> names;
  if (h5::exists(tstt.get(), "elements")) {
    const Id el = h5::open_group(tstt.get(), "elements");
    names = h5::children(el.get(), "/tstt/elements");
    for (const std::string& n : names) groups.push_back(read_group(el.get(), n, nodes));
  }
  int dim = 0;
  for (const Group& g : groups) {
    if (g.count > 0) dim = std::max(dim, dimension(g.topology));
  }
  if (dim == 0) fail("it has no elements; a mesh needs at least one");

  IdRange sets{1, 0, -1, "/tstt/sets/list"};
  if (h5::exists(tstt.get(), "sets/list")) {
    const Id ds = h5::open_dataset(tstt.get(), "sets/list");
    const auto ext = h5::extent(ds.get(), sets.where);
    if (ext.size() != 2 || ext[1] != 4) fail(sets.where + " must have 4 columns");
    if (ext[0] > static_cast<hsize_t>(std::numeric_limits<LO>::max())) {
      fail(sets.where + " holds more sets than the LO type of this build can number");
    }
    sets.count = static_cast<GO>(ext[0]);
    if (sets.count > 0) sets.first = start_id(ds.get(), sets.where);
  }
  std::vector<IdRange> ranges{{nodes.start, nodes.count, 0, "/tstt/nodes/coordinates"}, sets};
  for (const Group& g : groups) ranges.push_back({g.start, g.count, dimension(g.topology), g.where});
  check_ranges(ranges, max_id);

  MeshBuilder b(dim);
  {
    std::vector<GO> gids(static_cast<std::size_t>(nodes.count));
    std::iota(gids.begin(), gids.end(), nodes.start);
    b.add_vertices(gids, nodes.xyz);
  }
  for (Group& g : groups) {
    if (g.count == 0) continue;
    std::vector<GO> gids(static_cast<std::size_t>(g.count));
    std::iota(gids.begin(), gids.end(), g.start);
    if (g.topology == Topology::polygon) {
      b.add_polygons(gids, g.offsets, g.conn);
    } else {
      b.add_elements(g.topology, gids, g.conn);
    }
    std::vector<GO>().swap(g.conn);
  }
  HostMesh mesh = std::move(b).build(options.ordering);
  if (options.tags && h5::exists(tstt.get(), "tags")) {
    const Locator at(mesh, ranges);
    import_tags(mesh, tstt.get(), nodes, groups, names, sets, at, report);
  }
  return mesh;
}

}  // namespace

bool h5m_available() noexcept { return true; }

HostMesh read_h5m(const fs::path& path, const H5mReadOptions& options, H5mReport& report) {
  report = {};
  try {
    return read(path, options, report);
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception& e) {
    throw std::runtime_error("read_h5m: " + path.string() + ": " + e.what());
  }
}

HostMesh read_h5m(const fs::path& path, const H5mReadOptions& options) {
  H5mReport report;
  return read_h5m(path, options, report);
}

}  // namespace soma
