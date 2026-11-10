// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_IO_H5M_READ_HPP_
#define SOMA_SRC_IO_H5M_READ_HPP_

// Pieces of the .h5m reader shared by the serial and the parallel read,
// private to the library.

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/dense_ids.hpp"
#include "hdf5.hpp"
#include "soma/core/handle_ref.hpp"
#include "soma/core/lid_runs.hpp"
#include "soma/core/mesh.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/topology/nodes.hpp"

namespace soma::h5m_read {

using h5::Id;

class Failure : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] inline void fail(const std::string& what) { throw Failure(what); }

inline std::string str(std::uint64_t v) { return std::to_string(v); }
inline std::string str(std::int64_t v) { return std::to_string(v); }

inline std::string hex(std::int64_t v) {
  std::array<char, 24> buf{};
  const auto end = std::to_chars(buf.data(), buf.data() + buf.size(), v, 16).ptr;
  std::string digits(buf.data(), end);
  if (digits.size() < 2) digits.insert(0, 2 - digits.size(), '0');
  return v < 0 ? "-0x" + digits.substr(1) : "0x" + digits;
}

inline GO start_id(hid_t dataset, const std::string& where) {
  if (!h5::has_attribute(dataset, "start_id")) fail(where + " has no start_id attribute");
  const auto s = h5::read_attribute<std::int64_t>(dataset, "start_id", where);
  if (s < 1) fail(where + " has start_id " + str(s) + "; file ids start at 1");
  return s;
}

// File ids first .. first + count - 1, held by one dataset.
struct IdRange {
  GO first = 1;
  GO count = 0;
  int dim = set_dim;  // set_dim: entity sets
  std::string where;
};

inline std::optional<Topology> topology_named(std::string_view name) {
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
inline std::string element_type(hid_t group, const std::string& where) {
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
  std::uint64_t arity = 0;      // entries per element; 0 for a poly_indices list
  std::uint64_t nnz = 0;        // connectivity entries
  std::vector<Offset> offsets;  // polygons: count + 1 (of the rows read)
  std::vector<GO> conn;         // vertex file ids
  int nodes = 0;                // an element's nodes, corners first; 0 for polygons
};

inline Offset to_offset(std::uint64_t n, const std::string& where) {
  if (n > static_cast<std::uint64_t>(std::numeric_limits<Offset>::max())) {
    fail(where + " has " + str(n) + " entries, more than the Offset type of this build can index");
  }
  return static_cast<Offset>(n);
}

// The vertices' file ids and coordinate columns, checked.
struct NodeInfo {
  GO start = 1;
  GO count = 0;
  std::size_t columns = 3;
};

inline NodeInfo describe_nodes(hid_t tstt) {
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
  NodeInfo n;
  n.count = static_cast<GO>(ext[0]);
  n.columns = static_cast<std::size_t>(ext[1]);
  n.start = start_id(ds.get(), where);
  return n;
}

// A group's type, extents and start id, checked, without its connectivity.
inline Group describe_group(hid_t elements, const std::string& name) {
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
  if (h5::exists(grp.get(), "poly_indices")) {
    if (g.topology != Topology::polygon) {
      fail(g.where + " has poly_indices, but its elements are " + type + ", not polygons");
    }
    if (ext.size() != 1) fail(cw + " must be one-dimensional beside poly_indices");
    g.nnz = ext[0];
    const std::string pw = g.where + "/poly_indices";
    const Id pi = h5::open_dataset(grp.get(), "poly_indices");
    const auto pext = h5::extent(pi.get(), pw);
    if (pext.size() != 1) fail(pw + " must be one-dimensional");
    if (pext[0] > static_cast<hsize_t>(std::numeric_limits<LO>::max())) {
      fail(g.where + " holds more elements than the LO type of this build can number");
    }
    to_offset(g.nnz, cw);
    g.count = static_cast<GO>(pext[0]);
  } else {
    if (ext.size() != 2) fail(cw + " must be two-dimensional (elements by vertices)");
    if (ext[0] > static_cast<hsize_t>(std::numeric_limits<LO>::max())) {
      fail(g.where + " holds more elements than the LO type of this build can number");
    }
    if (ext[1] != 0 && ext[0] > std::numeric_limits<std::uint64_t>::max() / ext[1]) {
      fail(cw + " has more entries than an index can count");
    }
    to_offset(ext[0] * ext[1], cw);  // so that i * arity fits too
    g.count = static_cast<GO>(ext[0]);
    g.arity = ext[1];
    if (g.topology == Topology::polygon) {
      if (g.arity < 3) fail(cw + " has " + str(g.arity) + " vertices per polygon; polygons need 3");
    } else if (g.arity > 27 || mid_node_bits(g.topology, static_cast<int>(g.arity)) < 0) {
      fail(cw + " has " + str(g.arity) + " nodes per " + type + "; no " + type +
           " has that many (its corners, with or without mid-edge, mid-face and mid-region nodes)");
    } else {
      g.nodes = static_cast<int>(g.arity);
    }
    g.nnz = ext[0] * ext[1];
  }
  if (g.count > 0) g.start = start_id(ds.get(), cw);
  return g;
}

// Offsets of polygons first .. first + last.size() - 1 of a poly_indices
// list from their last indices, `prev` the last index of the polygon before
// (-1 for none); `final` when the last of them is the group's last.
inline std::vector<Offset> polygon_offsets(const Group& g, std::span<const std::int64_t> last, std::uint64_t first,
                                           std::int64_t prev, bool final) {
  const std::string cw = g.where + "/connectivity";
  const std::string pw = g.where + "/poly_indices";
  std::vector<Offset> offsets(last.size() + 1, 0);
  offsets[0] = to_offset(static_cast<std::uint64_t>(prev + 1), cw);
  for (std::size_t i = 0; i < last.size(); ++i) {
    // prev is in [-1, nnz), so prev + 3 cannot overflow.
    if (last[i] < prev + 3) {
      fail(pw + ": polygon " + str(first + i) + " ends at index " + str(last[i]) +
           ", leaving it fewer than 3 vertices");
    }
    if (static_cast<std::uint64_t>(last[i]) >= g.nnz) {
      fail(pw + ": polygon " + str(first + i) + " ends at index " + str(last[i]) + ", but " + cw + " has " +
           str(g.nnz) + " entries");
    }
    prev = last[i];
    offsets[i + 1] = to_offset(static_cast<std::uint64_t>(prev) + 1, cw);
  }
  if (final && static_cast<std::uint64_t>(prev + 1) != g.nnz) {
    fail(pw + " ends at index " + str(prev) + ", but " + cw + " has " + str(g.nnz) + " entries");
  }
  return offsets;
}

// Connectivity entries `ids` of rows first on of a group (offsets as from
// polygon_offsets for a poly_indices list, which start at ids[0]), checked to
// be vertex file ids.
inline std::vector<GO> vertex_ids(const Group& g, std::span<const std::uint64_t> ids, std::uint64_t first,
                                  std::span<const Offset> offsets, GO vertex_start, GO vertex_count) {
  const auto lo = static_cast<std::uint64_t>(vertex_start);
  const auto end = lo + static_cast<std::uint64_t>(vertex_count);
  std::vector<GO> conn(ids.size());
  for (std::size_t j = 0; j < ids.size(); ++j) {
    if (ids[j] < lo || ids[j] >= end) {
      std::uint64_t row = 0;
      if (g.arity != 0) {
        row = j / g.arity;
      } else {
        const auto at = static_cast<Offset>(j) + offsets.front();
        row = static_cast<std::uint64_t>(std::upper_bound(offsets.begin(), offsets.end(), at) - offsets.begin() - 1);
      }
      fail(g.where + "/connectivity: element " + str(static_cast<std::uint64_t>(g.start) + first + row) +
           " refers to file id " + str(ids[j]) + ", which is not a vertex");
    }
    conn[j] = static_cast<GO>(ids[j]);
  }
  return conn;
}

inline void check_ranges(std::vector<IdRange> ranges, std::uint64_t max_id) {
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

// File id -> (slot, local id) through each dimension's (gid, lid) pairs; a
// set's local id is its index, its file id less the first set's.
class Locator {
 public:
  Locator(const HostMesh& mesh, std::vector<IdRange> ranges) : ranges_(std::move(ranges)) {
    std::erase_if(ranges_, [](const IdRange& r) { return r.count == 0; });
    std::sort(ranges_.begin(), ranges_.end(),
              [](const IdRange& a, const IdRange& b) { return a.first < b.first; });
    for (const IdRange& r : ranges_) {
      if (r.dim == set_dim) set_first_ = r.first;
    }
    for (int d = 0; d <= mesh.dim(); ++d) {
      const auto gid = mesh.gid(d);
      const auto n = static_cast<std::size_t>(gid.extent(0));
      auto& p = pairs_[static_cast<std::size_t>(d)];
      p.resize(n);
      // By gid: through a table over the gids' range when it is at most twice
      // their count (kept for lid), else by sorting.
      const auto [lo, span] = n == 0 ? std::pair<GO, std::uint64_t>{0, 0}
                                     : detail::value_range<GO>(n, [&](std::size_t l) { return gid(l); });
      if (n > 0 && span < 2 * static_cast<std::uint64_t>(n)) {
        auto& table = table_[static_cast<std::size_t>(d)];
        table_lo_[static_cast<std::size_t>(d)] = lo;
        table.assign(static_cast<std::size_t>(span) + 1, invalid_lo);
        for (std::size_t l = 0; l < n; ++l) table[static_cast<std::size_t>(gid(l) - lo)] = static_cast<LO>(l);
        std::size_t k = 0;
        for (std::size_t g = 0; g < table.size(); ++g) {
          if (table[g] != invalid_lo) p[k++] = {lo + static_cast<GO>(g), table[g]};
        }
      } else {
        for (std::size_t l = 0; l < n; ++l) p[l] = {gid(l), static_cast<LO>(l)};
        if (!std::is_sorted(p.begin(), p.end())) std::sort(p.begin(), p.end());
      }
      auto& sg = segs_[static_cast<std::size_t>(d)];
      for (const auto& [g, l] : p) {
        if (!sg.empty() && g == sg.back().gid + sg.back().len && l == sg.back().lid + sg.back().len) {
          ++sg.back().len;
        } else {
          sg.push_back({g, l, 1});
        }
      }
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
    if (dim == set_dim) return static_cast<LO>(id - set_first_);
    const auto& table = table_[static_cast<std::size_t>(dim)];
    if (!table.empty()) return table[static_cast<std::size_t>(id - table_lo_[static_cast<std::size_t>(dim)])];
    const auto& p = pairs_[static_cast<std::size_t>(dim)];
    const auto it = std::lower_bound(p.begin(), p.end(), std::pair<GO, LO>{id, 0});
    return it->second;
  }

  // Local ids of the count entities of dimension dim from file id first on,
  // as runs (not sorted); every id of the range must name an entity. Costs the
  // runs, not the entities: one run when the lids follow the file ids.
  std::vector<Interval> lid_runs(int dim, GO first, GO count) const {
    const auto& sg = segs_[static_cast<std::size_t>(dim)];
    auto it = std::upper_bound(sg.begin(), sg.end(), first, [](GO v, const Seg& s) { return v < s.gid; });
    --it;
    std::vector<Interval> out;
    for (GO id = first; id < first + count; ++it) {
      const GO end = std::min(first + count, it->gid + it->len);
      const auto l = static_cast<LO>(it->lid + (id - it->gid));
      out.push_back(Interval{l, static_cast<LO>(l + (end - id))});
      id = end;
    }
    return out;
  }

  // Local ids of the count entities of dimension dim from file id first on.
  std::vector<LO> lids(int dim, GO first, GO count) const {
    if (dim == set_dim) {
      std::vector<LO> out(static_cast<std::size_t>(count));
      std::iota(out.begin(), out.end(), static_cast<LO>(first - set_first_));
      return out;
    }
    const auto& p = pairs_[static_cast<std::size_t>(dim)];
    auto it = std::lower_bound(p.begin(), p.end(), std::pair<GO, LO>{first, 0});
    std::vector<LO> out(static_cast<std::size_t>(count));
    for (auto& l : out) l = (it++)->second;
    return out;
  }

 private:
  // gid gid .. gid + len - 1 at lids lid .. lid + len - 1.
  struct Seg {
    GO gid;
    LO lid;
    LO len;
  };
  std::vector<IdRange> ranges_;
  std::array<std::vector<std::pair<GO, LO>>, 4> pairs_;
  std::array<std::vector<LO>, 4> table_;  // lid by gid - table_lo_, over a dense range
  std::array<GO, 4> table_lo_{};
  std::array<std::vector<Seg>, 4> segs_;
  GO set_first_ = 1;
};

// How a tag's values are held in memory.
struct TagKind {
  DataType type = DataType::u8;
  int ncomp = 1;
  Id mem;
  TagFormat::Kind kind = TagFormat::Kind::plain;
  int nbits = 0;
};

inline std::optional<DataType> scalar_type(hid_t t, std::string& why) {
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

inline hid_t native_of(DataType t) {
  hid_t out = H5I_INVALID_HID;
  dispatch_data_type(t, [&]<class T>() { out = h5::native<T>(); });
  return out;
}

inline std::optional<TagKind> tag_kind(hid_t type, std::string& why) {
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
      k.kind = TagFormat::Kind::opaque;
      return k;
    case H5T_BITFIELD:
      if (size != 1) {
        why = "bit tags wider than 8 bits are not read";
        return std::nullopt;
      }
      k.type = DataType::u8;
      k.mem = h5::check(H5Tcopy(H5T_NATIVE_B8), "copy the bitfield type");
      k.kind = TagFormat::Kind::bit;
      k.nbits = static_cast<int>(H5Tget_precision(type));  // 1 to 8 in one byte
      return k;
    default:
      why = "values of this HDF5 type class are not read";
      return std::nullopt;
  }
}

// Handle values read as GO: the tag's shape over a native signed 64-bit type.
// A file id beyond the GO range converts to its largest value, which names
// nothing and is rejected with the others.
inline TagKind handle_kind(hid_t type, const TagKind& k, const std::string& where) {
  if (k.type != DataType::u64 && k.type != DataType::i64) {
    fail(where + " holds handles, but its values are " + std::string(name(k.type)) + ", not 64-bit file ids");
  }
  TagKind h;
  h.type = DataType::i64;
  h.ncomp = k.ncomp;
  if (H5Tget_class(type) == H5T_ARRAY) {
    const int rank = H5Tget_array_ndims(type);
    std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
    h5::check(H5Tget_array_dims2(type, dims.data()), "get the dimensions of an array type");
    h.mem = h5::check(H5Tarray_create2(H5T_NATIVE_INT64, static_cast<unsigned>(rank), dims.data()),
                      "create an array type");
  } else {
    h.mem = h5::check(H5Tcopy(H5T_NATIVE_INT64), "copy a native type");
  }
  return h;
}

// Handle values in the file are 0 (no entity) or the file id of an entity or
// set; each becomes make_handle(slot, file id), the file id being the gid.
// owner(i) names the holder of value i.
template <class T, class Owner>
void to_handles(std::span<T> v, const Locator& at, const std::string& where, Owner&& owner) {
  if constexpr (std::is_same_v<T, GO>) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (v[i] == 0) continue;
      const IdRange* r = at.range_of(v[i]);
      if (r == nullptr) {
        fail(where + ": " + owner(i) + " holds the handle " + str(std::int64_t{v[i]}) +
             ", which names no entity or set");
      }
      const GO h = make_handle(r->dim, v[i]);
      if (h == 0) {
        fail(where + ": " + owner(i) + " holds the handle " + str(std::int64_t{v[i]}) +
             ", a file id beyond the gids a handle can hold");
      }
      v[i] = h;
    }
  }
}

template <class T>
void to_handle_options(FieldOptions<T>& o, const Locator& at, const std::string& where) {
  const auto named = [](const char* what) { return [what](std::size_t) { return std::string("the ") + what; }; };
  if (o.default_value) to_handles<T>(*o.default_value, at, where, named("default"));
  if (o.global_value) to_handles<T>(*o.global_value, at, where, named("global"));
}

// A tag with values on no entity, kept with the mesh.
template <class T>
MeshValue mesh_value(const FieldOptions<T>& o, int ncomp, const TagFormat& format) {
  MeshValue v;
  v.type = data_type_of<T>();
  v.ncomp = ncomp;
  v.handle = o.handle;
  v.format = format;
  if (o.default_value) v.default_value = MeshValue::bytes_of<T>(*o.default_value);
  if (o.global_value) v.global_value = MeshValue::bytes_of<T>(*o.global_value);
  return v;
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

// The domain of every entity of a dimension, built (and fingerprinted) once
// for all the tags that cover it.
class FullDomains {
 public:
  explicit FullDomains(const HostMesh& mesh) : mesh_(mesh) {}
  const Domain<Kokkos::HostSpace>& get(int d) {
    auto& slot = all_[static_cast<std::size_t>(d)];
    if (!slot) slot = Domain<Kokkos::HostSpace>::all(mesh_, d);
    return *slot;
  }

 private:
  const HostMesh& mesh_;
  std::array<std::optional<Domain<Kokkos::HostSpace>>, num_field_slots> all_;
};

// The default or global of a variable-length tag: one value of an array type
// of any length, or a single value of the tag's type.
template <class T>
std::optional<std::vector<T>> read_var_attribute(hid_t tag, const char* name, const TagKind& k,
                                                 const std::string& where) {
  if (!h5::has_attribute(tag, name)) return std::nullopt;
  const std::string what = std::string("the ") + name + " of " + where;
  const Id a = h5::check(H5Aopen(tag, name, H5P_DEFAULT), "open " + what);
  h5::require_one_value(a.get(), what);
  const Id t = h5::check(H5Aget_type(a.get()), "get the type of " + what);
  if (H5Tget_class(t.get()) != H5T_ARRAY) {
    std::vector<T> v(1);
    h5::check(H5Aread(a.get(), k.mem.get(), v.data()), "read " + what);
    return v;
  }
  if (H5Tget_array_ndims(t.get()) != 1) fail(what + " is an array of more than one dimension");
  hsize_t n = 0;
  H5Tget_array_dims2(t.get(), &n);
  const Id mem = h5::check(H5Tarray_create2(k.mem.get(), 1, &n), "make the memory type of " + what);
  std::vector<T> v(static_cast<std::size_t>(n));
  h5::check(H5Aread(a.get(), mem.get(), v.data()), "read " + what);
  return v;
}

// A dense tag dataset: one row per entity of the group whose file ids start at first.
struct Dense {
  std::string path;  // below /tstt
  int dim = 0;
  GO first = 1;
  GO count = 0;
};

// Entries of a tag per slot in the whole file, and whether every entity of
// the slot has one: a slot with entries gets a field (empty where none is
// held), over all its entities when every one has a value.
struct SlotTotals {
  std::array<std::uint64_t, num_field_slots> entries{};
  std::array<bool, num_field_slots> every{};
};

// A tag's values on the entities held here, lids[d] and values[d] (ncomp
// per lid) per slot, installed as fields; a tag with no entries at all is a
// mesh value.
template <class T>
void install_tag(HostMesh& mesh, const std::string& name, const TagKind& k,
                 std::array<std::vector<LO>, num_field_slots>& lids, std::array<std::vector<T>, num_field_slots>& values,
                 const FieldOptions<T>& options, bool handle, const TagFormat& format, FullDomains& full,
                 const SlotTotals& totals) {
  const std::string where = "/tstt/tags/" + name;
  const auto nc = static_cast<std::size_t>(k.ncomp);
  bool any = false;
  for (int d = 0; d < num_field_slots; ++d) {
    auto& l = lids[static_cast<std::size_t>(d)];
    if (totals.entries[static_cast<std::size_t>(d)] == 0) continue;
    any = true;
    // Values by lid. Input in lid order (Ordering::file) keeps its order;
    // otherwise each value is placed at its lid, which also finds repeats.
    std::vector<std::size_t> order;  // value index of each member; empty: in order
    std::vector<LO> placed;
    std::span<const LO> sorted(l);
    if (std::adjacent_find(l.begin(), l.end(), std::greater_equal<>()) != l.end()) {
      constexpr auto none = std::numeric_limits<std::size_t>::max();
      std::vector<std::size_t> at_lid(static_cast<std::size_t>(detail::slot_count(mesh, d)), none);
      LO repeat = std::numeric_limits<LO>::max();
      for (std::size_t i = 0; i < l.size(); ++i) {
        std::size_t& slot = at_lid[static_cast<std::size_t>(l[i])];
        if (slot != none) repeat = std::min(repeat, l[i]);
        slot = i;
      }
      if (repeat != std::numeric_limits<LO>::max()) {
        fail(where + " has two values on file id " + str(detail::slot_gid(mesh, d)(static_cast<std::size_t>(repeat))));
      }
      placed.reserve(l.size());
      order.reserve(l.size());
      for (std::size_t lid = 0; lid < at_lid.size(); ++lid) {
        if (at_lid[lid] == none) continue;
        placed.push_back(static_cast<LO>(lid));
        order.push_back(at_lid[lid]);
      }
      sorted = placed;
    }
    // A tag on every entity of d is a full domain.
    const bool every = totals.every[static_cast<std::size_t>(d)];
    if (every && sorted.size() != static_cast<std::size_t>(detail::slot_count(mesh, d))) {
      fail(where + " has values on every entity, but not on every one held here");
    }
    const auto domain = every ? full.get(d) : Domain<Kokkos::HostSpace>::from_list(mesh, d, sorted);
    const auto f = mesh.fields().add<T>(name, Space<Kokkos::HostSpace>(domain, Layout::per_entity()),
                                        k.ncomp, options);
    // A per_entity field has one row per member, in member order.
    const auto view = f.view();
    const auto& src = values[static_cast<std::size_t>(d)];
    if (order.empty() && nc == 1) {  // one column of a LayoutLeft view is contiguous
      std::copy(src.begin(), src.end(), view.data());
      continue;
    }
    for (std::size_t i = 0; i < sorted.size(); ++i) {
      const std::size_t from = (order.empty() ? i : order[i]) * nc;
      for (std::size_t c = 0; c < nc; ++c) view(i, c) = src[from + c];
    }
  }
  if (any) {
    mesh.fields().set_tag_format(mesh.fields().family<T>(name, k.ncomp, Layout::per_entity(), handle), format);
  } else {
    mesh.values().set(name, mesh_value(options, k.ncomp, format));
  }
}

// A run of a variable-length tag's values: those of one entity.
struct Run {
  LO lid;
  std::size_t begin;
  std::size_t end;
};

template <class T>
void install_var_tag(HostMesh& mesh, const std::string& name, std::array<std::vector<Run>, num_field_slots>& runs,
                     const std::vector<T>& values, const FieldOptions<T>& options, bool handle,
                     const TagFormat& format, FullDomains& full, const SlotTotals& totals) {
  const std::string where = "/tstt/tags/" + name;
  bool any = false;
  for (int d = 0; d < num_field_slots; ++d) {
    auto& rs = runs[static_cast<std::size_t>(d)];
    if (totals.entries[static_cast<std::size_t>(d)] == 0) continue;
    any = true;
    std::sort(rs.begin(), rs.end(), [](const Run& a, const Run& b) { return a.lid < b.lid; });
    const auto twice =
        std::adjacent_find(rs.begin(), rs.end(), [](const Run& a, const Run& b) { return a.lid == b.lid; });
    if (twice != rs.end()) {
      fail(where + " has two values on file id " +
           str(detail::slot_gid(mesh, d)(static_cast<std::size_t>(twice->lid))));
    }
    std::vector<LO> lids(rs.size());
    std::vector<Offset> offsets(rs.size() + 1, 0);
    std::vector<T> held;
    for (std::size_t i = 0; i < rs.size(); ++i) {
      lids[i] = rs[i].lid;
      held.insert(held.end(), values.begin() + static_cast<std::ptrdiff_t>(rs[i].begin),
                  values.begin() + static_cast<std::ptrdiff_t>(rs[i].end));
      offsets[i + 1] = to_offset(held.size(), where + "/values");
    }
    const bool every = totals.every[static_cast<std::size_t>(d)];
    if (every && lids.size() != static_cast<std::size_t>(detail::slot_count(mesh, d))) {
      fail(where + " has values on every entity, but not on every one held here");
    }
    const auto domain = every ? full.get(d) : Domain<Kokkos::HostSpace>::from_list(mesh, d, lids);
    mesh.fields().add_var<T>(name, Space<Kokkos::HostSpace>(domain, Layout::per_entity()), offsets, held, options);
  }
  if (any) {
    mesh.fields().set_tag_format(mesh.fields().family<T>(name, 0, Layout::per_entity(), handle), format);
  } else {
    mesh.values().set(name, mesh_value(options, 0, format));
  }
}

// A cached graph's group name: pair_<from>_<to> or bridge_<dim>_<via>, one
// digit each.
struct GraphName {
  bool pair = true;
  int a = 0, b = 0;
};
inline std::optional<GraphName> graph_named(std::string_view n) {
  GraphName g;
  if (n.starts_with("pair_")) {
    n.remove_prefix(5);
  } else if (n.starts_with("bridge_")) {
    g.pair = false;
    n.remove_prefix(7);
  } else {
    return std::nullopt;
  }
  if (n.size() != 3 || n[1] != '_' || n[0] < '0' || n[0] > '3' || n[2] < '0' || n[2] > '3') return std::nullopt;
  g.a = n[0] - '0';
  g.b = n[2] - '0';
  return g;
}


}  // namespace soma::h5m_read

#endif  // SOMA_SRC_IO_H5M_READ_HPP_
