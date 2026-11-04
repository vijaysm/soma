// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "hdf5.hpp"
#include "soma/core/handle_ref.hpp"
#include "soma/core/sets.hpp"
#include "soma/core/topology.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/tag_format.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/id_runs.hpp"
#include "soma/topology/sides.hpp"
#include "temporary.hpp"

namespace soma {

namespace {

namespace fs = std::filesystem;
using h5::Id;

// MOAB's element type names, by Topology value (mhdf_public.h).
constexpr std::array<const char*, topology_count> type_names{
    nullptr, "Edge", "Tri", "Quad", "Polygon", "Tet", "Pyramid", "Prism", "Knife", "Hex", "Polyhedron"};

// The entities of one element group: one topology and node count, rows in
// gid order, file ids from start on.
struct Group {
  int dim = 0;
  Topology type = Topology::vertex;
  int arity = 0;  // nodes an element: corners, then those of high-order elements
  std::vector<LO> rows;
  std::uint64_t start = 0;
  std::string name() const { return type_names[static_cast<std::size_t>(type)] + std::to_string(arity); }
};

// The set flag of contents stored as (start, count) pairs.
constexpr std::uint8_t ranged_contents = 0x8;

// /tstt/sets: per set the inclusive ends of its contents, children and
// parents, then its flags; and those three columns.
struct SetTable {
  std::vector<std::int64_t> list;
  std::vector<std::uint64_t> contents, children, parents;
};

// A run of file ids that holds dense tag values under `path`/tags: the
// vertices, an element group, or the sets.
struct Block {
  std::string path;
  std::uint64_t start = 0;
  std::uint64_t count = 0;
};

// One MOAB tag: its values by file id, in id order, and the blocks they
// cover whole (dense; the other values, and all variable-length ones, are
// sparse). ncomp 0: variable length.
struct TagPlan {
  std::string name;
  DataType type = DataType::u8;
  int ncomp = 1;
  bool handle = false;  // values, default and global are file ids
  TagFormat format{};
  std::optional<std::vector<std::byte>> def, global;
  std::size_t width = 0;  // bytes per component
  std::vector<std::uint64_t> ids;
  std::vector<std::byte> values;
  std::vector<std::int64_t> ends;  // variable length: the last index of each id's values
  std::vector<bool> dense;  // by block
};

// What is written and under which file ids, settled before the file is
// touched.
// A cached adjacency graph (options.adjacencies): rows in file id order of
// its source entities, targets as file ids.
struct GraphPlan {
  std::string name;  // pair_<from>_<to> or bridge_<dim>_<via>
  bool pair = true;
  std::vector<std::int64_t> offsets;
  std::vector<std::uint64_t> targets;
  std::vector<std::uint8_t> codes;  // pairs only
};

struct Plan {
  std::array<bool, 4> dims{};
  std::vector<LO> nodes;  // vertex rows in gid order
  std::vector<Group> groups;
  // By slot (dimension, or set_dim) and local id (set index); 0 = not written.
  std::array<std::vector<std::uint64_t>, num_field_slots> fid;
  std::uint64_t set_start = 0;
  SetTable sets;
  std::uint64_t max_id = 0;
  std::vector<Block> blocks;  // in file id order
  std::vector<TagPlan> tags;  // families, then mesh values, each in name order
  std::optional<HandleRef<Kokkos::HostSpace>> handles;  // when the mesh has handle values
  int compression = 0;  // deflate level; 0: datasets stored contiguous
  std::vector<GraphPlan> graphs;
};

// The number of temporary file names write_h5m tries before it gives up.
constexpr int temporary_attempts = 100;

// The bytes of a compressed dataset's chunks, short of its whole size.
constexpr std::size_t chunk_bytes = std::size_t{1} << 20;

[[noreturn]] void reject(const std::string& why) { throw std::invalid_argument("write_h5m: " + why); }

// Local ids of the listed rows in gid order (local id order among equal gids).
// Rows usually come in gid order already, so that is checked first.
void sort_by_gid(std::vector<LO>& rows, const View1<const GO, Kokkos::HostSpace>& gid) {
  const auto by_gid = [&](LO a, LO b) { return gid(static_cast<std::size_t>(a)) < gid(static_cast<std::size_t>(b)); };
  if (!std::is_sorted(rows.begin(), rows.end(), by_gid)) std::stable_sort(rows.begin(), rows.end(), by_gid);
}

template <class T>
std::optional<std::vector<T>> present(const std::vector<T>& v) {
  return v.empty() ? std::nullopt : std::optional(v);
}

// Calls f with every handle the mesh holds: in handle fields (values,
// defaults and globals) and in handle mesh values.
template <class F>
void each_handle(const HostMesh& mesh, F&& f) {
  const auto all = [&](const auto& v) {
    if (v) {
      for (const GO h : *v) f(h);
    }
  };
  for (const FieldInfo& info : mesh.fields().list()) {
    if (!info.handle) continue;
    if (info.variable) {
      const auto v = mesh.fields().get_var<GO>(info.id);
      const auto values = v.values();
      for (std::size_t i = 0; i < values.extent(0); ++i) f(values(i));
      all(v.default_value());
      all(v.global_value());
    } else {
      const auto v = mesh.fields().get<GO>(info.id);
      const auto values = v.view();
      for (std::size_t r = 0; r < values.extent(0); ++r) {
        for (std::size_t c = 0; c < values.extent(1); ++c) f(values(r, c));
      }
      all(present(v.default_value()));
      all(present(v.global_value()));
    }
  }
  for (const std::string& name : mesh.values().names()) {
    const MeshValue& v = mesh.values().at(name);
    if (!v.handle) continue;
    all(v.default_as<GO>());
    all(v.global_as<GO>());
  }
}

// The resolver of the mesh's handles, when it holds any.
std::optional<HandleRef<Kokkos::HostSpace>> handle_ref(const HostMesh& mesh) {
  bool any = false;
  for (const FieldInfo& info : mesh.fields().list()) any = any || info.handle;
  for (const std::string& name : mesh.values().names()) any = any || mesh.values().at(name).handle;
  if (!any) return std::nullopt;
  return HandleRef<Kokkos::HostSpace>::build(mesh);
}

// The gid of the first entity of dimension k that is not a side of a
// written entity of a higher dimension, which a reader could not rebuild;
// none when every one is.
std::optional<GO> not_rebuilt(const HostMesh& mesh, const std::array<bool, 4>& w, int k) {
  using Key = std::array<LO, 5>;  // vertex count, then the vertices sorted
  const auto key = [](const topology::Side& s) {
    Key x{static_cast<LO>(s.n), -1, -1, -1, -1};
    std::copy(s.v, s.v + s.n, x.begin() + 1);
    std::sort(x.begin() + 1, x.begin() + 1 + s.n);
    return x;
  };
  const auto m = mesh.view();
  std::vector<Key> sides;
  for (int d = k + 1; d <= mesh.dim(); ++d) {
    if (!w[static_cast<std::size_t>(d)]) continue;
    for (LO e = 0; e < mesh.count(d); ++e) {
      const int n = topology::side_count(m, d, e, k);
      for (int s = 0; s < n; ++s) sides.push_back(key(topology::side_of(m, d, e, k, s)));
    }
  }
  std::sort(sides.begin(), sides.end());
  for (LO e = 0; e < mesh.count(k); ++e) {
    const topology::Side s = topology::entity_as_side(m, k, e);
    if (s.n > 4 || !std::binary_search(sides.begin(), sides.end(), key(s))) {
      return mesh.gid(k)(static_cast<std::size_t>(e));
    }
  }
  return std::nullopt;
}

// The vertices, mesh.dim(), the dimensions the mesh was built with, and
// those that fields have values on, sets hold or handles name; then
// options.dims where set. A dimension left out must be one a reader can
// rebuild from the written ones.
std::array<bool, 4> written_dims(const HostMesh& mesh, const H5mWriteOptions& options,
                                 const std::optional<HandleRef<Kokkos::HostSpace>>& handles) {
  if (options.dims[0] == false) reject("the vertices are always written; dims[0] cannot be false");
  std::array<bool, 4> w{};
  for (int d = 0; d <= mesh.dim(); ++d) w[static_cast<std::size_t>(d)] = d == 0 || d == mesh.dim() || mesh.input(d);
  for (const FieldInfo& info : mesh.fields().list()) {
    for (const int d : info.dims) {
      if (d < set_dim) w[static_cast<std::size_t>(d)] = true;
    }
  }
  const auto& sets = mesh.sets();
  for (int d = 1; d < mesh.dim(); ++d) {
    for (LO s = 0; s < sets.count() && !w[static_cast<std::size_t>(d)] && sets.has_members(d); ++s) {
      w[static_cast<std::size_t>(d)] = !sets.members_of(s, d).empty();
    }
  }
  if (handles) {
    each_handle(mesh, [&](GO h) {
      const HandleTarget x = handles->resolve(h);
      if (x.valid() && x.slot < set_dim) w[static_cast<std::size_t>(x.slot)] = true;
    });
  }
  // The dimensions of cached graphs: both of a pair (but the connectivity),
  // that of a bridge.
  for (int a = 0; a <= mesh.dim(); ++a) {
    for (int b = 0; b <= mesh.dim(); ++b) {
      if (b != 0 && options.adjacencies.has(a, b)) w[static_cast<std::size_t>(a)] = w[static_cast<std::size_t>(b)] = true;
      if (options.adjacencies.has_bridge(a, b)) w[static_cast<std::size_t>(a)] = true;
    }
  }
  // Without a dimension turned off, every one left out was derived from
  // written ones.
  bool off = false;
  for (std::size_t d = 0; d < w.size(); ++d) {
    if (options.dims[d]) {
      w[d] = *options.dims[d];
      off = off || !w[d];
    }
  }
  for (int k = 1; off && k <= mesh.dim(); ++k) {
    if (w[static_cast<std::size_t>(k)] || mesh.count(k) == 0) continue;
    if (mesh.has_nodes(k)) {
      const auto no = mesh.node_offsets(k);
      for (LO e = 0; e < mesh.count(k); ++e) {
        if (no(static_cast<std::size_t>(e) + 1) == no(static_cast<std::size_t>(e))) continue;
        reject("dimension " + std::to_string(k) + " cannot be left out: its entity of gid " +
               std::to_string(mesh.gid(k)(static_cast<std::size_t>(e))) +
               " has nodes beyond its corners, which a reader could not rebuild");
      }
    }
    if (const auto g = not_rebuilt(mesh, w, k)) {
      reject("dimension " + std::to_string(k) + " cannot be left out: its entity of gid " + std::to_string(*g) +
             " is not a side of a written entity of a higher dimension, so a reader could not rebuild it");
    }
  }
  return w;
}

// Each set's contents in file ids: an ordered set's sequence as stored, an
// unordered set's members sorted, as pairs where MOAB would store them so;
// children and parents as set file ids.
void plan_sets(const HostMesh& mesh, Plan& p) {
  const auto& sets = mesh.sets();
  SetTable& t = p.sets;
  for (LO s = 0; s < sets.count(); ++s) {
    const std::string set = "set " + std::to_string(sets.gid()(static_cast<std::size_t>(s)));
    const auto fid = [&](int slot, LO lid) {
      const auto& f = p.fid[static_cast<std::size_t>(slot)];
      const std::uint64_t id = static_cast<std::size_t>(lid) < f.size() ? f[static_cast<std::size_t>(lid)] : 0;
      if (id == 0) reject(set + " holds an entity of dimension " + std::to_string(slot) + ", which is not written");
      return id;
    };
    std::vector<std::uint64_t> ids;
    if (sets.ordered(s)) {
      for (const SetEntry& e : sets.sequence(s)) ids.push_back(fid(e.slot, e.lid));
    } else {
      for (int slot = 0; slot < num_field_slots; ++slot) {
        if (slot != set_dim && slot > mesh.dim()) continue;
        if (!sets.has_members(slot)) {
          reject(set + ": the mesh does not hold the set members of dimension " + std::to_string(slot));
        }
        for (const LO l : sets.members_of(s, slot)) ids.push_back(fid(slot, l));
      }
      std::sort(ids.begin(), ids.end());
    }
    const io::SetContents c = io::set_contents(ids, sets.ordered(s));
    t.contents.insert(t.contents.end(), c.column.begin(), c.column.end());
    for (const LO x : sets.children(s)) t.children.push_back(fid(set_dim, x));
    for (const LO x : sets.parents(s)) t.parents.push_back(fid(set_dim, x));
    t.list.push_back(static_cast<std::int64_t>(t.contents.size()) - 1);
    t.list.push_back(static_cast<std::int64_t>(t.children.size()) - 1);
    t.list.push_back(static_cast<std::int64_t>(t.parents.size()) - 1);
    t.list.push_back((sets.flags(s) & set_flag_bits) | (c.ranged ? ranged_contents : 0));
  }
}

template <class T>
std::vector<std::byte> bytes_of(const std::vector<T>& v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  if (!v.empty()) std::memcpy(out.data(), v.data(), out.size());
  return out;
}

// The file id of the entity or set a handle names; 0 for the null handle.
GO handle_fid(const HandleRef<Kokkos::HostSpace>& ref, const Plan& p, GO h, const std::string& where) {
  if (h == 0) return 0;
  const HandleTarget x = ref.resolve(h);
  if (!x.valid()) reject(where + " holds the handle " + std::to_string(h) + ", which names nothing in the mesh");
  const auto& fid = p.fid[static_cast<std::size_t>(x.slot)];
  const std::uint64_t id = static_cast<std::size_t>(x.lid) < fid.size() ? fid[static_cast<std::size_t>(x.lid)] : 0;
  if (id == 0) reject(where + " names an entity of dimension " + std::to_string(x.slot) + ", which is not written");
  return static_cast<GO>(id);
}

// The one value a family's members agree on (their defaults, or their global
// values), or none when no member has one.
template <class T>
std::optional<std::vector<T>> agreed(const std::string& name, const char* what,
                                     const std::vector<std::optional<std::vector<T>>>& values) {
  std::optional<std::vector<T>> one;
  for (const auto& v : values) {
    if (!v) continue;
    if (one && *one != *v) reject("the fields named '" + name + "' have different " + what + " values");
    one = v;
  }
  return one;
}

// A family's values on every written entity that has one, sorted by file id:
// ncomp values each, or any number for a variable-length family.
template <class T>
void collect_family(const HostMesh& mesh, const Plan& p, const std::vector<FieldInfo>& members, TagPlan& t) {
  std::vector<std::optional<std::vector<T>>> defs, globals;
  struct Entry {
    std::uint64_t id = 0;
    std::size_t first = 0, count = 0;  // in values
    bool operator<(const Entry& o) const { return id < o.id; }
  };
  std::vector<Entry> order;
  std::vector<T> values;
  // Walks the members of each block of a field's space: their file id and
  // field row. GLOBAL_ID values on a dimension left out go with it: a reader
  // numbers the entities it rebuilds.
  const auto each = [&](const FieldInfo& info, const Space<Kokkos::HostSpace>& space, const auto& append) {
    for (int k = 0; k < space.num_blocks(); ++k) {
      const auto block = space.block(k);
      const int slot = block.domain.dim;
      const auto& fid = p.fid[static_cast<std::size_t>(slot)];
      const LO n = slot == set_dim ? mesh.sets().count() : mesh.count(slot);
      if (fid.size() != static_cast<std::size_t>(n)) {
        if (info.name == "GLOBAL_ID" && slot != set_dim && !p.dims[static_cast<std::size_t>(slot)]) continue;
        reject("field '" + info.name + "' has values on dimension " + std::to_string(slot) + ", which is not written");
      }
      for (LO m = 0; m < block.domain.size; ++m) {
        const std::size_t first = values.size();
        append(static_cast<std::size_t>(block.row_begin + block.first(m)));
        order.push_back(Entry{fid[static_cast<std::size_t>(block.domain.entity(m))], first, values.size() - first});
      }
    }
  };
  for (const FieldInfo& info : members) {
    if (info.variable) {
      const auto f = mesh.fields().template get_var<T>(info.id);
      defs.push_back(f.default_value());
      globals.push_back(f.global_value());
      const auto offsets = f.offsets();
      const auto data = f.values();
      each(info, f.space(), [&](std::size_t r) {
        values.insert(values.end(), data.data() + offsets(r), data.data() + offsets(r + 1));
      });
    } else {
      const auto f = mesh.fields().template get<T>(info.id);
      defs.push_back(present(f.default_value()));
      globals.push_back(present(f.global_value()));
      const auto view = f.view();
      each(info, f.space(), [&](std::size_t r) {
        for (int c = 0; c < t.ncomp; ++c) values.push_back(view(r, static_cast<std::size_t>(c)));  // LayoutLeft
      });
    }
  }
  auto def = agreed(t.name, "default", defs);
  auto global = agreed(t.name, "global", globals);
  t.width = sizeof(T);
  // Members usually come in file id order already; then the values are too.
  std::vector<T> sorted;
  if (std::is_sorted(order.begin(), order.end())) {
    sorted = std::move(values);
  } else {
    std::stable_sort(order.begin(), order.end());
    sorted.reserve(values.size());
    for (const Entry& e : order) {
      sorted.insert(sorted.end(), values.begin() + static_cast<std::ptrdiff_t>(e.first),
                    values.begin() + static_cast<std::ptrdiff_t>(e.first + e.count));
    }
  }
  t.ids.reserve(order.size());
  std::int64_t end = -1;
  for (const Entry& e : order) {
    t.ids.push_back(e.id);
    end += static_cast<std::int64_t>(e.count);
    if (t.ncomp == 0) t.ends.push_back(end);
  }
  if constexpr (std::is_same_v<T, GO>) {
    if (t.handle) {
      const std::string where = "field '" + t.name + "'";
      for (GO& h : sorted) h = handle_fid(*p.handles, p, h, where);
      for (auto* v : {&def, &global}) {
        if (*v) {
          for (GO& h : **v) h = handle_fid(*p.handles, p, h, where);
        }
      }
    }
  }
  if (def) t.def = bytes_of(*def);
  if (global) t.global = bytes_of(*global);
  t.values = bytes_of(sorted);
}

// The blocks a fixed-length tag's values cover whole, and the tag's storage
// class when the mesh did not record one: 0 for bit tags, 2 when every value
// is in a covered block, else 1.
void place(const std::vector<Block>& blocks, TagPlan& t) {
  t.dense.assign(blocks.size(), false);
  std::size_t covered = 0;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const Block& k = blocks[b];
    const auto lo = std::lower_bound(t.ids.begin(), t.ids.end(), k.start);
    const auto hi = std::lower_bound(lo, t.ids.end(), k.start + k.count);
    t.dense[b] = t.ncomp > 0 && k.count > 0 && static_cast<std::uint64_t>(hi - lo) == k.count;
    if (t.dense[b]) covered += k.count;
  }
  if (t.format.storage_class < 0) {
    t.format.storage_class = t.format.kind == TagFormat::Kind::bit ? 0
                             : !t.ids.empty() && covered == t.ids.size() ? 2
                                                                         : 1;
  }
}

// GLOBAL_ID as MOAB declares it (dense i32, default and global -1) holding
// the gid of every written vertex and element.
TagPlan gid_tag(const HostMesh& mesh, const Plan& p) {
  std::vector<std::int32_t> gids;
  const auto add = [&](int d, const std::vector<LO>& rows) {
    const auto g = mesh.gid(d);
    for (const LO l : rows) {
      const GO v = g(static_cast<std::size_t>(l));
      if (!std::in_range<std::int32_t>(v)) {
        reject("the gid " + std::to_string(v) + " of dimension " + std::to_string(d) +
               " does not fit GLOBAL_ID, a 32-bit integer in MOAB's format");
      }
      gids.push_back(static_cast<std::int32_t>(v));
    }
  };
  add(0, p.nodes);
  for (const Group& g : p.groups) add(g.dim, g.rows);
  TagPlan t;
  t.name = "GLOBAL_ID";
  t.type = DataType::i32;
  t.format = {2, TagFormat::Kind::plain, 0};
  t.def = bytes_of(std::vector<std::int32_t>{-1});
  t.global = t.def;
  t.width = sizeof(std::int32_t);
  t.ids.resize(gids.size());
  for (std::size_t i = 0; i < gids.size(); ++i) t.ids[i] = i + 1;
  t.values = bytes_of(gids);
  return t;
}

// One tag per field family, checked: one family per name, per_entity
// layouts, no pending writes, members that agree on their default and
// global values; and one per mesh value.
void plan_tags(const HostMesh& mesh, Plan& p) {
  std::map<std::string, std::vector<FieldInfo>> by_name;
  for (const FieldInfo& info : mesh.fields().list()) by_name[info.name].push_back(info);
  for (const auto& [name, members] : by_name) {
    const FieldInfo& first = members.front();
    for (const FieldInfo& info : members) {
      if (info.family != first.family) {
        reject("the fields named '" + name + "' differ in value type, component count or handle mark; one MOAB tag "
               "cannot hold them all");
      }
      if (info.layout.kind != Layout::Kind::per_entity) {
        reject("field '" + name + "' has the layout " + describe(info.layout) + "; only per_entity fields are written");
      }
      if (const std::size_t n = mesh.fields().pending_count(info.id); n != 0) {
        reject("field '" + name + "' has " + std::to_string(n) + " uncommitted writes; commit it before writing");
      }
    }
    TagPlan t;
    t.name = name;
    t.type = first.type;
    t.ncomp = first.ncomp;
    t.handle = first.handle;
    t.format = mesh.fields().tag_format(first.family);
    dispatch_data_type(t.type, [&]<class T>() { collect_family<T>(mesh, p, members, t); });
    place(p.blocks, t);
    p.tags.push_back(std::move(t));
  }
  // Mesh values: tags with no entity values; class 3 (mesh) when they hold a
  // global value and the mesh recorded no class.
  for (const std::string& name : mesh.values().names()) {
    if (by_name.contains(name)) reject("a field and a mesh value are both named '" + name + "'");
    if (name == "GLOBAL_ID") continue;
    const MeshValue& v = mesh.values().at(name);
    TagPlan t;
    t.name = name;
    t.type = v.type;
    t.ncomp = v.ncomp;
    t.handle = v.handle;
    t.format = v.format;
    t.def = v.default_value;
    t.global = v.global_value;
    if (v.handle) {
      const auto mapped = [&](const std::optional<std::vector<GO>>& h) -> std::optional<std::vector<std::byte>> {
        if (!h) return std::nullopt;
        std::vector<GO> ids = *h;
        for (GO& x : ids) x = handle_fid(*p.handles, p, x, "mesh value '" + name + "'");
        return bytes_of(ids);
      };
      t.def = mapped(v.default_as<GO>());
      t.global = mapped(v.global_as<GO>());
    }
    dispatch_data_type(t.type, [&]<class T>() { t.width = sizeof(T); });
    if (t.format.storage_class < 0 && t.global) t.format.storage_class = 3;
    place(p.blocks, t);
    p.tags.push_back(std::move(t));
  }
  // GLOBAL_ID: the mesh's own field; or else the gids, declared as the mesh
  // value of that name declares the tag (a file with the tag but no values
  // read so), or as MOAB does.
  if (!by_name.contains("GLOBAL_ID")) {
    TagPlan t = gid_tag(mesh, p);
    if (mesh.values().contains("GLOBAL_ID")) {
      const MeshValue& v = mesh.values().at("GLOBAL_ID");
      if (v.type != DataType::i32 || v.ncomp != 1 || v.handle) {
        reject("the mesh value 'GLOBAL_ID' cannot declare the gids: it is not one 32-bit integer");
      }
      t.def = v.default_value;
      t.global = v.global_value;
      t.format = v.format;
    }
    place(p.blocks, t);
    p.tags.push_back(std::move(t));
  }
}

// The graphs options.adjacencies names, but the connectivity (d, 0): each
// one the mesh holds, over written dimensions.
void plan_graphs(const HostMesh& mesh, const AdjacencyRequest& request, Plan& p) {
  for (int slot = 0; slot < adjacency_slots; ++slot) {
    if (!request.has_slot(slot)) continue;
    const bool pair = slot < 16;
    const int from = (pair ? slot : slot - 16) / 4;
    const int other = slot % 4;  // the target dimension of a pair, the via of a bridge
    if (pair && other == 0) continue;
    const int to = pair ? other : from;
    const std::string what = pair ? "(" + std::to_string(from) + ", " + std::to_string(to) + ") adjacency"
                                  : "bridge (" + std::to_string(from) + " via " + std::to_string(other) + ")";
    if (pair ? !mesh.has_adjacency(from, to) : !mesh.has_bridge(from, other)) {
      reject("the mesh holds no " + what + "; build it with build_adjacencies to write it");
    }
    for (const int d : {from, to}) {
      if (!p.dims[static_cast<std::size_t>(d)]) {
        reject("the " + what + " names dimension " + std::to_string(d) + ", which is not written");
      }
    }
    const AdjacencyGraph<Kokkos::HostSpace> g = pair ? mesh.adjacency(from, to) : mesh.bridge(from, other);
    // The source entities in file id order.
    std::vector<LO> rows;
    if (from == 0) {
      rows = p.nodes;
    } else {
      for (const Group& grp : p.groups) {
        if (grp.dim == from) rows.insert(rows.end(), grp.rows.begin(), grp.rows.end());
      }
    }
    const auto& target_fid = p.fid[static_cast<std::size_t>(to)];
    GraphPlan out;
    out.name = (pair ? "pair_" : "bridge_") + std::to_string(from) + "_" + std::to_string(other);
    out.pair = pair;
    out.offsets.reserve(rows.size() + 1);
    out.offsets.push_back(0);
    for (const LO r : rows) {
      const auto e = static_cast<std::size_t>(r);
      for (auto k = static_cast<std::size_t>(g.offsets(e)); k < static_cast<std::size_t>(g.offsets(e + 1)); ++k) {
        out.targets.push_back(target_fid[static_cast<std::size_t>(g.targets(k))]);
        if (pair) out.codes.push_back(g.codes(k));
      }
      out.offsets.push_back(static_cast<std::int64_t>(out.targets.size()));
    }
    p.graphs.push_back(std::move(out));
  }
}

Plan plan(const HostMesh& mesh, const H5mWriteOptions& options) {
  Plan p;
  if (options.compression < 0 || options.compression > 9) {
    reject("compression is a deflate level in [0, 9], not " + std::to_string(options.compression));
  }
  p.compression = options.compression;
  p.handles = handle_ref(mesh);
  p.dims = written_dims(mesh, options, p.handles);
  const auto nv = static_cast<std::size_t>(mesh.count(0));
  if (mesh.coords().extent(0) != nv || mesh.gid(0).extent(0) != nv) {
    reject("the mesh holds " + std::to_string(nv) + " vertices but not their coordinates and gids");
  }
  p.nodes.resize(nv);
  for (std::size_t l = 0; l < nv; ++l) p.nodes[l] = static_cast<LO>(l);
  sort_by_gid(p.nodes, mesh.gid(0));

  const auto view = mesh.view();
  for (int d = 1; d < 4; ++d) {
    if (!p.dims[static_cast<std::size_t>(d)]) continue;
    const auto n = static_cast<std::size_t>(mesh.count(d));
    const auto topo = mesh.topology(d);
    std::vector<Group> found;
    for (std::size_t e = 0; e < n; ++e) {
      const auto t = static_cast<Topology>(topo(e));
      if (t == Topology::knife || t == Topology::polyhedron || t == Topology::vertex) {
        reject("dimension " + std::to_string(d) + " holds a " + name(t) + ", which write_h5m does not write");
      }
      const auto arity = static_cast<int>(view.num_nodes(d, static_cast<LO>(e)));
      auto g = std::find_if(found.begin(), found.end(), [&](const Group& x) { return x.type == t && x.arity == arity; });
      if (g == found.end()) g = found.insert(found.end(), Group{d, t, arity, {}, 0});
      g->rows.push_back(static_cast<LO>(e));
    }
    for (Group& g : found) sort_by_gid(g.rows, mesh.gid(d));
    p.groups.insert(p.groups.end(), found.begin(), found.end());
  }
  std::stable_sort(p.groups.begin(), p.groups.end(), [](const Group& a, const Group& b) {
    return std::pair(static_cast<int>(a.type), a.arity) < std::pair(static_cast<int>(b.type), b.arity);
  });

  std::uint64_t next = 1;
  p.fid[0].assign(nv, 0);
  for (const LO l : p.nodes) p.fid[0][static_cast<std::size_t>(l)] = next++;
  for (int d = 1; d < 4; ++d) {
    if (p.dims[static_cast<std::size_t>(d)]) p.fid[static_cast<std::size_t>(d)].assign(static_cast<std::size_t>(mesh.count(d)), 0);
  }
  for (Group& g : p.groups) {
    g.start = next;
    auto& fid = p.fid[static_cast<std::size_t>(g.dim)];
    for (const LO l : g.rows) fid[static_cast<std::size_t>(l)] = next++;
  }
  // Sets come last, in set order (gid order).
  p.set_start = next;
  p.fid[set_dim].resize(static_cast<std::size_t>(mesh.sets().count()));
  for (auto& id : p.fid[set_dim]) id = next++;
  p.max_id = next - 1;
  plan_sets(mesh, p);

  p.blocks.push_back(Block{"nodes", 1, nv});
  for (const Group& g : p.groups) p.blocks.push_back(Block{"elements/" + g.name(), g.start, g.rows.size()});
  p.blocks.push_back(Block{"sets", p.set_start, p.fid[set_dim].size()});
  plan_tags(mesh, p);
  plan_graphs(mesh, options.adjacencies, p);
  return p;
}

// Writes the file at `name` with the default HDF5 property lists, as MOAB
// does, but no dataset modification times, so that equal meshes give equal
// bytes. (In the default file format groups and committed types carry none.)
// At a compression level above 0 datasets with values are chunked, shuffled
// and deflated.
class File {
 public:
  File(const std::string& name, int compression) : compression_(compression) {
    dcpl_ = h5::check(H5Pcreate(H5P_DATASET_CREATE), "create a dataset property list");
    h5::check(H5Pset_obj_track_times(dcpl_.get(), false), "turn off dataset times");
    file_ = h5::check(H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), "create the file");
  }

  hid_t get() const { return file_.get(); }

  Id group(hid_t loc, const std::string& path) const {
    return h5::check(H5Gcreate2(loc, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "create the group " + path);
  }

  void commit_type(hid_t loc, const std::string& path, hid_t type) const {
    h5::check(H5Tcommit2(loc, path.c_str(), type, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "commit the type " + path);
  }

  // A dataset of rank 1 or 2 written whole from `data` (memory type `mem`).
  Id dataset(hid_t loc, const std::string& path, hid_t type, std::vector<hsize_t> dims, hid_t mem,
             const void* data) const {
    const Id space = h5::check(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr),
                               "create the dataspace of " + path);
    hsize_t n = 1;
    for (const hsize_t x : dims) n *= x;
    Id chunked;
    if (compression_ > 0 && n > 0) {
      chunked = h5::check(H5Pcopy(dcpl_.get()), "copy the dataset property list");
      const hsize_t row = H5Tget_size(type) * n / dims[0];
      std::vector<hsize_t> chunk = dims;
      chunk[0] = std::clamp<hsize_t>(chunk_bytes / row, 1, dims[0]);
      h5::check(H5Pset_chunk(chunked.get(), static_cast<int>(chunk.size()), chunk.data()), "chunk " + path);
      h5::check(H5Pset_shuffle(chunked.get()), "shuffle " + path);
      h5::check(H5Pset_deflate(chunked.get(), static_cast<unsigned>(compression_)), "deflate " + path);
    }
    Id ds = h5::check(H5Dcreate2(loc, path.c_str(), type, space.get(), H5P_DEFAULT,
                                 chunked.get() >= 0 ? chunked.get() : dcpl_.get(), H5P_DEFAULT),
                      "create the dataset " + path);
    if (n > 0) h5::check(H5Dwrite(ds.get(), mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), "write " + path);
    return ds;
  }

  static void attribute(hid_t obj, const char* name, hid_t type, hid_t mem, const void* value) {
    const Id space = h5::check(H5Screate(H5S_SCALAR), std::string("create the dataspace of ") + name);
    const Id a = h5::check(H5Acreate2(obj, name, type, space.get(), H5P_DEFAULT, H5P_DEFAULT),
                           std::string("create the attribute ") + name);
    h5::check(H5Awrite(a.get(), mem, value), std::string("write the attribute ") + name);
  }

  void flush() const { h5::check(H5Fflush(file_.get(), H5F_SCOPE_GLOBAL), "flush the file"); }

 private:
  int compression_ = 0;
  Id dcpl_, file_;
};

// The HDF5 types of a tag's values: `file` as committed (an array of ncomp
// for plain tags of several components, a bitfield of nbits for bit tags,
// opaque of ncomp bytes for opaque ones; one component for variable-length
// tags) and `mem` as they are in memory.
struct TagTypes {
  Id file, mem;
};

TagTypes tag_types(const TagPlan& t) {
  TagTypes out;
  const std::string what = "the type of tag " + t.name;
  if (t.format.kind == TagFormat::Kind::bit) {
    out.file = h5::check(H5Tcopy(H5T_STD_B8LE), "copy " + what);
    h5::check(H5Tset_precision(out.file.get(), static_cast<std::size_t>(t.format.nbits)), "set the bits of " + what);
    out.mem = h5::check(H5Tcopy(out.file.get()), "copy " + what);
    return out;
  }
  if (t.format.kind == TagFormat::Kind::opaque) {
    out.file = h5::check(H5Tcreate(H5T_OPAQUE, static_cast<std::size_t>(std::max(t.ncomp, 1))), "create " + what);
    h5::check(H5Tset_tag(out.file.get(), ""), "tag " + what);
    out.mem = h5::check(H5Tcopy(out.file.get()), "copy " + what);
    return out;
  }
  hid_t file = -1, mem = -1;
  if (t.handle) {
    file = H5T_STD_U64LE;
    mem = H5T_NATIVE_UINT64;
  } else {
  dispatch_data_type(t.type, [&]<class T>() {
    mem = h5::native<T>();
    file = std::is_floating_point_v<T> ? (sizeof(T) == 4 ? H5T_IEEE_F32LE : H5T_IEEE_F64LE)
           : std::is_signed_v<T> ? (sizeof(T) == 1 ? H5T_STD_I8LE : sizeof(T) == 4 ? H5T_STD_I32LE : H5T_STD_I64LE)
                                 : (sizeof(T) == 1 ? H5T_STD_U8LE : sizeof(T) == 4 ? H5T_STD_U32LE : H5T_STD_U64LE);
  });
  }
  if (t.ncomp <= 1) {
    out.file = h5::check(H5Tcopy(file), "copy " + what);
    out.mem = h5::check(H5Tcopy(mem), "copy " + what);
  } else {
    const hsize_t n = static_cast<hsize_t>(t.ncomp);
    out.file = h5::check(H5Tarray_create2(file, 1, &n), "create " + what);
    out.mem = h5::check(H5Tarray_create2(mem, 1, &n), "create " + what);
  }
  return out;
}

// A variable-length tag: its default and global as arrays of their own
// lengths (none for an empty one, which HDF5 cannot type), and every value
// sparse, var_indices holding the last index of each entity's values.
void write_var_tag(const File& f, hid_t g, const TagTypes& types, const TagPlan& t) {
  const std::int32_t one = 1;
  File::attribute(g, "variable_length", H5T_STD_I32LE, H5T_NATIVE_INT32, &one);
  for (const auto& [attr, value] : {std::pair{"default", &t.def}, std::pair{"global", &t.global}}) {
    if (!*value || (*value)->empty()) continue;
    const hsize_t n = (*value)->size() / t.width;
    const Id file = h5::check(H5Tarray_create2(types.file.get(), 1, &n), std::string("type the ") + attr);
    const Id mem = h5::check(H5Tarray_create2(types.mem.get(), 1, &n), std::string("type the ") + attr);
    File::attribute(g, attr, file.get(), mem.get(), (*value)->data());
  }
  if (t.ids.empty()) return;
  f.dataset(g, "id_list", H5T_STD_U64LE, {t.ids.size()}, H5T_NATIVE_UINT64, t.ids.data());
  f.dataset(g, "var_indices", H5T_STD_I64LE, {t.ends.size()}, H5T_NATIVE_INT64, t.ends.data());
  f.dataset(g, "values", types.file.get(), {t.values.size() / t.width}, types.mem.get(), t.values.data());
}

// /tstt/tags/<name> with its committed type, class, default and global, and
// the sparse values; the dense ones go to <block>/tags/<name>.
void write_tag(const File& f, hid_t tstt, const std::vector<Block>& blocks, const TagPlan& t) {
  const Id tags = h5::open_group(tstt, "tags");
  const Id g = f.group(tags.get(), t.name);
  h5::check(H5Oset_comment(g.get(), t.name.c_str()), "comment the tag " + t.name);
  const TagTypes types = tag_types(t);
  f.commit_type(g.get(), "type", types.file.get());
  const std::int32_t cls = t.format.storage_class;
  File::attribute(g.get(), "class", H5T_STD_I32LE, H5T_NATIVE_INT32, &cls);
  if (t.handle) {
    const std::int32_t one = 1;
    File::attribute(g.get(), "is_handle", H5T_STD_I32LE, H5T_NATIVE_INT32, &one);
  }
  if (t.ncomp == 0) {
    write_var_tag(f, g.get(), types, t);
    return;
  }
  // MOAB types the default and global of a handle tag as plain file ids, not
  // with the committed type.
  const Id plain = h5::check(H5Tcopy(types.file.get()), "copy the type of tag " + t.name);
  const hid_t attr = t.handle ? plain.get() : types.file.get();
  if (t.def) File::attribute(g.get(), "default", attr, types.mem.get(), t.def->data());
  if (t.global) File::attribute(g.get(), "global", attr, types.mem.get(), t.global->data());

  std::vector<std::uint64_t> ids;
  std::vector<std::byte> values;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const Block& k = blocks[b];
    const auto lo = static_cast<std::size_t>(std::lower_bound(t.ids.begin(), t.ids.end(), k.start) - t.ids.begin());
    const auto hi = static_cast<std::size_t>(std::lower_bound(t.ids.begin(), t.ids.end(), k.start + k.count) - t.ids.begin());
    const std::size_t size = static_cast<std::size_t>(t.ncomp) * t.width;
    const std::byte* first = t.values.data() + lo * size;
    if (t.dense[b]) {
      const Id dir = h5::open_group(tstt, k.path + "/tags");
      f.dataset(dir.get(), t.name, types.file.get(), {k.count}, types.mem.get(), first);
    } else {
      ids.insert(ids.end(), t.ids.begin() + static_cast<std::ptrdiff_t>(lo), t.ids.begin() + static_cast<std::ptrdiff_t>(hi));
      values.insert(values.end(), first, t.values.data() + hi * size);
    }
  }
  if (!ids.empty()) {
    f.dataset(g.get(), "id_list", H5T_STD_U64LE, {ids.size()}, H5T_NATIVE_UINT64, ids.data());
    f.dataset(g.get(), "values", types.file.get(), {ids.size()}, types.mem.get(), values.data());
  }
}

void write_file(const std::string& name, const HostMesh& mesh, const Plan& p) {
  const File f(name, p.compression);
  const Id tstt = f.group(f.get(), "tstt");

  const Id types = h5::check(H5Tenum_create(H5T_STD_U8LE), "create the element type enumeration");
  for (int t = 1; t < topology_count; ++t) {
    const auto v = static_cast<std::uint8_t>(t);
    h5::check(H5Tenum_insert(types.get(), type_names[static_cast<std::size_t>(t)], &v), "name an element type");
  }
  f.commit_type(tstt.get(), "elemtypes", types.get());

  {
    const Id nodes = f.group(tstt.get(), "nodes");
    const auto xyz = mesh.coords();
    std::vector<double> buf(3 * p.nodes.size());
    for (std::size_t i = 0; i < p.nodes.size(); ++i) {
      for (std::size_t c = 0; c < 3; ++c) buf[3 * i + c] = xyz(static_cast<std::size_t>(p.nodes[i]), c);
    }
    const Id ds = f.dataset(nodes.get(), "coordinates", H5T_IEEE_F64LE, {p.nodes.size(), 3}, H5T_NATIVE_DOUBLE,
                            buf.data());
    const std::int64_t start = 1;
    File::attribute(ds.get(), "start_id", H5T_STD_I64LE, H5T_NATIVE_INT64, &start);
    f.group(nodes.get(), "tags");
  }

  const Id elements = f.group(tstt.get(), "elements");
  const auto& fid0 = p.fid[0];
  const auto view = mesh.view();
  for (const Group& g : p.groups) {
    const Id grp = f.group(elements.get(), g.name());
    const auto t = static_cast<std::uint8_t>(g.type);
    File::attribute(grp.get(), "element_type", types.get(), types.get(), &t);
    const auto k = static_cast<std::size_t>(g.arity);
    std::vector<std::uint64_t> buf(g.rows.size() * k);
    for (std::size_t i = 0; i < g.rows.size(); ++i) {
      for (std::size_t j = 0; j < k; ++j) {
        buf[k * i + j] = fid0[static_cast<std::size_t>(view.node(g.dim, g.rows[i], static_cast<LO>(j)))];
      }
    }
    const Id ds = f.dataset(grp.get(), "connectivity", H5T_STD_U64LE, {g.rows.size(), k}, H5T_NATIVE_UINT64,
                            buf.data());
    const auto start = static_cast<std::int64_t>(g.start);
    File::attribute(ds.get(), "start_id", H5T_STD_I64LE, H5T_NATIVE_INT64, &start);
    f.group(grp.get(), "tags");
  }

  {
    const Id sets = f.group(tstt.get(), "sets");
    const SetTable& t = p.sets;
    if (!t.list.empty()) {
      const Id ds = f.dataset(sets.get(), "list", H5T_STD_I64LE, {t.list.size() / 4, 4}, H5T_NATIVE_INT64,
                              t.list.data());
      const auto start = static_cast<std::int64_t>(p.set_start);
      File::attribute(ds.get(), "start_id", H5T_STD_I64LE, H5T_NATIVE_INT64, &start);
    }
    for (const auto& [column, ids] : {std::pair{"contents", &t.contents}, std::pair{"children", &t.children},
                                      std::pair{"parents", &t.parents}}) {
      if (!ids->empty()) f.dataset(sets.get(), column, H5T_STD_U64LE, {ids->size()}, H5T_NATIVE_UINT64, ids->data());
    }
    f.group(sets.get(), "tags");
  }
  f.group(tstt.get(), "tags");
  for (const TagPlan& t : p.tags) write_tag(f, tstt.get(), p.blocks, t);
  if (!p.graphs.empty()) {
    const Id soma = f.group(tstt.get(), "soma");
    const std::int32_t version = 1;
    File::attribute(soma.get(), "version", H5T_STD_I32LE, H5T_NATIVE_INT32, &version);
    const Id adjacency = f.group(soma.get(), "adjacency");
    for (const GraphPlan& g : p.graphs) {
      const Id grp = f.group(adjacency.get(), g.name);
      f.dataset(grp.get(), "offsets", H5T_STD_I64LE, {g.offsets.size()}, H5T_NATIVE_INT64, g.offsets.data());
      f.dataset(grp.get(), "targets", H5T_STD_U64LE, {g.targets.size()}, H5T_NATIVE_UINT64, g.targets.data());
      if (g.pair) f.dataset(grp.get(), "codes", H5T_STD_U8LE, {g.codes.size()}, H5T_NATIVE_UINT8, g.codes.data());
    }
  }
  File::attribute(tstt.get(), "max_id", H5T_STD_U64LE, H5T_NATIVE_UINT64, &p.max_id);
  f.flush();
}

}  // namespace

void write_h5m(const HostMesh& mesh, const fs::path& path, const H5mWriteOptions& options) {
  const Plan p = plan(mesh, options);
  if (p.compression > 0 && H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0) {
    throw std::runtime_error(io::cannot_write("write_h5m", path, "this HDF5 has no deflate filter"));
  }
  io::Temporary file("write_h5m", path, io::random_names(path), temporary_attempts);
  try {
    file.close();
    const h5::Session quiet;
    write_file(file.name(), mesh, p);
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const h5::Error& e) {
    throw std::runtime_error(io::cannot_write("write_h5m", path, e.what()));
  }
  file.commit();
}

}  // namespace soma
