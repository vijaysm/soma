// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "h5m_read.hpp"
#include "hdf5.hpp"
#include "soma/core/handle_ref.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/io/h5m.hpp"
#include "soma/topology/build.hpp"
#include "soma/topology/nodes.hpp"

namespace soma {

namespace {

namespace fs = std::filesystem;
using h5::Id;
using namespace h5m_read;

struct Nodes {
  GO start = 1;
  GO count = 0;
  std::vector<Real> xyz;
};

Nodes read_nodes(hid_t tstt) {
  const NodeInfo info = describe_nodes(tstt);
  const std::string where = "/tstt/nodes/coordinates";
  const Id ds = h5::open_dataset(tstt, "nodes/coordinates");
  Nodes n;
  n.start = info.start;
  n.count = info.count;
  const auto rows = static_cast<std::size_t>(info.count);
  const auto raw = h5::read_all<double>(ds.get(), rows * info.columns, where);
  n.xyz.assign(3 * rows, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t c = 0; c < info.columns; ++c) n.xyz[3 * i + c] = raw[info.columns * i + c];
  }
  return n;
}

Group read_group(hid_t elements, const std::string& name, const Nodes& nodes) {
  Group g = describe_group(elements, name);
  const Id grp = h5::open_group(elements, name);
  const std::string cw = g.where + "/connectivity";
  if (g.arity == 0) {
    const std::string pw = g.where + "/poly_indices";
    const Id pi = h5::open_dataset(grp.get(), "poly_indices");
    const auto last = h5::read_all<std::int64_t>(pi.get(), static_cast<std::size_t>(g.count), pw);
    g.offsets = polygon_offsets(g, last, 0, -1, true);
  } else if (g.topology == Topology::polygon) {
    g.offsets.assign(static_cast<std::size_t>(g.count) + 1, 0);
    for (std::size_t i = 0; i <= static_cast<std::size_t>(g.count); ++i) g.offsets[i] = to_offset(i * g.arity, cw);
  }
  if (g.count == 0) return g;
  const Id ds = h5::open_dataset(grp.get(), "connectivity");
  const auto ids = h5::read_all<std::uint64_t>(ds.get(), static_cast<std::size_t>(g.nnz), cw);
  g.conn = vertex_ids(g, ids, 0, g.offsets, nodes.start, nodes.count);
  return g;
}

// The sets of /tstt/sets/list, installed on mesh. A row's contents name
// entities and sets by file id, one by one or (flag 0x8) as (start, count)
// ranges; its children and parents name sets.
void read_sets(HostMesh& mesh, hid_t tstt, const IdRange& sets, std::uint64_t max_id, const Locator& at) {
  if (sets.count == 0) return;
  constexpr std::int64_t ranged = 0x8;
  constexpr std::int64_t known_flags = set_flag_bits | ranged;
  const auto n = static_cast<std::size_t>(sets.count);
  const auto list = h5::read_all<std::int64_t>(h5::open_dataset(tstt, "sets/list").get(), 4 * n, sets.where);
  const auto set_name = [&](std::size_t i) { return "set " + str(sets.first + static_cast<GO>(i)); };

  // Column c of the list ends each set's run of /tstt/sets/<column c>.
  static constexpr std::array<const char*, 3> column{"contents", "children", "parents"};
  std::array<std::vector<std::uint64_t>, 3> ids;
  for (std::size_t c = 0; c < column.size(); ++c) {
    std::int64_t prev = -1;
    for (std::size_t i = 0; i < n; ++i) {
      const std::int64_t end = list[4 * i + c];
      if (end < prev) {
        fail(sets.where + ": " + set_name(i) + " ends its " + column[c] + " at index " + str(end) +
             "; the ends must be at least -1 and never decrease");
      }
      prev = end;
    }
    const std::string path = std::string("sets/") + column[c];
    const std::string where = "/tstt/" + path;
    if (!h5::exists(tstt, path)) {
      if (prev != -1) fail(sets.where + " ends " + column[c] + " at index " + str(prev) + ", but there is no " + where);
      continue;
    }
    const Id ds = h5::open_dataset(tstt, path);
    const auto ext = h5::extent(ds.get(), where);
    if (ext.size() != 1) fail(where + " must be one-dimensional");
    if (static_cast<std::uint64_t>(prev) + 1 != ext[0]) {  // prev >= -1, so this is exact
      fail(where + " has " + str(std::uint64_t{ext[0]}) + " entries, but " + sets.where + " ends " + column[c] +
           " at index " + str(prev));
    }
    ids[c] = h5::read_all<std::uint64_t>(ds.get(), static_cast<std::size_t>(ext[0]), where);
  }

  const auto range_of = [&](std::uint64_t id) {
    return id <= static_cast<std::uint64_t>(std::numeric_limits<GO>::max()) ? at.range_of(static_cast<GO>(id))
                                                                           : nullptr;
  };
  const auto no_entity = [&](std::size_t i, std::uint64_t id) {
    fail("/tstt/sets/contents: " + set_name(i) + " lists file id " + str(id) + ", which names no entity");
  };
  const auto set_ids = [&](std::size_t i, std::size_t c, std::size_t from, std::size_t to) {
    std::vector<GO> out;
    out.reserve(to - from);
    for (std::size_t j = from; j < to; ++j) {
      const IdRange* r = range_of(ids[c][j]);
      if (r == nullptr || r->dim != set_dim) {
        fail("/tstt/sets/" + std::string(column[c]) + ": " + set_name(i) + " lists file id " + str(ids[c][j]) +
             ", which is not a set");
      }
      out.push_back(static_cast<GO>(ids[c][j]));
    }
    return out;
  };

  SetBuilder b;
  std::array<std::size_t, 3> from{};
  for (std::size_t i = 0; i < n; ++i) {
    const std::int64_t flags = list[4 * i + 3];
    if (flags < 0 || (flags & ~known_flags) != 0) {
      fail(sets.where + ": " + set_name(i) + " has flags " + hex(flags) + ", outside " + hex(known_flags));
    }
    std::array<std::size_t, 3> to{};
    for (std::size_t c = 0; c < 3; ++c) to[c] = static_cast<std::size_t>(list[4 * i + c] + 1);
    SetSpec spec;
    spec.gid = sets.first + static_cast<GO>(i);
    spec.flags = static_cast<std::uint8_t>(flags & set_flag_bits);
    std::vector<SetRun> spec_runs;
    // A piece of a range within one dataset: set gids are file ids, entities
    // become runs of lids.
    const auto add = [&](const IdRange& r, GO first, GO count) {
      if (r.dim == set_dim) {
        spec_runs.push_back({set_dim, first, count});
        return;
      }
      for (const Interval& l : at.lid_runs(r.dim, first, count)) {
        spec_runs.push_back({r.dim, GO{l.begin}, GO{l.end - l.begin}});
      }
    };
    const std::span<const std::uint64_t> contents(ids[0].data() + from[0], to[0] - from[0]);
    if ((flags & ranged) == 0) {
      spec.contents.reserve(contents.size());
      for (const std::uint64_t id : contents) {
        const IdRange* r = range_of(id);
        if (r == nullptr) no_entity(i, id);
        const auto g = static_cast<GO>(id);
        spec.contents.push_back({r->dim, r->dim == set_dim ? g : GO{at.lid(r->dim, g)}});
      }
    } else {
      if ((flags & set_ordered) != 0) {
        fail(sets.where + ": " + set_name(i) + " is both ordered and ranged, which MOAB does not write");
      }
      if (contents.size() % 2 != 0) {
        fail("/tstt/sets/contents: " + set_name(i) + " is ranged but has " + str(std::uint64_t{contents.size()}) +
             " entries, not (start, count) pairs");
      }
      std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;  // [first, last]
      for (std::size_t j = 0; j < contents.size(); j += 2) {
        const std::uint64_t start = contents[j];
        const std::uint64_t count = contents[j + 1];
        if (count == 0) fail("/tstt/sets/contents: " + set_name(i) + " has a range of no entities at " + str(start));
        if (start == 0 || count > max_id || start > max_id - count + 1) {
          fail("/tstt/sets/contents: " + set_name(i) + " has the range of " + str(count) + " file ids from " +
               str(start) + ", beyond the max_id " + str(max_id) + " of /tstt");
        }
        runs.emplace_back(start, start + count - 1);
      }
      // Merged runs are disjoint, so they hold at most max_id ids.
      std::sort(runs.begin(), runs.end());
      std::size_t m = 0;
      for (std::size_t j = 1; j < runs.size(); ++j) {
        if (runs[j].first - 1 <= runs[m].second) {
          runs[m].second = std::max(runs[m].second, runs[j].second);
        } else {
          runs[++m] = runs[j];
        }
      }
      if (!runs.empty()) runs.resize(m + 1);
      for (const auto& [first, last] : runs) {
        // A range may span datasets; each piece must lie in one.
        for (std::uint64_t id = first; id <= last;) {
          const IdRange* r = range_of(id);
          if (r == nullptr) no_entity(i, id);
          const auto end = std::min(last, static_cast<std::uint64_t>(r->first + r->count - 1));
          add(*r, static_cast<GO>(id), static_cast<GO>(end - id + 1));
          id = end + 1;
        }
      }
    }
    spec.children = set_ids(i, 1, from[1], to[1]);
    spec.parents = set_ids(i, 2, from[2], to[2]);
    from = to;
    b.add(std::move(spec), std::move(spec_runs));
  }
  std::move(b).install(mesh);
}

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
void import_tag(HostMesh& mesh, hid_t tstt, const std::string& name, const TagKind& k,
                const std::vector<Dense>& dense, const Locator& at, FullDomains& full,
                bool handle, const TagFormat& format) {
  const std::string where = "/tstt/tags/" + name;
  const auto nc = static_cast<std::size_t>(k.ncomp);
  std::array<std::vector<LO>, num_field_slots> lids;
  std::array<std::vector<T>, num_field_slots> values;

  for (const Dense& s : dense) {
    std::vector<T> v = read_rows<T>(tstt, s.path, k, s.count);
    const auto d = static_cast<std::size_t>(s.dim);
    std::vector<LO> l = at.lids(s.dim, s.first, s.count);
    if (lids[d].empty()) {
      lids[d] = std::move(l);
      values[d] = std::move(v);
    } else {
      lids[d].insert(lids[d].end(), l.begin(), l.end());
      values[d].insert(values[d].end(), v.begin(), v.end());
    }
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
      const auto d = static_cast<std::size_t>(r->dim);
      lids[d].push_back(at.lid(r->dim, static_cast<GO>(ids[i])));
      values[d].insert(values[d].end(), v.begin() + static_cast<std::ptrdiff_t>(i * nc),
                       v.begin() + static_cast<std::ptrdiff_t>((i + 1) * nc));
    }
  }

  FieldOptions<T> options;
  options.default_value = read_value_attribute<T>(tag.get(), "default", k, where);
  options.global_value = read_value_attribute<T>(tag.get(), "global", k, where);
  options.handle = handle;
  if (handle) {
    to_handle_options(options, at, where);
    for (int d = 0; d < num_field_slots; ++d) {
      const auto& l = lids[static_cast<std::size_t>(d)];
      const auto gid = detail::slot_gid(mesh, d);
      to_handles<T>(values[static_cast<std::size_t>(d)], at, where, [&](std::size_t i) {
        return "file id " + str(gid(static_cast<std::size_t>(l[i / nc])));
      });
    }
  }

  SlotTotals totals;
  for (int d = 0; d < num_field_slots; ++d) {
    const auto k_ = static_cast<std::size_t>(d);
    totals.entries[k_] = lids[k_].size();
    totals.every[k_] = lids[k_].size() == static_cast<std::size_t>(detail::slot_count(mesh, d));
  }
  install_tag<T>(mesh, name, k, lids, values, options, handle, format, full, totals);
}

// A variable-length tag: id_list names the entities and sets, var_indices
// holds the last index of each one's run of values.
template <class T>
void import_var_tag(HostMesh& mesh, hid_t tstt, const std::string& name, const TagKind& k,
                    const std::vector<Dense>& dense, const Locator& at, FullDomains& full, bool handle,
                    const TagFormat& format) {
  const std::string where = "/tstt/tags/" + name;
  if (!dense.empty()) {
    fail("/tstt/" + dense.front().path + " holds dense values of a variable-length tag, which MOAB does not write");
  }
  std::array<std::vector<Run>, num_field_slots> runs;
  std::vector<T> values;
  const Id tag = h5::open_group(tstt, "tags/" + name);
  if (h5::exists(tag.get(), "id_list")) {
    const std::string iw = where + "/id_list";
    const Id ids_ds = h5::open_dataset(tag.get(), "id_list");
    const auto ext = h5::extent(ids_ds.get(), iw);
    if (ext.size() != 1) fail(iw + " must be one-dimensional");
    const auto n = static_cast<std::size_t>(ext[0]);
    const auto ids = h5::read_all<std::uint64_t>(ids_ds.get(), n, iw);
    const std::string xw = where + "/var_indices";
    if (!h5::exists(tag.get(), "var_indices")) fail(where + " is variable-length but has no var_indices");
    const Id x_ds = h5::open_dataset(tag.get(), "var_indices");
    const auto xext = h5::extent(x_ds.get(), xw);
    if (xext.size() != 1 || xext[0] != ext[0]) fail(xw + " must hold one index per entry of " + iw);
    const auto last = h5::read_all<std::int64_t>(x_ds.get(), n, xw);
    std::int64_t prev = -1;
    for (std::size_t i = 0; i < n; ++i) {
      if (last[i] < prev) {
        fail(xw + ": file id " + str(ids[i]) + " ends its values at index " + str(last[i]) +
             "; the ends must be at least -1 and never decrease");
      }
      prev = last[i];
    }
    if (!h5::exists(tag.get(), "values")) fail(where + " has an id_list but no values");
    const std::uint64_t held_values = static_cast<std::uint64_t>(prev) + 1;  // prev >= -1
    if (held_values > static_cast<std::uint64_t>(std::numeric_limits<GO>::max())) {
      fail(xw + " ends at index " + str(prev) + ", beyond what an index can count");
    }
    values = read_rows<T>(tstt, "tags/" + name + "/values", k, static_cast<GO>(held_values));
    if (handle) {
      to_handles<T>(values, at, where, [&](std::size_t i) {
        const auto j = static_cast<std::size_t>(
            std::lower_bound(last.begin(), last.end(), static_cast<std::int64_t>(i)) - last.begin());
        return "file id " + str(ids[j]);
      });
    }
    std::size_t begin = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const IdRange* r = ids[i] <= static_cast<std::uint64_t>(std::numeric_limits<GO>::max())
                             ? at.range_of(static_cast<GO>(ids[i]))
                             : nullptr;
      if (r == nullptr) fail(iw + " lists file id " + str(ids[i]) + ", which names no entity");
      const auto end = static_cast<std::size_t>(last[i] + 1);
      runs[static_cast<std::size_t>(r->dim)].push_back({at.lid(r->dim, static_cast<GO>(ids[i])), begin, end});
      begin = end;
    }
  }

  FieldOptions<T> options;
  options.default_value = read_var_attribute<T>(tag.get(), "default", k, where);
  options.global_value = read_var_attribute<T>(tag.get(), "global", k, where);
  options.handle = handle;
  if (handle) to_handle_options(options, at, where);
  SlotTotals totals;
  for (int d = 0; d < num_field_slots; ++d) {
    const auto k_ = static_cast<std::size_t>(d);
    totals.entries[k_] = runs[k_].size();
    totals.every[k_] = runs[k_].size() == static_cast<std::size_t>(detail::slot_count(mesh, d));
  }
  install_var_tag<T>(mesh, name, runs, values, options, handle, format, full, totals);
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
  collect("sets", set_dim, sets.first, sets.count);

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

  FullDomains full(mesh);
  for (const std::string& name : names) {
    const std::string where = "/tstt/tags/" + name;
    const Id tag = h5::open_group(tstt, "tags/" + name);
    const bool variable = h5::has_attribute(tag.get(), "variable_length") &&
                          h5::read_attribute<std::int32_t>(tag.get(), "variable_length", where) != 0;
    const bool handle = h5::has_attribute(tag.get(), "is_handle") &&
                        h5::read_attribute<std::int32_t>(tag.get(), "is_handle", where) != 0;
    const int storage_class =
        h5::has_attribute(tag.get(), "class") ? h5::read_attribute<std::int32_t>(tag.get(), "class", where) : -1;
    if (!h5::exists(tag.get(), "type")) fail(where + " has no type");
    const Id type = h5::check(H5Topen2(tag.get(), "type", H5P_DEFAULT), "open the type of " + where);
    std::string why;
    auto kind = tag_kind(type.get(), why);
    if (!kind) {
      report.skipped.push_back({name, why});
      continue;
    }
    if (handle) kind = handle_kind(type.get(), *kind, where);
    if (storage_class < -1 || storage_class > 3) {
      fail(where + " has class " + std::to_string(storage_class) + ", not 0 (bit), 1 (sparse), 2 (dense) or 3 (mesh)");
    }
    const TagFormat format{storage_class, kind->kind, kind->nbits};
    const auto it = dense.find(name);
    const std::vector<Dense> none;
    const std::vector<Dense>& sources = it != dense.end() ? it->second : none;
    if (variable) {
      if (kind->ncomp != 1) {
        report.skipped.push_back({name, "variable-length tags of " + std::to_string(kind->ncomp) +
                                            "-component values are not read"});
        continue;
      }
      dispatch_data_type(kind->type, [&]<class T>() {
        import_var_tag<T>(mesh, tstt, name, *kind, sources, at, full, handle, format);
      });
      continue;
    }
    dispatch_data_type(kind->type, [&]<class T>() {
      import_tag<T>(mesh, tstt, name, *kind, sources, at, full, handle, format);
    });
  }
}

// One dataset of a cached graph: rank 1, read whole.
template <class T>
std::vector<T> read_column(hid_t group, const char* name, const std::string& where) {
  const std::string path = where + "/" + name;
  if (!h5::exists(group, name)) fail(path + " is missing");
  const Id ds = h5::open_dataset(group, name);
  const auto ext = h5::extent(ds.get(), path);
  if (ext.size() != 1) fail(path + " must have rank 1");
  return h5::read_all<T>(ds.get(), static_cast<std::size_t>(ext[0]), path);
}

// The adjacency graphs a file caches (/tstt/soma/adjacency), installed on
// the mesh as build_adjacencies would make them: rows and targets mapped
// from file ids to local ids, downward rows in their side order, upward and
// bridge rows sorted by (target, code).
void read_graphs(HostMesh& mesh, hid_t tstt, std::uint64_t max_id) {
  if (!h5::exists(tstt, "soma")) return;
  const Id soma = h5::open_group(tstt, "soma");
  if (!h5::has_attribute(soma.get(), "version") ||
      h5::read_attribute<std::int32_t>(soma.get(), "version", "/tstt/soma") != 1) {
    fail("/tstt/soma is not of version 1, the only one this reader knows");
  }
  if (!h5::exists(soma.get(), "adjacency")) return;
  const Id graphs = h5::open_group(soma.get(), "adjacency");
  // File id -> dimension and local id of each vertex and element (their gids
  // are their file ids), and each dimension's local ids in file id order.
  std::vector<std::int8_t> dim_of(static_cast<std::size_t>(max_id) + 1, -1);
  std::vector<LO> lid_of(dim_of.size(), invalid_lo);
  for (int d = 0; d <= mesh.dim(); ++d) {
    const auto gid = mesh.gid(d);
    for (std::size_t l = 0; l < gid.extent(0); ++l) {
      const auto id = static_cast<std::size_t>(gid(l));
      dim_of[id] = static_cast<std::int8_t>(d);
      lid_of[id] = static_cast<LO>(l);
    }
  }
  std::array<std::vector<LO>, 4> in_file_order;
  for (std::size_t id = 1; id < dim_of.size(); ++id) {
    if (dim_of[id] >= 0) in_file_order[static_cast<std::size_t>(dim_of[id])].push_back(lid_of[id]);
  }
  for (const std::string& name : h5::children(graphs.get(), "/tstt/soma/adjacency")) {
    const std::string where = "/tstt/soma/adjacency/" + name;
    const auto g = graph_named(name);
    if (!g) fail(where + " is not a graph this reader knows (pair_<from>_<to> or bridge_<dim>_<via>)");
    if (g->a == g->b || g->a > mesh.dim() || g->b > mesh.dim()) {
      fail(where + " names dimensions that are not two distinct ones of this mesh");
    }
    if (g->pair && g->b == 0) fail(where + " caches the connectivity, which is not cached");
    const int from = g->a;
    const int to = g->pair ? g->b : g->a;
    const Id grp = h5::open_group(graphs.get(), name);
    const auto offsets = read_column<std::int64_t>(grp.get(), "offsets", where);
    const auto targets = read_column<std::uint64_t>(grp.get(), "targets", where);
    std::vector<std::uint8_t> codes;
    if (g->pair) {
      codes = read_column<std::uint8_t>(grp.get(), "codes", where);
      if (codes.size() != targets.size()) fail(where + "/codes must hold one code per target");
    } else if (h5::exists(grp.get(), "codes")) {
      fail(where + "/codes is not part of a bridge");
    }
    const std::vector<LO>& sources = in_file_order[static_cast<std::size_t>(from)];
    if (offsets.size() != sources.size() + 1) {
      fail(where + "/offsets must hold " + str(static_cast<std::uint64_t>(sources.size() + 1)) + " entries");
    }
    if (offsets.front() != 0 || !std::is_sorted(offsets.begin(), offsets.end()) ||
        offsets.back() != static_cast<std::int64_t>(targets.size())) {
      fail(where + "/offsets must rise from 0 to the number of targets");
    }
    const Offset total = to_offset(targets.size(), where + "/targets");
    Adjacency<Kokkos::HostSpace> out;
    out.offsets = View1<Offset, Kokkos::HostSpace>("h5m_adjacency_offsets", sources.size() + 1);
    out.targets = View1<LO, Kokkos::HostSpace>("h5m_adjacency_targets", static_cast<std::size_t>(total));
    if (g->pair) out.codes = View1<std::uint8_t, Kokkos::HostSpace>("h5m_adjacency_codes", static_cast<std::size_t>(total));
    for (std::size_t i = 0; i < sources.size(); ++i) {
      out.offsets(static_cast<std::size_t>(sources[i]) + 1) = static_cast<Offset>(offsets[i + 1] - offsets[i]);
    }
    for (std::size_t l = 0; l < sources.size(); ++l) out.offsets(l + 1) += out.offsets(l);
    const bool sorted_rows = from < to || !g->pair;
    std::vector<std::pair<LO, std::uint8_t>> row;
    for (std::size_t i = 0; i < sources.size(); ++i) {
      row.clear();
      for (auto k = static_cast<std::size_t>(offsets[i]); k < static_cast<std::size_t>(offsets[i + 1]); ++k) {
        const std::uint64_t id = targets[k];
        if (id >= dim_of.size() || dim_of[static_cast<std::size_t>(id)] != to) {
          fail(where + "/targets names " + str(id) + ", which is not an entity of dimension " + std::to_string(to));
        }
        row.emplace_back(lid_of[static_cast<std::size_t>(id)], g->pair ? codes[k] : std::uint8_t{0});
      }
      if (sorted_rows) std::sort(row.begin(), row.end());
      auto k = static_cast<std::size_t>(out.offsets(static_cast<std::size_t>(sources[i])));
      for (const auto& [lid, code] : row) {
        out.targets(k) = lid;
        if (g->pair) out.codes(k) = code;
        ++k;
      }
    }
    soma::detail::TopologyAccess::store(mesh).put(g->pair ? pair_slot(from, to) : bridge_slot(from, g->b), std::move(out));
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

  IdRange sets{1, 0, set_dim, "/tstt/sets/list"};
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
      b.add_elements(g.topology, gids, g.conn, g.nodes);
    }
    std::vector<GO>().swap(g.conn);
  }
  HostMesh mesh = std::move(b).build(options.ordering);
  const Locator at(mesh, ranges);
  read_sets(mesh, tstt.get(), sets, max_id, at);
  if (options.adjacencies) read_graphs(mesh, tstt.get(), max_id);
  if (options.tags) {
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
