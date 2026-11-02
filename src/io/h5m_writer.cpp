// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hdf5.hpp"
#include "soma/core/sets.hpp"
#include "soma/core/topology.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/id_runs.hpp"
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
  int arity = 0;
  std::vector<LO> rows;
  std::uint64_t start = 0;
  std::string name() const { return type_names[static_cast<int>(type)] + std::to_string(arity); }
};

// The set flag of contents stored as (start, count) pairs.
constexpr std::uint8_t ranged_contents = 0x8;

// /tstt/sets: per set the inclusive ends of its contents, children and
// parents, then its flags; and those three columns.
struct SetTable {
  std::vector<std::int64_t> list;
  std::vector<std::uint64_t> contents, children, parents;
};

// What is written and under which file ids, settled before the file is
// touched.
struct Plan {
  std::array<bool, 4> dims{};
  std::vector<LO> nodes;  // vertex rows in gid order
  std::vector<Group> groups;
  // By slot (dimension, or set_dim) and local id (set index); 0 = not written.
  std::array<std::vector<std::uint64_t>, num_field_slots> fid;
  std::uint64_t set_start = 0;
  SetTable sets;
  std::uint64_t max_id = 0;
};

// The number of temporary file names write_h5m tries before it gives up.
constexpr int temporary_attempts = 100;

[[noreturn]] void reject(const std::string& why) { throw std::invalid_argument("write_h5m: " + why); }

// Local ids of the listed rows in gid order (local id order among equal gids).
void sort_by_gid(std::vector<LO>& rows, const View1<const GO, Kokkos::HostSpace>& gid) {
  std::stable_sort(rows.begin(), rows.end(), [&](LO a, LO b) {
    return gid(static_cast<std::size_t>(a)) < gid(static_cast<std::size_t>(b));
  });
}

std::array<bool, 4> written_dims(const HostMesh& mesh, const H5mWriteOptions& options) {
  std::array<bool, 4> w{};
  if (options.dims) {
    if (!(*options.dims)[0]) reject("the vertices are always written; dims[0] must be true");
    w = *options.dims;
  } else {
    for (int d = 0; d <= mesh.dim(); ++d) w[static_cast<std::size_t>(d)] = d == 0 || d == mesh.dim() || mesh.input(d);
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

Plan plan(const HostMesh& mesh, const H5mWriteOptions& options) {
  Plan p;
  p.dims = written_dims(mesh, options);
  const auto nv = static_cast<std::size_t>(mesh.count(0));
  if (mesh.coords().extent(0) != nv || mesh.gid(0).extent(0) != nv) {
    reject("the mesh holds " + std::to_string(nv) + " vertices but not their coordinates and gids");
  }
  p.nodes.resize(nv);
  for (std::size_t l = 0; l < nv; ++l) p.nodes[l] = static_cast<LO>(l);
  sort_by_gid(p.nodes, mesh.gid(0));

  for (int d = 1; d < 4; ++d) {
    if (!p.dims[static_cast<std::size_t>(d)]) continue;
    const auto n = static_cast<std::size_t>(mesh.count(d));
    const auto off = mesh.conn_offsets(d);
    const auto topo = mesh.topology(d);
    std::vector<Group> found;
    for (std::size_t e = 0; e < n; ++e) {
      const auto t = static_cast<Topology>(topo(e));
      if (t == Topology::knife || t == Topology::polyhedron || t == Topology::vertex) {
        reject("dimension " + std::to_string(d) + " holds a " + name(t) + ", which write_h5m does not write");
      }
      const auto arity = static_cast<int>(off(e + 1) - off(e));
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
  return p;
}

// Writes the file at `name` with the default HDF5 property lists, as MOAB
// does, but no dataset modification times, so that equal meshes give equal
// bytes. (In the default file format groups and committed types carry none.)
class File {
 public:
  explicit File(const std::string& name) {
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
    Id ds = h5::check(H5Dcreate2(loc, path.c_str(), type, space.get(), H5P_DEFAULT, dcpl_.get(), H5P_DEFAULT),
                      "create the dataset " + path);
    hsize_t n = 1;
    for (const hsize_t x : dims) n *= x;
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
  Id dcpl_, file_;
};

void write_file(const std::string& name, const HostMesh& mesh, const Plan& p) {
  const File f(name);
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
  for (const Group& g : p.groups) {
    const Id grp = f.group(elements.get(), g.name());
    const auto t = static_cast<std::uint8_t>(g.type);
    File::attribute(grp.get(), "element_type", types.get(), types.get(), &t);
    const auto off = mesh.conn_offsets(g.dim);
    const auto conn = mesh.conn(g.dim);
    const auto k = static_cast<std::size_t>(g.arity);
    std::vector<std::uint64_t> buf(g.rows.size() * k);
    for (std::size_t i = 0; i < g.rows.size(); ++i) {
      const auto first = static_cast<std::size_t>(off(static_cast<std::size_t>(g.rows[i])));
      for (std::size_t j = 0; j < k; ++j) buf[k * i + j] = fid0[static_cast<std::size_t>(conn(first + j))];
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
  File::attribute(tstt.get(), "max_id", H5T_STD_U64LE, H5T_NATIVE_UINT64, &p.max_id);
  f.flush();
}

}  // namespace

void write_h5m(const HostMesh& mesh, const fs::path& path, const H5mWriteOptions& options) {
  const Plan p = plan(mesh, options);
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
