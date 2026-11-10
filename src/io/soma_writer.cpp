// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The SOMA format's writer: each rank lays out its owned rows
// of every dataset, one exscan places them, and the ranks write them
// collectively over MPI-IO, through M aggregators that alone open the file,
// or through rank 0 with a serial HDF5.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../parallel/closure.hpp"
#include "hdf5.hpp"
#include "soma/core/handle_ref.hpp"
#include "soma/core/mesh_values.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma_image.hpp"
#include "soma_layout.hpp"
#include "temporary.hpp"

namespace soma {

namespace {

namespace fs = std::filesystem;
using h5::Id;
using soma_file::dim_group;
using soma_file::field_group;
using soma_file::value_group;

constexpr const char* who = "write_soma";

// The number of temporary file names rank 0 tries before it gives up.
constexpr int temporary_attempts = 100;

// The bytes of a compressed dataset's chunks, short of its whole size.
constexpr std::size_t chunk_bytes = std::size_t{1} << 20;

// Message tags of the streamed sink (route uses 32000 and above).
constexpr int tag_size = 31000;
constexpr int tag_rows = 31001;
constexpr int tag_group_size = 31002;
constexpr int tag_group_rows = 31003;

template <class T>
void append(std::vector<std::byte>& out, const T& v) {
  const std::size_t at = out.size();
  out.resize(at + sizeof(T));
  std::memcpy(out.data() + at, &v, sizeof(T));
}

template <class T>
void append(std::vector<std::byte>& out, std::span<const T> v) {
  const std::size_t at = out.size();
  out.resize(at + v.size_bytes());
  if (!v.empty()) std::memcpy(out.data() + at, v.data(), v.size_bytes());
}

using soma_file::Attribute;
using soma_file::Dataset;
using soma_file::Image;

bool streamed = false;  // detail::soma_streamed_sink

bool internal(const std::string& name) { return name == sharing_field || name == owner_field; }

bool has_sharing(const HostMesh& mesh, int d) {
  for (const FieldInfo& f : mesh.fields().list()) {
    if (f.name == sharing_field && std::find(f.dims.begin(), f.dims.end(), d) != f.dims.end()) return true;
  }
  return false;
}

std::vector<std::int64_t> layout_words(const Layout& l) {
  return {static_cast<std::int64_t>(l.kind),        static_cast<std::int64_t>(l.family), l.order,
          static_cast<std::int64_t>(l.ordering),    static_cast<std::int64_t>(l.serendipity), l.quad_rule};
}

std::vector<std::int64_t> format_words(const TagFormat& t) {
  return {t.storage_class, static_cast<std::int64_t>(t.kind), t.nbits};
}

// What every rank must agree on before laying anything out, in one
// reduction: failures, the hashes of sets and fields, the dimensions
// written and those with high-order nodes.
struct Agreed {
  int dim = 0;
  std::array<bool, 4> written{};
  std::array<bool, 4> input{};
  std::array<bool, 4> nodes{};
  std::uint32_t graphs = 0;
};

Agreed agree(const Comm& comm, const HostMesh& mesh, const std::string& local_error, bool derived,
             std::uint64_t access) {
  const std::uint64_t hs = detail::sets_hash(mesh), hf = detail::fields_hash(mesh);
  // Maxima: flags, then each hash and its complement (equal on every rank
  // only when the maxima are complements), the dimension, then per
  // dimension input, set members, high-order nodes, the graphs held, and
  // per dimension whether any entity is held, then the hash of the access
  // options and its complement.
  std::array<std::uint64_t, 26> f{};
  f[0] = local_error.empty() ? 0 : 1;
  f[1] = mesh.edited() && comm.size() > 1 ? 1 : 0;
  f[2] = hs;
  f[3] = ~hs;
  f[4] = hf;
  f[5] = ~hf;
  f[6] = static_cast<std::uint64_t>(mesh.dim());
  for (int d = 0; d <= mesh.dim(); ++d) {
    const auto k = static_cast<std::size_t>(d);
    f[7 + k] = mesh.count(d) > 0 && mesh.input(d) ? 1 : 0;
    // Members of some set (storage alone, as derivation leaves it, is not).
    const auto& sets = mesh.sets();
    for (LO s = 0; s < sets.count() && f[11 + k] == 0 && sets.has_members(d); ++s) {
      f[11 + k] = sets.members_of(s, d).empty() ? 0 : 1;
    }
    f[15 + k] = mesh.count(d) > 0 && d > 0 && mesh.has_nodes(d) ? 1 : 0;
    f[20 + k] = mesh.count(d) > 0 ? 1 : 0;
  }
  f[19] = mesh.adjacencies().bits();
  f[24] = access;
  f[25] = ~access;
  comm.allreduce(std::span<std::uint64_t>(f), Op::max);
  if (f[0] != 0) {
    if (!local_error.empty()) throw std::invalid_argument(std::string(who) + ": " + local_error);
    throw std::invalid_argument(std::string(who) + ": another rank cannot write its part");
  }
  if (f[1] != 0) throw std::logic_error(std::string(who) + ": a rank's mesh holds edits; compress() it first");
  if (f[2] != ~f[3]) {
    throw std::invalid_argument(std::string(who) +
                                ": the ranks' sets differ (gids, flags, children, parents or sets they contain)");
  }
  if (f[4] != ~f[5]) {
    throw std::invalid_argument(std::string(who) +
                                ": the ranks' fields or mesh values differ (beyond their values on entities)");
  }
  if (f[24] != ~f[25]) {
    throw std::invalid_argument(std::string(who) + ": the ranks' aggregators or MPI-IO hints differ");
  }
  Agreed a;
  a.dim = static_cast<int>(f[6]);
  for (std::size_t d = 0; d < 4; ++d) {
    a.input[d] = f[7 + d] != 0;
    a.written[d] = d == 0 || static_cast<int>(d) == a.dim || f[7 + d] != 0 || f[11 + d] != 0;
    a.written[d] = a.written[d] || (derived && f[20 + d] != 0);
    a.nodes[d] = f[15 + d] != 0;
  }
  for (const FieldInfo& fi : mesh.fields().list()) {
    if (internal(fi.name)) continue;
    for (const int d : fi.dims) {
      if (d != set_dim) a.written[static_cast<std::size_t>(d)] = true;
    }
  }
  for (std::size_t d = 0; d < 4; ++d) a.written[d] = a.written[d] && static_cast<int>(d) <= a.dim;
  // The graphs a reader builds again, as migrate does: those held on any
  // rank, and the connectivity of each dimension held but neither input nor
  // built by them.
  const auto held = static_cast<std::uint32_t>(f[19]);
  std::array<bool, 4> built{};
  for (int s = 0; s < adjacency_slots; ++s) {
    if (((held >> s) & 1u) == 0) continue;
    const int x = s < 16 ? s / 4 : (s - 16) / 4, y = s < 16 ? s % 4 : (s - 16) % 4;
    built[static_cast<std::size_t>(x)] = built[static_cast<std::size_t>(y)] = true;
  }
  AdjacencyRequest r = AdjacencyRequest::from_bits(held);
  for (int d = 1; d < a.dim; ++d) {
    const auto k = static_cast<std::size_t>(d);
    if (f[20 + k] != 0 && !a.input[k] && !built[k] && !a.written[k]) r.add(d, 0);
  }
  a.graphs = r.bits();
  return a;
}

// The owned rows of each written dimension, and for each owned shared
// entity the ranks holding copies.
void lay_out_entities(const HostMesh& mesh, const Agreed& a, Image& img) {
  const auto view = mesh.view();
  for (int d = 0; d <= a.dim; ++d) {
    if (!a.written[static_cast<std::size_t>(d)]) continue;
    const std::string g = dim_group(d);
    img.groups.push_back(g);
    img.groups.push_back(g + "/shared");
    const bool here = d <= mesh.dim() && mesh.count(d) > 0;
    const LO owned = here ? mesh.segment_table(d).owned_end : 0;
    auto& gid = img.column(g + "/gid", DataType::i64);
    const auto gids = here ? mesh.gid(d) : decltype(mesh.gid(d)){};
    for (LO e = 0; e < owned; ++e) append(gid.data, static_cast<std::int64_t>(gids(e)));
    if (d == 0) {
      auto& xyz = img.column(g + "/coords", DataType::f64, 3);
      for (LO e = 0; e < owned; ++e) {
        for (int c = 0; c < 3; ++c) append(xyz.data, static_cast<double>(view.x(e, c)));
      }
    } else {
      auto& topo = img.column(g + "/topology", DataType::u8);
      auto& counts = img.column(g + "/counts", DataType::u32);
      auto& conn = img.column(g + "/conn", DataType::i64);
      for (LO e = 0; e < owned; ++e) {
        append(topo.data, static_cast<std::uint8_t>(view.topology(d, e)));
        const LO k = view.num_vertices(d, e);
        append(counts.data, static_cast<std::uint32_t>(k));
        for (LO j = 0; j < k; ++j) append(conn.data, static_cast<std::int64_t>(view.gid(0, view.vertex(d, e, j))));
      }
      if (a.nodes[static_cast<std::size_t>(d)]) {
        auto& ncounts = img.column(g + "/node_counts", DataType::u32);
        auto& nodes = img.column(g + "/nodes", DataType::i64);
        for (LO e = 0; e < owned; ++e) {
          const LO k = view.num_vertices(d, e), n = view.num_nodes(d, e);
          append(ncounts.data, static_cast<std::uint32_t>(n - k));
          for (LO j = k; j < n; ++j) append(nodes.data, static_cast<std::int64_t>(view.gid(0, view.node(d, e, j))));
        }
      }
    }
    auto& index = img.column(g + "/shared/index", DataType::i64);
    auto& scounts = img.column(g + "/shared/counts", DataType::u32);
    auto& parts = img.column(g + "/shared/parts", DataType::i32);
    if (here && has_sharing(mesh, d)) {
      const SegmentTable& t = mesh.segment_table(d);
      const auto s = mesh.fields().get_var<int>(sharing_field, d);
      for (LO e = t.interior_end; e < t.owned_end; ++e) {
        const auto row = s.value_or_default(d, e);
        append(index.data, static_cast<std::int64_t>(e));
        append(scounts.data, static_cast<std::uint32_t>(row.size()));
        append(parts.data, std::span<const int>(row.data(), row.size()));
      }
    }
  }
}

// One field: its options as attributes, values on sets once, and per
// dimension each part's values on its owned entities, with the members when
// the part's domain is not full there.
template <class T>
void lay_out_field(const HostMesh& mesh, const FieldInfo& info, std::size_t index, const std::array<bool, 4>& full,
                   Image& img) {
  const auto& reg = mesh.fields();
  const std::string g = field_group(index);
  img.groups.push_back(g);
  img.text(g, "name", info.name);
  std::vector<std::byte> def, glob;
  bool has_def = false, has_glob = false;
  const Domain<Kokkos::HostSpace>* dom = nullptr;
  std::optional<Field<T, Kokkos::HostSpace>> fixed;
  std::optional<VarField<T, Kokkos::HostSpace>> var;
  if (info.variable) {
    var = reg.template get_var<T>(info.id);
    dom = &var->domain();
    if (var->default_value()) {
      has_def = true;
      append(def, std::span<const T>(*var->default_value()));
    }
    if (var->global_value()) {
      has_glob = true;
      append(glob, std::span<const T>(*var->global_value()));
    }
  } else {
    fixed = reg.template get<T>(info.id);
    dom = &fixed->domain();
    if (fixed->has_default()) {
      has_def = true;
      append(def, std::span<const T>(fixed->default_value()));
    }
    if (fixed->has_global()) {
      has_glob = true;
      append(glob, std::span<const T>(fixed->global_value()));
    }
  }
  std::vector<std::int64_t> meta{static_cast<std::int64_t>(info.type), info.ncomp, info.variable ? 1 : 0,
                                 info.handle ? 1 : 0};
  const auto lw = layout_words(info.layout);
  meta.insert(meta.end(), lw.begin(), lw.end());
  const auto fw = format_words(reg.tag_format(info.family));
  meta.insert(meta.end(), fw.begin(), fw.end());
  meta.push_back(has_def ? 1 : 0);
  meta.push_back(has_glob ? 1 : 0);
  std::int64_t full_bits = 0;
  for (std::size_t d = 0; d < 4; ++d) full_bits |= full[d] ? std::int64_t{1} << d : 0;
  meta.push_back(full_bits);
  meta.push_back(static_cast<std::int64_t>(info.dims.size()));
  for (const int d : info.dims) meta.push_back(d);
  img.ints(g, "meta", std::move(meta));
  img.bytes(g, "default", def);
  img.bytes(g, "global", glob);

  const auto nc = static_cast<std::size_t>(info.variable ? 1 : info.ncomp);
  // Values per entity are counted unless there is one row each.
  const bool counted = info.variable || info.layout.kind != Layout::Kind::per_entity;
  const auto rows_of = [&](int d, LO l) {
    return info.variable ? RowRange{0, 0} : fixed->space().entity_rows(d, l);
  };
  // Appends entity l's values: its rows (fixed) or its count and values.
  const auto put_values = [&](int d, LO l, Dataset& values, Dataset* counts) {
    if (info.variable) {
      const auto v = var->value_or_default(d, l);
      append(counts->data, static_cast<std::int64_t>(v.size()));
      append(values.data, std::span<const T>(v.data(), v.size()));
      return;
    }
    const RowRange r = rows_of(d, l);
    if (counts != nullptr) append(counts->data, static_cast<std::int64_t>(r.end - r.begin));
    for (LO k = r.begin; k < r.end; ++k) {
      for (std::size_t c = 0; c < nc; ++c) append(values.data, fixed->view()(k, static_cast<int>(c)));
    }
  };
  for (const int d : info.dims) {
    const int b = dom->find_block(d);
    const std::string dg = d == set_dim ? g + "/sets" : g + "/" + dim_group(d);
    img.groups.push_back(dg);
    if (d == set_dim) {  // the same on every rank: written once
      auto& members = img.single(dg + "/domain", DataType::i64);
      auto& values = img.single(dg + "/values", info.type, nc);
      Dataset* counts = counted ? &img.single(dg + "/counts", DataType::i64) : nullptr;
      if (b >= 0) {
        for (const LO l : dom->members(b)) {
          append(members.data, static_cast<std::int64_t>(l));
          put_values(d, l, values, counts);
        }
      }
      continue;
    }
    auto& part_full = img.column(dg + "/full", DataType::u8);
    auto& members = img.column(dg + "/domain", DataType::i64);
    auto& values = img.column(dg + "/values", info.type, nc);
    Dataset* counts = counted ? &img.column(dg + "/counts", DataType::i64) : nullptr;
    // The part's members among its owned entities, listed unless they are
    // all of them.
    const bool here = d <= mesh.dim() && mesh.count(d) > 0 && b >= 0;
    std::vector<LO> owned_members;
    if (here) {
      const LO owned = mesh.segment_table(d).owned_end;
      for (const LO l : dom->members(b)) {
        if (l < owned) owned_members.push_back(l);
      }
    }
    const bool full_here = here && static_cast<LO>(owned_members.size()) == mesh.segment_table(d).owned_end;
    append(part_full.data, static_cast<std::uint8_t>(full_here ? 1 : 0));
    for (const LO l : owned_members) {
      if (!full_here) append(members.data, static_cast<std::int64_t>(l));
      put_values(d, l, values, counts);
    }
  }
}

void lay_out_fields(const Comm& comm, const HostMesh& mesh, Image& img) {
  std::vector<const FieldInfo*> fields;
  const auto list = mesh.fields().list();
  for (const FieldInfo& f : list) {
    if (!internal(f.name)) fields.push_back(&f);
  }
  // Whether each field is full on each dimension on every rank: one reduction.
  std::vector<std::int32_t> full(4 * fields.size(), 1);
  for (std::size_t i = 0; i < fields.size(); ++i) {
    const FieldInfo& f = *fields[i];
    dispatch_data_type(f.type, [&]<class T>() {
      const auto& reg = mesh.fields();
      const Domain<Kokkos::HostSpace> dom =
          f.variable ? reg.template get_var<T>(f.id).domain() : reg.template get<T>(f.id).domain();
      for (int b = 0; b < dom.num_blocks(); ++b) {
        const int d = dom.info(b).dim;
        if (d != set_dim && dom.info(b).kind != DomainKind::full) full[4 * i + static_cast<std::size_t>(d)] = 0;
      }
    });
  }
  comm.allreduce(std::span<std::int32_t>(full), Op::min);
  img.groups.push_back("fields");
  for (std::size_t i = 0; i < fields.size(); ++i) {
    std::array<bool, 4> f{};
    for (std::size_t d = 0; d < 4; ++d) f[d] = full[4 * i + d] != 0;
    dispatch_data_type(fields[i]->type, [&]<class T>() { lay_out_field<T>(mesh, *fields[i], i, f, img); });
  }
  img.groups.push_back("values");
  const auto names = mesh.values().names();
  for (std::size_t i = 0; i < names.size(); ++i) {
    const MeshValue& v = mesh.values().at(names[i]);
    const std::string g = value_group(i);
    img.groups.push_back(g);
    img.text(g, "name", names[i]);
    std::vector<std::int64_t> meta{static_cast<std::int64_t>(v.type), v.ncomp, v.handle ? 1 : 0};
    const auto fw = format_words(v.format);
    meta.insert(meta.end(), fw.begin(), fw.end());
    meta.push_back(v.default_value ? 1 : 0);
    meta.push_back(v.global_value ? 1 : 0);
    img.ints(g, "meta", std::move(meta));
    if (v.default_value) img.bytes(g, "default", *v.default_value);
    if (v.global_value) img.bytes(g, "global", *v.global_value);
  }
}

// The sets: their list once; each part's members of its owned entities,
// (set, dimension, index, position in an ordered set or -1); and, from
// rank 0, the sets ordered sets list, (set, position, gid).
void lay_out_sets(const Comm& comm, const HostMesh& mesh, Image& img) {
  img.groups.push_back("sets");
  const auto& sets = mesh.sets();
  const auto sgid = sets.gid();
  auto& gid = img.single("sets/gid", DataType::i64);
  auto& flags = img.single("sets/flags", DataType::u8);
  auto& ccount = img.single("sets/children_counts", DataType::i64);
  auto& children = img.single("sets/children", DataType::i64);
  auto& pcount = img.single("sets/parents_counts", DataType::i64);
  auto& parents = img.single("sets/parents", DataType::i64);
  auto& mcount = img.single("sets/set_members_counts", DataType::i64);
  auto& smembers = img.single("sets/set_members", DataType::i64);
  auto& members = img.column("sets/members", DataType::i64, 4);
  auto& entries = img.column("sets/entries", DataType::i64, 3);
  for (LO i = 0; i < sets.count(); ++i) {
    append(gid.data, static_cast<std::int64_t>(sgid(i)));
    append(flags.data, sets.flags(i));
    const auto c = sets.children(i), p = sets.parents(i);
    append(ccount.data, static_cast<std::int64_t>(c.size()));
    for (const LO l : c) append(children.data, static_cast<std::int64_t>(sgid(l)));
    append(pcount.data, static_cast<std::int64_t>(p.size()));
    for (const LO l : p) append(parents.data, static_cast<std::int64_t>(sgid(l)));
    const auto member = [&](int d, LO l, std::int64_t pos) {
      for (const std::int64_t x : {std::int64_t{i}, std::int64_t{d}, std::int64_t{l}, pos}) append(members.data, x);
    };
    if (sets.ordered(i)) {
      append(mcount.data, std::int64_t{0});
      // Positions among the entries this rank writes (owned entities, and
      // on rank 0 set entries), so that copies and ghosts do not change them.
      std::int64_t pos = -1;
      for (const SetEntry& x : sets.sequence(i)) {
        const bool written =
            x.slot == set_dim ? comm.rank() == 0 : x.lid < mesh.segment_table(x.slot).owned_end;
        if (written) ++pos;
        if (x.slot == set_dim) {
          if (comm.rank() == 0) {
            for (const std::int64_t y : {std::int64_t{i}, pos, static_cast<std::int64_t>(sgid(x.lid))}) {
              append(entries.data, y);
            }
          }
        } else if (written) {
          member(x.slot, x.lid, pos);
        }
      }
      continue;
    }
    std::int64_t n = 0;
    if (sets.has_members(set_dim)) {
      for (const LO l : sets.members_of(i, set_dim)) {
        append(smembers.data, static_cast<std::int64_t>(sgid(l)));
        ++n;
      }
    }
    append(mcount.data, n);
    for (int d = 0; d <= mesh.dim(); ++d) {
      if (!sets.has_members(d)) continue;
      const LO owned = mesh.segment_table(d).owned_end;
      for (const LO l : sets.members_of(i, d)) {
        if (l < owned) member(d, l, -1);
      }
    }
  }
}

// The dataset creation list: no times, and at a compression level above 0
// chunks of about chunk_bytes with shuffle and deflate.
Id creation(int compression, hsize_t rows, std::size_t row_bytes, std::size_t width) {
  Id dcpl = h5::check(H5Pcreate(H5P_DATASET_CREATE), "create a dataset property list");
  h5::check(H5Pset_obj_track_times(dcpl.get(), false), "turn off dataset times");
  if (compression > 0 && rows > 0) {
    const std::array<hsize_t, 2> chunk{std::clamp<hsize_t>(chunk_bytes / row_bytes, 1, rows), width};
    h5::check(H5Pset_chunk(dcpl.get(), 2, chunk.data()), "chunk a dataset");
    h5::check(H5Pset_shuffle(dcpl.get()), "shuffle a dataset");
    h5::check(H5Pset_deflate(dcpl.get(), static_cast<unsigned>(compression)), "deflate a dataset");
  }
  return dcpl;
}

Id dataset(hid_t root, const std::string& path, hid_t type, hsize_t rows, std::size_t width, std::size_t row_bytes,
           int compression) {
  const std::array<hsize_t, 2> dims{rows, width};
  const Id space = h5::check(H5Screate_simple(2, dims.data(), nullptr), "create the dataspace of " + path);
  const Id dcpl = creation(compression, rows, row_bytes, width);
  return h5::check(H5Dcreate2(root, path.c_str(), type, space.get(), H5P_DEFAULT, dcpl.get(), H5P_DEFAULT),
                   "create the dataset " + path);
}

void write_attribute(hid_t root, const Attribute& a) {
  const Id obj = h5::check(H5Oopen(root, a.object.empty() ? "." : a.object.c_str(), H5P_DEFAULT),
                           "open " + a.object);
  const std::string what = "the attribute " + a.name + " of /soma/" + a.object;
  if (a.kind == Attribute::Kind::text) {
    const Id type = h5::check(H5Tcopy(H5T_C_S1), "copy the string type");
    h5::check(H5Tset_size(type.get(), std::max<std::size_t>(a.text.size(), 1)), "size a string type");
    const Id space = h5::check(H5Screate(H5S_SCALAR), "create the dataspace of " + what);
    const Id at = h5::check(H5Acreate2(obj.get(), a.name.c_str(), type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT),
                            "create " + what);
    std::string text = a.text;
    if (text.empty()) text.push_back('\0');
    h5::check(H5Awrite(at.get(), type.get(), text.data()), "write " + what);
    return;
  }
  const bool ints = a.kind == Attribute::Kind::ints;
  const hsize_t n = ints ? a.ints.size() : a.bytes.size();
  const Id space = h5::check(H5Screate_simple(1, &n, nullptr), "create the dataspace of " + what);
  const Id at = h5::check(H5Acreate2(obj.get(), a.name.c_str(), ints ? H5T_STD_I64LE : H5T_STD_U8LE, space.get(),
                                     H5P_DEFAULT, H5P_DEFAULT),
                          "create " + what);
  h5::check(H5Awrite(at.get(), ints ? H5T_NATIVE_INT64 : H5T_NATIVE_UINT8,
                     ints ? static_cast<const void*>(a.ints.data()) : static_cast<const void*>(a.bytes.data())),
            "write " + what);
}

// Where the columns of every part go: this part's first row of each, and
// the rows of each in all.
struct Placement {
  std::vector<std::uint64_t> first, total;
};

// The file's structure: groups, attributes and every dataset, created alike
// on every rank that writes (each rank with MPI-IO, else rank 0). The part
// table has a row per part and one more, the totals.
struct Created {
  Id file, root;
  std::vector<Id> columns, once;
  Id table;
  std::vector<std::uint64_t> totals;  // rows of each column
};

Created create(hid_t file, const Image& img, const Placement& at, std::uint64_t parts, int compression) {
  Created c;
  c.totals = at.total;
  c.root = h5::check(H5Gcreate2(file, soma_file::root, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "create /soma");
  for (const std::string& g : img.groups) {
    if (H5Lexists(c.root.get(), g.c_str(), H5P_DEFAULT) > 0) continue;
    (void)h5::check(H5Gcreate2(c.root.get(), g.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "create the group " + g);
  }
  for (std::size_t k = 0; k < img.columns.size(); ++k) {
    const Dataset& d = img.columns[k];
    c.columns.push_back(dataset(c.root.get(), d.path, soma_file::file_type(d.type), at.total[k], d.width,
                                d.row_bytes(), compression));
  }
  for (const Dataset& d : img.once) {
    c.once.push_back(dataset(c.root.get(), d.path, soma_file::file_type(d.type), d.rows(), d.width, d.row_bytes(),
                             compression));
  }
  {
    (void)h5::check(H5Gcreate2(c.root.get(), "parts", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "create the group parts");
    const std::array<hsize_t, 2> dims{parts + 1, img.columns.size()};
    const Id space = h5::check(H5Screate_simple(2, dims.data(), nullptr), "create the dataspace of the part table");
    const Id dcpl = creation(0, 0, 0, 0);
    c.table = h5::check(H5Dcreate2(c.root.get(), "parts/offsets", H5T_STD_U64LE, space.get(), H5P_DEFAULT, dcpl.get(),
                                   H5P_DEFAULT),
                        "create the part table");
  }
  for (const Attribute& a : img.attributes) write_attribute(c.root.get(), a);
  std::string names;
  for (const Dataset& d : img.columns) names += d.path + "\n";
  Attribute cols;
  cols.object = "parts/offsets";
  cols.name = "columns";
  cols.text = names;
  cols.kind = Attribute::Kind::text;
  write_attribute(c.root.get(), cols);
  return c;
}

// Rows [first, first + rows) of a dataset of `width` columns, or none.
struct Slab {
  Id file, memory;
};

Slab slab(hid_t ds, std::uint64_t first, std::uint64_t rows, std::size_t width) {
  Slab s;
  s.file = h5::check(H5Dget_space(ds), "get a dataspace");
  if (rows == 0) {
    h5::check(H5Sselect_none(s.file.get()), "select nothing");
    s.memory = h5::check(H5Scopy(s.file.get()), "copy a dataspace");
    h5::check(H5Sselect_none(s.memory.get()), "select nothing");
    return s;
  }
  const std::array<hsize_t, 2> start{first, 0}, count{rows, width};
  h5::check(H5Sselect_hyperslab(s.file.get(), H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr),
            "select rows");
  s.memory = h5::check(H5Screate_simple(2, count.data(), nullptr), "create a memory dataspace");
  return s;
}

// The rows a rank writes: its rows of each column at their place, its
// row of the part table (and the totals, from the last part), and every
// dataset written once when `once`.
struct Writes {
  std::vector<hid_t> datasets, types;
  std::vector<Slab> slabs;
  std::vector<const void*> buffers;
};

// A dataset without rows on any rank is left alone; HDF5 wants a buffer
// even where a rank selects nothing.
void add(Writes& w, hid_t ds, hid_t mem, Slab s, const void* data, std::uint64_t total) {
  static const std::uint64_t none = 0;
  if (total == 0) return;
  if (data == nullptr) data = &none;
  w.datasets.push_back(ds);
  w.types.push_back(mem);
  w.slabs.push_back(std::move(s));
  w.buffers.push_back(data);
}

void run(const Writes& w, hid_t dxpl) {
  if (w.datasets.empty()) return;
  std::vector<hid_t> mem, file;
  for (const Slab& s : w.slabs) {
    mem.push_back(s.memory.get());
    file.push_back(s.file.get());
  }
#if H5_VERSION_GE(1, 14, 0)
  // One transfer for them all.
  std::vector<hid_t> ds = w.datasets, types = w.types;
  std::vector<const void*> buf = w.buffers;
  h5::check(H5Dwrite_multi(ds.size(), ds.data(), types.data(), mem.data(), file.data(), dxpl, buf.data()),
            "write the datasets");
#else
  for (std::size_t k = 0; k < w.datasets.size(); ++k) {
    h5::check(H5Dwrite(w.datasets[k], w.types[k], mem[k], file[k], dxpl, w.buffers[k]), "write a dataset");
  }
#endif
}

// The part table rows a rank writes: its offsets, and the totals after them
// when it writes the last part.
std::vector<std::uint64_t> table_rows(const Placement& at, bool last) {
  std::vector<std::uint64_t> rows = at.first;
  if (last) rows.insert(rows.end(), at.total.begin(), at.total.end());
  return rows;
}

// Writes the rows of `img` (one part's, or a group's) at `first`, and
// `table`, part table rows from row `part` on (one per part, then the
// totals after the last part), and the datasets written once when `once`.
void write_rows(const Created& c, const Image& img, const std::vector<std::uint64_t>& first, std::uint64_t part,
                const std::vector<std::uint64_t>& table, bool once, hid_t dxpl) {
  Writes w;
  for (std::size_t k = 0; k < img.columns.size(); ++k) {
    const Dataset& d = img.columns[k];
    add(w, c.columns[k].get(), soma_file::memory_type(d.type), slab(c.columns[k].get(), first[k], d.rows(), d.width),
        d.data.data(), c.totals[k]);
  }
  const std::uint64_t rows = img.columns.empty() ? 0 : table.size() / img.columns.size();
  add(w, c.table.get(), H5T_NATIVE_UINT64, slab(c.table.get(), part, rows, img.columns.size()), table.data(), 1);
  for (std::size_t k = 0; k < img.once.size(); ++k) {
    const Dataset& d = img.once[k];
    add(w, c.once[k].get(), soma_file::memory_type(d.type), slab(c.once[k].get(), 0, once ? d.rows() : 0, d.width),
        d.data.data(), d.rows());
  }
  run(w, dxpl);
}

void write_part(const Created& c, const Image& img, const Placement& at, std::uint64_t part, bool last, bool once,
                hid_t dxpl) {
  write_rows(c, img, at.first, part, table_rows(at, last), once, dxpl);
}

Image lay_out(const Comm& comm, const HostMesh& mesh, const Agreed& a) {
  Image img;
  lay_out_entities(mesh, a, img);
  lay_out_fields(comm, mesh, img);
  lay_out_sets(comm, mesh, img);
  std::int64_t written = 0, input = 0, nodes = 0;
  for (std::size_t d = 0; d < 4; ++d) {
    written |= a.written[d] ? std::int64_t{1} << d : 0;
    input |= a.input[d] ? std::int64_t{1} << d : 0;
    nodes |= a.nodes[d] ? std::int64_t{1} << d : 0;
  }
  std::int64_t fields = 0;
  for (const FieldInfo& f : mesh.fields().list()) fields += internal(f.name) ? 0 : 1;
  img.ints("", "meta",
           {soma_file::format_version, a.dim, comm.size(), written, input, nodes, static_cast<std::int64_t>(a.graphs),
            mesh.sets().count(), fields, static_cast<std::int64_t>(mesh.values().size()),
            static_cast<std::int64_t>(mesh.ordering())});
  return img;
}

// Every column's rows as one message, after their counts.
std::vector<std::byte> pack(const Image& img) {
  std::vector<std::byte> out;
  for (const Dataset& d : img.columns) append(out, d.rows());
  for (const Dataset& d : img.columns) append(out, std::span<const std::byte>(d.data));
  return out;
}

void unpack(std::span<const std::byte> in, Image& img) {
  std::size_t at = 0;
  std::vector<std::uint64_t> rows(img.columns.size());
  for (auto& r : rows) {
    if (at + sizeof r > in.size()) throw std::runtime_error("a part's rows were cut short");
    std::memcpy(&r, in.data() + at, sizeof r);
    at += sizeof r;
  }
  for (std::size_t k = 0; k < img.columns.size(); ++k) {
    Dataset& d = img.columns[k];
    const std::size_t n = rows[k] * d.row_bytes();
    if (at + n > in.size()) throw std::runtime_error("a part's rows were cut short");
    d.data.assign(in.begin() + static_cast<std::ptrdiff_t>(at), in.begin() + static_cast<std::ptrdiff_t>(at + n));
    at += n;
  }
}

// With MPI-IO: every rank creates the structure and writes its rows in one
// collective transfer.
void write_collective(const Comm& comm, const std::string& name, const Image& img, const Placement& at,
                      int compression, const MpiIoHints& hints) {
  const Id fapl = soma_file::file_access(comm, true, false, hints);
  const Id file = h5::check(H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl.get()), "create the file");
  Created c = create(file.get(), img, at, static_cast<std::uint64_t>(comm.size()), compression);
  const Id dxpl = soma_file::transfer(true);
  write_part(c, img, at, static_cast<std::uint64_t>(comm.rank()), comm.rank() + 1 == comm.size(), comm.rank() == 0,
             dxpl.get());
}

// Without: rank 0 creates the file and writes its rows, then each other
// rank's in turn, as they arrive; a failure there still drains them.
void write_streamed(const Comm& comm, const std::string& name, Image& img, const Placement& at, int compression,
                    const MpiIoHints& hints) {
  if (comm.rank() != 0) {
    const std::vector<std::byte> rows = pack(img);
    const std::uint64_t n = rows.size();
    Request size = comm.issend(std::span<const std::uint64_t>(&n, 1), 0, tag_size);
    size.wait();
    Request data = comm.issend(std::span<const std::byte>(rows), 0, tag_rows);
    data.wait();
    return;
  }
  std::exception_ptr failure;
  std::optional<Id> file;
  Created c;
  const Id dxpl = soma_file::transfer(false);
  try {
    const Id fapl = soma_file::file_access(comm, false, false, hints);
    file = h5::check(H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl.get()), "create the file");
    c = create(file->get(), img, at, static_cast<std::uint64_t>(comm.size()), compression);
    write_part(c, img, at, 0, comm.size() == 1, true, dxpl.get());
  } catch (...) {
    failure = std::current_exception();
  }
  Placement next = at;
  for (std::size_t k = 0; k < img.columns.size(); ++k) next.first[k] += img.columns[k].rows();
  for (int r = 1; r < comm.size(); ++r) {
    std::uint64_t n = 0;
    Request size = comm.irecv(std::span<std::uint64_t>(&n, 1), r, tag_size);
    size.wait();
    std::vector<std::byte> rows(n);
    Request data = comm.irecv(std::span<std::byte>(rows), r, tag_rows);
    data.wait();
    if (failure) continue;
    try {
      unpack(rows, img);
      write_part(c, img, next, static_cast<std::uint64_t>(r), r + 1 == comm.size(), false, dxpl.get());
      for (std::size_t k = 0; k < img.columns.size(); ++k) next.first[k] += img.columns[k].rows();
    } catch (...) {
      failure = std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);
}

// With M aggregators: each rank sends its rows to its group's first rank;
// those M ranks alone open the file (over MPI-IO on their own communicator)
// and each writes its group's rows, contiguous in every column, and the
// group's part table rows. A failure in a leader still drains its members.
void write_aggregated(const Comm& comm, const std::string& name, Image& img, const Placement& at, int compression,
                      const MpiIoHints& hints, int m) {
  const int p = comm.size(), me = comm.rank(), leader = soma_file::leader_of(me, m, p);
  const SplitComm leaders = comm.split(leader == me ? 0 : -1, me);
  if (leader != me) {
    const std::vector<std::byte> rows = pack(img);
    const std::uint64_t n = rows.size();
    Request size = comm.issend(std::span<const std::uint64_t>(&n, 1), leader, tag_group_size);
    size.wait();
    Request data = comm.issend(std::span<const std::byte>(rows), leader, tag_group_rows);
    data.wait();
    return;
  }
  // This group's parts: this rank's, then each member's in rank order.
  int end = me + 1;
  while (end < p && soma_file::leader_of(end, m, p) == me) ++end;
  std::exception_ptr failure;
  std::vector<std::uint64_t> table = at.first, next = at.first;
  for (std::size_t k = 0; k < img.columns.size(); ++k) next[k] += img.columns[k].rows();
  Image member;  // a member's rows, in columns shaped like this rank's
  for (const Dataset& d : img.columns) member.columns.push_back(Dataset{d.path, d.type, d.width, {}});
  for (int r = me + 1; r < end; ++r) {
    std::uint64_t n = 0;
    Request size = comm.irecv(std::span<std::uint64_t>(&n, 1), r, tag_group_size);
    size.wait();
    std::vector<std::byte> rows(n);
    Request data = comm.irecv(std::span<std::byte>(rows), r, tag_group_rows);
    data.wait();
    if (failure) continue;
    try {
      unpack(rows, member);
      table.insert(table.end(), next.begin(), next.end());
      for (std::size_t k = 0; k < img.columns.size(); ++k) {
        Dataset& d = img.columns[k];
        d.data.insert(d.data.end(), member.columns[k].data.begin(), member.columns[k].data.end());
        next[k] += member.columns[k].rows();
      }
    } catch (...) {
      failure = std::current_exception();
    }
  }
  if (end == p) table.insert(table.end(), at.total.begin(), at.total.end());
  // Every leader joins the collective calls, a failed one with nothing.
  const Comm& group = leaders.comm();
  const Id fapl = soma_file::file_access(group, soma_file::parallel_hdf5(), false, hints);
  const Id file = h5::check(H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl.get()), "create the file");
  const Created c = create(file.get(), img, at, static_cast<std::uint64_t>(p), compression);
  const Id dxpl = soma_file::transfer(soma_file::parallel_hdf5());
  if (failure) {
    for (Dataset& d : img.columns) d.data.clear();
    table.clear();
  }
  write_rows(c, img, at.first, static_cast<std::uint64_t>(me), table, me == 0 && !failure, dxpl.get());
  if (failure) std::rethrow_exception(failure);
}

// Runs f; a failure on any rank throws on every rank (std::runtime_error
// naming the path), the failing ones with their own cause.
template <class F>
void collectively(const Comm& comm, const fs::path& path, const char* writer, F&& f) {
  std::string cause;
  try {
    f();
  } catch (const std::bad_alloc&) {
    cause = "out of memory";
  } catch (const std::exception& e) {
    cause = e.what();
  }
  if (comm.allreduce(static_cast<std::int32_t>(cause.empty() ? 0 : 1), Op::max) != 0) {
    throw std::runtime_error(io::cannot_write(writer, path, cause.empty() ? "failed on another rank" : cause));
  }
}

}  // namespace

void write_soma(const Comm& comm, const HostMesh& mesh, const fs::path& path, const SomaWriteOptions& options) {
  if (mesh.edited() && comm.size() == 1) {
    HostMesh compact = mesh.clone();
    compact.compress();
    write_soma(comm, compact, path, options);
    return;
  }
  std::string error;
  if (options.compression < 0 || options.compression > 9) {
    error = "compression is a deflate level in [0, 9], not " + std::to_string(options.compression);
  } else if (options.compression > 0 && H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0) {
    error = "this HDF5 has no deflate filter";
  } else if (static_cast<std::size_t>(mesh.count(0)) != mesh.coords().extent(0)) {
    error = "the mesh holds " + std::to_string(mesh.count(0)) + " vertices but not their coordinates";
  } else {
    error = soma_file::access_error(options.hints, options.aggregators, comm.size());
  }
  const Agreed agreed =
      agree(comm, mesh, error, options.derived, soma_file::access_hash(options.hints, options.aggregators));
  Image img = lay_out(comm, mesh, agreed);

  soma_file::write_image(comm, path, img, options, who);
}

void soma_file::write_image(const Comm& comm, const fs::path& path, Image& img, const SomaWriteOptions& options,
                            const char* writer) {
  // Where each part's rows go: one exscan and one sum over all columns.
  Placement at;
  for (const Dataset& d : img.columns) at.first.push_back(d.rows());
  at.total = at.first;
  comm.exscan(std::span<std::uint64_t>(at.first));
  comm.allreduce(std::span<std::uint64_t>(at.total), Op::sum);

  // Rank 0 makes the temporary file and tells the others its name.
  std::optional<io::Temporary> file;
  std::array<char, 4096> name{};
  std::string cause;
  if (comm.rank() == 0) {
    try {
      file.emplace(writer, path, io::random_names(path), temporary_attempts);
      file->close();
      if (file->name().size() >= name.size()) throw std::runtime_error("the temporary file name is too long");
      std::memcpy(name.data(), file->name().data(), file->name().size());
    } catch (const std::exception& e) {
      cause = e.what();
      file.reset();
    }
  }
  comm.bcast(std::span<char>(name), 0);
  if (name[0] == '\0') {
    if (!cause.empty()) throw std::runtime_error(cause);
    throw std::runtime_error(io::cannot_write(writer, path, "rank 0 could not create the file"));
  }
  const std::string temporary(name.data());
  const bool collective = soma_file::parallel_hdf5() && comm.size() > 1 && !streamed;
  collectively(comm, path, writer, [&] {
    const h5::Session quiet;
    if (options.aggregators > 0 && soma_file::parallel_hdf5() && !streamed) {
      write_aggregated(comm, temporary, img, at, options.compression, options.hints, options.aggregators);
    } else if (collective) {
      write_collective(comm, temporary, img, at, options.compression, options.hints);
    } else {
      write_streamed(comm, temporary, img, at, options.compression, options.hints);
    }
  });
  // Rank 0 renames it over `path`; the others learn whether that worked.
  std::int32_t ok = 1;
  if (comm.rank() == 0) {
    try {
      file->commit();
    } catch (const std::exception& e) {
      ok = 0;
      cause = e.what();
    }
  }
  comm.bcast(std::span<std::int32_t>(&ok, 1), 0);
  if (ok == 0) {
    if (!cause.empty()) throw std::runtime_error(cause);
    throw std::runtime_error(io::cannot_write(writer, path, "rank 0 could not rename the file"));
  }
}

void detail::soma_streamed_sink(bool on) noexcept { streamed = on; }

void write_soma(const HostMesh& mesh, const fs::path& path, const SomaWriteOptions& options) {
  write_soma(Comm::self(), mesh, path, options);
}

}  // namespace soma
