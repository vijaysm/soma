// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Interface: load_file and write_file. Reading takes a file's fields and sets
// into tags and sets kept here and keeps the mesh bare; writing puts them
// back on a copy of the mesh and writes that.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "core_state.hpp"
#include "soma/compat/cn.hpp"
#include "soma/compat/core.hpp"
#include "soma/core/handle_ref.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/vtk_reader.hpp"
#include "soma/io/vtk_writer.hpp"
#include "soma/topology/canonical.hpp"
#include "soma/topology/nodes.hpp"

namespace soma::compat {

namespace {

using detail::CoreState;
using detail::SetData;

std::string lower_extension(const char* file_name) {
  std::string ext = std::filesystem::path(file_name).extension().string();
  for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return ext;
}

bool no_options(const char* options) { return options == nullptr || *options == '\0'; }

// --- Reading --------------------------------------------------------------

EntityHandle handle_of(const CoreState& s, int slot, LO lid) {
  if (slot == set_dim) return make_handle(MBENTITYSET, static_cast<EntityID>(lid) + 1);
  return s.handle_of[static_cast<std::size_t>(slot)][static_cast<std::size_t>(lid)];
}

void import_sets(CoreState& s, const HostMesh& m) {
  const auto& sets = m.sets();
  s.sets.assign(static_cast<std::size_t>(sets.count()), SetData{});
  if (sets.count() > 0) s.alloc_run(MBENTITYSET, 0, sets.count());
  for (LO x = 0; x < sets.count(); ++x) {
    SetData& d = s.sets[static_cast<std::size_t>(x)];
    d.flags = sets.flags(x);
    if (sets.ordered(x)) {
      for (const SetEntry& e : sets.sequence(x)) d.sequence.push_back(handle_of(s, e.slot, e.lid));
    } else {
      std::vector<EntityHandle> all;
      for (int slot = 0; slot <= set_dim; ++slot) {
        if (!sets.has_members(slot)) continue;
        for (const LO l : sets.members_of(x, slot)) all.push_back(handle_of(s, slot, l));
      }
      std::sort(all.begin(), all.end());
      for (const EntityHandle h : all) d.contents.insert(h);
    }
    for (const LO c : sets.children(x)) d.children.push_back(handle_of(s, set_dim, c));
    for (const LO p : sets.parents(x)) d.parents.push_back(handle_of(s, set_dim, p));
  }
}

// High-order elements that disagree on the node of a side they share, named
// MOAB's way, or an empty string: MOAB's structures need such nodes shared.
std::string nonconforming(const CoreState& s) {
  const auto v = s.mesh.view();
  std::map<std::vector<LO>, std::pair<LO, EntityHandle>> seen;  // side dim, sorted corners -> node, element
  const auto name = [](EntityHandle h) { return std::string(CN::EntityTypeName(type_of(h))) + " " + std::to_string(id_of(h)); };
  for (int d = 2; d <= s.mesh.dim(); ++d) {
    for (LO e = 0; e < s.mesh.count(d); ++e) {
      const LO n = v.num_nodes(d, e);
      const LO corners = v.num_vertices(d, e);
      if (n == corners) continue;
      const Topology t = v.topology(d, e);
      for (LO k = corners; k < n; ++k) {
        const NodeParent p = node_parent(t, static_cast<int>(n), static_cast<int>(k));
        if (p.dim <= 0 || p.dim >= d) continue;  // nodes of the element itself are its own
        const canonical::SideDef side = canonical::side(t, p.dim, p.index);
        std::vector<LO> key{p.dim};
        for (int q = 0; q < side.num_vertices; ++q) key.push_back(v.node(d, e, side.v[q]));
        std::sort(key.begin() + 1, key.end());
        const EntityHandle h = handle_of(s, d, e);
        const auto [it, fresh] = seen.emplace(key, std::pair<LO, EntityHandle>{v.node(d, e, k), h});
        if (!fresh && it->second.first != v.node(d, e, k)) {
          return name(it->second.second) + " and " + name(h) + " disagree on the node of a shared " +
                 (p.dim == 1 ? "edge" : "face");
        }
      }
    }
  }
  return {};
}

// How a field's values are kept as a tag.
struct Kind {
  DataType type = MB_TYPE_OPAQUE;
  int length = 1;  // values per entity (bits for bit tags)
  int bytes = 0;
};

Kind kind_of(soma::DataType t, int ncomp, bool handle, const TagFormat& f) {
  if (handle) return {MB_TYPE_HANDLE, ncomp, ncomp * value_size(MB_TYPE_HANDLE)};
  if (f.kind == TagFormat::Kind::bit) return {MB_TYPE_BIT, f.nbits, 1};
  if (f.kind == TagFormat::Kind::opaque) return {MB_TYPE_OPAQUE, ncomp, ncomp};
  switch (t) {
    case soma::DataType::f32:
    case soma::DataType::f64: return {MB_TYPE_DOUBLE, ncomp, ncomp * value_size(MB_TYPE_DOUBLE)};
    case soma::DataType::i64:
    case soma::DataType::u64: return {MB_TYPE_OPAQUE, 8 * ncomp, 8 * ncomp};  // kept whole, as bytes
    default: return {MB_TYPE_INTEGER, ncomp, ncomp * value_size(MB_TYPE_INTEGER)};
  }
}

// Values of type T as the bytes of their tag kind (a handle value becomes the
// handle of what it names here, 0 when it names nothing).
template <class T>
void append_value(const CoreState& s, const HandleRef<Kokkos::HostSpace>* ref, const Kind& k, const T* v, int n,
                  std::vector<unsigned char>& out) {
  const auto put = [&](const auto& x) {
    const auto* p = reinterpret_cast<const unsigned char*>(&x);
    out.insert(out.end(), p, p + sizeof(x));
  };
  for (int c = 0; c < n; ++c) {
    switch (k.type) {
      case MB_TYPE_DOUBLE: put(static_cast<double>(v[c])); break;
      case MB_TYPE_INTEGER: put(static_cast<int>(v[c])); break;
      case MB_TYPE_HANDLE: {
        EntityHandle h = 0;
        if constexpr (std::is_same_v<T, GO>) {
          const HandleTarget x = ref != nullptr ? ref->resolve(v[c]) : HandleTarget{};
          if (x.valid()) h = handle_of(s, x.slot, x.lid);
        }
        put(h);
        break;
      }
      default: put(v[c]);  // opaque and bit bytes, and whole 64-bit values
    }
  }
}

template <class T>
std::vector<unsigned char> bytes_of(const CoreState& s, const HandleRef<Kokkos::HostSpace>* ref, const Kind& k,
                                    const std::vector<T>& v) {
  std::vector<unsigned char> out;
  append_value(s, ref, k, v.data(), static_cast<int>(v.size()), out);
  return out;
}

TagInfo* tag_for(CoreState& s, const std::string& name, const Kind& k, bool varlen, int storage_class, bool full) {
  TagInfo* t = s.find_tag(name);
  if (t != nullptr && (t->type != k.type || t->varlen != varlen || (!varlen && t->bytes != k.bytes))) {
    // A file's tag of another shape replaces the instance's declaration.
    s.tags.erase(std::find_if(s.tags.begin(), s.tags.end(), [&](const auto& p) { return p.get() == t; }));
    t = nullptr;
  }
  if (t == nullptr) {
    auto made = std::make_unique<TagInfo>();
    made->name = name;
    made->type = k.type;
    made->varlen = varlen;
    made->length = k.length;
    made->bytes = varlen ? 0 : k.bytes;
    made->storage = k.type == MB_TYPE_BIT       ? MB_TAG_BIT
                    : storage_class == 2         ? MB_TAG_DENSE
                    : storage_class == 3         ? MB_TAG_MESH
                    : storage_class == 1 || !full ? MB_TAG_SPARSE
                                                  : MB_TAG_DENSE;
    t = made.get();
    s.tags.push_back(std::move(made));
  }
  return t;
}

template <class T>
void import_field(CoreState& s, const HostMesh& m, const FieldInfo& info, const HandleRef<Kokkos::HostSpace>* ref) {
  const TagFormat& f = m.fields().tag_format(info.family);
  const Kind k = kind_of(info.type, info.variable ? 1 : info.ncomp, info.handle, f);
  const auto full = [&](const Space<Kokkos::HostSpace>& space) {
    for (int b = 0; b < space.num_blocks(); ++b) {
      const auto block = space.block(b);
      const LO n = block.domain.dim == set_dim ? m.sets().count() : m.count(block.domain.dim);
      if (block.domain.size != n) return false;
    }
    return true;
  };
  const auto each = [&](const Space<Kokkos::HostSpace>& space, const auto& visit) {
    for (int b = 0; b < space.num_blocks(); ++b) {
      const auto block = space.block(b);
      for (LO q = 0; q < block.domain.size; ++q) {
        visit(handle_of(s, block.domain.dim, block.domain.entity(q)),
              static_cast<std::size_t>(block.row_begin + block.first(q)));
      }
    }
  };
  if (info.variable) {
    const auto field = m.fields().template get_var<T>(info.id);
    TagInfo* t = tag_for(s, info.name, k, true, f.storage_class, full(field.space()));
    if (field.default_value() && !t->has_def) {
      t->def = bytes_of(s, ref, k, *field.default_value());
      t->has_def = true;
    }
    if (field.global_value() && !t->has_mesh) {
      t->mesh_value = bytes_of(s, ref, k, *field.global_value());
      t->has_mesh = true;
    }
    const auto offsets = field.offsets();
    const auto values = field.values();
    each(field.space(), [&](EntityHandle h, std::size_t r) {
      const auto kt = static_cast<std::size_t>(type_of(h));
      const auto at = static_cast<std::size_t>(id_of(h) - 1);
      if (t->vdata[kt].size() <= at) t->vdata[kt].resize(at + 1);
      if (t->has[kt].size() <= at) t->has[kt].resize(at + 1, 0);
      t->vdata[kt][at].clear();
      append_value(s, ref, k, values.data() + offsets(r), static_cast<int>(offsets(r + 1) - offsets(r)), t->vdata[kt][at]);
      t->has[kt][at] = 1;
    });
    return;
  }
  const auto field = m.fields().template get<T>(info.id);
  TagInfo* t = tag_for(s, info.name, k, false, f.storage_class, full(field.space()));
  if (!field.default_value().empty() && !t->has_def) {
    t->def = bytes_of(s, ref, k, field.default_value());
    t->has_def = true;
  }
  if (!field.global_value().empty() && !t->has_mesh) {
    t->mesh_value = bytes_of(s, ref, k, field.global_value());
    t->has_mesh = true;
  }
  const auto view = field.view();
  std::vector<T> row(static_cast<std::size_t>(info.ncomp));
  std::vector<unsigned char> value;
  each(field.space(), [&](EntityHandle h, std::size_t r) {
    for (int c = 0; c < info.ncomp; ++c) row[static_cast<std::size_t>(c)] = view(r, static_cast<std::size_t>(c));
    value.clear();
    append_value(s, ref, k, row.data(), info.ncomp, value);
    std::memcpy(t->slot(type_of(h), id_of(h)), value.data(), value.size());
    t->has[static_cast<std::size_t>(type_of(h))][static_cast<std::size_t>(id_of(h) - 1)] = 1;
    allocate_dense(s, *t, type_of(h), id_of(h));
  });
}

void import_value(CoreState& s, const std::string& name, const MeshValue& v,
                  const HandleRef<Kokkos::HostSpace>* ref) {
  const Kind k = kind_of(v.type, v.ncomp == 0 ? 1 : v.ncomp, v.handle, v.format);
  TagInfo* t = tag_for(s, name, k, v.ncomp == 0, v.format.storage_class, true);
  dispatch_data_type(v.type, [&]<class T>() {
    if (v.default_value && !t->has_def) {
      t->def = bytes_of(s, ref, k, *v.default_as<T>());
      t->has_def = true;
    }
    if (v.global_value && !t->has_mesh) {
      t->mesh_value = bytes_of(s, ref, k, *v.global_as<T>());
      t->has_mesh = true;
    }
  });
}

// --- Writing --------------------------------------------------------------

struct Export {
  const CoreState& s;
  HostMesh& out;
  std::vector<GO> set_gid;  // by set index; 0 for deleted sets
};

GO soma_handle(const Export& x, EntityHandle h) {
  if (h == 0) return 0;
  if (type_of(h) == MBENTITYSET) {
    const auto i = static_cast<std::size_t>(id_of(h) - 1);
    return x.s.is_set(h) ? soma::make_handle(set_dim, x.set_gid[i]) : 0;
  }
  const LO l = x.s.lid(h);
  if (l == invalid_lo) return 0;
  const int d = CN::Dimension(type_of(h));
  return soma::make_handle(d, x.out.gid(d)(static_cast<std::size_t>(l)));
}

template <class T>
std::vector<T> typed(const Export& x, const TagInfo& t, const unsigned char* p, std::size_t n) {
  std::vector<T> out;
  if constexpr (std::is_same_v<T, GO>) {
    for (std::size_t i = 0; i < n / sizeof(EntityHandle); ++i) {
      EntityHandle h = 0;
      std::memcpy(&h, p + i * sizeof(EntityHandle), sizeof(h));
      out.push_back(t.type == MB_TYPE_HANDLE ? soma_handle(x, h) : static_cast<GO>(h));
    }
  } else {
    out.resize(n / sizeof(T));
    if (n > 0) std::memcpy(out.data(), p, out.size() * sizeof(T));
  }
  return out;
}

// One slot's entities with values: their local ids (sorted) and handles.
struct SlotValues {
  std::vector<LO> lids;
  std::vector<EntityHandle> handles;
};

SlotValues values_on(const Export& x, const TagInfo& t, int slot) {
  SlotValues v;
  if (slot == set_dim) {
    LO next = 0;
    for (std::size_t i = 0; i < x.s.sets.size(); ++i) {
      if (!x.s.sets[i].alive) continue;
      const EntityHandle h = make_handle(MBENTITYSET, static_cast<EntityID>(i + 1));
      if (t.has_value(MBENTITYSET, id_of(h))) {
        v.lids.push_back(next);
        v.handles.push_back(h);
      }
      ++next;
    }
    return v;
  }
  const auto& hs = x.s.handle_of[static_cast<std::size_t>(slot)];
  for (std::size_t l = 0; l < hs.size(); ++l) {
    const EntityHandle h = hs[l];
    if (x.s.lid(h) == invalid_lo || !t.has_value(type_of(h), id_of(h))) continue;
    v.lids.push_back(static_cast<LO>(l));
    v.handles.push_back(h);
  }
  return v;
}

template <class T>
void export_tag(Export& x, const TagInfo& t, int ncomp, const TagFormat& format) {
  FieldOptions<T> options;
  options.handle = t.type == MB_TYPE_HANDLE;
  if (t.has_def) options.default_value = typed<T>(x, t, t.def.data(), t.def.size());
  if (t.has_mesh) options.global_value = typed<T>(x, t, t.mesh_value.data(), t.mesh_value.size());
  bool any = false;
  for (int slot = 0; slot <= set_dim; ++slot) {
    if (slot != set_dim && slot > x.out.dim()) continue;
    const SlotValues v = values_on(x, t, slot);
    if (v.lids.empty()) continue;
    any = true;
    const Space<Kokkos::HostSpace> space(Domain<Kokkos::HostSpace>::from_list(x.out, slot, v.lids),
                                         Layout::per_entity());
    if (t.varlen) {
      std::vector<Offset> offsets{0};
      std::vector<T> values;
      for (const EntityHandle h : v.handles) {
        const auto& b = t.vdata[static_cast<std::size_t>(type_of(h))][static_cast<std::size_t>(id_of(h) - 1)];
        const std::vector<T> one = typed<T>(x, t, b.data(), b.size());
        values.insert(values.end(), one.begin(), one.end());
        offsets.push_back(static_cast<Offset>(values.size()));
      }
      x.out.fields().template add_var<T>(t.name, space, offsets, values, options);
    } else {
      auto f = x.out.fields().template add<T>(t.name, space, ncomp, options);
      const auto view = f.view();
      for (std::size_t r = 0; r < v.handles.size(); ++r) {
        const EntityHandle h = v.handles[r];
        const unsigned char* p = t.data[static_cast<std::size_t>(type_of(h))].data() +
                                 static_cast<std::size_t>(id_of(h) - 1) * static_cast<std::size_t>(t.bytes);
        const std::vector<T> one = typed<T>(x, t, p, static_cast<std::size_t>(t.bytes));
        for (int c = 0; c < ncomp; ++c) view(r, static_cast<std::size_t>(c)) = one[static_cast<std::size_t>(c)];
      }
    }
  }
  if (any) {
    for (const FieldInfo& info : x.out.fields().list()) {
      if (info.name == t.name) x.out.fields().set_tag_format(info.family, format);
    }
    return;
  }
  // No entity holds a value: the tag is declared as a mesh value.
  MeshValue mv;
  mv.type = data_type_of<T>();
  mv.ncomp = t.varlen ? 0 : ncomp;
  mv.handle = options.handle;
  mv.format = format;
  if (options.default_value) mv.default_value = MeshValue::bytes_of(std::span<const T>(*options.default_value));
  if (options.global_value) mv.global_value = MeshValue::bytes_of(std::span<const T>(*options.global_value));
  x.out.values().set(t.name, std::move(mv));
}

void export_tags(Export& x) {
  for (const auto& p : x.s.tags) {
    const TagInfo& t = *p;
    TagFormat format;
    format.storage_class = t.storage == MB_TAG_BIT     ? 0
                           : t.storage == MB_TAG_DENSE ? 2
                           : t.storage == MB_TAG_MESH  ? 3
                                                       : 1;
    switch (t.type) {
      case MB_TYPE_INTEGER: export_tag<std::int32_t>(x, t, t.length, format); break;
      case MB_TYPE_DOUBLE: export_tag<double>(x, t, t.length, format); break;
      case MB_TYPE_HANDLE: export_tag<GO>(x, t, t.length, format); break;
      case MB_TYPE_BIT:
        format.kind = TagFormat::Kind::bit;
        format.nbits = t.length;
        export_tag<std::uint8_t>(x, t, 1, format);
        break;
      default:
        format.kind = TagFormat::Kind::opaque;
        export_tag<std::uint8_t>(x, t, t.length, format);
    }
  }
}

void export_sets(Export& x) {
  GO next = 1;
  x.set_gid.assign(x.s.sets.size(), 0);
  for (std::size_t i = 0; i < x.s.sets.size(); ++i) {
    if (x.s.sets[i].alive) x.set_gid[i] = next++;
  }
  const auto item = [&](EntityHandle h) {
    if (type_of(h) == MBENTITYSET) return SetItem{set_dim, x.set_gid[static_cast<std::size_t>(id_of(h) - 1)]};
    return SetItem{CN::Dimension(type_of(h)), static_cast<GO>(x.s.lid(h))};
  };
  SetBuilder b;
  for (std::size_t i = 0; i < x.s.sets.size(); ++i) {
    const SetData& d = x.s.sets[i];
    if (!d.alive) continue;
    SetSpec spec;
    spec.gid = x.set_gid[i];
    spec.flags = static_cast<std::uint8_t>(d.flags & set_flag_bits);
    if (d.ordered()) {
      for (const EntityHandle h : d.sequence) spec.contents.push_back(item(h));
    } else {
      for (const EntityHandle h : d.contents) spec.contents.push_back(item(h));
    }
    for (const EntityHandle c : d.children) spec.children.push_back(x.set_gid[static_cast<std::size_t>(id_of(c) - 1)]);
    for (const EntityHandle p : d.parents) spec.parents.push_back(x.set_gid[static_cast<std::size_t>(id_of(p) - 1)]);
    b.add(std::move(spec));
  }
  std::move(b).install(x.out);
}

}  // namespace

ErrorCode Interface::load_file(const char* file_name, const EntityHandle* file_set, const char* options,
                               const char* set_tag_name, const int* /*set_tag_values*/, int num_set_tag_values) {
  if (!no_options(options)) return s_->fail(MB_UNHANDLED_OPTION, std::string("load_file: options '") + options + "' are not supported");
  if (set_tag_name != nullptr || num_set_tag_values > 0) {
    return s_->fail(MB_NOT_IMPLEMENTED, "load_file: reading part of a file is not supported");
  }
  s_->flush();
  if (s_->loaded || s_->mesh.count(0) > 0 || !s_->sets.empty()) {
    return s_->fail(MB_NOT_IMPLEMENTED, "load_file: this instance holds a mesh already");
  }
  if (file_name == nullptr || !std::filesystem::exists(file_name)) {
    return s_->fail(MB_FILE_DOES_NOT_EXIST, std::string("load_file: no file '") + (file_name ? file_name : "") + "'");
  }
  const std::string ext = lower_extension(file_name);
  HostMesh m;
  try {
    if (ext == ".vtk") {
      m = read_vtk(file_name);
    } else if (ext == ".h5m") {
      if (!h5m_available()) return s_->fail(MB_NOT_IMPLEMENTED, "load_file: SOMA was built without HDF5");
      m = read_h5m(file_name);
    } else {
      return s_->fail(MB_NOT_IMPLEMENTED, "load_file: '" + ext + "' files are not supported");
    }
    s_->mesh = m.mirror<Kokkos::HostSpace>(TransferOptions{{true, true, true, true}, true, {}, {}, {}, false});
    MeshValues none;
    s_->mesh.values().swap(none);
    s_->number_entities();
    if (const std::string bad = nonconforming(*s_); !bad.empty()) throw std::runtime_error("load_file: " + bad);
    import_sets(*s_, m);
    std::optional<HandleRef<Kokkos::HostSpace>> ref;
    const auto handles = [&]() -> const HandleRef<Kokkos::HostSpace>* {
      if (!ref) ref = HandleRef<Kokkos::HostSpace>::build(m);
      return &*ref;
    };
    for (const FieldInfo& info : m.fields().list()) {
      if (info.layout.kind != Layout::Kind::per_entity) continue;
      dispatch_data_type(info.type, [&]<class T>() {
        import_field<T>(*s_, m, info, info.handle ? handles() : nullptr);
      });
    }
    for (const std::string& name : m.values().names()) {
      const MeshValue& v = m.values().at(name);
      import_value(*s_, name, v, v.handle ? handles() : nullptr);
    }
  } catch (const std::exception& e) {
    s_ = std::make_unique<detail::CoreState>();
    detail::declare_default_tags(*this);
    return s_->fail(MB_FAILURE, e.what());
  }
  s_->loaded = true;
  if (file_set != nullptr && *file_set != 0) {
    Range all;
    get_entities_by_handle(0, all);
    return add_entities(*file_set, all);
  }
  return MB_SUCCESS;
}

ErrorCode Interface::load_mesh(const char* file_name, const int* block_ids, int num_blocks) {
  if (block_ids != nullptr && num_blocks > 0) return s_->fail(MB_NOT_IMPLEMENTED, "load_mesh: reading blocks is not supported");
  return load_file(file_name);
}

ErrorCode Interface::write_file(const char* file_name, const char* file_type, const char* options,
                                const EntityHandle* output_sets, int num_output_sets, const Tag* tag_list,
                                int num_tags) {
  if (!no_options(options)) return s_->fail(MB_UNHANDLED_OPTION, std::string("write_file: options '") + options + "' are not supported");
  if ((output_sets != nullptr && num_output_sets > 0) || (tag_list != nullptr && num_tags > 0)) {
    return s_->fail(MB_NOT_IMPLEMENTED, "write_file: writing part of a mesh is not supported");
  }
  std::string ext = file_type != nullptr && *file_type != '\0' ? std::string(".") + file_type : lower_extension(file_name);
  for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  s_->flush();
  try {
    TransferOptions o{{true, true, true, true}, true, {}, {}, {}, false};
    HostMesh out = s_->mesh.mirror<Kokkos::HostSpace>(o);
    Export x{*s_, out, {}};
    export_sets(x);
    export_tags(x);
    if (ext == ".h5m") {
      if (!h5m_available()) return s_->fail(MB_NOT_IMPLEMENTED, "write_file: SOMA was built without HDF5");
      write_h5m(out, file_name);
    } else if (ext == ".vtk") {
      // Tags with a value on every vertex, or on every element of the mesh
      // dimension, as MOAB writes them (integers and reals only).
      std::vector<VtkField> fields;
      for (const FieldInfo& info : out.fields().list()) {
        if (info.variable || info.handle || info.dims.size() != 1) continue;
        const int d = info.dims[0];
        if (d != 0 && d != out.dim()) continue;
        if (info.rows != out.num_live(d)) continue;
        if (info.type != soma::DataType::i32 && info.type != soma::DataType::f64) continue;
        if (out.fields().tag_format(info.family).kind != TagFormat::Kind::plain) continue;
        fields.push_back(VtkField{info.name, d});
      }
      write_vtk(out, file_name, fields, VtkWriteOptions{VtkLegacyVersion::v3_0});
    } else {
      return s_->fail(MB_NOT_IMPLEMENTED, "write_file: '" + ext + "' files are not supported");
    }
  } catch (const std::exception& e) {
    return s_->fail(MB_FILE_WRITE_ERROR, e.what());
  }
  return MB_SUCCESS;
}

ErrorCode Interface::write_file(const char* file_name, const char* file_type, const char* options,
                                const Range& output_sets, const Tag* tag_list, int num_tags) {
  const std::vector<EntityHandle> v(output_sets.begin(), output_sets.end());
  return write_file(file_name, file_type, options, v.data(), static_cast<int>(v.size()), tag_list, num_tags);
}

ErrorCode Interface::write_mesh(const char* file_name, const EntityHandle* output_sets, int num_sets) {
  return write_file(file_name, nullptr, nullptr, output_sets, num_sets);
}

ErrorCode Interface::delete_mesh() {
  s_ = std::make_unique<detail::CoreState>();
  detail::declare_default_tags(*this);
  return MB_SUCCESS;
}

}  // namespace soma::compat
