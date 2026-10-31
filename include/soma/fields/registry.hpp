// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_REGISTRY_HPP_
#define SOMA_FIELDS_REGISTRY_HPP_

// Host-side owner of fields. Type erasure exists only here; kernels receive
// concretely typed Views. A field is identified by (name, value type, space);
// a second definition with all three equal is a FieldClash. FieldIds are
// indices that are never reused, so a removed field leaves a gap.
//
// Fields with the same name, value type, component count, and layout form a
// family (the layout is part of the key so that a per_entity field and a node
// field of one name can coexist on the same entities).
// Members of a family must be disjoint: add() throws FieldOverlap otherwise
// (the check is exact and only compares members that share a dimension). A
// family resolves an entity to (member, row) through a per-dimension index that
// is built on first use and rebuilt after the epoch or the membership changes;
// see resolver.hpp. Fields of one family on the same dimension may differ in
// layout, so a family lookup returns the first value point of the entity.
//
// The registry holds the current mesh epoch. Mesh (or a test) sets it with
// set_epoch(), and fetching a field whose domain was built against another
// epoch throws StaleField. Kernels get Views and are not checked on the device.
// The registry needs only entity counts (and optionally global ids) per
// dimension, supplied through set_entities(); it never includes the mesh.
//
// Thread safety: none, const members included. Lazy family-index builds and the
// automatic commit in mirror_into write to mutable members, so a const registry
// is not safe for concurrent use. Take family handles before spawning threads.

#include <algorithm>
#include <any>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/errors.hpp"
#include "soma/fields/field.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/overlap.hpp"
#include "soma/fields/resolver.hpp"
#include "soma/fields/space.hpp"

namespace soma {

template <class MemSpace>
class FieldRegistry {
 public:
  using memory_space = MemSpace;
  using GidView = View1<const GO, MemSpace>;

  FieldRegistry() = default;
  FieldRegistry(const FieldRegistry&) = delete;  // Views would alias silently
  FieldRegistry& operator=(const FieldRegistry&) = delete;
  // A View "move" leaves the source referencing the data, so move by swapping
  // with an empty registry; the moved-from registry is genuinely empty.
  FieldRegistry(FieldRegistry&& other) noexcept { swap(other); }
  FieldRegistry& operator=(FieldRegistry&& other) noexcept {
    FieldRegistry tmp(std::move(other));
    swap(tmp);
    return *this;
  }

  void swap(FieldRegistry& o) noexcept {
    entries_.swap(o.entries_);
    families_.swap(o.families_);
    std::swap(epoch_, o.epoch_);
    std::swap(index_builds_, o.index_builds_);
    std::swap(entities_, o.entities_);
  }

  std::uint64_t epoch() const noexcept { return epoch_; }
  // Public so that tests can drive it; Mesh::bump_epoch() is the only caller in
  // production code.
  void set_epoch(std::uint64_t e) noexcept {
    if (e == epoch_) return;
    epoch_ = e;
    for (Family& f : families_) ++f.version;
  }

  // Entity count and optional global ids of one dimension; used by add_tag
  // and as the identity behind full domains the registry builds itself.
  void set_entities(int dim, LO count, GidView gids = {}) {
    check_dim(dim);
    entities_[static_cast<std::size_t>(dim)] = Entities{count, gids, true};
  }
  bool has_entities(int dim) const noexcept {
    return dim >= 0 && dim < 4 && entities_[static_cast<std::size_t>(dim)].set;
  }

  template <class T>
  Field<T, MemSpace> add(std::string name, const Space<MemSpace>& space, int ncomp,
                         FieldOptions<T> options = {}) {
    if (ncomp < 1) throw std::invalid_argument("field '" + name + "': ncomp must be positive");
    if (space.domain().empty()) {
      throw std::invalid_argument("field '" + name + "': the domain has no blocks");
    }
    if (space.domain().epoch() != epoch_) throw StaleField(stale_message(name, space.domain()));
    check_values(name, options.default_value, ncomp, "default");
    check_values(name, options.global_value, ncomp, "global");

    const DataType type = data_type_of<T>();
    for (const Entry& e : entries_) {
      if (e.alive && e.info.name == name && e.info.type == type && e.info.ncomp == ncomp &&
          e.space == space) {
        throw FieldClash("field '" + name + "' (" + type_text(type, ncomp) + ") on " +
                         describe(space.layout()) + " with this domain already exists");
      }
    }

    check_disjoint(name, type, ncomp, space);
    check_member_limit(name, type, ncomp, space);

    View2<T, MemSpace> data(
        Kokkos::view_alloc("soma::field::" + name, Kokkos::WithoutInitializing),
        static_cast<std::size_t>(space.rows()), static_cast<std::size_t>(ncomp));
    fill(data, options.default_value);

    FieldInfo info;
    info.id = static_cast<FieldId>(entries_.size());
    info.family = family_for(name, type, ncomp, space.layout());
    families_[info.family].members.push_back(info.id);
    ++families_[info.family].version;
    info.name = std::move(name);
    info.type = type;
    info.ncomp = ncomp;
    info.layout = space.layout();
    info.rows = space.rows();
    for (int b = 0; b < space.domain().num_blocks(); ++b) info.dims.push_back(space.domain().info(b).dim);

    Field<T, MemSpace> f(info.id, space, data, options.default_value.value_or(std::vector<T>{}),
                         options.global_value.value_or(std::vector<T>{}));
    entries_.push_back(Entry{std::move(info), space, std::any(f), true});
    return f;
  }

  // A dense MOAB-style tag: one value per entity of `dim`, on a full domain.
  template <class T>
  Field<T, MemSpace> add_tag(std::string name, int dim, int ncomp, FieldOptions<T> options = {}) {
    check_dim(dim);
    const Entities& e = entities_[static_cast<std::size_t>(dim)];
    if (!e.set) {
      throw std::logic_error("add_tag: entity count of dimension " + std::to_string(dim) +
                             " is not set; call set_entities first");
    }
    const auto domain = Domain<MemSpace>::all(dim, e.count, epoch_, e.gids);
    return add<T>(std::move(name), Space<MemSpace>(domain, Layout::per_entity()), ncomp,
                  std::move(options));
  }

  // The field called `name`. Throws std::out_of_range when there is none and
  // std::invalid_argument when several fields share the name (pick one with
  // the overloads below) or when T is not the stored value type.
  template <class T>
  Field<T, MemSpace> get(std::string_view name) const {
    const Entry* found = nullptr;
    for (const Entry& e : entries_) {
      if (!e.alive || e.info.name != name) continue;
      if (found != nullptr) {
        throw std::invalid_argument("field name '" + std::string(name) +
                                    "' is ambiguous; select by dimension, space, or id");
      }
      found = &e;
    }
    if (found == nullptr) throw std::out_of_range("no field named '" + std::string(name) + "'");
    return fetch<T>(*found);
  }

  // The field called `name` that has members on dimension `dim`.
  template <class T>
  Field<T, MemSpace> get(std::string_view name, int dim) const {
    const Entry* found = nullptr;
    for (const Entry& e : entries_) {
      if (!e.alive || e.info.name != name) continue;
      if (std::find(e.info.dims.begin(), e.info.dims.end(), dim) == e.info.dims.end()) continue;
      if (found != nullptr) {
        throw std::invalid_argument("field name '" + std::string(name) + "' has several fields on dimension " +
                                    std::to_string(dim) + "; select by space or id");
      }
      found = &e;
    }
    if (found == nullptr) {
      throw std::out_of_range("no field '" + std::string(name) + "' on dimension " +
                              std::to_string(dim));
    }
    return fetch<T>(*found);
  }

  template <class T>
  Field<T, MemSpace> get(std::string_view name, const Space<MemSpace>& space) const {
    const Entry* e = find(name, data_type_of<T>(), space);
    if (e == nullptr) throw std::out_of_range("no field '" + std::string(name) + "' on this space");
    return fetch<T>(*e);
  }

  template <class T>
  Field<T, MemSpace> get(FieldId id) const {
    return fetch<T>(entry(id));
  }

  bool contains(std::string_view name) const noexcept {
    return std::any_of(entries_.begin(), entries_.end(),
                       [&](const Entry& e) { return e.alive && e.info.name == name; });
  }

  const FieldInfo& info(FieldId id) const { return entry(id).info; }

  std::vector<FieldInfo> list() const {
    std::vector<FieldInfo> out;
    for (const Entry& e : entries_) {
      if (e.alive) out.push_back(e.info);
    }
    return out;
  }

  // ---- families -----------------------------------------------------------

  // The family of fields called `name` with value type T, `ncomp` components,
  // and `layout`. Throws std::out_of_range when no field ever created one.
  template <class T>
  FamilyId family(std::string_view name, int ncomp, const Layout& layout = Layout::per_entity()) const {
    const FamilyId id = find_family(name, data_type_of<T>(), ncomp, layout);
    if (id == invalid_family) {
      throw std::out_of_range("no family '" + std::string(name) + "' (" +
                              type_text(data_type_of<T>(), ncomp) + ")");
    }
    return id;
  }

  // Fields of a family that have members on `dim`, in lookup slot order (the
  // slot that FamilyHit reports). Builds the index if needed.
  std::vector<FieldId> family_members(FamilyId fam, int dim) const {
    return dim_index(fam, dim).slots;
  }

  DimIndexKind index_kind(FamilyId fam, int dim) const { return dim_index(fam, dim).kind; }

  // Number of dimension indexes built so far (a cache probe for tests).
  std::uint64_t index_builds() const noexcept { return index_builds_; }

  // Handle on the family's current indexes, for kernels: handle.view() is the
  // device-copyable lookup. The handle keeps those indexes alive and is not
  // updated by later changes to the family (see FamilyHandle).
  FamilyHandle<MemSpace> family_handle(FamilyId fam) const {
    std::array<typename FamilyHandle<MemSpace>::Index, 4> index;
    for (std::size_t d = 0; d < 4; ++d) index[d] = dim_index_ptr(fam, static_cast<int>(d));
    return FamilyHandle<MemSpace>(std::move(index));
  }

  // Host lookup of one entity: member slot and first row.
  FamilyHit lookup(FamilyId fam, int dim, LO lid) const {
    return dim_index(fam, dim).host_view().lookup(lid);
  }

  // Host read of component `comp` at the first value point of (dim, lid): the
  // stored value, or the family default (the first member's default value, or
  // zero) when no member has the entity. A convenience path: every call
  // fetches the field through std::any, tens of nanoseconds against about one
  // for a raw View read. Bulk or kernel access should take a family_handle()
  // and read the field Views directly.
  template <class T>
  T value(FamilyId fam, int dim, LO lid, int comp = 0) const {
    static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible,
                  "value reads on the host; mirror the registry first");
    const auto& idx = dim_index(fam, dim);
    const FamilyHit h = idx.host_view().lookup(lid);
    if (h.found()) {
      const FieldId id = idx.slots[static_cast<std::size_t>(h.slot)];
      return get<T>(id).view()(h.row, comp);
    }
    for (const FieldId id : families_[fam].members) {
      const auto f = get<T>(id);
      if (f.has_default()) return f.default_value()[static_cast<std::size_t>(comp)];
    }
    return T{};
  }

  // Values of component `comp` for a list of entities (dim, lid), host side.
  template <class T>
  void gather(FamilyId fam, std::span<const std::pair<int, LO>> entities, int comp,
              std::span<T> out) const {
    if (out.size() != entities.size()) {
      throw std::invalid_argument("gather: output has " + std::to_string(out.size()) +
                                  " slots for " + std::to_string(entities.size()) + " entities");
    }
    for (std::size_t i = 0; i < entities.size(); ++i) {
      out[i] = value<T>(fam, entities[i].first, entities[i].second, comp);
    }
  }

  // Frees the storage. The id stays retired.
  void remove(FieldId id) {
    Entry& e = entry_mut(id);
    Family& fam = families_[e.info.family];
    std::erase(fam.members, id);
    ++fam.version;
    e.alive = false;
    e.field.reset();
    e.space = Space<MemSpace>();
  }

  // Copies the selected fields into the empty registry `dst`, keeping their
  // ids. The entity counts and global ids (set_entities) belong to whoever owns
  // those arrays, so the caller sets them on `dst`. names: {"*"} = all. A field is skipped when any dimension it has
  // members on is switched off in `dims`. Domains and spaces are mirrored once
  // and shared by every field that shares them. alias = true reuses storage
  // when Dst can access it (create_mirror_view_and_copy); alias = false always
  // allocates (create_mirror + deep_copy).
  template <class Dst>
  void mirror_into(FieldRegistry<Dst>& dst, const std::vector<std::string>& names,
                   const std::array<bool, 4>& dims, bool alias) {
    if (!dst.entries_.empty()) {
      throw std::logic_error("mirror_into: the destination registry must be empty");
    }
    const bool all = std::find(names.begin(), names.end(), "*") != names.end();
    if (!all) {
      for (const std::string& n : names) {
        if (!contains(n)) throw std::out_of_range("mirror: no field named '" + n + "'");
      }
    }
    dst.epoch_ = epoch_;
    for (const Family& f : families_) dst.families_.push_back(typename FieldRegistry<Dst>::Family{f.key});

    std::map<const void*, Domain<Dst>> domains;
    std::map<const void*, Space<Dst>> spaces;
    for (Entry& e : entries_) {
      typename FieldRegistry<Dst>::Entry out;
      out.info = e.info;
      const bool wanted = all || std::find(names.begin(), names.end(), e.info.name) != names.end();
      const bool dims_on = std::all_of(e.info.dims.begin(), e.info.dims.end(), [&](int d) {
        return dims[static_cast<std::size_t>(d)];
      });
      if (e.alive && wanted && dims_on) {
        auto sp = spaces.find(e.space.storage_id());
        if (sp == spaces.end()) {
          auto dm = domains.find(e.space.domain().storage_id());
          if (dm == domains.end()) {
            dm = domains.emplace(e.space.domain().storage_id(), e.space.domain().template mirror<Dst>())
                     .first;
          }
          sp = spaces.emplace(e.space.storage_id(), e.space.template mirror<Dst>(dm->second)).first;
        }
        out.space = sp->second;
        dispatch_data_type(e.info.type, [&]<class T>() {
          const auto src = std::any_cast<const Field<T, MemSpace>&>(e.field);
          View2<T, Dst> view;
          if (alias) {
            view = Kokkos::create_mirror_view_and_copy(Dst{}, src.view());
          } else {
            view = Kokkos::create_mirror(Kokkos::view_alloc(Dst{}, Kokkos::WithoutInitializing),
                                         src.view());
            Kokkos::deep_copy(view, src.view());
          }
          out.field = Field<T, Dst>(src.id(), out.space, view, src.default_value(),
                                    src.global_value());
        });
        out.alive = true;
        dst.families_[out.info.family].members.push_back(out.info.id);
      }
      dst.entries_.push_back(std::move(out));
    }
  }

 private:
  template <class>
  friend class FieldRegistry;

  struct Entities {
    LO count = 0;
    GidView gids{};
    bool set = false;
  };

  struct Entry {
    FieldInfo info;
    Space<MemSpace> space;
    std::any field;  // Field<T, MemSpace>
    bool alive = false;
  };

  struct FamilyKey {
    std::string name;
    DataType type = DataType::f64;
    int ncomp = 1;
    Layout layout{};
  };

  struct Family {
    explicit Family(FamilyKey k) : key(std::move(k)) {}
    FamilyKey key;
    std::vector<FieldId> members;  // live fields, in id order
    std::uint64_t version = 1;     // bumped when the membership or the epoch changes
    mutable std::array<std::shared_ptr<const detail::FamilyDimData<MemSpace>>, 4> index;
    mutable std::array<std::uint64_t, 4> built{};
  };

  static void check_dim(int dim) {
    if (dim < 0 || dim > 3) {
      throw std::invalid_argument("dimension " + std::to_string(dim) + " is not in [0, 3]");
    }
  }

  static std::string type_text(DataType t, int ncomp) {
    return std::string(name(t)) + "x" + std::to_string(ncomp);
  }

  std::string stale_message(const std::string& name, const Domain<MemSpace>& d) const {
    return "field '" + name + "' was defined at mesh epoch " + std::to_string(d.epoch()) +
           " but the mesh is at epoch " + std::to_string(epoch_) + "; rebuild its domain";
  }

  template <class T>
  static void check_values(const std::string& name, const std::optional<std::vector<T>>& v,
                           int ncomp, const char* what) {
    if (v && static_cast<int>(v->size()) != ncomp) {
      throw std::invalid_argument("field '" + name + "': " + what + " value has " +
                                  std::to_string(v->size()) + " entries for " +
                                  std::to_string(ncomp) + " components");
    }
  }

  template <class T>
  static void fill(const View2<T, MemSpace>& data, const std::optional<std::vector<T>>& def) {
    if (data.extent(0) == 0) return;
    if (!def) {
      Kokkos::deep_copy(data, T{});
      return;
    }
    for (std::size_t c = 0; c < data.extent(1); ++c) {
      Kokkos::deep_copy(Kokkos::subview(data, Kokkos::ALL, c), (*def)[c]);
    }
  }

  FamilyId find_family(std::string_view name, DataType type, int ncomp,
                       const Layout& layout) const noexcept {
    for (std::size_t i = 0; i < families_.size(); ++i) {
      const FamilyKey& k = families_[i].key;
      if (k.name == name && k.type == type && k.ncomp == ncomp && k.layout == layout) {
        return static_cast<FamilyId>(i);
      }
    }
    return invalid_family;
  }

  FamilyId family_for(const std::string& name, DataType type, int ncomp, const Layout& layout) {
    const FamilyId found = find_family(name, type, ncomp, layout);
    if (found != invalid_family) return found;
    families_.push_back(Family{FamilyKey{name, type, ncomp, layout}});
    return static_cast<FamilyId>(families_.size() - 1);
  }

  std::string describe_members(const Space<MemSpace>& sp, int b) const {
    const DomainBlockInfo& bi = sp.domain().info(b);
    return std::string("{") + name(bi.kind) + " of dimension " + std::to_string(bi.dim) + ", " +
           std::to_string(bi.size) + " of " + std::to_string(bi.count) + " entities, " +
           describe(sp.layout()) + "}";
  }

  // Throws FieldOverlap when `space` shares an entity with a live member of the
  // family it would join. Nothing is modified.
  void check_disjoint(const std::string& name, DataType type, int ncomp,
                      const Space<MemSpace>& space, FieldId skip = invalid_field) const {
    const FamilyId fid = find_family(name, type, ncomp, space.layout());
    if (fid == invalid_family) return;
    for (const FieldId other : families_[fid].members) {
      if (other == skip) continue;
      const Entry& e = entries_[other];
      for (int b = 0; b < space.domain().num_blocks(); ++b) {
        const int dim = space.domain().info(b).dim;
        const int ob = e.space.domain().find_block(dim);
        if (ob < 0) continue;
        const LO lid = detail::first_overlap(space.domain().host_block(b),
                                             e.space.domain().host_block(ob));
        if (lid == invalid_lo) continue;
        throw FieldOverlap("field '" + name + "' (" + type_text(type, ncomp) + ") on " +
                           describe_members(space, b) + " intersects existing '" + e.info.name +
                           "' (" + type_text(e.info.type, e.info.ncomp) + ") on " +
                           describe_members(e.space, ob) + " in dim " + std::to_string(dim) +
                           " at entity " + entity_text(dim, lid));
      }
    }
  }

  // Throws std::length_error when `space` would give its family more members on
  // one dimension than the resolver can name (max_family_members). Nothing is
  // modified.
  void check_member_limit(const std::string& name, DataType type, int ncomp,
                          const Space<MemSpace>& space) const {
    const FamilyId fid = find_family(name, type, ncomp, space.layout());
    if (fid == invalid_family) return;
    for (int b = 0; b < space.domain().num_blocks(); ++b) {
      const int dim = space.domain().info(b).dim;
      int count = 1;
      for (const FieldId other : families_[fid].members) {
        if (entries_[other].space.domain().find_block(dim) >= 0) ++count;
      }
      if (count > max_family_members) {
        throw std::length_error("field '" + name + "' (" + type_text(type, ncomp) +
                                "): a family holds at most " + std::to_string(max_family_members) +
                                " members on dimension " + std::to_string(dim));
      }
    }
  }

  std::string entity_text(int dim, LO lid) const {
    const Entities& e = entities_[static_cast<std::size_t>(dim)];
    if (e.set && e.gids.extent(0) != 0) {
      const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, e.gids);
      return "gid " + std::to_string(h(static_cast<std::size_t>(lid)));
    }
    return "lid " + std::to_string(lid);
  }

  using IndexPtr = std::shared_ptr<const detail::FamilyDimData<MemSpace>>;

  const detail::FamilyDimData<MemSpace>& dim_index(FamilyId fid, int dim) const {
    return *dim_index_ptr(fid, dim);
  }

  // The index is replaced, never modified, so handles that still hold the old
  // one keep a consistent snapshot.
  const IndexPtr& dim_index_ptr(FamilyId fid, int dim) const {
    check_dim(dim);
    if (fid >= families_.size()) throw std::out_of_range("no family with id " + std::to_string(fid));
    const Family& fam = families_[fid];
    const auto d = static_cast<std::size_t>(dim);
    if (!fam.index[d] || fam.built[d] != fam.version) {
      std::vector<detail::MemberBlock> blocks;
      std::vector<FieldId> slots;
      for (const FieldId id : fam.members) {
        const Entry& e = entries_[id];
        if (e.space.domain().epoch() != epoch_) {
          throw StaleField(stale_message(e.info.name, e.space.domain()));
        }
        const int b = e.space.domain().find_block(dim);
        if (b < 0) continue;
        blocks.push_back(detail::MemberBlock{e.space.domain().host_block(b),
                                             e.space.block_row_begin(b), e.space.block_stride(b),
                                             e.space.block_host_offsets(b)});
        slots.push_back(id);
      }
      auto built = std::make_shared<detail::FamilyDimData<MemSpace>>(
          detail::build_dim_index<MemSpace>(blocks));
      built->slots = std::move(slots);
      fam.index[d] = std::move(built);
      fam.built[d] = fam.version;
      ++index_builds_;
    }
    return fam.index[d];
  }

  const Entry& entry(FieldId id) const {
    if (id >= entries_.size() || !entries_[id].alive) {
      throw std::out_of_range("no field with id " + std::to_string(id));
    }
    return entries_[id];
  }
  Entry& entry_mut(FieldId id) {
    return const_cast<Entry&>(std::as_const(*this).entry(id));
  }

  const Entry* find(std::string_view name, DataType type, const Space<MemSpace>& space) const {
    for (const Entry& e : entries_) {
      if (e.alive && e.info.name == name && e.info.type == type && e.space == space) return &e;
    }
    return nullptr;
  }

  template <class T>
  Field<T, MemSpace> fetch(const Entry& e) const {
    if (e.info.type != data_type_of<T>()) {
      throw std::invalid_argument("field '" + e.info.name + "' requested as the wrong value type (stored " +
                                  name(e.info.type) + ")");
    }
    if (e.space.domain().epoch() != epoch_) throw StaleField(stale_message(e.info.name, e.space.domain()));
    return std::any_cast<Field<T, MemSpace>>(e.field);
  }

  std::vector<Entry> entries_;
  std::vector<Family> families_;
  mutable std::uint64_t index_builds_ = 0;
  std::uint64_t epoch_ = 1;
  std::array<Entities, 4> entities_{};
};

}  // namespace soma

#endif  // SOMA_FIELDS_REGISTRY_HPP_
