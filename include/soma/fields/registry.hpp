// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_REGISTRY_HPP_
#define SOMA_FIELDS_REGISTRY_HPP_

// Host-side owner of fields. Type erasure exists only here; kernels receive
// concretely typed Views. A field is identified by (name, value type, space);
// a second definition with all three equal is a FieldClash. FieldIds are
// indices that are never reused, so a removed field leaves a gap.
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
    std::swap(entities_, o.entities_);
  }

  std::uint64_t epoch() const noexcept { return epoch_; }
  // Public so that tests can drive it; Mesh::bump_epoch() is the only caller in
  // production code.
  void set_epoch(std::uint64_t e) noexcept { epoch_ = e; }

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

    View2<T, MemSpace> data(
        Kokkos::view_alloc("soma::field::" + name, Kokkos::WithoutInitializing),
        static_cast<std::size_t>(space.rows()), static_cast<std::size_t>(ncomp));
    fill(data, options.default_value);

    FieldInfo info;
    info.id = static_cast<FieldId>(entries_.size());
    info.family = family_for(name, type, ncomp);
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

  // Frees the storage. The id stays retired.
  void remove(FieldId id) {
    Entry& e = entry_mut(id);
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
    dst.families_ = families_;

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
    DataType type;
    int ncomp;
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

  FamilyId family_for(const std::string& name, DataType type, int ncomp) {
    for (std::size_t i = 0; i < families_.size(); ++i) {
      const FamilyKey& k = families_[i];
      if (k.name == name && k.type == type && k.ncomp == ncomp) return static_cast<FamilyId>(i);
    }
    families_.push_back(FamilyKey{name, type, ncomp});
    return static_cast<FamilyId>(families_.size() - 1);
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
  std::vector<FamilyKey> families_;
  std::uint64_t epoch_ = 1;
  std::array<Entities, 4> entities_{};
};

}  // namespace soma

#endif  // SOMA_FIELDS_REGISTRY_HPP_
