// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_REGISTRY_HPP_
#define SOMA_FIELDS_REGISTRY_HPP_

// Host-side owner of fields. Type erasure exists only here; kernels receive
// concretely typed Views. A field is defined by (name, value type, component
// count, space); a second definition with all four equal is a FieldClash.
// FieldIds are indices that are never reused, so a removed field leaves a gap.
//
// A family is the set of fields that share a name, a value type, a component
// count, and a layout (the family key; the domain is not part of it). The
// layout is in the key so that a per_entity field and a node field of one name
// can coexist on the same entities, and the component count so that a
// 1-component and a 3-component field of one name stay apart.
// Members of a family must be disjoint: add() throws FieldOverlap otherwise
// (the check is exact and only compares members that share a dimension). A
// family resolves an entity to (member, row) through a per-dimension index that
// is built on first use and rebuilt after the epoch or the membership changes;
// see resolver.hpp. Fields of one family on the same dimension may differ in
// domain only, so a family lookup returns the first value point of the entity.
//
// The registry holds the current mesh epoch. Mesh (or a test) sets it with
// set_epoch(), and fetching a field whose domain was built against another
// epoch throws StaleField. Kernels get Views and are not checked on the device.
// The registry needs only entity counts (and optionally global ids) per
// dimension, supplied through set_entities(); it never includes the mesh.
//
// Thread safety: none for mutation, and const members are not read-only. Lazy
// family-index builds (family_handle(), lookup(), value(), family_members())
// and the commit that mirror_into performs write to mutable members, so two
// threads may not call into one registry unless it is only read after every
// index has been built: take family_handle() for each family (or do one
// lookup per family and dimension) before spawning threads, make no write,
// add, remove, commit, mirror_into or set_epoch call meanwhile, and let the
// threads use the handles' views or the Field Views rather than the registry.

#include <algorithm>
#include <any>
#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
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

namespace detail {

template <class T, class MemSpace>
struct RowScatter {
  View2<T, MemSpace> data;
  View1<const T, MemSpace> values;
  std::size_t row;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t c) const { data(row, c) = values(c); }
};

// Writing one row of a field. Plain HostSpace memory is assigned directly: a
// per-component deep_copy fences the execution space every time, which costs
// tens of microseconds with threads. Another host-accessible space (managed or
// unified memory) may have a kernel in flight on the device, so it gets one
// fence and then the same direct writes. Anything else is staged.
enum class RowWritePath { direct, fenced_direct, staged };

template <class MemSpace>
constexpr RowWritePath write_row_path() {
  if constexpr (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
    return RowWritePath::direct;
  } else if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible) {
    return RowWritePath::fenced_direct;
  } else {
    return RowWritePath::staged;
  }
}

template <class T, class MemSpace>
void write_row_direct(const View2<T, MemSpace>& data, std::size_t row, std::span<const T> values) {
  for (std::size_t c = 0; c < values.size(); ++c) data(row, c) = values[c];
}

// Stages the values in one host View, moves them with one deep_copy, and
// scatters them into the row with one kernel.
template <class T, class MemSpace>
void write_row_staged(const View2<T, MemSpace>& data, std::size_t row, std::span<const T> values) {
  View1<T, Kokkos::HostSpace> host(
      Kokkos::view_alloc("soma::field::row", Kokkos::WithoutInitializing), values.size());
  for (std::size_t c = 0; c < values.size(); ++c) host(c) = values[c];
  const View1<T, MemSpace> staged = Kokkos::create_mirror_view_and_copy(MemSpace{}, host);
  using Policy = Kokkos::RangePolicy<typename MemSpace::execution_space, std::size_t>;
  Kokkos::parallel_for("soma::field::write_row", Policy(0, values.size()),
                       RowScatter<T, MemSpace>{data, staged, row});
  Kokkos::fence("soma::field::write_row");
}

template <RowWritePath path, class T, class MemSpace>
void write_row_via(const View2<T, MemSpace>& data, std::size_t row, std::span<const T> values) {
  if constexpr (path == RowWritePath::direct) {
    write_row_direct<T, MemSpace>(data, row, values);
  } else if constexpr (path == RowWritePath::fenced_direct) {
    Kokkos::fence("soma::field::write_row");
    write_row_direct<T, MemSpace>(data, row, values);
  } else {
    write_row_staged<T, MemSpace>(data, row, values);
  }
}

template <class T, class MemSpace>
void write_row(const View2<T, MemSpace>& data, std::size_t row, std::span<const T> values) {
  write_row_via<write_row_path<MemSpace>(), T, MemSpace>(data, row, values);
}

}  // namespace detail

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
    std::swap(serial_, o.serial_);
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
    check_pending(name, type, ncomp, space);
    check_member_limit(name, type, ncomp, space);

    View2<T, MemSpace> data(
        Kokkos::view_alloc("soma::field::" + name, Kokkos::WithoutInitializing),
        static_cast<std::size_t>(space.rows()), static_cast<std::size_t>(ncomp));
    fill(data, options.default_value);

    const FamilyId existing = find_family(name, type, ncomp, space.layout());
    FieldInfo info;
    info.id = static_cast<FieldId>(entries_.size());
    info.family = existing != invalid_family ? existing : static_cast<FamilyId>(families_.size());
    info.name = name;
    info.type = type;
    info.ncomp = ncomp;
    info.layout = space.layout();
    info.rows = space.rows();
    for (int b = 0; b < space.domain().num_blocks(); ++b) {
      info.dims.push_back(space.domain().info(b).dim);
    }
    std::optional<Family> created;
    if (existing == invalid_family) {
      created.emplace(FamilyKey{name, type, ncomp, space.layout()});
      created->members.reserve(1);
    }

    Field<T, MemSpace> f(info.id, space, data, options.default_value.value_or(std::vector<T>{}),
                         options.global_value.value_or(std::vector<T>{}), 0, serial_);
    Entry entry{std::move(info), space, std::any(f), true};

    // Everything that can throw has happened by now. Make room for the family
    // first; entries_.push_back is then the only step left that can throw, and
    // it does so before changing anything because Entry moves without throwing.
    // The pushes after it cannot fail.
    static_assert(std::is_nothrow_move_constructible_v<Entry>);
    if (created) {
      grow_for_one(families_);
    } else {
      grow_for_one(families_[existing].members);
    }
    entries_.push_back(std::move(entry));
    if (created) families_.push_back(std::move(*created));
    Family& fam = families_[entries_.back().info.family];
    fam.members.push_back(f.id());
    ++fam.version;
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

  // True when `f` still refers to the storage this registry holds for its id.
  // A commit that merges pending writes replaces the storage, so a handle kept
  // across it is stale: it stays readable, but writes through it go to storage
  // the registry no longer uses. Fetch the field again with get(). False for a
  // removed field, an unknown id, a default-constructed field, and a handle
  // from another registry (a mirror counts as another registry, aliasing or
  // not). The mesh epoch is not judged here (get() does that).
  template <class T>
  bool is_current(const Field<T, MemSpace>& f) const noexcept {
    if (f.id() >= entries_.size()) return false;
    const Entry& e = entries_[f.id()];
    return e.alive && f.owner() == serial_ && f.revision() == e.info.revision;
  }

  // Revision of the field's current storage (the value Field::revision() has
  // after a fresh get()); empty for a removed or unknown id. A handle whose
  // revision() differs is stale.
  std::optional<std::uint64_t> current_revision(FieldId id) const noexcept {
    if (id >= entries_.size() || !entries_[id].alive) return std::nullopt;
    return entries_[id].info.revision;
  }

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

  // ---- sparse growth: pending writes and commit --------------------------

  // Host write of one entity's values (ncomp of them) into a per_entity field
  // with one block. A member entity is written straight into the data, in the
  // storage that `id` names now: a Field handle taken before a commit points at
  // replaced storage, so check is_current() on handles kept across commits.
  // A non-member of a subset field is recorded in a sorted host buffer and
  // joins the domain at the next commit, which every mirror of the field
  // performs first. A field over a full domain has no non-members.
  // Throws FieldOverlap when the entity is a member of another field of the
  // same family, or has a pending write in one: such a write could never be
  // committed. A throw buffers nothing.
  template <class T>
  void set_pending(FieldId id, LO lid, std::span<const T> values) {
    Entry& e = entry_mut(id);
    if (e.info.type != data_type_of<T>()) {
      throw std::invalid_argument("field '" + e.info.name + "' written as the wrong value type (stored " +
                                  name(e.info.type) + ")");
    }
    require_growable(e);
    require_current(e);
    if (static_cast<int>(values.size()) != e.info.ncomp) {
      throw std::invalid_argument("field '" + e.info.name + "': " + std::to_string(values.size()) +
                                  " values for " + std::to_string(e.info.ncomp) + " components");
    }
    const DomainBlockInfo& bi = e.space.domain().info(0);
    if (lid < 0 || lid >= bi.count) {
      throw std::out_of_range("field '" + e.info.name + "': entity " + std::to_string(lid) +
                              " is outside [0, " + std::to_string(bi.count) + ")");
    }
    const auto field = std::any_cast<const Field<T, MemSpace>&>(e.field);
    const LO row = e.space.domain().host_block(0).row(lid);
    if (row != invalid_lo) {
      detail::write_row<T, MemSpace>(field.view(), static_cast<std::size_t>(row), values);
      return;
    }
    using Buffer = std::map<LO, std::vector<T>>;
    const Buffer* held = e.pending.has_value() ? std::any_cast<Buffer>(&e.pending) : nullptr;
    if (held == nullptr || held->find(lid) == held->end()) check_free(e, lid);
    // Everything that allocates happens before the buffer is touched, so a
    // throw leaves the old buffer and its count as they were.
    std::vector<T> value(values.begin(), values.end());
    if (held == nullptr) {
      Buffer fresh;
      fresh.emplace(lid, std::move(value));
      e.pending = std::move(fresh);
    } else {
      auto& buf = std::any_cast<Buffer&>(e.pending);
      const auto it = buf.lower_bound(lid);
      if (it != buf.end() && it->first == lid) {
        it->second = std::move(value);
      } else {
        buf.emplace_hint(it, lid, std::move(value));
      }
    }
    e.pending_count = std::any_cast<Buffer&>(e.pending).size();
  }

  // Drops the buffered writes of one field. Data already written to member
  // rows is not affected.
  void discard_pending(FieldId id) {
    Entry& e = entry_mut(id);
    e.pending.reset();
    e.pending_count = 0;
  }

  // Entities waiting to be merged into the field's domain.
  std::size_t pending_count(FieldId id) const { return entry(id).pending_count; }

  // Merges the pending writes of one field into a new domain and data. The
  // field keeps its id and its mesh epoch; its revision advances. Field handles
  // taken earlier then refer to the old storage and report is_current() ==
  // false; fetch the field again with get(). Idempotent and cheap when nothing
  // is pending. If the grown domain would overlap another member of the family,
  // FieldOverlap is thrown and the field keeps its previous domain, data, and
  // pending buffer.
  void commit(FieldId id) { commit_entry(entry_mut(id)); }

  // Throws StaleField if any live field is stale, like commit on that field.
  void commit_all() { commit_everything(); }

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
  // those arrays, so the caller sets them on `dst`. names: {"*"} = all. A field
  // is skipped when any dimension it has members on is switched off in `dims`.
  // Domains and spaces are mirrored once and shared by every field that shares
  // them. Pending writes of the selected fields are committed first, so the
  // mirror and the source agree; fields that are not selected are left alone,
  // stale or with pending writes. A selected stale field throws StaleField.
  // The selected fields are committed one after the other before anything is
  // copied, and a commit is not undone: if a later selected field makes the
  // call throw (for example by running out of memory), the earlier selected
  // fields stay committed on the source, and `dst` is unchanged and can be used
  // again. A stale selected field is found first and throws before any commit.
  // Note that this const member commits on the source: the registry keeps its
  // storage mutable for that reason. alias = false always allocates and copies
  // (create_mirror + deep_copy). alias = true reuses storage when Dst can access
  // it (create_mirror_view_and_copy): from host to host (or any pair of
  // spaces with the same memory) the mirror then shares the Views of the
  // source, so writes through either are seen by both, and the destination
  // does not follow later commits on the source (a commit gives the source new
  // storage; mirror again to see it). A destination that cannot access the
  // source always gets its own copy.
  template <class Dst>
  void mirror_into(FieldRegistry<Dst>& dst, const std::vector<std::string>& names,
                   const std::array<bool, 4>& dims, bool alias) const {
    if (!dst.entries_.empty()) {
      throw std::logic_error("mirror_into: the destination registry must be empty");
    }
    const bool all = std::find(names.begin(), names.end(), "*") != names.end();
    if (!all) {
      for (const std::string& n : names) {
        if (!contains(n)) throw std::out_of_range("mirror: no field named '" + n + "'");
      }
    }
    for (const Entry& e : entries_) {
      if (e.alive && selected(e, all, names, dims)) require_current(e);
    }
    for (Entry& e : entries_) {
      if (e.alive && selected(e, all, names, dims)) commit_entry(e);
    }
    // Built aside and swapped in, so a throw leaves `dst` empty and reusable.
    FieldRegistry<Dst> built;
    built.epoch_ = epoch_;
    for (const Family& f : families_) {
      built.families_.push_back(typename FieldRegistry<Dst>::Family{f.key});
    }

    std::map<const void*, Domain<Dst>> domains;
    std::map<const void*, Space<Dst>> spaces;
    for (const Entry& e : entries_) {
      typename FieldRegistry<Dst>::Entry out;
      out.info = e.info;
      if (e.alive && selected(e, all, names, dims)) {
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
                                    src.global_value(), src.revision(), dst.serial_);
        });
        out.alive = true;
        built.families_[out.info.family].members.push_back(out.info.id);
      }
      built.entries_.push_back(std::move(out));
    }
    dst.entries_.swap(built.entries_);
    dst.families_.swap(built.families_);
    dst.epoch_ = built.epoch_;
  }

 private:
  template <class>
  friend class FieldRegistry;

  static std::uint64_t next_serial() noexcept {
    static std::atomic<std::uint64_t> counter{0};
    return ++counter;
  }

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
    std::any pending{};  // std::map<LO, std::vector<T>>, sorted by entity
    std::size_t pending_count = 0;
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

  // Reserves room for one more element when the vector is full, so that the
  // push_back that follows cannot throw. Doubles, to keep growth amortized.
  template <class V>
  static void grow_for_one(V& v) {
    if (v.size() == v.capacity()) v.reserve(v.empty() ? 4 : 2 * v.size());
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

  // Throws FieldOverlap when `space` contains an entity that a live member of the
  // family it would join has a pending write for: that write could then never
  // be committed. Cost: one inverse-map lookup per pending entity. Nothing is
  // modified.
  void check_pending(const std::string& name, DataType type, int ncomp,
                     const Space<MemSpace>& space) const {
    const FamilyId fid = find_family(name, type, ncomp, space.layout());
    if (fid == invalid_family) return;
    for (const FieldId other : families_[fid].members) {
      const Entry& e = entries_[other];
      if (e.pending_count == 0) continue;
      const int dim = e.space.domain().info(0).dim;
      const int b = space.domain().find_block(dim);
      if (b < 0) continue;
      const auto block = space.domain().host_block(b);
      dispatch_data_type(e.info.type, [&]<class T>() {
        using Buffer = std::map<LO, std::vector<T>>;
        for (const auto& [lid, values] : std::any_cast<const Buffer&>(e.pending)) {
          if (block.row(lid) == invalid_lo) continue;
          throw FieldOverlap("field '" + name + "' (" + type_text(type, ncomp) + ") on " +
                             describe_members(space, b) + " contains entity " +
                             entity_text(dim, lid) + " in dim " + std::to_string(dim) +
                             ", which has a pending write of existing '" + e.info.name + "' (" +
                             type_text(e.info.type, e.info.ncomp) + ")");
        }
      });
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

  // Throws FieldOverlap when another live member of e's family already holds
  // entity `lid`, as a member row (one lookup in the family index of that
  // dimension) or as a pending write (one map lookup per sibling).
  void check_free(const Entry& e, LO lid) const {
    const int dim = e.space.domain().info(0).dim;
    for (const FieldId other : families_[e.info.family].members) {
      if (other == e.info.id) continue;
      const Entry& o = entries_[other];
      if (o.pending_count == 0 || o.space.domain().info(0).dim != dim) continue;
      dispatch_data_type(o.info.type, [&]<class T>() {
        const auto& buf = std::any_cast<const std::map<LO, std::vector<T>>&>(o.pending);
        if (buf.find(lid) == buf.end()) return;
        throw FieldOverlap("field '" + e.info.name + "' (" + type_text(e.info.type, e.info.ncomp) +
                           ") cannot take entity " + entity_text(dim, lid) + " in dim " +
                           std::to_string(dim) + ": existing '" + o.info.name + "' (" +
                           type_text(o.info.type, o.info.ncomp) +
                           ") has a pending write for it");
      });
    }
    const detail::FamilyDimData<MemSpace>& idx = dim_index(e.info.family, dim);
    const FamilyHit hit = idx.host_view().lookup(lid);
    if (!hit.found()) return;
    const Entry& other = entries_[idx.slots[static_cast<std::size_t>(hit.slot)]];
    throw FieldOverlap("field '" + e.info.name + "' (" + type_text(e.info.type, e.info.ncomp) +
                       ") cannot take entity " + entity_text(dim, lid) + " in dim " +
                       std::to_string(dim) + ": it belongs to existing '" + other.info.name + "' (" +
                       type_text(other.info.type, other.info.ncomp) + ") on " +
                       describe_members(other.space, 0));
  }

  static bool selected(const Entry& e, bool all, const std::vector<std::string>& names,
                       const std::array<bool, 4>& dims) {
    const bool wanted = all || std::find(names.begin(), names.end(), e.info.name) != names.end();
    const bool dims_on = std::all_of(e.info.dims.begin(), e.info.dims.end(), [&](int d) {
      return dims[static_cast<std::size_t>(d)];
    });
    return wanted && dims_on;
  }

  void require_current(const Entry& e) const {
    if (e.space.domain().epoch() != epoch_) {
      throw StaleField(stale_message(e.info.name, e.space.domain()));
    }
  }

  static void require_growable(const Entry& e) {
    if (e.info.layout.kind != Layout::Kind::per_entity || e.space.domain().num_blocks() != 1) {
      throw std::invalid_argument("field '" + e.info.name +
                                  "': pending writes need a per_entity field over one dimension");
    }
  }

  void commit_everything() const {
    for (Entry& e : entries_) {
      if (e.alive) commit_entry(e);
    }
  }

  void commit_entry(Entry& e) const {
    require_current(e);
    if (e.pending_count == 0) return;
    dispatch_data_type(e.info.type, [&]<class T>() { commit_typed<T>(e); });
  }

  template <class T>
  void commit_typed(Entry& e) const {
    using Buffer = std::map<LO, std::vector<T>>;
    const auto& buf = std::any_cast<const Buffer&>(e.pending);
    const auto old_field = std::any_cast<const Field<T, MemSpace>&>(e.field);
    const Domain<MemSpace>& old_domain = e.space.domain();
    const DomainBlockInfo& bi = old_domain.info(0);
    const int ncomp = e.info.ncomp;

    // Merged member list, sorted by entity; remember where each old row goes.
    const std::vector<LO> old_members = old_domain.members(0);
    std::vector<LO> lids;
    lids.reserve(old_members.size() + buf.size());
    std::vector<std::size_t> from_old(old_members.size() + buf.size(), old_members.size());
    {
      auto o = old_members.begin();
      auto p = buf.begin();
      while (o != old_members.end() || p != buf.end()) {
        if (p == buf.end() || (o != old_members.end() && *o < p->first)) {
          from_old[lids.size()] = static_cast<std::size_t>(o - old_members.begin());
          lids.push_back(*o++);
        } else {
          lids.push_back((p++)->first);
        }
      }
    }
    GidView gids;
    const Entities& ent = entities_[static_cast<std::size_t>(bi.dim)];
    if (ent.set) gids = ent.gids;
    const auto domain = Domain<MemSpace>::from_list(bi.dim, bi.count, old_domain.epoch(), lids, gids);
    const Space<MemSpace> space(domain, Layout::per_entity());
    // The checks in add() and set_pending() keep members and buffers disjoint, so
    // this cannot fire through the public API today. It stays a thrown
    // FieldOverlap rather than an assert so that a broken family invariant
    // cannot corrupt a release build silently.
    check_disjoint(e.info.name, e.info.type, ncomp, space, e.info.id);

    const auto old_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, old_field.view());
    View2<T, Kokkos::HostSpace> merged(
        Kokkos::view_alloc("soma::field::" + e.info.name, Kokkos::WithoutInitializing),
        lids.size(), static_cast<std::size_t>(ncomp));
    for (std::size_t r = 0; r < lids.size(); ++r) {
      const auto it = buf.find(lids[r]);
      for (std::size_t c = 0; c < static_cast<std::size_t>(ncomp); ++c) {
        merged(r, c) = it != buf.end() ? it->second[c] : old_host(from_old[r], c);
      }
    }
    const View2<T, MemSpace> data = Kokkos::create_mirror_view_and_copy(MemSpace{}, merged);

    // Everything that allocates happens before the first change to the entry,
    // and the changes below cannot throw, so a failure leaves the entry as it
    // was.
    Field<T, MemSpace> grown(e.info.id, space, data, old_field.default_value(),
                             old_field.global_value(), e.info.revision + 1, serial_);
    std::any grown_any(grown);
    Family& family = families_[e.info.family];
    e.field.swap(grown_any);
    e.space = space;
    e.info.rows = space.rows();
    ++e.info.revision;
    e.pending.reset();
    e.pending_count = 0;
    ++family.version;
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

  // Mutable because commit is logically const: it replaces storage by storage
  // with the same observable content, and mirrors perform it on a const source.
  mutable std::vector<Entry> entries_;
  mutable std::vector<Family> families_;
  mutable std::uint64_t index_builds_ = 0;
  std::uint64_t epoch_ = 1;
  // Identifies this registry in the handles it hands out; moves carry it along
  // with the storage. Never 0, which marks a handle that no registry made.
  std::uint64_t serial_ = next_serial();
  std::array<Entities, 4> entities_{};
};

}  // namespace soma

#endif  // SOMA_FIELDS_REGISTRY_HPP_
