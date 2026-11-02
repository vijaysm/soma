// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_FIELD_HPP_
#define SOMA_FIELDS_FIELD_HPP_

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/fields/space.hpp"

namespace soma {

using FieldId = std::uint32_t;
using FamilyId = std::uint32_t;
inline constexpr FieldId invalid_field = std::numeric_limits<FieldId>::max();
inline constexpr FamilyId invalid_family = std::numeric_limits<FamilyId>::max();

// The value of an entity that was never set. default_value (ncomp values) is
// also the initial fill; without one, fields start zeroed and has_default() is
// false. global_value is the one value per mesh that MOAB tags can carry. For
// a variable-length field either may have any length. A handle field holds
// gids of entities or sets (0 for none) and needs T = GO; it has a family of
// its own, apart from plain GO fields of the same name.
template <class T>
struct FieldOptions {
  std::optional<std::vector<T>> default_value;
  std::optional<std::vector<T>> global_value;
  bool handle = false;
};

struct FieldInfo {
  FieldId id = invalid_field;
  FamilyId family = invalid_family;
  std::string name;
  DataType type = DataType::f64;
  int ncomp = 1;  // 0 for a variable-length field
  bool variable = false;  // a VarField, fetched with get_var
  bool handle = false;    // values are gids (FieldOptions::handle)
  Layout layout{};
  LO rows = 0;
  std::vector<int> dims;  // dimensions the domain has members on
  std::uint64_t revision = 0;  // advances each time a commit replaces the domain
};

// Typed handle on one field. It aliases the registry's storage, so it stays
// valid after the registry changes but not after remove(). Kernels capture
// view() (a View, not a Field). A moved-from Field is empty: no rows, no
// components, an empty space, and invalid_field as its id.
template <class T, class MemSpace>
class Field {
 public:
  using value_type = T;
  using memory_space = MemSpace;
  using ViewType = View2<T, MemSpace>;

  Field() = default;
  Field(FieldId id, Space<MemSpace> space, ViewType data, std::vector<T> default_value,
        std::vector<T> global_value, std::uint64_t revision = 0, std::uint64_t owner = 0)
      : revision_(revision),
        owner_(owner),
        id_(id),
        space_(std::move(space)),
        data_(std::move(data)),
        default_(std::move(default_value)),
        global_(std::move(global_value)) {}

  Field(const Field&) = default;
  Field& operator=(const Field&) = default;
  // A moved View still references its data, so move by swapping with an empty
  // field.
  Field(Field&& o) noexcept { swap(o); }
  Field& operator=(Field&& o) noexcept {
    Field tmp(std::move(o));
    swap(tmp);
    return *this;
  }

  void swap(Field& o) noexcept {
    std::swap(revision_, o.revision_);
    std::swap(owner_, o.owner_);
    std::swap(id_, o.id_);
    std::swap(space_, o.space_);
    std::swap(data_, o.data_);
    default_.swap(o.default_);
    global_.swap(o.global_);
  }

  FieldId id() const noexcept { return id_; }
  // Handles are snapshots: after a commit replaces the domain, fetch the field
  // again (FieldRegistry::is_current).
  std::uint64_t revision() const noexcept { return revision_; }
  // Serial of the registry that produced this handle (0: none). Together with
  // id() and revision() it identifies the storage the handle belongs to; it
  // does not depend on the View, whose data pointer can be null when empty and
  // is shared between a source and an aliasing mirror.
  std::uint64_t owner() const noexcept { return owner_; }
  // The reference-returning accessors have an rvalue overload that returns a
  // copy, so `for (x : reg.get<T>("n").default_value())` does not read a
  // destroyed temporary. Named fields still get references.
  const Space<MemSpace>& space() const& noexcept { return space_; }
  Space<MemSpace> space() && { return space_; }
  const Domain<MemSpace>& domain() const& noexcept { return space_.domain(); }
  Domain<MemSpace> domain() && { return space_.domain(); }
  // The View of this handle's storage. After a commit replaces the field the
  // registry holds new storage; writes through a stale handle's view are lost.
  // FieldRegistry::is_current(field) tells.
  ViewType view() const noexcept { return data_; }
  LO rows() const noexcept { return static_cast<LO>(data_.extent(0)); }
  int ncomp() const noexcept { return static_cast<int>(data_.extent(1)); }

  bool has_default() const noexcept { return !default_.empty(); }
  const std::vector<T>& default_value() const& noexcept { return default_; }
  std::vector<T> default_value() && { return default_; }
  bool has_global() const noexcept { return !global_.empty(); }
  const std::vector<T>& global_value() const& noexcept { return global_; }
  std::vector<T> global_value() && { return global_; }

  // Host read of component `c` at the first value point of entity (dim, lid):
  // the stored value for a member, the default otherwise (zero without one).
  // Not available for fields in a device-only memory space.
  T value_or_default(int dim, LO lid, int c = 0) const {
    static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible,
                  "value_or_default reads on the host; mirror the field first");
    const RowRange r = space_.entity_rows(dim, lid);
    if (r.valid()) return data_(r.begin, c);
    return has_default() ? default_[static_cast<std::size_t>(c)] : T{};
  }

 private:
  std::uint64_t revision_ = 0;
  std::uint64_t owner_ = 0;
  FieldId id_ = invalid_field;
  Space<MemSpace> space_;
  ViewType data_;
  std::vector<T> default_;
  std::vector<T> global_;
};

}  // namespace soma

#endif  // SOMA_FIELDS_FIELD_HPP_
