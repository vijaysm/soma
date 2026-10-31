// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_FIELD_HPP_
#define SOMA_FIELDS_FIELD_HPP_

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
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
// false. global_value is the one value per mesh that MOAB tags can carry.
template <class T>
struct FieldOptions {
  std::optional<std::vector<T>> default_value;
  std::optional<std::vector<T>> global_value;
};

struct FieldInfo {
  FieldId id = invalid_field;
  FamilyId family = invalid_family;
  std::string name;
  DataType type = DataType::f64;
  int ncomp = 1;
  Layout layout{};
  LO rows = 0;
  std::vector<int> dims;  // dimensions the domain has members on
};

// Typed handle on one field. It aliases the registry's storage, so it stays
// valid after the registry changes but not after remove(). Kernels capture
// view() (a View, not a Field).
template <class T, class MemSpace>
class Field {
 public:
  using value_type = T;
  using memory_space = MemSpace;
  using ViewType = View2<T, MemSpace>;

  Field() = default;
  Field(FieldId id, Space<MemSpace> space, ViewType data, std::vector<T> default_value,
        std::vector<T> global_value)
      : id_(id),
        space_(std::move(space)),
        data_(std::move(data)),
        default_(std::move(default_value)),
        global_(std::move(global_value)) {}

  FieldId id() const noexcept { return id_; }
  const Space<MemSpace>& space() const noexcept { return space_; }
  const Domain<MemSpace>& domain() const noexcept { return space_.domain(); }
  ViewType view() const noexcept { return data_; }
  LO rows() const noexcept { return static_cast<LO>(data_.extent(0)); }
  int ncomp() const noexcept { return static_cast<int>(data_.extent(1)); }

  bool has_default() const noexcept { return !default_.empty(); }
  const std::vector<T>& default_value() const noexcept { return default_; }
  bool has_global() const noexcept { return !global_.empty(); }
  const std::vector<T>& global_value() const noexcept { return global_; }

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
  FieldId id_ = invalid_field;
  Space<MemSpace> space_;
  ViewType data_;
  std::vector<T> default_;
  std::vector<T> global_;
};

}  // namespace soma

#endif  // SOMA_FIELDS_FIELD_HPP_
