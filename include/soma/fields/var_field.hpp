// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_VAR_FIELD_HPP_
#define SOMA_FIELDS_VAR_FIELD_HPP_

#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/field.hpp"
#include "soma/fields/space.hpp"

namespace soma {

/// Typed handle on one variable-length field: a CSR over a per_entity space.
/// Member r holds values()[offsets()(r) .. offsets()(r + 1)). The values can
/// be written; the lengths are fixed when the field is added. The default and
/// the global value are rows of any length, each present or not. Like Field it
/// aliases the registry's storage, and a moved-from VarField is empty.
template <class T, class MemSpace>
class VarField {
 public:
  using value_type = T;
  using memory_space = MemSpace;
  using OffsetView = View1<const Offset, MemSpace>;
  using ValueView = View1<T, MemSpace>;

  VarField() = default;
  VarField(FieldId id, Space<MemSpace> space, OffsetView offsets, ValueView values,
           std::optional<std::vector<T>> default_value, std::optional<std::vector<T>> global_value,
           std::uint64_t revision = 0, std::uint64_t owner = 0)
      : revision_(revision),
        owner_(owner),
        id_(id),
        space_(std::move(space)),
        offsets_(std::move(offsets)),
        values_(std::move(values)),
        default_(std::move(default_value)),
        global_(std::move(global_value)) {}

  VarField(const VarField&) = default;
  VarField& operator=(const VarField&) = default;
  VarField(VarField&& o) noexcept { swap(o); }
  VarField& operator=(VarField&& o) noexcept {
    VarField tmp(std::move(o));
    swap(tmp);
    return *this;
  }

  void swap(VarField& o) noexcept {
    std::swap(revision_, o.revision_);
    std::swap(owner_, o.owner_);
    std::swap(id_, o.id_);
    std::swap(space_, o.space_);
    std::swap(offsets_, o.offsets_);
    std::swap(values_, o.values_);
    default_.swap(o.default_);
    global_.swap(o.global_);
  }

  FieldId id() const noexcept { return id_; }
  std::uint64_t revision() const noexcept { return revision_; }
  std::uint64_t owner() const noexcept { return owner_; }
  const Space<MemSpace>& space() const& noexcept { return space_; }
  Space<MemSpace> space() && { return space_; }
  const Domain<MemSpace>& domain() const& noexcept { return space_.domain(); }
  Domain<MemSpace> domain() && { return space_.domain(); }
  /// rows() + 1 offsets; empty for an empty handle.
  OffsetView offsets() const noexcept { return offsets_; }
  ValueView values() const noexcept { return values_; }
  LO rows() const noexcept { return offsets_.extent(0) == 0 ? 0 : static_cast<LO>(offsets_.extent(0) - 1); }

  bool has_default() const noexcept { return default_.has_value(); }
  const std::optional<std::vector<T>>& default_value() const& noexcept { return default_; }
  std::optional<std::vector<T>> default_value() && { return default_; }
  bool has_global() const noexcept { return global_.has_value(); }
  const std::optional<std::vector<T>>& global_value() const& noexcept { return global_; }
  std::optional<std::vector<T>> global_value() && { return global_; }

  /// Host read of entity (dim, lid): its values for a member, the default
  /// otherwise (no values without one). The span refers to this handle.
  std::span<const T> value_or_default(int dim, LO lid) const {
    static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible,
                  "value_or_default reads on the host; mirror the field first");
    const RowRange r = space_.entity_rows(dim, lid);
    if (r.valid()) {
      const auto b = static_cast<std::size_t>(offsets_(static_cast<std::size_t>(r.begin)));
      const auto e = static_cast<std::size_t>(offsets_(static_cast<std::size_t>(r.begin) + 1));
      return {values_.data() + b, e - b};
    }
    if (default_) return *default_;
    return {};
  }

 private:
  std::uint64_t revision_ = 0;
  std::uint64_t owner_ = 0;
  FieldId id_ = invalid_field;
  Space<MemSpace> space_;
  OffsetView offsets_;
  ValueView values_;
  std::optional<std::vector<T>> default_;
  std::optional<std::vector<T>> global_;
};

}  // namespace soma

#endif  // SOMA_FIELDS_VAR_FIELD_HPP_
