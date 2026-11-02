// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_MESH_VALUES_HPP_
#define SOMA_CORE_MESH_VALUES_HPP_

#include <cstddef>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "soma/fields/data_type.hpp"

namespace soma {

// A value of the mesh rather than of its entities: a MOAB tag of class 3, or
// one declared with values on no entity. Its default and global value, each
// present or not, are the bytes of ncomp values of `type` (any number of them
// for a variable-length tag); default_as and global_as read them typed.
struct MeshValue {
  DataType type = DataType::u8;
  int ncomp = 1;           // 0: variable length
  bool handle = false;     // values are handles (FieldOptions::handle)
  int storage_class = -1;  // MOAB's: 0 bit, 1 sparse, 2 dense, 3 mesh; -1 unknown
  std::optional<std::vector<std::byte>> default_value;
  std::optional<std::vector<std::byte>> global_value;

  template <class T>
  static std::vector<std::byte> bytes_of(std::span<const T> v) {
    std::vector<std::byte> out(v.size_bytes());
    if (!v.empty()) std::memcpy(out.data(), v.data(), out.size());
    return out;
  }

  // Throws std::invalid_argument when T is not the stored type.
  template <class T>
  std::optional<std::vector<T>> default_as() const {
    return typed<T>(default_value);
  }
  template <class T>
  std::optional<std::vector<T>> global_as() const {
    return typed<T>(global_value);
  }

 private:
  template <class T>
  std::optional<std::vector<T>> typed(const std::optional<std::vector<std::byte>>& b) const {
    if (data_type_of<T>() != type) {
      throw std::invalid_argument(std::string("MeshValue: read as the wrong value type (stored ") + name(type) +
                                  ")");
    }
    if (!b) return std::nullopt;
    if (b->size() % sizeof(T) != 0) {
      throw std::logic_error("MeshValue: " + std::to_string(b->size()) + " bytes are no whole number of values");
    }
    std::vector<T> out(b->size() / sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), b->data(), b->size());
    return out;
  }
};

// The mesh values by name. Host only; every mirror of a mesh copies them.
class MeshValues {
 public:
  // Adds the value, or replaces the one of that name.
  void set(std::string name, MeshValue v) { values_.insert_or_assign(std::move(name), std::move(v)); }
  bool contains(std::string_view name) const { return values_.find(name) != values_.end(); }
  // Throws std::out_of_range when there is none.
  const MeshValue& at(std::string_view name) const {
    const auto it = values_.find(name);
    if (it == values_.end()) throw std::out_of_range("no mesh value named '" + std::string(name) + "'");
    return it->second;
  }
  bool erase(std::string_view name) {
    const auto it = values_.find(name);
    if (it == values_.end()) return false;
    values_.erase(it);
    return true;
  }
  std::size_t size() const noexcept { return values_.size(); }
  bool empty() const noexcept { return values_.empty(); }
  // In name order.
  std::vector<std::string> names() const {
    std::vector<std::string> out;
    out.reserve(values_.size());
    for (const auto& [n, v] : values_) out.push_back(n);
    return out;
  }
  void swap(MeshValues& o) noexcept { values_.swap(o.values_); }

 private:
  std::map<std::string, MeshValue, std::less<>> values_;
};

}  // namespace soma

#endif  // SOMA_CORE_MESH_VALUES_HPP_
