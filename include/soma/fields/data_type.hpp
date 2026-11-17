// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_DATA_TYPE_HPP_
#define SOMA_FIELDS_DATA_TYPE_HPP_

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace soma {

enum class DataType : std::uint8_t { i8, u8, i32, u32, i64, u64, f32, f64 };

template <class T>
constexpr DataType data_type_of() {
  if constexpr (std::is_same_v<T, std::int8_t>) return DataType::i8;
  else if constexpr (std::is_same_v<T, std::uint8_t>) return DataType::u8;
  else if constexpr (std::is_same_v<T, std::int32_t>) return DataType::i32;
  else if constexpr (std::is_same_v<T, std::uint32_t>) return DataType::u32;
  else if constexpr (std::is_same_v<T, std::int64_t>) return DataType::i64;
  else if constexpr (std::is_same_v<T, std::uint64_t>) return DataType::u64;
  else if constexpr (std::is_same_v<T, float>) return DataType::f32;
  else if constexpr (std::is_same_v<T, double>) return DataType::f64;
  else static_assert(sizeof(T) == 0, "unsupported field value type");
}

constexpr const char* name(DataType t) noexcept {
  switch (t) {
    case DataType::i8: return "i8";
    case DataType::u8: return "u8";
    case DataType::i32: return "i32";
    case DataType::u32: return "u32";
    case DataType::i64: return "i64";
    case DataType::u64: return "u64";
    case DataType::f32: return "f32";
    case DataType::f64: return "f64";
  }
  return "unknown";
}

/// Invokes f.template operator()<T>() for the C++ type matching `type`.
template <class F>
void dispatch_data_type(DataType type, F&& f) {
  switch (type) {
    case DataType::i8: f.template operator()<std::int8_t>(); return;
    case DataType::u8: f.template operator()<std::uint8_t>(); return;
    case DataType::i32: f.template operator()<std::int32_t>(); return;
    case DataType::u32: f.template operator()<std::uint32_t>(); return;
    case DataType::i64: f.template operator()<std::int64_t>(); return;
    case DataType::u64: f.template operator()<std::uint64_t>(); return;
    case DataType::f32: f.template operator()<float>(); return;
    case DataType::f64: f.template operator()<double>(); return;
  }
  throw std::invalid_argument("dispatch_data_type: invalid DataType value");
}

}  // namespace soma

#endif  // SOMA_FIELDS_DATA_TYPE_HPP_
