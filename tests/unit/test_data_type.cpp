// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "soma/fields/data_type.hpp"

using soma::DataType;

static_assert(soma::data_type_of<std::int8_t>() == DataType::i8);
static_assert(soma::data_type_of<std::uint64_t>() == DataType::u64);
static_assert(soma::data_type_of<float>() == DataType::f32);
static_assert(soma::data_type_of<double>() == DataType::f64);

TEST(DataType, DispatchInvokesTheMatchingType) {
  const DataType all[] = {DataType::i8,  DataType::u8,  DataType::i32, DataType::u32,
                          DataType::i64, DataType::u64, DataType::f32, DataType::f64};
  for (const DataType t : all) {
    DataType seen = DataType::i8;
    std::size_t bytes = 0;
    soma::dispatch_data_type(t, [&]<class T>() {
      seen = soma::data_type_of<T>();
      bytes = sizeof(T);
    });
    EXPECT_EQ(seen, t) << soma::name(t);
    EXPECT_GT(bytes, 0u);
  }
}

TEST(DataType, DispatchPassesWidths) {
  std::vector<std::size_t> widths;
  for (const DataType t : {DataType::i8, DataType::i32, DataType::i64, DataType::f32,
                           DataType::f64}) {
    soma::dispatch_data_type(t, [&]<class T>() { widths.push_back(sizeof(T)); });
  }
  EXPECT_EQ(widths, (std::vector<std::size_t>{1, 4, 8, 4, 8}));
}

TEST(DataType, SignednessIsPreserved) {
  bool is_signed = true;
  soma::dispatch_data_type(DataType::u32, [&]<class T>() { is_signed = std::is_signed_v<T>; });
  EXPECT_FALSE(is_signed);
  soma::dispatch_data_type(DataType::i32, [&]<class T>() { is_signed = std::is_signed_v<T>; });
  EXPECT_TRUE(is_signed);
}

TEST(DataType, NamesAreStable) {
  EXPECT_EQ(std::string(soma::name(DataType::f64)), "f64");
  EXPECT_EQ(std::string(soma::name(DataType::u8)), "u8");
}

TEST(DataType, InvalidValueThrows) {
  EXPECT_THROW(soma::dispatch_data_type(static_cast<DataType>(200), []<class T>() {}),
               std::invalid_argument);
}
