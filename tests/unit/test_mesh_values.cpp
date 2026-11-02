// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_values.hpp"
#include "soma/core/transfer.hpp"

namespace {

soma::MeshValue ints(std::vector<std::int32_t> global) {
  soma::MeshValue v;
  v.type = soma::DataType::i32;
  v.ncomp = static_cast<int>(global.size());
  v.storage_class = 3;
  v.global_value = soma::MeshValue::bytes_of<std::int32_t>(global);
  return v;
}

}  // namespace

TEST(MeshValues, HoldValuesByName) {
  soma::MeshValues values;
  EXPECT_TRUE(values.empty());
  values.set("b", ints({1, 2}));
  values.set("a", ints({3}));
  EXPECT_EQ(values.names(), (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(values.at("b").global_as<std::int32_t>(), (std::vector<std::int32_t>{1, 2}));
  EXPECT_FALSE(values.at("b").default_as<std::int32_t>().has_value());
  values.set("b", ints({7}));  // replaces
  EXPECT_EQ(values.size(), 2u);
  EXPECT_EQ(values.at("b").global_as<std::int32_t>(), std::vector<std::int32_t>{7});
  EXPECT_TRUE(values.erase("a"));
  EXPECT_FALSE(values.erase("a"));
  EXPECT_FALSE(values.contains("a"));
  EXPECT_THROW(values.at("a"), std::out_of_range);
}

TEST(MeshValues, ReadOnlyAsTheirType) {
  const auto v = ints({1});
  try {
    (void)v.global_as<std::int64_t>();
    ADD_FAILURE() << "read as i64";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("MeshValue: read as the wrong value type (stored i32)"), std::string::npos);
  }
  soma::MeshValue odd = ints({});
  odd.global_value = std::vector<std::byte>(6);
  EXPECT_THROW((void)odd.global_as<std::int32_t>(), std::logic_error);
  soma::MeshValue empty = ints({});
  EXPECT_EQ(empty.global_as<std::int32_t>(), std::vector<std::int32_t>{});  // present, of length 0
}

// Every mirror and clone carries the values, even one that keeps nothing else.
TEST(MeshValues, TravelWithEveryMirror) {
  soma::HostMesh mesh = soma_test::two_quads_and_a_triangle();
  mesh.values().set("m", ints({5}));
  soma::TransferOptions lean;
  lean.dims = {false, false, false, false};
  lean.coordinates = false;
  lean.sets = false;
  for (const auto& o : {soma::TransferOptions{}, lean}) {
    const auto dev = mesh.mirror<Kokkos::DefaultExecutionSpace::memory_space>(o);
    EXPECT_EQ(dev.values().at("m").global_as<std::int32_t>(), std::vector<std::int32_t>{5});
  }
  const auto copy = mesh.clone();
  mesh.values().set("m", ints({6}));
  EXPECT_EQ(copy.values().at("m").global_as<std::int32_t>(), std::vector<std::int32_t>{5});
  soma::HostMesh moved = std::move(mesh);
  EXPECT_TRUE(moved.values().contains("m"));
  EXPECT_TRUE(mesh.values().empty());  // NOLINT(bugprone-use-after-move)
}
