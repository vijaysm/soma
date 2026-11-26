// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Field rules: typed, kept by the field's family, carried by copies and
// mirrors, gone with the family's last field.

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/fields/field_rule.hpp"

using soma::EntityBatch;
using soma::FieldRule;
using soma::GO;
using soma::LO;
using soma::Real;
using soma::Topology;

using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace {

// Three points (1,2,3), (4,5,6), (7,8,9).
struct Points {
  std::vector<LO> offsets{0, 1, 2, 3};
  std::vector<Real> xyz{1, 2, 3, 4, 5, 6, 7, 8, 9};
  EntityBatch batch() const { return EntityBatch{0, {}, offsets, xyz, {}}; }
};

FieldRule<double> x_plus_y() {
  return [](const EntityBatch& b) {
    std::vector<double> v(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) v[i] = b.x(i, 0, 0) + b.x(i, 0, 1);
    return v;
  };
}

}  // namespace

TEST(FieldRule, ABatchNamesItsEntitiesCorners) {
  // Two edges: (0,0,0)-(2,0,0) and (0,0,0)-(0,4,0)-(0,0,6) (a triangle).
  const std::vector<LO> offsets{0, 2, 5};
  const std::vector<Real> xyz{0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 6};
  const std::vector<Topology> topo{Topology::edge, Topology::tri};
  const EntityBatch b{1, topo, offsets, xyz, {}};
  ASSERT_EQ(b.size(), 2u);
  EXPECT_EQ(b.corners(0), 2);
  EXPECT_EQ(b.corners(1), 3);
  EXPECT_DOUBLE_EQ(b.x(1, 2, 2), 6.0);
  EXPECT_EQ(b.center(0), (std::array<Real, 3>{1, 0, 0}));
  EXPECT_EQ(b.center(1), (std::array<Real, 3>{0, 1, 2}));
  EXPECT_EQ(EntityBatch{}.size(), 0u);
}

TEST(FieldRule, ARuleGivesNcompValuesPerEntityOrThrows) {
  const Points p;
  const auto r = x_plus_y();
  ASSERT_TRUE(r);
  EXPECT_EQ(r(p.batch(), 1), (std::vector<double>{3, 9, 15}));
  EXPECT_THROW((void)r(p.batch(), 2), std::logic_error);
  EXPECT_FALSE(FieldRule<double>{});
}

TEST(FieldRule, PointwiseEvaluatesAtEachCenter) {
  const Points p;
  const FieldRule<std::int32_t> id = soma::pointwise([](Real x, Real, Real) { return static_cast<std::int32_t>(x); });
  EXPECT_EQ(id(p.batch(), 1), (std::vector<std::int32_t>{1, 4, 7}));
  const FieldRule<float> two = soma::pointwise(
      [](Real x, Real y, Real z) { return std::array<float, 2>{static_cast<float>(x * y), static_cast<float>(z)}; });
  EXPECT_EQ(two(p.batch(), 2), (std::vector<float>{2, 3, 20, 6, 56, 9}));
}

TEST(FieldRule, AddTagKeepsTheRuleOnTheFamily) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  soma::FieldOptions<double> o;
  o.rule = x_plus_y();
  const auto u = mesh.fields().add_tag<double>("u", 0, 1, std::move(o));
  const auto plain = mesh.fields().add_tag<double>("plain", 0, 1);
  EXPECT_TRUE(mesh.fields().has_rule(u.id()));
  EXPECT_FALSE(mesh.fields().has_rule(plain.id()));
  EXPECT_FALSE(mesh.fields().rule<double>(plain.id()));
  const Points p;
  EXPECT_EQ(mesh.fields().rule<double>(u.id())(p.batch(), 1), (std::vector<double>{3, 9, 15}));
  EXPECT_THROW((void)mesh.fields().rule<float>(u.id()), std::invalid_argument);
  EXPECT_EQ(mesh.fields().rule_names(), (std::vector<std::string>{"u"}));

  // The same name on the cells is the same family: it shares the rule.
  const auto on_cells = mesh.fields().add_tag<double>("u", 2, 1);
  EXPECT_TRUE(mesh.fields().has_rule(on_cells.id()));
}

TEST(FieldRule, SetRuleByFieldOrByNameAndClearIt) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  const auto u = mesh.fields().add_tag<double>("u", 0, 1);
  const auto k = mesh.fields().add_tag<std::int32_t>("k", 2, 1);
  mesh.fields().set_rule(u, [](const EntityBatch& b) { return std::vector<double>(b.size(), 1.5); });
  EXPECT_TRUE(mesh.fields().has_rule(u.id()));
  mesh.fields().set_rule<std::int32_t>("k", soma::pointwise([](Real, Real, Real) { return std::int32_t{4}; }));
  EXPECT_TRUE(mesh.fields().has_rule(k.id()));
  EXPECT_THROW(mesh.fields().set_rule<double>("k", x_plus_y()), std::invalid_argument);
  EXPECT_THROW(mesh.fields().set_rule<double>("none", x_plus_y()), std::out_of_range);
  mesh.fields().set_rule(u, FieldRule<double>{});
  EXPECT_FALSE(mesh.fields().has_rule(u.id()));
  EXPECT_EQ(mesh.fields().rule_names(), (std::vector<std::string>{"k"}));
}

TEST(FieldRule, AVariableLengthFieldTakesNone) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  const auto space = mesh.fields().get<double>(mesh.fields().add_tag<double>("tmp", 2, 1).id()).space();
  const std::vector<soma::Offset> offsets(static_cast<std::size_t>(space.rows()) + 1, 0);
  soma::FieldOptions<double> o;
  o.rule = x_plus_y();
  EXPECT_THROW(mesh.fields().add_var<double>("var", space, offsets, std::span<const double>{}, std::move(o)),
               std::invalid_argument);
}

TEST(FieldRule, ClonesAndMirrorsCarryTheRule) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  soma::FieldOptions<double> o;
  o.rule = x_plus_y();
  mesh.fields().add_tag<double>("u", 0, 1, std::move(o));
  const Points p;
  const auto copy = mesh.clone();
  EXPECT_EQ(copy.fields().rule<double>(copy.fields().get<double>("u").id())(p.batch(), 1),
            (std::vector<double>{3, 9, 15}));
  const auto device = mesh.mirror<DevMem>(soma::TransferOptions::everything());
  EXPECT_TRUE(device.fields().has_rule(device.fields().get<double>("u").id()));
  soma::HostMesh moved = std::move(mesh);
  EXPECT_TRUE(moved.fields().has_rule(moved.fields().get<double>("u").id()));
}

TEST(FieldRule, TheRuleGoesWithTheFamilysLastField) {
  auto mesh = soma_test::two_quads_and_a_triangle();
  soma::FieldOptions<double> o;
  o.rule = x_plus_y();
  const auto a = mesh.fields().add_tag<double>("u", 0, 1, std::move(o));
  const auto b = mesh.fields().add_tag<double>("u", 2, 1);
  mesh.fields().remove(a.id());
  EXPECT_TRUE(mesh.fields().has_rule(b.id()));
  mesh.fields().remove(b.id());
  const auto again = mesh.fields().add_tag<double>("u", 0, 1);
  EXPECT_FALSE(mesh.fields().has_rule(again.id()));
}
