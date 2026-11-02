// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/errors.hpp"
#include "soma/fields/field.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/registry.hpp"
#include "soma/fields/space.hpp"

using soma::Layout;
using soma::LO;
using soma::NodeFamily;
using soma::Topology;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace soma_test {

template <class T, class M>
double sum_on_space(soma::View2<T, M> v) {
  double s = 0;
  Kokkos::parallel_reduce(
      "soma::test::field_sum", Kokkos::RangePolicy<typename M::execution_space>(0, v.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i, double& acc) {
        for (std::size_t c = 0; c < v.extent(1); ++c) acc += static_cast<double>(v(i, c));
      },
      s);
  return s;
}

template <class T, class M>
void write_on_space(soma::View2<T, M> v, T value) {
  Kokkos::parallel_for(
      "soma::test::field_write", Kokkos::RangePolicy<typename M::execution_space>(0, v.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        for (std::size_t c = 0; c < v.extent(1); ++c) v(i, c) = value;
      });
  Kokkos::fence();
}

template <class M>
soma::ElementInfo<M> hexes(LO n) {
  soma::View1<std::uint8_t, Kokkos::HostSpace> h("soma::test::topo", static_cast<std::size_t>(n));
  for (LO i = 0; i < n; ++i) h(i) = static_cast<std::uint8_t>(Topology::hex);
  soma::ElementInfo<M> info;
  info.topology[3] = Kokkos::create_mirror_view_and_copy(M{}, h);
  return info;
}

}  // namespace soma_test

template <class M>
class RegistryTyped : public ::testing::Test {
 protected:
  soma::FieldRegistry<M> reg;
  void SetUp() override {
    reg.set_entities(0, 10);
    reg.set_entities(2, 7);
    reg.set_entities(3, 4);
  }
};
using Spaces = ::testing::Types<HostMem, DevMem>;
TYPED_TEST_SUITE(RegistryTyped, Spaces);

TYPED_TEST(RegistryTyped, AddTagInitializesAndGetReturnsSameStorage) {
  soma::FieldOptions<double> opt;
  opt.default_value = std::vector<double>{2.5, 2.5, 2.5};
  auto a = this->reg.template add_tag<double>("temperature", 0, 3, opt);
  EXPECT_EQ(a.rows(), 10);
  EXPECT_EQ(a.ncomp(), 3);
  EXPECT_DOUBLE_EQ((soma_test::sum_on_space<double, TypeParam>(a.view())), 75.0);
  auto b = this->reg.template get<double>("temperature");
  EXPECT_EQ(a.view().data(), b.view().data());
  EXPECT_TRUE(b.has_default());
}

TYPED_TEST(RegistryTyped, WithoutDefaultFieldsAreZeroFilled) {
  auto a = this->reg.template add_tag<std::int32_t>("zero", 2, 2);
  EXPECT_FALSE(a.has_default());
  EXPECT_DOUBLE_EQ((soma_test::sum_on_space<std::int32_t, TypeParam>(a.view())), 0.0);
}

TYPED_TEST(RegistryTyped, PerComponentDefaultsFillTheirColumns) {
  soma::FieldOptions<float> opt;
  opt.default_value = std::vector<float>{1.0F, 10.0F};
  auto a = this->reg.template add_tag<float>("two", 0, 2, opt);
  EXPECT_DOUBLE_EQ((soma_test::sum_on_space<float, TypeParam>(a.view())), 10 * 11.0);
}

TYPED_TEST(RegistryTyped, TypeAndDimensionAreChecked) {
  this->reg.template add_tag<std::int32_t>("material", 3, 1);
  EXPECT_THROW(this->reg.template get<double>("material"), std::invalid_argument);
  EXPECT_THROW(this->reg.template get<std::int32_t>("material", 2), std::out_of_range);
  EXPECT_THROW(this->reg.template get<std::int32_t>("missing"), std::out_of_range);
  EXPECT_NO_THROW(this->reg.template get<std::int32_t>("material", 3));
  EXPECT_THROW(this->reg.template add_tag<std::int32_t>("x", 5, 1), std::invalid_argument);
  EXPECT_THROW(this->reg.template add_tag<std::int32_t>("x", -1, 1), std::invalid_argument);
  EXPECT_THROW(this->reg.template add_tag<std::int32_t>("x", soma::set_dim, 1), std::logic_error);
  this->reg.set_entities(soma::set_dim, 6);
  EXPECT_EQ(this->reg.template add_tag<std::int32_t>("x", soma::set_dim, 1).rows(), 6);
  EXPECT_TRUE(this->reg.has_entities(soma::set_dim));
  EXPECT_FALSE(this->reg.has_entities(5));
  EXPECT_THROW(this->reg.template add_tag<std::int32_t>("x", 3, 0), std::invalid_argument);
}

TYPED_TEST(RegistryTyped, DefaultSizeMustMatchComponents) {
  soma::FieldOptions<double> opt;
  opt.default_value = std::vector<double>{1.0};
  EXPECT_THROW(this->reg.template add_tag<double>("bad", 0, 3, opt), std::invalid_argument);
  soma::FieldOptions<double> g;
  g.global_value = std::vector<double>{1.0, 2.0};
  EXPECT_THROW(this->reg.template add_tag<double>("bad", 0, 3, g), std::invalid_argument);
}

TYPED_TEST(RegistryTyped, AddTagNeedsEntityCounts) {
  soma::FieldRegistry<TypeParam> bare;
  EXPECT_THROW(bare.template add_tag<double>("t", 0, 1), std::logic_error);
}

TYPED_TEST(RegistryTyped, IdentityClashNeedsNameTypeAndSpace) {
  using D = soma::Domain<TypeParam>;
  this->reg.template add_tag<double>("T", 3, 1);
  EXPECT_THROW(this->reg.template add_tag<double>("T", 3, 1), soma::FieldClash);
  EXPECT_NO_THROW(this->reg.template add_tag<float>("T", 3, 1));
  EXPECT_NO_THROW(this->reg.template add_tag<double>("T", 3, 2));
  EXPECT_NO_THROW(this->reg.template add_tag<double>("T", 2, 1));
  // A per_entity field and a node field on the same hexes are different spaces.
  const auto info = soma_test::hexes<TypeParam>(4);
  const soma::Space<TypeParam> sem(D::all(3, 4, 1), Layout::element_nodes(NodeFamily::gll, 2), info);
  EXPECT_NO_THROW(this->reg.template add<double>("T", sem, 1));
  EXPECT_THROW(this->reg.template add<double>("T", sem, 1), soma::FieldClash);
}

TYPED_TEST(RegistryTyped, ClashIsStructuralNotByPointer) {
  using D = soma::Domain<TypeParam>;
  const soma::Space<TypeParam> a(D::from_list(2, 7, 1, std::vector<LO>{1, 3}), Layout::per_entity());
  const soma::Space<TypeParam> b(D::from_list(2, 7, 1, std::vector<LO>{1, 3}), Layout::per_entity());
  ASSERT_NE(a.storage_id(), b.storage_id());
  this->reg.template add<double>("q", a, 1);
  EXPECT_THROW(this->reg.template add<double>("q", b, 1), soma::FieldClash);
}

TYPED_TEST(RegistryTyped, SameNameOnDifferentDomainsAreDistinctFields) {
  using D = soma::Domain<TypeParam>;
  auto v = this->reg.template add_tag<std::int32_t>("GLOBAL_ID", 0, 1);
  auto f = this->reg.template add<std::int32_t>(
      "GLOBAL_ID", soma::Space<TypeParam>(D::all(2, 7, 1), Layout::per_entity()), 1);
  EXPECT_NE(v.id(), f.id());
  EXPECT_THROW(this->reg.template get<std::int32_t>("GLOBAL_ID"), std::invalid_argument);
  EXPECT_EQ(this->reg.template get<std::int32_t>("GLOBAL_ID", 2).id(), f.id());
  EXPECT_EQ(this->reg.template get<std::int32_t>("GLOBAL_ID", 0).id(), v.id());
  EXPECT_EQ(this->reg.info(v.id()).family, this->reg.info(f.id()).family);
}

TYPED_TEST(RegistryTyped, ListReportsMetadata) {
  this->reg.template add_tag<float>("f", 2, 2);
  const auto info = this->reg.list();
  ASSERT_EQ(info.size(), 1u);
  EXPECT_EQ(info[0].name, "f");
  EXPECT_EQ(info[0].dims, std::vector<int>{2});
  EXPECT_EQ(info[0].ncomp, 2);
  EXPECT_EQ(info[0].type, soma::DataType::f32);
  EXPECT_EQ(info[0].rows, 7);
  EXPECT_EQ(info[0].layout, Layout::per_entity());
}

TYPED_TEST(RegistryTyped, FieldIdsAreStableAndNeverReused) {
  const auto a = this->reg.template add_tag<double>("a", 0, 1);
  const auto b = this->reg.template add_tag<double>("b", 0, 1);
  this->reg.remove(a.id());
  EXPECT_FALSE(this->reg.contains("a"));
  EXPECT_THROW(this->reg.template get<double>(a.id()), std::out_of_range);
  EXPECT_THROW(this->reg.remove(a.id()), std::out_of_range);
  const auto c = this->reg.template add_tag<double>("a", 0, 1);
  EXPECT_NE(c.id(), a.id());
  EXPECT_EQ(this->reg.template get<double>(b.id()).id(), b.id());
  EXPECT_EQ(this->reg.list().size(), 2u);
}

TYPED_TEST(RegistryTyped, StaleFieldThrowsAfterAnEpochChange) {
  this->reg.template add_tag<double>("t", 3, 1);
  this->reg.set_epoch(this->reg.epoch() + 1);
  EXPECT_THROW(this->reg.template get<double>("t"), soma::StaleField);
  EXPECT_THROW(this->reg.template get<double>("t", 3), soma::StaleField);
  this->reg.set_epoch(this->reg.epoch() - 1);
  EXPECT_NO_THROW(this->reg.template get<double>("t"));
}

TYPED_TEST(RegistryTyped, AddingOnAStaleDomainThrows) {
  using D = soma::Domain<TypeParam>;
  this->reg.set_epoch(5);
  const soma::Space<TypeParam> old_space(D::all(3, 4, 4), Layout::per_entity());
  EXPECT_THROW(this->reg.template add<double>("t", old_space, 1), soma::StaleField);
}

TYPED_TEST(RegistryTyped, SparseFieldReportsTheDefaultForNonMembers) {
  using D = soma::Domain<TypeParam>;
  soma::FieldOptions<double> opt;
  opt.default_value = std::vector<double>{-1.0};
  const soma::Space<TypeParam> sp(D::from_list(2, 7, 1, std::vector<LO>{2, 5}), Layout::per_entity());
  auto f = this->reg.template add<double>("flux", sp, 1, opt);
  EXPECT_EQ(f.rows(), 2);
  soma::FieldRegistry<HostMem> host;
  this->reg.template mirror_into<HostMem>(host, {"*"}, {true, true, true, true}, false);
  const auto hf = host.template get<double>("flux");
  EXPECT_DOUBLE_EQ(hf.value_or_default(2, 2), -1.0);
  EXPECT_DOUBLE_EQ(hf.value_or_default(2, 0), -1.0);
  EXPECT_DOUBLE_EQ(hf.value_or_default(2, 5), -1.0);
}

TEST(FieldRegistry, ValueOrDefaultReturnsStoredValuesForMembers) {
  using D = soma::Domain<HostMem>;
  soma::FieldRegistry<HostMem> reg;
  soma::FieldOptions<double> opt;
  opt.default_value = std::vector<double>{-1.0};
  auto f = reg.add<double>(
      "flux", soma::Space<HostMem>(D::from_list(2, 9, 1, std::vector<LO>{2, 5}), Layout::per_entity()),
      1, opt);
  f.view()(1, 0) = 8.0;
  EXPECT_DOUBLE_EQ(f.value_or_default(2, 5), 8.0);
  EXPECT_DOUBLE_EQ(f.value_or_default(2, 2), -1.0);
  EXPECT_DOUBLE_EQ(f.value_or_default(2, 3), -1.0);
  EXPECT_DOUBLE_EQ(f.value_or_default(3, 5), -1.0);
}

TEST(FieldRegistry, ValueOrDefaultWithoutDefaultIsZero) {
  using D = soma::Domain<HostMem>;
  soma::FieldRegistry<HostMem> reg;
  auto f = reg.add<std::int32_t>(
      "n", soma::Space<HostMem>(D::from_list(2, 9, 1, std::vector<LO>{2}), Layout::per_entity()), 1);
  EXPECT_EQ(f.value_or_default(2, 4), 0);
}

TEST(FieldRegistry, GlobalValueIsKept) {
  soma::FieldRegistry<HostMem> reg;
  reg.set_entities(0, 3);
  soma::FieldOptions<double> opt;
  opt.global_value = std::vector<double>{4.5};
  const auto f = reg.add_tag<double>("g", 0, 1, opt);
  ASSERT_TRUE(f.has_global());
  EXPECT_DOUBLE_EQ(f.global_value()[0], 4.5);
  EXPECT_FALSE(f.has_default());
}

TEST(FieldRegistry, MoveEmptiesTheSource) {
  soma::FieldRegistry<HostMem> a;
  a.set_entities(0, 4);
  a.add_tag<double>("t", 0, 1);
  soma::FieldRegistry<HostMem> b = std::move(a);
  EXPECT_EQ(b.list().size(), 1u);
  EXPECT_TRUE(a.list().empty());  // NOLINT(bugprone-use-after-move)
  EXPECT_FALSE(a.has_entities(0));
  soma::FieldRegistry<HostMem> c;
  c = std::move(b);
  EXPECT_EQ(c.list().size(), 1u);
  EXPECT_TRUE(b.list().empty());  // NOLINT(bugprone-use-after-move)
}

static_assert(!std::is_copy_constructible_v<soma::FieldRegistry<HostMem>>);
static_assert(!std::is_copy_assignable_v<soma::FieldRegistry<HostMem>>);

TYPED_TEST(RegistryTyped, CloneIsDeep) {
  auto a = this->reg.template add_tag<double>("u", 0, 1);
  soma_test::write_on_space<double, TypeParam>(a.view(), 1.0);
  soma::FieldRegistry<TypeParam> copy;
  this->reg.template mirror_into<TypeParam>(copy, {"*"}, {true, true, true, true}, false);
  auto c = copy.template get<double>("u");
  EXPECT_NE(c.view().data(), a.view().data());
  EXPECT_DOUBLE_EQ((soma_test::sum_on_space<double, TypeParam>(c.view())), 10.0);
  soma_test::write_on_space<double, TypeParam>(c.view(), 3.0);
  EXPECT_DOUBLE_EQ((soma_test::sum_on_space<double, TypeParam>(a.view())), 10.0);
}

TEST(FieldRegistry, MirrorAliasesWhenSpaceIsSame) {
  soma::FieldRegistry<HostMem> reg;
  reg.set_entities(0, 4);
  auto a = reg.add_tag<double>("u", 0, 1);
  soma::FieldRegistry<HostMem> alias;
  reg.mirror_into<HostMem>(alias, {"u"}, {true, true, true, true}, true);
  EXPECT_EQ(alias.get<double>("u").view().data(), a.view().data());
}

TEST(FieldRegistry, MirrorRejectsUnknownNameAndNonEmptyDestination) {
  soma::FieldRegistry<HostMem> reg;
  reg.set_entities(0, 4);
  reg.add_tag<double>("u", 0, 1);
  soma::FieldRegistry<HostMem> out;
  EXPECT_THROW(reg.mirror_into<HostMem>(out, {"missing"}, {true, true, true, true}, false),
               std::out_of_range);
  EXPECT_TRUE(out.list().empty());
  soma::FieldRegistry<HostMem> full;
  reg.mirror_into<HostMem>(full, {"*"}, {true, true, true, true}, false);
  EXPECT_THROW(reg.mirror_into<HostMem>(full, {"*"}, {true, true, true, true}, false),
               std::logic_error);
}

TEST(FieldRegistry, MirrorSelectsByNameAndDimension) {
  soma::FieldRegistry<HostMem> reg;
  reg.set_entities(0, 4);
  reg.set_entities(3, 2);
  reg.add_tag<double>("v", 0, 1);
  reg.add_tag<double>("c", 3, 1);
  soma::FieldRegistry<HostMem> only_v;
  reg.mirror_into<HostMem>(only_v, {"v"}, {true, true, true, true}, false);
  EXPECT_TRUE(only_v.contains("v"));
  EXPECT_FALSE(only_v.contains("c"));
  soma::FieldRegistry<HostMem> no_cells;
  reg.mirror_into<HostMem>(no_cells, {"*"}, {true, true, true, false}, false);
  EXPECT_TRUE(no_cells.contains("v"));
  EXPECT_FALSE(no_cells.contains("c"));
}

TEST(FieldRegistry, MirrorKeepsFieldIdsAndFamilies) {
  soma::FieldRegistry<HostMem> reg;
  reg.set_entities(0, 4);
  reg.set_entities(3, 2);
  const auto a = reg.add_tag<double>("a", 0, 1);
  const auto b = reg.add_tag<double>("b", 3, 1);
  soma::FieldRegistry<HostMem> only_b;
  reg.mirror_into<HostMem>(only_b, {"b"}, {true, true, true, true}, false);
  EXPECT_EQ(only_b.get<double>("b").id(), b.id());
  EXPECT_EQ(only_b.info(b.id()).family, reg.info(b.id()).family);
  EXPECT_THROW(only_b.get<double>(a.id()), std::out_of_range);
  only_b.set_entities(0, 4);
  const auto later = only_b.add_tag<double>("a", 0, 1);
  EXPECT_NE(later.id(), a.id());
}

TYPED_TEST(RegistryTyped, MirrorSharesOneDomainBetweenFieldsOnASpace) {
  using D = soma::Domain<HostMem>;
  soma::FieldRegistry<HostMem> host;
  const soma::Space<HostMem> shared(D::from_list(2, 7, 1, std::vector<LO>{1, 4, 6}),
                                    Layout::per_entity());
  const auto a = host.add<double>("a", shared, 1);
  host.add<float>("b", shared, 2);
  const soma::Space<HostMem> other(shared.domain(), Layout::per_entity());
  host.add<double>("c", other, 1);
  soma::FieldRegistry<TypeParam> dst;
  host.mirror_into<TypeParam>(dst, {"*"}, {true, true, true, true}, false);
  const auto da = dst.template get<double>("a");
  const auto db = dst.template get<float>("b");
  const auto dc = dst.template get<double>("c");
  EXPECT_EQ(da.space().storage_id(), db.space().storage_id());
  EXPECT_TRUE(da.domain().shares_storage_with(db.domain()));
  EXPECT_NE(da.space().storage_id(), dc.space().storage_id());
  EXPECT_TRUE(da.domain().shares_storage_with(dc.domain()));
  EXPECT_EQ(da.domain().members(0), a.domain().members(0));
}

TYPED_TEST(RegistryTyped, MirroredFieldKeepsDefaultsAndValues) {
  using D = soma::Domain<HostMem>;
  soma::FieldRegistry<HostMem> host;
  soma::FieldOptions<double> opt;
  opt.default_value = std::vector<double>{-2.0};
  opt.global_value = std::vector<double>{9.0};
  auto f = host.add<double>(
      "s", soma::Space<HostMem>(D::from_list(2, 9, 1, std::vector<LO>{3, 8}), Layout::per_entity()),
      1, opt);
  f.view()(1, 0) = 5.5;
  soma::FieldRegistry<TypeParam> dst;
  host.mirror_into<TypeParam>(dst, {"s"}, {true, true, true, true}, false);
  const auto m = dst.template get<double>("s");
  EXPECT_TRUE(m.has_default());
  EXPECT_DOUBLE_EQ(m.global_value()[0], 9.0);
  EXPECT_DOUBLE_EQ((soma_test::sum_on_space<double, TypeParam>(m.view())), -2.0 + 5.5);
}

TYPED_TEST(RegistryTyped, MovedFromFieldIsEmptyAndUsable) {
  auto a = this->reg.template add_tag<double>("temperature", 0, 3);
  const auto keep = a;
  soma::Field<double, TypeParam> b(std::move(a));
  EXPECT_EQ(a.rows(), 0);
  EXPECT_EQ(a.ncomp(), 0);
  EXPECT_EQ(a.id(), soma::invalid_field);
  EXPECT_EQ(a.revision(), 0u);
  EXPECT_EQ(a.view().extent(0), 0u);
  EXPECT_EQ(a.view().data(), nullptr);
  EXPECT_EQ(a.space().rows(), 0);
  EXPECT_TRUE(a.domain().empty());
  EXPECT_FALSE(a.has_default());
  EXPECT_FALSE(a.has_global());
  EXPECT_EQ(b.rows(), 10);
  EXPECT_EQ(b.view().data(), keep.view().data());

  soma::Field<double, TypeParam> c;
  c = std::move(b);
  EXPECT_EQ(b.rows(), 0);
  EXPECT_EQ(b.view().data(), nullptr);
  EXPECT_EQ(c.rows(), 10);

  b = keep;
  EXPECT_EQ(b.rows(), 10);
  soma::Field<double, TypeParam>& alias = b;
  b = std::move(alias);
  EXPECT_EQ(b.rows(), 10);
}

TEST(Registry, MovedFromFieldReadsOnTheHost) {
  soma::FieldRegistry<HostMem> reg;
  reg.set_entities(2, 7);
  soma::FieldOptions<double> opt;
  opt.default_value = std::vector<double>{4.5};
  auto a = reg.add_tag<double>("q", 2, 1, opt);
  soma::Field<double, HostMem> b(std::move(a));
  EXPECT_DOUBLE_EQ(a.value_or_default(2, 3), 0.0);
  EXPECT_DOUBLE_EQ(b.value_or_default(2, 3), 4.5);
}

TYPED_TEST(RegistryTyped, RegistryStaysConsistentWhileItsHandlesAreMoved) {
  auto a = this->reg.template add_tag<double>("q", 2, 1);
  soma::Field<double, TypeParam> moved(std::move(a));
  const auto again = this->reg.template get<double>("q");
  EXPECT_EQ(again.view().data(), moved.view().data());
  soma::FieldRegistry<TypeParam> other(std::move(this->reg));
  EXPECT_TRUE(this->reg.list().empty());
  EXPECT_EQ(other.template get<double>("q").rows(), 7);
}

// A range-for over an accessor of a temporary Field keeps the temporary alive
// only if the accessor returns by value; with a reference the loop reads freed
// memory (AddressSanitizer: stack-use-after-scope).
TYPED_TEST(RegistryTyped, RangeForOverAccessorsOfATemporaryFieldIsSafe) {
  soma::FieldOptions<double> opt;
  opt.default_value = std::vector<double>{1.5, 2.5, 3.5};
  opt.global_value = std::vector<double>{7.0, 8.0, 9.0};
  this->reg.template add_tag<double>("temperature", 0, 3, opt);
  double defaults = 0;
  for (const double d : this->reg.template get<double>("temperature").default_value()) {
    defaults += d;
  }
  EXPECT_DOUBLE_EQ(defaults, 7.5);
  double globals = 0;
  for (const double g : this->reg.template get<double>("temperature").global_value()) globals += g;
  EXPECT_DOUBLE_EQ(globals, 24.0);
}

TYPED_TEST(RegistryTyped, AccessorsOfATemporaryFieldAreCopiesAndNamedFieldsStayReferences) {
  using F = soma::Field<double, TypeParam>;
  using Values = std::vector<double>;
  using Sp = soma::Space<TypeParam>;
  using Dm = soma::Domain<TypeParam>;
  static_assert(std::is_same_v<decltype(std::declval<F&>().default_value()), const Values&>);
  static_assert(std::is_same_v<decltype(std::declval<const F&>().global_value()), const Values&>);
  static_assert(std::is_same_v<decltype(std::declval<const F&>().space()), const Sp&>);
  static_assert(std::is_same_v<decltype(std::declval<const F&>().domain()), const Dm&>);
  static_assert(std::is_same_v<decltype(std::declval<F&&>().default_value()), std::vector<double>>);
  static_assert(std::is_same_v<decltype(std::declval<F&&>().global_value()), std::vector<double>>);
  static_assert(std::is_same_v<decltype(std::declval<F&&>().space()), soma::Space<TypeParam>>);
  static_assert(std::is_same_v<decltype(std::declval<F&&>().domain()), soma::Domain<TypeParam>>);
  const auto f = this->reg.template add_tag<double>("t", 2, 1);
  EXPECT_EQ(&f.default_value(), &f.default_value());
  EXPECT_EQ(this->reg.template get<double>("t").domain().size(), 7);
  EXPECT_EQ(this->reg.template get<double>("t").space().rows(), 7);
  EXPECT_EQ(this->reg.template get<double>("t").view().extent(0), 7u);
}

// Binding a reference to what an accessor of a temporary Field returns is the
// idiom that dangles when the accessor returns a reference into the temporary.
// The Field here is the last owner of its Space, so its destruction frees the
// Domain the reference would point into.
TYPED_TEST(RegistryTyped, AReferenceBoundToTheDomainOfATemporaryFieldIsSafe) {
  this->reg.template add_tag<double>("t", 2, 1);
  this->reg.template add_tag<double>("u", 2, 1);
  const auto make = [&](const char* name) {
    auto field = this->reg.template get<double>(name);
    this->reg.remove(field.id());
    return field;
  };
  const auto& domain = make("t").domain();
  const auto& space = make("u").space();
  EXPECT_EQ(domain.size(), 7);
  EXPECT_EQ(space.rows(), 7);
  EXPECT_EQ(domain.row(2, 3), 3);
}
