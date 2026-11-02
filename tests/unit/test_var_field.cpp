// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/errors.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/registry.hpp"
#include "soma/fields/space.hpp"
#include "soma/fields/var_field.hpp"

using soma::Layout;
using soma::LO;
using soma::Offset;
using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace {

template <class M>
class VarFieldTyped : public ::testing::Test {
 protected:
  using D = soma::Domain<M>;
  soma::FieldRegistry<M> reg;
  // Entities 1, 3 and 6 of the 7 of dimension 2, holding 2, 0 and 3 values.
  soma::Space<M> space{D::from_list(2, 7, 1, std::vector<LO>{1, 3, 6}), Layout::per_entity()};
  std::vector<Offset> offsets{0, 2, 2, 5};
  std::vector<double> values{1, 2, 3, 4, 5};

  soma::VarField<double, M> add(soma::FieldOptions<double> o = {}) {
    return reg.template add_var<double>("v", space, offsets, values, std::move(o));
  }
};

using Spaces = ::testing::Types<HostMem, DevMem>;
TYPED_TEST_SUITE(VarFieldTyped, Spaces);

std::vector<double> as_vector(std::span<const double> s) { return {s.begin(), s.end()}; }

std::string what_of(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

// Each member keeps its own number of values; non-members read the default.
TYPED_TEST(VarFieldTyped, MembersHoldRowsOfAnyLength) {
  soma::FieldOptions<double> o;
  o.default_value = std::vector<double>{-1, -2};
  o.global_value = std::vector<double>{};
  const auto f = this->add(o);
  EXPECT_EQ(f.rows(), 3);
  const soma::FieldInfo& info = this->reg.info(f.id());
  EXPECT_TRUE(info.variable);
  EXPECT_EQ(info.ncomp, 0);
  EXPECT_EQ(info.rows, 3);
  EXPECT_EQ(info.dims, std::vector<int>{2});

  soma::FieldRegistry<HostMem> host;
  this->reg.template mirror_into<HostMem>(host, {"*"}, {true, true, true, true, true}, false);
  const auto h = host.template get_var<double>("v");
  EXPECT_EQ(as_vector(h.value_or_default(2, 1)), (std::vector<double>{1, 2}));
  EXPECT_TRUE(h.value_or_default(2, 3).empty());
  EXPECT_EQ(as_vector(h.value_or_default(2, 6)), (std::vector<double>{3, 4, 5}));
  EXPECT_EQ(as_vector(h.value_or_default(2, 0)), (std::vector<double>{-1, -2}));
  EXPECT_EQ(as_vector(h.value_or_default(0, 1)), (std::vector<double>{-1, -2}));
  ASSERT_TRUE(h.has_global());
  EXPECT_TRUE(h.global_value()->empty());  // present, of length 0
  EXPECT_EQ(h.offsets().extent(0), 4u);
}

TYPED_TEST(VarFieldTyped, WithoutADefaultNonMembersHoldNothing) {
  soma::FieldRegistry<HostMem> host;
  this->add();
  this->reg.template mirror_into<HostMem>(host, {"*"}, {true, true, true, true, true}, false);
  const auto h = host.template get_var<double>("v", 2);
  EXPECT_FALSE(h.has_default());
  EXPECT_FALSE(h.has_global());
  EXPECT_TRUE(h.value_or_default(2, 0).empty());
}

// Kernels read and write the values in place; the lengths stay.
TYPED_TEST(VarFieldTyped, KernelsWriteTheValues) {
  const auto f = this->add();
  const auto off = f.offsets();
  const auto val = f.values();
  using Exec = typename TypeParam::execution_space;
  Kokkos::parallel_for(
      "soma::test::var_scale", Kokkos::RangePolicy<Exec>(0, 3), KOKKOS_LAMBDA(const int r) {
        for (Offset k = off(r); k < off(r + 1); ++k) val(k) *= 10.0 * (r + 1);
      });
  Kokkos::fence();
  const auto again = this->reg.template get_var<double>(f.id());
  const auto hv = Kokkos::create_mirror_view_and_copy(HostMem{}, again.values());
  const std::vector<double> got(hv.data(), hv.data() + hv.extent(0));
  EXPECT_EQ(got, (std::vector<double>{10, 20, 90, 120, 150}));
  EXPECT_TRUE(this->reg.is_current(again));
}

TYPED_TEST(VarFieldTyped, OffsetsMustDescribeTheValues) {
  auto& reg = this->reg;
  const auto& sp = this->space;
  const std::vector<double> v{1, 2, 3, 4, 5};
  const auto reject = [&](std::vector<Offset> off, const std::string& needle) {
    const std::string what = what_of([&] { reg.template add_var<double>("v", sp, off, v); });
    EXPECT_NE(what.find(needle), std::string::npos) << what;
  };
  reject({0, 2, 5}, "field 'v': 3 offsets for 3 members; it needs one more than the members");
  reject({1, 2, 2, 5}, "the offsets must start at 0, never decrease and end at 5, the number of values");
  reject({0, 3, 2, 5}, "never decrease");
  reject({0, 2, 2, 4}, "end at 5");
  EXPECT_TRUE(reg.list().empty());
}

TYPED_TEST(VarFieldTyped, OnlyPerEntityLayoutsHoldVariableLengths) {
  using M = TypeParam;
  soma::View1<std::uint8_t, HostMem> topo("soma::test::topo", 4);
  for (std::size_t i = 0; i < 4; ++i) topo(i) = static_cast<std::uint8_t>(soma::Topology::hex);
  soma::ElementInfo<M> info;
  info.topology[3] = Kokkos::create_mirror_view_and_copy(M{}, topo);
  const soma::Space<M> nodes(soma::Domain<M>::all(3, 4, 1), Layout::element_nodes(soma::NodeFamily::gll, 1),
                             info);
  const std::vector<Offset> off(static_cast<std::size_t>(nodes.rows()) + 1, 0);
  const std::string what = what_of([&] { this->reg.template add_var<double>("v", nodes, off, std::vector<double>{}); });
  EXPECT_NE(what.find("a variable-length field needs a per_entity layout"), std::string::npos) << what;
}

// get and get_var each fetch their own kind and name the other on a mismatch.
TYPED_TEST(VarFieldTyped, FixedAndVariableFetchesDoNotMix) {
  const auto f = this->add();
  this->reg.set_entities(0, 4);
  this->reg.template add_tag<double>("fixed", 0, 1);
  EXPECT_NE(what_of([&] { this->reg.template get<double>("v"); }).find("is variable-length; fetch it with get_var"),
            std::string::npos);
  EXPECT_NE(what_of([&] { this->reg.template get_var<double>("fixed"); })
                .find("has fixed-length values; fetch it with get"),
            std::string::npos);
  EXPECT_NE(what_of([&] { this->reg.template get_var<float>(f.id()); }).find("wrong value type"), std::string::npos);
  EXPECT_THROW(this->reg.template get_var<double>("none"), std::out_of_range);
}

// A variable-length field has a family of its own: a fixed field of the same
// name on the same entities neither clashes with nor overlaps it, whichever
// comes first; a second variable one does.
TYPED_TEST(VarFieldTyped, FamiliesKeepVariableAndFixedApart) {
  const auto fixed = this->reg.template add<double>("v", this->space, 1);
  const auto f = this->add();
  const soma::Space<TypeParam> one(soma::Domain<TypeParam>::from_list(2, 7, 1, std::vector<LO>{6}),
                                   Layout::per_entity());
  EXPECT_NO_THROW(this->reg.template add<double>("v", one, 2));
  EXPECT_NE(this->reg.info(f.id()).family, this->reg.info(fixed.id()).family);
  const soma::Space<TypeParam> other(soma::Domain<TypeParam>::from_list(2, 7, 1, std::vector<LO>{0, 6}),
                                     Layout::per_entity());
  const std::vector<Offset> off{0, 1, 1};
  const std::vector<double> v{7};
  EXPECT_THROW(this->reg.template add_var<double>("v", other, off, v), soma::FieldOverlap);
  EXPECT_THROW(this->reg.template add_var<double>("v", this->space, this->offsets, this->values), soma::FieldClash);
  const soma::Space<TypeParam> rest(soma::Domain<TypeParam>::from_list(2, 7, 1, std::vector<LO>{0, 2}),
                                    Layout::per_entity());
  const auto g = this->reg.template add_var<double>("v", rest, off, v);
  EXPECT_EQ(this->reg.info(g.id()).family, this->reg.info(f.id()).family);
  const soma::FamilyHit hit = this->reg.lookup(this->reg.info(f.id()).family, 2, 6);
  ASSERT_TRUE(hit.found());
  EXPECT_EQ(this->reg.family_members(this->reg.info(f.id()).family, 2)[static_cast<std::size_t>(hit.slot)], f.id());
  EXPECT_EQ(hit.row, 2);
}

TYPED_TEST(VarFieldTyped, LengthsTakeNoPendingWrites) {
  const auto f = this->add();
  const std::vector<double> one{1};
  const std::string what = what_of([&] { this->reg.template set_pending<double>(f.id(), 0, one); });
  EXPECT_NE(what.find("field 'v' is variable-length; its lengths are fixed when it is added"), std::string::npos)
      << what;
  EXPECT_EQ(this->reg.pending_count(f.id()), 0u);
}

TYPED_TEST(VarFieldTyped, StaleDomainsThrow) {
  const auto f = this->add();
  this->reg.set_epoch(2);
  EXPECT_THROW(this->reg.template get_var<double>(f.id()), soma::StaleField);
}

TEST(VarField, MirrorsAliasOrCopy) {
  soma::FieldRegistry<HostMem> reg;
  const soma::Space<HostMem> sp(soma::Domain<HostMem>::all(0, 2, 1), Layout::per_entity());
  const std::vector<Offset> off{0, 1, 3};
  const std::vector<std::int32_t> v{4, 5, 6};
  soma::FieldOptions<std::int32_t> o;
  o.default_value = std::vector<std::int32_t>{0, 0, 0};
  const auto f = reg.add_var<std::int32_t>("v", sp, off, v, o);
  soma::FieldRegistry<HostMem> alias;
  reg.mirror_into<HostMem>(alias, {"v"}, {true, true, true, true, true}, true);
  const auto a = alias.get_var<std::int32_t>("v");
  EXPECT_EQ(a.values().data(), f.values().data());
  EXPECT_EQ(a.offsets().data(), f.offsets().data());
  EXPECT_TRUE(alias.is_current(a));
  EXPECT_FALSE(reg.is_current(a));  // a handle of another registry
  soma::FieldRegistry<HostMem> copy;
  reg.mirror_into<HostMem>(copy, {"*"}, {true, true, true, true, true}, false);
  const auto c = copy.get_var<std::int32_t>("v");
  EXPECT_NE(c.values().data(), f.values().data());
  EXPECT_NE(c.offsets().data(), f.offsets().data());
  EXPECT_EQ(c.default_value(), o.default_value);
  c.values()(0) = 40;
  EXPECT_EQ(f.values()(0), 4);
  EXPECT_EQ(std::vector<std::int32_t>(c.value_or_default(0, 1).begin(), c.value_or_default(0, 1).end()),
            (std::vector<std::int32_t>{5, 6}));
  // Without dimension 0 the field stays behind.
  soma::FieldRegistry<HostMem> none;
  reg.mirror_into<HostMem>(none, {"*"}, {false, true, true, true, true}, false);
  EXPECT_FALSE(none.contains("v"));
}

TEST(VarField, RemoveRetiresTheId) {
  soma::FieldRegistry<HostMem> reg;
  const soma::Space<HostMem> sp(soma::Domain<HostMem>::all(0, 1, 1), Layout::per_entity());
  const std::vector<Offset> off{0, 1};
  const std::vector<double> v{1};
  const auto f = reg.add_var<double>("v", sp, off, v);
  reg.remove(f.id());
  EXPECT_FALSE(reg.contains("v"));
  EXPECT_FALSE(reg.is_current(f));
  const auto g = reg.add_var<double>("v", sp, off, v);
  EXPECT_NE(g.id(), f.id());
}

TEST(VarField, MovedFromHandleIsEmpty) {
  soma::FieldRegistry<HostMem> reg;
  const soma::Space<HostMem> sp(soma::Domain<HostMem>::all(0, 1, 1), Layout::per_entity());
  const std::vector<Offset> off{0, 2};
  const std::vector<double> v{1, 2};
  auto f = reg.add_var<double>("v", sp, off, v);
  const auto g = std::move(f);
  EXPECT_EQ(g.rows(), 1);
  EXPECT_EQ(f.rows(), 0);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(f.id(), soma::invalid_field);
}

}  // namespace
