// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_GOLDEN_SERIAL_DERIVATION_HPP_
#define SOMA_TESTS_GOLDEN_SERIAL_DERIVATION_HPP_

// Serial reference derivation of unique sub-entities (edges, faces), kept to
// check the parallel one: a hash map keyed by the sorted vertex ids, filled in
// element order. Ported from moabX (src/mesh/subentity_derivation.cpp).
//
// Orientation: the first element to insert a sub-entity fixes its vertex
// order. Later elements report sense +1 if their traversal agrees with that
// order and -1 if reversed. A conforming interior face sees one +1 and one -1;
// a boundary face exactly one incidence.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"

namespace soma_golden {

using soma::LO;

// One (element, local side) reference to a derived sub-entity.
struct Incidence {
  LO element = 0;
  std::uint8_t local_index = 0;
  std::int8_t sense = 0;  // +1 as stored, -1 reversed, 0 no match
};

struct Derived {
  std::vector<LO> connectivity{};
  std::vector<LO> offsets{0};
  std::vector<Incidence> incidences{};
  std::vector<LO> incidence_offsets{0};
  // Elements without a table for this dimension (polyhedra), or with too few
  // vertices for their topology.
  std::vector<LO> unsupported_elements{};

  std::size_t count() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }
  std::span<const LO> vertices(LO id) const noexcept {
    const auto b = static_cast<std::size_t>(offsets[static_cast<std::size_t>(id)]);
    const auto e = static_cast<std::size_t>(offsets[static_cast<std::size_t>(id) + 1]);
    return {connectivity.data() + b, e - b};
  }
  std::span<const Incidence> incidences_of(LO id) const noexcept {
    const auto b = static_cast<std::size_t>(incidence_offsets[static_cast<std::size_t>(id)]);
    const auto e = static_cast<std::size_t>(incidence_offsets[static_cast<std::size_t>(id) + 1]);
    return {incidences.data() + b, e - b};
  }
};

struct Elements {
  std::span<const LO> offsets{};
  std::span<const LO> indices{};
  std::span<const soma::Topology> topologies{};

  std::size_t count() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }
  std::span<const LO> connectivity(LO e) const noexcept {
    const auto b = static_cast<std::size_t>(offsets[static_cast<std::size_t>(e)]);
    const auto n = static_cast<std::size_t>(offsets[static_cast<std::size_t>(e) + 1]) - b;
    return {indices.data() + b, n};
  }
};

// All unique sub-entities of sub_dimension (1 = edges, 2 = faces), numbered in
// first-encounter order.
Derived derive(const Elements& elements, int sub_dimension);

}  // namespace soma_golden

#endif  // SOMA_TESTS_GOLDEN_SERIAL_DERIVATION_HPP_
