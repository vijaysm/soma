// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_UNIT_MESH_TEXT_HPP_
#define SOMA_TESTS_UNIT_MESH_TEXT_HPP_

// A mesh's content keyed by gids, independent of local ids, for comparing a
// mesh with what a file read back gives: coordinates, entities, sets, fields
// and mesh values.

#include <array>
#include <cstddef>
#include <cstring>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/mesh_values.hpp"
#include "soma/core/sets.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/fields/space.hpp"
#include "soma/fields/tag_format.hpp"

namespace soma_test {

using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Topology;

// Entities of one dimension by gid: topology and vertex gids.
struct Entity {
  Topology type = Topology::vertex;
  std::vector<GO> vertices;
  friend bool operator==(const Entity&, const Entity&) = default;
};

inline std::map<GO, Entity> entities(const HostMesh& m, int d) {
  std::map<GO, Entity> out;
  if (d > m.dim()) return out;
  const auto gid = m.gid(d);
  const auto g0 = m.gid(0);
  for (std::size_t e = 0; e < static_cast<std::size_t>(m.count(d)); ++e) {
    Entity x;
    if (d > 0) {
      x.type = static_cast<Topology>(m.topology(d)(e));
      const auto off = m.conn_offsets(d);
      for (auto k = off(e); k < off(e + 1); ++k) x.vertices.push_back(g0(static_cast<std::size_t>(m.conn(d)(static_cast<std::size_t>(k)))));
    }
    out[gid(e)] = x;
  }
  return out;
}

// Sets by gid: flags, members by slot and gid (in order for an ordered set),
// children and parents by gid.
struct SetText {
  std::uint8_t flags = 0;
  std::vector<std::pair<int, GO>> members;
  std::vector<GO> children, parents;
  friend bool operator==(const SetText&, const SetText&) = default;
};

inline std::map<GO, SetText> sets_of(const HostMesh& m) {
  std::map<GO, SetText> out;
  const auto& sets = m.sets();
  const auto gid_of = [&](int slot, LO l) {
    return slot == soma::set_dim ? sets.gid()(static_cast<std::size_t>(l)) : m.gid(slot)(static_cast<std::size_t>(l));
  };
  for (LO s = 0; s < sets.count(); ++s) {
    SetText t;
    t.flags = sets.flags(s);
    if (sets.ordered(s)) {
      for (const auto& e : sets.sequence(s)) t.members.emplace_back(e.slot, gid_of(e.slot, e.lid));
    } else {
      for (int slot = 0; slot < soma::num_field_slots; ++slot) {
        if (slot != soma::set_dim && slot > m.dim()) continue;
        for (const LO l : sets.members_of(s, slot)) t.members.emplace_back(slot, gid_of(slot, l));
      }
      std::sort(t.members.begin(), t.members.end());
    }
    for (const LO x : sets.children(s)) t.children.push_back(gid_of(soma::set_dim, x));
    for (const LO x : sets.parents(s)) t.parents.push_back(gid_of(soma::set_dim, x));
    out[sets.gid()(static_cast<std::size_t>(s))] = t;
  }
  return out;
}

// Every field by name and slot: its values by entity gid, default, global
// value and file format, as bytes.
struct FieldText {
  soma::DataType type = soma::DataType::u8;
  int ncomp = 0;
  bool handle = false;
  soma::TagFormat format{};
  std::optional<std::vector<std::byte>> def, global;
  std::map<GO, std::vector<std::byte>> values;
};

template <class T>
std::vector<std::byte> value_bytes(std::span<const T> v) {
  std::vector<std::byte> out(v.size_bytes());
  if (!v.empty()) std::memcpy(out.data(), v.data(), out.size());
  return out;
}

inline std::map<std::pair<std::string, int>, FieldText> fields_of(const HostMesh& m) {
  std::map<std::pair<std::string, int>, FieldText> out;
  const auto gid_of = [&](int slot, LO l) {
    return slot == soma::set_dim ? m.sets().gid()(static_cast<std::size_t>(l)) : m.gid(slot)(static_cast<std::size_t>(l));
  };
  const auto opt = [](const auto& v) -> std::optional<std::vector<std::byte>> {
    if (!v) return std::nullopt;
    return value_bytes(std::span(*v));
  };
  for (const soma::FieldInfo& info : m.fields().list()) {
    soma::dispatch_data_type(info.type, [&]<class T>() {
      for (const int slot : info.dims) {
        FieldText& t = out[{info.name, slot}];
        t.type = info.type;
        t.ncomp = info.ncomp;
        t.handle = info.handle;
        t.format = m.fields().tag_format(info.family);
        const LO n = slot == soma::set_dim ? m.sets().count() : m.count(slot);
        if (info.variable) {
          const auto f = m.fields().get_var<T>(info.id);
          t.def = opt(f.default_value());
          t.global = opt(f.global_value());
          for (LO l = 0; l < n; ++l) {
            if (f.space().entity_rows(slot, l).valid()) t.values[gid_of(slot, l)] = value_bytes(f.value_or_default(slot, l));
          }
          continue;
        }
        const auto f = m.fields().get<T>(info.id);
        if (f.has_default()) t.def = value_bytes(std::span<const T>(f.default_value()));
        if (f.has_global()) t.global = value_bytes(std::span<const T>(f.global_value()));
        for (LO l = 0; l < n; ++l) {
          const auto r = f.space().entity_rows(slot, l);
          if (!r.valid()) continue;
          std::vector<T> row;
          for (int c = 0; c < info.ncomp; ++c) row.push_back(f.view()(static_cast<std::size_t>(r.begin), static_cast<std::size_t>(c)));
          t.values[gid_of(slot, l)] = value_bytes(std::span<const T>(row));
        }
      }
    });
  }
  return out;
}

// The first difference between two field listings, or "".
inline std::string difference(const std::map<std::pair<std::string, int>, FieldText>& a,
                       const std::map<std::pair<std::string, int>, FieldText>& b) {
  for (const auto& [k, x] : a) {
    const std::string at = k.first + " on " + std::to_string(k.second) + ": ";
    const auto it = b.find(k);
    if (it == b.end()) return at + "not in the second";
    const FieldText& y = it->second;
    if (x.type != y.type || x.ncomp != y.ncomp || x.handle != y.handle) return at + "type";
    if (!(x.format == y.format)) {
      return at + "format " + std::to_string(x.format.storage_class) + "/" +
             std::to_string(static_cast<int>(x.format.kind)) + "/" + std::to_string(x.format.nbits) + " vs " +
             std::to_string(y.format.storage_class) + "/" + std::to_string(static_cast<int>(y.format.kind)) + "/" +
             std::to_string(y.format.nbits);
    }
    if (x.def != y.def) return at + "default";
    if (x.global != y.global) return at + "global";
    if (x.values.size() != y.values.size()) {
      return at + std::to_string(x.values.size()) + " vs " + std::to_string(y.values.size()) + " values";
    }
    for (const auto& [g, v] : x.values) {
      const auto w = y.values.find(g);
      if (w == y.values.end()) return at + "no value on gid " + std::to_string(g);
      if (w->second != v) {
        std::string text = at + "gid " + std::to_string(g) + ":";
        for (std::size_t i = 0; i < v.size(); ++i) {
          text += " " + std::to_string(static_cast<int>(v[i])) + "/" + std::to_string(static_cast<int>(w->second[i]));
        }
        return text;
      }
    }
  }
  for (const auto& [k, x] : b) {
    if (!a.contains(k)) return k.first + " on " + std::to_string(k.second) + ": not in the first";
  }
  return "";
}

inline LO lid_of(const HostMesh& m, int d, GO gid) {
  for (std::size_t i = 0; i < static_cast<std::size_t>(m.count(d)); ++i) {
    if (m.gid(d)(i) == gid) return static_cast<LO>(i);
  }
  return -1;
}

inline std::map<GO, std::array<double, 3>> coordinates(const HostMesh& m) {
  std::map<GO, std::array<double, 3>> out;
  const auto xyz = m.coords();
  for (std::size_t v = 0; v < static_cast<std::size_t>(m.count(0)); ++v) {
    out[m.gid(0)(v)] = {xyz(v, 0), xyz(v, 1), xyz(v, 2)};
  }
  return out;
}

// The first difference between two meshes, or "": dimension, input marks,
// coordinates, entities, sets, fields and mesh values, all by gid.
inline std::string mesh_difference(const HostMesh& a, const HostMesh& b) {
  if (a.dim() != b.dim()) return "dimension";
  if (coordinates(a) != coordinates(b)) return "coordinates";
  for (int d = 1; d <= a.dim(); ++d) {
    if (a.input(d) != b.input(d)) return "input mark of dimension " + std::to_string(d);
    if (entities(a, d) != entities(b, d)) return "entities of dimension " + std::to_string(d);
  }
  if (sets_of(a) != sets_of(b)) return "sets";
  if (const std::string f = difference(fields_of(a), fields_of(b)); !f.empty()) return f;
  if (a.values().names() != b.values().names()) return "mesh value names";
  for (const std::string& n : a.values().names()) {
    const soma::MeshValue& x = a.values().at(n);
    const soma::MeshValue& y = b.values().at(n);
    if (!(x.type == y.type && x.ncomp == y.ncomp && x.handle == y.handle && x.format == y.format &&
          x.default_value == y.default_value && x.global_value == y.global_value)) {
      return "mesh value " + n;
    }
  }
  return "";
}

}  // namespace soma_test

#endif  // SOMA_TESTS_UNIT_MESH_TEXT_HPP_
