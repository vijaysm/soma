// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// moab_describe <file> [<other>]: loads a mesh file in MOAB and prints what
// it holds, entity by entity: entities with their coordinates or
// connectivity (vertex ids), sets with their flags, contents, children and
// parents, and every tag's values, reals to 12 significant digits. Given two
// files, compares their descriptions instead, numbers to a relative
// tolerance of 1e-9 (MOAB writes VTK reals to 10 digits), prints the first
// lines that differ and exits with 1 when any does.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <moab/CN.hpp>
#include <moab/Core.hpp>

using namespace moab;

namespace {

std::string name_of(Interface& mb, EntityHandle h) {
  return std::string(CN::EntityTypeName(mb.type_from_handle(h))) + " " + std::to_string(mb.id_from_handle(h));
}

std::string real(double x) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.12g", x);
  return buf;
}

void print_value(std::ostream& out, Interface& mb, DataType type, const void* p, int n) {
  for (int i = 0; i < n; ++i) {
    switch (type) {
      case MB_TYPE_INTEGER: out << " " << static_cast<const int*>(p)[i]; break;
      case MB_TYPE_DOUBLE: out << " " << real(static_cast<const double*>(p)[i]); break;
      case MB_TYPE_HANDLE: {
        const EntityHandle h = static_cast<const EntityHandle*>(p)[i];
        out << " " << (h == 0 ? std::string("0") : name_of(mb, h));
        break;
      }
      default: out << " " << static_cast<int>(static_cast<const unsigned char*>(p)[i]);
    }
  }
}

bool same_token(const std::string& a, const std::string& b) {
  if (a == b) return true;
  char* ea = nullptr;
  char* eb = nullptr;
  const double x = std::strtod(a.c_str(), &ea);
  const double y = std::strtod(b.c_str(), &eb);
  if (*ea != '\0' || *eb != '\0' || ea == a.c_str() || eb == b.c_str()) return false;
  return std::fabs(x - y) <= 1e-9 * std::max({1.0, std::fabs(x), std::fabs(y)});
}

bool same_line(const std::string& a, const std::string& b) {
  std::istringstream sa(a), sb(b);
  std::string x, y;
  while (true) {
    const bool ga = static_cast<bool>(sa >> x);
    const bool gb = static_cast<bool>(sb >> y);
    if (ga != gb) return false;
    if (!ga) return true;
    if (!same_token(x, y)) return false;
  }
}

int describe(const char* file, std::ostream& out);

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "usage: moab_describe <file> [<other>]\n";
    return 2;
  }
  if (argc == 2) return describe(argv[1], std::cout);
  std::ostringstream a, b;
  if (describe(argv[1], a) != 0 || describe(argv[2], b) != 0) return 1;
  std::vector<std::string> la, lb;
  std::string line;
  for (std::istringstream in(a.str()); std::getline(in, line);) la.push_back(line);
  for (std::istringstream in(b.str()); std::getline(in, line);) lb.push_back(line);
  int shown = 0;
  for (std::size_t i = 0; i < std::max(la.size(), lb.size()) && shown < 10; ++i) {
    const std::string x = i < la.size() ? la[i] : "<none>";
    const std::string y = i < lb.size() ? lb[i] : "<none>";
    if (same_line(x, y)) continue;
    std::cout << "line " << i + 1 << ":\n  " << argv[1] << ": " << x << "\n  " << argv[2] << ": " << y << "\n";
    ++shown;
  }
  return shown == 0 ? 0 : 1;
}

namespace {

int describe(const char* file, std::ostream& out) {
  Core mb;
  if (mb.load_file(file) != MB_SUCCESS) {
    std::cerr << "moab_describe: cannot load " << file << "\n";
    return 1;
  }
  for (EntityType t = MBVERTEX; t <= MBENTITYSET; ++t) {
    Range r;
    mb.get_entities_by_type(0, t, r);
    if (r.empty()) continue;
    out << CN::EntityTypeName(t) << " count " << r.size() << "\n";
    for (const EntityHandle h : r) {
      out << name_of(mb, h) << ":";
      if (t == MBVERTEX) {
        double x[3];
        mb.get_coords(&h, 1, x);
        out << " " << real(x[0]) << " " << real(x[1]) << " " << real(x[2]);
      } else if (t == MBENTITYSET) {
        unsigned flags = 0;
        mb.get_meshset_options(h, flags);
        std::vector<EntityHandle> contents, children, parents;
        mb.get_entities_by_handle(h, contents);
        mb.get_child_meshsets(h, children);
        mb.get_parent_meshsets(h, parents);
        out << " flags " << (flags & 0x6) << " contents";
        for (const EntityHandle c : contents) out << " " << name_of(mb, c);
        out << " children";
        for (const EntityHandle c : children) out << " " << name_of(mb, c);
        out << " parents";
        for (const EntityHandle c : parents) out << " " << name_of(mb, c);
      } else {
        const EntityHandle* conn = nullptr;
        int n = 0;
        mb.get_connectivity(h, conn, n);
        for (int i = 0; i < n; ++i) out << " " << mb.id_from_handle(conn[i]);
      }
      out << "\n";
    }
  }
  std::vector<Tag> tags;
  mb.tag_get_tags(tags);
  std::vector<std::pair<std::string, Tag>> named;
  for (Tag t : tags) {
    std::string n;
    mb.tag_get_name(t, n);
    named.emplace_back(n, t);
  }
  std::sort(named.begin(), named.end());
  for (const auto& [n, t] : named) {
    DataType type;
    mb.tag_get_data_type(t, type);
    int length = 0;
    const bool varlen = mb.tag_get_length(t, length) == MB_VARIABLE_DATA_LENGTH;
    out << "tag " << n << " type " << type << " length " << (varlen ? -1 : length) << "\n";
    for (EntityType et = MBVERTEX; et <= MBENTITYSET; ++et) {
      Range r;
      mb.get_entities_by_type_and_tag(0, et, &t, nullptr, 1, r);
      for (const EntityHandle h : r) {
        const void* p = nullptr;
        int size = 0;
        if (mb.tag_get_by_ptr(t, &h, 1, &p, &size) != MB_SUCCESS) continue;
        out << "  " << name_of(mb, h) << ":";
        print_value(out, mb, type, p, size);
        out << "\n";
      }
    }
  }
  return 0;
}

}  // namespace
