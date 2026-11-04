// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/compat/range.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "soma/compat/cn.hpp"

namespace soma::compat {

namespace {

using Pair = Range::Pair;

// The first run whose last handle is at least h.
std::vector<Pair>::const_iterator run_reaching(const std::vector<Pair>& runs, EntityHandle h) {
  return std::lower_bound(runs.begin(), runs.end(), h, [](const Pair& p, EntityHandle v) { return p.second < v; });
}

}  // namespace

Range::const_iterator& Range::const_iterator::operator+=(EntityID step) {
  if (step < 0) return *this -= -step;
  auto left = static_cast<EntityHandle>(step);
  while (left > 0 && run_ < runs_->size()) {
    const EntityHandle room = (*runs_)[run_].second - value_;
    if (left <= room) {
      value_ += left;
      return *this;
    }
    left -= room + 1;
    ++run_;
    value_ = run_ < runs_->size() ? (*runs_)[run_].first : 0;
  }
  return *this;
}

Range::const_iterator& Range::const_iterator::operator-=(EntityID step) {
  if (step < 0) return *this += -step;
  auto left = static_cast<EntityHandle>(step);
  while (left > 0) {
    if (run_ < runs_->size()) {
      const EntityHandle room = value_ - (*runs_)[run_].first;
      if (left <= room) {
        value_ -= left;
        return *this;
      }
      left -= room + 1;
    } else {
      --left;
    }
    if (run_ == 0) throw std::out_of_range("Range: iterator moved before begin");
    --run_;
    value_ = (*runs_)[run_].second;
  }
  return *this;
}

EntityID operator-(const Range::const_iterator& a, const Range::const_iterator& b) {
  // Positions from the start of the range, run by run.
  const auto position = [](const Range::const_iterator& it) {
    EntityID p = 0;
    for (std::size_t r = 0; r < it.run_ && r < it.runs_->size(); ++r) {
      p += static_cast<EntityID>((*it.runs_)[r].second - (*it.runs_)[r].first + 1);
    }
    if (it.run_ < it.runs_->size()) p += static_cast<EntityID>(it.value_ - (*it.runs_)[it.run_].first);
    return p;
  };
  return position(a) - position(b);
}

std::size_t Range::size() const {
  std::size_t n = 0;
  for (const Pair& p : runs_) n += static_cast<std::size_t>(p.second - p.first + 1);
  return n;
}

Range::iterator Range::insert(EntityHandle first, EntityHandle last) {
  if (first > last) std::swap(first, last);
  // Runs overlapping or touching [first, last] merge with it.
  auto lo = std::lower_bound(runs_.begin(), runs_.end(), first,
                             [](const Pair& p, EntityHandle v) { return v > 0 && p.second < v - 1; });
  auto hi = lo;
  while (hi != runs_.end() && (hi->first <= last || hi->first == last + 1)) ++hi;
  if (lo != hi) {
    first = std::min(first, lo->first);
    last = std::max(last, (hi - 1)->second);
    lo = runs_.erase(lo + 1, hi) - 1;
    *lo = {first, last};
  } else {
    lo = runs_.insert(lo, Pair{first, last});
  }
  const auto r = static_cast<std::size_t>(lo - runs_.begin());
  return {&runs_, r, first};
}

void Range::insert(const_iterator first, const_iterator last) {
  // A run at a time; last may end inside a run.
  while (first != last) {
    const bool last_run = last.run_ == first.run_;
    insert(*first, last_run ? *last - 1 : (*first.runs_)[first.run_].second);
    if (last_run) break;
    ++first.run_;
    first.value_ = first.run_ < first.runs_->size() ? (*first.runs_)[first.run_].first : 0;
  }
}

void Range::merge(const Range& r) {
  if (&r == this) return;
  if (empty()) {
    runs_ = r.runs_;
    return;
  }
  for (const Pair& p : r.runs_) insert(p.first, p.second);
}

Range::iterator Range::erase(iterator it) {
  const EntityHandle h = *it;
  erase(h);
  return lower_bound(h);
}

Range::iterator Range::erase(iterator first, iterator last) {
  if (first == last) return last;
  const EntityHandle a = *first;
  const bool to_end = last == end();
  const EntityHandle b = to_end ? back() : *last - 1;
  *this -= Range(a, b);
  return to_end ? end() : lower_bound(b + 1);
}

Range::iterator Range::erase(EntityHandle h) {
  auto it = run_reaching(runs_, h);
  if (it == runs_.end() || it->first > h) return end();
  const auto r = static_cast<std::size_t>(it - runs_.begin());
  if (it->first == h && it->second == h) {
    runs_.erase(it);
  } else if (it->first == h) {
    ++runs_[r].first;
  } else if (it->second == h) {
    --runs_[r].second;
  } else {
    const Pair upper{h + 1, it->second};
    runs_[r].second = h - 1;
    runs_.insert(runs_.begin() + static_cast<std::ptrdiff_t>(r) + 1, upper);
  }
  return lower_bound(h);
}

EntityHandle Range::pop_front() {
  const EntityHandle h = front();
  erase(h);
  return h;
}

EntityHandle Range::pop_back() {
  const EntityHandle h = back();
  erase(h);
  return h;
}

Range::const_iterator Range::find(EntityHandle h) const {
  const auto it = run_reaching(runs_, h);
  if (it == runs_.end() || it->first > h) return end();
  return {&runs_, static_cast<std::size_t>(it - runs_.begin()), h};
}

Range::const_iterator Range::lower_bound(const_iterator first, const_iterator last, EntityHandle h) {
  const std::vector<Pair>& runs = *first.runs_;
  if (first == last || h <= *first) return first;
  auto it = run_reaching(runs, h);
  const auto r = static_cast<std::size_t>(it - runs.begin());
  const_iterator out = it == runs.end() ? const_iterator{&runs, runs.size(), 0}
                                        : const_iterator{&runs, r, std::max(h, it->first)};
  // Past last means last.
  if (last.run_ < out.run_ || (last.run_ == out.run_ && last.run_ < runs.size() && *last <= *out)) return last;
  return out;
}

Range::const_iterator Range::upper_bound(const_iterator first, const_iterator last, EntityHandle h) {
  return h == ~EntityHandle{0} ? last : lower_bound(first, last, h + 1);
}

Range::const_iterator Range::upper_bound(EntityType t) const {
  return lower_bound(make_handle(static_cast<EntityType>(t + 1), 0));
}

Range::const_iterator Range::upper_bound(EntityType t, const_iterator first) const {
  return lower_bound(first, end(), make_handle(static_cast<EntityType>(t + 1), 0));
}

bool Range::all_of_type(EntityType t) const {
  return empty() || (type_of(front()) == t && type_of(back()) == t);
}

bool Range::all_of_dimension(int d) const {
  return empty() || (CN::Dimension(type_of(front())) == d && CN::Dimension(type_of(back())) == d);
}

unsigned Range::num_of_type(EntityType t) const {
  return static_cast<unsigned>(subset_by_type(t).size());
}

unsigned Range::num_of_dimension(int d) const {
  return static_cast<unsigned>(subset_by_dimension(d).size());
}

Range Range::subset_by_type(EntityType t) const {
  Range out;
  const EntityHandle lo = make_handle(t, 0);
  const EntityHandle hi = make_handle(t, MB_END_ID);
  for (auto it = run_reaching(runs_, lo); it != runs_.end() && it->first <= hi; ++it) {
    out.runs_.emplace_back(std::max(it->first, lo), std::min(it->second, hi));
  }
  return out;
}

Range Range::subset_by_dimension(int d) const {
  if (d < 0 || d > 4) return {};
  const CN::DimensionPair types = CN::TypeDimensionMap[d];
  Range out;
  const EntityHandle lo = make_handle(types.first, 0);
  const EntityHandle hi = make_handle(types.second, MB_END_ID);
  for (auto it = run_reaching(runs_, lo); it != runs_.end() && it->first <= hi; ++it) {
    out.runs_.emplace_back(std::max(it->first, lo), std::min(it->second, hi));
  }
  return out;
}

bool Range::contains(const Range& other) const {
  for (const Pair& p : other.runs_) {
    const auto it = run_reaching(runs_, p.first);
    if (it == runs_.end() || it->first > p.first || it->second < p.second) return false;
  }
  return true;
}

EntityHandle Range::operator[](EntityID i) const {
  auto left = static_cast<EntityHandle>(i);
  for (const Pair& p : runs_) {
    const EntityHandle n = p.second - p.first + 1;
    if (left < n) return p.first + left;
    left -= n;
  }
  throw std::out_of_range("Range: index " + std::to_string(i) + " past the end");
}

int Range::index(EntityHandle h) const {
  EntityHandle before = 0;
  for (const Pair& p : runs_) {
    if (h < p.first) return -1;
    if (h <= p.second) return static_cast<int>(before + (h - p.first));
    before += p.second - p.first + 1;
  }
  return -1;
}

Range& Range::operator-=(const Range& rhs) {
  if (empty() || rhs.empty()) return *this;
  std::vector<Pair> out;
  out.reserve(runs_.size());
  auto b = rhs.runs_.begin();
  for (Pair p : runs_) {
    while (b != rhs.runs_.end() && b->second < p.first) ++b;
    auto c = b;
    while (c != rhs.runs_.end() && c->first <= p.second) {
      if (c->first > p.first) out.emplace_back(p.first, c->first - 1);
      if (c->second >= p.second) {
        p.first = 1;
        p.second = 0;  // used up
        break;
      }
      p.first = c->second + 1;
      ++c;
    }
    if (p.first <= p.second) out.push_back(p);
  }
  runs_.swap(out);
  return *this;
}

std::string Range::str_rep(const char* indent) const {
  std::ostringstream s;
  print(s, indent);
  return s.str();
}

void Range::print(const char* indent) const { print(std::cout, indent); }

void Range::print(std::ostream& s, const char* indent) const {
  const char* pre = indent != nullptr ? indent : "";
  if (empty()) {
    s << pre << "\tempty" << std::endl;
    return;
  }
  for (const Pair& p : runs_) {
    s << pre << '\t' << CN::EntityTypeName(type_of(p.first)) << ' ' << id_of(p.first);
    if (p.second != p.first) {
      s << " - ";
      if (type_of(p.second) != type_of(p.first)) s << CN::EntityTypeName(type_of(p.second)) << ' ';
      s << id_of(p.second);
    }
    s << std::endl;
  }
}

void Range::sanity_check() const {
  for (std::size_t r = 0; r < runs_.size(); ++r) {
    if (runs_[r].first > runs_[r].second) throw std::logic_error("Range: a run ends before it starts");
    if (r > 0 && runs_[r - 1].second + 1 >= runs_[r].first) throw std::logic_error("Range: runs overlap or touch");
  }
}

Range intersect(const Range& a, const Range& b) {
  Range out = a;
  out -= subtract(a, b);
  return out;
}

Range subtract(const Range& a, const Range& b) {
  Range out = a;
  out -= b;
  return out;
}

Range unite(const Range& a, const Range& b) {
  Range out = a;
  out.merge(b);
  return out;
}

std::ostream& operator<<(std::ostream& s, const Range& r) {
  r.print(s);
  return s;
}

}  // namespace soma::compat
