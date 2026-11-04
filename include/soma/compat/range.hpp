// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_COMPAT_RANGE_HPP_
#define SOMA_COMPAT_RANGE_HPP_

// MOAB's Range: a sorted set of handles kept as closed runs [first, second].
// The runs live in a vector, so inserting in increasing order (the common
// case) appends, while an insertion in the middle moves the later runs; as in
// MOAB, iterators are invalidated by any change.

#include <cstddef>
#include <iosfwd>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "soma/compat/types.hpp"

namespace soma::compat {

class Range {
 public:
  using value_type = EntityHandle;
  using Pair = std::pair<EntityHandle, EntityHandle>;
  using const_pair_iterator = std::vector<Pair>::const_iterator;

  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = EntityHandle;
    using difference_type = EntityID;
    using pointer = const EntityHandle*;
    using reference = const EntityHandle&;

    const_iterator() = default;
    const EntityHandle& operator*() const { return value_; }
    const EntityHandle* operator->() const { return &value_; }
    const_iterator& operator++() {
      if (value_ < (*runs_)[run_].second) {
        ++value_;
      } else {
        ++run_;
        value_ = run_ < runs_->size() ? (*runs_)[run_].first : 0;
      }
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator old = *this;
      ++*this;
      return old;
    }
    const_iterator& operator--() {
      if (run_ < runs_->size() && value_ > (*runs_)[run_].first) {
        --value_;
      } else {
        --run_;
        value_ = (*runs_)[run_].second;
      }
      return *this;
    }
    const_iterator operator--(int) {
      const_iterator old = *this;
      --*this;
      return old;
    }
    const_iterator& operator+=(EntityID step);
    const_iterator& operator-=(EntityID step);
    bool operator==(const const_iterator& o) const { return run_ == o.run_ && value_ == o.value_; }
    bool operator!=(const const_iterator& o) const { return !(*this == o); }
    // The last and first handles of this iterator's run.
    const_iterator end_of_block() const { return {runs_, run_, (*runs_)[run_].second}; }
    const_iterator start_of_block() const { return {runs_, run_, (*runs_)[run_].first}; }

   private:
    friend class Range;
    friend EntityID operator-(const const_iterator& a, const const_iterator& b);
    const_iterator(const std::vector<Pair>* runs, std::size_t run, EntityHandle value)
        : runs_(runs), run_(run), value_(value) {}
    const std::vector<Pair>* runs_ = nullptr;
    std::size_t run_ = 0;
    EntityHandle value_ = 0;
  };
  using iterator = const_iterator;

  // Keeps the handle it points at, as const_iterator does (std::reverse_iterator
  // would hand out a reference into a temporary).
  class const_reverse_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = EntityHandle;
    using difference_type = EntityID;
    using pointer = const EntityHandle*;
    using reference = const EntityHandle&;

    const_reverse_iterator() = default;
    explicit const_reverse_iterator(const_iterator base) : base_(base) {}
    const EntityHandle& operator*() const {
      const_iterator at = base_;
      value_ = *--at;
      return value_;
    }
    const_reverse_iterator& operator++() {
      --base_;
      return *this;
    }
    const_reverse_iterator operator++(int) {
      const_reverse_iterator old = *this;
      --base_;
      return old;
    }
    const_reverse_iterator& operator--() {
      ++base_;
      return *this;
    }
    const_reverse_iterator operator--(int) {
      const_reverse_iterator old = *this;
      ++base_;
      return old;
    }
    bool operator==(const const_reverse_iterator& o) const { return base_ == o.base_; }
    bool operator!=(const const_reverse_iterator& o) const { return base_ != o.base_; }
    const_iterator base() const { return base_; }

   private:
    const_iterator base_;  // one past the handle pointed at
    mutable EntityHandle value_ = 0;
  };

  Range() = default;
  Range(EntityHandle first, EntityHandle last) { insert(first, last); }

  const_iterator begin() const { return {&runs_, 0, runs_.empty() ? 0 : runs_.front().first}; }
  const_iterator end() const { return {&runs_, runs_.size(), 0}; }
  const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
  const_pair_iterator const_pair_begin() const { return runs_.begin(); }
  const_pair_iterator const_pair_end() const { return runs_.end(); }
  const_pair_iterator pair_begin() const { return runs_.begin(); }
  const_pair_iterator pair_end() const { return runs_.end(); }

  std::size_t size() const;
  std::size_t psize() const { return runs_.size(); }
  bool empty() const { return runs_.empty(); }

  iterator insert(EntityHandle h) { return insert(h, h); }
  iterator insert(EntityHandle first, EntityHandle last);
  iterator insert(iterator /*hint*/, EntityHandle h) { return insert(h, h); }
  iterator insert(iterator /*hint*/, EntityHandle first, EntityHandle last) { return insert(first, last); }
  void insert(const_iterator first, const_iterator last);
  void insert(const Range& r) { merge(r); }
  template <class It>
  iterator insert_list(It first, It last) {
    iterator at = end();
    for (; first != last; ++first) at = insert(static_cast<EntityHandle>(*first));
    return at;
  }
  template <class T>
  iterator insert(const T* first, const T* last) {
    return insert_list(first, last);
  }
  void merge(const Range& r);
  void merge(const_iterator first, const_iterator last) { insert(first, last); }

  iterator erase(iterator it);
  iterator erase(iterator first, iterator last);
  iterator erase(EntityHandle h);

  const EntityHandle& front() const { return runs_.front().first; }
  const EntityHandle& back() const { return runs_.back().second; }
  EntityHandle pop_front();
  EntityHandle pop_back();

  const_iterator find(EntityHandle h) const;
  static const_iterator lower_bound(const_iterator first, const_iterator last, EntityHandle h);
  static const_iterator upper_bound(const_iterator first, const_iterator last, EntityHandle h);
  const_iterator lower_bound(EntityHandle h) const { return lower_bound(begin(), end(), h); }
  const_iterator upper_bound(EntityHandle h) const { return upper_bound(begin(), end(), h); }
  const_iterator lower_bound(EntityType t) const { return lower_bound(make_handle(t, 0)); }
  const_iterator upper_bound(EntityType t) const;
  const_iterator lower_bound(EntityType t, const_iterator first) const {
    return lower_bound(first, end(), make_handle(t, 0));
  }
  const_iterator upper_bound(EntityType t, const_iterator first) const;
  std::pair<const_iterator, const_iterator> equal_range(EntityType t) const {
    return {lower_bound(t), upper_bound(t)};
  }

  bool all_of_type(EntityType t) const;
  bool all_of_dimension(int d) const;
  unsigned num_of_type(EntityType t) const;
  unsigned num_of_dimension(int d) const;
  Range subset_by_type(EntityType t) const;
  Range subset_by_dimension(int d) const;
  bool contains(const Range& other) const;

  // The handle at position i, and the position of h (-1 if absent).
  EntityHandle operator[](EntityID i) const;
  int index(EntityHandle h) const;

  void clear() { runs_.clear(); }
  void swap(Range& o) noexcept { runs_.swap(o.runs_); }
  Range& operator-=(const Range& rhs);

  // MOAB's text: one line per run, "<indent>\t<Type> <id>[ - [<Type> ]<id>]",
  // "<indent>\tempty" when empty.
  std::string str_rep(const char* indent = nullptr) const;
  void print(const char* indent = nullptr) const;
  void print(std::ostream& s, const char* indent = nullptr) const;

  unsigned long get_memory_use() const { return sizeof(*this) + runs_.capacity() * sizeof(Pair); }
  double compactness() const { return empty() ? 1.0 : static_cast<double>(psize()) / static_cast<double>(size()); }
  void sanity_check() const;

  friend bool operator==(const Range& a, const Range& b) { return a.runs_ == b.runs_; }
  friend bool operator!=(const Range& a, const Range& b) { return !(a == b); }

 private:
  std::vector<Pair> runs_;
};

EntityID operator-(const Range::const_iterator& a, const Range::const_iterator& b);
inline Range::const_iterator operator+(Range::const_iterator it, EntityID step) { return it += step; }
inline Range::const_iterator operator-(Range::const_iterator it, EntityID step) { return it -= step; }

Range intersect(const Range& a, const Range& b);
Range subtract(const Range& a, const Range& b);
Range unite(const Range& a, const Range& b);
std::ostream& operator<<(std::ostream& s, const Range& r);

}  // namespace soma::compat

#endif  // SOMA_COMPAT_RANGE_HPP_
