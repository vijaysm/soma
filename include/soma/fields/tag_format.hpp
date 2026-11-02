// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_TAG_FORMAT_HPP_
#define SOMA_FIELDS_TAG_FORMAT_HPP_

#include <cstdint>

namespace soma {

// How a legacy .h5m file stores a family's tag: MOAB's storage class (0 bit,
// 1 sparse, 2 dense, 3 mesh; -1 when no file gave one) and the kind of its
// values. Bit values are u8 with the low nbits used (1 to 8); opaque values
// are u8, one component per byte. read_h5m records it, write_h5m restores it.
struct TagFormat {
  enum class Kind : std::uint8_t { plain, bit, opaque };
  int storage_class = -1;
  Kind kind = Kind::plain;
  int nbits = 0;  // bit tags only
  friend bool operator==(const TagFormat&, const TagFormat&) = default;
};

}  // namespace soma

#endif  // SOMA_FIELDS_TAG_FORMAT_HPP_
