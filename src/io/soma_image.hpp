// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_IO_SOMA_IMAGE_HPP_
#define SOMA_SRC_IO_SOMA_IMAGE_HPP_

// What a SOMA file holds, laid out in memory before the file is touched, and
// the writer every kind of SOMA file shares: one exscan places each part's
// rows, then the ranks write collectively over MPI-IO (or through rank 0
// with a serial HDF5) into a temporary that rank 0 renames over the path.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "soma/fields/data_type.hpp"
#include "soma/parallel/comm.hpp"
#include "soma_layout.hpp"

namespace soma::soma_file {

// A dataset of rows of `width` values of `type`: `data` holds this rank's
// rows (a column, written by every part) or every row (written once, the
// same on every rank).
struct Dataset {
  std::string path;  // below /soma
  DataType type = DataType::u8;
  std::size_t width = 1;
  std::vector<std::byte> data;

  [[nodiscard]] std::size_t row_bytes() const { return width * soma_file::value_size(type); }
  [[nodiscard]] std::uint64_t rows() const { return data.size() / row_bytes(); }
};

// An attribute the same on every rank: text, or an array of i64 or u8.
struct Attribute {
  std::string object;  // below /soma; "" for /soma
  std::string name;
  std::string text;
  std::vector<std::int64_t> ints;
  std::vector<std::uint8_t> bytes;
  enum class Kind : std::uint8_t { text, ints, bytes } kind = Kind::ints;
};

// Everything the file holds, laid out before it is touched.
struct Image {
  std::vector<std::string> groups;  // below /soma, parents first
  // Deques, so that the references column() and single() give stay valid.
  std::deque<Dataset> columns;  // written by every part, in part order
  std::deque<Dataset> once;     // the same on every rank, written by rank 0
  std::vector<Attribute> attributes;

  Dataset& column(std::string path, DataType type, std::size_t width = 1) {
    columns.push_back({std::move(path), type, width, {}});
    return columns.back();
  }
  Dataset& single(std::string path, DataType type, std::size_t width = 1) {
    once.push_back({std::move(path), type, width, {}});
    return once.back();
  }
  void ints(std::string object, std::string name, std::vector<std::int64_t> v) {
    Attribute a;
    a.object = std::move(object);
    a.name = std::move(name);
    a.ints = std::move(v);
    attributes.push_back(std::move(a));
  }
  void text(std::string object, std::string name, std::string v) {
    Attribute a;
    a.object = std::move(object);
    a.name = std::move(name);
    a.text = std::move(v);
    a.kind = Attribute::Kind::text;
    attributes.push_back(std::move(a));
  }
  void bytes(std::string object, std::string name, std::span<const std::byte> v) {
    if (v.empty()) return;  // HDF5 has no empty attribute; the meta array says it is there
    Attribute a;
    a.object = std::move(object);
    a.name = std::move(name);
    a.bytes.resize(v.size());
    std::memcpy(a.bytes.data(), v.data(), v.size());
    a.kind = Attribute::Kind::bytes;
    attributes.push_back(std::move(a));
  }
};

// Writes `img` to `path` whole or not at all, collectively over comm; I/O
// failures throw std::runtime_error naming `writer` and the path on every
// rank. `compression` is a checked deflate level.
void write_image(const Comm& comm, const std::filesystem::path& path, Image& img, const SomaWriteOptions& options,
                 const char* writer);

}  // namespace soma::soma_file

#endif  // SOMA_SRC_IO_SOMA_IMAGE_HPP_
