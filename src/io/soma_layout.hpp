// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_IO_SOMA_LAYOUT_HPP_
#define SOMA_SRC_IO_SOMA_LAYOUT_HPP_

// What the SOMA format's writer and reader share: names, the
// version, the HDF5 types of value types, and the file access and transfer
// properties, MPI-IO and collective with a parallel HDF5 at more than one
// rank.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

#include "hdf5.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/comm.hpp"

namespace soma::soma_file {

inline constexpr std::int64_t format_version = 1;
inline constexpr const char* root = "soma";

// Datasets of every part, concatenated in part order; /soma/parts/offsets
// holds where each part starts in each (columns named in its attribute
// "columns", one per line).
inline std::string dim_group(int d) { return "dim" + std::to_string(d); }
inline std::string field_group(std::size_t i) { return "fields/f" + std::to_string(i); }
inline std::string value_group(std::size_t i) { return "values/v" + std::to_string(i); }

// The HDF5 file type of a value type, little-endian as MOAB writes.
inline hid_t file_type(DataType t) {
  switch (t) {
    case DataType::i8: return H5T_STD_I8LE;
    case DataType::u8: return H5T_STD_U8LE;
    case DataType::i32: return H5T_STD_I32LE;
    case DataType::u32: return H5T_STD_U32LE;
    case DataType::i64: return H5T_STD_I64LE;
    case DataType::u64: return H5T_STD_U64LE;
    case DataType::f32: return H5T_IEEE_F32LE;
    case DataType::f64: return H5T_IEEE_F64LE;
  }
  return H5T_STD_U8LE;
}

inline hid_t memory_type(DataType t) {
  switch (t) {
    case DataType::i8: return H5T_NATIVE_INT8;
    case DataType::u8: return H5T_NATIVE_UINT8;
    case DataType::i32: return H5T_NATIVE_INT32;
    case DataType::u32: return H5T_NATIVE_UINT32;
    case DataType::i64: return H5T_NATIVE_INT64;
    case DataType::u64: return H5T_NATIVE_UINT64;
    case DataType::f32: return H5T_NATIVE_FLOAT;
    case DataType::f64: return H5T_NATIVE_DOUBLE;
  }
  return H5T_NATIVE_UINT8;
}

// Bytes of one value of type t.
inline std::size_t value_size(DataType t) {
  std::size_t n = 0;
  dispatch_data_type(t, [&]<class T>() { n = sizeof(T); });
  return n;
}

// True when the HDF5 is parallel: files at more than one rank are then
// opened with MPI-IO and moved with collective transfers.
constexpr bool parallel_hdf5() {
#ifdef H5_HAVE_PARALLEL
  return true;
#else
  return false;
#endif
}

// Why the hints and aggregator count cannot be used at `ranks` ranks, or
// nothing.
inline std::string access_error(const MpiIoHints& h, int aggregators, int ranks) {
  if (aggregators < 0 || aggregators > ranks) {
    return "aggregators must be in [0, " + std::to_string(ranks) + "], not " + std::to_string(aggregators);
  }
  if (aggregators > 1 && !parallel_hdf5()) return "more than one aggregator needs a parallel HDF5";
  if (h.cb_nodes < 0 || h.striping_factor < 0) return "a negative count in the MPI-IO hints";
  if ((h.alignment & (h.alignment - 1)) != 0) {
    return "alignment " + std::to_string(h.alignment) + " is not a power of two";
  }
  return "";
}

// The first rank of the group of rank r, of `m` aggregators over `p` ranks:
// group floor(rm/p), whose first rank is ceil(gp/m). A group's ranks are
// consecutive, so its parts' rows are contiguous in every dataset.
inline int leader_of(int r, int m, int p) {
  const auto g = static_cast<std::int64_t>(r) * m / p;
  return static_cast<int>((g * p + m - 1) / m);
}

// A hash of the options every rank must give alike.
inline std::uint64_t access_hash(const MpiIoHints& h, int aggregators) {
  std::uint64_t x = 1469598103934665603ULL;
  for (const std::uint64_t v : {static_cast<std::uint64_t>(aggregators), static_cast<std::uint64_t>(h.cb_nodes),
                                static_cast<std::uint64_t>(h.cb_buffer_size),
                                static_cast<std::uint64_t>(h.collective_buffering ? 1 + *h.collective_buffering : 0),
                                static_cast<std::uint64_t>(h.striping_factor),
                                static_cast<std::uint64_t>(h.striping_unit),
                                static_cast<std::uint64_t>(h.alignment)}) {
    x = (x ^ v) * 1099511628211ULL;
  }
  return x;
}

// A collective (MPI-IO) file over `comm` when `collective`, else the default
// driver; metadata is read and written collectively too unless
// `independent_metadata` (when ranks open different datasets). The hints go
// to MPI-IO; the alignment applies with either driver. Objects are created
// in the HDF5 1.14 file format, whatever newer library writes them.
inline h5::Id file_access(const Comm& comm, bool collective, bool independent_metadata = false,
                          const MpiIoHints& hints = {}) {
  h5::Id fapl = h5::check(H5Pcreate(H5P_FILE_ACCESS), "make a file access property list");
  h5::check(H5Pset_libver_bounds(fapl.get(), H5F_LIBVER_V114, H5F_LIBVER_LATEST), "use the HDF5 1.14 format");
  if (hints.alignment > 1) {
    h5::check(H5Pset_alignment(fapl.get(), hints.alignment, hints.alignment), "align datasets");
  }
#ifdef H5_HAVE_PARALLEL
  if (collective) {
    MPI_Info info = MPI_INFO_NULL;
    const auto set = [&](const char* key, const std::string& value) {
      if (info == MPI_INFO_NULL) MPI_Info_create(&info);
      MPI_Info_set(info, key, value.c_str());
    };
    if (hints.cb_nodes > 0) set("cb_nodes", std::to_string(hints.cb_nodes));
    if (hints.cb_buffer_size > 0) set("cb_buffer_size", std::to_string(hints.cb_buffer_size));
    if (hints.collective_buffering) {
      const char* v = *hints.collective_buffering ? "enable" : "disable";
      set("romio_cb_write", v);
      set("romio_cb_read", v);
    }
    if (hints.striping_factor > 0) set("striping_factor", std::to_string(hints.striping_factor));
    if (hints.striping_unit > 0) set("striping_unit", std::to_string(hints.striping_unit));
    const herr_t mpio = H5Pset_fapl_mpio(fapl.get(), comm.native(), info);  // copies the info
    if (info != MPI_INFO_NULL) MPI_Info_free(&info);
    h5::check(mpio, "use MPI-IO");
    if (!independent_metadata) {
      h5::check(H5Pset_all_coll_metadata_ops(fapl.get(), true), "read metadata collectively");
      h5::check(H5Pset_coll_metadata_write(fapl.get(), true), "write metadata collectively");
    }
  }
#else
  (void)comm;
  (void)collective;
  (void)independent_metadata;
#endif
  return fapl;
}

// Collective transfers when `collective`, else the default.
inline h5::Id transfer(bool collective) {
  h5::Id dxpl = h5::check(H5Pcreate(H5P_DATASET_XFER), "make a transfer property list");
#ifdef H5_HAVE_PARALLEL
  if (collective) h5::check(H5Pset_dxpl_mpio(dxpl.get(), H5FD_MPIO_COLLECTIVE), "transfer collectively");
#else
  (void)collective;
#endif
  return dxpl;
}

// Reads of a file opened once over more than one rank with MPI-IO: the
// first rank alone reads its metadata (the file is opened with metadata read
// independently, since HDF5 cannot read it independently on one rank of a
// file opened for collective metadata reads), and every rank its rows,
// straight from the file in one collective MPI-IO read where a dataset is
// stored as the rows read.

// Where a dataset's rows are: `raw` when they are stored at `at` as read in
// memory type `memory` (contiguous, allocated, of that type).
struct Location {
  bool raw = false;
  std::uint64_t at = 0;
};

inline Location location(hid_t dataset, hid_t memory, const std::string& where) {
  const h5::Id dcpl = h5::check(H5Dget_create_plist(dataset), "get the creation properties of " + where);
  const h5::Id type = h5::check(H5Dget_type(dataset), "get the type of " + where);
  const haddr_t at = H5Dget_offset(dataset);
  const bool raw = H5Pget_layout(dcpl.get()) == H5D_CONTIGUOUS && H5Pget_external_count(dcpl.get()) == 0 &&
                   H5Tequal(type.get(), memory) > 0 && at != HADDR_UNDEF;
  return {raw, raw ? static_cast<std::uint64_t>(at) : 0};
}

// Bytes of a file to read: `bytes` at offset `at` into `to`.
struct Piece {
  std::uint64_t at = 0, bytes = 0;
  std::byte* to = nullptr;
};

// The pieces, in one collective MPI-IO read on the file HDF5 opened with
// MPI-IO; every rank calls it, with pieces or none. Throws h5::Error when
// MPI-IO fails, std::runtime_error when the file is shorter than the pieces.
inline void read_raw(hid_t file, std::vector<Piece> pieces) {
#ifdef H5_HAVE_PARALLEL
  void* handle = nullptr;
  h5::check(H5Fget_vfd_handle(file, H5P_DEFAULT, &handle), "get the MPI-IO file");
  MPI_File fh = *static_cast<MPI_File*>(handle);
  std::sort(pieces.begin(), pieces.end(), [](const Piece& x, const Piece& y) { return x.at < y.at; });
  constexpr std::uint64_t most = std::uint64_t{1} << 30;  // bytes a block's int length holds
  std::vector<int> lengths;
  std::vector<MPI_Aint> at, to;
  std::uint64_t total = 0;
  for (const Piece& p : pieces) {
    for (std::uint64_t k = 0; k < p.bytes; k += most) {
      lengths.push_back(static_cast<int>(std::min(most, p.bytes - k)));
      at.push_back(static_cast<MPI_Aint>(p.at + k));
      MPI_Aint a = 0;
      MPI_Get_address(p.to + k, &a);
      to.push_back(a);
    }
    total += p.bytes;
  }
  const int n = static_cast<int>(lengths.size());
  MPI_Datatype file_view = MPI_BYTE, memory = MPI_BYTE;
  if (n > 0) {
    MPI_Type_create_hindexed(n, lengths.data(), at.data(), MPI_BYTE, &file_view);
    MPI_Type_commit(&file_view);
    MPI_Type_create_hindexed(n, lengths.data(), to.data(), MPI_BYTE, &memory);
    MPI_Type_commit(&memory);
  }
  // A failed view still joins the read, reading nothing; the view HDF5
  // expects is restored after.
  const bool viewed = MPI_File_set_view(fh, 0, MPI_BYTE, file_view, "native", MPI_INFO_NULL) == MPI_SUCCESS;
  MPI_Status status;
  const bool read = MPI_File_read_at_all(fh, 0, n > 0 ? MPI_BOTTOM : nullptr, viewed && n > 0 ? 1 : 0, memory,
                                         &status) == MPI_SUCCESS;
  const bool reset = MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL) == MPI_SUCCESS;
  MPI_Count got = 0;
  if (read) MPI_Get_elements_x(&status, MPI_BYTE, &got);
  if (n > 0) {
    MPI_Type_free(&file_view);
    MPI_Type_free(&memory);
  }
  if (!viewed || !read || !reset) throw h5::Error("HDF5 could not read the datasets' rows with MPI-IO");
  if (static_cast<std::uint64_t>(got) != total) throw std::runtime_error("the file is shorter than its datasets");
#else
  (void)file;
  (void)pieces;
#endif
}

// Rank 0's message, given to every rank: in one broadcast of first_bcast
// bytes, and a second for what does not fit.
inline constexpr std::size_t first_bcast = std::size_t{64} << 10;

inline std::vector<std::byte> broadcast(const Comm& comm, std::vector<std::byte> message) {
  constexpr std::size_t room = first_bcast - sizeof(std::uint64_t);
  std::vector<std::byte> head(first_bcast);
  std::uint64_t n = message.size();
  if (comm.rank() == 0) {
    std::memcpy(head.data(), &n, sizeof n);
    std::copy_n(message.begin(), std::min<std::size_t>(n, room), head.begin() + sizeof n);
  }
  comm.bcast(std::span<std::byte>(head), 0);
  std::memcpy(&n, head.data(), sizeof n);
  if (comm.rank() != 0) {
    const auto begin = head.begin() + sizeof n;
    message.assign(begin, begin + static_cast<std::ptrdiff_t>(std::min<std::size_t>(n, room)));
    message.resize(n);
  }
  if (n > room) comm.bcast(std::span<std::byte>(message).subspan(room), 0);
  return message;
}

}  // namespace soma::soma_file

#endif  // SOMA_SRC_IO_SOMA_LAYOUT_HPP_
