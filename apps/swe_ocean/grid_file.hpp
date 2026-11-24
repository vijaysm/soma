// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_GRID_FILE_HPP_
#define SWE_OCEAN_GRID_FILE_HPP_

// Latitude-longitude grids in NetCDF-4 files (ETOPO1, coastlines), read through HDF5 by window: a rank reads
// only the rows and columns its points need, in strips of a few rows, never the whole grid. open_grid reads
// the two coordinate axes; windows_for turns a latitude-longitude box into the index windows of the file;
// read_window reads one. Without HDF5 (SWE_OCEAN_HAVE_HDF5 unset) open_grid and read_window throw
// std::runtime_error saying HDF5 is needed.
//
// Axes are uniform: latitudes and longitudes of the file are checked to be equally spaced (to a millionth of
// the spacing) and ascending in longitude; latitudes may run either way (south_first records which).

#include <cstddef>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "soma/parallel/comm.hpp"

namespace ocean {

// The file and the names of its variable and axes; the defaults are ETOPO1's (z over y, x).
struct GridSpec {
  std::filesystem::path file;
  std::string variable = "z", lat = "y", lon = "x";
};

// The grid as the program sees it, normalised.
//   lat: degrees, ascending whatever the file's order; lat[i] is the i-th row from the south. south_first
//        records the file's order: true when file row r is lat[r], false when it is lat[nlat - 1 - r].
//   lon: degrees as stored, ascending, with the duplicate periodic column dropped. A grid-registered global
//        file repeats its first meridian in its last column (lon spans 360); the repeat is dropped. A
//        cell-registered global file (lon spans 360 - dlon) keeps every column. Columns are in the file's
//        order, so window column indices are file column indices.
//   periodic: the columns tile the full circle (dlon * lon.size() = 360 to a millionth of dlon): longitudes
//        wrap, and windows_for splits a window that crosses the seam.
//   dlat, dlon: the spacings (degrees, positive).
struct GridInfo {
  std::vector<double> lat, lon;
  bool south_first = true, periodic = false;
  double dlat = 0, dlon = 0;
};

// Rows lat0 .. lat0 + nlat - 1 and columns lon0 .. lon0 + nlon - 1 of the file, in its own index order:
// lat0 counts file rows from the file's first row (the south for south_first, else the north), lon0 file
// columns from its first; lon0 + nlon <= lon.size() (the dropped duplicate column is never read).
struct Window {
  std::size_t lat0, nlat, lon0, nlon;
};

// Reads the axes of the file and checks the variable is (lat, lon) shaped. Collective on comm: rank 0 reads
// and broadcasts, and a failure throws std::runtime_error, with the same message, on every rank (a missing
// file, axis or variable, an axis that is not one-dimensional or uniform, a variable not nlat x nlon (or
// with one more column than the normalised grid: the periodic repeat)).
GridInfo open_grid(const soma::Comm& comm, const GridSpec& spec);

// Reads a window by strips of at most strip_rows rows. visit(i, row) is called once per row in the file's row
// order (south to north when south_first, else north to south), with i the row's index in the normalised
// ascending latitudes (info.lat[i]) and row the nlon values of the window's columns in the file's column
// order (double, converted from the stored type). The row span is valid only during the call. Not collective:
// each rank reads its own windows, opening the file for the call. Throws std::invalid_argument for strip_rows
// 0 or a window outside the grid and std::runtime_error naming the file and variable for a read failure.
// Every value read adds to values_read().
void read_window(const GridSpec& spec, const GridInfo& info, const Window& window, std::size_t strip_rows,
                 const std::function<void(std::size_t, std::span<const double>)>& visit);

// The number of values read from files by this process since it started, summed over read_window calls
// (thread-safe). A test hook: a test reads it before and after to check what a rank really read.
std::size_t values_read();

// The windows covering the box lat_min .. lat_max (degrees), lon_min .. lon_max. Each is the smallest set of
// rows and columns whose coordinates bracket the box (the row at or below lat_min through the row at or
// above lat_max; likewise for columns), so every point of the box lies between two included samples;
// latitudes beyond the grid clamp to its first and last rows. On a periodic grid longitudes wrap: lon_min >
// lon_max (or any range reaching past the file's last column) gives two windows, the first up to the last
// column, the second from column 0 (nothing is covered twice); a range of 360 degrees or more, or one
// needing every column, gives the one window of all columns. On a grid that is not periodic longitudes clamp
// to the file's columns and lon_min > lon_max throws std::invalid_argument. Latitudes: lat_min > lat_max
// throws std::invalid_argument. Not collective.
std::vector<Window> windows_for(const GridInfo& info, double lat_min, double lat_max, double lon_min, double lon_max);

}  // namespace ocean

#endif  // SWE_OCEAN_GRID_FILE_HPP_
