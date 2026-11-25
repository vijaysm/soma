// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.
// avtSOMAOptions.h -- the SOMA plugin's options on opening a database.

#ifndef AVT_SOMA_OPTIONS_H
#define AVT_SOMA_OPTIONS_H

class DBOptionsAttributes;

namespace SOMAOptions {
// Close cells with hanging vertices (in 3D, as several zones each); off: cells as stored, one zone each.
extern const char *const kCloseHanging;
}

// The caller owns what these return.
DBOptionsAttributes *GetSOMAReadOptions(void);
DBOptionsAttributes *GetSOMAWriteOptions(void);

#endif
