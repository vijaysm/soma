// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.
// avtSOMAOptions.C -- the SOMA plugin's options on opening a database.

#include "avtSOMAOptions.h"

#include <DBOptionsAttributes.h>

const char *const SOMAOptions::kCloseHanging = "Close cells at hanging vertices";

DBOptionsAttributes *GetSOMAReadOptions(void) {
  DBOptionsAttributes *rv = new DBOptionsAttributes;
  rv->SetBool(SOMAOptions::kCloseHanging, true);
  return rv;
}

DBOptionsAttributes *GetSOMAWriteOptions(void) { return new DBOptionsAttributes; }
