/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2022-2026, Vijay Mahadevan. */

/// @file
/// The C API: SOMA's operations behind opaque handles, for C callers and the
/// Fortran module.
///
/// Conventions:
/// - Every function returns a status: SOMA_OK (0) or one of the SOMA_ERR_*
///   codes. soma_last_error() gives the calling thread's last message. No C++
///   exception crosses this interface.
/// - Ids, CSR offsets and segment bounds are 0-based, as in C++. (The Fortran
///   module shifts them to 1-based.) Gids are global and never shift.
/// - Handles are reference counted: soma_<h>_clone() gives a second handle on
///   the same object, and the object lives until its last handle is freed.
///   Handles built from a mesh (edits, halo exchanges, remap operators) keep
///   the mesh alive too. soma_<h>_free(NULL) is a no-op.
/// - A collective call checks its arguments on every rank and agrees on the
///   outcome: when one rank's arguments are bad, every rank returns an error,
///   and none enters the collective step alone.
/// - An array of count 0 may be NULL.
/// - Pointers into a mesh's storage (coordinates, connectivity, gids, fields)
///   stay valid until a call changes that mesh's topology or numbering (edit
///   apply, compress, ghosts, adjacencies) or frees it; field pointers also
///   until a field of the mesh is added or removed.

#ifndef SOMA_C_SOMA_H_
#define SOMA_C_SOMA_H_

#include <stddef.h>
#include <stdint.h>

#include "soma/c/soma_config.h"

#if SOMA_C_HAVE_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if SOMA_C_LO_BYTES == 8
typedef int64_t soma_lo_t;
#else
typedef int32_t soma_lo_t;
#endif
#if SOMA_C_OFFSET_BYTES == 8
typedef int64_t soma_offset_t;
#else
typedef int32_t soma_offset_t;
#endif
typedef int64_t soma_go_t;

typedef struct soma_comm soma_comm_t;
typedef struct soma_builder soma_builder_t;
typedef struct soma_mesh soma_mesh_t;
typedef struct soma_edit soma_edit_t;
typedef struct soma_renumbering soma_renumbering_t;
typedef struct soma_halo soma_halo_t;
typedef struct soma_refine soma_refine_t;
typedef struct soma_remap soma_remap_t;

/* Status codes. */
enum {
  SOMA_OK = 0,
  SOMA_ERR_INVALID_ARGUMENT = 1, /**< a bad argument (C++ std::invalid_argument) */
  SOMA_ERR_OUT_OF_RANGE = 2,     /**< an index or name not found (std::out_of_range) */
  SOMA_ERR_TYPE = 3,             /**< the wrong data type */
  SOMA_ERR_STATE = 4,            /**< a null, consumed or stale handle */
  SOMA_ERR_RUNTIME = 5           /**< any other failure */
};

/* Entity shapes (soma::Topology). */
enum {
  SOMA_VERTEX = 0,
  SOMA_EDGE = 1,
  SOMA_TRI = 2,
  SOMA_QUAD = 3,
  SOMA_POLYGON = 4,
  SOMA_TET = 5,
  SOMA_PYRAMID = 6,
  SOMA_PRISM = 7,
  SOMA_KNIFE = 8,
  SOMA_HEX = 9,
  SOMA_POLYHEDRON = 10
};

/* Entity orderings of a built mesh (soma::Ordering); DEFAULT is the library's. */
enum {
  SOMA_ORDERING_DEFAULT = -1,
  SOMA_ORDERING_FILE = 0,
  SOMA_ORDERING_TYPE_MAJOR_SFC = 1,
  SOMA_ORDERING_TILED = 2
};

/* Ownership classes of a segment (soma::Ownership). */
enum { SOMA_INTERIOR = 0, SOMA_SHARED = 1, SOMA_SHARED_COPY = 2, SOMA_GHOST = 3 };

/* Field value types (soma::DataType). */
enum {
  SOMA_I8 = 0,
  SOMA_U8 = 1,
  SOMA_I32 = 2,
  SOMA_U32 = 3,
  SOMA_I64 = 4,
  SOMA_U64 = 5,
  SOMA_F32 = 6,
  SOMA_F64 = 7
};

/* Reductions. */
enum { SOMA_SUM = 0, SOMA_MIN = 1, SOMA_MAX = 2, SOMA_PROD = 3 };

/* File formats. */
enum {
  SOMA_FORMAT_LEGACY_H5M = 0,
  SOMA_FORMAT_SOMA = 1,
  SOMA_FORMAT_SOMA_AMR = 2,
  SOMA_FORMAT_LEGACY_VTK = 3,
  SOMA_FORMAT_UNKNOWN = 4
};

/* Remap options. */
enum { SOMA_SEARCH_GRID = 0, SOMA_SEARCH_ARBORX = 1 };
enum { SOMA_SURFACE_PLANE = 0, SOMA_SURFACE_SPHERE = 1 };
enum { SOMA_SPHERE_CLIP_GNOMONIC = 0, SOMA_SPHERE_CLIP_GREAT_CIRCLE = 1 };

/* Marked refinement: how new vertices on the boundary between ranks match. */
enum { SOMA_VERTICES_RECORDED = 0, SOMA_VERTICES_COORDINATES = 1 };

/* Refinement maps. */
enum { SOMA_PARENT = 0, SOMA_CHILDREN = 1, SOMA_HANGING = 2 };
enum { SOMA_CHILD_OFFSETS = 0, SOMA_HANGING_OFFSETS = 1 };

/* ---- Runtime ---------------------------------------------------------- */

/// Copies the calling thread's last error message into buf (truncated, always
/// NUL-terminated when len > 0); returns the message's full length.
size_t soma_last_error(char* buf, size_t len);

/// Initializes Kokkos (host) unless it is initialized; MPI is the caller's.
int soma_initialize(void);
/// Finalizes Kokkos when soma_initialize initialized it. Objects still alive
/// then leave their memory to the operating system; freeing them is safe.
int soma_finalize(void);

/// Test hooks: the number of live handles, and a failure of the given status.
int soma_testing_live_handles(long* count);
int soma_testing_raise(int status);

/* ---- Communicators ---------------------------------------------------- */

#if SOMA_C_HAVE_MPI
/// A handle on comm, which the caller keeps alive (the handle does not own it).
int soma_comm_create(MPI_Comm comm, soma_comm_t** out);
#endif
/// The same from a Fortran MPI handle (MPI_Comm_f2c); needs an MPI build.
int soma_comm_create_f(int fcomm, soma_comm_t** out);
/// Every rank (MPI_COMM_WORLD), or the one rank without MPI.
int soma_comm_world(soma_comm_t** out);
/// This rank alone.
int soma_comm_self(soma_comm_t** out);
int soma_comm_rank(const soma_comm_t* comm, int* rank);
int soma_comm_size(const soma_comm_t* comm, int* size);
int soma_comm_barrier(const soma_comm_t* comm);
int soma_comm_clone(const soma_comm_t* comm, soma_comm_t** out);
void soma_comm_free(soma_comm_t* comm);

#ifdef __cplusplus
}
#endif

#endif /* SOMA_C_SOMA_H_ */
