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

/* ---- Mesh creation ---------------------------------------------------- */

/// A builder of this rank's vertices and elements, by gid, for a mesh of
/// dimension dim.
int soma_builder_create(int dim, soma_builder_t** out);
/// n vertices: gids[n] and coordinates xyz[3 n], interleaved (x, y, z).
int soma_builder_add_vertices(soma_builder_t* builder, soma_lo_t n, const soma_go_t* gids, const double* xyz);
/// n elements of one topology: gids[n] and their corners' vertex gids,
/// element after element (nv of them).
int soma_builder_add_elements(soma_builder_t* builder, int topology, soma_lo_t n, const soma_go_t* gids,
                              soma_offset_t nv, const soma_go_t* vertex_gids);
int soma_builder_clone(const soma_builder_t* builder, soma_builder_t** out);
void soma_builder_free(soma_builder_t* builder);

/// Collective: this rank's part of the mesh the builders of every rank
/// describe, with its sharing. Consumes the builder (later use: SOMA_ERR_STATE).
int soma_build_local_part(const soma_comm_t* comm, soma_builder_t* builder, int ordering, soma_mesh_t** out);

/// Collective: a box [0, lx] x [0, ly] x [0, lz] of nx x ny x nz cells of hex,
/// tet, quad or tri elements (nz = 1 for quad and tri), split across the ranks
/// in layers along its last axis.
int soma_box_mesh(const soma_comm_t* comm, int nx, int ny, int nz, int element, double lx, double ly, double lz,
                  int ordering, soma_mesh_t** out);

/* ---- Mesh queries ----------------------------------------------------- */

int soma_mesh_dim(const soma_mesh_t* mesh, int* dim);
/// Entities of dimension d held here: owned, copies and ghosts.
int soma_mesh_count(const soma_mesh_t* mesh, int d, soma_lo_t* count);
/// Entities owned here: lids [0, owned).
int soma_mesh_owned_count(const soma_mesh_t* mesh, int d, soma_lo_t* owned);
/// Entities held by this rank alone: lids [0, interior).
int soma_mesh_interior_count(const soma_mesh_t* mesh, int d, soma_lo_t* interior);
/// Runs of entities alike: lids [begin, end), topology, arity, ownership.
int soma_mesh_segment_count(const soma_mesh_t* mesh, int d, int* count);
int soma_mesh_segment(const soma_mesh_t* mesh, int d, int i, soma_lo_t* begin, soma_lo_t* end, int* topology,
                      int* arity, int* ownership);
/// The vertex coordinates, writable, shared with the mesh: column-major
/// (n, 3), so x[v + n i] is coordinate i of vertex v.
int soma_mesh_coords(soma_mesh_t* mesh, double** x, soma_lo_t* n);
/// Copies of the first n vertices' coordinates, interleaved: xyz[3 v + i].
int soma_mesh_get_coords_interleaved(const soma_mesh_t* mesh, soma_lo_t n, double* xyz);
int soma_mesh_set_coords_interleaved(soma_mesh_t* mesh, soma_lo_t n, const double* xyz);
/// Global ids of dimension d (count of them), shared, read-only.
int soma_mesh_gids(const soma_mesh_t* mesh, int d, const soma_go_t** gids, soma_lo_t* count);
/// Corner vertex lids of dimension d: entity e's are
/// conn[offsets[e], offsets[e + 1]); count + 1 offsets. Shared, read-only.
/// Vertices store no connectivity: dimension 0 gives count 0.
int soma_mesh_conn(const soma_mesh_t* mesh, int d, const soma_lo_t** conn, const soma_offset_t** offsets,
                   soma_lo_t* count);
/// The same, 1-based (lids and offsets plus one), for Fortran: a copy made
/// once and kept with the mesh until its topology changes.
int soma_mesh_conn1(soma_mesh_t* mesh, int d, const soma_lo_t** conn, const soma_offset_t** offsets,
                    soma_lo_t* count);
/// Each entity's topology (a SOMA_* shape value). Shared, read-only.
/// Vertices store none: dimension 0 gives count 0.
int soma_mesh_topology(const soma_mesh_t* mesh, int d, const uint8_t** topology, soma_lo_t* count);
/// The rank owning each of the count(d) entities of dimension d, into owners.
int soma_mesh_owner_ranks(const soma_comm_t* comm, const soma_mesh_t* mesh, int d, int* owners);
/// Collective: builds the adjacency graphs pairs[2 k] -> pairs[2 k + 1] and the
/// bridges (bridges[2 k] through bridges[2 k + 1]), deriving edges and faces
/// with their sharing.
int soma_mesh_build_adjacencies(const soma_comm_t* comm, soma_mesh_t* mesh, int npairs, const int* pairs,
                                int nbridges, const int* bridges);
int soma_mesh_clone(const soma_mesh_t* mesh, soma_mesh_t** out);
void soma_mesh_free(soma_mesh_t* mesh);

/* ---- Edits ----------------------------------------------------------- */

/// Batched edits of a mesh (which the edit keeps alive), made visible
/// together by soma_edit_apply. Existing entities keep their lids; added ones
/// follow; removed ones stay, masked, until compressed.
int soma_edit_create(soma_mesh_t* mesh, soma_edit_t** out);
/// n vertices, xyz[3 n] interleaved; gids[n] or NULL (the next free gids).
/// first: the lid the first will have.
int soma_edit_add_vertices(soma_edit_t* edit, soma_lo_t n, const double* xyz, const soma_go_t* gids,
                           soma_lo_t* first);
/// Elements of one topology by their nodes' lids (nlids of them, element after
/// element; lids of vertices added by this edit count), nodes_per_element
/// each (0: the topology's corners); gids[ngids] (one per element) or NULL.
int soma_edit_add_elements(soma_edit_t* edit, int topology, soma_offset_t nlids, const soma_lo_t* vertex_lids,
                           soma_lo_t ngids, const soma_go_t* gids, int nodes_per_element, soma_lo_t* first);
/// Polygons: polygon k's corners are lids[offsets[k], offsets[k + 1]).
int soma_edit_add_polygons(soma_edit_t* edit, soma_lo_t noffsets, const soma_offset_t* offsets, soma_offset_t nlids,
                           const soma_lo_t* lids, soma_lo_t* first);
/// Entities of dimension d that exist before this edit.
int soma_edit_remove(soma_edit_t* edit, int d, soma_lo_t n, const soma_lo_t* lids);
/// apply compresses when a dimension's removed fraction reaches t, in [0, 1].
int soma_edit_set_compress_threshold(soma_edit_t* edit, double t);
/// Applies the edits: added[d] and removed[d] count the entities per
/// dimension; *compressed (when not NULL) is the renumbering if apply
/// compressed the mesh, else NULL. The edit is then empty and can be used again.
int soma_edit_apply(soma_edit_t* edit, soma_lo_t added[4], soma_lo_t removed[4], soma_renumbering_t** compressed);
int soma_edit_clone(const soma_edit_t* edit, soma_edit_t** out);
void soma_edit_free(soma_edit_t* edit);

/// Drops the removed entities; *renumbering gives each old lid's new one.
int soma_mesh_compress(soma_mesh_t* mesh, soma_renumbering_t** renumbering);
/// The new lid of each old lid of dimension d (-1: removed); count of them.
int soma_renumbering_new_lids(const soma_renumbering_t* renumbering, int d, const soma_lo_t** new_lids,
                              soma_lo_t* count);
int soma_renumbering_clone(const soma_renumbering_t* renumbering, soma_renumbering_t** out);
void soma_renumbering_free(soma_renumbering_t* renumbering);

#ifdef __cplusplus
}
#endif

#endif /* SOMA_C_SOMA_H_ */
