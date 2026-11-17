! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

!> The C API (soma/c/soma.h) as bind(C) interfaces, for the module soma.
!> Arrays pass as type(c_ptr) by value (c_null_ptr when empty); outputs by
!> reference. Ids here are 0-based, as in C.
module soma_c_api
  use, intrinsic :: iso_c_binding
  use soma_kinds
  implicit none
  public

  interface

    ! ---- Runtime
    integer(c_size_t) function c_last_error(buf, len) bind(C, name='soma_last_error')
      import :: c_char, c_size_t
      character(kind=c_char), intent(out) :: buf(*)
      integer(c_size_t), value :: len
    end function
    integer(c_int) function c_set_last_error(status, message) bind(C, name='soma_set_last_error')
      import :: c_int, c_char
      integer(c_int), value :: status
      character(kind=c_char), intent(in) :: message(*)
    end function
    integer(c_int) function c_initialize() bind(C, name='soma_initialize')
      import :: c_int
    end function
    integer(c_int) function c_finalize() bind(C, name='soma_finalize')
      import :: c_int
    end function
    subroutine c_abort(message) bind(C, name='soma_abort')
      import :: c_char
      character(kind=c_char), intent(in) :: message(*)
    end subroutine
    integer(c_int) function c_testing_live_handles(count) bind(C, name='soma_testing_live_handles')
      import :: c_int, c_long
      integer(c_long), intent(out) :: count
    end function

    ! ---- Communicators
    integer(c_int) function c_comm_create_f(fcomm, out) bind(C, name='soma_comm_create_f')
      import :: c_int, c_ptr
      integer(c_int), value :: fcomm
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_comm_world(out) bind(C, name='soma_comm_world')
      import :: c_int, c_ptr
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_comm_self(out) bind(C, name='soma_comm_self')
      import :: c_int, c_ptr
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_comm_rank(comm, rank) bind(C, name='soma_comm_rank')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm
      integer(c_int), intent(out) :: rank
    end function
    integer(c_int) function c_comm_size(comm, size) bind(C, name='soma_comm_size')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm
      integer(c_int), intent(out) :: size
    end function
    integer(c_int) function c_comm_barrier(comm) bind(C, name='soma_comm_barrier')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm
    end function
    integer(c_int) function c_comm_clone(comm, out) bind(C, name='soma_comm_clone')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm
      type(c_ptr), intent(out) :: out
    end function
    subroutine c_comm_free(comm) bind(C, name='soma_comm_free')
      import :: c_ptr
      type(c_ptr), value :: comm
    end subroutine

    ! ---- Mesh creation
    integer(c_int) function c_builder_create(dim, out) bind(C, name='soma_builder_create')
      import :: c_int, c_ptr
      integer(c_int), value :: dim
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_builder_add_vertices(b, n, gids, xyz) bind(C, name='soma_builder_add_vertices')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: b, gids, xyz
      integer(soma_lo), value :: n
    end function
    integer(c_int) function c_builder_add_elements(b, topology, n, gids, nv, vertex_gids) &
        bind(C, name='soma_builder_add_elements')
      import :: c_int, c_ptr, soma_lo, soma_offset
      type(c_ptr), value :: b, gids, vertex_gids
      integer(c_int), value :: topology
      integer(soma_lo), value :: n
      integer(soma_offset), value :: nv
    end function
    integer(c_int) function c_builder_clone(b, out) bind(C, name='soma_builder_clone')
      import :: c_int, c_ptr
      type(c_ptr), value :: b
      type(c_ptr), intent(out) :: out
    end function
    subroutine c_builder_free(b) bind(C, name='soma_builder_free')
      import :: c_ptr
      type(c_ptr), value :: b
    end subroutine
    integer(c_int) function c_build_local_part(comm, b, ordering, out) bind(C, name='soma_build_local_part')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm, b
      integer(c_int), value :: ordering
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_box_mesh(comm, nx, ny, nz, element, lx, ly, lz, ordering, out) &
        bind(C, name='soma_box_mesh')
      import :: c_int, c_ptr, c_double
      type(c_ptr), value :: comm
      integer(c_int), value :: nx, ny, nz, element, ordering
      real(c_double), value :: lx, ly, lz
      type(c_ptr), intent(out) :: out
    end function

    ! ---- Mesh queries
    integer(c_int) function c_mesh_dim(m, dim) bind(C, name='soma_mesh_dim')
      import :: c_int, c_ptr
      type(c_ptr), value :: m
      integer(c_int), intent(out) :: dim
    end function
    integer(c_int) function c_mesh_count(m, d, n) bind(C, name='soma_mesh_count')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: m
      integer(c_int), value :: d
      integer(soma_lo), intent(out) :: n
    end function
    integer(c_int) function c_mesh_owned_count(m, d, n) bind(C, name='soma_mesh_owned_count')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: m
      integer(c_int), value :: d
      integer(soma_lo), intent(out) :: n
    end function
    integer(c_int) function c_mesh_interior_count(m, d, n) bind(C, name='soma_mesh_interior_count')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: m
      integer(c_int), value :: d
      integer(soma_lo), intent(out) :: n
    end function
    integer(c_int) function c_mesh_segment_count(m, d, n) bind(C, name='soma_mesh_segment_count')
      import :: c_int, c_ptr
      type(c_ptr), value :: m
      integer(c_int), value :: d
      integer(c_int), intent(out) :: n
    end function
    integer(c_int) function c_mesh_segment(m, d, i, begin, end, topology, arity, ownership) &
        bind(C, name='soma_mesh_segment')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: m
      integer(c_int), value :: d, i
      integer(soma_lo), intent(out) :: begin, end
      integer(c_int), intent(out) :: topology, arity, ownership
    end function
    integer(c_int) function c_mesh_coords(m, x, n) bind(C, name='soma_mesh_coords')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: m
      type(c_ptr), intent(out) :: x
      integer(soma_lo), intent(out) :: n
    end function
    integer(c_int) function c_mesh_get_coords_interleaved(m, n, xyz) bind(C, name='soma_mesh_get_coords_interleaved')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: m, xyz
      integer(soma_lo), value :: n
    end function
    integer(c_int) function c_mesh_set_coords_interleaved(m, n, xyz) bind(C, name='soma_mesh_set_coords_interleaved')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: m, xyz
      integer(soma_lo), value :: n
    end function
    integer(c_int) function c_mesh_gids(m, d, gids, n) bind(C, name='soma_mesh_gids')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: m
      integer(c_int), value :: d
      type(c_ptr), intent(out) :: gids
      integer(soma_lo), intent(out) :: n
    end function
    integer(c_int) function c_mesh_conn1(m, d, conn, offsets, n) bind(C, name='soma_mesh_conn1')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: m
      integer(c_int), value :: d
      type(c_ptr), intent(out) :: conn, offsets
      integer(soma_lo), intent(out) :: n
    end function
    integer(c_int) function c_mesh_topology(m, d, topology, n) bind(C, name='soma_mesh_topology')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: m
      integer(c_int), value :: d
      type(c_ptr), intent(out) :: topology
      integer(soma_lo), intent(out) :: n
    end function
    integer(c_int) function c_mesh_owner_ranks(comm, m, d, owners) bind(C, name='soma_mesh_owner_ranks')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm, m, owners
      integer(c_int), value :: d
    end function
    integer(c_int) function c_mesh_build_adjacencies(comm, m, npairs, pairs, nbridges, bridges) &
        bind(C, name='soma_mesh_build_adjacencies')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm, m, pairs, bridges
      integer(c_int), value :: npairs, nbridges
    end function
    integer(c_int) function c_mesh_clone(m, out) bind(C, name='soma_mesh_clone')
      import :: c_int, c_ptr
      type(c_ptr), value :: m
      type(c_ptr), intent(out) :: out
    end function
    subroutine c_mesh_free(m) bind(C, name='soma_mesh_free')
      import :: c_ptr
      type(c_ptr), value :: m
    end subroutine

    ! ---- Edits
    integer(c_int) function c_edit_create(m, out) bind(C, name='soma_edit_create')
      import :: c_int, c_ptr
      type(c_ptr), value :: m
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_edit_add_vertices(e, n, xyz, gids, first) bind(C, name='soma_edit_add_vertices')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: e, xyz, gids
      integer(soma_lo), value :: n
      integer(soma_lo), intent(out) :: first
    end function
    integer(c_int) function c_edit_add_elements(e, topology, nlids, lids, ngids, gids, npe, first) &
        bind(C, name='soma_edit_add_elements')
      import :: c_int, c_ptr, soma_lo, soma_offset
      type(c_ptr), value :: e, lids, gids
      integer(c_int), value :: topology, npe
      integer(soma_offset), value :: nlids
      integer(soma_lo), value :: ngids
      integer(soma_lo), intent(out) :: first
    end function
    integer(c_int) function c_edit_add_polygons(e, noffsets, offsets, nlids, lids, first) &
        bind(C, name='soma_edit_add_polygons')
      import :: c_int, c_ptr, soma_lo, soma_offset
      type(c_ptr), value :: e, offsets, lids
      integer(soma_lo), value :: noffsets
      integer(soma_offset), value :: nlids
      integer(soma_lo), intent(out) :: first
    end function
    integer(c_int) function c_edit_remove(e, d, n, lids) bind(C, name='soma_edit_remove')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: e, lids
      integer(c_int), value :: d
      integer(soma_lo), value :: n
    end function
    integer(c_int) function c_edit_set_compress_threshold(e, t) bind(C, name='soma_edit_set_compress_threshold')
      import :: c_int, c_ptr, c_double
      type(c_ptr), value :: e
      real(c_double), value :: t
    end function
    integer(c_int) function c_edit_apply(e, added, removed, compressed) bind(C, name='soma_edit_apply')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: e
      integer(soma_lo), intent(out) :: added(4), removed(4)
      type(c_ptr), intent(out) :: compressed
    end function
    integer(c_int) function c_edit_clone(e, out) bind(C, name='soma_edit_clone')
      import :: c_int, c_ptr
      type(c_ptr), value :: e
      type(c_ptr), intent(out) :: out
    end function
    subroutine c_edit_free(e) bind(C, name='soma_edit_free')
      import :: c_ptr
      type(c_ptr), value :: e
    end subroutine
    integer(c_int) function c_mesh_compress(m, out) bind(C, name='soma_mesh_compress')
      import :: c_int, c_ptr
      type(c_ptr), value :: m
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_renumbering_new_lids(r, d, lids, n) bind(C, name='soma_renumbering_new_lids')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: r
      integer(c_int), value :: d
      type(c_ptr), intent(out) :: lids
      integer(soma_lo), intent(out) :: n
    end function
    integer(c_int) function c_renumbering_clone(r, out) bind(C, name='soma_renumbering_clone')
      import :: c_int, c_ptr
      type(c_ptr), value :: r
      type(c_ptr), intent(out) :: out
    end function
    subroutine c_renumbering_free(r) bind(C, name='soma_renumbering_free')
      import :: c_ptr
      type(c_ptr), value :: r
    end subroutine

    ! ---- Fields
    integer(c_int) function c_mesh_add_field(m, name, dim, ncomp, dtype, data, rows) bind(C, name='soma_mesh_add_field')
      import :: c_int, c_ptr, c_char, soma_lo
      type(c_ptr), value :: m
      character(kind=c_char), intent(in) :: name(*)
      integer(c_int), value :: dim, ncomp, dtype
      type(c_ptr), intent(out) :: data
      integer(soma_lo), intent(out) :: rows
    end function
    integer(c_int) function c_mesh_field(m, name, dim, dtype, data, rows, ncomp) bind(C, name='soma_mesh_field')
      import :: c_int, c_ptr, c_char, soma_lo
      type(c_ptr), value :: m
      character(kind=c_char), intent(in) :: name(*)
      integer(c_int), value :: dim
      integer(c_int), intent(out) :: dtype, ncomp
      type(c_ptr), intent(out) :: data
      integer(soma_lo), intent(out) :: rows
    end function
    integer(c_int) function c_mesh_field_count(m, n) bind(C, name='soma_mesh_field_count')
      import :: c_int, c_ptr
      type(c_ptr), value :: m
      integer(c_int), intent(out) :: n
    end function
    integer(c_int) function c_mesh_field_info(m, i, name, len, length, dtype, ncomp, rows, dims, variable) &
        bind(C, name='soma_mesh_field_info')
      import :: c_int, c_ptr, c_char, c_size_t, soma_lo
      type(c_ptr), value :: m
      integer(c_int), value :: i
      character(kind=c_char), intent(out) :: name(*)
      integer(c_size_t), value :: len
      integer(c_size_t), intent(out) :: length
      integer(c_int), intent(out) :: dtype, ncomp, dims, variable
      integer(soma_lo), intent(out) :: rows
    end function
    integer(c_int) function c_mesh_remove_field(m, name, dim) bind(C, name='soma_mesh_remove_field')
      import :: c_int, c_ptr, c_char
      type(c_ptr), value :: m
      character(kind=c_char), intent(in) :: name(*)
      integer(c_int), value :: dim
    end function
    integer(c_int) function c_mesh_get_field_interleaved(m, name, dim, dtype, rows, values) &
        bind(C, name='soma_mesh_get_field_interleaved')
      import :: c_int, c_ptr, c_char, soma_lo
      type(c_ptr), value :: m, values
      character(kind=c_char), intent(in) :: name(*)
      integer(c_int), value :: dim, dtype
      integer(soma_lo), value :: rows
    end function
    integer(c_int) function c_mesh_set_field_interleaved(m, name, dim, dtype, rows, values) &
        bind(C, name='soma_mesh_set_field_interleaved')
      import :: c_int, c_ptr, c_char, soma_lo
      type(c_ptr), value :: m, values
      character(kind=c_char), intent(in) :: name(*)
      integer(c_int), value :: dim, dtype
      integer(soma_lo), value :: rows
    end function

    ! ---- Halo exchange
    integer(c_int) function c_halo_create(comm, m, dim, out) bind(C, name='soma_halo_create')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm, m
      integer(c_int), value :: dim
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_halo_update(h, comm, dtype, data, rows, ncomp) bind(C, name='soma_halo_update')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: h, comm, data
      integer(c_int), value :: dtype, ncomp
      integer(soma_lo), value :: rows
    end function
    integer(c_int) function c_halo_reduce(h, comm, dtype, data, rows, ncomp, op) bind(C, name='soma_halo_reduce')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: h, comm, data
      integer(c_int), value :: dtype, ncomp, op
      integer(soma_lo), value :: rows
    end function
    integer(c_int) function c_halo_clone(h, out) bind(C, name='soma_halo_clone')
      import :: c_int, c_ptr
      type(c_ptr), value :: h
      type(c_ptr), intent(out) :: out
    end function
    subroutine c_halo_free(h) bind(C, name='soma_halo_free')
      import :: c_ptr
      type(c_ptr), value :: h
    end subroutine

    ! ---- File I/O
    integer(c_int) function c_read_file(comm, path, out) bind(C, name='soma_read_file')
      import :: c_int, c_ptr, c_char
      type(c_ptr), value :: comm
      character(kind=c_char), intent(in) :: path(*)
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_write_file(comm, m, path, nvtk, vtk_fields) bind(C, name='soma_write_file')
      import :: c_int, c_ptr, c_char
      type(c_ptr), value :: comm, m, vtk_fields
      character(kind=c_char), intent(in) :: path(*)
      integer(c_int), value :: nvtk
    end function
    integer(c_int) function c_detect_format(path, format) bind(C, name='soma_detect_format')
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: path(*)
      integer(c_int), intent(out) :: format
    end function
    integer(c_int) function c_convert_mesh_file(comm, src, dst, compression, written) &
        bind(C, name='soma_convert_mesh_file')
      import :: c_int, c_ptr, c_char
      type(c_ptr), value :: comm
      character(kind=c_char), intent(in) :: src(*), dst(*)
      integer(c_int), value :: compression
      integer(c_int), intent(out) :: written
    end function

    ! ---- Parallel
    integer(c_int) function c_partition_by_curve(comm, n, xyz, gids, dest) bind(C, name='soma_partition_by_curve')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: comm, xyz, gids, dest
      integer(soma_lo), value :: n
    end function
    integer(c_int) function c_migrate(comm, m, n, dest, ordering, out) bind(C, name='soma_migrate')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: comm, m, dest
      integer(soma_lo), value :: n
      integer(c_int), value :: ordering
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_rebalance(comm, m, ordering, out) bind(C, name='soma_rebalance')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm, m
      integer(c_int), value :: ordering
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_mesh_add_ghosts(comm, m, layers, bridge_dim, sets) bind(C, name='soma_mesh_add_ghosts')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm, m
      integer(c_int), value :: layers, bridge_dim, sets
    end function
    integer(c_int) function c_mesh_remove_ghosts(comm, m) bind(C, name='soma_mesh_remove_ghosts')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm, m
    end function

    ! ---- Smoothing, refinement, remap
    integer(c_int) function c_laplacian_smooth(comm, m, iterations, bridge_dim, tolerance, nfixed, fixed, done, &
                                               largest) bind(C, name='soma_laplacian_smooth')
      import :: c_int, c_ptr, c_double, soma_lo
      type(c_ptr), value :: comm, m, fixed
      integer(c_int), value :: iterations, bridge_dim
      real(c_double), value :: tolerance
      integer(soma_lo), value :: nfixed
      integer(c_int), intent(out) :: done
      real(c_double), intent(out) :: largest
    end function
    integer(c_int) function c_refine_uniform(comm, m, levels, out) bind(C, name='soma_refine_uniform')
      import :: c_int, c_ptr
      type(c_ptr), value :: comm, m
      integer(c_int), value :: levels
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_refine_marked(comm, m, n, marks, conforming, vertices, out) &
        bind(C, name='soma_refine_marked')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: comm, m, marks
      integer(soma_lo), value :: n
      integer(c_int), value :: conforming, vertices
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_refine_mesh(r, out) bind(C, name='soma_refine_mesh')
      import :: c_int, c_ptr
      type(c_ptr), value :: r
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_refine_lids(r, which, lids, n) bind(C, name='soma_refine_lids')
      import :: c_int, c_ptr, c_int64_t
      type(c_ptr), value :: r
      integer(c_int), value :: which
      type(c_ptr), intent(out) :: lids
      integer(c_int64_t), intent(out) :: n
    end function
    integer(c_int) function c_refine_offsets(r, which, offsets, n) bind(C, name='soma_refine_offsets')
      import :: c_int, c_ptr, c_int64_t
      type(c_ptr), value :: r
      integer(c_int), value :: which
      type(c_ptr), intent(out) :: offsets
      integer(c_int64_t), intent(out) :: n
    end function
    integer(c_int) function c_refine_masters(r, gids, n) bind(C, name='soma_refine_masters')
      import :: c_int, c_ptr, c_int64_t
      type(c_ptr), value :: r
      type(c_ptr), intent(out) :: gids
      integer(c_int64_t), intent(out) :: n
    end function
    integer(c_int) function c_refine_weights(r, weights, n) bind(C, name='soma_refine_weights')
      import :: c_int, c_ptr, c_int64_t
      type(c_ptr), value :: r
      type(c_ptr), intent(out) :: weights
      integer(c_int64_t), intent(out) :: n
    end function
    integer(c_int) function c_refine_dropped_count(r, n) bind(C, name='soma_refine_dropped_count')
      import :: c_int, c_ptr
      type(c_ptr), value :: r
      integer(c_int), intent(out) :: n
    end function
    integer(c_int) function c_refine_dropped_name(r, i, name, len, length) bind(C, name='soma_refine_dropped_name')
      import :: c_int, c_ptr, c_char, c_size_t
      type(c_ptr), value :: r
      integer(c_int), value :: i
      character(kind=c_char), intent(out) :: name(*)
      integer(c_size_t), value :: len
      integer(c_size_t), intent(out) :: length
    end function
    integer(c_int) function c_refine_clone(r, out) bind(C, name='soma_refine_clone')
      import :: c_int, c_ptr
      type(c_ptr), value :: r
      type(c_ptr), intent(out) :: out
    end function
    subroutine c_refine_free(r) bind(C, name='soma_refine_free')
      import :: c_ptr
      type(c_ptr), value :: r
    end subroutine
    integer(c_int) function c_remap_create(comm, s, t, merge_tolerance, search, surface, radius, sphere_clip, out) &
        bind(C, name='soma_remap_create')
      import :: c_int, c_ptr, c_double
      type(c_ptr), value :: comm, s, t
      real(c_double), value :: merge_tolerance, radius
      integer(c_int), value :: search, surface, sphere_clip
      type(c_ptr), intent(out) :: out
    end function
    integer(c_int) function c_remap_apply(r, comm, ns, us, nt, ut) bind(C, name='soma_remap_apply')
      import :: c_int, c_ptr, soma_lo
      type(c_ptr), value :: r, comm, us, ut
      integer(soma_lo), value :: ns, nt
    end function
    integer(c_int) function c_remap_clone(r, out) bind(C, name='soma_remap_clone')
      import :: c_int, c_ptr
      type(c_ptr), value :: r
      type(c_ptr), intent(out) :: out
    end function
    subroutine c_remap_free(r) bind(C, name='soma_remap_free')
      import :: c_ptr
      type(c_ptr), value :: r
    end subroutine

  end interface

end module soma_c_api
