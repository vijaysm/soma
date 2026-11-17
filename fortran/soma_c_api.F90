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

  end interface

end module soma_c_api
