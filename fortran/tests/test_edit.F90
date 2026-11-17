! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! The Fortran module's edits, with 1-based ids in and out.

program test_edit
#if SOMA_TEST_MPI
  use mpi
#endif
  use soma
  use soma_check
  implicit none
  integer :: ierr

#if SOMA_TEST_MPI
  call MPI_Init(ierr)
#endif
  call soma_initialize()
  block
    type(soma_comm) :: self
    type(soma_mesh) :: m, alias
    type(soma_edit) :: e
    type(soma_renumbering) :: r
    integer(soma_lo) :: v, q, first
    integer(soma_lo) :: added(0:3), removed(0:3)
    integer(soma_lo), pointer :: conn(:)
    integer(soma_offset), pointer :: off(:)
    real(c_double), pointer :: x(:, :)
    integer(soma_lo), allocatable :: new_ids(:)

    self = soma_comm_self()
    m = soma_box_mesh(self, 2, 2, 1, element=SOMA_QUAD)  ! 9 vertices, 4 quads
    alias = m  ! the same mesh: it sees every edit made through m
    call alias%conn(2, conn, off)
    call check(size(off) == 5, 'four quads through the alias')
    e = soma_edit(m)
    call e%set_compress_threshold(1d0)
    call e%add_vertices(reshape([3d0, 3d0, 0d0], [3, 1]), gids=[100_soma_go], first=v)
    call check(v == 10, 'the first added vertex is id 10')
    call e%add_elements(SOMA_QUAD, [5_soma_lo, 6_soma_lo, 9_soma_lo, v], gids=[50_soma_go], first=q)
    call check(q == 5, 'the added quad is id 5')
    call e%remove(2, [1_soma_lo])
    call e%apply(added, removed)
    call check(added(2) == 1 .and. removed(2) == 1, 'apply reports per dimension')
    call check(m%count(2) == 5, 'the removed quad stays until compress')

    ! The added quad's corners, through the 1-based connectivity.
    call m%conn(2, conn, off)
    x => m%coords()
    call check(all(conn(off(5):off(6) - 1) == [5, 6, 9, 10]), 'the new quad''s corners, 1-based')
    call check(all(x(conn(off(6) - 1), :) == [3d0, 3d0, 0d0]), 'x(conn(k), :) of the added vertex')
    call alias%conn(2, conn, off)
    call check(size(off) == 6, 'the alias sees the added quad')
    call check(all(conn(off(5):off(6) - 1) == [5, 6, 9, 10]), 'and its corners')

    r = m%compress()
    new_ids = r%new_ids(2)
    call check(size(new_ids) == 5, 'a renumbering of 5 quads')
    call check(new_ids(1) == 0, 'a removed quad maps to 0')
    call check(all(new_ids(2:) == [1, 2, 3, 4]), 'the rest move down, 1-based')
    call check(m%count(2) == 4, 'compressed')

    ! Polygons by 1-based offsets and ids, and errors.
    first = -1
    call e%add_polygons([1_soma_offset, 6_soma_offset], [1_soma_lo, 2_soma_lo, 5_soma_lo, 8_soma_lo, 4_soma_lo], &
                        first)
    call check(first == 5, 'the polygon is id 5')
    call e%apply()
    call e%remove(0, [0_soma_lo], ierr)
    call check(ierr /= SOMA_OK, 'id 0 is not an entity')
    call e%remove(0, [99_soma_lo], ierr)
    call check(ierr == SOMA_ERR_INVALID_ARGUMENT, 'id 99 is not an entity')

    ! The edit keeps the mesh alive.
    call m%free()
    call e%add_vertices(reshape([0d0, 0d0, 1d0], [3, 1]))
    call e%apply()
  end block
  call check(soma_live_handles() == 0, 'every handle freed')
  call soma_finalize()
  call check_done('test_edit')
end program
