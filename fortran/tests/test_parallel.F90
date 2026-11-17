! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! The Fortran module's partitioning, migration and ghost layers.

program test_parallel
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
    type(soma_comm) :: comm
    type(soma_mesh) :: m, moved, even, q
    integer, allocatable :: dest(:)
    real(c_double), allocatable :: c(:, :)
    real(c_double), pointer :: x(:, :)
    integer(soma_lo), pointer :: conn(:)
    integer(soma_offset), pointer :: off(:)
    integer(soma_go), pointer :: g(:)
    integer(soma_lo) :: e, n, held
    integer(8) :: cells

    comm = soma_comm_world()
    m = soma_box_mesh(comm, 4, 4, 4)
    cells = allsum(int(m%owned_count(3), 8))
    n = m%owned_count(3)
    x => m%coords()
    call m%conn(3, conn, off)
    allocate (c(3, n))
    do e = 1, n
      c(:, e) = sum(x(conn(off(e):off(e + 1) - 1), :), dim=1) / 8
    end do
    g => m%gids(3)
    allocate (dest(m%count(3)))
    dest = my_rank()
    call soma_partition_by_curve(comm, c, g(1:n), dest(1:n))
    call check(all(dest >= 0 .and. dest < nranks()), 'destinations are ranks')
    moved = soma_migrate(comm, m, dest)
    call check(allsum(int(moved%owned_count(3), 8)) == cells, 'migrate keeps every cell')
    even = soma_rebalance(comm, m)
    call check(allsum(int(even%owned_count(3), 8)) == cells, 'rebalance keeps every cell')
    dest(1:min(1, size(dest))) = 1000
    moved = soma_migrate(comm, m, dest, ierr=ierr)
    call check(ierr == SOMA_ERR_INVALID_ARGUMENT, 'a bad destination fails everywhere')

    ! Ghosts in place, and the 1-based connectivity after them.
    held = m%count(3)
    call m%add_ghosts(comm, 1, 0)
    call check(allsum(int(m%owned_count(3), 8)) == cells, 'ghosts keep the owned cells')
    if (nranks() > 1) call check(allsum(merge(1_8, 0_8, m%count(3) > held)) > 0, 'ghosts added somewhere')
    call m%conn(3, conn, off)
    call check(size(off) == m%count(3) + 1 .and. all(conn >= 1 .and. conn <= m%count(0)), 'conn after ghosts')
    call m%remove_ghosts(comm)
    call check(m%count(3) == held, 'ghosts removed')
    call m%add_ghosts(comm, 1, 7, ierr=ierr)
    call check(ierr == SOMA_ERR_INVALID_ARGUMENT, 'bridge_dim above the mesh')

    ! Ranks holding nothing: zero-size arrays.
    q = soma_box_mesh(comm, 2, 2, 1, element=SOMA_QUAD)
    n = q%owned_count(2)
    deallocate (c, dest)
    allocate (c(3, n), dest(n))
    c = 0
    g => q%gids(2)
    call soma_partition_by_curve(comm, c, g(1:n), dest)
    call q%add_ghosts(comm, 1, 0)
  end block
  call soma_finalize()
  call check_done('test_parallel')
end program
