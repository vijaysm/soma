! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! The Fortran module's fields and halo exchange.

program test_fields
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
    comm = soma_comm_world()
    call fields(comm)
    call halo(comm)
    call halo_empty_ranks(comm)
  end block
  call soma_finalize()
  call check_done('test_fields')

contains

  subroutine fields(comm)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh) :: m
    real(c_double), pointer :: u(:, :), again(:, :)
    integer(c_int32_t), pointer :: k(:, :)
    integer(c_int64_t), pointer :: g(:, :)
    character(len=32) :: name
    character(len=256), allocatable :: names(:)
    real(c_double), allocatable :: copy(:, :)
    integer(soma_lo) :: r

    m = soma_box_mesh(comm, 2, 2, 2)
    call m%add_field('u', 3, 2, u)
    call check(size(u, 1) == m%count(3) .and. size(u, 2) == 2, 'u is (rows, ncomp)')
    do r = 1, size(u, 1)
      u(r, :) = [real(r, c_double), -real(r, c_double)]
    end do
    call m%add_field('k', 0, 1, k)
    call m%add_field('g', 0, 1, g)
    name = 'u'  ! blank-padded
    call m%field(name, again)
    if (size(u) > 0) call check(associated(again, u), 'the field is shared')  ! false for zero size
    call m%field('k', u, ierr=ierr)
    call check(ierr == SOMA_ERR_TYPE, 'an i32 field is not real')
    call m%field('nope', u, ierr=ierr)
    call check(ierr == SOMA_ERR_OUT_OF_RANGE, 'no such field')
    names = m%field_names()
    call check(any(names == 'u') .and. any(names == 'k') .and. any(names == 'g'), 'field names')

    allocate (copy(2, m%count(3)))
    call m%get_field(name, copy)
    if (m%count(3) > 0) call check(all(copy(:, 1) == [1d0, -1d0]), 'get_field is interleaved (ncomp, rows)')
    copy = -copy
    call m%set_field('u', copy)
    if (m%count(3) > 0) call check(again(1, 2) == 1d0, 'set_field writes the field')
    deallocate (copy)
    allocate (copy(3, m%count(3)))
    call m%get_field('u', copy, ierr=ierr)
    call check(ierr == SOMA_ERR_INVALID_ARGUMENT, 'get_field needs ncomp rows')
    call m%remove_field('k')
    call m%field('k', k, ierr=ierr)
    call check(ierr == SOMA_ERR_OUT_OF_RANGE, 'a removed field is gone')
  end subroutine

  subroutine halo(comm)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh) :: m
    type(soma_halo) :: h
    real(c_double), pointer :: v(:, :), w(:, :)
    integer(soma_go), pointer :: g(:)
    integer(soma_lo) :: n

    m = soma_box_mesh(comm, 2, 2, 4)
    call m%add_field('v', 0, 1, v)
    g => m%gids(0)
    n = m%owned_count(0)
    v(:, 1) = -1
    v(1:n, 1) = real(g(1:n), c_double)
    h = soma_halo(comm, m, 0)
    call h%update(comm, v)
    call check(all(v(:, 1) == real(g, c_double)), 'update: copies get their owners'' values')

    call m%add_field('w', 0, 1, w)
    w = 1
    call h%reduce(comm, w, SOMA_SUM)
    call check(abs(allsum_real(sum(1 / w(:, 1))) - 45d0) < 1d-9, 'reduce sum counts the holders')

    call m%add_ghosts(comm, 1, 0)
    call m%add_field('z', 0, 1, w)
    call h%update(comm, w, ierr)
    call check(ierr == SOMA_ERR_STATE, 'an exchange is stale after ghosts')
  end subroutine

  subroutine halo_empty_ranks(comm)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh) :: m
    type(soma_halo) :: h
    integer(c_int64_t), pointer :: v(:, :)
    integer(soma_go), pointer :: g(:)
    integer(soma_lo) :: n
    m = soma_box_mesh(comm, 2, 2, 2)  ! empty ranks at 3 and 4
    call m%add_field('v2', 0, 2, v)
    g => m%gids(0)
    n = m%owned_count(0)
    v = -1
    v(1:n, 1) = g(1:n)
    v(1:n, 2) = 2 * g(1:n)
    h = soma_halo(comm, m, 0)
    call h%update(comm, v)
    call check(all(v(:, 1) == g) .and. all(v(:, 2) == 2 * g), 'two components, empty ranks')
    call h%reduce(comm, v, SOMA_MAX)
  end subroutine

  real(c_double) function allsum_real(x)
    real(c_double), intent(in) :: x
#if SOMA_TEST_MPI
    integer :: e
    call MPI_Allreduce(x, allsum_real, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, e)
#else
    allsum_real = x
#endif
  end function

end program
