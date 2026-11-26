! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! The Fortran module's fields and halo exchange.

! Field rules, as C functions of the batch.
module test_fields_rules
  use, intrinsic :: iso_c_binding, only: c_ptr, c_f_pointer
  use soma
  implicit none
contains
  ! u = (x + 2y, gid): f64, two components.
  integer(c_int) function rule_u(batch, ncomp, values, context) bind(C)
    type(soma_entity_batch), intent(in) :: batch
    integer(c_int), value :: ncomp
    type(c_ptr), value :: values, context
    integer(soma_lo), pointer :: offsets(:)
    real(c_double), pointer :: xyz(:, :), v(:, :)
    integer(soma_go), pointer :: gid(:)
    integer(c_int), pointer :: calls
    integer :: i
    call c_f_pointer(context, calls)
    calls = calls + 1
    call soma_batch_arrays(batch, offsets, xyz, gid)
    call c_f_pointer(values, v, [int(ncomp), int(batch%count)])
    do i = 1, int(batch%count)
      v(1, i) = xyz(1, offsets(i) + 1) + 2 * xyz(2, offsets(i) + 1)
      v(2, i) = real(gid(i), c_double)
    end do
    rule_u = 0
  end function

  ! k = 10 * corners: i32.
  integer(c_int) function rule_k(batch, ncomp, values, context) bind(C)
    type(soma_entity_batch), intent(in) :: batch
    integer(c_int), value :: ncomp
    type(c_ptr), value :: values, context
    integer(soma_lo), pointer :: offsets(:)
    real(c_double), pointer :: xyz(:, :)
    integer(soma_go), pointer :: gid(:)
    integer(c_int32_t), pointer :: v(:)
    integer :: i
    call soma_batch_arrays(batch, offsets, xyz, gid)
    call c_f_pointer(values, v, [int(batch%count)])
    do i = 1, int(batch%count)
      v(i) = int(10 * (offsets(i + 1) - offsets(i)), c_int32_t)
    end do
    rule_k = 0
  end function
end module

program test_fields
#if SOMA_TEST_MPI
  use mpi
#endif
  use, intrinsic :: iso_c_binding, only: c_loc
  use soma
  use soma_check
  use test_fields_rules
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
    call rules(comm)
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
    character(len=soma_name_len), allocatable :: names(:)
    character(len=:), allocatable :: long_name, longer_name, huge_name
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

    ! Names are never cut off: whole up to soma_name_len, an error beyond it.
    long_name = repeat('a', 300)
    call m%add_field(long_name, 3, 1, u)
    names = m%field_names()
    call check(any(names == long_name), 'a 300-character name comes back whole')
    call check(soma_name_len >= 1024, 'soma_name_len is 1024 by default')
    longer_name = repeat('b', soma_name_len + 1)
    call m%add_field(longer_name, 3, 1, u)
    names = m%field_names(ierr)
    call check(ierr == SOMA_ERR_OUT_OF_RANGE, 'a name longer than soma_name_len is an error')
    call check(index(soma_error_message(), 'SOMA_FORTRAN_NAME_LENGTH') > 0, 'naming the option that sets it')
    call m%remove_field(longer_name)
    call m%remove_field(long_name)
    huge_name = repeat('c', 2000)
    call m%field(huge_name, u, ierr=ierr)
    call check(ierr == SOMA_ERR_OUT_OF_RANGE, 'no field of that name')
    call check(index(soma_error_message(), huge_name) > 0, 'the message is whole, however long')
    call m%field('u', u)
    call m%field('u', again)  ! both pointers again: adding and removing fields moved them

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

  ! Rules: sampled at vertices and cells, carried to the vertices refinement
  ! makes; a type other than the field's is an error.
  subroutine rules(comm)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh), target :: m
    type(soma_mesh) :: fine
    type(soma_refine) :: r
    real(c_double), pointer :: u(:, :), x(:, :)
    integer(c_int32_t), pointer :: k(:, :)
    integer(soma_go), pointer :: g(:)
    integer(c_int), target :: calls
    integer :: v, wrong

    m = soma_box_mesh(comm, 2, 2, 2)
    call m%add_field('u', 0, 2, u)
    call m%add_field('k', 3, 1, k)
    calls = 0
    call m%set_rule('u', SOMA_F32, rule_u, c_loc(calls), ierr)
    call check(ierr == SOMA_ERR_TYPE, 'a rule of another type than the field''s')
    call m%set_rule('u', SOMA_F64, rule_u, c_loc(calls))
    call m%set_rule('k', SOMA_I32, rule_k)
    call check(m%has_rule('u') .and. m%has_rule('k', 3), 'the fields have rules')
    call m%sample('u')
    call m%sample('k')
    if (m%count(0) > 0) call check(calls == 1, 'one call for every vertex')
    call m%field('u', u)
    call m%field('k', k)
    x => m%coords()
    g => m%gids(0)
    wrong = 0
    do v = 1, size(u, 1)
      if (u(v, 1) /= x(v, 1) + 2 * x(v, 2) .or. u(v, 2) /= real(g(v), c_double)) wrong = wrong + 1
    end do
    if (size(k) > 0) call check(all(k(:, 1) == 80), 'k is 10 a corner')

    r = soma_refine_uniform(comm, m, 1)
    fine = r%mesh()
    call check(fine%has_rule('u'), 'refinement keeps the rule')
    call fine%field('u', u)
    x => fine%coords()
    g => fine%gids(0)
    do v = 1, size(u, 1)
      if (u(v, 1) /= x(v, 1) + 2 * x(v, 2) .or. u(v, 2) /= real(g(v), c_double)) wrong = wrong + 1
    end do
    call check(wrong == 0, 'every vertex holds the rule''s values')
    call fine%clear_rule('u', SOMA_F64)
    call check(.not. fine%has_rule('u'), 'cleared')
  end subroutine

end program
