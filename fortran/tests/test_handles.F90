! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! Handles in arrays and in components of user types: each element is its own
! reference, copied by cloning and freed when its array goes.

program test_handles
#if SOMA_TEST_MPI
  use mpi
#endif
  use soma
  use soma_check
  implicit none
  type :: application
    type(soma_mesh) :: meshes(2)
  end type
  integer :: ierr
  integer(c_long) :: live0

#if SOMA_TEST_MPI
  call MPI_Init(ierr)
#endif
  call soma_initialize()
  live0 = soma_live_handles()

  block
    type(soma_comm) :: self
    type(soma_mesh) :: a
    type(soma_mesh), allocatable :: arr(:), arr2(:)
    type(application) :: app1, app2
    integer(c_long) :: before

    self = soma_comm_self()
    a = soma_box_mesh(self, 2, 2, element=SOMA_QUAD)
    before = soma_live_handles()

    ! A scalar assigned to every element: one new reference each.
    allocate (arr(2), arr2(2))
    arr = a
    call check(soma_live_handles() == before + 2, 'arr = a clones into each element')
    call arr(1)%free()
    call check(a%dim() == 2, 'freeing an element leaves the original')

    ! Array to array.
    arr2 = arr
    call check(soma_live_handles() == before + 2, 'arr2 = arr clones the live element')
    call check(arr2(2)%dim() == 2, 'the copy is usable')
    deallocate (arr)
    call check(soma_live_handles() == before + 1, 'deallocating an array frees its elements')
    call check(arr2(2)%dim() == 2 .and. a%dim() == 2, 'and leaves the copies')
    deallocate (arr2)
    call check(soma_live_handles() == before, 'arr2 freed')

    ! Arrays of handles inside a user type, copied by intrinsic assignment.
    app1%meshes(1) = a
    app1%meshes(2) = a
    app2 = app1
    call check(soma_live_handles() == before + 4, 'copying the type clones its handles')
    call app1%meshes(1)%free()
    call app1%meshes(2)%free()
    call check(app2%meshes(1)%dim() == 2 .and. app2%meshes(2)%dim() == 2, 'the copy outlives the original''s')
  end block
  call check(soma_live_handles() == live0, 'every handle freed at the end of its block')

  call soma_finalize()
  call check_done('test_handles')
end program
