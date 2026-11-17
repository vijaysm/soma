! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! The Fortran module's runtime: initialization, communicators, errors and
! handle copies.

program test_runtime
#if SOMA_TEST_MPI
  use mpi
#endif
  use soma
  use soma_check
  implicit none
  integer :: ierr
  integer(c_long) :: live0, live1

#if SOMA_TEST_MPI
  call MPI_Init(ierr)
#endif
  call soma_initialize()
  call soma_initialize(ierr)
  call check(ierr == SOMA_OK, 'initialize twice')
  live0 = soma_live_handles()

  block
    type(soma_comm) :: world, self, copy
    type(soma_mesh) :: m, m2, empty
    integer(soma_lo) :: n
#if SOMA_TEST_MPI
    world = soma_comm(MPI_COMM_WORLD)
#else
    world = soma_comm_world()
#endif
    call check(world%rank() == my_rank(), 'comm rank')
    call check(world%size() == nranks(), 'comm size')
    self = soma_comm_self()
    call check(self%size() == 1, 'self size')
    copy = world
    call world%free()
    call check(copy%size() == nranks(), 'a copy outlives the original')
    call world%free()  ! freeing twice is harmless

    ! A handle never assigned: an error status and a message, no crash.
    n = empty%count(0, ierr)
    call check(ierr == SOMA_ERR_STATE, 'null mesh handle is a state error')
    call check(index(soma_error_message(), 'null') > 0, 'null handle message')

    m = soma_box_mesh(copy, 2, 2, 2)
    m2 = m
    call m%free()
    call check(m2%count(3) >= 0, 'a mesh copy outlives the original')
    m = m2  ! assignment over a freed handle
    m = m   ! self-assignment
    call check(m%dim() == 3, 'self assignment keeps the mesh')
    call check(soma_live_handles() > live0, 'handles are live')
  end block
  live1 = soma_live_handles()
  call check(live1 == live0, 'every handle freed at the end of its block')

  call soma_finalize()
  call check_done('test_runtime')
end program
