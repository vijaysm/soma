! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! Calls before soma_initialize and after soma_finalize return SOMA_ERR_STATE
! instead of aborting.

program test_uninit
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
  block
    type(soma_mesh) :: m
    m = soma_box_mesh(soma_comm_self(), 2, 2, ierr=ierr)
    call check(ierr == SOMA_ERR_STATE, 'a mesh before soma_initialize')
    call check(index(soma_error_message(), 'soma_initialize') > 0, 'the message names soma_initialize')
  end block
  call soma_initialize()
  block
    type(soma_mesh) :: m
    m = soma_box_mesh(soma_comm_self(), 2, 2, ierr=ierr)
    call check(ierr == SOMA_OK, 'a mesh after soma_initialize')
  end block
  call soma_finalize()
  block
    type(soma_mesh) :: m
    m = soma_box_mesh(soma_comm_self(), 2, 2, ierr=ierr)
    call check(ierr == SOMA_ERR_STATE, 'a mesh after soma_finalize')
  end block
  call check_done('test_uninit')
end program
