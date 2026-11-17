! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! Without ierr, a failure stops the program with SOMA's message (run by
! expect_failure.cmake, which requires the non-zero exit and the message).

program test_stop
#if SOMA_TEST_MPI
  use mpi
#endif
  use soma
  implicit none
  integer :: ierr
  type(soma_mesh) :: m
#if SOMA_TEST_MPI
  call MPI_Init(ierr)
#endif
  call soma_initialize()
  m = soma_box_mesh(soma_comm_world(), 2, 2, 2, element=SOMA_POLYGON)  ! no ierr: stops here
  print '(a)', 'not reached'
end program
