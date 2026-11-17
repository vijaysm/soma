! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! Without ierr, a failure stops the program with SOMA's message (run by
! expect_failure.cmake, which requires the non-zero exit, the message, and the
! output written before the failure). Built twice: once SOMA initializes
! Kokkos, once the host program does (SOMA_TEST_HOST_KOKKOS), as a C++ code
! that calls Fortran would.

program test_stop
#if SOMA_TEST_MPI
  use mpi
#endif
  use soma
  implicit none
#if SOMA_TEST_HOST_KOKKOS
  interface
    subroutine host_kokkos_initialize() bind(C, name='test_host_kokkos_initialize')
    end subroutine
  end interface
#endif
  integer :: ierr
  type(soma_mesh) :: m
#if SOMA_TEST_MPI
  call MPI_Init(ierr)
#endif
#if SOMA_TEST_HOST_KOKKOS
  call host_kokkos_initialize()
#endif
  call soma_initialize()
  print '(a)', 'before the failure'
  m = soma_box_mesh(soma_comm_world(), 2, 2, 2, element=SOMA_POLYGON)  ! no ierr: stops here
  print '(a)', 'not reached'
end program
