! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! A Fortran program on an installed SOMA: a box mesh on this rank, its cells counted.

program consumer
#if HAVE_MPI
  use mpi
#endif
  use soma
  implicit none
  integer :: ierr
  integer(soma_lo) :: cells
#if HAVE_MPI
  call MPI_Init(ierr)
#endif
  call soma_initialize()
  block
    type(soma_mesh) :: m
    m = soma_box_mesh(soma_comm_self(), 2, 2, 2)
    cells = m%count(3)
  end block
  call soma_finalize()
#if HAVE_MPI
  call MPI_Finalize(ierr)
#endif
  if (cells /= 8) stop 1
  print '(a, i0, a)', 'fortran consumer: ', cells, ' cells'
end program
