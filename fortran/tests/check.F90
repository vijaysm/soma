! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! A small test harness for the Fortran tests: checks count failures, and
! check_done agrees on them across the ranks and sets the exit status.

module soma_check
#if SOMA_TEST_MPI
  use mpi
#endif
  use, intrinsic :: iso_fortran_env, only: error_unit
  implicit none
  private
  public :: check, check_done, my_rank, nranks, allsum

  integer, save :: failures = 0

contains

  subroutine check(cond, label)
    logical, intent(in) :: cond
    character(*), intent(in) :: label
    if (.not. cond) then
      failures = failures + 1
      write (error_unit, '(a, i0, 2a)') 'rank ', my_rank(), ': FAILED: ', label
    end if
  end subroutine

  integer function my_rank()
#if SOMA_TEST_MPI
    integer :: ierr
    call MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr)
#else
    my_rank = 0
#endif
  end function

  integer function nranks()
#if SOMA_TEST_MPI
    integer :: ierr
    call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)
#else
    nranks = 1
#endif
  end function

  !> The sum over the ranks.
  integer(8) function allsum(v)
    integer(8), intent(in) :: v
#if SOMA_TEST_MPI
    integer :: ierr
    call MPI_Allreduce(v, allsum, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, ierr)
#else
    allsum = v
#endif
  end function

  !> Agrees on the failures, finalizes MPI, and stops with status 1 on any.
  subroutine check_done(name)
    character(*), intent(in) :: name
    integer :: total, rank
    rank = my_rank()
#if SOMA_TEST_MPI
    block
      integer :: ierr
      call MPI_Allreduce(failures, total, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
      call MPI_Finalize(ierr)
    end block
#else
    total = failures
#endif
    if (total /= 0) then
      if (rank == 0) write (error_unit, '(2a, i0, a)') name, ': ', total, ' failed checks'
      stop 1
    end if
    if (rank == 0) print '(2a)', name, ': all checks passed'
  end subroutine

end module soma_check
