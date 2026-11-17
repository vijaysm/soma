! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! The Fortran module's file I/O (HDF5 builds).

program test_io
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
    type(soma_mesh) :: m, r
    real(c_double), pointer :: u(:, :)
    character(len=256) :: dir, path
    integer :: written

    comm = soma_comm_world()
    call get_environment_variable('SOMA_TEST_OUT', dir)
    m = soma_box_mesh(comm, 2, 2, 4)
    call m%add_field('u', 3, 1, u)
    u = 1
    path = trim(dir) // '/box.soma'  ! blank-padded
    call soma_write_file(comm, m, path)
    call check(soma_detect_format(path) == SOMA_FORMAT_SOMA, 'detect .soma')
    r = soma_read_file(comm, path)
    call check(allsum(int(r%owned_count(3), 8)) == 16, 'read back the cells')
    call check(allsum(int(r%owned_count(0), 8)) == 45, 'read back the vertices')

    written = soma_convert_mesh_file(comm, path, trim(dir) // '/box.h5m')
    call check(written == SOMA_FORMAT_LEGACY_H5M, 'converted to .h5m')
    call soma_write_file(comm, m, trim(dir) // '/box.vtk', vtk_fields=['u'])
    call soma_write_file(comm, m, trim(dir) // '/bad.vtk', vtk_fields=['nope'], ierr=ierr)
    call check(ierr /= SOMA_OK, 'an unknown vtk field fails')
    r = soma_read_file(comm, trim(dir) // '/missing.soma', ierr)
    call check(ierr /= SOMA_OK, 'a missing file fails on every rank')
  end block
  call soma_finalize()
  call check_done('test_io')
end program
