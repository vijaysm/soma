! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! pipeline_f <out_dir>: the steps of examples/pipeline.cpp through the Fortran
! module, with the same results to the last bit:
!
!   1. a 6^3 box of tets, split across the ranks, with a ghost layer through vertices;
!   2. its interior vertices moved off the grid by a gid-based offset;
!   3. ten sweeps of Laplacian smoothing (the box's faces fixed), then no ghosts;
!   4. u = 1 + x + 2y + 3z at the cell centroids, remapped to a 5^3 box of tets;
!   5. both meshes written to <out_dir> as source and target, .soma and .vtk.
!
! Rank 0 prints the digests of the vertex coordinates and of the remapped
! values (in gid order) and the integral of the remapped u. Built without MPI
! (SOMA_PIPELINE_MPI 0) it runs on one process, threaded as Kokkos is. The scalar
! arithmetic is parenthesized as the C++ example's and compiled without fused
! multiply-adds; digests and printing are the C++ example's own (pipeline_arith).

module pipeline_report
  use, intrinsic :: iso_c_binding
  implicit none
  interface
    integer(c_int64_t) function pipeline_fnv(n, values) bind(C, name='pipeline_fnv')
      import :: c_int64_t, c_double
      integer(c_int64_t), value :: n
      real(c_double), intent(in) :: values(*)
    end function
    subroutine pipeline_print(coords, remap, integral) bind(C, name='pipeline_print')
      import :: c_int64_t, c_double
      integer(c_int64_t), value :: coords, remap
      real(c_double), value :: integral
    end subroutine
  end interface
end module pipeline_report

program pipeline
#if SOMA_PIPELINE_MPI
  use mpi
#endif
  use soma
  use pipeline_report
  implicit none
  integer :: ierr, rank
  character(len=4096) :: out

  rank = 0
#if SOMA_PIPELINE_MPI
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
#endif
  if (command_argument_count() /= 1) then
    if (rank == 0) write (*, '(a)') 'usage: pipeline_f <out_dir>'
#if SOMA_PIPELINE_MPI
    call MPI_Finalize(ierr)
#endif
    stop 2
  end if
  call get_command_argument(1, out)
  call soma_initialize()
#if SOMA_PIPELINE_MPI
  call run(soma_comm(MPI_COMM_WORLD), trim(out))
#else
  call run(soma_comm_world(), trim(out))
#endif
  call soma_finalize()
#if SOMA_PIPELINE_MPI
  call MPI_Finalize(ierr)
#endif

contains

  subroutine run(comm, out)
    type(soma_comm), intent(in) :: comm
    character(*), intent(in) :: out
    type(soma_mesh) :: source, target
    type(soma_remap) :: op
    real(c_double), pointer :: x(:, :), u(:, :), ut(:, :), tx(:, :)
    integer(soma_lo), pointer :: conn(:), tconn(:)
    integer(soma_offset), pointer :: off(:), toff(:)
    integer(soma_go), pointer :: g(:), tg(:)
    logical, allocatable :: fixed(:)
    real(c_double), allocatable :: xyz(:), coords(:), rows(:), remapped(:), values(:)
    real(c_double) :: h, a, gd, c(3), integral
    integer(soma_lo) :: v, e, nv, nt
    integer :: i, k
    integer(c_int64_t) :: dc, dr

    ! 1. The source box, with ghosts.
    source = soma_box_mesh(comm, 6, 6, 6, element=SOMA_TET)
    call source%add_ghosts(comm, 1, 0)

    ! 2. Interior vertices off the grid.
    h = 1.0d0 / 6.0d0
    a = 0.05d0 * h
    x => source%coords()
    g => source%gids(0)
    allocate (fixed(source%count(0)))
    do v = 1, source%count(0)
      fixed(v) = any(abs(x(v, :)) < 1d-12 .or. abs(x(v, :) - 1.0d0) < 1d-12)
      if (fixed(v)) cycle
      gd = real(g(v), c_double)
      x(v, 1) = moved(x(v, 1), a, sin(gd))
      x(v, 2) = moved(x(v, 2), a, cos(gd))
      x(v, 3) = moved(x(v, 3), a, sin(2.0d0 * gd))
    end do

    ! 3. Smoothing, then no ghosts.
    call soma_laplacian_smooth(comm, source, iterations=10, fixed=fixed)
    call source%remove_ghosts(comm)

    ! 4. u at the centroids, remapped.
    x => source%coords()
    call source%conn(3, conn, off)
    call source%add_field('u', 3, 1, u)
    do e = 1, source%count(3)
      do i = 1, 3
        c(i) = centroid(x(conn(off(e)), i), x(conn(off(e) + 1), i), x(conn(off(e) + 2), i), x(conn(off(e) + 3), i))
      end do
      u(e, 1) = linear(c(1), c(2), c(3))
    end do
    target = soma_box_mesh(comm, 5, 5, 5, element=SOMA_TET)
    call target%add_field('u', 3, 1, ut)
    op = soma_remap(comm, source, target)
    call op%apply(comm, u(:, 1), ut(:, 1))

    ! 5. Files.
    call soma_write_file(comm, source, out // '/source.soma', vtk_fields=['u'])
    call soma_write_file(comm, target, out // '/target.soma', vtk_fields=['u'])
    call soma_write_file(comm, source, out // '/source.vtk', vtk_fields=['u'])
    call soma_write_file(comm, target, out // '/target.vtk', vtk_fields=['u'])

    ! Digests, in gid order.
    nv = source%owned_count(0)
    g => source%gids(0)
    allocate (xyz(3 * nv))
    do v = 1, nv
      xyz(3 * v - 2:3 * v) = x(v, :)
    end do
    coords = by_gid(g(1:nv), xyz, 3)

    nt = target%owned_count(3)
    tx => target%coords()
    call target%conn(3, tconn, toff)
    tg => target%gids(3)
    allocate (rows(2 * nt))
    do e = 1, nt
      rows(2 * e - 1) = ut(e, 1)
      rows(2 * e) = tet_volume(tx, tconn(toff(e):toff(e) + 3))
    end do
    remapped = by_gid(tg(1:nt), rows, 2)

    dc = pipeline_fnv(int(size(coords), c_int64_t), coords)
    integral = 0
    k = size(remapped) / 2
    allocate (values(k))
    do i = 1, k
      values(i) = remapped(2 * i - 1)
      integral = accumulate(integral, remapped(2 * i - 1), remapped(2 * i))
    end do
    dr = pipeline_fnv(int(k, c_int64_t), values)
    if (comm%rank() == 0) call pipeline_print(dc, dr, integral)
  end subroutine

  ! Rows of `width` values of the owned entities, gathered to every rank (with
  ! MPI) and ordered by gid.
  function by_gid(gids, rows, width) result(out)
    integer(soma_go), intent(in) :: gids(:)
    real(c_double), intent(in) :: rows(:)
    integer, intent(in) :: width
    real(c_double), allocatable :: out(:)
    integer(soma_go), allocatable :: all_gids(:)
    real(c_double), allocatable :: all_rows(:)
    integer, allocatable :: order(:)
    integer :: n, i, total
#if SOMA_PIPELINE_MPI
    integer, allocatable :: counts(:), displs(:)
    integer :: p, ierr
    call MPI_Comm_size(MPI_COMM_WORLD, p, ierr)
    allocate (counts(p), displs(p))
    n = size(gids)
    call MPI_Allgather(n, 1, MPI_INTEGER, counts, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)
    displs(1) = 0
    do i = 2, p
      displs(i) = displs(i - 1) + counts(i - 1)
    end do
    total = sum(counts)
    allocate (all_gids(total), all_rows(width * total))
    call MPI_Allgatherv(gids, n, MPI_INTEGER8, all_gids, counts, displs, MPI_INTEGER8, MPI_COMM_WORLD, ierr)
    call MPI_Allgatherv(rows, width * n, MPI_DOUBLE_PRECISION, all_rows, width * counts, width * displs, &
                        MPI_DOUBLE_PRECISION, MPI_COMM_WORLD, ierr)
#else
    n = size(gids)
    total = n
    all_gids = gids
    all_rows = rows
#endif
    order = sorted(all_gids)
    allocate (out(width * total))
    do i = 1, total
      out(width * (i - 1) + 1:width * i) = all_rows(width * (order(i) - 1) + 1:width * order(i))
    end do
  end function

  ! The positions of keys in increasing order (insertion sort; the keys are unique).
  function sorted(keys) result(order)
    integer(soma_go), intent(in) :: keys(:)
    integer, allocatable :: order(:)
    integer :: i, j, t
    order = [(i, i = 1, size(keys))]
    do i = 2, size(keys)
      t = order(i)
      j = i - 1
      do while (j >= 1)
        if (keys(order(j)) <= keys(t)) exit
        order(j + 1) = order(j)
        j = j - 1
      end do
      order(j + 1) = t
    end do
  end function

  real(c_double) function moved(x, a, s)  ! x + a s
    real(c_double), intent(in) :: x, a, s
    moved = x + (a * s)
  end function

  real(c_double) function centroid(x0, x1, x2, x3)  ! (((x0 + x1) + x2) + x3) / 4
    real(c_double), intent(in) :: x0, x1, x2, x3
    centroid = (((x0 + x1) + x2) + x3) / 4.0d0
  end function

  real(c_double) function linear(x, y, z)  ! ((1 + x) + 2 y) + 3 z
    real(c_double), intent(in) :: x, y, z
    linear = ((1.0d0 + x) + (2.0d0 * y)) + (3.0d0 * z)
  end function

  real(c_double) function accumulate(s, u, v)  ! s + u v
    real(c_double), intent(in) :: s, u, v
    accumulate = s + (u * v)
  end function

  ! The volume of the tet with corners corner(1:4), as pipeline::tet_volume.
  real(c_double) function tet_volume(x, corner)
    real(c_double), intent(in) :: x(:, :)
    integer(soma_lo), intent(in) :: corner(4)
    real(c_double) :: e(3, 3), cx, cy, cz, det
    integer :: i
    do i = 1, 3
      e(1, i) = x(corner(2), i) - x(corner(1), i)
      e(2, i) = x(corner(3), i) - x(corner(1), i)
      e(3, i) = x(corner(4), i) - x(corner(1), i)
    end do
    cx = (e(2, 2) * e(3, 3)) - (e(2, 3) * e(3, 2))
    cy = (e(2, 3) * e(3, 1)) - (e(2, 1) * e(3, 3))
    cz = (e(2, 1) * e(3, 2)) - (e(2, 2) * e(3, 1))
    det = ((e(1, 1) * cx) + (e(1, 2) * cy)) + (e(1, 3) * cz)
    tet_volume = abs(det) / 6.0d0
  end function

end program pipeline
