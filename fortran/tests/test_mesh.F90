! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! The Fortran module's mesh creation and queries, with 1-based ids end to end.

program test_mesh
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
    comm = soma_comm_world()
    call hex_box(comm)
    call builder(comm)
    call empty_ranks(comm)
    call adjacencies(comm)
  end block
  call soma_finalize()
  call check_done('test_mesh')

contains

  subroutine hex_box(comm)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh) :: m
    real(c_double), pointer :: x(:, :)
    integer(soma_lo), pointer :: conn(:)
    integer(soma_offset), pointer :: off(:)
    integer(soma_go), pointer :: gids(:)
    integer(c_int8_t), pointer :: topo(:)
    real(c_double), allocatable :: xyz(:, :)
    type(soma_segment), allocatable :: segs(:)
    integer(soma_lo) :: e, nc
    integer(soma_offset) :: k
    integer :: i
    real(c_double) :: lo(3), h(3)
    logical :: inside
    integer, allocatable :: owners(:)

    m = soma_box_mesh(comm, 2, 3, 4)
    call check(m%dim() == 3, 'dim')
    call check(allsum(int(m%owned_count(3), 8)) == 24, 'owned cells')
    call check(allsum(int(m%owned_count(0), 8)) == 3 * 4 * 5, 'owned vertices')
    x => m%coords()
    call check(size(x, 1) == m%count(0) .and. size(x, 2) == 3, 'coords shape (n, 3)')
    call m%conn(3, conn, off)
    nc = m%count(3)
    call check(size(off) == nc + 1, 'count + 1 offsets')
    if (nc > 0) call check(off(1) == 1, 'offsets start at 1')
    ! Every corner of cell e, x(conn(k), :), lies in the cell's box: ids are 1-based.
    h = [0.5d0, 1d0 / 3, 0.25d0]
    inside = .true.
    do e = 1, nc
      call check(off(e + 1) - off(e) == 8, 'eight corners')
      do i = 1, 3
        lo(i) = minval(x(conn(off(e):off(e + 1) - 1), i))
      end do
      do k = off(e), off(e + 1) - 1
        do i = 1, 3
          inside = inside .and. x(conn(k), i) >= lo(i) - 1d-12 .and. x(conn(k), i) <= lo(i) + h(i) + 1d-12
        end do
      end do
    end do
    call check(inside, 'x(conn(k), :) gives the corners of the cell')
    call check(all(conn >= 1) .and. all(conn <= m%count(0)), 'conn in 1..count(0)')

    ! Interleaved copies, and the sharing of x.
    allocate (xyz(3, m%count(0)))
    call m%get_coords(xyz)
    call check(all(xyz == transpose(x)), 'get_coords gives xyz(:, v) = x(v, :)')
    if (m%count(0) > 0) then
      x(1, 2) = 42d0
      call m%get_coords(xyz)
      call check(xyz(2, 1) == 42d0, 'a write through x is the mesh''s')
      xyz(3, 1) = -1d0
      call m%set_coords(xyz(:, 1:1))
      call check(x(1, 3) == -1d0, 'set_coords writes the mesh')
    end if
    deallocate (xyz)
    allocate (xyz(2, 1))
    call m%get_coords(xyz, ierr)
    call check(ierr == SOMA_ERR_INVALID_ARGUMENT, 'get_coords needs 3 rows')

    gids => m%gids(3)
    call check(size(gids) == nc, 'gids of cells')
    topo => m%topology(3)
    call check(all(topo == SOMA_HEX), 'every cell a hex')
    segs = m%segments(3)
    if (size(segs) > 0) then
      call check(segs(1)%first == 1, 'segments start at 1')
      call check(segs(size(segs))%last == nc, 'segments end at count')
    end if
    owners = m%owner_ranks(comm, 0)
    call check(size(owners) == m%count(0), 'one owner per vertex')
    call check(all(owners(1:m%owned_count(0)) == my_rank()), 'owned vertices are mine')
  end subroutine

  subroutine builder(comm)
    type(soma_comm), intent(in) :: comm
    type(soma_builder) :: b
    type(soma_mesh) :: m
    real(c_double) :: xyz(3, 8)
    integer(soma_go) :: vg(8)
    integer :: i
    b = soma_builder(3)
    if (my_rank() == 0) then
      vg = [(int(i, soma_go), i = 1, 8)]
      xyz = reshape([0d0, 0d0, 0d0, 1d0, 0d0, 0d0, 1d0, 1d0, 0d0, 0d0, 1d0, 0d0, &
                     0d0, 0d0, 1d0, 1d0, 0d0, 1d0, 1d0, 1d0, 1d0, 0d0, 1d0, 1d0], [3, 8])
      call b%add_vertices(vg, xyz)
      call b%add_elements(SOMA_HEX, [1_soma_go], vg)
    end if
    m = soma_build_local_part(comm, b)
    call check(allsum(int(m%owned_count(3), 8)) == 1, 'one hex built')
    m = soma_build_local_part(comm, b, ierr=ierr)
    call check(ierr == SOMA_ERR_STATE, 'a builder builds once')
  end subroutine

  subroutine empty_ranks(comm)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh) :: m
    real(c_double), pointer :: x(:, :)
    integer(soma_lo), pointer :: conn(:)
    integer(soma_offset), pointer :: off(:)
    real(c_double), allocatable :: xyz(:, :)
    integer, allocatable :: owners(:)
    m = soma_box_mesh(comm, 2, 2, 1, element=SOMA_QUAD)  ! ranks from 2 on hold nothing
    x => m%coords()
    call m%conn(2, conn, off)
    call check(size(off) == m%count(2) + 1, 'offsets on an empty rank')
    allocate (xyz(3, m%count(0)))
    call m%get_coords(xyz)
    owners = m%owner_ranks(comm, 2)
    call check(size(owners) == m%count(2), 'owners on an empty rank')
  end subroutine

  subroutine adjacencies(comm)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh) :: m
    integer(soma_lo), pointer :: conn(:)
    integer(soma_offset), pointer :: off(:)
    m = soma_box_mesh(comm, 2, 2, 3)
    call m%build_adjacencies(comm, reshape([3, 1], [2, 1]))
    call m%conn(1, conn, off)
    if (m%count(3) > 0) call check(m%count(1) > 0, 'edges built')
    call check(all(off(2:) - off(:size(off) - 1) == 2), 'two corners an edge')
  end subroutine

end program
