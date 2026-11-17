! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

! The Fortran module's smoothing, refinement and remap.

program test_algorithms
#if SOMA_TEST_MPI
  use mpi
#endif
  use soma
  use soma_check
  implicit none
  ! The tolerance of computed values: these tests check that values pass
  ! through the bindings, not the algorithms' accuracy (the C++ tests pin that,
  ! to 1e-14). 1e-10 leaves room for other compilers, rank and thread counts,
  ! and still catches a wrong or missing value (about 1/cells, 1e-3 here).
  real(c_double), parameter :: tolerance = 1d-10
  integer :: ierr

#if SOMA_TEST_MPI
  call MPI_Init(ierr)
#endif
  call soma_initialize()
  block
    type(soma_comm) :: comm
    comm = soma_comm_world()
    call smooth(comm)
    call refine(comm)
    call remap(comm, 4)
    call remap(comm, 2)
  end block
  call soma_finalize()
  call check_done('test_algorithms')

contains

  subroutine smooth(comm)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh) :: m
    real(c_double), pointer :: x(:, :)
    real(c_double), allocatable :: grid(:, :)
    logical, allocatable :: fixed(:)
    integer(soma_go), pointer :: g(:)
    integer :: done
    real(c_double) :: largest, before
    integer(soma_lo) :: v

    m = soma_box_mesh(comm, 4, 4, 4)
    call m%add_ghosts(comm, 1, 0)
    x => m%coords()
    grid = x
    fixed = any(abs(x) < 1d-12 .or. abs(x - 1) < 1d-12, dim=2)
    g => m%gids(0)
    do v = 1, m%count(0)
      if (fixed(v)) cycle
      x(v, :) = x(v, :) + 1d-2 * [sin(real(g(v), 8)), cos(real(g(v), 8)), sin(2 * real(g(v), 8))]
    end do
    before = 0
    if (size(x) > 0) before = maxval(abs(x - grid))
    call soma_laplacian_smooth(comm, m, iterations=10, fixed=fixed, done=done, largest_move=largest)
    call check(done == 10, 'ten sweeps')
    call check(all(x(pack([(v, v = 1, m%count(0))], fixed), :) == grid(pack([(v, v = 1, m%count(0))], fixed), :)), &
               'the fixed vertices stay')
    if (size(x) > 0) call check(maxval(abs(x - grid)) <= before, 'smoothing moves the interior toward the grid')
    call soma_laplacian_smooth(comm, m, fixed=fixed(1:max(0, size(fixed) - 1)), ierr=ierr)
    call check(ierr == SOMA_ERR_INVALID_ARGUMENT .or. size(fixed) == 0, 'a short fixed mask fails')
    ! An empty mask is a mask, not "no mask": on a rank with vertices it fails, on every rank.
    call soma_laplacian_smooth(comm, m, fixed=fixed(1:0), ierr=ierr)
    call check(ierr == SOMA_ERR_INVALID_ARGUMENT .or. allsum(int(size(fixed), 8)) == 0, 'an empty fixed mask fails')
  end subroutine

  subroutine refine(comm)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh) :: m, fine
    type(soma_refine) :: r
    integer(soma_lo), allocatable :: parent(:), children(:), hanging(:)
    integer(soma_offset), allocatable :: off(:), hoff(:)
    real(c_double), allocatable :: w(:)
    logical, allocatable :: marks(:)
    integer(soma_lo) :: p
    integer(soma_offset) :: k
    integer :: i

    m = soma_box_mesh(comm, 2, 2, 2, element=SOMA_TET)
    r = soma_refine_uniform(comm, m, 1)
    fine = r%mesh()
    call check(allsum(int(fine%owned_count(3), 8)) == 8 * 48, 'uniform refinement: 8 children a tet')
    parent = r%parent()
    off = r%child_offsets()
    children = r%children()
    call check(size(parent) == fine%count(3), 'one parent a cell')
    if (size(off) > 0) call check(off(1) == 1 .and. off(size(off)) == size(children) + 1, '1-based offsets')
    do p = 1, size(off) - 1
      do k = off(p), off(p + 1) - 1
        call check(parent(children(k)) == p, 'a child''s parent, 1-based')
      end do
    end do
    call check(size(r%dropped()) == 0, 'no field dropped')

    m = soma_box_mesh(comm, 2, 2, 2)
    allocate (marks(m%count(3)))
    marks = .false.
    if (m%owned_count(3) > 0) marks(1) = .true.
    r = soma_refine_marked(comm, m, marks)
    hanging = r%hanging()
    hoff = r%hanging_offsets()
    w = r%hanging_weights()
    call check(size(hoff) == size(hanging) + 1, 'hanging offsets')
    call check(size(r%hanging_masters()) == size(w), 'a weight a master')
    do i = 1, size(hanging)
      call check(abs(sum(w(hoff(i):hoff(i + 1) - 1)) - 1) < tolerance, 'weights sum to one')
    end do
    call check(all(hanging >= 1), 'hanging vertices are 1-based ids')
    call check(allsum(int(size(hanging), 8)) > 0, 'one marked hex leaves hanging vertices')
  end subroutine

  subroutine remap(comm, nz)
    type(soma_comm), intent(in) :: comm
    integer, intent(in) :: nz
    type(soma_mesh) :: s, t
    type(soma_remap) :: op
    real(c_double), pointer :: us(:, :), ut(:, :)
    real(c_double) :: si, ti
    s = soma_box_mesh(comm, 4, 4, nz, element=SOMA_TET)
    t = soma_box_mesh(comm, 3, 3, 3, element=SOMA_TET)
    call s%add_field('u', 3, 1, us)
    call t%add_field('u', 3, 1, ut)
    call linear(s, us(:, 1))
    op = soma_remap(comm, s, t)
    call s%free()  ! the operator keeps both meshes
    call op%apply(comm, us(:, 1), ut(:, 1))
    si = integral(op_source_volumes(comm, nz), us(:, 1))
    ti = integral(volumes(t), ut(:, 1))
    call check(abs(si - ti) <= tolerance * abs(si), 'remap conserves the integral')
    call op%apply(comm, us(:, 1), ut(1:max(0, size(ut, 1) - 1), 1), ierr)
    call check(ierr == SOMA_ERR_INVALID_ARGUMENT, 'too few target values fail on every rank')
  end subroutine

  ! u = 1 + x + 2y + 3z at each cell's centroid.
  subroutine linear(m, u)
    type(soma_mesh), intent(in) :: m
    real(c_double), intent(out) :: u(:)
    real(c_double), pointer :: x(:, :)
    integer(soma_lo), pointer :: conn(:)
    integer(soma_offset), pointer :: off(:)
    real(c_double) :: c(3)
    integer(soma_lo) :: e
    x => m%coords()
    call m%conn(3, conn, off)
    do e = 1, m%count(3)
      c = sum(x(conn(off(e):off(e + 1) - 1), :), dim=1) / 4
      u(e) = 1 + c(1) + 2 * c(2) + 3 * c(3)
    end do
  end subroutine

  ! The source's volumes, from a fresh box (the operator holds the freed handle's mesh).
  function op_source_volumes(comm, nz) result(v)
    type(soma_comm), intent(in) :: comm
    integer, intent(in) :: nz
    real(c_double), allocatable :: v(:)
    v = volumes(soma_box_mesh(comm, 4, 4, nz, element=SOMA_TET))
  end function

  ! Volumes of the owned tets.
  function volumes(m) result(v)
    type(soma_mesh), intent(in) :: m
    real(c_double), allocatable :: v(:)
    real(c_double), pointer :: x(:, :)
    integer(soma_lo), pointer :: conn(:)
    integer(soma_offset), pointer :: off(:)
    real(c_double) :: a(3), b(3), d(3)
    integer(soma_lo) :: e
    integer(soma_offset) :: k
    x => m%coords()
    call m%conn(3, conn, off)
    allocate (v(m%owned_count(3)))
    do e = 1, size(v)
      k = off(e)
      a = x(conn(k + 1), :) - x(conn(k), :)
      b = x(conn(k + 2), :) - x(conn(k), :)
      d = x(conn(k + 3), :) - x(conn(k), :)
      v(e) = abs(a(1) * (b(2) * d(3) - b(3) * d(2)) - a(2) * (b(1) * d(3) - b(3) * d(1)) + &
                 a(3) * (b(1) * d(2) - b(2) * d(1))) / 6
    end do
  end function

  real(c_double) function integral(vol, u)
    real(c_double), intent(in) :: vol(:), u(:)
    real(c_double) :: s
#if SOMA_TEST_MPI
    integer :: e
    s = sum(vol * u(1:size(vol)))
    call MPI_Allreduce(s, integral, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, e)
#else
    integral = sum(vol * u(1:size(vol)))
#endif
  end function

end program
