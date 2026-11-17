! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

!> SOMA from Fortran 2008: meshes, fields, I/O, parallel operations, smoothing,
!> refinement and remap, over the C API.
!>
!> Conventions:
!> - Local ids, CSR offsets and segment bounds are 1-based; SOMA's "no id"
!>   (-1) is 0. Global ids and MPI ranks are never shifted.
!> - Coordinates are shared with the mesh as x(n, 3) (column-major, no copy);
!>   get_coords/set_coords copy to and from an interleaved xyz(3, n). Fields
!>   are shared as u(rows, ncomp).
!> - Connectivity is a 1-based copy made once per change of the mesh's
!>   topology, then shared: x(conn(k), :) are the corners' coordinates.
!> - Pointers into a mesh are valid until a call changes its topology or
!>   numbering (an edit's apply, compress, ghosts, adjacencies) or it is freed;
!>   field pointers also until a field is added or removed.
!> - Every procedure takes an optional integer ierr. Present, it receives the
!>   status (SOMA_OK or SOMA_ERR_*); absent, a failure writes SOMA's message
!>   to stderr and stops the program.
!> - Handles are reference counted: assignment shares the object, and the
!>   last handle freed (by free, or when it goes out of scope) frees it.
module soma
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: error_unit
  use soma_kinds
  use soma_c_api
  implicit none
  private

  public :: soma_lo, soma_offset, soma_go, soma_real
  public :: c_double, c_int, c_long, c_int8_t, c_int32_t, c_int64_t

  ! Status codes.
  integer, parameter, public :: SOMA_OK = 0, SOMA_ERR_INVALID_ARGUMENT = 1, SOMA_ERR_OUT_OF_RANGE = 2, &
                                SOMA_ERR_TYPE = 3, SOMA_ERR_STATE = 4, SOMA_ERR_RUNTIME = 5
  ! Entity shapes.
  integer, parameter, public :: SOMA_VERTEX = 0, SOMA_EDGE = 1, SOMA_TRI = 2, SOMA_QUAD = 3, SOMA_POLYGON = 4, &
                                SOMA_TET = 5, SOMA_PYRAMID = 6, SOMA_PRISM = 7, SOMA_KNIFE = 8, SOMA_HEX = 9, &
                                SOMA_POLYHEDRON = 10
  ! Orderings of a built mesh.
  integer, parameter, public :: SOMA_ORDERING_DEFAULT = -1, SOMA_ORDERING_FILE = 0, &
                                SOMA_ORDERING_TYPE_MAJOR_SFC = 1, SOMA_ORDERING_TILED = 2
  ! Ownership classes of a segment.
  integer, parameter, public :: SOMA_INTERIOR = 0, SOMA_SHARED = 1, SOMA_SHARED_COPY = 2, SOMA_GHOST = 3

  !> A communicator: the ranks a collective call runs on.
  type, public :: soma_comm
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: rank => comm_rank
    procedure :: size => comm_size
    procedure :: barrier => comm_barrier
    procedure :: free => comm_free
    procedure, private :: comm_assign
    generic :: assignment(=) => comm_assign
    final :: comm_final
  end type

  !> One rank's vertices and elements, by global id, for soma_build_local_part.
  type, public :: soma_builder
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: add_vertices => builder_add_vertices
    procedure :: add_elements => builder_add_elements
    procedure :: free => builder_free
    procedure, private :: builder_assign
    generic :: assignment(=) => builder_assign
    final :: builder_final
  end type

  !> A run of entities alike: ids first..last (1-based, inclusive).
  type, public :: soma_segment
    integer(soma_lo) :: first = 0, last = -1
    integer :: topology = 0, arity = 0, ownership = 0
  end type

  !> One rank's part of a mesh.
  type, public :: soma_mesh
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: dim => mesh_dim
    procedure :: count => mesh_count
    procedure :: owned_count => mesh_owned_count
    procedure :: interior_count => mesh_interior_count
    procedure :: segments => mesh_segments
    procedure :: coords => mesh_coords
    procedure :: get_coords => mesh_get_coords
    procedure :: set_coords => mesh_set_coords
    procedure :: gids => mesh_gids
    procedure :: conn => mesh_conn
    procedure :: topology => mesh_topology
    procedure :: owner_ranks => mesh_owner_ranks
    procedure :: build_adjacencies => mesh_build_adjacencies
    procedure :: free => mesh_free
    procedure, private :: mesh_assign
    generic :: assignment(=) => mesh_assign
    final :: mesh_final
  end type

  interface soma_comm
    module procedure comm_from_fortran
  end interface
  interface soma_builder
    module procedure builder_new
  end interface

  public :: soma_initialize, soma_finalize, soma_error_message, soma_live_handles
  public :: soma_comm_world, soma_comm_self
  public :: soma_build_local_part, soma_box_mesh

  ! Zero-size targets for pointers to empty arrays.
  real(c_double), target, save :: empty_real2(0, 3)
  integer(soma_lo), target, save :: empty_lo(0)
  integer(soma_go), target, save :: empty_go(0)
  integer(c_int8_t), target, save :: empty_i8(0)
  integer(soma_offset), target, save :: one_offset(1) = [1]

contains

  ! ---- Errors -------------------------------------------------------------

  !> The calling thread's last error message.
  function soma_error_message() result(msg)
    character(:), allocatable :: msg
    character(kind=c_char) :: buf(1024)
    integer(c_size_t) :: n
    integer :: i, k
    n = c_last_error(buf, int(size(buf), c_size_t))
    k = int(min(n, int(size(buf) - 1, c_size_t)))
    allocate (character(k) :: msg)
    do i = 1, k
      msg(i:i) = buf(i)
    end do
  end function

  ! Hands status to ierr, or stops with the message when ierr is absent. SOMA
  ! is finalized first: a process that exits with Kokkos initialized aborts.
  subroutine finish(status, ierr)
    integer(c_int), intent(in) :: status
    integer, intent(out), optional :: ierr
    integer(c_int) :: ignored
    if (present(ierr)) then
      ierr = status
    else if (status /= SOMA_OK) then
      write (error_unit, '(2a)') 'soma: ', soma_error_message()
      ignored = c_finalize()
      error stop 1
    end if
  end subroutine

  ! A failure found on the Fortran side: recorded as SOMA's last error.
  subroutine fail(status, message, ierr)
    integer, intent(in) :: status
    character(*), intent(in) :: message
    integer, intent(out), optional :: ierr
    call finish(c_set_last_error(int(status, c_int), message // c_null_char), ierr)
  end subroutine

  ! A NUL-terminated copy of s without its trailing blanks.
  function c_str(s) result(cs)
    character(*), intent(in) :: s
    character(:, kind=c_char), allocatable :: cs
    cs = trim(s) // c_null_char
  end function

  ! ---- Runtime ------------------------------------------------------------

  !> Initializes SOMA (Kokkos) unless it is; after MPI_Init.
  subroutine soma_initialize(ierr)
    integer, intent(out), optional :: ierr
    call finish(c_initialize(), ierr)
  end subroutine

  !> Finalizes what soma_initialize initialized; before MPI_Finalize. Objects
  !> still alive leave their memory to the operating system.
  subroutine soma_finalize(ierr)
    integer, intent(out), optional :: ierr
    call finish(c_finalize(), ierr)
  end subroutine

  !> Test hook: the number of live handles.
  integer(c_long) function soma_live_handles()
    integer(c_int) :: status
    status = c_testing_live_handles(soma_live_handles)
  end function

  ! ---- Communicators ------------------------------------------------------

  !> A communicator from a Fortran MPI handle (mpi_f08: pass comm%MPI_VAL).
  function comm_from_fortran(fcomm, ierr) result(c)
    integer, intent(in) :: fcomm
    integer, intent(out), optional :: ierr
    type(soma_comm) :: c
    call finish(c_comm_create_f(int(fcomm, c_int), c%ptr), ierr)
  end function

  !> Every rank (MPI_COMM_WORLD), or the one rank without MPI.
  function soma_comm_world(ierr) result(c)
    integer, intent(out), optional :: ierr
    type(soma_comm) :: c
    call finish(c_comm_world(c%ptr), ierr)
  end function

  !> This rank alone.
  function soma_comm_self(ierr) result(c)
    integer, intent(out), optional :: ierr
    type(soma_comm) :: c
    call finish(c_comm_self(c%ptr), ierr)
  end function

  integer function comm_rank(self, ierr)
    class(soma_comm), intent(in) :: self
    integer, intent(out), optional :: ierr
    integer(c_int) :: r
    r = -1
    call finish(c_comm_rank(self%ptr, r), ierr)
    comm_rank = r
  end function

  integer function comm_size(self, ierr)
    class(soma_comm), intent(in) :: self
    integer, intent(out), optional :: ierr
    integer(c_int) :: s
    s = -1
    call finish(c_comm_size(self%ptr, s), ierr)
    comm_size = s
  end function

  subroutine comm_barrier(self, ierr)
    class(soma_comm), intent(in) :: self
    integer, intent(out), optional :: ierr
    call finish(c_comm_barrier(self%ptr), ierr)
  end subroutine

  subroutine comm_free(self)
    class(soma_comm), intent(inout) :: self
    call c_comm_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  subroutine comm_assign(lhs, rhs)
    class(soma_comm), intent(inout) :: lhs
    type(soma_comm), intent(in) :: rhs
    type(c_ptr) :: p
    integer(c_int) :: status
    p = c_null_ptr
    if (c_associated(rhs%ptr)) status = c_comm_clone(rhs%ptr, p)
    call c_comm_free(lhs%ptr)
    lhs%ptr = p
  end subroutine

  subroutine comm_final(self)
    type(soma_comm), intent(inout) :: self
    call c_comm_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  ! ---- Builders -----------------------------------------------------------

  !> A builder of a mesh of dimension dim.
  function builder_new(dim, ierr) result(b)
    integer, intent(in) :: dim
    integer, intent(out), optional :: ierr
    type(soma_builder) :: b
    call finish(c_builder_create(int(dim, c_int), b%ptr), ierr)
  end function

  !> Vertices by global id, with their coordinates xyz(3, n).
  subroutine builder_add_vertices(self, gids, xyz, ierr)
    class(soma_builder), intent(inout) :: self
    integer(soma_go), intent(in), target, contiguous :: gids(:)
    real(c_double), intent(in), target, contiguous :: xyz(:, :)
    integer, intent(out), optional :: ierr
    if (size(xyz, 1) /= 3 .or. size(xyz, 2) /= size(gids)) then
      call fail(SOMA_ERR_INVALID_ARGUMENT, 'add_vertices: xyz must be (3, size(gids))', ierr)
      return
    end if
    call finish(c_builder_add_vertices(self%ptr, int(size(gids), soma_lo), loc_go(gids), loc_real2(xyz)), ierr)
  end subroutine

  !> Elements of one topology by global id, with their corners' vertex gids,
  !> element after element.
  subroutine builder_add_elements(self, topology, gids, vertex_gids, ierr)
    class(soma_builder), intent(inout) :: self
    integer, intent(in) :: topology
    integer(soma_go), intent(in), target, contiguous :: gids(:), vertex_gids(:)
    integer, intent(out), optional :: ierr
    call finish(c_builder_add_elements(self%ptr, int(topology, c_int), int(size(gids), soma_lo), loc_go(gids), &
                                       int(size(vertex_gids), soma_offset), loc_go(vertex_gids)), ierr)
  end subroutine

  subroutine builder_free(self)
    class(soma_builder), intent(inout) :: self
    call c_builder_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  subroutine builder_assign(lhs, rhs)
    class(soma_builder), intent(inout) :: lhs
    type(soma_builder), intent(in) :: rhs
    type(c_ptr) :: p
    integer(c_int) :: status
    p = c_null_ptr
    if (c_associated(rhs%ptr)) status = c_builder_clone(rhs%ptr, p)
    call c_builder_free(lhs%ptr)
    lhs%ptr = p
  end subroutine

  subroutine builder_final(self)
    type(soma_builder), intent(inout) :: self
    call c_builder_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  !> Collective: this rank's part of the mesh the builders of every rank
  !> describe. Consumes the builder.
  function soma_build_local_part(comm, builder, ordering, ierr) result(m)
    type(soma_comm), intent(in) :: comm
    type(soma_builder), intent(inout) :: builder
    integer, intent(in), optional :: ordering
    integer, intent(out), optional :: ierr
    type(soma_mesh) :: m
    integer(c_int) :: o
    o = SOMA_ORDERING_DEFAULT
    if (present(ordering)) o = int(ordering, c_int)
    call finish(c_build_local_part(comm%ptr, builder%ptr, o, m%ptr), ierr)
  end function

  !> Collective: a box [0, lx] x [0, ly] x [0, lz] of hex, tet, quad or tri
  !> elements (nz = 1 for quad and tri), in layers across the ranks.
  function soma_box_mesh(comm, nx, ny, nz, element, lx, ly, lz, ordering, ierr) result(m)
    type(soma_comm), intent(in) :: comm
    integer, intent(in) :: nx, ny
    integer, intent(in), optional :: nz, element, ordering
    real(c_double), intent(in), optional :: lx, ly, lz
    integer, intent(out), optional :: ierr
    type(soma_mesh) :: m
    integer(c_int) :: z, e, o
    real(c_double) :: a, b, c
    z = 1
    e = SOMA_HEX
    o = SOMA_ORDERING_FILE
    a = 1
    b = 1
    c = 1
    if (present(nz)) z = int(nz, c_int)
    if (present(element)) e = int(element, c_int)
    if (present(ordering)) o = int(ordering, c_int)
    if (present(lx)) a = lx
    if (present(ly)) b = ly
    if (present(lz)) c = lz
    call finish(c_box_mesh(comm%ptr, int(nx, c_int), int(ny, c_int), z, e, a, b, c, o, m%ptr), ierr)
  end function

  ! ---- Mesh queries ---------------------------------------------------------

  integer function mesh_dim(self, ierr)
    class(soma_mesh), intent(in) :: self
    integer, intent(out), optional :: ierr
    integer(c_int) :: d
    d = -1
    call finish(c_mesh_dim(self%ptr, d), ierr)
    mesh_dim = d
  end function

  !> Entities of dimension d held here (owned, copies and ghosts).
  integer(soma_lo) function mesh_count(self, d, ierr)
    class(soma_mesh), intent(in) :: self
    integer, intent(in) :: d
    integer, intent(out), optional :: ierr
    mesh_count = -1
    call finish(c_mesh_count(self%ptr, int(d, c_int), mesh_count), ierr)
  end function

  !> Entities of dimension d owned here: ids 1..owned_count.
  integer(soma_lo) function mesh_owned_count(self, d, ierr)
    class(soma_mesh), intent(in) :: self
    integer, intent(in) :: d
    integer, intent(out), optional :: ierr
    mesh_owned_count = -1
    call finish(c_mesh_owned_count(self%ptr, int(d, c_int), mesh_owned_count), ierr)
  end function

  !> Entities of dimension d held by this rank alone: ids 1..interior_count.
  integer(soma_lo) function mesh_interior_count(self, d, ierr)
    class(soma_mesh), intent(in) :: self
    integer, intent(in) :: d
    integer, intent(out), optional :: ierr
    mesh_interior_count = -1
    call finish(c_mesh_interior_count(self%ptr, int(d, c_int), mesh_interior_count), ierr)
  end function

  !> Runs of entities alike, with 1-based bounds.
  function mesh_segments(self, d, ierr) result(segs)
    class(soma_mesh), intent(in) :: self
    integer, intent(in) :: d
    integer, intent(out), optional :: ierr
    type(soma_segment), allocatable :: segs(:)
    integer(c_int) :: n, i, t, a, o, status
    integer(soma_lo) :: b, e
    n = 0
    status = c_mesh_segment_count(self%ptr, int(d, c_int), n)
    allocate (segs(n))
    do i = 1, n
      if (status /= SOMA_OK) exit
      status = c_mesh_segment(self%ptr, int(d, c_int), i - 1, b, e, t, a, o)
      segs(i) = soma_segment(b + 1, e, t, a, o)
    end do
    call finish(status, ierr)
  end function

  !> The vertex coordinates x(n, 3), writable, shared with the mesh.
  function mesh_coords(self, ierr) result(x)
    class(soma_mesh), intent(in) :: self
    integer, intent(out), optional :: ierr
    real(c_double), pointer :: x(:, :)
    type(c_ptr) :: p
    integer(soma_lo) :: n
    integer(c_int) :: status
    x => empty_real2
    n = 0
    status = c_mesh_coords(self%ptr, p, n)
    if (status == SOMA_OK .and. n > 0) call c_f_pointer(p, x, [int(n), 3])
    call finish(status, ierr)
  end function

  !> Copies the first size(xyz, 2) vertices' coordinates into xyz(3, :).
  subroutine mesh_get_coords(self, xyz, ierr)
    class(soma_mesh), intent(in) :: self
    real(c_double), intent(inout), target, contiguous :: xyz(:, :)
    integer, intent(out), optional :: ierr
    if (size(xyz, 1) /= 3) then
      call fail(SOMA_ERR_INVALID_ARGUMENT, 'get_coords: xyz must have 3 rows', ierr)
      return
    end if
    call finish(c_mesh_get_coords_interleaved(self%ptr, int(size(xyz, 2), soma_lo), loc_real2(xyz)), ierr)
  end subroutine

  !> Sets the first size(xyz, 2) vertices' coordinates from xyz(3, :).
  subroutine mesh_set_coords(self, xyz, ierr)
    class(soma_mesh), intent(inout) :: self
    real(c_double), intent(in), target, contiguous :: xyz(:, :)
    integer, intent(out), optional :: ierr
    if (size(xyz, 1) /= 3) then
      call fail(SOMA_ERR_INVALID_ARGUMENT, 'set_coords: xyz must have 3 rows', ierr)
      return
    end if
    call finish(c_mesh_set_coords_interleaved(self%ptr, int(size(xyz, 2), soma_lo), loc_real2(xyz)), ierr)
  end subroutine

  !> Global ids of dimension d, shared, not to be written.
  function mesh_gids(self, d, ierr) result(g)
    class(soma_mesh), intent(in) :: self
    integer, intent(in) :: d
    integer, intent(out), optional :: ierr
    integer(soma_go), pointer :: g(:)
    type(c_ptr) :: p
    integer(soma_lo) :: n
    integer(c_int) :: status
    g => empty_go
    n = 0
    status = c_mesh_gids(self%ptr, int(d, c_int), p, n)
    if (status == SOMA_OK .and. n > 0) call c_f_pointer(p, g, [n])
    call finish(status, ierr)
  end function

  !> Connectivity of dimension d, 1-based: entity e's corners are
  !> conn(offsets(e) : offsets(e + 1) - 1). Shared, not to be written.
  subroutine mesh_conn(self, d, conn, offsets, ierr)
    class(soma_mesh), intent(in) :: self
    integer, intent(in) :: d
    integer(soma_lo), pointer, intent(out) :: conn(:)
    integer(soma_offset), pointer, intent(out) :: offsets(:)
    integer, intent(out), optional :: ierr
    type(c_ptr) :: pc, po
    integer(soma_lo) :: n
    integer(c_int) :: status
    conn => empty_lo
    offsets => one_offset
    n = 0
    status = c_mesh_conn1(self%ptr, int(d, c_int), pc, po, n)
    if (status == SOMA_OK) then
      call c_f_pointer(po, offsets, [n + 1])
      if (offsets(n + 1) > 1) call c_f_pointer(pc, conn, [offsets(n + 1) - 1])
    end if
    call finish(status, ierr)
  end subroutine

  !> Each entity's shape (SOMA_HEX, ...); vertices store none (size 0).
  function mesh_topology(self, d, ierr) result(t)
    class(soma_mesh), intent(in) :: self
    integer, intent(in) :: d
    integer, intent(out), optional :: ierr
    integer(c_int8_t), pointer :: t(:)
    type(c_ptr) :: p
    integer(soma_lo) :: n
    integer(c_int) :: status
    t => empty_i8
    n = 0
    status = c_mesh_topology(self%ptr, int(d, c_int), p, n)
    if (status == SOMA_OK .and. n > 0) call c_f_pointer(p, t, [n])
    call finish(status, ierr)
  end function

  !> The rank owning each entity of dimension d (a copy).
  function mesh_owner_ranks(self, comm, d, ierr) result(owners)
    class(soma_mesh), intent(in) :: self
    type(soma_comm), intent(in) :: comm
    integer, intent(in) :: d
    integer, intent(out), optional :: ierr
    integer, allocatable, target :: owners(:)
    integer(soma_lo) :: n
    integer(c_int) :: status
    n = 0
    status = c_mesh_count(self%ptr, int(d, c_int), n)
    allocate (owners(max(n, 0_soma_lo)))
    if (status == SOMA_OK) status = c_mesh_owner_ranks(comm%ptr, self%ptr, int(d, c_int), loc_int(owners))
    call finish(status, ierr)
  end function

  !> Collective: builds the graphs pairs(1, k) -> pairs(2, k) and the
  !> bridges (bridges(1, k) through bridges(2, k)).
  subroutine mesh_build_adjacencies(self, comm, pairs, bridges, ierr)
    class(soma_mesh), intent(inout) :: self
    type(soma_comm), intent(in) :: comm
    integer, intent(in), target, contiguous :: pairs(:, :)
    integer, intent(in), target, contiguous, optional :: bridges(:, :)
    integer, intent(out), optional :: ierr
    integer(c_int) :: nb
    type(c_ptr) :: pb
    nb = 0
    pb = c_null_ptr
    if (present(bridges)) then
      nb = int(size(bridges, 2), c_int)
      if (size(bridges) > 0) pb = c_loc(bridges(1, 1))
    end if
    if (size(pairs) > 0) then
      call finish(c_mesh_build_adjacencies(comm%ptr, self%ptr, int(size(pairs, 2), c_int), c_loc(pairs(1, 1)), &
                                           nb, pb), ierr)
    else
      call finish(c_mesh_build_adjacencies(comm%ptr, self%ptr, 0_c_int, c_null_ptr, nb, pb), ierr)
    end if
  end subroutine

  subroutine mesh_free(self)
    class(soma_mesh), intent(inout) :: self
    call c_mesh_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  subroutine mesh_assign(lhs, rhs)
    class(soma_mesh), intent(inout) :: lhs
    type(soma_mesh), intent(in) :: rhs
    type(c_ptr) :: p
    integer(c_int) :: status
    p = c_null_ptr
    if (c_associated(rhs%ptr)) status = c_mesh_clone(rhs%ptr, p)
    call c_mesh_free(lhs%ptr)
    lhs%ptr = p
  end subroutine

  subroutine mesh_final(self)
    type(soma_mesh), intent(inout) :: self
    call c_mesh_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  ! ---- Array addresses (c_null_ptr for an empty array) -------------------

  type(c_ptr) function loc_real2(a)
    real(c_double), intent(in), target, contiguous :: a(:, :)
    loc_real2 = c_null_ptr
    if (size(a) > 0) loc_real2 = c_loc(a(1, 1))
  end function

  type(c_ptr) function loc_go(a)
    integer(soma_go), intent(in), target, contiguous :: a(:)
    loc_go = c_null_ptr
    if (size(a) > 0) loc_go = c_loc(a(1))
  end function

  type(c_ptr) function loc_int(a)
    integer, intent(in), target, contiguous :: a(:)
    loc_int = c_null_ptr
    if (size(a) > 0) loc_int = c_loc(a(1))
  end function

end module soma
