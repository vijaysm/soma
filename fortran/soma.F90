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
!>   status (SOMA_OK or SOMA_ERR_*); absent, a failure flushes stdout, writes
!>   SOMA's message to stderr and ends the process (soma_abort: Kokkos
!>   finalized, then MPI_Abort, or exit status 1 without MPI).
!> - Names passed in may be of any length; names handed back are
!>   character(len=soma_name_len), and a longer one is an error, never cut off.
!> - Queries may run on several threads; calls that change a mesh may not run
!>   alongside other calls on it.
!> - Handles are reference counted: assignment shares the object, and the
!>   last handle freed (by free, or when it goes out of scope) frees it.
!>   Assignment and finalization are elemental, so arrays of handles and
!>   handle components of user types share and free element by element;
!>   allocate(source=) and transfer copy without sharing and must not be used.
module soma
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: error_unit, output_unit
  use soma_kinds
  use soma_c_api
  implicit none
  private

  public :: soma_lo, soma_offset, soma_go, soma_real, soma_name_len
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
  ! Field value types.
  integer, parameter, public :: SOMA_I8 = 0, SOMA_U8 = 1, SOMA_I32 = 2, SOMA_U32 = 3, SOMA_I64 = 4, SOMA_U64 = 5, &
                                SOMA_F32 = 6, SOMA_F64 = 7
  ! Reductions.
  integer, parameter, public :: SOMA_SUM = 0, SOMA_MIN = 1, SOMA_MAX = 2, SOMA_PROD = 3
  ! File formats.
  integer, parameter, public :: SOMA_FORMAT_LEGACY_H5M = 0, SOMA_FORMAT_SOMA = 1, SOMA_FORMAT_SOMA_AMR = 2, &
                                SOMA_FORMAT_LEGACY_VTK = 3, SOMA_FORMAT_UNKNOWN = 4
  ! Remap options.
  integer, parameter, public :: SOMA_SEARCH_GRID = 0, SOMA_SEARCH_ARBORX = 1, SOMA_SURFACE_PLANE = 0, &
                                SOMA_SURFACE_SPHERE = 1, SOMA_SPHERE_CLIP_GNOMONIC = 0, &
                                SOMA_SPHERE_CLIP_GREAT_CIRCLE = 1
  ! Marked refinement: how new vertices on rank boundaries match.
  integer, parameter, public :: SOMA_VERTICES_RECORDED = 0, SOMA_VERTICES_COORDINATES = 1

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
    procedure, private :: mesh_add_field_f64, mesh_add_field_i32, mesh_add_field_i64
    generic :: add_field => mesh_add_field_f64, mesh_add_field_i32, mesh_add_field_i64
    procedure, private :: mesh_field_f64, mesh_field_i32, mesh_field_i64
    generic :: field => mesh_field_f64, mesh_field_i32, mesh_field_i64
    procedure :: field_names => mesh_field_names
    procedure :: remove_field => mesh_remove_field
    procedure, private :: mesh_get_field_f64, mesh_get_field_i32, mesh_get_field_i64
    generic :: get_field => mesh_get_field_f64, mesh_get_field_i32, mesh_get_field_i64
    procedure, private :: mesh_set_field_f64, mesh_set_field_i32, mesh_set_field_i64
    generic :: set_field => mesh_set_field_f64, mesh_set_field_i32, mesh_set_field_i64
    procedure :: compress => mesh_compress
    procedure :: add_ghosts => mesh_add_ghosts
    procedure :: remove_ghosts => mesh_remove_ghosts
    procedure :: free => mesh_free
    procedure, private :: mesh_assign
    generic :: assignment(=) => mesh_assign
    final :: mesh_final
  end type

  !> The new id of each old id of a dimension, after compression.
  type, public :: soma_renumbering
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: new_ids => renumbering_new_ids
    procedure :: free => renumbering_free
    procedure, private :: renumbering_assign
    generic :: assignment(=) => renumbering_assign
    final :: renumbering_final
  end type

  !> Batched edits of a mesh (kept alive by the edit), applied together.
  type, public :: soma_edit
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: add_vertices => edit_add_vertices
    procedure :: add_elements => edit_add_elements
    procedure :: add_polygons => edit_add_polygons
    procedure :: remove => edit_remove
    procedure :: set_compress_threshold => edit_set_compress_threshold
    procedure :: apply => edit_apply
    procedure :: free => edit_free
    procedure, private :: edit_assign
    generic :: assignment(=) => edit_assign
    final :: edit_final
  end type

  !> The exchange of the shared copies and ghosts of one dimension.
  type, public :: soma_halo
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure, private :: halo_update_f64, halo_update_i32, halo_update_i64
    generic :: update => halo_update_f64, halo_update_i32, halo_update_i64
    procedure, private :: halo_reduce_f64, halo_reduce_i32, halo_reduce_i64
    generic :: reduce => halo_reduce_f64, halo_reduce_i32, halo_reduce_i64
    procedure :: free => halo_free
    procedure, private :: halo_assign
    generic :: assignment(=) => halo_assign
    final :: halo_final
  end type

  !> A refined mesh and its maps to the input (1-based copies).
  type, public :: soma_refine
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: mesh => refine_mesh
    procedure :: parent => refine_parent
    procedure :: children => refine_children
    procedure :: hanging => refine_hanging
    procedure :: child_offsets => refine_child_offsets
    procedure :: hanging_offsets => refine_hanging_offsets
    procedure :: hanging_masters => refine_hanging_masters
    procedure :: hanging_weights => refine_hanging_weights
    procedure :: dropped => refine_dropped
    procedure :: free => refine_free
    procedure, private :: refine_assign
    generic :: assignment(=) => refine_assign
    final :: refine_final
  end type

  !> First-order conservative remap between the cells of two meshes.
  type, public :: soma_remap
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: apply => remap_apply
    procedure :: free => remap_free
    procedure, private :: remap_assign
    generic :: assignment(=) => remap_assign
    final :: remap_final
  end type

  interface soma_comm
    module procedure comm_from_fortran
  end interface
  interface soma_edit
    module procedure edit_new
  end interface
  interface soma_halo
    module procedure halo_new
  end interface
  interface soma_remap
    module procedure remap_new
  end interface
  interface soma_builder
    module procedure builder_new
  end interface

  public :: soma_initialize, soma_finalize, soma_error_message, soma_live_handles
  public :: soma_comm_world, soma_comm_self
  public :: soma_build_local_part, soma_box_mesh
  public :: soma_read_file, soma_write_file, soma_detect_format, soma_convert_mesh_file
  public :: soma_partition_by_curve, soma_migrate, soma_rebalance
  public :: soma_laplacian_smooth, soma_refine_uniform, soma_refine_marked

  ! Zero-size targets for pointers to empty arrays.
  real(c_double), target, save :: empty_real2(0, 3)
  integer(soma_lo), target, save :: empty_lo(0)
  integer(soma_go), target, save :: empty_go(0)
  integer(c_int8_t), target, save :: empty_i8(0)
  integer(soma_offset), target, save :: one_offset(1) = [1]
  real(c_double), target, save :: empty_f64(0)
  integer(c_int32_t), target, save :: empty_i32(0)
  integer(c_int64_t), target, save :: empty_i64(0)

contains

  ! ---- Errors -------------------------------------------------------------

  !> The calling thread's last error message.
  function soma_error_message() result(msg)
    character(:), allocatable :: msg
    character(kind=c_char) :: probe(1)
    character(kind=c_char), allocatable :: buf(:)
    integer(c_size_t) :: n
    integer :: i
    n = c_last_error(probe, 0_c_size_t)  ! the length, so the message is whole
    allocate (buf(n + 1))
    n = c_last_error(buf, int(size(buf), c_size_t))
    allocate (character(int(n)) :: msg)
    do i = 1, int(n)
      msg(i:i) = buf(i)
    end do
  end function

  ! Hands status to ierr, or stops with the message when ierr is absent. SOMA
  ! is finalized first: a process that exits with Kokkos initialized aborts.
  subroutine finish(status, ierr)
    integer(c_int), intent(in) :: status
    integer, intent(out), optional :: ierr
    if (present(ierr)) then
      ierr = status
    else if (status /= SOMA_OK) then
      ! The process ends in C (MPI_Abort, or _Exit without MPI), not by error
      ! stop: what a compiler runs at error stop, and whether Kokkos is still up
      ! then, is not SOMA's to know. Output written so far is flushed first.
      flush (output_unit)
      flush (error_unit)
      call c_abort(soma_error_message() // c_null_char)
    end if
  end subroutine

  ! A failure found on the Fortran side: recorded as SOMA's last error.
  subroutine fail(status, message, ierr)
    integer, intent(in) :: status
    character(*), intent(in) :: message
    integer, intent(out), optional :: ierr
    call finish(c_set_last_error(int(status, c_int), message // c_null_char), ierr)
  end subroutine

  ! The first length characters of a C name.
  function from_c(buf, length) result(name)
    character(kind=c_char), intent(in) :: buf(:)
    integer(c_size_t), intent(in) :: length
    character(len=int(length)) :: name
    integer :: k
    do k = 1, int(length)
      name(k:k) = buf(k)
    end do
  end function

  ! The error for a name longer than soma_name_len.
  function too_long(what, length) result(message)
    character(*), intent(in) :: what
    integer(c_size_t), intent(in) :: length
    character(:), allocatable :: message
    character(len=24) :: n, limit
    write (n, '(i0)') length
    write (limit, '(i0)') soma_name_len
    message = what // ': a name of ' // trim(n) // ' characters is longer than soma_name_len (' // trim(limit) // &
              '); configure SOMA with a larger SOMA_FORTRAN_NAME_LENGTH'
  end function

  ! A NUL-terminated copy of s without its trailing blanks.
  ! A Fortran string as a C string: its characters without trailing blanks, then
  ! NUL. An array, not a deferred-length string, so every compiler passes it to
  ! the C interfaces' name(*) as it is.
  function c_str(s) result(cs)
    character(*), intent(in) :: s
    character(kind=c_char) :: cs(len_trim(s) + 1)
    integer :: k
    do k = 1, len_trim(s)
      cs(k) = s(k:k)
    end do
    cs(len_trim(s) + 1) = c_null_char
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

  impure elemental subroutine comm_assign(lhs, rhs)
    class(soma_comm), intent(inout) :: lhs
    type(soma_comm), intent(in) :: rhs
    type(c_ptr) :: p
    integer(c_int) :: status
    p = c_null_ptr
    if (c_associated(rhs%ptr)) status = c_comm_clone(rhs%ptr, p)
    call c_comm_free(lhs%ptr)
    lhs%ptr = p
  end subroutine

  impure elemental subroutine comm_final(self)
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

  impure elemental subroutine builder_assign(lhs, rhs)
    class(soma_builder), intent(inout) :: lhs
    type(soma_builder), intent(in) :: rhs
    type(c_ptr) :: p
    integer(c_int) :: status
    p = c_null_ptr
    if (c_associated(rhs%ptr)) status = c_builder_clone(rhs%ptr, p)
    call c_builder_free(lhs%ptr)
    lhs%ptr = p
  end subroutine

  impure elemental subroutine builder_final(self)
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

  impure elemental subroutine mesh_assign(lhs, rhs)
    class(soma_mesh), intent(inout) :: lhs
    type(soma_mesh), intent(in) :: rhs
    type(c_ptr) :: p
    integer(c_int) :: status
    p = c_null_ptr
    if (c_associated(rhs%ptr)) status = c_mesh_clone(rhs%ptr, p)
    call c_mesh_free(lhs%ptr)
    lhs%ptr = p
  end subroutine

  impure elemental subroutine mesh_final(self)
    type(soma_mesh), intent(inout) :: self
    call c_mesh_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  ! ---- Fields ---------------------------------------------------------------

  ! A new field's storage, or a field's, with its type, rows and components.
  integer(c_int) function field_storage(self, name, dim, ncomp, dtype, add, p, rows, nc) result(status)
    class(soma_mesh), intent(in) :: self
    character(*), intent(in) :: name
    integer, intent(in) :: dim, ncomp, dtype
    logical, intent(in) :: add
    type(c_ptr), intent(out) :: p
    integer(soma_lo), intent(out) :: rows
    integer, intent(out) :: nc
    integer(c_int) :: t, k
    p = c_null_ptr
    rows = 0
    nc = 0
    if (add) then
      status = c_mesh_add_field(self%ptr, c_str(name), int(dim, c_int), int(ncomp, c_int), int(dtype, c_int), p, rows)
      nc = ncomp
      return
    end if
    status = c_mesh_field(self%ptr, c_str(name), int(dim, c_int), t, p, rows, k)
    nc = k
    if (status == SOMA_OK .and. t /= dtype) then
      status = c_set_last_error(int(SOMA_ERR_TYPE, c_int), 'field ''' // trim(name) // ''' holds another type' // &
                                c_null_char)
    end if
  end function

  integer function dim_or_any(dim)
    integer, intent(in), optional :: dim
    dim_or_any = -1
    if (present(dim)) dim_or_any = dim
  end function

  !> A new field on every entity of dimension dim: u(rows, ncomp), shared.
  subroutine mesh_add_field_f64(self, name, dim, ncomp, u, ierr)
    class(soma_mesh), intent(inout) :: self
    character(*), intent(in) :: name
    integer, intent(in) :: dim, ncomp
    real(c_double), pointer, intent(out) :: u(:, :)
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    integer(soma_lo) :: rows
    integer :: nc
    integer(c_int) :: status
    u(1:0, 1:0) => empty_f64
    status = field_storage(self, name, dim, ncomp, SOMA_F64, .true., p, rows, nc)
    if (status == SOMA_OK) then
      if (rows > 0) then
        call c_f_pointer(p, u, [int(rows), nc])
      else
        u(1:0, 1:nc) => empty_f64  ! no rows here, but the field's components
      end if
    end if
    call finish(status, ierr)
  end subroutine

  subroutine mesh_add_field_i32(self, name, dim, ncomp, u, ierr)
    class(soma_mesh), intent(inout) :: self
    character(*), intent(in) :: name
    integer, intent(in) :: dim, ncomp
    integer(c_int32_t), pointer, intent(out) :: u(:, :)
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    integer(soma_lo) :: rows
    integer :: nc
    integer(c_int) :: status
    u(1:0, 1:0) => empty_i32
    status = field_storage(self, name, dim, ncomp, SOMA_I32, .true., p, rows, nc)
    if (status == SOMA_OK) then
      if (rows > 0) then
        call c_f_pointer(p, u, [int(rows), nc])
      else
        u(1:0, 1:nc) => empty_i32  ! no rows here, but the field's components
      end if
    end if
    call finish(status, ierr)
  end subroutine

  subroutine mesh_add_field_i64(self, name, dim, ncomp, u, ierr)
    class(soma_mesh), intent(inout) :: self
    character(*), intent(in) :: name
    integer, intent(in) :: dim, ncomp
    integer(c_int64_t), pointer, intent(out) :: u(:, :)
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    integer(soma_lo) :: rows
    integer :: nc
    integer(c_int) :: status
    u(1:0, 1:0) => empty_i64
    status = field_storage(self, name, dim, ncomp, SOMA_I64, .true., p, rows, nc)
    if (status == SOMA_OK) then
      if (rows > 0) then
        call c_f_pointer(p, u, [int(rows), nc])
      else
        u(1:0, 1:nc) => empty_i64  ! no rows here, but the field's components
      end if
    end if
    call finish(status, ierr)
  end subroutine

  !> A field's values u(rows, ncomp), shared (on dimension dim when the name
  !> is on several).
  subroutine mesh_field_f64(self, name, u, dim, ierr)
    class(soma_mesh), intent(in) :: self
    character(*), intent(in) :: name
    real(c_double), pointer, intent(out) :: u(:, :)
    integer, intent(in), optional :: dim
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    integer(soma_lo) :: rows
    integer :: nc
    integer(c_int) :: status
    u(1:0, 1:0) => empty_f64
    status = field_storage(self, name, dim_or_any(dim), 0, SOMA_F64, .false., p, rows, nc)
    if (status == SOMA_OK) then
      if (rows > 0) then
        call c_f_pointer(p, u, [int(rows), nc])
      else
        u(1:0, 1:nc) => empty_f64  ! no rows here, but the field's components
      end if
    end if
    call finish(status, ierr)
  end subroutine

  subroutine mesh_field_i32(self, name, u, dim, ierr)
    class(soma_mesh), intent(in) :: self
    character(*), intent(in) :: name
    integer(c_int32_t), pointer, intent(out) :: u(:, :)
    integer, intent(in), optional :: dim
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    integer(soma_lo) :: rows
    integer :: nc
    integer(c_int) :: status
    u(1:0, 1:0) => empty_i32
    status = field_storage(self, name, dim_or_any(dim), 0, SOMA_I32, .false., p, rows, nc)
    if (status == SOMA_OK) then
      if (rows > 0) then
        call c_f_pointer(p, u, [int(rows), nc])
      else
        u(1:0, 1:nc) => empty_i32  ! no rows here, but the field's components
      end if
    end if
    call finish(status, ierr)
  end subroutine

  subroutine mesh_field_i64(self, name, u, dim, ierr)
    class(soma_mesh), intent(in) :: self
    character(*), intent(in) :: name
    integer(c_int64_t), pointer, intent(out) :: u(:, :)
    integer, intent(in), optional :: dim
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    integer(soma_lo) :: rows
    integer :: nc
    integer(c_int) :: status
    u(1:0, 1:0) => empty_i64
    status = field_storage(self, name, dim_or_any(dim), 0, SOMA_I64, .false., p, rows, nc)
    if (status == SOMA_OK) then
      if (rows > 0) then
        call c_f_pointer(p, u, [int(rows), nc])
      else
        u(1:0, 1:nc) => empty_i64  ! no rows here, but the field's components
      end if
    end if
    call finish(status, ierr)
  end subroutine

  !> Every field's name.
  function mesh_field_names(self, ierr) result(names)
    class(soma_mesh), intent(in) :: self
    integer, intent(out), optional :: ierr
    character(len=soma_name_len), allocatable :: names(:)
    character(kind=c_char) :: buf(soma_name_len + 1)
    integer(c_int) :: n, i, t, nc, dims, variable, status
    integer(c_size_t) :: length
    integer(soma_lo) :: rows
    n = 0
    status = c_mesh_field_count(self%ptr, n)
    allocate (names(n))
    names = ' '
    do i = 1, n
      if (status /= SOMA_OK) exit
      length = 0
      status = c_mesh_field_info(self%ptr, i - 1, buf, int(size(buf), c_size_t), length, t, nc, rows, dims, variable)
      if (status == SOMA_OK) names(i) = from_c(buf, length)
    end do
    if (status /= SOMA_OK .and. length > soma_name_len) then
      call fail(SOMA_ERR_OUT_OF_RANGE, too_long('field_names', length), ierr)
      return
    end if
    call finish(status, ierr)
  end function

  subroutine mesh_remove_field(self, name, dim, ierr)
    class(soma_mesh), intent(inout) :: self
    character(*), intent(in) :: name
    integer, intent(in), optional :: dim
    integer, intent(out), optional :: ierr
    call finish(c_mesh_remove_field(self%ptr, c_str(name), int(dim_or_any(dim), c_int)), ierr)
  end subroutine

  ! Copies between a field and an interleaved u(ncomp, rows) the caller holds.
  integer(c_int) function field_copy(self, name, dim, dtype, u, ncomp, rows, set) result(status)
    class(soma_mesh), intent(in) :: self
    character(*), intent(in) :: name
    integer, intent(in) :: dim, dtype, ncomp
    type(c_ptr), intent(in) :: u
    integer(soma_lo), intent(in) :: rows
    logical, intent(in) :: set
    type(c_ptr) :: p
    integer(soma_lo) :: frows
    integer :: nc
    status = field_storage(self, name, dim, 0, dtype, .false., p, frows, nc)
    if (status /= SOMA_OK) return
    if (ncomp /= nc) then
      status = c_set_last_error(int(SOMA_ERR_INVALID_ARGUMENT, c_int), 'field ''' // trim(name) // &
                                ''': the array''s first extent must be the field''s components' // c_null_char)
      return
    end if
    if (set) then
      status = c_mesh_set_field_interleaved(self%ptr, c_str(name), int(dim, c_int), int(dtype, c_int), rows, u)
    else
      status = c_mesh_get_field_interleaved(self%ptr, c_str(name), int(dim, c_int), int(dtype, c_int), rows, u)
    end if
  end function

  !> Copies the first size(u, 2) rows of a field into u(ncomp, :), interleaved.
  subroutine mesh_get_field_f64(self, name, u, dim, ierr)
    class(soma_mesh), intent(in) :: self
    character(*), intent(in) :: name
    real(c_double), intent(inout), target, contiguous :: u(:, :)
    integer, intent(in), optional :: dim
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(field_copy(self, name, dim_or_any(dim), SOMA_F64, p, size(u, 1), int(size(u, 2), soma_lo), .false.), &
                ierr)
  end subroutine

  subroutine mesh_get_field_i32(self, name, u, dim, ierr)
    class(soma_mesh), intent(in) :: self
    character(*), intent(in) :: name
    integer(c_int32_t), intent(inout), target, contiguous :: u(:, :)
    integer, intent(in), optional :: dim
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(field_copy(self, name, dim_or_any(dim), SOMA_I32, p, size(u, 1), int(size(u, 2), soma_lo), .false.), &
                ierr)
  end subroutine

  subroutine mesh_get_field_i64(self, name, u, dim, ierr)
    class(soma_mesh), intent(in) :: self
    character(*), intent(in) :: name
    integer(c_int64_t), intent(inout), target, contiguous :: u(:, :)
    integer, intent(in), optional :: dim
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(field_copy(self, name, dim_or_any(dim), SOMA_I64, p, size(u, 1), int(size(u, 2), soma_lo), .false.), &
                ierr)
  end subroutine

  !> Sets the first size(u, 2) rows of a field from u(ncomp, :), interleaved.
  subroutine mesh_set_field_f64(self, name, u, dim, ierr)
    class(soma_mesh), intent(inout) :: self
    character(*), intent(in) :: name
    real(c_double), intent(in), target, contiguous :: u(:, :)
    integer, intent(in), optional :: dim
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(field_copy(self, name, dim_or_any(dim), SOMA_F64, p, size(u, 1), int(size(u, 2), soma_lo), .true.), &
                ierr)
  end subroutine

  subroutine mesh_set_field_i32(self, name, u, dim, ierr)
    class(soma_mesh), intent(inout) :: self
    character(*), intent(in) :: name
    integer(c_int32_t), intent(in), target, contiguous :: u(:, :)
    integer, intent(in), optional :: dim
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(field_copy(self, name, dim_or_any(dim), SOMA_I32, p, size(u, 1), int(size(u, 2), soma_lo), .true.), &
                ierr)
  end subroutine

  subroutine mesh_set_field_i64(self, name, u, dim, ierr)
    class(soma_mesh), intent(inout) :: self
    character(*), intent(in) :: name
    integer(c_int64_t), intent(in), target, contiguous :: u(:, :)
    integer, intent(in), optional :: dim
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(field_copy(self, name, dim_or_any(dim), SOMA_I64, p, size(u, 1), int(size(u, 2), soma_lo), .true.), &
                ierr)
  end subroutine

  ! ---- Halo exchange --------------------------------------------------------

  !> Collective: the exchange of dimension dim of the mesh (kept alive). It
  !> fails with SOMA_ERR_STATE once the mesh's topology changes.
  function halo_new(comm, mesh, dim, ierr) result(h)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh), intent(in) :: mesh
    integer, intent(in) :: dim
    integer, intent(out), optional :: ierr
    type(soma_halo) :: h
    call finish(c_halo_create(comm%ptr, mesh%ptr, int(dim, c_int), h%ptr), ierr)
  end function

  !> Collective: owners' values overwrite every copy's; u(rows, ncomp).
  subroutine halo_update_f64(self, comm, u, ierr)
    class(soma_halo), intent(in) :: self
    type(soma_comm), intent(in) :: comm
    real(c_double), intent(inout), target, contiguous :: u(:, :)
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(c_halo_update(self%ptr, comm%ptr, int(SOMA_F64, c_int), p, int(size(u, 1), soma_lo), &
                              int(size(u, 2), c_int)), ierr)
  end subroutine

  subroutine halo_update_i32(self, comm, u, ierr)
    class(soma_halo), intent(in) :: self
    type(soma_comm), intent(in) :: comm
    integer(c_int32_t), intent(inout), target, contiguous :: u(:, :)
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(c_halo_update(self%ptr, comm%ptr, int(SOMA_I32, c_int), p, int(size(u, 1), soma_lo), &
                              int(size(u, 2), c_int)), ierr)
  end subroutine

  subroutine halo_update_i64(self, comm, u, ierr)
    class(soma_halo), intent(in) :: self
    type(soma_comm), intent(in) :: comm
    integer(c_int64_t), intent(inout), target, contiguous :: u(:, :)
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(c_halo_update(self%ptr, comm%ptr, int(SOMA_I64, c_int), p, int(size(u, 1), soma_lo), &
                              int(size(u, 2), c_int)), ierr)
  end subroutine

  !> Collective: every holder gets all holders' values combined by op (SOMA_SUM, ...).
  subroutine halo_reduce_f64(self, comm, u, op, ierr)
    class(soma_halo), intent(in) :: self
    type(soma_comm), intent(in) :: comm
    real(c_double), intent(inout), target, contiguous :: u(:, :)
    integer, intent(in) :: op
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(c_halo_reduce(self%ptr, comm%ptr, int(SOMA_F64, c_int), p, int(size(u, 1), soma_lo), &
                              int(size(u, 2), c_int), int(op, c_int)), ierr)
  end subroutine

  subroutine halo_reduce_i32(self, comm, u, op, ierr)
    class(soma_halo), intent(in) :: self
    type(soma_comm), intent(in) :: comm
    integer(c_int32_t), intent(inout), target, contiguous :: u(:, :)
    integer, intent(in) :: op
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(c_halo_reduce(self%ptr, comm%ptr, int(SOMA_I32, c_int), p, int(size(u, 1), soma_lo), &
                              int(size(u, 2), c_int), int(op, c_int)), ierr)
  end subroutine

  subroutine halo_reduce_i64(self, comm, u, op, ierr)
    class(soma_halo), intent(in) :: self
    type(soma_comm), intent(in) :: comm
    integer(c_int64_t), intent(inout), target, contiguous :: u(:, :)
    integer, intent(in) :: op
    integer, intent(out), optional :: ierr
    type(c_ptr) :: p
    p = c_null_ptr
    if (size(u) > 0) p = c_loc(u(1, 1))
    call finish(c_halo_reduce(self%ptr, comm%ptr, int(SOMA_I64, c_int), p, int(size(u, 1), soma_lo), &
                              int(size(u, 2), c_int), int(op, c_int)), ierr)
  end subroutine

  ! ---- Edits ----------------------------------------------------------------

  !> Batched edits of mesh; the edit keeps the mesh alive.
  function edit_new(mesh, ierr) result(e)
    type(soma_mesh), intent(in) :: mesh
    integer, intent(out), optional :: ierr
    type(soma_edit) :: e
    call finish(c_edit_create(mesh%ptr, e%ptr), ierr)
  end function

  !> Vertices xyz(3, n) (gids given or the next free ones); first: the id of
  !> the first added vertex.
  subroutine edit_add_vertices(self, xyz, gids, first, ierr)
    class(soma_edit), intent(inout) :: self
    real(c_double), intent(in), target, contiguous :: xyz(:, :)
    integer(soma_go), intent(in), target, contiguous, optional :: gids(:)
    integer(soma_lo), intent(out), optional :: first
    integer, intent(out), optional :: ierr
    integer(soma_lo) :: f
    integer(c_int) :: status
    type(c_ptr) :: pg
    if (size(xyz, 1) /= 3) then
      call fail(SOMA_ERR_INVALID_ARGUMENT, 'add_vertices: xyz must have 3 rows', ierr)
      return
    end if
    pg = c_null_ptr
    if (present(gids)) then
      if (size(gids) /= size(xyz, 2)) then
        call fail(SOMA_ERR_INVALID_ARGUMENT, 'add_vertices: one gid a vertex', ierr)
        return
      end if
      pg = loc_go(gids)
      if (size(gids) == 0) pg = c_null_ptr
    end if
    f = -1
    status = c_edit_add_vertices(self%ptr, int(size(xyz, 2), soma_lo), loc_real2(xyz), pg, f)
    if (present(first)) first = f + 1
    call finish(status, ierr)
  end subroutine

  !> Elements of one topology by their vertex ids (1-based, element after
  !> element; ids of vertices added by this edit count).
  subroutine edit_add_elements(self, topology, vertex_ids, gids, nodes_per_element, first, ierr)
    class(soma_edit), intent(inout) :: self
    integer, intent(in) :: topology
    integer(soma_lo), intent(in) :: vertex_ids(:)
    integer(soma_go), intent(in), target, contiguous, optional :: gids(:)
    integer, intent(in), optional :: nodes_per_element
    integer(soma_lo), intent(out), optional :: first
    integer, intent(out), optional :: ierr
    integer(soma_lo), allocatable, target :: lids(:)
    integer(soma_lo) :: f, ng
    integer(c_int) :: npe, status
    type(c_ptr) :: pg
    lids = vertex_ids - 1
    npe = 0
    if (present(nodes_per_element)) npe = int(nodes_per_element, c_int)
    pg = c_null_ptr
    ng = 0
    if (present(gids)) then
      ng = size(gids)
      if (ng > 0) pg = c_loc(gids(1))
    end if
    f = -1
    status = c_edit_add_elements(self%ptr, int(topology, c_int), int(size(lids), soma_offset), loc_lo(lids), ng, pg, &
                                 npe, f)
    if (present(first)) first = f + 1
    call finish(status, ierr)
  end subroutine

  !> Polygons: polygon k's corners are vertex_ids(offsets(k) : offsets(k + 1) - 1),
  !> both 1-based.
  subroutine edit_add_polygons(self, offsets, vertex_ids, first, ierr)
    class(soma_edit), intent(inout) :: self
    integer(soma_offset), intent(in) :: offsets(:)
    integer(soma_lo), intent(in) :: vertex_ids(:)
    integer(soma_lo), intent(out), optional :: first
    integer, intent(out), optional :: ierr
    integer(soma_offset), allocatable, target :: off(:)
    integer(soma_lo), allocatable, target :: lids(:)
    integer(soma_lo) :: f
    integer(c_int) :: status
    type(c_ptr) :: po
    off = offsets - 1
    lids = vertex_ids - 1
    po = c_null_ptr
    if (size(off) > 0) po = c_loc(off(1))
    f = -1
    status = c_edit_add_polygons(self%ptr, int(size(off), soma_lo), po, int(size(lids), soma_offset), loc_lo(lids), f)
    if (present(first)) first = f + 1
    call finish(status, ierr)
  end subroutine

  !> Entities of dimension d (ids, 1-based) that exist before this edit.
  subroutine edit_remove(self, d, ids, ierr)
    class(soma_edit), intent(inout) :: self
    integer, intent(in) :: d
    integer(soma_lo), intent(in) :: ids(:)
    integer, intent(out), optional :: ierr
    integer(soma_lo), allocatable, target :: lids(:)
    lids = ids - 1
    call finish(c_edit_remove(self%ptr, int(d, c_int), int(size(lids), soma_lo), loc_lo(lids)), ierr)
  end subroutine

  subroutine edit_set_compress_threshold(self, t, ierr)
    class(soma_edit), intent(inout) :: self
    real(c_double), intent(in) :: t
    integer, intent(out), optional :: ierr
    call finish(c_edit_set_compress_threshold(self%ptr, t), ierr)
  end subroutine

  !> Applies the edits. added(0:3) and removed(0:3) count the entities by
  !> dimension; renumbering is set when apply compressed the mesh (else
  !> freed). The edit can then be used again.
  subroutine edit_apply(self, added, removed, renumbering, ierr)
    class(soma_edit), intent(inout) :: self
    integer(soma_lo), intent(out), optional :: added(0:3), removed(0:3)
    type(soma_renumbering), intent(inout), optional :: renumbering
    integer, intent(out), optional :: ierr
    integer(soma_lo) :: a(4), r(4)
    type(c_ptr) :: p
    integer(c_int) :: status
    a = 0
    r = 0
    p = c_null_ptr
    status = c_edit_apply(self%ptr, a, r, p)
    if (present(added)) added = a
    if (present(removed)) removed = r
    if (present(renumbering)) then
      call renumbering%free()
      renumbering%ptr = p
    else
      call c_renumbering_free(p)
    end if
    call finish(status, ierr)
  end subroutine

  subroutine edit_free(self)
    class(soma_edit), intent(inout) :: self
    call c_edit_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  impure elemental subroutine edit_assign(lhs, rhs)
    class(soma_edit), intent(inout) :: lhs
    type(soma_edit), intent(in) :: rhs
    type(c_ptr) :: p
    integer(c_int) :: status
    p = c_null_ptr
    if (c_associated(rhs%ptr)) status = c_edit_clone(rhs%ptr, p)
    call c_edit_free(lhs%ptr)
    lhs%ptr = p
  end subroutine

  impure elemental subroutine edit_final(self)
    type(soma_edit), intent(inout) :: self
    call c_edit_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  !> Drops the removed entities; returns the renumbering.
  function mesh_compress(self, ierr) result(r)
    class(soma_mesh), intent(inout) :: self
    integer, intent(out), optional :: ierr
    type(soma_renumbering) :: r
    call finish(c_mesh_compress(self%ptr, r%ptr), ierr)
  end function

  !> The new id of each old id of dimension d (1-based; 0: removed).
  function renumbering_new_ids(self, d, ierr) result(ids)
    class(soma_renumbering), intent(in) :: self
    integer, intent(in) :: d
    integer, intent(out), optional :: ierr
    integer(soma_lo), allocatable :: ids(:)
    integer(soma_lo), pointer :: lids(:)
    type(c_ptr) :: p
    integer(soma_lo) :: n
    integer(c_int) :: status
    n = 0
    status = c_renumbering_new_lids(self%ptr, int(d, c_int), p, n)
    allocate (ids(0))
    if (status == SOMA_OK .and. n > 0) then
      call c_f_pointer(p, lids, [n])
      ids = lids + 1
    end if
    call finish(status, ierr)
  end function

  subroutine renumbering_free(self)
    class(soma_renumbering), intent(inout) :: self
    call c_renumbering_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  impure elemental subroutine renumbering_assign(lhs, rhs)
    class(soma_renumbering), intent(inout) :: lhs
    type(soma_renumbering), intent(in) :: rhs
    type(c_ptr) :: p
    integer(c_int) :: status
    p = c_null_ptr
    if (c_associated(rhs%ptr)) status = c_renumbering_clone(rhs%ptr, p)
    call c_renumbering_free(lhs%ptr)
    lhs%ptr = p
  end subroutine

  impure elemental subroutine renumbering_final(self)
    type(soma_renumbering), intent(inout) :: self
    call c_renumbering_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  ! ---- File I/O ---------------------------------------------------------------

  !> Collective: reads a mesh file (.h5m, .soma, .vtk; by content), partitioned.
  function soma_read_file(comm, path, ierr) result(m)
    type(soma_comm), intent(in) :: comm
    character(*), intent(in) :: path
    integer, intent(out), optional :: ierr
    type(soma_mesh) :: m
    call finish(c_read_file(comm%ptr, c_str(path), m%ptr), ierr)
  end function

  !> Collective: writes the mesh by the path's extension (.h5m, .soma, .vtk);
  !> vtk_fields: the fields a .vtk file holds (default: every one VTK can hold).
  subroutine soma_write_file(comm, mesh, path, vtk_fields, ierr)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh), intent(in) :: mesh
    character(*), intent(in) :: path
    character(*), intent(in), optional :: vtk_fields(:)
    integer, intent(out), optional :: ierr
    character(kind=c_char), allocatable, target :: buf(:)
    type(c_ptr), allocatable, target :: names(:)
    type(c_ptr) :: pn
    integer :: i, k, at, n
    if (.not. present(vtk_fields)) then
      call finish(c_write_file(comm%ptr, mesh%ptr, c_str(path), -1_c_int, c_null_ptr), ierr)
      return
    end if
    ! The names, NUL-terminated, one after another in buf; names(i) points at the i-th.
    n = size(vtk_fields)
    allocate (buf(n * (len(vtk_fields) + 1) + 1), names(max(n, 1)))
    at = 1
    do i = 1, n
      names(i) = c_loc(buf(at))
      do k = 1, len_trim(vtk_fields(i))
        buf(at) = vtk_fields(i)(k:k)
        at = at + 1
      end do
      buf(at) = c_null_char
      at = at + 1
    end do
    pn = c_null_ptr
    if (n > 0) pn = c_loc(names(1))
    call finish(c_write_file(comm%ptr, mesh%ptr, c_str(path), int(n, c_int), pn), ierr)
  end subroutine

  !> The format of a file, by content (SOMA_FORMAT_*).
  integer function soma_detect_format(path, ierr)
    character(*), intent(in) :: path
    integer, intent(out), optional :: ierr
    integer(c_int) :: f
    f = SOMA_FORMAT_UNKNOWN
    call finish(c_detect_format(c_str(path), f), ierr)
    soma_detect_format = f
  end function

  !> Collective: converts a MOAB .h5m file to a SOMA file or back; returns the
  !> format written.
  integer function soma_convert_mesh_file(comm, src, dst, compression, ierr)
    type(soma_comm), intent(in) :: comm
    character(*), intent(in) :: src, dst
    integer, intent(in), optional :: compression
    integer, intent(out), optional :: ierr
    integer(c_int) :: c, w
    c = 0
    if (present(compression)) c = int(compression, c_int)
    w = SOMA_FORMAT_UNKNOWN
    call finish(c_convert_mesh_file(comm%ptr, c_str(src), c_str(dst), c, w), ierr)
    soma_convert_mesh_file = w
  end function

  ! ---- Parallel ---------------------------------------------------------------

  !> Collective: dest(e), the rank of element e, by its centroid centroids(:, e)
  !> on a space-filling curve.
  subroutine soma_partition_by_curve(comm, centroids, gids, dest, ierr)
    type(soma_comm), intent(in) :: comm
    real(c_double), intent(in), target, contiguous :: centroids(:, :)
    integer(soma_go), intent(in), target, contiguous :: gids(:)
    integer, intent(out), target, contiguous :: dest(:)
    integer, intent(out), optional :: ierr
    integer(c_int) :: status
    if (size(centroids, 1) /= 3 .or. size(centroids, 2) /= size(gids) .or. size(dest) /= size(gids)) then
      ! A count the C layer refuses on every rank, so no rank enters the
      ! collective step alone; then this rank's own message.
      status = c_partition_by_curve(comm%ptr, -1_soma_lo, c_null_ptr, c_null_ptr, c_null_ptr)
      call fail(SOMA_ERR_INVALID_ARGUMENT, 'partition_by_curve: centroids(3, n), gids(n) and dest(n)', ierr)
      return
    end if
    call finish(c_partition_by_curve(comm%ptr, int(size(gids), soma_lo), loc_real2(centroids), loc_go(gids), &
                                     loc_int(dest)), ierr)
  end subroutine

  !> Collective: a new mesh with each element held here moved to rank dest(e).
  function soma_migrate(comm, mesh, dest, ordering, ierr) result(m)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh), intent(in) :: mesh
    integer, intent(in), target, contiguous :: dest(:)
    integer, intent(in), optional :: ordering
    integer, intent(out), optional :: ierr
    type(soma_mesh) :: m
    integer(c_int) :: o
    o = SOMA_ORDERING_DEFAULT
    if (present(ordering)) o = int(ordering, c_int)
    call finish(c_migrate(comm%ptr, mesh%ptr, int(size(dest), soma_lo), loc_int(dest), o, m%ptr), ierr)
  end function

  !> Collective: a new mesh with the elements spread evenly along a curve.
  function soma_rebalance(comm, mesh, ordering, ierr) result(m)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh), intent(in) :: mesh
    integer, intent(in), optional :: ordering
    integer, intent(out), optional :: ierr
    type(soma_mesh) :: m
    integer(c_int) :: o
    o = SOMA_ORDERING_DEFAULT
    if (present(ordering)) o = int(ordering, c_int)
    call finish(c_rebalance(comm%ptr, mesh%ptr, o, m%ptr), ierr)
  end function

  !> Collective, in place: layers of ghost elements through entities of bridge_dim.
  subroutine mesh_add_ghosts(self, comm, layers, bridge_dim, sets, ierr)
    class(soma_mesh), intent(inout) :: self
    type(soma_comm), intent(in) :: comm
    integer, intent(in), optional :: layers, bridge_dim
    logical, intent(in), optional :: sets
    integer, intent(out), optional :: ierr
    integer(c_int) :: l, b, s
    l = 1
    b = 0
    s = 0
    if (present(layers)) l = int(layers, c_int)
    if (present(bridge_dim)) b = int(bridge_dim, c_int)
    if (present(sets)) then
      if (sets) s = 1
    end if
    call finish(c_mesh_add_ghosts(comm%ptr, self%ptr, l, b, s), ierr)
  end subroutine

  !> Collective, in place: drops the ghosts.
  subroutine mesh_remove_ghosts(self, comm, ierr)
    class(soma_mesh), intent(inout) :: self
    type(soma_comm), intent(in) :: comm
    integer, intent(out), optional :: ierr
    call finish(c_mesh_remove_ghosts(comm%ptr, self%ptr), ierr)
  end subroutine

  ! ---- Smoothing ----------------------------------------------------------------

  !> Collective, in place: Laplacian smoothing. fixed(v) keeps vertex v
  !> (default: the skin stays); done: sweeps made; largest_move: in the last
  !> sweep (0 unless tolerance > 0).
  subroutine soma_laplacian_smooth(comm, mesh, iterations, bridge_dim, tolerance, fixed, done, largest_move, ierr)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh), intent(inout) :: mesh
    integer, intent(in), optional :: iterations, bridge_dim
    real(c_double), intent(in), optional :: tolerance
    logical, intent(in), optional :: fixed(:)
    integer, intent(out), optional :: done
    real(c_double), intent(out), optional :: largest_move
    integer, intent(out), optional :: ierr
    integer(c_int8_t), allocatable, target :: flags(:)
    integer(c_int) :: it, b, d, status
    real(c_double) :: t, largest
    type(c_ptr) :: pf
    integer(soma_lo) :: nf
    it = 1
    b = -1
    t = 0
    if (present(iterations)) it = int(iterations, c_int)
    if (present(bridge_dim)) b = int(bridge_dim, c_int)
    if (present(tolerance)) t = tolerance
    pf = c_null_ptr
    nf = 0
    if (present(fixed)) then
      ! A mask given, even empty, is a non-null address (C reads null as "no
      ! mask"): at least one element, as c_loc may not take a zero-size array.
      allocate (flags(max(size(fixed), 1)))
      flags = 0
      flags(1:size(fixed)) = merge(1_c_int8_t, 0_c_int8_t, fixed)
      nf = size(fixed)
      pf = c_loc(flags(1))
    end if
    d = 0
    largest = 0
    status = c_laplacian_smooth(comm%ptr, mesh%ptr, it, b, t, nf, pf, d, largest)
    if (present(done)) done = d
    if (present(largest_move)) largest_move = largest
    call finish(status, ierr)
  end subroutine

  ! ---- Refinement ---------------------------------------------------------------

  !> Collective: every edge split in two, levels times (default 1).
  function soma_refine_uniform(comm, mesh, levels, ierr) result(r)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh), intent(in) :: mesh
    integer, intent(in), optional :: levels
    integer, intent(out), optional :: ierr
    type(soma_refine) :: r
    integer(c_int) :: l
    l = 1
    if (present(levels)) l = int(levels, c_int)
    call finish(c_refine_uniform(comm%ptr, mesh%ptr, l, r%ptr), ierr)
  end function

  !> Collective: the cells marked (marks(e) true, one a cell held) refined;
  !> conforming: no hanging vertices; vertices: SOMA_VERTICES_*.
  function soma_refine_marked(comm, mesh, marks, conforming, vertices, ierr) result(r)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh), intent(in) :: mesh
    logical, intent(in) :: marks(:)
    logical, intent(in), optional :: conforming
    integer, intent(in), optional :: vertices
    integer, intent(out), optional :: ierr
    type(soma_refine) :: r
    integer(c_int8_t), allocatable, target :: flags(:)
    integer(c_int) :: c, v
    type(c_ptr) :: pm
    flags = merge(1_c_int8_t, 0_c_int8_t, marks)
    c = 0
    if (present(conforming)) then
      if (conforming) c = 1
    end if
    v = SOMA_VERTICES_RECORDED
    if (present(vertices)) v = int(vertices, c_int)
    pm = c_null_ptr
    if (size(flags) > 0) pm = c_loc(flags)
    call finish(c_refine_marked(comm%ptr, mesh%ptr, int(size(flags), soma_lo), pm, c, v, r%ptr), ierr)
  end function

  !> The refined mesh.
  function refine_mesh(self, ierr) result(m)
    class(soma_refine), intent(in) :: self
    integer, intent(out), optional :: ierr
    type(soma_mesh) :: m
    call finish(c_refine_mesh(self%ptr, m%ptr), ierr)
  end function

  ! A map of lids, 1-based (-1 becomes 0).
  function refine_lid_map(self, which, ierr) result(ids)
    class(soma_refine), intent(in) :: self
    integer, intent(in) :: which
    integer, intent(out), optional :: ierr
    integer(soma_lo), allocatable :: ids(:)
    integer(soma_lo), pointer :: lids(:)
    type(c_ptr) :: p
    integer(c_int64_t) :: n
    integer(c_int) :: status
    n = 0
    status = c_refine_lids(self%ptr, int(which, c_int), p, n)
    allocate (ids(0))
    if (status == SOMA_OK .and. n > 0) then
      call c_f_pointer(p, lids, [n])
      ids = lids + 1
    end if
    call finish(status, ierr)
  end function

  ! A map of offsets, 1-based.
  function refine_offset_map(self, which, ierr) result(offs)
    class(soma_refine), intent(in) :: self
    integer, intent(in) :: which
    integer, intent(out), optional :: ierr
    integer(soma_offset), allocatable :: offs(:)
    integer(soma_offset), pointer :: o(:)
    type(c_ptr) :: p
    integer(c_int64_t) :: n
    integer(c_int) :: status
    n = 0
    status = c_refine_offsets(self%ptr, int(which, c_int), p, n)
    allocate (offs(0))
    if (status == SOMA_OK .and. n > 0) then
      call c_f_pointer(p, o, [n])
      offs = o + 1
    end if
    call finish(status, ierr)
  end function

  !> For each cell of the refined mesh, the input's owned cell it lies in.
  function refine_parent(self, ierr) result(ids)
    class(soma_refine), intent(in) :: self
    integer, intent(out), optional :: ierr
    integer(soma_lo), allocatable :: ids(:)
    ids = refine_lid_map(self, 0, ierr)
  end function

  !> The children of the input's owned cells: cell p's are
  !> children(child_offsets(p) : child_offsets(p + 1) - 1).
  function refine_children(self, ierr) result(ids)
    class(soma_refine), intent(in) :: self
    integer, intent(out), optional :: ierr
    integer(soma_lo), allocatable :: ids(:)
    ids = refine_lid_map(self, 1, ierr)
  end function

  !> The hanging vertices of the refined mesh.
  function refine_hanging(self, ierr) result(ids)
    class(soma_refine), intent(in) :: self
    integer, intent(out), optional :: ierr
    integer(soma_lo), allocatable :: ids(:)
    ids = refine_lid_map(self, 2, ierr)
  end function

  function refine_child_offsets(self, ierr) result(offs)
    class(soma_refine), intent(in) :: self
    integer, intent(out), optional :: ierr
    integer(soma_offset), allocatable :: offs(:)
    offs = refine_offset_map(self, 0, ierr)
  end function

  !> Hanging vertex i's masters are hanging_masters(o(i) : o(i + 1) - 1).
  function refine_hanging_offsets(self, ierr) result(offs)
    class(soma_refine), intent(in) :: self
    integer, intent(out), optional :: ierr
    integer(soma_offset), allocatable :: offs(:)
    offs = refine_offset_map(self, 1, ierr)
  end function

  !> The hanging vertices' masters, by gid.
  function refine_hanging_masters(self, ierr) result(gids)
    class(soma_refine), intent(in) :: self
    integer, intent(out), optional :: ierr
    integer(soma_go), allocatable :: gids(:)
    integer(soma_go), pointer :: g(:)
    type(c_ptr) :: p
    integer(c_int64_t) :: n
    integer(c_int) :: status
    n = 0
    status = c_refine_masters(self%ptr, p, n)
    allocate (gids(0))
    if (status == SOMA_OK .and. n > 0) then
      call c_f_pointer(p, g, [n])
      gids = g
    end if
    call finish(status, ierr)
  end function

  !> The masters' weights.
  function refine_hanging_weights(self, ierr) result(w)
    class(soma_refine), intent(in) :: self
    integer, intent(out), optional :: ierr
    real(c_double), allocatable :: w(:)
    real(c_double), pointer :: q(:)
    type(c_ptr) :: p
    integer(c_int64_t) :: n
    integer(c_int) :: status
    n = 0
    status = c_refine_weights(self%ptr, p, n)
    allocate (w(0))
    if (status == SOMA_OK .and. n > 0) then
      call c_f_pointer(p, q, [n])
      w = q
    end if
    call finish(status, ierr)
  end function

  !> Fields not carried to the refined mesh.
  function refine_dropped(self, ierr) result(names)
    class(soma_refine), intent(in) :: self
    integer, intent(out), optional :: ierr
    character(len=soma_name_len), allocatable :: names(:)
    character(kind=c_char) :: buf(soma_name_len + 1)
    integer(c_int) :: n, i, status
    integer(c_size_t) :: length
    n = 0
    status = c_refine_dropped_count(self%ptr, n)
    allocate (names(n))
    names = ' '
    do i = 1, n
      if (status /= SOMA_OK) exit
      length = 0
      status = c_refine_dropped_name(self%ptr, i - 1, buf, int(size(buf), c_size_t), length)
      if (status == SOMA_OK) names(i) = from_c(buf, length)
    end do
    if (status /= SOMA_OK .and. length > soma_name_len) then
      call fail(SOMA_ERR_OUT_OF_RANGE, too_long('dropped', length), ierr)
      return
    end if
    call finish(status, ierr)
  end function

  subroutine refine_free(self)
    class(soma_refine), intent(inout) :: self
    call c_refine_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  impure elemental subroutine refine_assign(lhs, rhs)
    class(soma_refine), intent(inout) :: lhs
    type(soma_refine), intent(in) :: rhs
    type(c_ptr) :: p
    integer(c_int) :: status
    p = c_null_ptr
    if (c_associated(rhs%ptr)) status = c_refine_clone(rhs%ptr, p)
    call c_refine_free(lhs%ptr)
    lhs%ptr = p
  end subroutine

  impure elemental subroutine refine_final(self)
    type(soma_refine), intent(inout) :: self
    call c_refine_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  ! ---- Remap --------------------------------------------------------------------

  !> Collective: the remap from the cells of source to those of target (both
  !> kept alive). search, surface, sphere_clip: SOMA_SEARCH_*, ...
  function remap_new(comm, source, target, merge_tolerance, search, surface, radius, sphere_clip, ierr) result(r)
    type(soma_comm), intent(in) :: comm
    type(soma_mesh), intent(in) :: source, target
    real(c_double), intent(in), optional :: merge_tolerance, radius
    integer, intent(in), optional :: search, surface, sphere_clip
    integer, intent(out), optional :: ierr
    type(soma_remap) :: r
    real(c_double) :: tol, rad
    integer(c_int) :: se, su, sc
    tol = 1d-12
    rad = 0
    se = SOMA_SEARCH_GRID
    su = SOMA_SURFACE_PLANE
    sc = SOMA_SPHERE_CLIP_GNOMONIC
    if (present(merge_tolerance)) tol = merge_tolerance
    if (present(radius)) rad = radius
    if (present(search)) se = int(search, c_int)
    if (present(surface)) su = int(surface, c_int)
    if (present(sphere_clip)) sc = int(sphere_clip, c_int)
    call finish(c_remap_create(comm%ptr, source%ptr, target%ptr, tol, se, su, rad, sc, r%ptr), ierr)
  end function

  !> Collective: u_target from u_source, at least one value per owned cell of
  !> each mesh (a field's column u(:, 1) serves).
  subroutine remap_apply(self, comm, u_source, u_target, ierr)
    class(soma_remap), intent(in) :: self
    type(soma_comm), intent(in) :: comm
    real(c_double), intent(in), target, contiguous :: u_source(:)
    real(c_double), intent(inout), target, contiguous :: u_target(:)
    integer, intent(out), optional :: ierr
    type(c_ptr) :: ps, pt
    ps = c_null_ptr
    pt = c_null_ptr
    if (size(u_source) > 0) ps = c_loc(u_source(1))
    if (size(u_target) > 0) pt = c_loc(u_target(1))
    call finish(c_remap_apply(self%ptr, comm%ptr, int(size(u_source), soma_lo), ps, int(size(u_target), soma_lo), &
                              pt), ierr)
  end subroutine

  subroutine remap_free(self)
    class(soma_remap), intent(inout) :: self
    call c_remap_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  impure elemental subroutine remap_assign(lhs, rhs)
    class(soma_remap), intent(inout) :: lhs
    type(soma_remap), intent(in) :: rhs
    type(c_ptr) :: p
    integer(c_int) :: status
    p = c_null_ptr
    if (c_associated(rhs%ptr)) status = c_remap_clone(rhs%ptr, p)
    call c_remap_free(lhs%ptr)
    lhs%ptr = p
  end subroutine

  impure elemental subroutine remap_final(self)
    type(soma_remap), intent(inout) :: self
    call c_remap_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  ! ---- Halo handles -------------------------------------------------------------

  subroutine halo_free(self)
    class(soma_halo), intent(inout) :: self
    call c_halo_free(self%ptr)
    self%ptr = c_null_ptr
  end subroutine

  impure elemental subroutine halo_assign(lhs, rhs)
    class(soma_halo), intent(inout) :: lhs
    type(soma_halo), intent(in) :: rhs
    type(c_ptr) :: p
    integer(c_int) :: status
    p = c_null_ptr
    if (c_associated(rhs%ptr)) status = c_halo_clone(rhs%ptr, p)
    call c_halo_free(lhs%ptr)
    lhs%ptr = p
  end subroutine

  impure elemental subroutine halo_final(self)
    type(soma_halo), intent(inout) :: self
    call c_halo_free(self%ptr)
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

  type(c_ptr) function loc_lo(a)
    integer(soma_lo), intent(in), target, contiguous :: a(:)
    loc_lo = c_null_ptr
    if (size(a) > 0) loc_lo = c_loc(a(1))
  end function

  type(c_ptr) function loc_int(a)
    integer, intent(in), target, contiguous :: a(:)
    loc_int = c_null_ptr
    if (size(a) > 0) loc_int = c_loc(a(1))
  end function

end module soma
