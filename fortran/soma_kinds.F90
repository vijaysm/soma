! SPDX-License-Identifier: BSD-3-Clause
! Copyright (c) 2022-2026, Vijay Mahadevan.

!> The kinds of SOMA's ids and values, as this build of SOMA has them.
module soma_kinds
  use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t, c_double
  implicit none
  private

#if SOMA_LO_64
  integer, parameter, public :: soma_lo = c_int64_t      !< local ids (1-based in Fortran)
#else
  integer, parameter, public :: soma_lo = c_int32_t      !< local ids (1-based in Fortran)
#endif
#if SOMA_OFFSET_64
  integer, parameter, public :: soma_offset = c_int64_t  !< CSR offsets (1-based in Fortran)
#else
  integer, parameter, public :: soma_offset = c_int32_t  !< CSR offsets (1-based in Fortran)
#endif
  integer, parameter, public :: soma_go = c_int64_t      !< global ids (never shifted)
  integer, parameter, public :: soma_real = c_double

end module soma_kinds
