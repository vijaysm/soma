# Williamson cases 5 and 6: T213 reference solutions from NCAR STSWM

How the reference files `case5.nc` and `case6.nc` in this directory were produced. They hold the
free-surface height $h + h_s$ of Williamson et al. (1992) test cases 5 and 6, computed with NCAR's
Spectral Transform Shallow Water Model (STSWM 2.0) at T213, with the parameters of NCAR's
published reference runs (Jakob-Chien, Hack and Williamson 1995).
`apps/swe_atm/bench/run_cases.sh` reads them from here by default.

STSWM's licence ("made available for scientific research and educational purposes only, and should
not be construed to be in the public domain") keeps its source, and the patched copy described
here, out of this repository; this file records what was run. As the licence asks, NCAR is
acknowledged as the source of the model that produced the reference solutions.

## Why we generated them

NCAR's original reference files (`REF0114.cdf` for case 5, `REF0092.cdf` for case 6) were
distributed from `ftp.ucar.edu:chammp/shallow` and later `www.csm.ornl.gov/chammp/stswm`. On
2026-11-18 neither host served them (`ftp.ucar.edu` no longer resolves; the ORNL page returns 404),
the Wayback Machine had no copy of the FTP directory, and the GitHub mirror
`jhaoli/shallow_spectral` carries only case 7's `REF0007.cdf`. The STSWM source is still on netlib,
and its documentation (`docu/description.txt`, Appendix D) lists the reference runs' parameters, so
we reran them.

## Source

- `https://netlib.org/chammp/stswm.tgz` (133 141 bytes; gzip of `stswm.tar`, dated 1993-10-07):
  STSWM Version 2.0, R. Jakob, NCAR, August 1992.
- Reference-run parameters, from `docu/description.txt`, Appendix D:

  | Case | NCAR file | Truncation | Time step | $\nabla^4$ diffusion | Length | Output |
  | --- | --- | --- | --- | --- | --- | --- |
  | 5 | `REF0114.cdf` | T213 | 360 s | 8.0e12 m^4/s | 360 h (15 days) | every 24 h |
  | 6 | `REF0092.cdf` | T213 | 180 s | 8.0e12 m^4/s | 336 h (14 days) | every 24 h |

  We ran case 6 to 360 h (15 days), the length the pass mark uses; days 0-14 are the same run.

## Modifications

All changes are to a copy of `src/` and `lib/`; nothing else in the model was touched.

1. **Resolution** (`params.i`): `MM = NN = KK = 213`, `NLAT = 320`, `NLON = 640`, `MAXH = 213`,
   `RTRUNC = 213` (triangular T213 on the 640 x 320 Gaussian grid; the file already lists this
   line commented out).
2. **Precision**: compiled with `-fdefault-real-8 -fdefault-double-8`. The code declares plain
   `REAL`/`COMPLEX` throughout and was run in 64-bit precision on the Cray; the shipped sample
   outputs come from a 32-bit Sun run.
3. **Output** (`stswm.f`, new `wrtref.f`): the two calls
   `CALL OUTPTP (ZETASC,DIVSC,PHISC,PHIBAR,LDIAG,...,OCTR)` were replaced by `CALL WRTREF(WS3,...)`.
   `OUTPTP` writes spectral coefficients of geopotential through the netCDF-2 Fortran API, which
   current netCDF-Fortran no longer provides by default. At those calls `WS3` already holds the
   gridded free-surface height, `(PHI + PHIBAR)/GRAV + MOUNT` (`stswm.f`, after `CALL STEP`; at
   $t = 0$ the analytic initial height). `WRTREF` appends it, with the model time in hours, to the
   stream file `ref.bin`; the first call also writes `NLAT`, `NLON`, the Gaussian latitudes
   (`GLAT`) and the longitudes (`GLON`, $2\pi (i - 1)/640$).

   ```fortran
         SUBROUTINE WRTREF(H,NEW)
         INCLUDE 'params.i'
         INCLUDE 'consts.i'
         REAL H(NLON+2,NLAT)
         LOGICAL NEW
         EXTERNAL GLAT, GLON
         REAL GLAT, GLON
         INTEGER I,J
         IF (NEW) THEN
            OPEN(77,FILE='ref.bin',ACCESS='STREAM',FORM='UNFORMATTED',
        $        STATUS='REPLACE')
            WRITE(77) NLAT, NLON
            WRITE(77) (GLAT(J),J=1,NLAT)
            WRITE(77) (GLON(I),I=1,NLON)
         ELSE
            OPEN(77,FILE='ref.bin',ACCESS='STREAM',FORM='UNFORMATTED',
        $        STATUS='OLD',POSITION='APPEND')
         ENDIF
         WRITE(77) TAU
         WRITE(77) ((H(I,J),I=1,NLON),J=1,NLAT)
         CLOSE(77)
         RETURN
         END
   ```
4. **Libraries**: linked the dummy `lib/ncarg.f` (NCAR Graphics is not free; plotting is off,
   `LGPHS = .FALSE.`) and the dummy `lib/netcdf.f`, with the shipped `lib/fft99f.f` (ECMWF FFT)
   and `lib/adquad.f` (D01AHE replacement).
5. **`netcdf.inc`**: the unused fill value `XDRDINF = 1.797693134862315900e+308` rounds above the
   largest double and gfortran rejects it; changed to `1.7976931348623157d+308`.

## Build

gfortran 16.2.0 (Homebrew), macOS arm64:

```sh
FF="-O2 -fdefault-real-8 -fdefault-double-8 -std=legacy -fallow-argument-mismatch -w"
for f in *.f; do gfortran $FF -c $f; done      # src/*.f, lib/{fft99f,adquad,ncarg,netcdf}.f, wrtref.f
gfortran $FF -o stswm *.o
```

## Check of the build

A copy built at the shipped T42 resolution ran the shipped `test5/exp.0005` (graphics off). Its
`stdout` has the same lines as NCAR's `test5/stdout.0005`; the diagnostics agree to the shipped
run's single precision:

| Quantity (t = 6 h) | NCAR `stdout.0005` (32-bit) | This build (64-bit) |
| --- | --- | --- |
| mass | 5.619941406E+03 | 5.619934569E+03 |
| total energy | 1.569060480E+08 | 1.569075311E+08 |
| potential enstrophy | 7.203561207E-13 | 7.204477303E-13 |

The 64-bit build holds mass constant to all printed digits; the 32-bit run drifts in the sixth.

## Runs

Namelist (case 5; case 6 has `DT = 180.0`, `ICOND = 6`):

```
 $PHYVAR
 ALPHA  =  0.0,
 HDC    =  8.0E12,
 $END
 $EXPDEF
 DT     =   360.0,
 EGYFRQ =   24.0,
 ERRFRQ =   999.0,
 SPCFRQ =   999.0,
 TAUE   =   360.0,
 TAUO   =   24.0,
 GPHFRQ =   999.0,
 SITS   =   .TRUE.,
 FORCED =   .FALSE.,
 MOMENT =   .FALSE.,
 ICOND  =   5,
 $END
 $PLTDEF
 LGPHS  = .FALSE.,
 $END
 $FNAMES
 $END
```

Constants are STSWM's defaults, which are Williamson's: $a = 6.37122 \times 10^6$ m,
$\Omega = 7.292 \times 10^{-5}$ s$^{-1}$, $g = 9.80616$ m s$^{-2}$. Semi-implicit leapfrog, Asselin
filter off (the shipped experiments' setting). Run as `stswm < exp`.

Results (2026-11-18, one core of an M2 Max):

| | Case 5 | Case 6 |
| --- | --- | --- |
| steps | 3600 | 7200 |
| wall time | 165 s | 326 s |
| Courant number (STSWM's estimate) | 0.24 | 0.60 |
| mass, day 0 -> 15 | 5.619925488E+03, unchanged | 9.522996556E+03, unchanged |
| total energy, day 0 -> 15 | 1.569075112E+08 -> 1.569075044E+08 | 4.625523878E+08 -> 4.625522232E+08 |
| day-15 free surface | 5033.240 .. 5952.897 m | 8087.011 .. 10540.786 m |

`ref.bin` holds 16 records, days 0 to 15 (26 222 216 bytes each case).

## Conversion to NetCDF-4

`ref2nc.py ref.bin caseN.nc N` (numpy, netCDF4-python) writes `h(time, lat, lon)` in metres,
`lat` and `lon` in degrees (latitudes run north to south, as STSWM's Gaussian grid; `swe_atm` sorts
them), `time` in hours. Records are days 0 to 15, so `--ref-time -1` (the default) is day 15. For
case 5 it checks the first record against Williamson's initial state
$h_0 - (a\Omega u_0 + u_0^2/2)\sin^2\phi / g$ ($h_0$ = 5960 m, $u_0$ = 20 m/s): agreement to
9e-13 m, which confirms orientation, units and that `WS3` is the free surface.

The files are NetCDF-4 already; no `nccopy` is needed.

The converter, `ref2nc.py`:

```python
import sys

import netCDF4
import numpy as np


def read(path):
    raw = open(path, "rb").read()
    nlat, nlon = np.frombuffer(raw, np.int32, 2, 0)
    off = 8
    lat = np.frombuffer(raw, np.float64, nlat, off); off += 8 * nlat
    lon = np.frombuffer(raw, np.float64, nlon, off); off += 8 * nlon
    rec = 1 + nlat * nlon
    body = np.frombuffer(raw, np.float64, offset=off)
    if body.size % rec:
        raise SystemExit(f"{path}: {body.size} values is not a whole number of records")
    body = body.reshape(-1, rec)
    times = body[:, 0]
    h = body[:, 1:].reshape(-1, nlat, nlon)  # Fortran H(I,J) -> [J][I]
    return np.degrees(lat), np.degrees(lon), times, h


def check_case5(lat, h0):
    radius, omega, g, u0 = 6.37122e6, 7.292e-5, 9.80616, 20.0
    s = np.sin(np.radians(lat))[:, None]
    exact = 5960.0 - (radius * omega * u0 + u0 * u0 / 2) * s * s / g
    err = np.abs(h0 - exact).max()
    print(f"case 5 t=0: max |h - exact| = {err:.3e} m")
    if err > 1e-6:
        raise SystemExit("case 5 initial state does not match Williamson's; orientation or units wrong")


def main(src, dst, case):
    lat, lon, times, h = read(src)
    if case == "5":
        check_case5(lat, h[0])
    with netCDF4.Dataset(dst, "w", format="NETCDF4") as nc:
        nc.createDimension("time", None)
        nc.createDimension("lat", lat.size)
        nc.createDimension("lon", lon.size)
        v = nc.createVariable("time", "f8", ("time",)); v[:] = times; v.units = "hours"
        v = nc.createVariable("lat", "f8", ("lat",)); v[:] = lat; v.units = "degrees_north"
        v = nc.createVariable("lon", "f8", ("lon",)); v[:] = lon; v.units = "degrees_east"
        v = nc.createVariable("h", "f8", ("time", "lat", "lon"))
        v[:] = h
        v.units = "m"
        v.long_name = "free-surface height h + hs"
        nc.source = "NCAR STSWM 2.0, T213 (640 x 320 Gaussian), del^4 8e12 m^4/s, semi-implicit"
        nc.williamson_case = int(case)


if __name__ == "__main__":
    main(*sys.argv[1:])
```

## Check against `swe_atm`

Both files read through `--reference` with the default `--ref-var h --ref-lat lat --ref-lon lon`.
At a coarse resolution (4 ranks x 2 threads, ne8 $p = 3$, 15 days, default viscosity):

| Case | $l_1$ | $l_2$ | $l_\infty$ |
| --- | --- | --- | --- |
| 5 | 1.83e-3 | 2.67e-3 | 1.13e-2 |
| 6 | 2.03e-2 | 2.78e-2 | 6.73e-2 |

The pass mark is at ne30 $p = 3$: `apps/swe_atm/bench/run_cases.sh NP`, which reads `case5.nc`,
`case6.nc` and `marks.txt` from this directory (or from `REFDIR` given as its second argument).

## Files

- `case5.nc`, `case6.nc` (26 238 899 bytes each): `h(time = 16, lat = 320, lon = 640)`, days 0 to
  15, double precision, uncompressed.

## References

- R. Jakob-Chien, J. J. Hack and D. L. Williamson (1995). Spectral transform solutions to the
  shallow water test set. *J. Comput. Phys.* 119, 164-187.
- J. J. Hack and R. Jakob (1992). Description of a global shallow water model based on the spectral
  transform method. NCAR Tech. Note TN-343+STR.
- D. L. Williamson, J. B. Drake, J. J. Hack, R. Jakob and P. N. Swarztrauber (1992). A standard test
  set for numerical approximations to the shallow water equations in spherical geometry.
  *J. Comput. Phys.* 102, 211-224.
