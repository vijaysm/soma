# swe_ocean

The barotropic ocean on SOMA's spectral elements: a free surface and velocity over real bathymetry
(ETOPO1), with coasts, wind stress, bottom drag and passive tracers on the levels of a
terrain-following (`s`) or z-star coordinate, on a cubed sphere refined toward the coast. Stommel's
gyre and a Kelvin wave verify it. Build with `-DSOMA_ENABLE_APPS=ON` (HDF5 for real bathymetry);
the equations, options, outputs and results are in
[docs/apps/swe-ocean.md](../../docs/apps/swe-ocean.md).

```sh
build/apps/swe_ocean/swe_ocean --case kelvin --ne 40 --order 6          # analytic checks
build/apps/swe_ocean/swe_ocean --case stommel --ne 8 --order 4
export OMP_NUM_THREADS=1                                                   # one thread per rank
mpiexec -n 8 -bind-to core build/apps/swe_ocean/swe_ocean --case wind --ne 30 --coast-km 50 \
    --bathymetry ETOPO1_Ice_g_gmt4.nc --days 5 --report-hours 24 --save-mesh ocean_ne30.soma
mpiexec -n 8 -bind-to core build/apps/swe_ocean/swe_ocean --case wind --mesh ocean_ne30.soma --days 60 --report-hours 24
```

Run one thread per rank and bind each rank to its own core (`-bind-to core` is MPICH's; Open MPI:
`--bind-to core`, Slurm: `srun --cpu-bind=cores`). Do not set `OMP_PROC_BIND` or `OMP_PLACES`
unless the launcher binds the ranks: an unbound rank's thread is then pinned to the node's first
core, with every other rank's. On NFS add `--io-aggregators 1`. See
[Running on a cluster](../../docs/apps/swe-ocean.md#running-on-a-cluster).

Frames of the snapshots (`--output-reports`), vorticity, eta and speed on a map: `render_ocean.py`; see
[Movies](../../docs/apps/swe-ocean.md#movies).
