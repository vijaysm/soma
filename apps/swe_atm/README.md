# swe_atm

The shallow-water atmosphere on SOMA's spectral elements: Williamson's cases 2, 5 and 6 and
Galewsky's jet on the cubed sphere. Build with `-DSOMA_ENABLE_APPS=ON`; the equations, options,
outputs and benchmark procedure are in [docs/apps/swe-atm.md](../../docs/apps/swe-atm.md).

```sh
mpiexec -n 8 build/apps/swe_atm/swe_atm --case w6 --ne 30 --order 3 --days 15 --report-every 24
apps/swe_atm/bench/run_cases.sh 8           # cases 2, 5, 6 (references in reference/)
apps/swe_atm/bench/galewsky.sh 8             # Galewsky at ne30, ne60, ne120
```

## Results (M2 Max)

| Run | Result | Simulated days per hour |
| --- | --- | --- |
| case 5, ne30 p = 3, 15 days | not yet run | |
| case 6, ne30 p = 3, 15 days | not yet run | |
| Galewsky, ne30 / ne60 / ne120 | not yet run | |
