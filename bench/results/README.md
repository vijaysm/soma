# Benchmark baselines

One JSON-lines file per recorded run of `bench_backends`, named
`<yyyy-mm-dd>-<machine>-<git>.jsonl`, where `<git>` is the commit the binary
was built from (`git describe --always`; every row carries it as `"git"`).
Only runs from a clean tree are committed: a `-dirty` tag means the numbers
cannot be reproduced from any commit.

Record a run (RelWithDebInfo or Release, the machine otherwise idle):

    MAX_THREADS=8 bench/run_backends.sh <build-dir> --repeats 5 \
        > bench/results/$(date +%F)-<machine>-$(git describe --always).jsonl

Compare the latest run against a baseline:

    bench/compare_bench.py bench/results/<baseline>.jsonl <current>.jsonl
    bench/compare_bench.py --min-change 0.05 --workflow derive <baseline> <current>

`ratio` is baseline / current: above 1 the current build is faster. Rows whose
change is inside the measurement spread are flagged `~` and are not results.
Same machine, same compiler, same Kokkos and same build type on both sides, or
the comparison measures the environment.

Add a baseline by committing the file; the tables in `docs/benchmarks.md`
quote the run they were made from.
