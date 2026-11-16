# soma_app: an application outside SOMA's tree

A small application that uses SOMA the way a downstream project does: through an installed
SOMA and `find_package(soma)` only. It builds or reads a mesh on any number of ranks, refines it
once, remaps a cell density onto the refined mesh conservatively, writes the result (a SOMA
file when the name ends in `.soma`, which needs SOMA built with HDF5; else MOAB `.h5m` or legacy
VTK), reads a SOMA file back and checks the mass and the field.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOMA_ENABLE_MPI=ON -DSOMA_ENABLE_HDF5=ON \
  -DCMAKE_PREFIX_PATH=/path/to/kokkos -DCMAKE_INSTALL_PREFIX=$HOME/soma
cmake --build build -j && cmake --install build
cmake -S examples/app -B app-build -DCMAKE_PREFIX_PATH="$HOME/soma;/path/to/kokkos"
cmake --build app-build
mpiexec -n 4 app-build/soma_app --n 16 out.soma
```

SOMA's test `packaging.example_app` installs SOMA into a scratch prefix and builds and runs this
application at one and two ranks, so it keeps working as SOMA changes.
