#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
#
# make_figures.sh <SOMA build dir> [figure...]
#
# Makes the documentation figures: builds soma_figure_data in the given build
# (configured with SOMA_ENABLE_DOC_FIGURES=ON), runs each data set the figures
# need at its rank count, reproducibly and on one thread, then draws the
# figures with ParaView's pvpython into docs/figures/img/. With no figures
# named it makes them all. It ends by listing the images that changed.
#
# Environment:
#   PVPYTHON     pvpython to use (default: pvpython on PATH, else the ParaView
#                application under /Applications)
#   MPIEXEC      default: MPIEXEC_EXECUTABLE of the build
#   FIGURE_DATA  where the data goes (default: <build>/docs/figures/data)
#
# See docs/developer/making-figures.md.

set -euo pipefail

if ((BASH_VERSINFO[0] < 4)); then
  echo "make_figures: needs bash 4 or newer (this is $BASH_VERSION); on macOS: brew install bash" >&2
  exit 1
fi

die() {
  echo "make_figures: $*" >&2
  exit 1
}

[[ $# -ge 1 ]] || { echo "usage: make_figures.sh <SOMA build dir> [figure...]" >&2; exit 2; }
build=$(cd "$1" 2>/dev/null && pwd) || die "no build directory $1"
shift
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
cache="$build/CMakeCache.txt"

[[ -f "$cache" ]] || die "$build is not a CMake build directory"
grep -q '^SOMA_ENABLE_DOC_FIGURES:BOOL=ON$' "$cache" ||
  die "$build was not configured with -DSOMA_ENABLE_DOC_FIGURES=ON (which needs MPI and HDF5)"

if [[ -z "${PVPYTHON:-}" ]]; then
  PVPYTHON=$(command -v pvpython || true)
  if [[ -z "$PVPYTHON" ]]; then
    for app in /Applications/ParaView*.app/Contents/bin/pvpython; do
      [[ -x "$app" ]] && PVPYTHON=$app
    done
  fi
fi
[[ -n "${PVPYTHON:-}" && -x "$PVPYTHON" ]] || die "pvpython not found; install ParaView or set PVPYTHON"
MPIEXEC=${MPIEXEC:-$(sed -n 's/^MPIEXEC_EXECUTABLE:FILEPATH=//p' "$cache")}
MPIEXEC=$(command -v "$MPIEXEC" || true)
[[ -n "$MPIEXEC" ]] || die "mpiexec not found; set MPIEXEC"
data=${FIGURE_DATA:-$build/docs/figures/data}

if ! log=$(cmake --build "$build" --target soma_figure_data 2>&1); then
  tail -n 40 <<<"$log" >&2
  die "building soma_figure_data failed"
fi
driver="$build/docs/figures/soma_figure_data"

# The figures asked for, and the data sets they need.
figures=("$@")
[[ ${#figures[@]} -gt 0 ]] || figures=(all)
listing=$("$PVPYTHON" "$here/render.py" --list 2>/dev/null) || die "$PVPYTHON could not run render.py"
needed=()
while read -r name _ sets; do
  if [[ "${figures[*]}" == all ]] || [[ " ${figures[*]} " == *" $name "* ]]; then
    IFS=, read -r -a parts <<<"$sets"
    needed+=("${parts[@]}")
  fi
done <<<"$listing"
[[ ${#needed[@]} -gt 0 ]] || die "no figure named ${figures[*]}; known: $(cut -d' ' -f1 <<<"$listing" | tr '\n' ' ')"

declare -A ranks=()
sets=$("$driver" --list) || die "soma_figure_data --list failed"
while read -r set n; do ranks[$set]=$n; done <<<"$sets"
for set in $(printf '%s\n' "${needed[@]}" | sort -u); do
  [[ -n "${ranks[$set]:-}" ]] || die "soma_figure_data has no data set $set"
  echo "make_figures: data set $set on ${ranks[$set]} rank(s)"
  # Only this run's files: a set may name its files by rank, and an earlier run may have had more.
  mkdir -p "$data/$set"
  rm -f "$data/$set"/*.json "$data/$set"/*.vtk
  OMP_NUM_THREADS=1 "$MPIEXEC" -n "${ranks[$set]}" "$driver" "$set" "$data/$set" --soma-reproducible >/dev/null ||
    die "data set $set failed"
done

"$PVPYTHON" "$here/render.py" "$data" "$here/img" "${figures[@]}" || die "rendering failed"
echo "make_figures: images changed:"
git -C "$root" status --short -- docs/figures/img
