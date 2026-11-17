# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Installs the build into a fresh prefix and requires the package there, and nowhere else.

Usage: check_install.py <cmake> <build_dir> <prefix> <python_version>   (version as 3.9)
"""

import subprocess
import sys
from pathlib import Path


def main(cmake, build, prefix, version):
    prefix = Path(prefix)
    log = subprocess.run([cmake, "--install", build, "--prefix", str(prefix)], capture_output=True, text=True)
    if log.returncode != 0:
        sys.exit(f"cmake --install failed:\n{log.stdout}\n{log.stderr}")
    want = prefix / "lib" / f"python{version}" / "site-packages" / "soma"
    for name in ("__init__.py", "vis.py"):
        if not (want / name).is_file():
            sys.exit(f"missing {want / name}")
    if not list(want.glob("_soma*")):
        sys.exit(f"missing the extension module in {want}")
    outside = [line for line in log.stdout.splitlines()
               if "soma" in line and ("Installing" in line or "Up-to-date" in line) and str(prefix) not in line]
    if outside:
        sys.exit("installed outside the prefix:\n" + "\n".join(outside))
    print(f"package installed under {want}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
