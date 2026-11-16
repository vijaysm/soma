# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Sphinx configuration for SOMA's documentation, written in MyST Markdown.
# Build locally with:
#
#   python3 -m pip install -r docs/requirements.txt
#   sphinx-build -W -b html docs docs/_build/html

import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent

project = "SOMA"
author = "Vijay Mahadevan"
copyright = "2022-2026, Vijay Mahadevan"
# The version CMake declares.
_match = re.search(r"project\(\s*soma\s+VERSION\s+([0-9.]+)", (root / "CMakeLists.txt").read_text(), re.IGNORECASE)
release = _match.group(1) if _match else "unknown"
version = release

extensions = ["myst_parser", "sphinx_copybutton"]
source_suffix = {".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "fieldlist", "attrs_inline"]
myst_heading_anchors = 3
exclude_patterns = ["_build", "plans", "requirements.txt", "CHANGELOG.md"]
suppress_warnings = []

html_theme = "furo"
html_title = f"SOMA {release}"
html_static_path = ["_static"]
pygments_style = "default"
