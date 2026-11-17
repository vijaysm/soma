# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Sphinx configuration for SOMA's documentation, written in MyST Markdown.
# Build locally with:
#
#   python3 -m pip install -r docs/requirements.txt
#   sphinx-build -W -b html docs docs/_build/html
#
# The API reference needs Doxygen (docs/Doxyfile): the build runs it into
# docs/_build/doxygen first, unless its XML is newer than every header.

import logging
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path(__file__).resolve().parent.parent

project = "SOMA"
author = "Vijay Mahadevan"
copyright = "2022-2026, Vijay Mahadevan"
# The version CMake declares.
_match = re.search(r"project\(\s*soma\s+VERSION\s+([0-9.]+)", (root / "CMakeLists.txt").read_text(), re.IGNORECASE)
release = _match.group(1) if _match else "unknown"
version = release

extensions = ["myst_parser", "sphinx_copybutton", "sphinxcontrib.mermaid", "breathe"]
source_suffix = {".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "fieldlist", "attrs_inline"]
myst_heading_anchors = 3
# A plain ```mermaid fence is a diagram here, as it is on GitHub and in the wiki.
myst_fence_as_directive = ["mermaid"]
exclude_patterns = ["_build", "plans", "requirements.txt", "CHANGELOG.md"]
# Files under docs/ that git ignores (private notes in a working copy) are
# not pages; a checkout without git, like a source tarball, has none.
try:
    _ignored = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "."],
        cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=True,
    ).stdout.split()
    exclude_patterns += [p.rstrip("/") for p in _ignored]
except (OSError, subprocess.CalledProcessError):
    pass
suppress_warnings = []

# The API reference: Breathe reads Doxygen's XML.
_doxygen_out = root / "docs" / "_build" / "doxygen"
breathe_projects = {"soma": str(_doxygen_out / "xml")}
breathe_default_project = "soma"
breathe_default_members = ("members",)
breathe_show_include = True

html_theme = "furo"
html_title = f"SOMA {release}"
html_static_path = ["_static"]
html_css_files = ["figures.css"]
pygments_style = "default"


def _run_doxygen():
    """Doxygen's XML for the reference, made again when a header or the Doxyfile is newer."""
    docs = root / "docs"
    index = _doxygen_out / "xml" / "index.xml"
    inputs = [docs / "Doxyfile", *(root / "include" / "soma").rglob("*.hpp")]
    if index.exists() and index.stat().st_mtime >= max(p.stat().st_mtime for p in inputs):
        return
    doxygen = shutil.which("doxygen")
    if doxygen is None:
        raise RuntimeError("the API reference needs Doxygen (see docs/developer/writing-docs.md)")
    _doxygen_out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SOMA_DOXYGEN_OUTPUT": str(_doxygen_out)}
    subprocess.run([doxygen, "Doxyfile"], cwd=docs, check=True, env=env)


class _NamespaceRepeat(logging.Filter):
    """Drops the warning Breathe causes on every header's page after the first:
    the page declares its namespaces again. Any other duplicate still warns."""

    _pattern = re.compile(r"Duplicate C\+\+ declaration.*Declaration is '\.\. cpp:type:: ([\w:]+)'", re.S)

    def __init__(self, namespaces):
        super().__init__()
        self.namespaces = namespaces

    def filter(self, record):
        m = self._pattern.search(record.getMessage())
        return not (m and m.group(1) in self.namespaces)


def setup(app):
    _run_doxygen()
    tree = ET.parse(_doxygen_out / "xml" / "index.xml")
    namespaces = {c.findtext("name") for c in tree.getroot().iter("compound") if c.get("kind") == "namespace"}
    # Before Sphinx's own filters, which count warnings for -W.
    for handler in logging.getLogger("sphinx").handlers:
        handler.filters.insert(0, _NamespaceRepeat(namespaces))
