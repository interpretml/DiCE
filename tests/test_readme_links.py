"""Lightweight regression test for README notebook links.

Issue #308 reported that the BaseVAE notebook link in README.rst pointed
at `docs/notebooks/...` instead of the actual `docs/source/notebooks/...`
path. The link had been broken for years because nothing in CI checked
that in-repo notebook references resolve to real files.

This test scans README.rst for any `https://github.com/interpretml/DiCE/
blob/<ref>/docs/...notebook.ipynb` URL and asserts each one resolves to
a file under the working tree. It only validates *in-repo* links — it
does not hit the network.
"""
from __future__ import annotations

import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
README_PATH = os.path.join(REPO_ROOT, "README.rst")

# Match `https://github.com/interpretml/DiCE/blob/<ref>/<path>.ipynb`.
_NOTEBOOK_RE = re.compile(
    r"https://github\.com/interpretml/DiCE/blob/[^/]+/(?P<path>[^>\s)]+\.ipynb)"
)


def _readme_notebook_paths():
    with open(README_PATH, "r", encoding="utf-8") as fh:
        text = fh.read()
    return list({m.group("path") for m in _NOTEBOOK_RE.finditer(text)})


def test_readme_lists_at_least_one_notebook_link():
    # Sanity: if this drops to 0 the regex got broken or README was rewritten.
    assert _readme_notebook_paths(), "README must list at least one notebook"


def test_readme_notebook_links_resolve():
    """Every notebook URL in README.rst must point to a file in the repo."""
    missing = [
        path
        for path in _readme_notebook_paths()
        if not os.path.isfile(os.path.join(REPO_ROOT, path))
    ]
    assert not missing, (
        "README.rst links to notebooks that do not exist in the repo: "
        f"{missing}. See issue #308 — make sure the URL path matches the "
        "actual on-disk path under docs/source/notebooks/."
    )
