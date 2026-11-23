# Copyright 2023--present rgpot developers

"""Find repository files from a checkout or a wheel test.

cibuildwheel runs the test command in a temporary directory. A path next
to this file is that directory, not the checkout, unless the wheel test
command copied the file there. ``RGPOT_PROJECT`` is that checkout when the
command exports it. Pytest's rootdir counts when it is the checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

# tests/python/conftest.py sets this once pytest has chosen a rootdir.
PYTEST_ROOT: Path | None = None

_ENV = "RGPOT_PROJECT"

# A copied wheel test tree has no meson.build. Any one of these means the
# directory is a source tree, so a missing sibling fails instead of skipping.
_MARKERS = (
    "meson.build",
    "pyproject.toml",
    "CppCore/tests/data/eam_table/Cu_u3.eam",
    "CppCore/tests/data/eam_table/cu_u3_cubic.ref",
    "CppCore/tests/data/eam_table/malformed_funcfl.eam",
    "third_party/vesin/VERSION",
    "third_party/vesin/vesin-single-build.cpp",
    "potctl/src/ci.rs",
)


def _checkout(path: Path) -> bool:
    return (path / "meson.build").is_file() and (path / "pyproject.toml").is_file()


def _has_marker(path: Path) -> bool:
    return any((path / marker).exists() for marker in _MARKERS)


def source_root() -> Path | None:
    """Checkout that holds repository data, or None when none is present.

    ``RGPOT_PROJECT`` is the only root when it is set. A missing file there
    fails the test. The same is true of pytest's rootdir and of a parent of
    this file when that directory is a checkout, and of the copied wheel
    test tree when it already contains repository data.
    """
    named = os.environ.get(_ENV, "").strip()
    if named:
        root = Path(named).expanduser().resolve()
        if not root.is_dir():
            raise AssertionError(f"{_ENV} does not name a directory: {root}")
        return root
    if PYTEST_ROOT is not None and _checkout(PYTEST_ROOT):
        return PYTEST_ROOT.resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if _checkout(parent):
            return parent
    if len(here.parents) > 2:
        copied = here.parents[2]
        if _has_marker(copied):
            return copied
    return None


def source_file(root: Path | None, relative: str) -> Path:
    """``relative`` inside ``root``.

    ``root is None`` means the caller will skip. A root that lacks the file
    is an error, so a reachable checkout cannot pass by skipping.
    """
    if root is None:
        return Path(relative)
    path = root / relative
    if not path.is_file():
        raise AssertionError(f"{relative} is not in the source tree {root}")
    return path
