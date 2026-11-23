# Copyright 2023--present rgpot developers
"""PATH filtering of engines/test_gromacs_missing_include.py.

Hiding gmx must keep every other program in the same directory reachable. In a
conda or pixi prefix gmx sits beside the compilers, and dropping the whole
directory leaves meson without a compiler.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "engines" / "test_gromacs_missing_include.py"
)
SPEC = importlib.util.spec_from_file_location("gromacs_missing_include_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

HIDDEN = {"gmx", "gmx_d", "gmx.exe"}


def _program(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def prefix(tmp_path: Path) -> Path:
    shared = tmp_path / "prefix" / "bin"
    shared.mkdir(parents=True)
    _program(shared, "gmx")
    _program(shared, "gmx_d")
    _program(shared, "cc")
    _program(shared, "ninja")
    return shared


def _which(name: str, path: str) -> str | None:
    return shutil.which(name, path=path)


@pytest.mark.skipif(os.name == "nt", reason="shadow directories use symlinks")
def test_compiler_beside_gmx_stays_reachable(prefix: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    _program(other, "python")
    path = os.pathsep.join([str(prefix), str(other)])

    hidden = probe._path_without(HIDDEN, path, tmp_path / "shadow")

    assert _which("gmx", hidden) is None
    assert _which("gmx_d", hidden) is None
    assert _which("cc", hidden) is not None
    assert _which("ninja", hidden) is not None
    assert _which("python", hidden) is not None
    assert str(other) in hidden.split(os.pathsep)
    assert str(prefix) not in hidden.split(os.pathsep)


@pytest.mark.skipif(os.name == "nt", reason="shadow directories use symlinks")
def test_directories_without_gmx_are_left_alone(prefix: Path, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    _program(plain, "cc")

    hidden = probe._path_without(HIDDEN, str(plain), tmp_path / "shadow")

    assert hidden == str(plain)
    assert not (tmp_path / "shadow").exists()


@pytest.mark.skipif(os.name == "nt", reason="shadow directories use symlinks")
def test_shadow_programs_resolve_to_the_originals(prefix: Path, tmp_path: Path) -> None:
    hidden = probe._path_without(HIDDEN, str(prefix), tmp_path / "shadow")

    found = _which("cc", hidden)
    assert found is not None
    assert Path(found).resolve() == (prefix / "cc").resolve()
