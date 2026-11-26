# Copyright (c) anneal developers
"""The CI python job installs ASE so calculator tests do not skip."""

from __future__ import annotations

from pathlib import Path


def _package_name(token: str) -> str:
    token = token.strip("\"'").split(";")[0].split("[")[0]
    for separator in (">=", "<=", "==", "~=", "!=", ">", "<"):
        token = token.split(separator)[0]
    return token


def test_ci_python_job_installs_ase():
    """The Test job pip line and its import check both name the ase package."""
    text = Path(".github/workflows/ci.yml").read_text()
    start = text.index("name: Install maturin + pytest + array-api-strict + mypy")
    step = text[start : text.index("name: maturin develop", start)]
    install = next(
        line for line in step.splitlines() if "pip install" in line and "maturin" in line
    )
    names = [_package_name(token) for token in install.split()]
    assert "ase" in names, install
    imported = next(
        line for line in step.splitlines() if line.strip().startswith(".venv/bin/python -c")
    )
    modules = imported.split("import", 1)[1].replace('"', " ").replace(",", " ").split()
    assert "ase" in modules, imported
