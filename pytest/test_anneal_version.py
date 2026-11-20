# Copyright (c) anneal developers
"""package_version runs when neither tomllib nor tomli imports.

cargo test on the Python 3.10 cell executes the release workflow's
version command before any dependency group is installed. That cell
has no tomllib and no tomli. Blocking both modules is that state on
every interpreter: 3.11 and newer would otherwise import tomllib and
never reach the failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The finder raises ModuleNotFoundError, which is what a missing module
# raises. The old importer treated that as a hard failure at import.
SANDBOX = r"""
import sys
from importlib.abc import MetaPathFinder


class RefuseToml(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in {"tomli", "tomllib"}:
            raise ModuleNotFoundError(fullname)
        return None


sys.meta_path.insert(0, RefuseToml())
sys.path.insert(0, "scripts")
import anneal_version

print(anneal_version.package_version())
"""


def _workspace_version() -> str:
    """The ``[workspace.package]`` version, read without a TOML library."""
    in_table = False
    for raw in (ROOT / "Cargo.toml").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            in_table = line == "[workspace.package]"
            continue
        if in_table and line.startswith('version = "') and line.endswith('"'):
            return line[len('version = "') : -1]
    raise AssertionError("Cargo.toml has no [workspace.package] version")


def test_package_version_matches_with_an_installed_parser():
    root = str(ROOT / "scripts")
    if root not in sys.path:
        sys.path.insert(0, root)
    import anneal_version

    assert anneal_version.package_version() == _workspace_version()


def test_package_version_needs_no_toml_library():
    result = subprocess.run(
        [sys.executable, "-c", SANDBOX],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == _workspace_version()

    # A version outside the table, and version.workspace, are not the answer.
    import anneal_version

    decoy = '\n'.join(
        [
            'version = "9.9.9"',
            "[workspace.package]",
            "version.workspace = true",
            'version = "1.2.3" # kept',
            "[dependencies]",
            'version = "8.8.8"',
        ]
    )
    assert anneal_version._table_string(decoy, "workspace.package", "version") == "1.2.3"
