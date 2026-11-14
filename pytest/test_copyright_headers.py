# Copyright (c) anneal developers
"""Every tracked shell and Python file carries the project header.

The header is the first line, or the second when the first is a shebang.
Files under ``vendor/`` keep their upstream headers.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HEADER = b"# Copyright (c) anneal developers"


def _tracked_sources():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.sh", "*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [p for p in out.decode().split("\0") if p and not p.startswith("vendor/")]


def _has_header(path):
    lines = (ROOT / path).read_bytes().splitlines()[:2]
    if lines and lines[0].startswith(b"#!"):
        lines = lines[1:]
    return bool(lines) and lines[0] == HEADER


def test_shell_and_python_files_have_copyright_header():
    missing = [p for p in _tracked_sources() if not _has_header(p)]
    assert not missing, "missing '# Copyright (c) anneal developers':\n" + "\n".join(
        missing
    )
