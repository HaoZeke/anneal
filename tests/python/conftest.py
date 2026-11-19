"""Pytest hooks for the free-threaded ASE suite.

Tests import the installed rgpot. The source tree under python/ is not
placed on sys.path, because that would shadow the wheel.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Local helpers such as support.py. This directory is not the package.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "thread_unsafe: not safe under pytest-run-parallel",
    )
