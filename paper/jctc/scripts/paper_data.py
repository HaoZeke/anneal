#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fetch HaoZeke/anneal-paper-data once before a script reads the dumps."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_DONE = False
_FETCHED: set[str] = set()


def fetch_paper_data(paths: list[str] | None = None) -> None:
    """Run scripts/data/fetch_data.py once in this process, for every manifest file or the named ones.

    Paths are relative to the repository root, as data/hf-files.csv lists them.
    """
    global _DONE
    if _DONE:
        return
    wanted = None if paths is None else [p for p in paths if p not in _FETCHED]
    if wanted == []:
        return
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "data" / "fetch_data.py"
    subprocess.run([sys.executable, str(script), *(wanted or [])], check=True)
    if wanted is None:
        _DONE = True
    else:
        _FETCHED.update(wanted)
