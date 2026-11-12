#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fetch rgoswami/anneal-paper-data once before a script reads the dumps."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_DONE = False


def fetch_paper_data() -> None:
    """Run scripts/data/fetch_data.py once in this process."""
    global _DONE
    if _DONE:
        return
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "data" / "fetch_data.py"
    subprocess.run([sys.executable, str(script)], check=True)
    _DONE = True
