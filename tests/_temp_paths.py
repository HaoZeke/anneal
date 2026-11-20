# Copyright 2023--present rgpot developers
from __future__ import annotations

from pathlib import Path


def temp_path_str(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
