# Copyright (c) anneal developers
"""Reassemble a sealed file stored as sub-1 MiB parts.

Git history rejects a blob over 1 MiB. A sealed input that does not fit is
stored as ``name.part000``, ``name.part001``, ... beside the path the scripts
already name. ``materialize`` returns that path when the file is whole, and
otherwise concatenates the parts into a temporary file.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path


def parts_of(path: Path) -> list[Path]:
    return sorted(path.parent.glob(path.name + ".part[0-9][0-9][0-9]"))


def materialize(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    parts = parts_of(path)
    if not parts:
        return path
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.read_bytes())
    cache = Path(tempfile.gettempdir()) / "anneal-sealed" / digest.hexdigest()
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file():
        with cache.open("wb") as out:
            for part in parts:
                out.write(part.read_bytes())
    return cache
