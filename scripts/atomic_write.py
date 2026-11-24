# Copyright (c) anneal developers
"""Replace a file whole or not at all."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def stage(path: Path, data: bytes | str) -> Path:
    """Write ``data`` to a temporary file beside ``path`` and return its name.

    The file is flushed to disk and takes the mode of ``path`` when that exists,
    else the mode a new file gets. ``path`` itself is not touched.
    """
    path = Path(path)
    payload = data.encode() if isinstance(data, str) else data
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    staged = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(staged, path.stat().st_mode & 0o7777)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(staged, 0o666 & ~umask)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def write_atomic(path: Path, data: bytes | str) -> None:
    """Replace ``path`` with ``data``. A failure leaves the old file and no temporary file."""
    staged = stage(path, data)
    try:
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
