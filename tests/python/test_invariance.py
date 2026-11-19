"""T5: SHA-256 of raw float64 bytes across Python and OpenMP thread counts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.thread_unsafe,
    # One pass spawns an interpreter per width. Under ThreadSanitizer that
    # pass is about 90s, so the parallel job's default of five repeats
    # exceeds the 180s pytest timeout.
    pytest.mark.iterations(1),
]

_PYTHON_THREADS = (1, 2, 8)
_OMP_THREADS = (1, 2, 3, 8)


def _pythonpath() -> str:
    root = Path(__file__).resolve().parents[2]
    tests = str(root / "tests" / "python")
    source = str(root / "python")
    parts = [tests]
    for part in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if part and part != source:
            parts.append(part)
    return os.pathsep.join(parts)


def _usable(python: str) -> bool:
    """True when this interpreter can import the built extension.

    A wheel test container often has another CPython on PATH that does
    not have numpy or the wheel. Those interpreters are not part of T5.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    proc = subprocess.run(
        [python, "-c", "import numpy, rgpot"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode == 0


def _interpreters() -> list[str]:
    found: list[str] = []
    candidates = [sys.executable]
    for name in ("python3.14", "python3.14t"):
        path = shutil.which(name)
        if path:
            candidates.append(path)
    for path in candidates:
        if path not in found and _usable(path):
            found.append(path)
    return found


def _run(python: str, threads: int, omp: int) -> str:
    worker = Path(__file__).resolve().parent / "invariance_worker.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    env["OMP_NUM_THREADS"] = str(omp)
    proc = subprocess.run(
        [python, str(worker), str(threads)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert lines, proc.stdout
    return lines[-1]


def test_rank_invariance_hashes() -> None:
    interpreters = _interpreters()
    reference: str | None = None
    seen: list[tuple[str, int, int, str]] = []
    for python in interpreters:
        for omp in _OMP_THREADS:
            for threads in _PYTHON_THREADS:
                digest = _run(python, threads, omp)
                seen.append((python, threads, omp, digest))
                if reference is None:
                    reference = digest
                assert digest == reference, seen
    assert reference is not None
    print("T5 interpreters", interpreters)
    print("T5 hash", reference)
    print("T5 points", len(seen))
