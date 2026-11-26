# Copyright (c) anneal developers
"""Quench classifies a kernel panic instead of always raising ValueError."""

from __future__ import annotations

from pathlib import Path


def _quench_body() -> str:
    source = Path("src/python.rs").read_text()
    start = source.index("fn quench(")
    end = source.index("\nfn energy_hits(", start)
    return source[start:end]


def test_quench_classifies_kernel_panics():
    """A panic payload is not handed to ValueError unconditionally.

    Input the wrapper can see (gtol, method, species length, dimension)
    stays ValueError before the kernel. A panic that still escapes is
    ValueError only when its text is an input contract, and RuntimeError
    when it is an internal failure.
    """
    body = _quench_body()
    assert "reported_panic" in body, "quench still maps every panic to ValueError"
    assert "map_err(PyValueError::new_err)" not in body, body
    assert "PyValueError::new_err(panic_payload_message" not in body, body
