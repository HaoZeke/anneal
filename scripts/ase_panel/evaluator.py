# Copyright (c) anneal developers
"""Read the immutable system specification printed by ``asg_eval``."""

from __future__ import annotations

import subprocess


def evaluator_spec(binary: str, system: str) -> tuple[int, int, float, float]:
    proc = subprocess.run(
        [binary, "spec", system],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = proc.stdout.splitlines()[0].split()
    if len(fields) != 6 or fields[0] != system:
        raise RuntimeError(f"invalid asg_eval spec: {proc.stdout!r}")
    return int(fields[1]), int(fields[3]), float(fields[4]), float(fields[5])
