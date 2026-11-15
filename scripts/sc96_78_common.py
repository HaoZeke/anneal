# Copyright (c) anneal developers
"""Shared pieces of the Sutton-Chen 9-6 N=78 measurements."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLE_URL = "https://www-wales.ch.cam.ac.uk/~wales/CCD/jon/structures/SC/9-6/table.9-6.html"
TABLE_SHA256 = "4c528a7d92090ff0b192c73110c062aae1661e837389076218dc93b0f185c138"
LENGTH = (2.0 ** -0.5) / (2.0 ** (1.0 / 6.0))


def binary(name: str) -> Path:
    meta = json.loads(
        subprocess.check_output(
            ["rustup", "run", "stable", "cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=ROOT,
        )
    )
    path = Path(meta["target_directory"]) / "release" / "examples" / name
    if not path.is_file():
        raise SystemExit(f"missing {path}; build the release example first")
    return path


def reference_energy(scratch: Path) -> float:
    scratch.mkdir(parents=True, exist_ok=True)
    page = scratch / "table.9-6.html"
    if not page.is_file():
        request = urllib.request.Request(TABLE_URL, headers={"User-Agent": "anneal"})
        page.write_bytes(urllib.request.urlopen(request, timeout=60).read())
    digest = hashlib.sha256(page.read_bytes()).hexdigest()
    if digest != TABLE_SHA256:
        raise SystemExit(f"the 9-6 table hash is {digest}, expected {TABLE_SHA256}")
    html = page.read_text(errors="replace")
    match = re.search(
        r'HREF="points/78">78</A></TD>\s*<TD[^>]*>\s*(-?\d+\.\d+)',
        html,
        re.IGNORECASE,
    )
    if not match:
        raise SystemExit("the 9-6 table has no energy for N=78")
    return float(match.group(1))


def write_start(path: Path, n: int, seed: int) -> None:
    rng = np.random.Generator(np.random.PCG64(seed))
    radius = 1.8 * LENGTH * n ** (1.0 / 3.0)
    minimum = 0.55 * LENGTH
    points: list[np.ndarray] = []
    draws = 0
    while len(points) < n:
        draws += 1
        if draws > 100000:
            minimum *= 0.9
            draws = 0
        trial = rng.normal(size=3)
        trial *= radius * rng.random() ** (1.0 / 3.0) / np.linalg.norm(trial)
        if all(np.linalg.norm(trial - kept) >= minimum for kept in points):
            points.append(trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{x:.10f} {y:.10f} {z:.10f}\n" for x, y, z in points))


def parse_run(text: str) -> dict[str, str]:
    parsed = {}
    for line in text.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            parsed[key] = value
    return parsed
