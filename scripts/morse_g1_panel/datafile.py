#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Generate the fixed Morse input files used by the G1 comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "examples" / "h2h_gmin_morse.data"
OUTPUT = ROOT / "bench" / "morse-g1-panel" / "data"
DENSITY = 17.77153175
CELLS = (
    ("m30r6", 30, 6.0),
    ("m30r14", 30, 14.0),
    ("m38r6", 38, 6.0),
    ("m38r14", 38, 14.0),
    ("m55r6", 55, 6.0),
    ("m55r14", 55, 14.0),
    ("m70r3", 70, 3.0),
    ("m70r6", 70, 6.0),
    ("m70r10", 70, 10.0),
    ("m70r14", 70, 14.0),
    ("m80r3", 80, 3.0),
    ("m80r10", 80, 10.0),
)


def radius(n: int) -> float:
    """The native Morse default from io1.f90."""
    return 2.0 + (3.0 * n / DENSITY) ** (1.0 / 3.0)


def render(n: int, rho: float, base: str) -> str:
    if not base.endswith("\n"):
        base += "\n"
    return base + f"MORSE {rho:.1f}\nRADIUS {radius(n):.10f}\n"


def generate(output: Path = OUTPUT, base_path: Path = BASE) -> None:
    base = base_path.read_text()
    output.mkdir(parents=True, exist_ok=True)
    for name, n, rho in CELLS:
        (output / f"{name}.data").write_text(render(n, rho, base))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    generate(args.output, args.base)


if __name__ == "__main__":
    main()
