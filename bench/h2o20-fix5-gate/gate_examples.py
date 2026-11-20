#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the panel drivers for one arm.

Each driver is an arm's own committed example, or ``h2h_blj.rs`` beside this
file for the binary Lennard-Jones cells, with one line added after the row
it prints: ``best_full <Debug f64> <to_bits>``. The committed rows round
``best`` to eight decimals; the added line carries it at full precision.
Nothing else changes.

    gate_examples.py <arm-tree> <dest-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVERS = {
    "gate_h2h_gmin_proof": Path("examples/h2h_gmin_proof.rs"),
    "gate_h2h_morse": Path("examples/h2h_morse.rs"),
    "gate_h2h_sc": Path("examples/h2h_sc.rs"),
    "gate_h2h_metal": Path("examples/h2h_metal.rs"),
    "gate_h2h_blj": None,
}
FULL = '    println!("best_full {:?} {}", outcome.best, outcome.best.to_bits());\n'


def with_full_best(text: str) -> str:
    lines = text.splitlines(keepends=True)
    rows = [i for i, line in enumerate(lines) if ",anneal,{seed}," in line]
    if len(rows) != 1:
        raise SystemExit(f"expected one row print, found {len(rows)}")
    end = next(i for i in range(rows[0], len(lines)) if lines[i].strip() == ");")
    return "".join(lines[: end + 1] + [FULL] + lines[end + 1 :])


def main() -> None:
    tree, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    for name, source in DRIVERS.items():
        path = HERE / "h2h_blj.rs" if source is None else tree / source
        (dest / f"{name}.rs").write_text(with_full_best(path.read_text()))


if __name__ == "__main__":
    main()
