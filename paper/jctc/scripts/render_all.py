#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw every figure of paper/jctc into figs/ from the committed inputs.

Run from the repository root or from paper/jctc:

    python3 scripts/render_all.py

The working directory of the child scripts is paper/jctc.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"
SCRIPTS = ROOT / "scripts"

STEPS = (
    ("fig_algorithm_stack.py", ("figs/algorithm_stack.pdf",)),
    ("fig_rec_arms.py", ("figs/rec_arms.pdf",)),
    ("fig_lj75_ecdf.py", ("figs/lj75_ecdf.pdf",)),
    ("fig_summary_rates.py", ("figs/summary_rates.pdf",)),
    ("fig_landfold_maps.py", ("figs/landfold_maps.pdf",)),
    ("fig_systems.py", ("figs/systems.pdf",)),
    ("fig_si_structures.py", ("figs/si_structures.pdf",)),
    ("fig_si_water.py", ("figs/si_water.pdf",)),
    ("fig_landscapes.py", ("figs/landscapes.pdf",)),
    ("fig_disconnectivity.py", ("figs/disconnectivity.pdf",)),
    ("toc_graphic.py", ("figs/toc_graphic.pdf",)),
    ("fig_si_panels.py", ("figs",)),
)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else FIGS
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    for script, args in STEPS:
        rewritten = []
        for arg in args:
            path = Path(arg)
            if path.parts and path.parts[0] == "figs":
                rewritten.append(str(out.joinpath(*path.parts[1:])))
            elif arg == "figs":
                rewritten.append(str(out))
            else:
                rewritten.append(arg)
        command = [sys.executable, str(Path("scripts") / script), *rewritten]
        print("RUN", " ".join(command[1:]), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            print("FAIL", script, completed.returncode, flush=True)
            return completed.returncode
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
