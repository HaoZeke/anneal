#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Rebuild every file in figs/ from its committed generator, then hash it.

Two clean runs are compared by the caller. The child environment pins
SOURCE_DATE_EPOCH to 0 and PYTHONHASHSEED to 0. PDF writers already
drop CreationDate, ModDate and Creator through figstyle.save.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import fetch_paper_data

ROOT = Path(__file__).resolve().parent.parent

# Each step is a generator and the figs/ names it writes.
STEPS = (
    ("fig_algorithm_stack.py", ("algorithm_stack.pdf",)),
    ("fig_rec_arms.py", ("rec_arms.pdf",)),
    ("fig_lj75_ecdf.py", ("lj75_ecdf.pdf",)),
    ("fig_water_crest.py", ("water_crest.pdf",)),
    ("fig_landfold_maps.py", ("landfold_maps.pdf",)),
    ("fig_systems.py", ("systems.pdf",)),
    ("fig_si_structures.py", ("si_structures.pdf",)),
    ("fig_si_water.py", ("si_water.pdf",)),
    ("fig_landscapes.py", ("landscapes.pdf",)),
    ("fig_disconnectivity.py", ("disconnectivity.pdf",)),
    ("toc_graphic.py", ("toc-graphic.pdf",)),
    (
        "fig_si_panels.py",
        (
            "lj38_landscape.png",
            "lj38_sketchmap.png",
            "lj75_landscape.png",
            "lj98_landscape.png",
            "occupancy_landfold.pdf",
            "occupancy_landfold_lj38.pdf",
            "occupancy_landfold_lj75.pdf",
            "occupancy_landfold_lj98.pdf",
        ),
    ),
)


def pinned_env() -> dict[str, str]:
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "0"
    env["PYTHONHASHSEED"] = "0"
    return env


def rebuild(out: Path) -> list[str]:
    fetch_paper_data()
    out.mkdir(parents=True, exist_ok=True)
    env = pinned_env()
    failed: list[str] = []
    for script, names in STEPS:
        if script == "fig_si_panels.py":
            args = [str(out)]
        else:
            args = [str(out / names[0])]
        command = [sys.executable, str(ROOT / "scripts" / script), *args]
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        missing = [name for name in names if not (out / name).is_file()]
        if completed.returncode != 0 or missing:
            failed.append(script)
            continue
    return failed


def blob(path: Path) -> str:
    proc = subprocess.run(
        ["git", "hash-object", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip())
    return proc.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("usage: rebuild_figures.py OUTDIR")
    out = Path(args[0])
    failed = rebuild(out)
    for script in failed:
        sys.stderr.write(f"not rebuilt: {script}\n")
    names = sorted(name for _script, group in STEPS for name in group)
    for name in names:
        path = out / name
        if path.is_file():
            sys.stdout.write(f"{name} {blob(path)}\n")
        else:
            sys.stdout.write(f"{name} not-rebuilt\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
