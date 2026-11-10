#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Re-quench the 24 saved panel structures with lj_requench.

This does not rerun the search and does not write into the raw logs.
A hit is re-quenched energy <= E_known + 1e-6. The re-quench calls are
the evaluations returned by WarmLbfgs::minimize and are kept apart from
the search calls.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MARGIN = 1e-6
SIZES = (38, 55, 75, 98)
SEEDS = (6000, 6001, 6002, 6003, 6004, 6005)


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    raise SystemExit("not inside a git work tree")


def load_known(path: Path) -> dict[int, float]:
    known = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("n\t"):
            continue
        n_text, energy, _file, _line = line.split("\t")
        known[int(n_text)] = float(energy)
    return known


def parse_log(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
        elif " " in line:
            key, value = line.split(" ", 1)
        else:
            continue
        fields[key] = value
    return fields


def requench(binary: Path, xyz: Path) -> tuple[float, int]:
    proc = subprocess.run(
        [str(binary), xyz.as_posix()],
        cwd=repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or f"{xyz.name} exited {proc.returncode}")
    fields = parse_log(proc.stdout)
    return float(fields["energy"]), int(fields["calls"])


def find_binary(root: Path, requested: str | None) -> Path:
    if requested:
        path = Path(requested)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise SystemExit("missing lj_requench_xyz binary")
        return path
    for rel in (
        "target/release/examples/lj_requench_xyz",
        "../tmp/anneal-target/release/examples/lj_requench_xyz",
    ):
        path = root / rel
        if path.is_file():
            return path
    raise SystemExit("missing lj_requench_xyz binary")


def main() -> None:
    here = Path(__file__).resolve().parent
    root = repo_root()
    binary = find_binary(root, sys.argv[1] if len(sys.argv) > 1 else None)
    known = load_known(here / "e_known.tsv")
    panel = here / "panel"
    rows = []
    for n in SIZES:
        for seed in SEEDS:
            matches = sorted(panel.glob(f"*-lj{n}-{seed}.xyz"))
            logs = sorted(panel.glob(f"*-lj{n}-{seed}.log"))
            if len(matches) != 1 or len(logs) != 1:
                raise SystemExit(f"expected one xyz and one log for LJ{n} seed {seed}")
            raw = float(parse_log(logs[0].read_text(encoding="utf-8"))["lowest"])
            energy, calls = requench(binary, matches[0])
            limit = known[n] + MARGIN
            hit = energy <= limit
            rows.append((n, seed, raw, energy, hit, calls))
    lines = ["n\tseed\traw_e\trequenched_e\thit\trequench_calls"]
    for n, seed, raw, energy, hit, calls in rows:
        lines.append(
            f"{n}\t{seed}\t{raw:.10f}\t{energy:.16f}\t{'true' if hit else 'false'}\t{calls}"
        )
    table = "\n".join(lines) + "\n"
    (here / "requench.tsv").write_text(table, encoding="utf-8")
    prose = [
        "Copyright (c) anneal developers",
        "",
        "Re-quenched panel. A hit is requenched_e <= E_known + 1e-6.",
        "requench_calls are the WarmLbfgs::minimize evaluations and are not search calls.",
        "raw_e is the lowest line of the untouched search log.",
        "",
    ]
    floors = {}
    for n in SIZES:
        group = [row for row in rows if row[0] == n]
        hits = sum(1 for row in group if row[4])
        floors[n] = hits
        prose.append(f"LJ{n} floor: {hits}/6. E_known {known[n]:.6f}.")
        prose.append("")
        prose.append("seed raw_e requenched_e hit requench_calls")
        for _n, seed, raw, energy, hit, calls in group:
            prose.append(
                f"{seed} {raw:.10f} {energy:.16f} {'hit' if hit else 'miss'} {calls}"
            )
        prose.append("")
    (here / "requench.md").write_text("\n".join(prose), encoding="utf-8")
    note_path = here / "PREDICTIONS.md"
    note = note_path.read_text(encoding="utf-8")
    for n, hits in floors.items():
        note, count = re.subn(
            rf"LJ{n} floor: .*",
            f"LJ{n} floor: {hits}/6.",
            note,
            count=1,
        )
        if count != 1:
            raise SystemExit(f"PREDICTIONS.md has no LJ{n} floor line")
    old_para = (
        "The four lines above were counted inside the 0.001 stop window. "
        "They are not re-quenched hits. `requench_panel.py` replaces them "
        "from the re-quenched energies. `floors.md` keeps the window count. "
        "A count below its floor is a mechanical FAIL."
    )
    new_para = (
        "The four lines above are the re-quench floors from requench.tsv, "
        "written by requench_panel.py. A hit is re-quenched E <= E_known + 1e-6. "
        "Per-seed raw energy, re-quenched energy, hit flag, and re-quench calls "
        "are in requench.md. floors.md keeps the earlier 0.001 window count and "
        "is not the floor. A count below its floor is a mechanical FAIL."
    )
    if old_para in note:
        note = note.replace(old_para, new_para, 1)
    elif new_para not in note:
        raise SystemExit("PREDICTIONS.md is missing the floor-status paragraph")
    note_path.write_text(note, encoding="utf-8")
    sys.stdout.write(table)


if __name__ == "__main__":
    main()
