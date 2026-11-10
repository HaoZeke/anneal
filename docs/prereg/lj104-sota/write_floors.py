#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the no-regression floors from the raw default-panel logs.

The logs are the untouched stdout of examples/h2h_seed.rs. A hit is a
lowest energy at or below the reference plus 0.001. The references are
the Lennard-Jones lines in examples/h2h_arm3.rs. The budgets are the
PAPER_BUDGETS entries named in PREDICTIONS.md.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REFERENCES = {
    38: -173.928427,
    55: -279.248470,
    75: -397.492331,
    98: -543.665361,
}
TOL = 0.001
SEEDS = (6000, 6001, 6002, 6003, 6004, 6005)


def parse_log(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key] = value
    return fields


def receipt_field(text: str, name: str) -> str:
    prefix = name + " "
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise SystemExit(f"receipt is missing {name}")


def main() -> None:
    here = Path(__file__).resolve().parent
    panel = here / "panel"
    logs = sorted(panel.glob("*.log"))
    if len(logs) != 24:
        raise SystemExit(f"expected 24 logs, found {len(logs)}")
    rows = []
    for path in logs:
        fields = parse_log(path.read_text(encoding="utf-8"))
        n = int(fields["n"])
        seed = int(fields["seed"])
        energy = float(fields["lowest"])
        calls = int(fields["calls"])
        wall = float(fields["wall_seconds"])
        hit = energy <= REFERENCES[n] + TOL
        span = (panel / f"{path.stem}.span").read_text(encoding="utf-8").split()
        rows.append(
            {
                "file": path.name,
                "n": n,
                "seed": seed,
                "energy": energy,
                "calls": calls,
                "wall": wall,
                "hit": hit,
                "span": (float(span[0]), float(span[1])),
            }
        )
    seen = {(row["n"], row["seed"]) for row in rows}
    expected = {(n, seed) for n in REFERENCES for seed in SEEDS}
    if seen != expected:
        raise SystemExit("logs are not the six seeds on LJ38, LJ55, LJ75, and LJ98")
    for earlier, later in zip(rows, rows[1:]):
        if later["span"][0] < earlier["span"][1]:
            raise SystemExit("panel spans overlap")

    scored = (panel / "rows.tsv").read_text(encoding="utf-8").splitlines()
    header = scored[0].split("\t")
    body = [dict(zip(header, line.split("\t"))) for line in scored[1:]]
    if len(body) != 24:
        raise SystemExit("rows.tsv does not have 24 rows")
    for row in body:
        for name in ("runner_blob", "head", "tree"):
            if not row.get(name):
                raise SystemExit(f"a scored row is missing {name}")
    blobs = {row["runner_blob"] for row in body}
    heads = {row["head"] for row in body}
    trees = {row["tree"] for row in body}
    if len(blobs) != 1 or len(heads) != 1 or len(trees) != 1:
        raise SystemExit("scored rows disagree on runner, HEAD, or TREE")

    start = (panel / "start.receipt").read_text(encoding="utf-8")
    end = (panel / "end.receipt").read_text(encoding="utf-8")
    if receipt_field(start, "boot_id") != receipt_field(end, "boot_id"):
        raise SystemExit("start and end boot_id differ")
    session = float(receipt_field(end, "uptime").split()[0]) - float(
        receipt_field(start, "uptime").split()[0]
    )
    wall_sum = sum(row["wall"] for row in rows)
    span_sum = sum(row["span"][1] - row["span"][0] for row in rows)

    lines = [
        "Copyright (c) anneal developers",
        "",
        "Floors from the raw logs of the default panel. A hit is lowest <= reference + 0.001.",
        f"runner_blob {next(iter(blobs))}",
        f"head {next(iter(heads))}",
        f"tree {next(iter(trees))}",
        f"boot_id {receipt_field(start, 'boot_id')}",
        f"wall_sum_seconds {wall_sum:.6f}",
        f"span_sum_seconds {span_sum:.6f}",
        f"session_uptime_seconds {session:.6f}",
        "",
    ]
    floors = {}
    for n in (38, 55, 75, 98):
        group = [row for row in rows if row["n"] == n]
        hits = sum(1 for row in group if row["hit"])
        floors[n] = hits
        lines.append(f"LJ{n} floor: {hits}/6.")
        for row in group:
            mark = "hit" if row["hit"] else "miss"
            lines.append(
                f"LJ{n} seed {row['seed']} {mark} energy {row['energy']:.10f} "
                f"calls {row['calls']} wall_seconds {row['wall']:.3f}"
            )
        lines.append("")
    text = "\n".join(lines)
    (here / "floors.md").write_text(text, encoding="utf-8")

    predictions = here / "PREDICTIONS.md"
    note = predictions.read_text(encoding="utf-8")
    for n, hits in floors.items():
        pattern = rf"LJ{n} floor: .*"
        replacement = f"LJ{n} floor: {hits}/6."
        note, count = re.subn(pattern, replacement, note, count=1)
        if count != 1:
            raise SystemExit(f"PREDICTIONS.md has no LJ{n} floor line")
    new_para = (
        "The four floors are the hit counts in the raw logs under panel/, "
        "written by write_floors.py. Per-seed energies, calls, and wall times "
        "are in floors.md. A count below its floor is a mechanical FAIL."
    )
    old_para = (
        "No committed file states a six-seed default hit count for these sizes. "
        "Counts that are committed use other sample sizes, and this note does not "
        "convert them into floors. Those records are "
        "src/methods/cluster_hopping/preset.rs lines 20-24, "
        "docs/orgmode/architecture.org lines 191-193, and "
        "scripts/elja_sota_guard.sh lines 3-8. While a floor is UNFILLED, that "
        "size cannot be passed."
    )
    if old_para in note:
        note = note.replace(old_para, new_para, 1)
    elif new_para not in note:
        raise SystemExit("PREDICTIONS.md is missing the floor paragraph")
    predictions.write_text(note, encoding="utf-8")
    print(text)
    print(f"wall_sum {wall_sum:.6f}")
    print(f"span_sum {span_sum:.6f}")
    print(f"session_uptime {session:.6f}")


if __name__ == "__main__":
    sys.exit(main())
