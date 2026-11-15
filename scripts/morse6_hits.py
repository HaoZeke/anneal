# Copyright (c) anneal developers
"""Hits of the Morse rho = 6 panel, reduced units against the pair length.

The search is ``examples/morse_table``. Sizes, seeds and the budget are
the how-to's. One row per run is written to ``data/morse6_hits.csv``,
and the how-to figure is drawn from that file.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "examples" / "fixtures" / "morse_ccd.csv"
OUT = ROOT / "data" / "morse6_hits.csv"
SCRATCH = ROOT / "build" / "morse6"
FIGURE = SCRATCH / "morse6-hits.svg"
RHO = "6"
SIZES = "35,37,43,44,60"
BUDGET = "100000"
SEED0 = "12000"
SEEDS = "16"
ARMS = ("length", "reduced")


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


def run_arm(exe: Path, arm: str) -> list[dict[str, str]]:
    cmd = [str(exe), str(TABLE), RHO, SIZES, BUDGET, SEED0, SEEDS]
    if arm == "reduced":
        cmd.append("reduced")
    text = subprocess.check_output(cmd, cwd=ROOT, text=True)
    rows = []
    for line in text.splitlines():
        rho, size, seed, hit, calls, best, gap, *_rest = line.split("\t")
        rows.append(
            {
                "rho": rho,
                "size": size,
                "seed": seed,
                "arm": arm,
                "hit": hit,
                "calls": calls,
                "best": best,
                "gap": gap,
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fields = ["rho", "size", "seed", "arm", "hit", "calls", "best", "gap"]
    for path in (OUT, SCRATCH / "morse6_hits.csv"):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def hits_by_size(rows: list[dict[str, str]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["size"], row["arm"])
        counts[key] = counts.get(key, 0) + int(row["hit"])
    return counts


def write_figure(rows: list[dict[str, str]]) -> None:
    counts = hits_by_size(rows)
    sizes = SIZES.split(",")
    length = [counts[(size, "length")] for size in sizes]
    reduced = [counts[(size, "reduced")] for size in sizes]
    positions = list(range(len(sizes)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(
        [p - width / 2 for p in positions],
        reduced,
        width,
        label="reduced units",
        color="#9e9e9e",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        length,
        width,
        label="pair length",
        color="#4c78a8",
    )
    ax.set_xticks(positions, [f"N = {size}" for size in sizes])
    ax.set_ylabel(f"Hits of {SEEDS} seeds")
    ax.set_ylim(0, int(SEEDS) + 0.5)
    ax.legend(frameon=False)
    ax.set_title(f"Morse rho = {RHO}, budget {BUDGET}")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)
    plt.close(fig)


def write_page(rows: list[dict[str, str]]) -> None:
    counts = hits_by_size(rows)
    lines = [
        "=scripts/morse6_hits.py= runs the search below and writes one row per seed to =data/morse6_hits.csv=.",
        "The figure is written to =build/morse6/morse6-hits.svg= from that file.",
        "",
    ]
    for size in SIZES.split(","):
        lines.append(
            f"- N = {size}: {counts[(size, 'length')]} hits at the pair length, "
            f"{counts[(size, 'reduced')]} on reduced units, of {SEEDS} seeds."
        )
    page = (ROOT / "docs" / "orgmode" / "howto" / "morse-clusters.org").read_text()
    start = page.index("=scripts/morse6_hits.py=")
    end = page.index("- Sizes:")
    page = page[:start] + "\n".join(lines) + "\n\n" + page[end:]
    # The committed image is not this run.
    image = """#+begin_export rst
.. image:: /_static/morse6-hits.svg
   :width: 520
   :align: center
   :alt: Hits of 16 seeds at rho = 6 for N = 35, 37, 43, 44 and 60, reduced units against the pair length scale
#+end_export

"""
    page = page.replace(image, "")
    (ROOT / "docs" / "orgmode" / "howto" / "morse-clusters.org").write_text(page)


def main() -> None:
    exe = binary("morse_table")
    rows: list[dict[str, str]] = []
    for arm in ARMS:
        rows.extend(run_arm(exe, arm))
    write_csv(rows)
    write_figure(rows)
    write_page(rows)
    counts = hits_by_size(rows)
    for size in SIZES.split(","):
        print(
            f"N={size} length {counts[(size, 'length')]} reduced {counts[(size, 'reduced')]}"
        )


if __name__ == "__main__":
    sys.exit(main())
