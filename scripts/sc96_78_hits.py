# Copyright (c) anneal developers
"""Score GMIN, iso and iso-hop on Sutton-Chen 9-6 N=78.

Seeds 9532 through 9563, two million energy calls, the Cambridge table
energy as the stop. Starts are drawn by this script into ``build/``.
``data/sc96_78_hits.csv`` has one row per seed and arm. This script runs
iso and iso-hop. The GMIN rows are written by
``scripts/metal_h2h/enter_sc96_gmin.sh``, and the table scores those of the
latest harness revision. The how-to figure and its table are written from
that file.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sc96_78_common import binary, parse_run, reference_energy, write_start  # noqa: E402

SCRATCH = ROOT / "build" / "sc96-78"
OUT = ROOT / "data" / "sc96_78_hits.csv"
FIGURE = SCRATCH / "hits.png"
HOWTO = ROOT / "docs" / "orgmode" / "howto" / "sutton-chen-head-to-head.org"
SEED0 = 9532
SEED1 = 9563
BUDGET = 2_000_000
N = 78
ARMS = ("gmin", "iso", "iso-hop")
ANNEAL_ARMS = ("iso", "iso-hop")
WORKERS = 6


def current_gmin(rows: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """The GMIN rows of the last harness revision in the file."""
    gmin = [row for row in rows if row["arm"] == "gmin" and row.get("harness_revision")]
    if not gmin:
        raise SystemExit(f"{OUT} has no GMIN rows from scripts/metal_h2h/enter_sc96_gmin.sh")
    harness = gmin[-1]["harness_revision"]
    current = sorted((row for row in gmin if row["harness_revision"] == harness), key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in current] != list(range(SEED0, SEED1 + 1)):
        raise SystemExit(f"GMIN rows of harness {harness[:8]} do not cover seeds {SEED0}-{SEED1}")
    return harness, current


def runs_from_rows(name: str, current: list[dict[str, str]]) -> Path:
    directory = SCRATCH / "rows" / name
    directory.mkdir(parents=True, exist_ok=True)
    for row in current:
        hit = row["hit"] == "1"
        (directory / f"seed{row['seed']}.out").write_text(
            f"hit: {'true' if hit else 'false'}\n"
            f"hit_calls: {row['hit_calls'] if hit else 'none'}\n"
            f"spent: {row['spent']}\n"
            f"best: {row['best']}\n"
        )
    return directory
def one(task: tuple[str, int, str, str]) -> dict[str, str]:
    arm, seed, reference, exe = task
    start = SCRATCH / "starts" / f"seed{seed}.xyz"
    dest = SCRATCH / arm / f"seed{seed}.out"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not start.is_file():
        write_start(start, N, seed)
    if not dest.is_file():
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        text = subprocess.check_output(
            [exe, arm, "9-6", str(N), str(BUDGET), str(seed), str(start), reference],
            cwd=ROOT,
            env=env,
            text=True,
        )
        dest.write_text(text)
    parsed = parse_run(dest.read_text())
    hit = parsed["hit"] == "true"
    return {
        "arm": arm,
        "seed": str(seed),
        "hit": "1" if hit else "0",
        "hit_calls": parsed["hit_calls"] if hit else "",
        "spent": parsed["spent"],
        "best": parsed["best"],
        "reference": reference,
    }


def write_csv(fields: list[str], rows: list[dict[str, str]]) -> None:
    for path in (OUT, SCRATCH / "hits.csv"):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def score(directories: dict[str, Path]) -> str:
    plot = ROOT / "docs" / "scripts" / "plot_sc_hits.py"
    labels = [f"{arm}={directories[arm]}" for arm in ARMS]
    text = subprocess.check_output(
        [sys.executable, str(plot), str(SEED0), str(SEED1), str(BUDGET), str(FIGURE), *labels],
        cwd=ROOT,
        text=True,
    )
    return text


def sc_step_text() -> str:
    """The Sutton-Chen step, from the constants in scripts/metal_h2h/run_sc.py."""
    path = ROOT / "scripts" / "metal_h2h" / "run_sc.py"
    spec = importlib.util.spec_from_file_location("run_sc_step", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return f"{module.sc_step():.6f}"


def result_paragraph(rows: list[dict[str, str]], harness: str) -> str:
    by_arm: dict[str, list[dict[str, str]]] = {arm: [] for arm in ARMS}
    earlier = []
    for row in rows:
        if row["arm"] != "gmin" or row.get("harness_revision") == harness:
            by_arm[row["arm"]].append(row)
        else:
            earlier.append(row)
    lines = [
        "The table is written from =data/sc96_78_hits.csv=.",
        "The figure is written to =build/sc96-78/hits.png=.",
        "GMIN is the Wales program at SVNREV aac1548a, linked against OpenBLAS.",
        f"Its rows are those of harness {harness[:8]}, one seed at a time on one core, "
        "from =scripts/metal_h2h/enter_sc96_gmin.sh=.",
        "Its file is the 9-6 row of =scripts/metal_h2h/run_sc.py=: temperature 10, "
        f"step {sc_step_text()}, "
        "the container GMIN sizes for Sutton-Chen, and the quench block there.",
    ]
    for arm in ARMS:
        arm_rows = by_arm[arm]
        hits = [row for row in arm_rows if row["hit"] == "1"]
        misses = len(arm_rows) - len(hits)
        calls = sorted(int(row["hit_calls"]) for row in hits)
        if not calls:
            median = "no hit"
        else:
            mid = len(calls) // 2
            mid = (calls[len(calls) // 2 - 1] + calls[len(calls) // 2]) / 2
            if len(calls) % 2:
                mid = calls[len(calls) // 2]
            median = f"{mid:.1f}".removesuffix(".0")
        lines.append(
            f"={arm}= hits {len(hits)} of {len(arm_rows)} seeds"
            + (f", median {median} calls to a hit." if hits else ".")
            + (f" It misses {misses}." if misses else "")
        )
    if earlier:
        lines.append(
            f"The earlier GMIN rows, with temperature 0.8, step 0.40 and a container radius of 30, "
            f"hit {sum(row['hit'] == '1' for row in earlier)} of {len(earlier)} seeds. They are kept in the file."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", action="store_true",
                        help="score every arm from the committed rows, without running iso and iso-hop")
    from_rows = parser.parse_args(argv).rows
    with OUT.open() as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    harness, current = current_gmin(rows)
    reference = f"{reference_energy(SCRATCH):.4f}"
    if any(row["reference"] != reference for row in current):
        raise SystemExit(f"GMIN rows of harness {harness[:8]} stop at another reference than {reference}")
    directories = {"gmin": runs_from_rows(f"gmin-{harness[:8]}", current)}
    if from_rows:
        for arm in ANNEAL_ARMS:
            directories[arm] = runs_from_rows(arm, [row for row in rows if row["arm"] == arm])
    else:
        exe = str(binary("sc_h2h"))
        for seed in range(SEED0, SEED1 + 1):
            write_start(SCRATCH / "starts" / f"seed{seed}.xyz", N, seed)
        tasks = [
            (arm, seed, reference, exe)
            for arm in ANNEAL_ARMS
            for seed in range(SEED0, SEED1 + 1)
        ]
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            fresh = {(row["arm"], row["seed"]): row for row in pool.map(one, tasks)}
        for row in rows:
            row.update(fresh.pop((row["arm"], row["seed"]), {}))
        rows.extend(fresh.values())
        write_csv(fields, rows)
        directories.update({arm: SCRATCH / arm for arm in ANNEAL_ARMS})
    table = score(directories).rstrip("\n") + "\n"
    paragraph = result_paragraph(rows, harness)
    page = HOWTO.read_text()
    start = page.index("#+begin_example\nlabel hits")
    end = page.index("#+end_example", start) + len("#+end_example")
    page = page[:start] + "#+begin_example\n" + table + "#+end_example" + page[end:]
    old = (
        "Here =runs/gmin= has the results of Wales's global minimisation program (GMIN) on the same 32 starts.\n"
        "On all 32 seeds =iso-hop= reaches the global minimum, with a median of 138950.5 calls.\n"
        "GMIN needs a median of 398438 calls and solves 31 seeds.\n"
        "=iso= stops short on 7 seeds.\n"
    )
    if old in page:
        page = page.replace(old, paragraph, 1)
    else:
        anchors = (
            "The table and the figure are written from",
            "The table is written from",
        )
        start_p = next(page.index(anchor) for anchor in anchors if anchor in page)
        end_p = page.index("When a curve rises earlier")
        page = page[:start_p] + paragraph + page[end_p:]
    old_captions = (
        "Fraction of the 32 evaluation seeds of 9-6 N=78 at the global minimum against energy calls, "
        "for Wales's global minimisation program (GMIN) and two anneal arms.",
        "Fraction of seeds 9532 to 9563 of 9-6 N=78 at the global minimum against energy calls, for the iso and iso-hop arms.",
    )
    new_caption = (
        "Fraction of seeds 9532 to 9563 of 9-6 N=78 at the global minimum against energy calls, "
        "for GMIN, iso and iso-hop."
    )
    for caption in old_captions:
        page = page.replace(caption, new_caption)
    old_cmd = """#+begin_src bash
python3 docs/scripts/plot_sc_hits.py 9532 9563 2000000 hits.png \\
  gmin=runs/gmin iso=runs/iso iso-hop=runs/iso-hop
#+end_src
"""
    new_cmd = """#+begin_src bash
python3 scripts/sc96_78_hits.py
#+end_src

The script draws the figure with =docs/scripts/plot_sc_hits.py= from the run logs.
"""
    if old_cmd in page:
        page = page.replace(old_cmd, new_cmd, 1)
    page = page.replace("[[file:fig/sc96-78-hits.png]]\n", "")
    HOWTO.write_text(page)
    print(table)
    print(paragraph)


if __name__ == "__main__":
    main(sys.argv[1:])
