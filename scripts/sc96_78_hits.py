# Copyright (c) anneal developers
"""Score GMIN, iso and iso-hop on Sutton-Chen 9-6 N=78.

Seeds 9532 through 9563, two million energy calls, the Cambridge table
energy as the stop. Starts are drawn by this script into ``build/``.
``data/sc96_78_hits.csv`` has one row per seed, arm and harness revision.
This script runs iso and iso-hop one seed at a time on core 0 and appends
their rows; a seed already stored for this harness revision is skipped.
The GMIN rows are written by ``scripts/metal_h2h/enter_sc96_gmin.sh``.
The table scores each arm's rows of the latest harness revision in which
it has rows. The how-to figure and its table are written from that file.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
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


def current(rows: list[dict[str, str]], arm: str) -> tuple[str, list[dict[str, str]]]:
    """The rows of `arm` from the last harness revision of that arm in the file.

    The first iso and iso-hop rows have no harness revision; GMIN rows must have one.
    """
    group = [row for row in rows if row["arm"] == arm and (arm != "gmin" or row.get("harness_revision"))]
    if not group:
        raise SystemExit(f"{OUT} has no {arm} rows" + (" from scripts/metal_h2h/enter_sc96_gmin.sh" if arm == "gmin" else ""))
    harness = group[-1].get("harness_revision", "")
    latest = sorted((row for row in group if row.get("harness_revision", "") == harness), key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in latest] != list(range(SEED0, SEED1 + 1)):
        raise SystemExit(f"{arm} rows of harness {harness[:8] or '(none)'} do not cover seeds {SEED0}-{SEED1}")
    return harness, latest


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
def run_arm(exe: str, arm: str, seed: int, start: Path, reference: str, log: Path) -> dict[str, str]:
    """One seed of an anneal arm on core 0, as a row without its provenance."""
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    log.parent.mkdir(parents=True, exist_ok=True)
    text = subprocess.check_output(
        ["taskset", "-c", "0", exe, arm, "9-6", str(N), str(BUDGET), str(seed), str(start), reference],
        env=env,
        text=True,
    )
    log.write_text(text)
    parsed = parse_run(text)
    hit = parsed["hit"] == "true"
    return {
        "arm": arm,
        "seed": str(seed),
        "hit": "1" if hit else "0",
        "hit_calls": parsed["hit_calls"] if hit else "",
        "spent": parsed["spent"],
        "best": parsed["best"],
        "reference": reference,
        "wall_s": f"{float(parsed['wall_s']):.6f}",
        "pot_s": f"{float(parsed['pot_s']):.6f}",
    }


def append_runs(exe: str, head: str, reference: str, starts: Path, logs: Path, path: Path = OUT) -> None:
    """Iso and iso-hop on every seed not yet stored for harness `head`, appended to `path`."""
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for arm in ANNEAL_ARMS:
        for seed in range(SEED0, SEED1 + 1):
            if any(r["arm"] == arm and r["seed"] == str(seed) and r.get("harness_revision") == head for r in rows):
                print(f"skip {arm} seed {seed}", flush=True)
                continue
            print(f"run {arm} seed {seed}", flush=True)
            row = run_arm(exe, arm, seed, starts / f"seed{seed}.xyz", reference, logs / arm / f"seed{seed}.out")
            row.update({"code_revision": head, "host": os.uname().nodename, "harness_revision": head})
            rows.append(row)
            write_csv(fields, rows, path)
            print(f"done {arm} seed {seed} best={row['best']} hit={row['hit']} wall={row['wall_s']}", flush=True)


def write_csv(fields: list[str], rows: list[dict[str, str]], path: Path = OUT) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    tmp.replace(path)


def score(directories: dict[str, Path]) -> str:
    plot = ROOT / "docs" / "scripts" / "plot_sc_hits.py"
    labels = [f"{arm}={directories[arm]}" for arm in ARMS]
    text = subprocess.check_output(
        [sys.executable, str(plot), str(SEED0), str(SEED1), str(BUDGET), str(FIGURE), *labels],
        cwd=ROOT,
        text=True,
    )
    return text


def median_text(values: list[float]) -> str:
    ordered = sorted(values)
    mid = len(ordered) // 2
    value = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return f"{value:.1f}".removesuffix(".0")


def same_runs(left: list[dict[str, str]], right: list[dict[str, str]]) -> bool:
    key = lambda row: (row["hit"], row["hit_calls"], row["spent"], row["best"])  # noqa: E731
    return {row["seed"]: key(row) for row in left} == {row["seed"]: key(row) for row in right}


def result_paragraph(rows: list[dict[str, str]]) -> str:
    latest = {arm: current(rows, arm) for arm in ARMS}
    harness = latest["gmin"][0]
    lines = [
        "The table is written from =data/sc96_78_hits.csv=.",
        "The figure is written to =build/sc96-78/hits.png=.",
        "GMIN is the Wales program at SVNREV aac1548a, linked against OpenBLAS.",
        f"Its rows are those of harness {harness[:8]}, one seed at a time on one core, "
        "from =scripts/metal_h2h/enter_sc96_gmin.sh=.",
        "Its file is the 9-6 row of =scripts/metal_h2h/run_sc.py=: temperature 10, step 0.251984, "
        "the container GMIN sizes for Sutton-Chen, and the quench block there.",
    ]
    anneal = {latest[arm][0] for arm in ANNEAL_ARMS}
    if len(anneal) == 1 and (only := next(iter(anneal))):
        lines.append(f"The iso and iso-hop rows are those of harness {only[:8]}, one seed at a time on core 0, "
                     "from =scripts/metal_h2h/enter_sc_anneal.sh=.")
    for arm in ARMS:
        arm_rows = latest[arm][1]
        hits = [row for row in arm_rows if row["hit"] == "1"]
        misses = len(arm_rows) - len(hits)
        lines.append(
            f"={arm}= hits {len(hits)} of {len(arm_rows)} seeds"
            + (f", median {median_text([int(row['hit_calls']) for row in hits])} calls to a hit." if hits else ".")
            + (f" It misses {misses}." if misses else "")
        )
    for arm in ARMS:
        arm_rows = latest[arm][1]
        if all(row.get("wall_s") and row.get("pot_s") for row in arm_rows):
            wall = median_text([round(float(row["wall_s"]), 3) for row in arm_rows])
            pot = median_text([round(float(row["pot_s"]), 3) for row in arm_rows])
            lines.append(f"The median ={arm}= seed takes {wall} s, {pot} s of it inside the potential.")
    for arm in ANNEAL_ARMS:
        earlier = [row for row in rows if row["arm"] == arm and row.get("harness_revision", "") != latest[arm][0]]
        if earlier:
            lines.append(f"The earlier ={arm}= rows, without a harness revision, hit "
                         f"{sum(row['hit'] == '1' for row in earlier)} of {len(earlier)} seeds. They are kept in the file.")
            if same_runs(earlier, latest[arm][1]):
                lines.append(f"Their calls to a hit, calls spent and best energies match the latest ={arm}= rows seed for seed.")
    earlier = [row for row in rows if row["arm"] == "gmin" and row.get("harness_revision") != harness]
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
    reference = f"{reference_energy(SCRATCH):.4f}"
    if not from_rows:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        for seed in range(SEED0, SEED1 + 1):
            write_start(SCRATCH / "starts" / f"seed{seed}.xyz", N, seed)
        append_runs(str(binary("sc_h2h")), head, reference, SCRATCH / "starts", SCRATCH)
    with OUT.open() as handle:
        rows = list(csv.DictReader(handle))
    directories = {}
    for arm in ARMS:
        harness, latest = current(rows, arm)
        if any(row["reference"] != reference for row in latest):
            raise SystemExit(f"{arm} rows of harness {harness[:8]} stop at another reference than {reference}")
        directories[arm] = runs_from_rows(f"{arm}-{harness[:8] or 'first'}", latest)
    table = score(directories).rstrip("\n") + "\n"
    paragraph = result_paragraph(rows)
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
