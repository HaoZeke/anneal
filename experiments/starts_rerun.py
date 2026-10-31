#!/usr/bin/env python3
"""Run the preset that separates molecules starting in covalent contact, and decide it.

``Config::separate_start_molecules`` moves every declared group rigidly away
from the mean of the group centroids by the smallest factor 1.05^k, k at most
60, that leaves no bond under the graph rule between atoms of different
groups. ``run`` runs one campaign job on the tree that adds it. ``table``
applies the decision rule of PROTOCOL-starts.md: every campaign run is line
for line its run on the hessopt tree, the LJ digest is the harness
handover's, no hit of the hessopt water board is lost, and every separated
board seed has a recorded best that keeps its separated start's graph. It
also reproduces the separation in Python on every stored start.

Usage::

    python experiments/starts_rerun.py --campaign C --out OUT run SYSTEM SEED
    python experiments/starts_rerun.py --campaign C --out OUT --hessopt H --board B \\
        --hessopt-board HB --graph-board IB --gap-board PB --lj-harness L --lj LJ table
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_rerun import run_seed  # noqa: E402
from graph_rerun import TOLERANCE, board, board_hits, graph_valid, read_xyz  # noqa: E402
from robustness_rerun import COVALENT, write_csv  # noqa: E402

RATIO = 1.05
STEPS = 60
BOARD_BUDGET = 4000
BOARD_FIELDS = ("isomer", "isomer_rmsd_A", "isomer_reason", "force_calls", "best_energy")
# Lines this harness adds, and the wall-clock time.
NEW_LINES = re.compile(r"^(separate_start_molecules |START_SEPARATION |WALL )")
SEPARATION = re.compile(r"^k (\d+) s (\S+)$")
WATERS = 6


def run_job(campaign: Path, out: Path, system: str, seed: int) -> None:
    run_seed(campaign, out, "preset", system, seed)
    path = out / f"{system}_s{seed}" / "preset.json"
    entry = json.loads(path.read_text())
    if "start_separation" in entry:
        return
    log = (out / f"{system}_s{seed}" / "preset" / "anneal.out").read_text()
    entry["start_separation"] = re.search(r"^START_SEPARATION (.+)$", log, re.M).group(1)
    path.write_text(json.dumps(entry, indent=1) + "\n")


def declared_groups(path: Path, n: int) -> list[list[int]]:
    comment = path.read_text().splitlines()[1].strip()
    if not comment.startswith("groups "):
        return [list(range(n))]
    return [
        [int(i) for i in piece.split(",")] for piece in comment[7:].split(";") if piece
    ]


def separation(symbols: list[str], x: np.ndarray, groups: list[list[int]]):
    """PROTOCOL-starts.md section 1: ``None``, ``"unresolved"`` or ``(k, s, moved)``."""
    n = len(symbols)
    if len(groups) < 2:
        return None
    declared: list[int | None] = [None] * n
    for g, atoms in enumerate(groups):
        for atom in atoms:
            if atom < n and declared[atom] is None:
                declared[atom] = g
    compact: dict[int, int] = {}
    numbered, count = [], 0
    for d in declared:
        if d is not None and d in compact:
            numbered.append(compact[d])
            continue
        numbered.append(count)
        if d is not None:
            compact[d] = count
        count += 1
    group = np.array(numbered)
    radii = np.array([COVALENT[s] for s in symbols])

    def bonded(y: np.ndarray) -> bool:
        for i in range(n):
            for j in range(i + 1, n):
                if group[i] == group[j]:
                    continue
                cut = TOLERANCE * (radii[i] + radii[j])
                d = y[i] - y[j]
                if float(d @ d) < cut * cut:
                    return True
        return False

    if not bonded(x):
        return None
    centroids = np.array([x[group == g].mean(axis=0) for g in range(count)])
    offsets = centroids - centroids.mean(axis=0)
    for k in range(1, STEPS + 1):
        s = RATIO**k
        moved = x + (s - 1.0) * offsets[group]
        if not bonded(moved):
            return k, s, moved
    return "unresolved"


def rust_separation(text: str | None):
    if text in (None, "none"):
        return None
    if text == "unresolved":
        return "unresolved"
    found = SEPARATION.match(text)
    return int(found.group(1)), float(found.group(2))


def same(python, rust, written: Path | None) -> bool:
    """Whether the Python separation is the Rust one, and its structure the one written."""
    if python is None or rust is None or python == "unresolved" or rust == "unresolved":
        return python == rust
    if python[0] != rust[0] or abs(python[1] - rust[1]) > 1e-6:
        return False
    return written is None or (
        written.exists() and float(np.abs(read_xyz(written)[1] - python[2]).max()) < 1e-8
    )


def table(args: argparse.Namespace) -> None:
    out = args.out
    stats: dict = {}
    lines: list[str] = []
    checks: dict = {}

    # E1 and T1 on the campaign; T3 on its starts.
    jobs = sorted(p.parent for p in args.hessopt.glob("*/preset.json"))
    differing, missing, accounting, campaign_t3 = [], [], [], []
    separated_jobs = []
    for root in jobs:
        name = root.name
        mine = out / name
        if not (mine / "preset.json").exists():
            missing.append(name)
            continue
        entry = json.loads((mine / "preset.json").read_text())
        new = (mine / "preset" / "anneal.out").read_text().splitlines()
        old = (root / "preset" / "anneal.out").read_text().splitlines()
        if [ln for ln in new if not NEW_LINES.match(ln)] != [
            ln for ln in old if not NEW_LINES.match(ln)
        ]:
            differing.append(name)
        if not entry["engine_calls"] == entry["charged"] <= entry["budget"]:
            accounting.append(name)
        if entry["start_separation"] != "none":
            separated_jobs.append(f"{name} {entry['start_separation']}")
        start = mine / "start.xyz"
        symbols, x = read_xyz(start)
        python = separation(symbols, x, declared_groups(start, len(symbols)))
        if not same(python, rust_separation(entry["start_separation"]), None):
            campaign_t3.append(name)
    e1 = len(jobs) == 96 and not missing and not differing
    checks["T1_campaign_accounting_failures"] = accounting
    checks["T3_campaign_disagreements"] = campaign_t3
    stats["campaign"] = {
        "jobs": len(jobs),
        "missing": missing,
        "differing": differing,
        "separated": separated_jobs,
    }

    # E2.
    harness = args.lj_harness.read_bytes() if args.lj_harness else None
    lj = {str(p): p.exists() and p.read_bytes() == harness for p in args.lj or []}
    e2 = harness is not None and len(lj) == 1 and all(lj.values())

    # The board: E3, E4, T1, T2, T3.
    new = board(args.board)
    hessopt = board(args.hessopt_board)
    invariant = board(args.graph_board) if args.graph_board else {}
    preset = board(args.gap_board) if args.gap_board else {}
    geometry = args.board.parent / "geometry"
    rows, t2, t3, e4_failures, over = [], [], [], [], []
    groups = [[3 * w, 3 * w + 1, 3 * w + 2] for w in range(WATERS)]
    for key in sorted(new, key=int):
        row = new[key]
        rust = rust_separation(row.get("start_separation"))
        start = geometry / f"start_s{key}.xyz"
        separated = geometry / f"separated_s{key}.xyz"
        best = geometry / f"best_s{key}.xyz"
        bound = separated if separated.exists() else start
        kept = graph_valid(bound, best) if best.exists() and bound.exists() else None
        symbols, x = read_xyz(start)
        python = separation(symbols, x, groups)
        if not same(python, rust, separated):
            t3.append(key)
        is_separated = isinstance(rust, tuple)
        if rust is None and any(row.get(f) != hessopt.get(key, {}).get(f) for f in BOARD_FIELDS):
            t2.append(key)
        if is_separated and not (best.exists() and kept):
            e4_failures.append(key)
        if int(row.get("force_calls") or 0) > BOARD_BUDGET:
            over.append(key)
        rows.append(
            {
                "seed": int(key),
                "start_separation": row.get("start_separation"),
                "python_k": "none"
                if python is None
                else python
                if python == "unresolved"
                else python[0],
                "hit": int(row.get("isomer_reason") == "inside gate"),
                "hessopt_hit": int(hessopt.get(key, {}).get("isomer_reason") == "inside gate"),
                "invariant_hit": int(invariant.get(key, {}).get("isomer_reason") == "inside gate"),
                "preset_hit": int(preset.get(key, {}).get("isomer_reason") == "inside gate"),
                "isomer": row.get("isomer"),
                "isomer_reason": row.get("isomer_reason"),
                "best_energy": row.get("best_energy"),
                "start_graph_rejected": row.get("start_graph_rejected"),
                "best_keeps_bound_graph": "none" if kept is None else int(kept),
                "first_record_force_calls": row.get("first_record_force_calls"),
                "force_calls": row.get("force_calls"),
            }
        )
    write_csv(out / "board_starts.csv", rows)
    hits, hessopt_hits = board_hits(new), board_hits(hessopt)
    lost = sorted(hessopt_hits - hits, key=int)
    e3 = len(new) == 32 and not lost
    separated_seeds = [r["seed"] for r in rows if str(r["start_separation"]).startswith("k ")]
    e4 = not e4_failures
    accepted = e1 and e2 and e3 and e4
    checks["T1_board_over_budget"] = over
    checks["T2_unseparated_board_seeds_differing"] = t2
    checks["T3_board_disagreements"] = t3
    stats["decision"] = {
        "E1_campaign_untouched": e1,
        "E2_lj_identical": e2,
        "E2_files": lj,
        "E3_no_board_hit_lost": e3,
        "E3_lost": lost,
        "E4_separated_starts_searched": e4,
        "E4_failures": e4_failures,
        "accepted": accepted,
    }

    def tally(seeds: list[int] | None) -> dict:
        pick = [r for r in rows if seeds is None or r["seed"] in seeds]
        return {
            "seeds": len(pick),
            "hits": sum(r["hit"] for r in pick),
            "hessopt_hits": sum(r["hessopt_hit"] for r in pick),
            "invariant_hits": sum(r["invariant_hit"] for r in pick),
            "preset_hits": sum(r["preset_hit"] for r in pick),
        }

    stats["board"] = {
        "all": tally(None),
        "separated": tally(separated_seeds),
        "separated_seeds": separated_seeds,
        "gained_vs_hessopt": sorted(hits - hessopt_hits, key=int),
        "lost_vs_hessopt": lost,
        "factors": {r["seed"]: r["start_separation"] for r in rows if r["seed"] in separated_seeds},
        "first_record_force_calls": {
            r["seed"]: r["first_record_force_calls"] for r in rows if r["seed"] in separated_seeds
        },
    }
    lines += [
        "## Decision",
        "",
        f"E1 campaign untouched: {e1} ({len(jobs)} hessopt runs, {len(missing)} missing, "
        f"{len(differing)} differing: {' '.join(differing) or 'none'})",
        f"E2 LJ digest identical: {e2} (" + ", ".join(f"{k} {v}" for k, v in lj.items()) + ")",
        f"E3 no board hit lost: {e3} ({len(lost)} of {len(hessopt_hits)} hessopt board hits lost: "
        f"{' '.join(lost) or 'none'})",
        f"E4 separated starts searched: {e4} ({len(separated_seeds)} separated seeds; failures "
        f"{' '.join(e4_failures) or 'none'})",
        f"-> {'accepted' if accepted else 'rejected'}",
        "",
        "## Water board (reported)",
        "",
        "| seeds | starts | hessopt | invariant | 0008 preset |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("all", "all"), ("separated", "separated")):
        t = stats["board"][key]
        lines.append(
            f"| {label} ({t['seeds']}) | {t['hits']} | {t['hessopt_hits']} "
            f"| {t['invariant_hits']} | {t['preset_hits']} |"
        )
    lines += [
        "",
        f"Gained against the hessopt board: "
        f"{' '.join(stats['board']['gained_vs_hessopt']) or 'none'}.",
        "",
        "| seed | separation | first record (calls) | best keeps the separated graph | isomer "
        "| hit |",
        "| ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for r in rows:
        if r["seed"] in separated_seeds:
            lines.append(
                f"| {r['seed']} | {r['start_separation']} | {r['first_record_force_calls']} "
                f"| {r['best_keeps_bound_graph']} | {r['isomer']} | {r['hit']} |"
            )
    stats["self_checks"] = checks
    lines += ["", "## Self-checks", ""]
    for key, value in checks.items():
        shown = (" ".join(map(str, value)) or "none") if isinstance(value, list) else value
        lines.append(f"{key} {shown}")
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=str) + "\n")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("starts-rerun"))
    parser.add_argument("--hessopt", type=Path, help="hessopt campaign runs")
    parser.add_argument("--board", type=Path, help="this arm's water board.txt")
    parser.add_argument("--hessopt-board", type=Path, help="hessopt water board.txt")
    parser.add_argument("--graph-board", type=Path, help="invariant water board.txt")
    parser.add_argument("--gap-board", type=Path, help="0008 preset water board.txt")
    parser.add_argument("--lj-harness", type=Path, help="lj_digest output of the harness")
    parser.add_argument("--lj", type=Path, action="append", help="lj_digest output to compare")
    parser.add_argument("command", choices=["run", "table"])
    parser.add_argument("job", nargs="*")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        system, seed = args.job
        run_job(args.campaign, args.out, system, int(seed))
        return
    if None in (args.hessopt, args.board, args.hessopt_board):
        parser.error("table needs --hessopt, --board and --hessopt-board")
    table(args)


if __name__ == "__main__":
    main()
