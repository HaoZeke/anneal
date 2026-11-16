#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Anneal alone on every Sutton-Chen system that has GMIN head-to-head rows.

The sc-n12 panel is seeds 2000-2011 at the budgets of run_sc.PAPER_BUDGETS,
from the sc-n12 starts, with h2h_sc of the harness revision. Its rows are
appended to bench/metal-cluster-h2h/sc-n12/seeds.csv. 9-6 at N=78 is the
iso and iso-hop arms of sc_h2h on seeds 9532-9563 at 2000000 calls,
appended to data/sc96_78_hits.csv by scripts/sc96_78_hits.py. Then sc96-55
on seeds 2012-2059 with h2h_sc of e3ed7dad and of the harness revision,
written to bench/metal-cluster-h2h/sc-n12/sc96-55-seeds.csv, which says
whether the calls of sc96-55 moved between the two.

--panel-system limits the panel to the systems named, --no-n78 and
--no-check skip the other two parts, and --rgpot-revision names the rgpot
build of RGPOT_SC_LIB in an rgpot_revision column of the panel rows.

Each row records the anneal revision, the host and the harness revision.
A seed already stored for this harness revision is skipped, and no row is
rewritten. One job at a time, on core 0. The process refuses to run
unless the UTS hostname is bench and the working directory is /work.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import run_sc as sc  # noqa: E402
import run_sc96_gmin as gm  # noqa: E402
import run_sc_n12 as n12  # noqa: E402
import sc96_78_hits as n78  # noqa: E402

CHECK_SYSTEM = "sc96-55"
CHECK_SEEDS = tuple(range(2012, 2060))
CHECK_FIELDS = (
    "system",
    "seed",
    "budget",
    "hit",
    "calls",
    "wall_s",
    "best",
    "code_revision",
    "host",
    "harness_revision",
)


def start(binary: Path, runs: Path, system: str, seed: int) -> Path:
    coords = runs / system / f"start-{seed}.xyz"
    coords.parent.mkdir(parents=True, exist_ok=True)
    if not coords.is_file():
        subprocess.run([str(binary), "coords", system, str(seed), str(coords)], check=True)
    return coords


def panel(args, head: str, revision: str) -> None:
    wanted = list(n12.SEED_FIELDS) + (["rgpot_revision"] if args.rgpot_revision else [])
    fields, rows = gm.load(args.panel_seeds) if args.panel_seeds.is_file() else (wanted, [])
    if list(fields) != wanted:
        raise SystemExit(f"{args.panel_seeds} columns are not {', '.join(wanted)}")
    systems = n12.EIGHTY + n12.SMALLER
    for system in [s for s in systems if not args.panel_system or s in args.panel_system]:
        for seed in n12.SEEDS:
            if any(r["system"] == system and r["arm"] == "anneal" and r["seed"] == str(seed)
                   and r["harness_revision"] == head for r in rows):
                print(f"skip {system} anneal seed {seed}", flush=True)
                continue
            print(f"run {system} anneal seed {seed}", flush=True)
            coords = start(args.binary, args.runs, system, seed)
            log = args.logs / system / f"anneal-{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            parsed, diag = n12.run_anneal(args.binary, system, seed, sc.PAPER_BUDGETS[system], coords, log, args.runs)
            diag.pop("_arms", None)
            row = n12.as_seed(system, "anneal", parsed, diag)
            row.update({"code_revision": head, "host": os.uname().nodename, "harness_revision": head})
            if args.rgpot_revision:
                row["rgpot_revision"] = args.rgpot_revision
            rows.append(row)
            args.panel_seeds.parent.mkdir(parents=True, exist_ok=True)
            gm.save(args.panel_seeds, wanted, rows)
            print(f"done {system} anneal seed {seed} best={row['best']} hit={row['hit']} wall={row['wall_s']}",
                  flush=True)
    if args.panel_summary:
        gm.save(args.panel_summary, list(n12.SUMMARY_FIELDS), n12.summarise(rows, head, revision))


def check(args, head: str) -> None:
    rows = n12.load_rows(args.check_rows)
    budget = sc.PAPER_BUDGETS[CHECK_SYSTEM]
    builds = ((args.check_binary, args.check_revision), (args.binary, head))
    for seed in CHECK_SEEDS:
        coords = start(args.binary, args.runs, CHECK_SYSTEM, seed)
        for binary, code in builds:
            if any(r["seed"] == str(seed) and r["code_revision"] == code and r["harness_revision"] == head
                   for r in rows):
                continue
            log = args.logs / f"{CHECK_SYSTEM}-check" / f"{code[:8]}-{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            parsed, _ = n12.run_anneal(binary, CHECK_SYSTEM, seed, budget, coords, log, args.runs / "check" / code[:8])
            hit = parsed["hit"] in {"1", "true", "True"}
            # Both builds stop at the hit, so the calls charged are the calls
            # to the hit; the e3ed7dad build prints the budget as evals_to_hit.
            rows.append({
                "system": CHECK_SYSTEM,
                "seed": str(seed),
                "budget": str(budget),
                "hit": "1" if hit else "0",
                "calls": str(int(float(parsed["charged"]))),
                "wall_s": f"{float(parsed['wall_s']):.6f}",
                "best": f"{float(parsed['best']):.8f}",
                "code_revision": code,
                "host": os.uname().nodename,
                "harness_revision": head,
            })
            gm.save(args.check_rows, list(CHECK_FIELDS), rows)
            print(f"done {CHECK_SYSTEM} {code[:8]} seed {seed} calls={rows[-1]['calls']} hit={rows[-1]['hit']}",
                  flush=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sc-h2h", type=Path)
    parser.add_argument("--check-binary", type=Path)
    parser.add_argument("--check-revision")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--panel-seeds", type=Path, required=True)
    parser.add_argument("--panel-summary", type=Path)
    parser.add_argument("--panel-system", action="append", choices=tuple(sc.PAPER_BUDGETS))
    parser.add_argument("--rgpot-revision")
    parser.add_argument("--check-rows", type=Path)
    parser.add_argument("--n78-rows", type=Path)
    parser.add_argument("--starts78", type=Path)
    parser.add_argument("--reference78", type=Path)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--no-n78", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args(argv)
    needed = ([] if args.no_n78 else ["sc_h2h", "n78_rows", "starts78", "reference78"]) + \
             ([] if args.no_check else ["check_binary", "check_revision", "check_rows"])
    if missing := [name for name in needed if getattr(args, name) is None]:
        parser.error("missing " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    if Path.cwd() != Path("/work"):
        print(f"working directory is {Path.cwd()}, expected /work", file=sys.stderr)
        return 2
    head = args.head_file.read_text().strip()
    panel(args, head, "")
    if not args.no_n78:
        n78.append_runs(str(args.sc_h2h), head, args.reference78.read_text().strip(), args.starts78,
                        args.logs / "sc96-78", args.n78_rows)
    if not args.no_check:
        check(args, head)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
