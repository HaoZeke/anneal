#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Two anneal revisions against each other and GMIN on the water_gmin rows.

Reads ``bench/water-gmin/rows.csv`` and keeps, for each size, the twelve
seeds of each named revision. ``calls`` is ``evals_to_hit``, the budget for
a miss. ``fisher_p`` is the two-sided Fisher exact test on the hit counts
and ``mwu_calls_p`` the two-sided Mann-Whitney U test on calls, branch
against develop; the ``_gmin`` columns test the branch against GMIN. Times
are type-7 medians in seconds.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_h2h import (  # noqa: E402
    HIT_TOL,
    SEEDS,
    fisher_exact_two_sided,
    mann_whitney_u_two_sided,
    type7,
)
from run_water_gmin import CAMBRIDGE, SIZES  # noqa: E402


def arm(rows: list[dict[str, str]], n: int, revision: str) -> list[dict[str, str]]:
    by_seed = {r["seed"]: r for r in rows
               if r["system"] == f"tip4p{n}" and r["code_revision"] == revision}
    missing = [s for s in SEEDS if str(s) not in by_seed]
    if missing:
        raise SystemExit(f"tip4p{n} {revision}: no rows for seeds {missing}")
    return [by_seed[str(s)] for s in SEEDS]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--develop", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--gmin", required=True)
    parser.add_argument("--output", type=argparse.FileType("w"), default=sys.stdout)
    args = parser.parse_args()
    with args.rows.open() as handle:
        rows = list(csv.DictReader(handle))
    writer = None
    for n in SIZES:
        ref = CAMBRIDGE[n][0]
        arms = {name: arm(rows, n, rev) for name, rev in
                (("develop", args.develop), ("branch", args.branch), ("gmin", args.gmin))}
        hits = {k: [float(r["best"]) <= ref + HIT_TOL for r in v] for k, v in arms.items()}
        calls = {k: [float(r["evals_to_hit"]) for r in v] for k, v in arms.items()}
        h = {k: sum(v) for k, v in hits.items()}
        seeds = len(SEEDS)

        def med(name: str, key: str) -> str:
            return f"{type7([float(r[key]) for r in arms[name]]):.6f}"

        out = {
            "system": f"tip4p{n}",
            "seeds": str(seeds),
            "develop_hits": f"{h['develop']}/{seeds}",
            "branch_hits": f"{h['branch']}/{seeds}",
            "gmin_hits": f"{h['gmin']}/{seeds}",
            "develop_calls_median": f"{type7(calls['develop']):.1f}",
            "branch_calls_median": f"{type7(calls['branch']):.1f}",
            "gmin_calls_median": f"{type7(calls['gmin']):.1f}",
            "develop_best_median_kj": med("develop", "best"),
            "branch_best_median_kj": med("branch", "best"),
            "develop_wall_median_s": med("develop", "wall_s"),
            "branch_wall_median_s": med("branch", "wall_s"),
            "gmin_wall_median_s": med("gmin", "wall_s"),
            "develop_pot_median_s": med("develop", "pot_s"),
            "develop_outside_median_s": med("develop", "outside_s"),
            "branch_pot_median_s": med("branch", "pot_s"),
            "branch_outside_median_s": med("branch", "outside_s"),
            "fisher_p": f"{fisher_exact_two_sided(h['branch'], seeds - h['branch'], h['develop'], seeds - h['develop']):.6f}",
            "mwu_calls_p": f"{mann_whitney_u_two_sided(calls['branch'], calls['develop']):.6f}",
            "fisher_p_gmin": f"{fisher_exact_two_sided(h['branch'], seeds - h['branch'], h['gmin'], seeds - h['gmin']):.6f}",
            "mwu_calls_p_gmin": f"{mann_whitney_u_two_sided(calls['branch'], calls['gmin']):.6f}",
            "develop_revision": args.develop,
            "branch_revision": args.branch,
            "gmin_revision": args.gmin,
        }
        if writer is None:
            writer = csv.DictWriter(args.output, fieldnames=list(out), lineterminator="\n")
            writer.writeheader()
        writer.writerow(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
