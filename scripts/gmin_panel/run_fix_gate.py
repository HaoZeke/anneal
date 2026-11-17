# Copyright (c) anneal developers
"""Same-session gate for the general GMIN-loss fix.

The gate runs the reviewed base and candidate serially from the same start on
all 13 paper systems. Both disable structural templates. The candidate adds
pair-potential length scaling, acceptance-driven plateau heat and a
budget-sharing independent population. Held-out mode also runs GMIN on the
two loss systems.
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_panel as panel

GATE_SEEDS = tuple(range(2000, 2012))
HELDOUT_SEEDS = tuple(range(3000, 3012))
LOSS_SYSTEMS = ("lj150", "m80r6")
ARMS = {
    "baseline": {
        "H2H_TEMPLATES": "0",
        "H2H_MACKAY": "0",
        "H2H_PAIR_SCALE": "0",
        "H2H_PLATEAU_HEAT": "0",
        "H2H_ACCEPTANCE_ADAPT": "0",
        "H2H_POPULATION": "0",
    },
    "fix": {
        "H2H_TEMPLATES": "0",
        "H2H_MACKAY": "0",
        "H2H_PAIR_SCALE": "1",
        "H2H_PLATEAU_HEAT": "1",
        "H2H_ACCEPTANCE_ADAPT": "1",
        "H2H_POPULATION": "1",
    },
}

SUMMARY_FIELDS = [
    "campaign",
    "system",
    "budget",
    "left",
    "right",
    "left_hits",
    "left_n",
    "right_hits",
    "right_n",
    "fisher_p",
    "left_evals_median",
    "right_evals_median",
    "mannwhitney_u_calls",
    "mannwhitney_p_calls",
    "left_wall_median_s",
    "right_wall_median_s",
    "left_pot_median_s",
    "right_pot_median_s",
    "left_outside_median_s",
    "right_outside_median_s",
    "same_seeds",
    "gained_seeds",
    "lost_seeds",
    "base_revision",
    "code_revision",
    "gmin_revision",
    "host",
    "seeds",
]


def rows_for(
    rows: list[dict[str, str]],
    harness_revision: str,
    base_revision: str,
    campaign: str,
    system: str,
    code: str,
) -> list[dict[str, str]]:
    seed_set = set(HELDOUT_SEEDS if campaign == "heldout" else GATE_SEEDS)
    expected_revision = {
        "baseline": base_revision,
        "fix": harness_revision,
        "gmin": panel.GMIN_REV,
    }[code]
    selected: dict[int, dict[str, str]] = {}
    for row in rows:
        seed = int(row["seed"])
        if (
            row["system"] == system
            and row["code"] == code
            and row["code_revision"] == expected_revision
            and row["harness_revision"] == harness_revision
            and seed in seed_set
        ):
            selected.setdefault(seed, row)
    return [selected[seed] for seed in sorted(selected)]


def comparison(
    campaign: str,
    system: str,
    left_name: str,
    right_name: str,
    left: list[dict[str, str]],
    right: list[dict[str, str]],
    revision: str,
    base_revision: str,
) -> dict[str, str]:
    def values(group: list[dict[str, str]], name: str) -> list[float]:
        return [float(row[name]) for row in group]

    left_hits = sum(row["hit"] == "1" for row in left)
    right_hits = sum(row["hit"] == "1" for row in right)
    u_calls, p_calls = panel.mann_whitney(values(left, "evals_to_hit"), values(right, "evals_to_hit"))
    left_by_seed = {int(row["seed"]): row for row in left}
    right_by_seed = {int(row["seed"]): row for row in right}
    common = sorted(set(left_by_seed) & set(right_by_seed))
    gained = [
        seed
        for seed in common
        if left_by_seed[seed]["hit"] == "0" and right_by_seed[seed]["hit"] == "1"
    ]
    lost = [
        seed
        for seed in common
        if left_by_seed[seed]["hit"] == "1" and right_by_seed[seed]["hit"] == "0"
    ]
    same = sum(
        left_by_seed[seed]["hit"] == right_by_seed[seed]["hit"]
        and left_by_seed[seed]["evals_to_hit"] == right_by_seed[seed]["evals_to_hit"]
        and float(left_by_seed[seed]["best"]).hex() == float(right_by_seed[seed]["best"]).hex()
        for seed in common
    )
    both = left + right
    return {
        "campaign": campaign,
        "system": system,
        "budget": str(panel.PAPER_BUDGETS[system]),
        "left": left_name,
        "right": right_name,
        "left_hits": str(left_hits),
        "left_n": str(len(left)),
        "right_hits": str(right_hits),
        "right_n": str(len(right)),
        "fisher_p": f"{panel.fisher(left_hits, len(left), right_hits, len(right)):.6g}",
        "left_evals_median": f"{panel.type7(values(left, 'evals_to_hit'), 0.5):.2f}",
        "right_evals_median": f"{panel.type7(values(right, 'evals_to_hit'), 0.5):.2f}",
        "mannwhitney_u_calls": f"{u_calls:.1f}",
        "mannwhitney_p_calls": f"{p_calls:.6g}",
        "left_wall_median_s": f"{panel.type7(values(left, 'wall_s'), 0.5):.6f}",
        "right_wall_median_s": f"{panel.type7(values(right, 'wall_s'), 0.5):.6f}",
        "left_pot_median_s": f"{panel.type7(values(left, 'pot_s'), 0.5):.6f}",
        "right_pot_median_s": f"{panel.type7(values(right, 'pot_s'), 0.5):.6f}",
        "left_outside_median_s": f"{panel.type7(values(left, 'outside_s'), 0.5):.6f}",
        "right_outside_median_s": f"{panel.type7(values(right, 'outside_s'), 0.5):.6f}",
        "same_seeds": str(same),
        "gained_seeds": " ".join(map(str, gained)),
        "lost_seeds": " ".join(map(str, lost)),
        "base_revision": base_revision,
        "code_revision": revision,
        "gmin_revision": panel.GMIN_REV if "gmin" in (left_name, right_name) else "",
        "host": " ".join(sorted({row["host"] for row in both})),
        "seeds": " ".join(str(seed) for seed in common),
    }


def summarise(
    rows: list[dict[str, str]], revision: str, base_revision: str
) -> list[dict[str, str]]:
    summary = []
    for system, *_rest in panel.SYSTEMS:
        baseline = rows_for(rows, revision, base_revision, "gate", system, "baseline")
        fix = rows_for(rows, revision, base_revision, "gate", system, "fix")
        if baseline and fix:
            summary.append(
                comparison(
                    "gate",
                    system,
                    "baseline",
                    "fix",
                    baseline,
                    fix,
                    revision,
                    base_revision,
                )
            )
    for system in LOSS_SYSTEMS:
        fix = rows_for(rows, revision, base_revision, "heldout", system, "fix")
        gmin = rows_for(rows, revision, base_revision, "heldout", system, "gmin")
        if fix and gmin:
            summary.append(
                comparison(
                    "heldout",
                    system,
                    "fix",
                    "gmin",
                    fix,
                    gmin,
                    revision,
                    base_revision,
                )
            )
    return summary


def write_summary(
    path: Path, rows: list[dict[str, str]], revision: str, base_revision: str
) -> None:
    panel.write_csv(path, SUMMARY_FIELDS, summarise(rows, revision, base_revision))


def shared_start(
    bin_dir: Path,
    tree: Path,
    system: str,
    seed: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = panel.spec(system)[6]
    subprocess.run(
        [str(bin_dir / binary), "coords", system, str(seed), str(path)],
        cwd=tree,
        env=panel.single_thread_env(),
        capture_output=True,
        text=True,
        check=True,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--base-bin", type=Path)
    parser.add_argument("--base-tree", type=Path)
    parser.add_argument("--bin", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--gmin-source", type=Path)
    parser.add_argument("--lj-data", type=Path)
    parser.add_argument("--morse-data", type=Path)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv[1:])
    if args.summarize:
        write_summary(
            args.summary,
            panel.read_rows(args.rows),
            args.revision,
            args.base_revision,
        )
        return 0
    if socket.gethostname() != "bench" or os.sched_getaffinity(0) != {0}:
        raise SystemExit("the gate requires hostname bench and affinity {0}")
    needed = (
        args.bin,
        args.tree,
        args.base_bin,
        args.base_tree,
        args.work,
        args.logs,
        args.gmin_source,
        args.lj_data,
        args.morse_data,
    )
    if any(value is None for value in needed):
        parser.error("the build, work, log, source and data paths are required")
    for tree, revision in (
        (args.tree, args.revision),
        (args.base_tree, args.base_revision),
    ):
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tree,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if head != revision:
            raise SystemExit(f"{tree} is at {head}, not {revision}")
    source = dict(
        line.split("=", 1)
        for line in args.gmin_source.read_text().splitlines()
        if "=" in line
    )
    if source.get("svnrev") != panel.GMIN_REV:
        raise SystemExit(f"GMIN SVNREV {source.get('svnrev')} is not {panel.GMIN_REV}")

    available_systems = LOSS_SYSTEMS if args.heldout else tuple(row[0] for row in panel.SYSTEMS)
    systems = tuple(system for system in available_systems if not args.system or system in args.system)
    default_seeds = HELDOUT_SEEDS if args.heldout else GATE_SEEDS
    seeds = tuple(args.seed) if args.seed else default_seeds
    codes = ("baseline", "fix", "gmin") if args.heldout else ("baseline", "fix")
    lj_data = args.lj_data.read_text()
    morse_data = args.morse_data.read_text()
    lj_data += "" if lj_data.endswith("\n") else "\n"
    morse_data += "" if morse_data.endswith("\n") else "\n"
    host = socket.gethostname()
    cpu = panel.cpu_model()
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    done = {panel.row_key(row) for row in panel.read_rows(args.rows)}

    for system in systems:
        for label, bin_dir, tree in (
            ("baseline", args.base_bin, args.base_tree),
            ("fix", args.bin, args.tree),
        ):
            budget, reference = panel.binary_budget(bin_dir, tree, system)
            if (
                budget != panel.PAPER_BUDGETS[system]
                or abs(reference - panel.spec(system)[3]) > 1e-8
            ):
                raise SystemExit(f"{system}: {label} binary budget or reference drift")
        for seed in seeds:
            coords = args.work / "starts" / system / f"{seed}.xyz"
            shared_start(args.base_bin, args.base_tree, system, seed, coords)
            for code in codes:
                code_revision = {
                    "baseline": args.base_revision,
                    "fix": args.revision,
                    "gmin": panel.GMIN_REV,
                }[code]
                key = (system, code, str(seed), code_revision, args.revision)
                if key in done:
                    continue
                log = args.logs / ("heldout" if args.heldout else "gate") / system / f"{code}-{seed}.log"
                if code == "gmin":
                    result = panel.run_gmin(
                        args.bin,
                        args.tree,
                        system,
                        seed,
                        args.work / "gmin" / system / str(seed),
                        log,
                        lj_data,
                        morse_data,
                    )
                else:
                    bin_dir, tree = (
                        (args.base_bin, args.base_tree)
                        if code == "baseline"
                        else (args.bin, args.tree)
                    )
                    result = panel.run_anneal(
                        bin_dir,
                        tree,
                        system,
                        seed,
                        log,
                        env_overrides=ARMS[code],
                        coords=coords,
                    )
                    result.pop("head", None)
                row = {
                    "system": system,
                    "code": code,
                    "driver": panel.spec(system)[5] if code != "gmin" else "GMIN",
                    "seed": str(seed),
                    "budget": str(panel.PAPER_BUDGETS[system]),
                    **result,
                    "code_revision": code_revision,
                    "harness_revision": args.revision,
                    "host": host,
                    "cpu": cpu,
                    "session": session,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                panel.append_row(args.rows, row)
                done.add(key)
                print(
                    f"{system} {code} {seed} hit={row['hit']} calls={row['evals_to_hit']} "
                    f"wall={row['wall_s']} inside={row['pot_s']} outside={row['outside_s']}",
                    flush=True,
                )
    write_summary(
        args.summary,
        panel.read_rows(args.rows),
        args.revision,
        args.base_revision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
