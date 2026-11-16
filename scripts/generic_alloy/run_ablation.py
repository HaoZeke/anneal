#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial ablation of the labelled-cluster search on the Gupta alloys.

Two builds take part: ``develop`` and ``branch``. Each arm is one build
with an environment. With the priors off (``H2H_PRIORS=0``) the arms are
develop, the branch with only the exchange descent, the branch with the
descent and the ranked polish, and the full branch. With the priors on the
arms are develop and the full branch, and LJ75 runs through ``h2h_seed``.

A Gupta hit is a finished relaxation within 0.001 eV of the reference. An
LJ75 hit is a call below the Cambridge energy plus 1e-4. A miss is stored
as the full budget. Within a system the (seed, arm) pairs are shuffled,
so a slow stretch of the host falls on every arm. One core.

The panel campaign runs develop and the full branch with the priors off
on the six alloys and Au55, and LJ75, which has no prior switch, on its
preset.

The two g2 campaigns run develop and the branch on the six alloys, Au55
and M80 at rho = 6 through ``h2h_morse``, one with the priors off and one
on. The priors-off campaign adds the branch with the stalled-incumbent
reorder (``H2H_REORDER=1``) and LJ75 on its preset.

The two g3 campaigns are the g2 campaigns again in one session, the
priors-off one with the reorder that records and does not adopt. The six
alloys, which move between arms, run 24 seeds; the rest run 12.

Rows are appended to a CSV, one line per run, with the revision of the
build and the host. A row already in that file for the same build
revision is kept and its run is skipped; the summary reads only the rows
of the two revisions being compared. The summary compares every arm with the control: Fisher's
exact test on hits, and Mann-Whitney on calls (misses at the budget),
wall, time inside the potential and time outside it.

This process refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

SEEDS = tuple(range(2000, 2012))
WIDE_SEEDS = tuple(range(2000, 2024))
SCHEDULE_SEED = 20261114
LJ75_REFERENCE = -397.492331
LJ75_BUDGET = 4_000_000
MORSE = {"m80r6": (4_000_000, -378.333471)}
LJ_HIT_TOL = 1.0e-4

ALLOYS = {
    "cu6au32": (400_000, -130.79994836),
    "cu19au19": (400_000, -122.99148391),
    "cu27au28": (1_000_000, -182.93136675),
    "pt12cu11": (200_000, -96.03443577),
    "cu10au45": (1_000_000, -191.30598075),
    "cu30au8": (400_000, -115.53258986),
    "au55": (1_000_000, -191.03085280),
}

# Name, build, priors, H2H_HOMOTOP, further environment. The first arm of
# a campaign is its control. Systems listed under "wide" run WIDE_SEEDS.
CAMPAIGNS = {
    "priors-off": {
        "systems": ("cu6au32", "cu19au19", "cu27au28", "pt12cu11", "cu10au45", "cu30au8"),
        "arms": (
            ("develop", "develop", "0", None, {}),
            ("descent", "branch", "0", "ranked,window", {}),
            ("descent-ranked", "branch", "0", "window", {}),
            ("full", "branch", "0", None, {}),
        ),
    },
    "priors-off-panel": {
        "systems": (
            "cu6au32",
            "cu19au19",
            "cu27au28",
            "pt12cu11",
            "cu10au45",
            "cu30au8",
            "au55",
            "lj75",
        ),
        "arms": (
            ("develop", "develop", "0", None, {}),
            ("full", "branch", "0", None, {}),
        ),
    },
    "priors-on": {
        "systems": (
            "cu6au32",
            "cu19au19",
            "cu27au28",
            "pt12cu11",
            "cu10au45",
            "cu30au8",
            "au55",
            "lj75",
        ),
        "arms": (
            ("develop", "develop", None, None, {}),
            ("full", "branch", None, None, {}),
        ),
    },
    "g2-priors-off": {
        "systems": (
            "cu6au32",
            "cu19au19",
            "cu27au28",
            "pt12cu11",
            "cu10au45",
            "cu30au8",
            "au55",
            "lj75",
            "m80r6",
        ),
        "arms": (
            ("develop", "develop", "0", None, {}),
            ("branch", "branch", "0", None, {}),
            ("reorder", "branch", "0", None, {"H2H_REORDER": "1"}),
        ),
    },
    "g2-priors-on": {
        "systems": (
            "cu6au32",
            "cu19au19",
            "cu27au28",
            "pt12cu11",
            "cu10au45",
            "cu30au8",
            "au55",
            "m80r6",
        ),
        "arms": (
            ("develop", "develop", None, None, {}),
            ("branch", "branch", None, None, {}),
        ),
    },
    "g3-priors-off": {
        "systems": (
            "cu6au32",
            "cu19au19",
            "cu27au28",
            "pt12cu11",
            "cu10au45",
            "cu30au8",
            "au55",
            "lj75",
            "m80r6",
        ),
        "wide": ("cu6au32", "cu19au19", "cu27au28", "pt12cu11", "cu10au45", "cu30au8"),
        "arms": (
            ("develop", "develop", "0", None, {}),
            ("branch", "branch", "0", None, {}),
            ("reorder", "branch", "0", None, {"H2H_REORDER": "1"}),
        ),
    },
    "g3-priors-on": {
        "systems": (
            "cu6au32",
            "cu19au19",
            "cu27au28",
            "pt12cu11",
            "cu10au45",
            "cu30au8",
            "au55",
            "m80r6",
        ),
        "wide": ("cu6au32", "cu19au19", "cu27au28", "pt12cu11", "cu10au45", "cu30au8"),
        "arms": (
            ("develop", "develop", None, None, {}),
            ("branch", "branch", None, None, {}),
        ),
    },
}

ROW_FIELDS = [
    "campaign",
    "system",
    "arm",
    "seed",
    "budget",
    "reference",
    "priors",
    "homotop",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "code_revision",
    "host",
    "cpu",
]

SUMMARY_FIELDS = [
    "campaign",
    "system",
    "arm",
    "control",
    "n",
    "hits",
    "hits_over_n",
    "budget",
    "evals_median_to_hit",
    "evals_median",
    "wall_median_s",
    "pot_median_s",
    "outside_median_s",
    "fisher_p",
    "mannwhitney_p_evals",
    "mannwhitney_p_wall",
    "mannwhitney_p_pot",
    "mannwhitney_p_outside",
    "code_revision",
    "host",
    "seeds",
    "schedule_seed",
]


def type7(sorted_values: list[float], p: float) -> float:
    """Hyndman-Fan type 7. ``sorted_values`` is ascending."""
    count = len(sorted_values)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return sorted_values[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def seeds_of(spec: dict, system: str) -> tuple[int, ...]:
    return WIDE_SEEDS if system in spec.get("wide", ()) else SEEDS


def schedule(
    systems: tuple[str, ...], spec: dict, arms: tuple[str, ...], rng_seed: int
) -> list[tuple[str, int, str]]:
    """Systems in the given order. Pairs inside a system are shuffled."""
    rng = random.Random(rng_seed)
    order: list[tuple[str, int, str]] = []
    for system in systems:
        jobs = [(system, seed, arm) for seed in seeds_of(spec, system) for arm in arms]
        rng.shuffle(jobs)
        order.extend(jobs)
    return order


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[key] = "1"
    for key in (
        "H2H_PRIORS",
        "H2H_HOMOTOP",
        "H2H_GENERAL",
        "H2H_BEST",
        "H2H_BUDGET",
        "H2H_REFERENCE",
        "H2H_HANDOVER",
        "H2H_REORDER",
    ):
        env.pop(key, None)
    return env


def run_gupta(
    binary: Path, system: str, seed: int, priors, homotop, extra: dict, log: Path
) -> dict:
    morse = system in MORSE
    budget, reference = MORSE[system] if morse else ALLOYS[system]
    env = single_thread_env()
    if not morse:
        env["H2H_BUDGET"] = str(budget)
        env["H2H_REFERENCE"] = repr(reference)
        if homotop is not None:
            env["H2H_HOMOTOP"] = homotop
        env.update(extra)
    if priors is not None:
        env["H2H_PRIORS"] = priors
    command = ["taskset", "-c", "0", str(binary), "run", system, str(seed)]
    proc = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    lines = [line for line in proc.stdout.splitlines() if ",anneal," in line]
    if len(lines) != 1:
        raise RuntimeError(f"expected one result row, saw {len(lines)}")
    fields = lines[0].split(",")
    if fields[0] != system or int(fields[2]) != seed or int(fields[3]) != budget:
        raise RuntimeError(f"row {lines[0]}")
    return {
        "budget": budget,
        "reference": reference,
        "hit": fields[4] == "true",
        "evals_to_hit": int(fields[5]),
        "charged": int(fields[6]),
        "wall_s": float(fields[7]),
        "best": float(fields[8]),
        "pot_s": float(fields[9]),
        "outside_s": float(fields[10]),
    }


def run_lj75(binary: Path, seed: int, log: Path) -> dict:
    stop = LJ75_REFERENCE + LJ_HIT_TOL
    with tempfile.TemporaryDirectory() as scratch:
        command = [
            "taskset",
            "-c",
            "0",
            str(binary),
            "lj",
            "75",
            "0",
            str(LJ75_BUDGET),
            str(seed),
            "recommended",
            repr(stop),
            str(Path(scratch) / "run"),
        ]
        proc = subprocess.run(
            command, env=single_thread_env(), capture_output=True, text=True, check=False
        )
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"lj75 seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    rows = [line for line in proc.stdout.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError(f"expected one h2h_row, saw {len(rows)}")
    values = dict(item.split("=", 1) for item in rows[0].split()[1:])
    if int(values["seed"]) != seed or int(values["budget"]) != LJ75_BUDGET:
        raise RuntimeError(f"row {rows[0]}")
    return {
        "budget": LJ75_BUDGET,
        "reference": LJ75_REFERENCE,
        "hit": values["hit"] == "1",
        "evals_to_hit": int(values["evals_to_hit"]),
        "charged": int(values["charged"]),
        "wall_s": float(values["wall_s"]),
        "best": float(values["best"]),
        "pot_s": float(values["pot_s"]),
        "outside_s": float(values["outside_s"]),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)


def fmt_p(value: float) -> str:
    return f"{value:.6g}"


def summarise(campaign: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    spec = CAMPAIGNS[campaign]
    control = spec["arms"][0][0]
    out: list[dict[str, str]] = []
    for system in spec["systems"]:
        groups: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            if row["campaign"] == campaign and row["system"] == system:
                groups.setdefault(row["arm"], []).append(row)
        if control not in groups:
            continue
        base = groups[control]
        for arm, *_ in spec["arms"]:
            group = groups.get(arm)
            if not group:
                continue
            hits = [row for row in group if row["hit"] == "1"]
            evals = [float(row["evals_to_hit"]) for row in group]
            walls = [float(row["wall_s"]) for row in group]
            pots = [float(row["pot_s"]) for row in group]
            outs = [float(row["outside_s"]) for row in group]
            tests = {key: "" for key in SUMMARY_FIELDS if key.startswith(("fisher", "mann"))}
            if arm != control:
                base_hits = sum(row["hit"] == "1" for row in base)
                table = [
                    [len(hits), len(group) - len(hits)],
                    [base_hits, len(base) - base_hits],
                ]
                tests["fisher_p"] = fmt_p(fisher_exact(table)[1])
                for key, column in (
                    ("evals", "evals_to_hit"),
                    ("wall", "wall_s"),
                    ("pot", "pot_s"),
                    ("outside", "outside_s"),
                ):
                    left = [float(row[column]) for row in group]
                    right = [float(row[column]) for row in base]
                    if len(set(left + right)) > 1:
                        p = mannwhitneyu(left, right, alternative="two-sided", method="asymptotic", use_continuity=True).pvalue
                        tests[f"mannwhitney_p_{key}"] = fmt_p(p)
                    else:
                        tests[f"mannwhitney_p_{key}"] = "1"
            out.append(
                {
                    "campaign": campaign,
                    "system": system,
                    "arm": arm,
                    "control": control,
                    "n": str(len(group)),
                    "hits": str(len(hits)),
                    "hits_over_n": f"{len(hits)}/{len(group)}",
                    "budget": group[0]["budget"],
                    "evals_median_to_hit": (
                        f"{median([float(row['evals_to_hit']) for row in hits]):.1f}"
                        if hits
                        else ""
                    ),
                    "evals_median": f"{median(evals):.1f}",
                    "wall_median_s": f"{median(walls):.6f}",
                    "pot_median_s": f"{median(pots):.6f}",
                    "outside_median_s": f"{median(outs):.6f}",
                    **tests,
                    "code_revision": " ".join(sorted({row["code_revision"] for row in group})),
                    "host": " ".join(sorted({row["host"] for row in group})),
                    "seeds": " ".join(sorted({row["seed"] for row in group}, key=int)),
                    "schedule_seed": str(SCHEDULE_SEED),
                }
            )
    return out


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12 or abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7")
    for campaign, spec in CAMPAIGNS.items():
        if any(len(arm) != 5 for arm in spec["arms"]):
            raise SystemExit(f"{campaign} arm fields")
        names = tuple(arm for arm, *_ in spec["arms"])
        if any(system not in spec["systems"] for system in spec.get("wide", ())):
            raise SystemExit(f"{campaign} wide systems")
        order = schedule(spec["systems"], spec, names, SCHEDULE_SEED)
        jobs = sum(len(seeds_of(spec, system)) for system in spec["systems"]) * len(names)
        if len(order) != jobs:
            raise SystemExit(f"{campaign} schedule size")
        firsts = [
            system
            for index, (system, _, _) in enumerate(order)
            if index == 0 or order[index - 1][0] != system
        ]
        if firsts != list(spec["systems"]):
            raise SystemExit(f"{campaign} system order {firsts}")
    common = {
        "campaign": "priors-off",
        "system": "pt12cu11",
        "budget": "10",
        "code_revision": "abc",
        "host": "bench",
        "pot_s": "1",
        "outside_s": "1",
    }
    rows = [
        {**common, "arm": "develop", "seed": "1", "hit": "0", "evals_to_hit": "10", "wall_s": "2"},
        {**common, "arm": "develop", "seed": "2", "hit": "1", "evals_to_hit": "6", "wall_s": "1"},
        {**common, "arm": "full", "seed": "1", "hit": "1", "evals_to_hit": "3", "wall_s": "1"},
        {**common, "arm": "full", "seed": "2", "hit": "1", "evals_to_hit": "5", "wall_s": "1"},
    ]
    summary = summarise("priors-off", rows)
    if [row["arm"] for row in summary] != ["develop", "full"]:
        raise SystemExit(f"arms {summary}")
    if summary[1]["hits_over_n"] != "2/2" or summary[1]["evals_median_to_hit"] != "4.0":
        raise SystemExit(f"full {summary[1]}")
    if summary[1]["fisher_p"] != "1" or summary[0]["fisher_p"] != "":
        raise SystemExit(f"fisher {summary}")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=tuple(CAMPAIGNS), required=True)
    parser.add_argument("--develop", type=Path, required=True, help="directory of develop examples")
    parser.add_argument("--branch", type=Path, required=True, help="directory of branch examples")
    parser.add_argument("--develop-revision", required=True)
    parser.add_argument("--branch-revision", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--system", action="append")
    args = parser.parse_args(argv[1:])

    spec = CAMPAIGNS[args.campaign]
    systems = tuple(args.system) if args.system else spec["systems"]
    for system in systems:
        if system not in spec["systems"]:
            raise SystemExit(f"{system} is not in {args.campaign}")
    builds = {
        "develop": (args.develop, args.develop_revision),
        "branch": (args.branch, args.branch_revision),
    }
    arms = {
        name: (build, priors, homotop, extra) for name, build, priors, homotop, extra in spec["arms"]
    }
    host = os.uname().nodename
    cpu = cpu_model()
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    done = {
        (row["system"], row["arm"], row["seed"], row["code_revision"])
        for row in read_rows(args.rows)
        if row["campaign"] == args.campaign
    }
    order = schedule(systems, spec, tuple(arms), SCHEDULE_SEED)
    for index, (system, seed, arm) in enumerate(order, start=1):
        build, priors, homotop, extra = arms[arm]
        directory, revision = builds[build]
        if (system, arm, str(seed), revision) in done:
            continue
        log = args.logs / f"{args.campaign}-{index:03d}-{system}-{arm}-{seed}.log"
        if system == "lj75":
            result = run_lj75(directory / "h2h_seed", seed, log)
        elif system in MORSE:
            result = run_gupta(directory / "h2h_morse", system, seed, priors, None, {}, log)
        else:
            result = run_gupta(directory / "h2h_metal", system, seed, priors, homotop, extra, log)
        row = {
            "campaign": args.campaign,
            "system": system,
            "arm": arm,
            "seed": seed,
            "budget": result["budget"],
            "reference": f"{result['reference']:.8f}",
            "priors": "preset" if system == "lj75" else "off" if priors == "0" else "on",
            "homotop": (
                "absent"
                if build == "develop" or system in MORSE or system == "lj75"
                else ("all" if homotop is None else f"off:{homotop}")
                + "".join(f" {key}={value}" for key, value in sorted(extra.items()))
            ),
            "hit": int(result["hit"]),
            "evals_to_hit": result["evals_to_hit"],
            "charged": result["charged"],
            "best": f"{result['best']:.8f}",
            "wall_s": f"{result['wall_s']:.6f}",
            "pot_s": f"{result['pot_s']:.6f}",
            "outside_s": f"{result['outside_s']:.6f}",
            "code_revision": revision,
            "host": host,
            "cpu": cpu,
        }
        append_row(args.rows, row)
        print(
            f"{index}/{len(order)} {system} {arm} seed {seed} hit {row['hit']} "
            f"evals {row['evals_to_hit']} wall {row['wall_s']} best {row['best']}",
            flush=True,
        )
    current = {args.develop_revision, args.branch_revision}
    rows = [row for row in read_rows(args.rows) if row["code_revision"] in current]
    write_summary(args.summary, summarise(args.campaign, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
