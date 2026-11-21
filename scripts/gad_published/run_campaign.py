#!/usr/bin/env python3
# Copyright (c) anneal developers
"""GAD, anneal and GMIN on LJ31 and LJ38 at twelve seeds.

GAD is rerun by ``gad_rerun.py``: CondEvo's CHARLES with the settings
the Zenodo 10.5281/zenodo.19001618 deposit records, started from the
same seeded cluster as anneal and GMIN. Anneal is
``Config::recommended`` and GMIN is Wales's public binary with the
evaluator cap. All three stop at 400000 potential calls, the LJ38
entry of the elja map, used here for LJ31 as well, and all three count
one call per energy and force evaluation. Seeds are 2000 through 2011,
one process at a time. A hit is the first call within 0.001 of the
Cambridge energy; a miss is stored as the full budget. Medians are
Hyndman-Fan type 7. Fisher is the two-sided exact test on hits against
anneal, and Mann-Whitney is two-sided on calls to the hit against
anneal. The deposited run is kept as one ``gad_deposit`` row with its
generation and evaluation bounds. Summary rows are appended, never
rewritten, with the anneal revision and the host. This process refuses
to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

from read_deposit import HIT_TOL, REFERENCES, ZENODO_DOI, read_zip
from gmin_pin import read_built_revision

BUDGET = 400_000
SEEDS = tuple(range(2000, 2012))
SIZES = (31, 38)
RERUNS = ("anneal", "gmin", "gad")
ARMS = ("gad_deposit",) + RERUNS
CONDEVO_COMMIT = "ae76bf0462de63fa6bf3bde97df7cc00237b8d1d"
CORE = "0"

SUMMARY_FIELDS = [
    "system",
    "arm",
    "provenance",
    "n",
    "hits",
    "budget",
    "seeds",
    "evals_median",
    "evals_iqr",
    "best_median",
    "wall_median_s",
    "pot_median_s",
    "outside_median_s",
    "inside_us",
    "outside_us",
    "fisher_vs",
    "fisher_p",
    "mannwhitney_p",
    "hit_generation",
    "evals_lower",
    "evals_upper",
    "code_revision",
    "host",
    "date",
    "gmin_revision",
    "condevo_commit",
    "zenodo",
]


def type7(ordered: list[float], p: float) -> float:
    n = len(ordered)
    if n < 1:
        raise ValueError("type 7 needs a sample")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    median = type7(ordered, 0.5)
    if len(ordered) == 1:
        return median, 0.0
    return median, type7(ordered, 0.75) - type7(ordered, 0.25)


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def fisher_exact_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided Fisher exact p for hits out of two sample sizes."""
    a = hits_a
    b = n_a - hits_a
    c = hits_b
    d = n_b - hits_b
    if min(a, b, c, d) < 0:
        raise ValueError("hit count exceeds its sample")

    def probability(aa: int, bb: int, cc: int, dd: int) -> float:
        total = aa + bb + cc + dd
        return comb(aa + bb, aa) * comb(cc + dd, cc) / comb(total, aa + cc)

    observed = probability(a, b, c, d)
    total = 0.0
    for aa in range(a + b + 1):
        cc = a + c - aa
        bb = a + b - aa
        dd = c + d - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        mass = probability(aa, bb, cc, dd)
        if mass <= observed + 1e-15:
            total += mass
    return min(1.0, total)


def mann_whitney_two_sided(left: list[float], right: list[float]) -> float:
    """Two-sided Mann-Whitney p. Exact when the pooled sample has no ties."""
    xs = [float(value) for value in left]
    ys = [float(value) for value in right]
    if not xs or not ys:
        raise ValueError("Mann-Whitney needs both samples")
    observed = _u(xs, ys)
    pooled = xs + ys
    if len(set(pooled)) != len(pooled):
        return _mann_whitney_normal(xs, ys, observed)
    return _exact_no_ties(len(xs), len(ys), observed)


def _exact_no_ties(n_x: int, n_y: int, observed: float) -> float:
    """Null tail of the rank-sum statistic. Ranks are 1..n_x+n_y."""
    n = n_x + n_y
    product = n_x * n_y
    smaller = min(observed, product - observed)
    max_s = n_x * (2 * n - n_x + 1) // 2
    ways = [[0] * (max_s + 1) for _ in range(n_x + 1)]
    ways[0][0] = 1
    for rank in range(1, n + 1):
        for k in range(min(n_x, rank), 0, -1):
            previous = ways[k - 1]
            current = ways[k]
            for total in range(max_s, rank - 1, -1):
                current[total] += previous[total - rank]
    counted = sum(ways[n_x])
    expected = math.comb(n, n_x)
    if counted != expected:
        raise RuntimeError(f"rank sums counted {counted}, expected {expected}")
    base = n_x * (n_x + 1) / 2.0
    extreme = 0
    for total, count in enumerate(ways[n_x]):
        if count == 0:
            continue
        statistic = total - base
        if min(statistic, product - statistic) <= smaller + 1e-9:
            extreme += count
    return extreme / expected


def _u(left: list[float], right: list[float]) -> float:
    statistic = 0.0
    for x_value in left:
        for y_value in right:
            if x_value > y_value:
                statistic += 1.0
            elif x_value == y_value:
                statistic += 0.5
    return statistic


def _mann_whitney_normal(left: list[float], right: list[float], observed: float) -> float:
    n_x = len(left)
    n_y = len(right)
    product = n_x * n_y
    mean = product / 2.0
    tie_term = 0.0
    counts: dict[float, int] = {}
    for value in left + right:
        counts[value] = counts.get(value, 0) + 1
    for count in counts.values():
        tie_term += count * (count - 1) * (count + 1)  # count^3 - count
    n = n_x + n_y
    variance = (product / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if variance <= 0.0:
        return 1.0
    z = (abs(observed - mean) - 0.5) / math.sqrt(variance)
    # Two-sided tail of the standard normal, without a statistics package.
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


def per_eval(wall_s: float, pot_s: float, charged: int) -> tuple[float, float]:
    if charged < 1:
        raise RuntimeError("per-evaluation split needs a charged call")
    inside = pot_s / charged * 1e6
    outside = max(wall_s - pot_s, 0.0) / charged * 1e6
    return inside, outside


def gad_rows(zip_path: Path) -> list[dict]:
    rows = []
    for size in read_zip(zip_path):
        for run in size["runs"]:
            reached = bool(run["reached"])
            lower = int(run["evals_lower"])
            upper = int(run["evals_upper"])
            rows.append(
                {
                    "system": run["system"],
                    "arm": "gad_deposit",
                    "provenance": "deposited",
                    "seed": run["run"],
                    "budget": BUDGET,
                    "reached": int(reached),
                    "hit": "",
                    "evals_lower": lower,
                    "evals_upper": upper,
                    "best": float(run["best"]),
                    "wall_s": "",
                    "inside_us": "",
                    "outside_us": "",
                    "samples": run["samples"],
                    "relax_steps": run["relax_steps"],
                    "step_source": run["step_source"],
                    "hit_generation": run["hit_generation"],
                    "generations_charged": run["generations_charged"],
                }
            )
    return rows


def parse_anneal(text: str, code: str = "anneal") -> dict[str, str]:
    lines = text.splitlines()
    data = [line for line in lines if f",{code}," in line]
    if len(data) != 1:
        raise ValueError(f"expected one {code} row, saw {len(data)}")
    header = next(line for line in lines if line.startswith("system,code,"))
    row = next(csv.DictReader([header, data[0]]))
    if row["hit"] not in ("0", "1"):
        raise ValueError(f"{code} hit {row['hit']}")
    return row


def parse_gmin(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in text.split()[1:]:
        key, value = part.split("=", 1)
        fields[key] = value
    if "hit" not in fields or "evals_to_hit" not in fields:
        raise ValueError(f"no gmin row in {text!r}")
    return fields


def rerun_record(n: int, arm: str, row: dict[str, str]) -> dict:
    hit = row["hit"] in ("1", "true")
    charged = int(float(row["charged"]))
    wall_s = float(row["wall_s"])
    pot_s = float(row["pot_s"])
    inside, outside = per_eval(wall_s, pot_s, charged)
    if int(row["budget"]) != BUDGET:
        raise RuntimeError(f"{arm} budget {row['budget']}")
    return {
        "system": f"lj{n}",
        "arm": arm,
        "provenance": "rerun",
        "seed": str(int(row["seed"])),
        "budget": BUDGET,
        "reached": int(hit),
        "hit": int(hit),
        "hit_generation": "",
        "evals_lower": "",
        "evals_upper": "",
        "evals": float(row["evals_to_hit"]),
        "best": float(row["best"]),
        "wall_s": wall_s,
        "pot_s": pot_s,
        "outside_s": wall_s - pot_s,
        "inside_us": inside,
        "outside_us": outside,
    }


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(BUDGET)
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["TQDM_DISABLE"] = "1"
    return env


def run_anneal(binary: Path, n: int, seed: int, log: Path) -> dict[str, str]:
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", CORE, str(binary), "run", f"lj{n}", str(seed)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=_env(),
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal LJ{n} seed {seed} exited {proc.returncode}")
    return parse_anneal(text)


def run_gmin(
    gmin: Path,
    data: Path,
    binary: Path,
    n: int,
    seed: int,
    work: Path,
    log: Path,
) -> dict[str, str]:
    work.mkdir(parents=True, exist_ok=True)
    coords = work / "coords"
    subprocess.run(
        [str(binary), "coords", f"lj{n}", str(seed), str(coords)],
        check=True,
        env=_env(),
    )
    (work / "data").write_text(data.read_text() + f"RANSEED {seed}\n")
    env = _env()
    env.update(
        {
            "H2H_SYSTEM": f"lj{n}",
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{REFERENCES[n]:.10f}",
            "H2H_TOL": f"{HIT_TOL}",
            "H2H_ROW": str(work / "row.txt"),
        }
    )
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", CORE, str(gmin)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=work,
        )
    row_path = work / "row.txt"
    if not row_path.is_file():
        raise RuntimeError(f"gmin LJ{n} seed {seed} exited {proc.returncode} with no row")
    return parse_gmin(row_path.read_text())


def run_gad(
    python: Path,
    script: Path,
    condevo: Path,
    binary: Path,
    n: int,
    seed: int,
    work: Path,
    log: Path,
) -> dict[str, str]:
    work.mkdir(parents=True, exist_ok=True)
    founder = work / "founder"
    subprocess.run(
        [str(binary), "coords", f"lj{n}", str(seed), str(founder)],
        check=True,
        env=_env(),
    )
    env = _env()
    env["PYTHONPATH"] = str(condevo)
    with log.open("w") as handle:
        proc = subprocess.run(
            [
                "taskset", "-c", CORE, str(python), "-u", str(script),
                f"lj{n}", str(seed), "--founder", str(founder), "--budget", str(BUDGET),
            ],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"gad LJ{n} seed {seed} exited {proc.returncode}")
    return parse_anneal(text, "gad")


def summarise(rows: list[dict], head: str, revision: str, host: str) -> list[dict[str, str]]:
    by_system: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        by_system.setdefault(row["system"], {}).setdefault(row["arm"], []).append(row)
    date = datetime.date.today().isoformat()
    out: list[dict[str, str]] = []
    for n in SIZES:
        system = f"lj{n}"
        arms = by_system.get(system, {})
        anneal = arms.get("anneal", [])
        for arm in ARMS:
            group = arms.get(arm, [])
            if not group:
                continue
            row = {key: "" for key in SUMMARY_FIELDS}
            row.update(
                system=system,
                arm=arm,
                provenance=group[0]["provenance"],
                n=str(len(group)),
                budget=str(BUDGET),
                best_median=f"{median_iqr([item['best'] for item in group])[0]:.8f}",
                code_revision=head,
                host=host,
                date=date,
                gmin_revision=revision,
                condevo_commit=CONDEVO_COMMIT,
                zenodo=ZENODO_DOI,
            )
            if arm == "gad_deposit":
                row.update(
                    seeds=" ".join(str(item["seed"]) for item in group),
                    hit_generation=str(group[0]["hit_generation"]),
                    evals_lower=f"{float(group[0]['evals_lower']):.0f}",
                    evals_upper=f"{float(group[0]['evals_upper']):.0f}",
                )
                out.append(row)
                continue
            seeds = sorted(int(item["seed"]) for item in group)
            evals_median, evals_iqr = median_iqr([item["evals"] for item in group])
            row.update(
                hits=str(sum(item["hit"] for item in group)),
                seeds=f"{seeds[0]}-{seeds[-1]}",
                evals_median=f"{evals_median:.1f}",
                evals_iqr=f"{evals_iqr:.1f}",
                wall_median_s=f"{median_iqr([item['wall_s'] for item in group])[0]:.6f}",
                pot_median_s=f"{median_iqr([item['pot_s'] for item in group])[0]:.6f}",
                outside_median_s=f"{median_iqr([item['outside_s'] for item in group])[0]:.6f}",
                inside_us=f"{median_iqr([item['inside_us'] for item in group])[0]:.4f}",
                outside_us=f"{median_iqr([item['outside_us'] for item in group])[0]:.4f}",
            )
            if arm != "anneal" and anneal:
                hits = sum(item["hit"] for item in group)
                anneal_hits = sum(item["hit"] for item in anneal)
                fisher = fisher_exact_two_sided(hits, len(group), anneal_hits, len(anneal))
                mann = mann_whitney_two_sided(
                    [item["evals"] for item in group], [item["evals"] for item in anneal]
                )
                row.update(fisher_vs="anneal", fisher_p=f"{fisher:.6g}", mannwhitney_p=f"{mann:.6g}")
            out.append(row)
    return out


def append_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle))
        if header != SUMMARY_FIELDS:
            raise SystemExit(f"{path} has another header; rows are appended, never rewritten")
        new = False
    else:
        new = True
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def load_source(path: Path) -> str:
    revision = ""
    for line in path.read_text().splitlines():
        if line.startswith("revision="):
            revision = line.split("=", 1)[1].strip()
    if not revision:
        raise SystemExit(f"{path} has no revision")
    return revision


def saved_rerun(path: Path, n: int, arm: str) -> dict | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    text = path.read_text()
    row = parse_gmin(text) if arm == "gmin" else parse_anneal(text, arm)
    return rerun_record(n, arm, row)


def note(progress: Path, line: str) -> None:
    print(line, flush=True)
    with progress.open("a") as handle:
        handle.write(line + "\n")


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7 median")
    if abs(median_iqr(sample)[1] - 2.5) > 1e-12:
        raise SystemExit("iqr")
    # 12 hits of 12 against 0 hits of 1. The only tables with those margins
    # are the observed one and the swap of the single miss.
    extreme = fisher_exact_two_sided(12, 12, 0, 1)
    if abs(extreme - (1.0 / 13.0)) > 1e-12:
        raise SystemExit(f"fisher {extreme}")
    certain = fisher_exact_two_sided(12, 12, 1, 1)
    if abs(certain - 1.0) > 1e-12:
        raise SystemExit(f"fisher certain {certain}")
    # One observation above twelve others. Two-sided exact p is 2/13.
    ranked = mann_whitney_two_sided([100.0], [float(value) for value in range(12)])
    if abs(ranked - (2.0 / 13.0)) > 1e-12:
        raise SystemExit(f"mann {ranked}")
    tied = mann_whitney_two_sided([1.0, 1.0], [1.0, 1.0])
    if not 0.0 <= tied <= 1.0:
        raise SystemExit(f"tied mann {tied}")
    if BUDGET != 400_000 or REFERENCES[38] != -173.928427 or REFERENCES[31] != -133.586422:
        raise SystemExit("references")
    if abs(HIT_TOL - 0.001) > 1e-15:
        raise SystemExit("tolerance")
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as scratch:
        path = Path(scratch) / "rows.csv"
        row = {key: "x" for key in SUMMARY_FIELDS}
        append_summary(path, [row])
        append_summary(path, [row])
        if len(path.read_text().splitlines()) != 3:
            raise SystemExit("append")
        path.write_text("system,arm\n")
        try:
            append_summary(path, [row])
        except SystemExit:
            pass
        else:
            raise SystemExit("header guard")
    gad = parse_anneal(
        "noise\nsystem,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,generations\n"
        "lj38,gad,2000,400000,0,400000,400000,200.7,-170.98986,30.2,170.5,19\n",
        "gad",
    )
    if rerun_record(38, "gad", gad)["outside_s"] != 200.7 - 30.2:
        raise SystemExit("gad row")
    print("ok")


def jobs(sizes: tuple[int, ...], seeds: tuple[int, ...], arms: tuple[str, ...]) -> list[tuple]:
    order = []
    for n in sizes:
        if "gad_deposit" in arms:
            order.append((n, "gad_deposit", None))
        for seed in seeds:
            for arm in RERUNS:
                if arm in arms:
                    order.append((n, arm, seed))
    return order


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--svnrev", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--size", type=int, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", action="append")
    parser.add_argument("--core", default="0")
    parser.add_argument("--gad-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gad-script", type=Path, default=Path(__file__).with_name("gad_rerun.py"))
    parser.add_argument("--condevo", type=Path)
    args = parser.parse_args(argv[1:])
    sizes = tuple(args.size) if args.size else SIZES
    seeds = tuple(args.seed) if args.seed else SEEDS
    arms = tuple(args.arm) if args.arm else ARMS
    for n in sizes:
        if n not in REFERENCES:
            raise SystemExit(f"no Cambridge energy for LJ{n}")
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}")
    global CORE
    CORE = str(args.core)
    if "gad" in arms and args.condevo is None:
        raise SystemExit("the gad arm needs --condevo")
    head = args.head_file.read_text().strip()
    host = socket.gethostname()
    revision = read_built_revision(args.svnrev, args.source)
    progress = args.runs.parent / "progress.txt"
    args.runs.mkdir(parents=True, exist_ok=True)
    deposited = gad_rows(args.zip)
    for row in deposited:
        if row["step_source"] != "relaxer cap" or row["relax_steps"] != 1000 or row["samples"] != 64:
            raise SystemExit(
                f"LJ deposit {row['system']} {row['run']} is not 64 samples at the recorded cap 1000 "
                f"({row['step_source']}, steps={row['relax_steps']}, samples={row['samples']})"
            )
        charged = int(row["generations_charged"])
        if row["evals_lower"] != charged * 64 or row["evals_upper"] != charged * 64 * 1000:
            raise SystemExit(f"LJ deposit bounds {row}")
    rows: list[dict] = []
    order = jobs(sizes, seeds, arms)
    note(progress, "schedule " + " ".join(
        f"lj{n}:{arm}" if seed is None else f"lj{n}:{arm}:{seed}" for n, arm, seed in order
    ))
    for n, arm, seed in order:
        if arm == "gad_deposit":
            matched = [row for row in deposited if row["system"] == f"lj{n}"]
            if not matched:
                raise SystemExit(f"no deposited LJ{n} run")
            rows.extend(matched)
            for row in matched:
                note(
                    progress,
                    f"deposited n={n} run={row['seed']} reached={row['reached']} "
                    f"generation={row['hit_generation']} lower={row['evals_lower']} "
                    f"upper={row['evals_upper']} best={row['best']:.8f}",
                )
            continue
        saved = args.runs / f"lj{n}" / f"{arm}-{seed}.row"
        log = args.runs / f"lj{n}" / f"{arm}-{seed}.log"
        saved.parent.mkdir(parents=True, exist_ok=True)
        existing = saved_rerun(saved, n, arm)
        if existing is not None:
            rows.append(existing)
            note(progress, f"skip n={n} arm={arm} seed={seed} hit={existing['hit']} evals={existing['evals']:.0f}")
            continue
        note(progress, f"start n={n} arm={arm} seed={seed} budget={BUDGET}")
        if arm == "anneal":
            raw = run_anneal(args.binary, n, seed, log)
            saved.write_text(log.read_text())
        elif arm == "gad":
            raw = run_gad(
                args.gad_python,
                args.gad_script,
                args.condevo,
                args.binary,
                n,
                seed,
                args.runs / f"lj{n}" / f"gad-{seed}",
                log,
            )
            saved.write_text(log.read_text())
        else:
            raw = run_gmin(
                args.gmin,
                args.data,
                args.binary,
                n,
                seed,
                args.runs / f"lj{n}" / f"gmin-{seed}",
                log,
            )
            saved.write_text((args.runs / f"lj{n}" / f"gmin-{seed}" / "row.txt").read_text())
        result = rerun_record(n, arm, raw)
        rows.append(result)
        note(
            progress,
            f"done n={n} arm={arm} seed={seed} hit={result['hit']} "
            f"evals={result['evals']:.0f} wall_s={result['wall_s']:.3f} best={result['best']:.6f}",
        )
    text_rows = summarise(rows, head, revision, host)
    if args.summary:
        append_summary(args.summary, text_rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(text_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
