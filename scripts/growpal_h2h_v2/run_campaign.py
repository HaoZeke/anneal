#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial GrowPAL against the recommended preset at the paper call budgets.

Systems are LJ38, LJ55, LJ75, LJ98, and the Cambridge Morse M38 at
rho = 6. Gupta Au55 is not run: neither GrowPAL nor aegon has a Gupta
potential, and ASE has no Gupta calculator to route through. Six seeds,
the same list on both arms, one process pinned to core 0. Systems run in
the listed order; inside a system the (seed, arm) pairs are shuffled.

A hit is a quenched energy within 0.001 of the Cambridge minimum. A miss
is stored as the full budget. Medians and interquartile ranges are
Hyndman-Fan type 7. ``wall_s`` is the search clock of each code, after
imports and JIT warmup. ``pot_s`` is the part of it inside the potential
and ``outside_s`` the rest. ``process_s`` is the child process from
spawn to exit. The ratio is the GrowPAL median wall over the anneal
median wall.

Rows are append-only. Each seed is written once to its own row file and
appended once to ``seeds.csv``. A summary row is appended to
``results.csv`` when its system has every planned seed on both arms, and
only if that (system, arm, campaign) key is not already there. This
process refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# (system, binary, budget, Cambridge minimum, GrowPAL length per anneal length)
SYSTEMS = (
    ("lj38", "h2h_gmin_proof", 400_000, -173.928427, 3.0 / 2.0 ** (1.0 / 6.0)),
    ("lj55", "h2h_gmin_proof", 1_000_000, -279.248470, 3.0 / 2.0 ** (1.0 / 6.0)),
    ("lj75", "h2h_gmin_proof", 4_000_000, -397.492331, 3.0 / 2.0 ** (1.0 / 6.0)),
    ("lj98", "h2h_gmin_proof", 4_000_000, -543.665361, 3.0 / 2.0 ** (1.0 / 6.0)),
    ("m38r6", "h2h_morse", 400_000, -157.477108, 3.0),
)
SIZES = {"lj38": 38, "lj55": 55, "lj75": 75, "lj98": 98, "m38r6": 38}
PAPER_BUDGETS = {name: budget for name, _bin, budget, _ref, _scale in SYSTEMS}
REFERENCES = {name: ref for name, _bin, _budget, ref, _scale in SYSTEMS}
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = ("anneal", "growpal")
SCHEDULE_SEED = 20261113
HIT_TOL = 0.001
IDENTITY_TOL = 1e-9
# The uncapped diagnostic is not the head-to-head. It asks how many calls
# the deterministic GrowPAL loop needs when nothing stops it, one run per
# system, under this safety ceiling.
UNCAPPED_CEILING = 500_000_000

SEED_FIELDS = [
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "inside_us",
    "outside_us",
    "process_s",
    "best",
    "stop",
    "reached",
]

SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "planned",
    "hits",
    "hits_over_n",
    "budget",
    "reference",
    "evals_median",
    "evals_iqr",
    "hit_evals_median",
    "wall_median_s",
    "wall_iqr_s",
    "pot_median_s",
    "outside_median_s",
    "inside_us",
    "outside_us",
    "process_median_s",
    "wall_ratio",
    "reached_median",
    "stops",
    "seeds",
    "schedule_seed",
    "growpal_version",
    "growpal_sdist_sha256",
    "growpal_commit",
    "aegon_version",
    "aegon_sdist_sha256",
    "ase_version",
    "anneal_head",
    "cpu",
]

IDENTITY_FIELDS = [
    "system",
    "seed",
    "scale",
    "anneal",
    "growpal",
    "exact",
    "growpal_minus_exact",
    "growpal_cutoff",
    "rmax",
]


def type7(sorted_values: list[float], p: float) -> float:
    """Hyndman-Fan type 7. ``sorted_values`` is ascending."""
    n = len(sorted_values)
    if n < 1:
        raise ValueError("type 7 needs a sample")
    if n == 1:
        return sorted_values[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return type7(ordered, 0.5), type7(ordered, 0.75) - type7(ordered, 0.25)


def schedule(systems, seeds, arms, rng_seed: int) -> list[tuple[str, int, str]]:
    """Systems in the given order. Pairs inside a system are shuffled."""
    rng = random.Random(rng_seed)
    order: list[tuple[str, int, str]] = []
    for system in systems:
        jobs = [(system, seed, arm) for seed in seeds for arm in arms]
        rng.shuffle(jobs)
        order.extend(jobs)
    return order


def per_eval(wall_s: float, pot_s: float, charged: int) -> tuple[float, float]:
    """Microseconds per charged evaluation inside and outside the potential."""
    if charged <= 0:
        raise ValueError("per-evaluation time needs a charged evaluation")
    return pot_s / charged * 1e6, (wall_s - pot_s) / charged * 1e6


def load_source(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    for key in (
        "growpal_version",
        "growpal_sdist_sha256",
        "growpal_commit",
        "aegon_version",
        "aegon_sdist_sha256",
        "ase_version",
    ):
        if not fields.get(key):
            raise SystemExit(f"{path} has no {key}")
    return fields


def child_env(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(budget)
    for name in (
        "OMP_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[name] = "1"
    return env


def binary_budget(binary: Path, system: str) -> tuple[int, float]:
    proc = subprocess.run(
        [str(binary), "budget", system], check=True, capture_output=True, text=True
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"budget for {system} returned {proc.stdout!r}")
    return int(lines[0]), float(lines[1])


def parse_anneal(text: str) -> dict[str, str]:
    lines = text.splitlines()
    header = next((line for line in lines if line.startswith("system,code,")), None)
    data = [line for line in lines if ",anneal," in line]
    if header is None or len(data) != 1:
        raise ValueError(f"expected one anneal row, saw {len(data)}")
    row = next(csv.DictReader([header, data[0]]))
    if row["hit"] not in ("0", "1", "true", "false"):
        raise ValueError(f"anneal hit {row['hit']}")
    hit = "1" if row["hit"] in ("1", "true") else "0"
    return {
        "system": row["system"],
        "arm": "anneal",
        "seed": row["seed"],
        "budget": row["budget"],
        "hit": hit,
        "evals_to_hit": row["evals_to_hit"],
        "charged": row["charged"],
        "wall_s": row["wall_s"],
        "pot_s": row["pot_s"],
        "best": row["best"],
        "stop": "hit" if hit == "1" else "budget",
        "reached": str(SIZES[row["system"]]),
    }


def finish(row: dict[str, str], process_s: float) -> dict[str, str]:
    wall, pot, charged = float(row["wall_s"]), float(row["pot_s"]), int(row["charged"])
    inside, outside = per_eval(wall, pot, charged)
    out = dict(row)
    out["outside_s"] = f"{wall - pot:.6f}"
    out["inside_us"] = f"{inside:.4f}"
    out["outside_us"] = f"{outside:.4f}"
    out["process_s"] = f"{process_s:.6f}"
    return out


def timed(cmd: list[str], env: dict[str, str], log: Path) -> tuple[int, str, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while log.exists():
        log = log.with_name(f"{log.name.split('.log')[0]}.log.{attempt}")
        attempt += 1
    with log.open("x") as handle:
        mark = time.perf_counter()
        proc = subprocess.run(cmd, check=False, stdout=handle, stderr=subprocess.STDOUT, env=env)
        elapsed = time.perf_counter() - mark
    return proc.returncode, log.read_text(), elapsed


def run_anneal(binary: Path, system: str, seed: int, budget: int, log: Path) -> dict[str, str]:
    cmd = ["taskset", "-c", "0", str(binary), "run", system, str(seed)]
    code, text, elapsed = timed(cmd, child_env(budget), log)
    if code != 0:
        raise RuntimeError(f"anneal {system} seed {seed} exited {code}\n{text[-500:]}")
    row = parse_anneal(text)
    if int(row["budget"]) != budget or int(row["charged"]) > budget:
        raise RuntimeError(f"anneal budget {row['budget']} charged {row['charged']}")
    return finish(row, elapsed)


def run_growpal(
    python: Path, arm: Path, system: str, seed: int, budget: int, seeds_xyz: Path, log: Path
) -> dict[str, str]:
    cmd = [
        "taskset", "-c", "0", str(python), "-u", "-W", "ignore", str(arm), "run",
        system, str(seed), str(budget), repr(REFERENCES[system]), str(seeds_xyz),
    ]
    code, text, elapsed = timed(cmd, child_env(budget), log)
    if code != 0:
        raise RuntimeError(f"growpal {system} seed {seed} exited {code}\n{text[-800:]}")
    rows = [line[4:] for line in text.splitlines() if line.startswith("ROW ")]
    if len(rows) != 1:
        raise RuntimeError(f"growpal {system} seed {seed} printed {len(rows)} rows")
    row = json.loads(rows[0])
    if int(row["charged"]) > budget:
        raise RuntimeError(f"growpal charged {row['charged']} past {budget}")
    return finish(row, elapsed)


def write_once(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: row[key] for key in fields})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Append under the existing header. Never rewrites what is there."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle))
        if header != fields:
            raise RuntimeError(f"{path} header {header} != {fields}")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fresh:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def summarise(
    system: str,
    rows: list[dict[str, str]],
    seeds: tuple[int, ...],
    source: dict[str, str],
    head: str,
    cpu: str,
) -> list[dict[str, str]]:
    arms = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    walls = {arm: median_iqr([float(r["wall_s"]) for r in group])[0] for arm, group in arms.items()}
    ratio = f"{walls['growpal'] / walls['anneal']:.6f}" if walls["anneal"] > 0.0 else ""
    out = []
    for arm in ARMS:
        group = arms[arm]

        def med(key: str) -> float:
            return median_iqr([float(r[key]) for r in group])[0]

        evals_median, evals_iqr = median_iqr([float(r["evals_to_hit"]) for r in group])
        wall_median, wall_iqr = median_iqr([float(r["wall_s"]) for r in group])
        hit_evals = [float(r["evals_to_hit"]) for r in group if r["hit"] == "1"]
        hits = len(hit_evals)
        stops: dict[str, int] = {}
        for r in group:
            stops[r["stop"]] = stops.get(r["stop"], 0) + 1
        out.append(
            {
                "system": system,
                "arm": arm,
                "n": str(len(group)),
                "planned": str(len(seeds)),
                "hits": str(hits),
                "hits_over_n": f"{hits}/{len(group)}",
                "budget": str(PAPER_BUDGETS[system]),
                "reference": f"{REFERENCES[system]:.6f}",
                "evals_median": f"{evals_median:.1f}",
                "evals_iqr": f"{evals_iqr:.1f}",
                "hit_evals_median": f"{median_iqr(hit_evals)[0]:.1f}" if hit_evals else "",
                "wall_median_s": f"{wall_median:.6f}",
                "wall_iqr_s": f"{wall_iqr:.6f}",
                "pot_median_s": f"{med('pot_s'):.6f}",
                "outside_median_s": f"{med('outside_s'):.6f}",
                "inside_us": f"{med('inside_us'):.4f}",
                "outside_us": f"{med('outside_us'):.4f}",
                "process_median_s": f"{med('process_s'):.6f}",
                "wall_ratio": ratio,
                "reached_median": f"{med('reached'):.1f}",
                "stops": " ".join(f"{key}:{stops[key]}" for key in sorted(stops)),
                "seeds": " ".join(str(seed) for seed in seeds),
                "schedule_seed": str(SCHEDULE_SEED),
                "growpal_version": source["growpal_version"],
                "growpal_sdist_sha256": source["growpal_sdist_sha256"],
                "growpal_commit": source["growpal_commit"],
                "aegon_version": source["aegon_version"],
                "aegon_sdist_sha256": source["aegon_sdist_sha256"],
                "ase_version": source["ase_version"],
                "anneal_head": head,
                "cpu": cpu,
            }
        )
    return out


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


def identity_rows(binaries: dict[str, Path], python: Path, arm: Path, work: Path) -> list[dict]:
    """Both codes' energy of one anneal start, before any timed run."""
    out = []
    for system, binary_name, _budget, _ref, scale in SYSTEMS:
        coords = work / "identity" / f"{system}-2000.xyz"
        coords.parent.mkdir(parents=True, exist_ok=True)
        binary = binaries[binary_name]
        subprocess.run([str(binary), "coords", system, "2000", str(coords)], check=True)
        theirs = subprocess.run(
            [str(python), "-W", "ignore", str(arm), "identity", system, str(coords), repr(scale)],
            check=True, capture_output=True, text=True,
        )
        result = json.loads(theirs.stdout.strip().splitlines()[-1])
        exact = float(result["exact"])
        growpal = float(result["growpal"])
        same_form = float(result["same_form"])
        anneal_text = ""
        if binary_name == "h2h_gmin_proof":
            ours = subprocess.run(
                [str(binary), "energy", system, str(coords)],
                check=True, capture_output=True, text=True,
            )
            anneal = float(ours.stdout.strip().splitlines()[-1])
            anneal_text = f"{anneal:.10f}"
            if abs(anneal - exact) > IDENTITY_TOL * max(1.0, abs(exact)):
                raise SystemExit(f"{system} anneal {anneal} != full sum {exact}")
        if abs(growpal - same_form) > IDENTITY_TOL * max(1.0, abs(same_form)):
            raise SystemExit(f"{system} GrowPAL {growpal} != its own form {same_form}")
        out.append(
            {
                "system": system,
                "seed": "2000",
                "scale": f"{scale:.12f}",
                "anneal": anneal_text,
                "growpal": f"{growpal:.10f}",
                "exact": f"{exact:.10f}",
                "growpal_minus_exact": f"{growpal - exact:.3e}",
                "growpal_cutoff": f"{float(result['cutoff']):.6f}",
                "rmax": f"{float(result['rmax']):.6f}",
            }
        )
    return out


def note(progress: Path, line: str) -> None:
    print(line, flush=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a") as handle:
        handle.write(line + "\n")


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12 or abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7")
    if PAPER_BUDGETS != {
        "lj38": 400_000,
        "lj55": 1_000_000,
        "lj75": 4_000_000,
        "lj98": 4_000_000,
        "m38r6": 400_000,
    }:
        raise SystemExit("paper budgets")
    inside, outside = per_eval(1.0, 0.25, 1000)
    if abs(inside - 250.0) > 1e-9 or abs(outside - 750.0) > 1e-9:
        raise SystemExit("per-evaluation split")
    names = [name for name, *_ in SYSTEMS]
    order = schedule(names, SEEDS, ARMS, SCHEDULE_SEED)
    seen: list[str] = []
    for system, _seed, _arm in order:
        if not seen or seen[-1] != system:
            seen.append(system)
    if seen != names:
        raise SystemExit(f"system order {seen}")
    for system in names:
        jobs = sorted((seed, arm) for name, seed, arm in order if name == system)
        if jobs != sorted((seed, arm) for seed in SEEDS for arm in ARMS):
            raise SystemExit(f"{system} is not the shared seed list")
    row = parse_anneal(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s\n"
        "m38r6,anneal,2000,400000,true,5,7,0.5,-157.47,0.2,0.3\n"
    )
    if row["hit"] != "1" or row["reached"] != "38":
        raise SystemExit(f"anneal Morse row {row}")
    print("ok")


UNCAPPED_FIELDS = SEED_FIELDS + [
    "growpal_version",
    "growpal_commit",
    "aegon_version",
    "ase_version",
    "anneal_head",
    "cpu",
]


def uncapped(args) -> int:
    source = load_source(args.source)
    head = args.head_file.read_text().strip()
    cpu = cpu_model()
    arm_script = Path(__file__).resolve().parent / "growpal_arm.py"
    names = [name for name, *_ in SYSTEMS]
    systems = tuple(args.system) if args.system else tuple(names)
    seed = (args.seed or [SEEDS[0]])[0]
    progress = args.runs / "progress.txt"
    path = args.out / "uncapped.csv"
    present = {(r["system"], r["seed"], r["anneal_head"]) for r in read_rows(path)}
    for system in systems:
        if (system, str(seed), head) in present:
            note(progress, f"skip uncapped {system} {seed}")
            continue
        note(progress, f"start uncapped {system} {seed} ceiling={UNCAPPED_CEILING}")
        log = args.runs / "uncapped" / system / f"growpal-{seed}.log"
        row = run_growpal(
            args.python, arm_script, system, seed, UNCAPPED_CEILING, args.seeds_xyz, log
        )
        row.update(
            {
                "growpal_version": source["growpal_version"],
                "growpal_commit": source["growpal_commit"],
                "aegon_version": source["aegon_version"],
                "ase_version": source["ase_version"],
                "anneal_head": head,
                "cpu": cpu,
            }
        )
        write_once(args.out / "rows" / system / f"growpal-uncapped-{seed}.row", UNCAPPED_FIELDS, row)
        append_rows(path, UNCAPPED_FIELDS, [row])
        note(
            progress,
            f"done uncapped {system} {seed} hit={row['hit']} evals={row['evals_to_hit']} "
            f"charged={row['charged']} wall_s={row['wall_s']} stop={row['stop']} "
            f"reached={row['reached']} best={row['best']}",
        )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seeds-xyz", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", choices=("anneal", "growpal", "both"), default="both")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--uncapped", action="store_true")
    args = parser.parse_args(argv[1:])
    if args.uncapped:
        return uncapped(args)
    names = [name for name, *_ in SYSTEMS]
    systems = tuple(args.system) if args.system else tuple(names)
    seeds = tuple(args.seed) if args.seed else SEEDS
    arms = ARMS if args.arm == "both" else (args.arm,)
    for system in systems:
        if system not in PAPER_BUDGETS:
            raise SystemExit(f"no paper budget for {system}")
        if args.budget is not None and args.budget != PAPER_BUDGETS[system]:
            raise SystemExit(f"{system} budget {args.budget} != {PAPER_BUDGETS[system]}")
    source = load_source(args.source)
    head = args.head_file.read_text().strip()
    cpu = cpu_model()
    arm_script = Path(__file__).resolve().parent / "growpal_arm.py"
    binaries = {
        "h2h_gmin_proof": args.bin_dir / "h2h_gmin_proof",
        "h2h_morse": args.bin_dir / "h2h_morse",
    }
    binary_of = {name: binaries[binary] for name, binary, *_ in SYSTEMS}
    progress = args.runs / "progress.txt"
    subprocess.run([str(args.python), "-W", "ignore", str(arm_script), "check"], check=True)
    identity_path = args.out / "identity.csv"
    if not identity_path.exists():
        for row in identity_rows(binaries, args.python, arm_script, args.runs):
            append_rows(identity_path, IDENTITY_FIELDS, [row])
    for system in systems:
        calls, reference = binary_budget(binary_of[system], system)
        if calls != PAPER_BUDGETS[system] or abs(reference - REFERENCES[system]) > 1e-6:
            raise SystemExit(f"{system} binary has {calls} calls at {reference}")
    order = schedule(systems, seeds, arms, SCHEDULE_SEED)
    note(progress, f"head {head} cpu {cpu}")
    note(progress, "schedule " + " ".join(f"{s}:{a}:{d}" for s, d, a in order))
    seeds_csv = args.out / "seeds.csv"
    results_csv = args.out / "results.csv"
    logged = {(r["system"], r["arm"], r["seed"]) for r in read_rows(seeds_csv)}
    done: dict[str, list[dict[str, str]]] = {}
    for system, seed, arm in order:
        budget = PAPER_BUDGETS[system]
        saved = args.out / "rows" / system / f"{arm}-{seed}.row"
        existing = read_rows(saved)
        if existing:
            row = existing[0]
            note(progress, f"skip {system} {arm} {seed} hit={row['hit']} evals={row['evals_to_hit']}")
        else:
            note(progress, f"start {system} {arm} {seed} budget={budget}")
            log = args.runs / system / f"{arm}-{seed}.log"
            if arm == "anneal":
                row = run_anneal(binary_of[system], system, seed, budget, log)
            else:
                row = run_growpal(args.python, arm_script, system, seed, budget, args.seeds_xyz, log)
            write_once(saved, SEED_FIELDS, row)
            note(
                progress,
                f"done {system} {arm} {seed} hit={row['hit']} evals={row['evals_to_hit']} "
                f"charged={row['charged']} wall_s={row['wall_s']} pot_s={row['pot_s']} "
                f"process_s={row['process_s']} best={row['best']} stop={row['stop']} "
                f"reached={row['reached']}",
            )
        key = (system, arm, str(seed))
        if key not in logged:
            append_rows(seeds_csv, SEED_FIELDS, [row])
            logged.add(key)
        done.setdefault(system, []).append(row)
    present = {
        (r["system"], r["arm"], r["schedule_seed"], r["anneal_head"], r["growpal_version"])
        for r in read_rows(results_csv)
    }
    written = []
    for system in systems:
        group = done.get(system, [])
        complete = all(
            any(r["arm"] == arm and r["seed"] == str(seed) for r in group)
            for seed in SEEDS
            for arm in ARMS
        )
        if not complete or seeds != SEEDS:
            continue
        for row in summarise(system, group, SEEDS, source, head, cpu):
            key = (system, row["arm"], str(SCHEDULE_SEED), head, source["growpal_version"])
            if key in present:
                continue
            append_rows(results_csv, SUMMARY_FIELDS, [row])
            present.add(key)
            written.append(row)
    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
