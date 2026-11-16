# Copyright (c) anneal developers
"""Same-session gate of ``plateau_heat`` against its base revision.

Two builds made in the same session: ``base`` at the revision the branch
starts from and ``heat`` at the revision that adds the flag, run with
``H2H_PLATEAU_HEAT=1``. Au55 runs seeds 2000-2023; LJ38, LJ55, LJ75, LJ98,
LJ104, LJ150, Morse M80 at rho 6 and 14 and Sutton-Chen sc108-55 run seeds
2000-2011, all at their example budgets. For each system and seed the base
run goes first and the heat run second, one process on ``taskset -c 0``,
one thread. Rows are appended with the code revision and the host after
every run and are never rewritten; a row already stored for the same
system, arm, seed and revision is kept and its run skipped.

``--summarize`` writes, per system, the hits of each arm, Fisher's exact
test on hits, the tie-corrected Mann-Whitney test (normal approximation,
continuity correction) on calls to the minimum with misses entered at the
budget, and the median wall, potential and outside-potential times.

This process refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
from pathlib import Path

SEEDS_AU55 = tuple(range(2000, 2024))
SEEDS = tuple(range(2000, 2012))
BUDGETS = {
    "au55": 1_000_000,
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj104": 4_000_000,
    "lj150": 4_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
    "sc108-55": 1_000_000,
}
# System, binary. Run in this order.
RUNS = (
    ("au55", "h2h_metal"),
    ("lj38", "h2h_gmin_proof"),
    ("lj55", "h2h_gmin_proof"),
    ("lj75", "h2h_gmin_proof"),
    ("lj98", "h2h_gmin_proof"),
    ("lj104", "h2h_gmin_proof"),
    ("lj150", "h2h_gmin_proof"),
    ("m80r6", "h2h_morse"),
    ("m80r14", "h2h_morse"),
    ("sc108-55", "h2h_sc"),
)
ARMS = ("base", "heat")

ROW_FIELDS = [
    "system",
    "driver",
    "arm",
    "code",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
    "code_revision",
    "host",
    "cpu",
]

SUMMARY_FIELDS = [
    "system",
    "driver",
    "budget",
    "n",
    "base_hits",
    "heat_hits",
    "fisher_p",
    "base_evals_median",
    "heat_evals_median",
    "mannwhitney_u",
    "mannwhitney_p",
    "base_wall_median_s",
    "heat_wall_median_s",
    "base_pot_median_s",
    "heat_pot_median_s",
    "base_outside_median_s",
    "heat_outside_median_s",
    "same_seeds",
    "base_revision",
    "heat_revision",
    "host",
    "seeds",
]


def seeds_of(system: str) -> tuple[int, ...]:
    return SEEDS_AU55 if system == "au55" else SEEDS


def type7(values: list[float], p: float) -> float:
    """Hyndman-Fan type 7."""
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip().replace(",", " ")
    return ""


def arm_env(arm: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[key] = "1"
    for key in list(env):
        if key.startswith("H2H_") or key.startswith("ANNEAL_"):
            del env[key]
    if arm == "heat":
        env["H2H_PLATEAU_HEAT"] = "1"
    return env


def csv_result(text: str, system: str, seed: int, budget: int) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("system,") or f",anneal,{seed}," in line]
    headers = [line for line in lines if line.startswith("system,")]
    data = [line for line in lines if not line.startswith("system,")]
    if len(headers) != 1 or len(data) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one header and one row\n{text[-2000:]}")
    row = dict(zip(headers[0].split(","), data[0].split(",")))
    if row["system"] != system or int(row["seed"]) != seed or int(row["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: row {data[0]}")
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    outside = float(row["outside_s"]) if "outside_s" in row else wall - pot
    return {
        "hit": "1" if row["hit"].strip().lower() in ("1", "true") else "0",
        "evals_to_hit": str(int(row["evals_to_hit"])),
        "charged": str(int(row["charged"])),
        "wall_s": f"{wall:.6f}",
        "best": row["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "head": row.get("head", ""),
    }


def run_one(bin_dir: Path, tree: Path, arm: str, system: str, binary: str, seed: int, log: Path) -> dict[str, str]:
    command = ["taskset", "-c", "0", str(bin_dir / binary), "run", system, str(seed)]
    proc = subprocess.run(command, cwd=tree, env=arm_env(arm), capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} {arm} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    return csv_result(proc.stdout, system, seed, BUDGETS[system])


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    from scipy.stats import fisher_exact

    return float(fisher_exact([[hits_a, n_a - hits_a], [hits_b, n_b - hits_b]])[1])


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """Two-sided, normal approximation with the tie and continuity corrections."""
    from scipy.stats import mannwhitneyu

    if len(set(left + right)) == 1:
        return len(left) * len(right) / 2.0, 1.0
    result = mannwhitneyu(left, right, alternative="two-sided", method="asymptotic", use_continuity=True)
    return float(result.statistic), float(result.pvalue)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def summarise(rows: list[dict[str, str]], revisions: dict[str, str]) -> list[dict[str, str]]:
    out = []
    for system, driver in RUNS:
        groups = {}
        for arm in ARMS:
            by_seed = {
                int(r["seed"]): r
                for r in rows
                if r["system"] == system and r["arm"] == arm and r["code_revision"] == revisions[arm]
            }
            groups[arm] = by_seed
        seeds = [s for s in seeds_of(system) if s in groups["base"] and s in groups["heat"]]
        if not seeds:
            continue
        base = [groups["base"][s] for s in seeds]
        heat = [groups["heat"][s] for s in seeds]
        base_hits = sum(r["hit"] == "1" for r in base)
        heat_hits = sum(r["hit"] == "1" for r in heat)
        base_calls = [float(r["evals_to_hit"]) for r in base]
        heat_calls = [float(r["evals_to_hit"]) for r in heat]
        u, p = mann_whitney(heat_calls, base_calls)
        same = sum(
            1
            for b, h in zip(base, heat)
            if b["hit"] == h["hit"] and b["evals_to_hit"] == h["evals_to_hit"] and b["best"] == h["best"]
        )

        def med(group: list[dict[str, str]], column: str) -> str:
            return fmt(type7([float(r[column]) for r in group], 0.5))

        out.append(
            {
                "system": system,
                "driver": driver,
                "budget": str(BUDGETS[system]),
                "n": str(len(seeds)),
                "base_hits": str(base_hits),
                "heat_hits": str(heat_hits),
                "fisher_p": f"{fisher(heat_hits, len(seeds), base_hits, len(seeds)):.6g}",
                "base_evals_median": f"{type7(base_calls, 0.5):.1f}",
                "heat_evals_median": f"{type7(heat_calls, 0.5):.1f}",
                "mannwhitney_u": f"{u:.1f}",
                "mannwhitney_p": f"{p:.6g}",
                "base_wall_median_s": med(base, "wall_s"),
                "heat_wall_median_s": med(heat, "wall_s"),
                "base_pot_median_s": med(base, "pot_s"),
                "heat_pot_median_s": med(heat, "pot_s"),
                "base_outside_median_s": med(base, "outside_s"),
                "heat_outside_median_s": med(heat, "outside_s"),
                "same_seeds": str(same),
                "base_revision": revisions["base"],
                "heat_revision": revisions["heat"],
                "host": " ".join(sorted({r["host"] for r in base + heat})),
                "seeds": f"{seeds[0]}-{seeds[-1]}",
            }
        )
    return out


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12 or abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7")
    if {run[0] for run in RUNS} != set(BUDGETS):
        raise SystemExit("every system runs")
    if len(SEEDS_AU55) != 24 or len(SEEDS) != 12:
        raise SystemExit("seed counts")
    if "H2H_PLATEAU_HEAT" in arm_env("base") or arm_env("heat").get("H2H_PLATEAU_HEAT") != "1":
        raise SystemExit("arm environment")
    text = "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s\n"
    text += "lj38,anneal,2000,400000,true,9000,9100,0.5,-173.92842700,0.25,0.25\n"
    row = csv_result(text, "lj38", 2000, 400_000)
    if row["hit"] != "1" or row["outside_s"] != "0.250000":
        raise SystemExit(f"csv row {row}")
    u, p = mann_whitney([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    if abs(u - 4.5) > 1e-12 or abs(p - 1.0) > 1e-12:
        raise SystemExit(f"mann whitney {u} {p}")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--heat-revision", required=True)
    parser.add_argument("--base-bin", type=Path)
    parser.add_argument("--heat-bin", type=Path)
    parser.add_argument("--base-tree", type=Path)
    parser.add_argument("--heat-tree", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--system", action="append")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv[1:])
    revisions = {"base": args.base_revision, "heat": args.heat_revision}
    if not args.summarize:
        if socket.gethostname() != "bench":
            print(f"uts hostname is {socket.gethostname()}, expected bench", file=sys.stderr)
            return 2
        bins = {"base": args.base_bin, "heat": args.heat_bin}
        trees = {"base": args.base_tree, "heat": args.heat_tree}
        if None in bins.values() or None in trees.values() or args.logs is None:
            parser.error("both bins, both trees and --logs are needed to run")
        for arm in ARMS:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=trees[arm], capture_output=True, text=True, check=True
            ).stdout.strip()
            if head != revisions[arm]:
                raise SystemExit(f"{trees[arm]} is at {head}, not {revisions[arm]}")
        host = socket.gethostname()
        cpu = cpu_model()
        done = {(r["system"], r["arm"], r["seed"], r["code_revision"]) for r in read_rows(args.rows)}
        for system, driver in RUNS:
            if args.system and system not in args.system:
                continue
            for seed in seeds_of(system):
                for arm in ARMS:
                    key = (system, arm, str(seed), revisions[arm])
                    if key in done:
                        print(f"skip {system} {arm} {seed}", flush=True)
                        continue
                    log = args.logs / system / f"{arm}-{seed}.log"
                    result = run_one(bins[arm], trees[arm], arm, system, driver, seed, log)
                    if result["head"] and result["head"] != revisions[arm]:
                        raise SystemExit(f"{system} {arm} {seed}: binary reports head {result['head']}")
                    row = {
                        "system": system,
                        "driver": driver,
                        "arm": arm,
                        "code": "anneal",
                        "seed": str(seed),
                        "budget": str(BUDGETS[system]),
                        **{k: v for k, v in result.items() if k != "head"},
                        "code_revision": revisions[arm],
                        "host": host,
                        "cpu": cpu,
                    }
                    append_row(args.rows, row)
                    done.add(key)
                    print(
                        f"done {system} {arm} {seed} hit={row['hit']} calls={row['evals_to_hit']} "
                        f"wall_s={row['wall_s']} pot_s={row['pot_s']} outside_s={row['outside_s']}",
                        flush=True,
                    )
    if args.summary:
        write_csv(args.summary, SUMMARY_FIELDS, summarise(read_rows(args.rows), revisions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
