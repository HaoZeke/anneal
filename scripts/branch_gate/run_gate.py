# Copyright (c) anneal developers
"""Same-session gate for `Config::branch_on_stall` against develop.

LJ38, LJ55, LJ75, LJ98, LJ102, LJ103, LJ104, LJ150, Gupta Au55 and Morse
M80 at rho 6 and 14, seeds 2000-2011, the paper budgets. Lennard-Jones runs
through ``h2h_gmin_proof``, Au55 through ``h2h_metal`` and Morse through
``h2h_morse``. LJ103 runs on a second build of each revision with
``bench/paper-0120/lj103-table.patch``.

Two arms in one session. ``develop`` is the develop build; ``branch`` is the
candidate build with ``H2H_BRANCH=1``, which turns the branch on in the
example's ``Config::recommended``. For each system and seed the develop
build runs, then the candidate, ``taskset -c 0``, one thread. Rows are
appended with the code revision, the source tree and the host after every
run and are never rewritten; a stored row for the same system, arm, seed
and revision is kept and its run skipped. The tree is ``git rev-parse
HEAD^{tree}`` of the checkout the binary was built from, so a row can be
matched to its source after the commit is rebuilt with a new date.

``--summarize`` writes the comparison from the rows: hits, Fisher's exact
test on hits, and the tie-corrected Mann-Whitney test (normal
approximation, continuity correction) on calls to the minimum, misses
entered at the budget, and on the time inside and outside the potential.
Medians are Hyndman-Fan type 7.

This process refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

SEEDS = tuple(range(2000, 2012))
PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj102": 4_000_000,
    "lj103": 4_000_000,
    "lj104": 4_000_000,
    "lj150": 4_000_000,
    "au55": 1_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
}

# System, driver, binary. Run in this order.
RUNS = (
    ("lj38", "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj55", "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj75", "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj98", "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj102", "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj103", "h2h_gmin_proof+lj103", "h2h_gmin_proof_lj103"),
    ("lj104", "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj150", "h2h_gmin_proof", "h2h_gmin_proof"),
    ("au55", "h2h_metal", "h2h_metal"),
    ("m80r6", "h2h_morse", "h2h_morse"),
    ("m80r14", "h2h_morse", "h2h_morse"),
)
ARMS = ("develop", "branch")

ROW_FIELDS = [
    "system",
    "driver",
    "arm",
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
    "code_tree",
    "host",
    "cpu",
]

COMPARE_FIELDS = [
    "system",
    "driver",
    "budget",
    "n",
    "branch_hits",
    "develop_hits",
    "fisher_p",
    "branch_evals_median",
    "develop_evals_median",
    "mannwhitney_u",
    "mannwhitney_p",
    "branch_wall_median_s",
    "develop_wall_median_s",
    "branch_pot_median_s",
    "develop_pot_median_s",
    "mannwhitney_p_pot",
    "branch_outside_median_s",
    "develop_outside_median_s",
    "mannwhitney_p_outside",
    "same_seeds",
    "branch_revision",
    "branch_tree",
    "develop_revision",
    "host",
]


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


def is_hit(text: str) -> bool:
    return text.strip().lower() in ("1", "true")


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip().replace(",", " ")
    return ""


def single_thread_env(arm: str) -> dict[str, str]:
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
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    if arm == "branch":
        env["H2H_BRANCH"] = "1"
    return env


def csv_result(text: str, system: str, seed: int, budget: int) -> dict[str, str]:
    """The result line of ``h2h_gmin_proof``, ``h2h_metal`` or ``h2h_morse``."""
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
        "hit": "1" if is_hit(row["hit"]) else "0",
        "evals_to_hit": str(int(row["evals_to_hit"])),
        "charged": str(int(row["charged"])),
        "wall_s": f"{wall:.6f}",
        "best": row["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
    }


def run_one(bin_dir: Path, tree: Path, run: tuple, arm: str, seed: int, log: Path) -> dict[str, str]:
    system, driver, binary = run
    budget = PAPER_BUDGETS[system]
    with tempfile.TemporaryDirectory():
        command = ["taskset", "-c", "0", str(bin_dir / binary), "run", system, str(seed)]
        proc = subprocess.run(
            command, cwd=tree, env=single_thread_env(arm), capture_output=True, text=True, check=False
        )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} {arm} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    return csv_result(proc.stdout, system, seed, budget)


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


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def arm_rows(rows: list[dict[str, str]], system: str, arm: str, revision: str) -> dict[int, dict[str, str]]:
    found: dict[int, dict[str, str]] = {}
    for row in rows:
        if row["system"] == system and row["arm"] == arm and row["code_revision"] == revision:
            found.setdefault(int(row["seed"]), row)
    return found


def compare(rows: list[dict[str, str]], develop: str, branch: str) -> list[dict[str, str]]:
    out = []
    for system, driver, _binary in RUNS:
        mine = arm_rows(rows, system, "branch", branch)
        base = arm_rows(rows, system, "develop", develop)
        seeds = sorted(set(mine) & set(base))
        if not seeds:
            continue
        a = [mine[s] for s in seeds]
        b = [base[s] for s in seeds]

        def col(group: list[dict[str, str]], name: str) -> list[float]:
            return [float(r[name]) for r in group]

        hits_a = sum(r["hit"] == "1" for r in a)
        hits_b = sum(r["hit"] == "1" for r in b)
        u, p = mann_whitney(col(a, "evals_to_hit"), col(b, "evals_to_hit"))
        _, p_pot = mann_whitney(col(a, "pot_s"), col(b, "pot_s"))
        _, p_out = mann_whitney(col(a, "outside_s"), col(b, "outside_s"))
        same = sum(
            1
            for x, y in zip(a, b)
            if x["hit"] == y["hit"] and x["evals_to_hit"] == y["evals_to_hit"] and x["best"] == y["best"]
        )
        out.append(
            {
                "system": system,
                "driver": driver,
                "budget": a[0]["budget"],
                "n": str(len(seeds)),
                "branch_hits": str(hits_a),
                "develop_hits": str(hits_b),
                "fisher_p": f"{fisher(hits_a, len(a), hits_b, len(b)):.6g}",
                "branch_evals_median": f"{type7(col(a, 'evals_to_hit'), 0.5):.1f}",
                "develop_evals_median": f"{type7(col(b, 'evals_to_hit'), 0.5):.1f}",
                "mannwhitney_u": f"{u:.1f}",
                "mannwhitney_p": f"{p:.6g}",
                "branch_wall_median_s": fmt(type7(col(a, "wall_s"), 0.5)),
                "develop_wall_median_s": fmt(type7(col(b, "wall_s"), 0.5)),
                "branch_pot_median_s": fmt(type7(col(a, "pot_s"), 0.5)),
                "develop_pot_median_s": fmt(type7(col(b, "pot_s"), 0.5)),
                "mannwhitney_p_pot": f"{p_pot:.6g}",
                "branch_outside_median_s": fmt(type7(col(a, "outside_s"), 0.5)),
                "develop_outside_median_s": fmt(type7(col(b, "outside_s"), 0.5)),
                "mannwhitney_p_outside": f"{p_out:.6g}",
                "same_seeds": str(same),
                "branch_revision": branch,
                "branch_tree": a[0]["code_tree"],
                "develop_revision": develop,
                "host": " ".join(sorted({r["host"] for r in a + b})),
            }
        )
    return out


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_text(tree: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=tree, capture_output=True, text=True, check=True).stdout.strip()


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12 or abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7")
    if {run[0] for run in RUNS} != set(PAPER_BUDGETS):
        raise SystemExit("every system runs")
    if len(SEEDS) != 12 or SEEDS[0] != 2000:
        raise SystemExit("seeds 2000-2011")
    text = "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s\n"
    text += "lj38,anneal,2000,400000,true,9000,9100,0.5,-173.92842700,abc,t,c,0,0.25\n"
    row = csv_result(text, "lj38", 2000, 400_000)
    if row["hit"] != "1" or row["outside_s"] != "0.250000":
        raise SystemExit(f"csv row {row}")
    u, p = mann_whitney([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    if abs(u - 4.5) > 1e-12 or abs(p - 1.0) > 1e-12:
        raise SystemExit(f"mann whitney {u} {p}")
    if single_thread_env("branch").get("H2H_BRANCH") != "1" or "H2H_BRANCH" in single_thread_env("develop"):
        raise SystemExit("arm environment")
    rows = []
    for arm, revision, hits in (("develop", "d", ("1", "0")), ("branch", "b", ("1", "1"))):
        for seed, hit in zip((2000, 2001), hits):
            rows.append(
                {
                    "system": "lj38", "driver": "h2h_gmin_proof", "arm": arm, "seed": str(seed),
                    "budget": "400000", "hit": hit, "evals_to_hit": "9000" if hit == "1" else "400000",
                    "charged": "9000", "wall_s": "1.0", "best": "-173.9", "pot_s": "0.5",
                    "outside_s": "0.5", "code_revision": revision, "code_tree": "t", "host": "bench", "cpu": "c",
                }
            )
    line = compare(rows, "d", "b")
    if len(line) != 1 or line[0]["branch_hits"] != "2" or line[0]["develop_hits"] != "1" or line[0]["same_seeds"] != "1":
        raise SystemExit(f"compare {line}")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--develop", required=True, help="develop revision")
    parser.add_argument("--branch", required=True, help="candidate revision")
    parser.add_argument("--develop-bin", type=Path)
    parser.add_argument("--branch-bin", type=Path)
    parser.add_argument("--develop-tree", type=Path)
    parser.add_argument("--branch-tree", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv[1:])
    if not args.summarize:
        if socket.gethostname() != "bench":
            print(f"uts hostname is {socket.gethostname()}, expected bench", file=sys.stderr)
            return 2
        needed = (args.develop_bin, args.branch_bin, args.develop_tree, args.branch_tree, args.logs)
        if any(value is None for value in needed):
            parser.error("--develop-bin, --branch-bin, --develop-tree, --branch-tree and --logs are needed to run")
        arms = {
            "develop": (args.develop, args.develop_bin, args.develop_tree),
            "branch": (args.branch, args.branch_bin, args.branch_tree),
        }
        trees = {}
        for arm, (revision, _bin, tree) in arms.items():
            head = git_text(tree, "rev-parse", "HEAD")
            if head != revision:
                raise SystemExit(f"{tree} is at {head}, not {revision}")
            trees[arm] = git_text(tree, "rev-parse", "HEAD^{tree}")
        host = socket.gethostname()
        cpu = cpu_model()
        stored = read_rows(args.rows)
        done = {(r["system"], r["arm"], r["seed"], r["code_revision"]) for r in stored}
        seeds = tuple(args.seed) if args.seed else SEEDS
        for run in RUNS:
            system, driver = run[0], run[1]
            if args.system and system not in args.system:
                continue
            for seed in seeds:
                for arm in ARMS:
                    revision, bin_dir, tree = arms[arm]
                    key = (system, arm, str(seed), revision)
                    if key in done:
                        print(f"skip {system} {arm} {seed}", flush=True)
                        continue
                    print(f"start {system} {arm} {seed}", flush=True)
                    log = args.logs / system / f"{arm}-{seed}.log"
                    result = run_one(bin_dir, tree, run, arm, seed, log)
                    row = {
                        "system": system,
                        "driver": driver,
                        "arm": arm,
                        "seed": str(seed),
                        "budget": str(PAPER_BUDGETS[system]),
                        **result,
                        "code_revision": revision,
                        "code_tree": trees[arm],
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
    if args.compare:
        write_csv(args.compare, COMPARE_FIELDS, compare(read_rows(args.rows), args.develop, args.branch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
