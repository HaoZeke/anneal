# Copyright (c) anneal developers
"""The paper headline panel on one release build, against stored develop rows.

LJ38, LJ55, LJ75, LJ98, LJ102, LJ103, LJ104, LJ150, Gupta Au55 and Morse
M80 at rho 6 and 14, seeds 2000-2011, the paper budgets. Each system runs
through the example its head-to-head rule runs: ``h2h_gmin_proof`` for
Lennard-Jones, ``h2h_metal`` for Au55 and ``h2h_morse`` for Morse.
``h2h_gmin_proof`` has no LJ103 entry, so LJ103 runs on a second build of
the same revision with ``bench/paper-0120/lj103-table.patch``, which adds
the Cambridge energy and the 4e6 budget and nothing else; its rows carry
driver ``h2h_gmin_proof+lj103``. The newest develop LJ75 rows were taken
through ``h2h_seed`` (``scripts/generic_alloy/run_ablation.py``), so LJ75
also runs that way, with that script's 1e-3 hit tolerance.

One process, ``taskset -c 0``, one thread. Rows are appended with the code
revision and the host after every run and are never rewritten; a row
already stored for the same system, driver, seed and revision is kept and
its run skipped. ``--summarize`` writes the summary and the comparison
from the rows: Fisher's exact test on hits and the tie-corrected
Mann-Whitney test (normal approximation) on calls to the minimum, misses
entered at the budget, against the develop rows named in ``DEVELOP``.

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

for _candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (_candidate / "target_tol.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
from target_tol import DEFAULT_HIT_TOL  # noqa: E402

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
LJ75_REFERENCE = -397.492331
SEED_HIT_TOL = DEFAULT_HIT_TOL

# System, driver, binary, arguments before the seed. Run in this order.
RUNS = (
    ("lj38", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj38")),
    ("lj55", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj55")),
    ("lj75", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj75")),
    ("lj98", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj98")),
    ("lj102", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj102")),
    ("lj103", "h2h_gmin_proof+lj103", "h2h_gmin_proof_lj103", ("run", "lj103")),
    ("lj104", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj104")),
    ("lj150", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj150")),
    ("au55", "h2h_metal", "h2h_metal", ("run", "au55")),
    ("m80r6", "h2h_morse", "h2h_morse", ("run", "m80r6")),
    ("m80r14", "h2h_morse", "h2h_morse", ("run", "m80r14")),
    ("lj75", "h2h_seed", "h2h_seed", ()),
)

# The newest develop rows in bench/ for each system and driver, seeds
# 2000-2011. "rows" files hold one line per seed; "summary" files hold only
# hit counts and medians, so they give Fisher and no Mann-Whitney.
# morse-panel stores no revision column; its rows are the tree that
# recorded them.
DEVELOP = {
    ("lj38", "h2h_gmin_proof"): {
        "path": "bench/perf-quench9/rows.csv",
        "where": {"code_revision": "65557416a95a7550f5fa8b50c5537d1621366f3e"},
        "calls": "calls",
    },
    ("lj55", "h2h_gmin_proof"): {
        "path": "bench/lj-n12-500aaf65/summary.csv",
        "where": {"arm": "anneal"},
        "summary": "500aaf651dcb613c644c202a5519417b872b0e6e",
    },
    ("lj75", "h2h_gmin_proof"): {
        "path": "bench/perf-quench9/rows.csv",
        "where": {"code_revision": "65557416a95a7550f5fa8b50c5537d1621366f3e"},
        "calls": "calls",
    },
    ("lj75", "h2h_seed"): {
        "path": "bench/generic-alloy/g3-priors-off-rows.csv",
        "where": {"arm": "develop", "code_revision": "7192e9f8d5fff6e581c00b4b1e30aa883dad00de"},
        "calls": "evals_to_hit",
    },
    ("lj98", "h2h_gmin_proof"): {
        "path": "bench/perf-quench9/rows.csv",
        "where": {"code_revision": "65557416a95a7550f5fa8b50c5537d1621366f3e"},
        "calls": "calls",
    },
    ("lj102", "h2h_gmin_proof"): {
        "path": "bench/lj-n12-500aaf65/summary.csv",
        "where": {"arm": "anneal"},
        "summary": "500aaf651dcb613c644c202a5519417b872b0e6e",
    },
    ("lj103", "h2h_gmin_proof+lj103"): {
        "path": "bench/lj-n12-500aaf65/summary.csv",
        "where": {"arm": "anneal"},
        "summary": "500aaf651dcb613c644c202a5519417b872b0e6e",
    },
    ("lj104", "h2h_gmin_proof"): {
        "path": "bench/regress-bb486bfd/rows.csv",
        "where": {"code_revision": "bb486bfdb2d0a1bbcbc31e9a55ca322b5441e707"},
        "calls": "evals_to_hit",
    },
    ("lj150", "h2h_gmin_proof"): {
        "path": "bench/regress-bb486bfd/rows.csv",
        "where": {"code_revision": "bb486bfdb2d0a1bbcbc31e9a55ca322b5441e707"},
        "calls": "evals_to_hit",
    },
    ("au55", "h2h_metal"): {
        "path": "bench/generic-alloy/g3-priors-on-rows.csv",
        "where": {"arm": "develop", "code_revision": "7192e9f8d5fff6e581c00b4b1e30aa883dad00de"},
        "calls": "evals_to_hit",
    },
    ("m80r6", "h2h_morse"): {
        "path": "bench/generic-alloy/g3-priors-on-rows.csv",
        "where": {"arm": "develop", "code_revision": "7192e9f8d5fff6e581c00b4b1e30aa883dad00de"},
        "calls": "evals_to_hit",
    },
    ("m80r14", "h2h_morse"): {
        "path": "bench/morse-panel/seeds.csv",
        "where": {"code": "anneal"},
        "calls": "evals_to_hit",
        "revision": "76ff29a6",
    },
}

ROW_FIELDS = [
    "system",
    "driver",
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
    "n",
    "hits",
    "hits_over_n",
    "budget",
    "evals_median",
    "evals_iqr",
    "evals_median_to_hit",
    "wall_median_s",
    "pot_median_s",
    "outside_median_s",
    "code_revision",
    "host",
    "seeds",
]

COMPARE_FIELDS = [
    "system",
    "driver",
    "budget",
    "release_hits",
    "release_n",
    "develop_hits",
    "develop_n",
    "fisher_p",
    "release_evals_median",
    "develop_evals_median",
    "mannwhitney_u",
    "mannwhitney_p",
    "release_wall_median_s",
    "develop_wall_median_s",
    "release_pot_median_s",
    "develop_pot_median_s",
    "release_outside_median_s",
    "develop_outside_median_s",
    "same_seeds",
    "release_revision",
    "develop_revision",
    "develop_source",
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
    for key in list(env):
        if key.startswith("H2H_"):
            del env[key]
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
        "head": row.get("head", ""),
    }


def seed_result(text: str, seed: int, budget: int) -> dict[str, str]:
    """The ``h2h_row`` line of ``h2h_seed``."""
    rows = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError(f"lj75 h2h_seed {seed}: expected one h2h_row\n{text[-2000:]}")
    values = dict(item.split("=", 1) for item in rows[0].split()[1:])
    if values["system"] != "lj75" or int(values["seed"]) != seed or int(values["budget"]) != budget:
        raise RuntimeError(f"lj75 h2h_seed {seed}: row {rows[0]}")
    return {
        "hit": "1" if is_hit(values["hit"]) else "0",
        "evals_to_hit": str(int(values["evals_to_hit"])),
        "charged": str(int(values["charged"])),
        "wall_s": f"{float(values['wall_s']):.6f}",
        "best": values["best"],
        "pot_s": f"{float(values['pot_s']):.6f}",
        "outside_s": f"{float(values['outside_s']):.6f}",
        "head": "",
    }


def run_one(bin_dir: Path, tree: Path, run: tuple, seed: int, log: Path) -> dict[str, str]:
    system, driver, binary, prefix = run
    budget = PAPER_BUDGETS[system]
    with tempfile.TemporaryDirectory() as scratch:
        if driver == "h2h_seed":
            stop = LJ75_REFERENCE + SEED_HIT_TOL
            args = ["lj", "75", "0", str(budget), str(seed), "recommended", repr(stop), str(Path(scratch) / "run")]
        else:
            args = [*prefix, str(seed)]
        command = ["taskset", "-c", "0", str(bin_dir / binary), *args]
        proc = subprocess.run(
            command, cwd=tree, env=single_thread_env(), capture_output=True, text=True, check=False
        )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} {driver} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    if driver == "h2h_seed":
        return seed_result(proc.stdout, seed, budget)
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


def develop_rows(root: Path, spec: dict) -> list[dict[str, str]]:
    with (root / spec["path"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if all(row.get(k) == v for k, v in spec["where"].items())]


def same_seed(mine: dict[str, str], base: dict[str, str], calls: str) -> bool:
    """Same hit and calls to the minimum, and the same best energy to 1e-8
    where the develop row stores one. Without it two misses match on calls."""
    if is_hit(base["hit"]) != (mine["hit"] == "1"):
        return False
    if int(float(base[calls])) != int(mine["evals_to_hit"]):
        return False
    if base.get("best"):
        return f"{float(base['best']):.8f}" == f"{float(mine['best']):.8f}"
    return True


def fmt(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def release_groups(rows: list[dict[str, str]], revision: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    for row in rows:
        if row["code_revision"] != revision:
            continue
        groups.setdefault((row["system"], row["driver"]), {}).setdefault(int(row["seed"]), row)
    return {key: [group[s] for s in sorted(group)] for key, group in groups.items()}


def summarise(rows: list[dict[str, str]], revision: str) -> list[dict[str, str]]:
    out = []
    groups = release_groups(rows, revision)
    for system, driver, _binary, _prefix in RUNS:
        group = groups.get((system, driver))
        if not group:
            continue
        calls = [float(r["evals_to_hit"]) for r in group]
        hits = [r for r in group if r["hit"] == "1"]
        out.append(
            {
                "system": system,
                "driver": driver,
                "n": str(len(group)),
                "hits": str(len(hits)),
                "hits_over_n": f"{len(hits)}/{len(group)}",
                "budget": group[0]["budget"],
                "evals_median": f"{type7(calls, 0.5):.1f}",
                "evals_iqr": f"{type7(calls, 0.75) - type7(calls, 0.25):.1f}",
                "evals_median_to_hit": (
                    f"{type7([float(r['evals_to_hit']) for r in hits], 0.5):.1f}" if hits else ""
                ),
                "wall_median_s": fmt(type7([float(r["wall_s"]) for r in group], 0.5)),
                "pot_median_s": fmt(type7([float(r["pot_s"]) for r in group], 0.5)),
                "outside_median_s": fmt(type7([float(r["outside_s"]) for r in group], 0.5)),
                "code_revision": revision,
                "host": " ".join(sorted({r["host"] for r in group})),
                "seeds": " ".join(r["seed"] for r in group),
            }
        )
    return out


def compare(root: Path, rows: list[dict[str, str]], revision: str) -> list[dict[str, str]]:
    out = []
    groups = release_groups(rows, revision)
    for system, driver, _binary, _prefix in RUNS:
        group = groups.get((system, driver))
        spec = DEVELOP.get((system, driver))
        if not group or spec is None:
            continue
        mine_hits = sum(r["hit"] == "1" for r in group)
        mine_calls = [float(r["evals_to_hit"]) for r in group]
        line = {key: "" for key in COMPARE_FIELDS}
        line.update(
            {
                "system": system,
                "driver": driver,
                "budget": group[0]["budget"],
                "release_hits": str(mine_hits),
                "release_n": str(len(group)),
                "release_evals_median": f"{type7(mine_calls, 0.5):.1f}",
                "release_wall_median_s": fmt(type7([float(r["wall_s"]) for r in group], 0.5)),
                "release_pot_median_s": fmt(type7([float(r["pot_s"]) for r in group], 0.5)),
                "release_outside_median_s": fmt(type7([float(r["outside_s"]) for r in group], 0.5)),
                "release_revision": revision,
                "develop_source": spec["path"],
            }
        )
        found = develop_rows(root, spec)
        if "summary" in spec:
            match = [r for r in found if r["system"] == system]
            if len(match) != 1 or int(match[0]["n"]) != len(SEEDS):
                raise SystemExit(f"{spec['path']} has no single twelve-seed {system} anneal row")
            base = match[0]
            if int(base["budget"]) != int(group[0]["budget"]):
                raise SystemExit(f"{system}: develop budget {base['budget']}")
            hits = int(base["hits"])
            line.update(
                {
                    "develop_hits": str(hits),
                    "develop_n": base["n"],
                    "fisher_p": f"{fisher(mine_hits, len(group), hits, int(base['n'])):.6g}",
                    "develop_evals_median": f"{float(base['evals_median']):.1f}",
                    "develop_wall_median_s": fmt(float(base["wall_median_s"])),
                    "develop_revision": spec["summary"],
                }
            )
            out.append(line)
            continue
        found = [r for r in found if r["system"] == system]
        by_seed = {int(r["seed"]): r for r in found}
        if sorted(by_seed) != list(SEEDS) or len(found) != len(SEEDS):
            raise SystemExit(f"{spec['path']}: {system} develop seeds {sorted(by_seed)}")
        base = [by_seed[s] for s in SEEDS]
        if "budget" in base[0] and int(base[0]["budget"]) != int(group[0]["budget"]):
            raise SystemExit(f"{system}: develop budget {base[0]['budget']}")
        base_hits = sum(is_hit(r["hit"]) for r in base)
        base_calls = [float(r[spec["calls"]]) for r in base]
        u, p = mann_whitney(mine_calls, base_calls)
        same = sum(1 for r in group if same_seed(r, by_seed[int(r["seed"])], spec["calls"]))

        def med(column: str) -> str:
            if column not in base[0]:
                return ""
            return fmt(type7([float(r[column]) for r in base], 0.5))

        revisions = sorted({r.get("code_revision", "") for r in base} - {""})
        line.update(
            {
                "develop_hits": str(base_hits),
                "develop_n": str(len(base)),
                "fisher_p": f"{fisher(mine_hits, len(group), base_hits, len(base)):.6g}",
                "develop_evals_median": f"{type7(base_calls, 0.5):.1f}",
                "mannwhitney_u": f"{u:.1f}",
                "mannwhitney_p": f"{p:.6g}",
                "develop_wall_median_s": med("wall_s"),
                "develop_pot_median_s": med("pot_s"),
                "develop_outside_median_s": med("outside_s"),
                "same_seeds": str(same),
                "develop_revision": " ".join(revisions) or spec.get("revision", ""),
            }
        )
        out.append(line)
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
    if {run[0] for run in RUNS} != set(PAPER_BUDGETS):
        raise SystemExit("every system runs")
    if any((run[0], run[1]) not in DEVELOP for run in RUNS):
        raise SystemExit("every run has a develop comparison")
    if len(SEEDS) != 12 or SEEDS[0] != 2000:
        raise SystemExit("seeds 2000-2011")
    text = "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s\n"
    text += "lj38,anneal,2000,400000,true,9000,9100,0.5,-173.92842700,abc,t,c,0,0.25\n"
    row = csv_result(text, "lj38", 2000, 400_000)
    if row["hit"] != "1" or row["outside_s"] != "0.250000" or row["head"] != "abc":
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--bin", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv[1:])
    if not args.summarize:
        if socket.gethostname() != "bench":
            print(f"uts hostname is {socket.gethostname()}, expected bench", file=sys.stderr)
            return 2
        if args.bin is None or args.tree is None or args.logs is None:
            parser.error("--bin, --tree and --logs are needed to run")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.tree, capture_output=True, text=True, check=True
        ).stdout.strip()
        if head != args.revision:
            raise SystemExit(f"{args.tree} is at {head}, not {args.revision}")
        host = socket.gethostname()
        cpu = cpu_model()
        stored = read_rows(args.rows)
        done = {(r["system"], r["driver"], r["seed"], r["code_revision"]) for r in stored}
        seeds = tuple(args.seed) if args.seed else SEEDS
        for run in RUNS:
            system, driver = run[0], run[1]
            if args.system and system not in args.system:
                continue
            for seed in seeds:
                key = (system, driver, str(seed), args.revision)
                if key in done:
                    print(f"skip {system} {driver} {seed}", flush=True)
                    continue
                print(f"start {system} {driver} {seed}", flush=True)
                log = args.logs / system / f"{driver}-{seed}.log"
                result = run_one(args.bin, args.tree, run, seed, log)
                if result["head"] and result["head"] != args.revision:
                    raise SystemExit(f"{system} {seed}: binary reports head {result['head']}")
                row = {
                    "system": system,
                    "driver": driver,
                    "code": "anneal",
                    "seed": str(seed),
                    "budget": str(PAPER_BUDGETS[system]),
                    **{k: v for k, v in result.items() if k != "head"},
                    "code_revision": args.revision,
                    "host": host,
                    "cpu": cpu,
                }
                append_row(args.rows, row)
                done.add(key)
                print(
                    f"done {system} {driver} {seed} hit={row['hit']} calls={row['evals_to_hit']} "
                    f"wall_s={row['wall_s']} pot_s={row['pot_s']} outside_s={row['outside_s']}",
                    flush=True,
                )
    rows = read_rows(args.rows)
    if args.summary:
        write_csv(args.summary, SUMMARY_FIELDS, summarise(rows, args.revision))
    if args.compare:
        write_csv(args.compare, COMPARE_FIELDS, compare(args.root, rows, args.revision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
