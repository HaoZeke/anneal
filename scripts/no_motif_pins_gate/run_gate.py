#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial two-arm no-loss gate on the non-Lennard-Jones panels.

Arm A and arm B are two builds of the same head-to-head drivers. Every driver
takes its call budget from its own paper table, so no budget is set here.
Each (system, seed) pair is a round. Both arms run back to back on one core,
and the arm that runs first flips with the seed parity. There is no early
exit, no warm-up run, and no second attempt: a run that fails or exceeds the
job timeout is recorded once and scored as a miss at the budget.

    run_gate.py run --arm-a-bin DIR --arm-b-bin DIR ...
    run_gate.py summarize --rows rows.csv --panels panels.csv --summary summary.csv

A hit is a quenched energy at or below the database energy plus the hit
tolerance of the driver. `calls` is `evals_to_hit`, with a miss entered at the
budget. `pot_s` is the time inside the potential and `outside_s` is the rest
of the wall.

The verdict reads each system on its own. B loses a system when it has fewer
hits than A and the two-sided Fisher exact test on hits is below 0.05, when
its calls rank higher than A's and the tie- and continuity-corrected two-sided
Mann-Whitney test on calls is below 0.05, or when a run of B fails where the
same seed of A does not. Any lost system is FIX. Otherwise the gate is PASS.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

SEEDS = tuple(range(2000, 2012))
ALPHA = 0.05
JOB_TIMEOUT_S = 7200.0

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
)

TIP4P_SIZES = (8, 12, 15, 16, 20, 21)
SC_SYSTEMS = tuple(f"sc{family}-{n}" for n in (38, 55, 80) for family in (126, 108, 96))
MORSE_SYSTEMS = ("m75r14", "m80r14", "m80r6")
BLJ_SYSTEMS = (
    "blj105n38",
    "blj115n38",
    "blj130n38",
    "blj105n55",
    "blj115n55",
    "blj130n55",
    "blj130n98",
)

# Cheapest systems first, so a stopped session still holds whole panels.
PANELS: tuple[tuple[str, str], ...] = (
    *((f"tip4p{n}", "tip4p") for n in TIP4P_SIZES),
    *((name, "sc") for name in SC_SYSTEMS),
    *((name, "morse") for name in MORSE_SYSTEMS),
    *((name, "blj") for name in BLJ_SYSTEMS),
)

DRIVERS = {
    "morse": "h2h_morse",
    "sc": "h2h_sc",
    "blj": "h2h_blj",
    "tip4p": "h2h_seed",
}

FIELDS = (
    "system",
    "arm",
    "seed",
    "position",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "status",
    "library",
    "harness",
)

PANEL_FIELDS = ("system", "driver", "budget", "reference")

SUMMARY_FIELDS = (
    "system",
    "budget",
    "reference",
    "hits_a",
    "hits_b",
    "runs_a",
    "runs_b",
    "fisher_p",
    "calls_median_a",
    "calls_median_b",
    "mw_p_calls",
    "calls_rank_mean_a",
    "calls_rank_mean_b",
    "wall_median_a",
    "wall_median_b",
    "pot_median_a",
    "pot_median_b",
    "outside_median_a",
    "outside_median_b",
    "mw_p_wall",
    "failed_a",
    "failed_b",
    "loss_hits",
    "loss_calls",
    "loss_failure",
    "gain_hits",
    "gain_calls",
    "verdict",
)


def type7(values: list[float], p: float = 0.5) -> float:
    """Hyndman-Fan type 7 quantile."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    if n == 1:
        return s[0]
    h = 1.0 + (n - 1) * p
    lo = int(h)
    frac = h - lo
    idx = lo - 1
    if frac == 0.0:
        return s[idx]
    return s[idx] * (1.0 - frac) + s[idx + 1] * frac


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the table [[a, b], [c, d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1, row2 = a + b, a + c, c + d

    def mass(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = mass(a)
    total = 0.0
    for x in range(max(0, col1 - row2), min(row1, col1) + 1):
        px = mass(x)
        if px <= observed * (1.0 + 1e-9):
            total += px
    return min(1.0, total)


def tie_groups(ordered: list[float]) -> list[tuple[int, int]]:
    """Index ranges [i, j] of equal values in a sorted list."""
    groups = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[i]:
            j += 1
        groups.append((i, j))
        i = j + 1
    return groups


def mann_whitney(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Two-sided Mann-Whitney U on the normal approximation.

    The variance carries the tie correction and the statistic the continuity
    correction. Returns the p value, then the mean rank of x and of y. Samples
    that are all tied give p = 1.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    ordered = sorted(x + y)
    n = len(ordered)
    ranks: dict[float, float] = {}
    tie = 0.0
    for i, j in tie_groups(ordered):
        ranks[ordered[i]] = (i + 1 + j + 1) / 2.0
        t = j - i + 1
        tie += t**3 - t
    r1 = sum(ranks[v] for v in x)
    r2 = sum(ranks[v] for v in y)
    if n < 2:
        return 1.0, r1 / n1, r2 / n2
    u1 = r1 - n1 * (n1 + 1) / 2.0
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if sigma2 <= 0.0:
        return 1.0, r1 / n1, r2 / n2
    z = max(0.0, abs(u1 - n1 * n2 / 2.0) - 0.5) / math.sqrt(sigma2)
    return min(1.0, math.erfc(z / math.sqrt(2.0))), r1 / n1, r2 / n2


class Arm(NamedTuple):
    name: str
    bin_dir: Path
    library: str


def read_tip4p_references(repo: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with (repo / "data" / "tip4p_ccd.csv").open() as handle:
        for row in csv.DictReader(handle):
            out[int(row["n"])] = float(row["energy"])
    return out


def tip4p_constants(repo: Path) -> tuple[int, float]:
    """Budget and hit tolerance of the TIP4P head-to-head."""
    sys.path.insert(0, str(repo / "scripts" / "tip4p_h2h"))
    import run_h2h  # noqa: PLC0415

    return int(run_h2h.BUDGET), float(run_h2h.HIT_TOL)


def tip4p_size(system: str) -> int:
    return int(system.removeprefix("tip4p"))


def job_env(work: Path, extra: dict[str, str]) -> dict[str, str]:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(work / "home"),
        "TMPDIR": str(work / "tmp"),
        "LANG": "C",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
    }
    for key in THREAD_VARS:
        env[key] = "1"
    env.update(extra)
    return env


def query_budget(binary: Path, system: str, env: dict[str, str]) -> tuple[int, float]:
    out = subprocess.run(
        [str(binary), "budget", system],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.split()
    return int(out[0]), float(out[1])


def preflight_energy(binary: Path, system: str, env: dict[str, str], scratch: Path) -> None:
    """Load the potential library once, so a missing one stops the session early."""
    start = scratch / f"preflight-{system}.xyz"
    subprocess.run(
        [str(binary), "coords", system, str(SEEDS[0]), str(start)],
        check=True,
        capture_output=True,
        env=env,
    )
    out = subprocess.run(
        [str(binary), "energy", system, str(start)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.split()
    if not out or not math.isfinite(float(out[0])):
        raise RuntimeError(f"{binary.name} gave no finite energy for {system}")


def command_of(
    system: str,
    driver: str,
    arm: Arm,
    seed: int,
    scratch: Path,
    tip4p: tuple[int, float, dict[int, float]],
) -> list[str]:
    binary = str(arm.bin_dir / DRIVERS[driver])
    if driver != "tip4p":
        return [binary, "run", system, str(seed)]
    budget, tol, references = tip4p
    n = tip4p_size(system)
    prefix = scratch / f"{system}-{arm.name}-s{seed}"
    return [
        binary,
        "tip4p",
        str(n),
        "0",
        str(budget),
        str(seed),
        "recommended",
        repr(references[n] + tol),
        str(prefix),
    ]


def parse_output(text: str, driver: str) -> dict[str, str]:
    if driver == "tip4p":
        for line in text.splitlines():
            if line.startswith("h2h_row "):
                return dict(part.split("=", 1) for part in line.split()[1:])
        raise ValueError("no h2h_row line")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("system,code,seed,"):
            continue
        header = line.split(",")
        for candidate in lines[i + 1 :]:
            values = candidate.split(",")
            if len(values) == len(header) and values[2].isdigit():
                row = dict(zip(header, values, strict=True))
                row["hit"] = "1" if row["hit"] in ("true", "1") else "0"
                return row
        break
    raise ValueError("no result row")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def run_job(
    argv: list[str], cpu: str, env: dict[str, str], cwd: Path, log: Path, timeout: float
) -> tuple[str, str, float]:
    """Run one job on one core and return its status, output and elapsed seconds."""
    pinned = [shutil.which("taskset") or "taskset", "-c", cpu, *argv]
    started = time.monotonic()
    proc = subprocess.Popen(
        pinned,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        status = "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        status = "timeout"
    elapsed = time.monotonic() - started
    log.write_text(out or "")
    return status, out or "", elapsed


def failed_row(base: dict[str, str], status: str, budget: int, elapsed: float) -> dict[str, str]:
    row = dict(base)
    row.update(
        {
            "budget": str(budget),
            "hit": "0",
            "evals_to_hit": str(budget),
            "status": status,
            "wall_s": f"{elapsed:.6f}",
        }
    )
    return row


def session_host() -> str:
    return Path("/proc/sys/kernel/hostname").read_text().strip()


def cmd_run(args: argparse.Namespace) -> int:
    if session_host() != args.session_host:
        print(
            f"refusing to run: the session hostname is {session_host()!r}, "
            f"not {args.session_host!r}",
            file=sys.stderr,
        )
        return 2
    repo = Path(args.repo).resolve()
    work = Path(args.work).resolve()
    for sub in ("home", "tmp", "logs", "scratch"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    scratch = work / "scratch"
    arms = (
        Arm("A", Path(args.arm_a_bin).resolve(), args.arm_a_library),
        Arm("B", Path(args.arm_b_bin).resolve(), args.arm_b_library),
    )
    extra: dict[str, str] = {}
    if args.rgpot_lib:
        lib = Path(args.rgpot_lib).resolve()
        extra["RGPOT_SC_LIB"] = str(lib)
        extra["LD_LIBRARY_PATH"] = str(lib.parent)
    env = job_env(work, extra)
    tip_budget, tip_tol = tip4p_constants(repo)
    references = read_tip4p_references(repo)
    tip4p = (tip_budget, tip_tol, references)
    panels = [p for p in PANELS if not args.only or p[0] in args.only]
    rows_path = Path(args.rows)
    rows_path.parent.mkdir(parents=True, exist_ok=True)

    table: list[dict[str, str]] = []
    for system, driver in panels:
        if driver == "tip4p":
            table.append(
                {
                    "system": system,
                    "driver": DRIVERS[driver],
                    "budget": str(tip_budget),
                    "reference": f"{references[tip4p_size(system)]:.10f}",
                }
            )
            continue
        seen = {query_budget(arm.bin_dir / DRIVERS[driver], system, env) for arm in arms}
        if len(seen) != 1:
            print(f"{system}: the arms disagree on budget and reference: {seen}", file=sys.stderr)
            return 2
        budget, reference = seen.pop()
        table.append(
            {
                "system": system,
                "driver": DRIVERS[driver],
                "budget": str(budget),
                "reference": f"{reference:.10f}",
            }
        )
        if driver == "sc":
            for arm in arms:
                preflight_energy(arm.bin_dir / DRIVERS[driver], system, env, scratch)
    with Path(args.panels).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=PANEL_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    budgets = {r["system"]: int(r["budget"]) for r in table}

    done = {(r["system"], r["arm"], int(r["seed"])) for r in read_rows(rows_path)}
    total = len(panels) * len(SEEDS) * len(arms)
    count = len(done)
    print(f"{total} runs, {count} already recorded", flush=True)
    for system, driver in panels:
        for seed in SEEDS:
            order = arms if seed % 2 == 0 else tuple(reversed(arms))
            for position, arm in enumerate(order, start=1):
                if (system, arm.name, seed) in done:
                    continue
                base = {
                    "system": system,
                    "arm": arm.name,
                    "seed": str(seed),
                    "position": str(position),
                    "library": arm.library,
                    "harness": args.harness,
                }
                argv = command_of(system, driver, arm, seed, scratch, tip4p)
                log = work / "logs" / f"{system}-{arm.name}-s{seed}.log"
                status, out, elapsed = run_job(
                    argv, args.cpu, env, scratch, log, args.job_timeout_s
                )
                if status != "ok":
                    row = failed_row(base, status, budgets[system], elapsed)
                else:
                    try:
                        parsed = parse_output(out, driver)
                        row = dict(base)
                        row.update({k: parsed[k] for k in (
                            "budget", "hit", "evals_to_hit", "charged", "best",
                            "wall_s", "pot_s", "outside_s",
                        )})
                        row["status"] = "ok"
                    except (KeyError, ValueError) as err:
                        row = failed_row(base, f"parse:{err}", budgets[system], elapsed)
                append_row(rows_path, row)
                count += 1
                print(
                    f"[{count}/{total}] {system} seed {seed} arm {arm.name} "
                    f"{row['status']} hit {row['hit']} calls {row['evals_to_hit']} "
                    f"wall {row['wall_s']}",
                    flush=True,
                )
    return 0


def system_summary(
    system: str, rows: list[dict[str, str]], panel: dict[str, str]
) -> dict[str, str]:
    seeds = {int(r["seed"]) for r in rows if r["system"] == system}
    by_arm = {
        arm: sorted(
            (r for r in rows if r["system"] == system and r["arm"] == arm),
            key=lambda r: int(r["seed"]),
        )
        for arm in ("A", "B")
    }
    good = {arm: [r for r in rs if r["status"] == "ok"] for arm, rs in by_arm.items()}
    failed = {arm: {r["seed"] for r in rs if r["status"] != "ok"} for arm, rs in by_arm.items()}
    # A failed run was written as a miss at the budget, so a crash cannot help.
    hits = {arm: sum(int(r["hit"]) for r in rs) for arm, rs in by_arm.items()}
    runs = {arm: len(rs) for arm, rs in by_arm.items()}
    p_hits = fisher_two_sided(
        hits["A"], runs["A"] - hits["A"], hits["B"], runs["B"] - hits["B"]
    )
    calls = {arm: [float(r["evals_to_hit"]) for r in rs] for arm, rs in by_arm.items()}
    p_calls, rank_a, rank_b = mann_whitney(calls["A"], calls["B"])

    def column(arm: str, key: str) -> list[float]:
        return [float(r[key]) for r in good[arm]]

    def median(arm: str, key: str) -> float:
        return type7(column(arm, key))

    p_wall = mann_whitney(column("A", "wall_s"), column("B", "wall_s"))[0]
    loss_hits = hits["B"] < hits["A"] and p_hits < ALPHA
    loss_calls = rank_b > rank_a and p_calls < ALPHA
    loss_failure = bool(failed["B"] - failed["A"])
    gain_hits = hits["B"] > hits["A"] and p_hits < ALPHA
    gain_calls = rank_b < rank_a and p_calls < ALPHA
    complete = seeds == set(SEEDS) and runs["A"] == len(SEEDS) and runs["B"] == len(SEEDS)
    if loss_hits or loss_calls or loss_failure:
        verdict = "FIX"
    else:
        verdict = "PASS" if complete else "INCOMPLETE"

    def f(value: float) -> str:
        return f"{value:.6g}"

    return {
        "system": system,
        "budget": panel["budget"],
        "reference": panel["reference"],
        "hits_a": str(hits["A"]),
        "hits_b": str(hits["B"]),
        "runs_a": str(runs["A"]),
        "runs_b": str(runs["B"]),
        "fisher_p": f(p_hits),
        "calls_median_a": f(type7(calls["A"])),
        "calls_median_b": f(type7(calls["B"])),
        "mw_p_calls": f(p_calls),
        "calls_rank_mean_a": f(rank_a),
        "calls_rank_mean_b": f(rank_b),
        "wall_median_a": f(median("A", "wall_s")),
        "wall_median_b": f(median("B", "wall_s")),
        "pot_median_a": f(median("A", "pot_s")),
        "pot_median_b": f(median("B", "pot_s")),
        "outside_median_a": f(median("A", "outside_s")),
        "outside_median_b": f(median("B", "outside_s")),
        "mw_p_wall": f(p_wall),
        "failed_a": str(len(failed["A"])),
        "failed_b": str(len(failed["B"])),
        "loss_hits": str(int(loss_hits)),
        "loss_calls": str(int(loss_calls)),
        "loss_failure": str(int(loss_failure)),
        "gain_hits": str(int(gain_hits)),
        "gain_calls": str(int(gain_calls)),
        "verdict": verdict,
    }


def overall_verdict(summary: list[dict[str, str]]) -> str:
    lost = [r["system"] for r in summary if r["verdict"] == "FIX"]
    if lost:
        return "FIX: " + ", ".join(lost)
    if len(summary) != len(PANELS) or any(r["verdict"] != "PASS" for r in summary):
        return "INCOMPLETE"
    return "PASS"


def markdown(summary: list[dict[str, str]]) -> str:
    head = (
        "| system | budget | hits A | hits B | Fisher p | calls A | calls B | MW p calls "
        "| wall A | wall B | inside A | inside B | outside A | outside B | verdict |"
    )
    lines = [head, "|" + " --- |" * 15]
    for r in summary:
        lines.append(
            f"| {r['system']} | {r['budget']} | {r['hits_a']}/{r['runs_a']} "
            f"| {r['hits_b']}/{r['runs_b']} | {r['fisher_p']} | {r['calls_median_a']} "
            f"| {r['calls_median_b']} | {r['mw_p_calls']} | {r['wall_median_a']} "
            f"| {r['wall_median_b']} | {r['pot_median_a']} | {r['pot_median_b']} "
            f"| {r['outside_median_a']} | {r['outside_median_b']} | {r['verdict']} |"
        )
    return "\n".join(lines)


def cmd_summarize(args: argparse.Namespace) -> int:
    rows = read_rows(Path(args.rows))
    if not rows:
        print("no rows", file=sys.stderr)
        return 2
    panels = {r["system"]: r for r in read_rows(Path(args.panels))}
    systems = [s for s, _ in PANELS if any(r["system"] == s for r in rows)]
    summary = [system_summary(s, rows, panels[s]) for s in systems]
    with Path(args.summary).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(overall_verdict(summary))
    print(markdown(summary))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run every job once, serially, on one core")
    run.add_argument("--arm-a-bin", required=True, help="directory with the arm A drivers")
    run.add_argument("--arm-b-bin", required=True, help="directory with the arm B drivers")
    run.add_argument("--arm-a-library", required=True, help="library identity of arm A")
    run.add_argument("--arm-b-library", required=True, help="library identity of arm B")
    run.add_argument("--harness", required=True, help="revision that carries the drivers")
    run.add_argument("--repo", required=True, help="checkout that holds the reference tables")
    run.add_argument("--work", required=True, help="scratch directory")
    run.add_argument("--rows", required=True, help="append-only result rows")
    run.add_argument("--panels", required=True, help="where the budgets and references go")
    run.add_argument("--session-host", required=True, help="hostname the session must have")
    run.add_argument("--rgpot-lib", help="shared library for the Sutton-Chen driver")
    run.add_argument("--cpu", default="0", help="core every job is pinned to")
    run.add_argument("--job-timeout-s", type=float, default=JOB_TIMEOUT_S)
    run.add_argument("--only", nargs="*", help="restrict to these systems")
    run.set_defaults(func=cmd_run)

    summ = sub.add_parser("summarize", help="tests and verdict from the rows")
    summ.add_argument("--rows", required=True)
    summ.add_argument("--panels", required=True)
    summ.add_argument("--summary", required=True)
    summ.set_defaults(func=cmd_summarize)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
