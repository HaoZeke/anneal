#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial runner for the contact-cutoff gate.

    run_gate.py --arm DEVELOP=<checkout>:<build> --arm CC=... --arm DEVELOP2=...
                --repo <gate checkout> --rows rows.csv --logs <dir>
                --expect DEVELOP=<tree> --expect CC=<tree> --expect DEVELOP2=<tree>

Part 1 is LJ38, LJ55 and LJ75, seeds 2000-2002, arms DEVELOP and CC.
Part 2 is every non-Lennard-Jones cell of the campaign panel, seeds
2000-2011, arms DEVELOP, CC and DEVELOP2 (the second build of develop).
Each run is the arm's own campaign driver (``harness/``), ``run <system>
<seed>``, at the budget the driver itself carries, which must equal the
campaign's ``PAPER_BUDGETS``. No ``H2H_*`` or ``ANNEAL_*`` variable reaches
a run and every thread count is 1. One run at a time on core 0.

Arm order alternates by seed: even seeds run the arms in the listed
order and odd seeds in reverse (DEVELOP, CC / CC, DEVELOP in part 1;
DEVELOP, CC, DEVELOP2 / DEVELOP2, CC, DEVELOP in part 2). There is no
warm-up run and no shortened run. A row is appended and synced after
every run and never rewritten.

Before the first run and after the last, every arm's checkout must be
clean and its tree must equal the expected tree; the runner stops
otherwise. Trees are compared, never commit ids.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

HIT_TOL = 1.0e-3
CORE = 0
PART1_SEEDS = (2000, 2001, 2002)
PART2_SEEDS = tuple(range(2000, 2012))
PART1_ARMS = ("DEVELOP", "CC")
PART2_ARMS = ("DEVELOP", "CC", "DEVELOP2")
PART1_CELLS = (
    ("lj38", "h2h_gmin_proof"),
    ("lj55", "h2h_gmin_proof"),
    ("lj75", "h2h_gmin_proof"),
)
PART2_CELLS = tuple(
    (f"sc{family}-{n}", "h2h_sc") for family in ("126", "108", "96") for n in (38, 55, 80)
) + (
    ("au55", "h2h_metal"),
    ("m55r6", "h2h_morse"),
    ("m55r14", "h2h_morse"),
    ("m80r6", "h2h_morse"),
    ("m80r14", "h2h_morse"),
)
# Where the campaign declares each cell's paper budget.
BUDGET_SOURCES = {
    "h2h_gmin_proof": "scripts/gmin_panel/run_panel.py",
    "h2h_metal": "scripts/gmin_panel/run_panel.py",
    "h2h_morse": "scripts/gmin_panel/run_panel.py",
    "h2h_sc": "scripts/metal_h2h/run_sc.py",
}
ROW_FIELDS = (
    "part",
    "system",
    "arm",
    "seed",
    "slot",
    "core",
    "hit",
    "calls",
    "charged",
    "wall",
    "inside_s",
    "outside_s",
    "best",
    "best_bits",
    "budget",
    "trace",
    "machine",
    "library_tree",
    "harness_blob",
)


def paper_budgets(repo: Path, source: str) -> dict[str, int]:
    tree = ast.parse((repo / source).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PAPER_BUDGETS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit(f"{source} has no PAPER_BUDGETS")


def run_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("H2H_", "ANNEAL_"))}
    for key in list(env):
        if key.endswith("_THREADS"):
            env[key] = "1"
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def check_trees(arms: dict[str, tuple[Path, Path]], expect: dict[str, str]) -> None:
    for arm, (checkout, build) in arms.items():
        tree = git(checkout, "rev-parse", "HEAD^{tree}")
        dirty = git(checkout, "status", "--porcelain")
        built = (build / "tree").read_text().strip()
        if tree != expect[arm] or built != expect[arm] or dirty:
            raise SystemExit(
                f"{arm}: checkout tree {tree}, build tree {built}, expected {expect[arm]}, "
                f"dirty={bool(dirty)}"
            )
        print(f"tree ok {arm} {tree}", flush=True)


def blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def driver_row(text: str, system: str, seed: int) -> dict[str, str]:
    headers = [line for line in text.splitlines() if line.startswith("system,")]
    data = [line for line in text.splitlines() if line.startswith(f"{system},anneal,{seed},")]
    if len(headers) != 1 or len(data) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one header and one row")
    return dict(zip(headers[0].split(","), data[0].split(",")))


def gate_outcome(text: str) -> tuple[str, float, int, str]:
    lines = [line for line in text.splitlines() if line.startswith("gate_outcome,")]
    if len(lines) != 1:
        raise RuntimeError("expected one gate_outcome line")
    _tag, bits, best, charged, trace = lines[0].split(",")
    value = float(best)
    if f"{struct_bits(value)}" != bits:
        raise RuntimeError(f"best {best} does not round-trip to {bits}")
    return bits, value, int(charged), trace


def struct_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def one_run(
    part: str,
    system: str,
    driver: str,
    arm: str,
    seed: int,
    slot: int,
    checkout: Path,
    build: Path,
    budget: int,
    reference: float,
    logs: Path,
    common: dict[str, str],
) -> dict[str, str]:
    command = ["taskset", "-c", str(CORE), str(build / "bin" / driver), "run", system, str(seed)]
    print(f"slot {slot} {part} {system} {arm} seed {seed}", flush=True)
    proc = subprocess.run(command, cwd=checkout, env=run_env(), capture_output=True, text=True)
    log = logs / part / system / f"{arm}-{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"exit={proc.returncode}\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} {arm} {seed} exited {proc.returncode}")
    row = driver_row(proc.stdout, system, seed)
    bits, best, charged, trace = gate_outcome(proc.stdout)
    hit = math.isfinite(best) and best <= reference + HIT_TOL
    driver_hit = row["hit"].strip().lower() in ("1", "true")
    calls = int(row["evals_to_hit"])
    wall = float(row["wall_s"])
    inside = float(row["pot_s"])
    outside = float(row["outside_s"]) if "outside_s" in row else max(wall - inside, 0.0)
    if hit != driver_hit or int(row["charged"]) != charged or int(row["budget"]) != budget:
        raise RuntimeError(f"{system} {arm} {seed}: driver row {row} disagrees with the outcome")
    if f"{best:.8f}" != row["best"]:
        raise RuntimeError(f"{system} {arm} {seed}: best {best!r} prints as {row['best']}")
    if hit and not (0 < calls <= charged <= budget):
        raise RuntimeError(f"{system} {arm} {seed}: hit with calls {calls} charged {charged}")
    if not hit and (calls != budget or charged != budget):
        raise RuntimeError(f"{system} {arm} {seed}: miss with calls {calls} charged {charged}")
    if "tree" in row and row["tree"] != common["library_tree"]:
        raise RuntimeError(f"{system} {arm} {seed}: driver reports tree {row['tree']}")
    return {
        "part": part,
        "system": system,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "core": str(CORE),
        "hit": "1" if hit else "0",
        "calls": str(calls),
        "charged": str(charged),
        "wall": row["wall_s"],
        "inside_s": row["pot_s"],
        "outside_s": f"{outside:.6f}",
        "best": repr(best),
        "best_bits": bits,
        "budget": str(budget),
        "trace": trace,
        **common,
    }


def append(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def schedule() -> list[tuple[str, str, str, int, str]]:
    jobs = []
    for part, cells, seeds, arms in (
        ("1", PART1_CELLS, PART1_SEEDS, PART1_ARMS),
        ("2", PART2_CELLS, PART2_SEEDS, PART2_ARMS),
    ):
        for system, driver in cells:
            for seed in seeds:
                order = arms if seed % 2 == 0 else tuple(reversed(arms))
                jobs.extend((part, system, driver, seed, arm) for arm in order)
    return jobs


def pairs(values: list[str], what: str) -> dict[str, str]:
    out = {}
    for item in values:
        key, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--{what} needs ARM=VALUE, got {item}")
        out[key] = value
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--expect", action="append", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--machine-id", type=Path, default=Path("/etc/machine-id"))
    args = parser.parse_args(argv)
    arms = {}
    for arm, spec in pairs(args.arm, "arm").items():
        checkout, _, build = spec.partition(":")
        arms[arm] = (Path(checkout), Path(build))
    expect = pairs(args.expect, "expect")
    if set(arms) != set(PART2_ARMS) or set(expect) != set(PART2_ARMS):
        raise SystemExit(f"arms must be {PART2_ARMS}")
    if args.rows.exists():
        raise SystemExit(f"{args.rows} exists; rows are never resumed or rewritten")
    os.sched_setaffinity(0, {CORE})
    machine = hashlib.sha256(args.machine_id.read_bytes().strip()).hexdigest()
    harness = {arm: blob(build / "harness" / "build.rs") for arm, (_c, build) in arms.items()}
    if len(set(harness.values())) != 1 or harness["DEVELOP"] != blob(
        Path(__file__).resolve().parent / "harness" / "build.rs"
    ):
        raise SystemExit(f"harness differs between builds: {harness}")
    check_trees(arms, expect)
    budgets = {}
    for system, driver in PART1_CELLS + PART2_CELLS:
        declared = paper_budgets(args.repo, BUDGET_SOURCES[driver])[system]
        for arm, (checkout, build) in arms.items():
            out = subprocess.run(
                [str(build / "bin" / driver), "budget", system],
                cwd=checkout,
                env=run_env(),
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
            budget, reference = int(out[0]), float(out[1])
            if budget != declared or budgets.get(system, (budget, reference)) != (budget, reference):
                raise SystemExit(f"{system} {arm}: budget {budget} against PAPER_BUDGETS {declared}")
            budgets[system] = (budget, reference)
    for slot, (part, system, driver, seed, arm) in enumerate(schedule(), start=1):
        checkout, build = arms[arm]
        budget, reference = budgets[system]
        common = {
            "machine": machine,
            "library_tree": expect[arm],
            "harness_blob": harness[arm],
        }
        row = one_run(
            part, system, driver, arm, seed, slot, checkout, build, budget, reference, args.logs, common
        )
        append(args.rows, row)
        print(
            f"done slot {slot} {system} {arm} {seed} hit={row['hit']} calls={row['calls']} "
            f"wall={row['wall']}",
            flush=True,
        )
    check_trees(arms, expect)
    print("all runs finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
