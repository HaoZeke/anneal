#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Paired gate for the rebased shelf series.

Arm A and arm B are two release builds. The harness points at those
builds and does not change a budget, a seed, or a start. Both arms use
Config::recommended through the drivers below.

Cells and budgets are the ones named by the shelf gate at
record/sc-shelf-r3-gate-v2 (788b7d725ddb7cb6a6893b32e520a481f1c4fdec).
Sutton-Chen names, budgets and references are PAPER_BUDGETS and
REFERENCES in scripts/metal_h2h/run_sc.py. LJ38, LJ75 and LJ98 use
PAPER_BUDGETS in scripts/gmin_panel/run_panel.py. LJ77 uses the single
budget that map assigns from LJ75 through LJ104. m30r14 uses the budget
and reference examples/h2h_morse.rs publishes. References for the
Lennard-Jones sizes are the Cambridge table in that gate.

Schedule. One run at a time, pinned to core 0, with every thread-count
variable set to 1. Seeds 2000 through 2011. Within a cell the two arms
of a seed run back to back and the first arm follows ABBA. There is no
warm-up and no quick flag. After that panel, LJ38 and LJ75 run a
same-binary A/A block on the same seeds, also ABBA.

Rule, declared before any run. Pass means B loses on no cell. A cell is
a loss when the exact one-sided Fisher p for fewer hits on B, multiplied
by the number of tests and capped at 1, is below 0.05, or when the exact
one-sided Mann-Whitney p for higher calls on B, adjusted the same way,
is below 0.05, or when the median outside-potential time per charged
call on B divided by the same median on A is above 1.1. Calls in the
Mann-Whitney test and in the call median are the calls to a hit on
hitting runs. The outside-time ratio uses every seed. The A/A block is
the noise band and is not a second loss rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

ALPHA = Fraction(1, 20)
SEEDS = tuple(range(2000, 2012))
AA_SYSTEMS = ("lj38", "lj75")
CORE = 0
HIT_TOL = Decimal("0.001")
OUTSIDE_LIMIT = Decimal("1.1")
ARM_A = "59397594b662be91ba86e07b2e013c6b2c9d8d55"
ARM_A_TREE = "f5589a1b53ed374600b4f25be10963c00e6e9364"
ARM_B_TREE = "a2471b8f410e30e522ddc0eca49e80a736587d3b"
R3 = "788b7d725ddb7cb6a6893b32e520a481f1c4fdec"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DOMAIN_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "NUMBA_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "JULIA_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
}

ROW_FIELDS = [
    "system",
    "family",
    "role",
    "comparison",
    "arm",
    "seed",
    "seed_index",
    "order",
    "budget",
    "reference",
    "hit",
    "calls",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "core",
    "affinity",
    "machine_id",
    "code_revision",
    "tree",
    "binary_sha256",
    "driver",
]


@dataclass(frozen=True)
class Cell:
    name: str
    family: str
    role: str
    driver: str
    budget: int
    reference: Decimal
    n: int


# Order follows the Sutton-Chen map, then the two named cells, then the
# no-regression sizes.
CELLS = (
    Cell("sc126-38", "sc", "sc", "sc_shelf_gate", 400_000, Decimal("-35419.9804"), 38),
    Cell("sc126-55", "sc", "sc", "sc_shelf_gate", 1_000_000, Decimal("-52884.6806"), 55),
    Cell("sc126-80", "sc", "sc", "sc_shelf_gate", 1_000_000, Decimal("-78414.6271"), 80),
    Cell("sc108-38", "sc", "sc", "sc_shelf_gate", 400_000, Decimal("-10117.2454"), 38),
    Cell("sc108-55", "sc", "sc", "sc_shelf_gate", 1_000_000, Decimal("-14814.5225"), 55),
    Cell("sc108-80", "sc", "sc", "sc_shelf_gate", 1_000_000, Decimal("-21819.9952"), 80),
    Cell("sc96-38", "sc", "sc", "sc_shelf_gate", 400_000, Decimal("-8917.7056"), 38),
    Cell("sc96-55", "sc", "sc", "sc_shelf_gate", 1_000_000, Decimal("-13217.8963"), 55),
    Cell("sc96-80", "sc", "sc", "sc_shelf_gate", 1_000_000, Decimal("-19560.8966"), 80),
    Cell("lj77", "lj", "named", "lj_shelf_gate", 4_000_000, Decimal("-409.083517"), 77),
    Cell("m30r14", "morse", "named", "h2h_morse", 400_000, Decimal("-106.835790"), 30),
    Cell("lj38", "lj", "no-regression", "lj_shelf_gate", 400_000, Decimal("-173.928427"), 38),
    Cell("lj75", "lj", "no-regression", "lj_shelf_gate", 4_000_000, Decimal("-397.492331"), 75),
    Cell("lj98", "lj", "no-regression", "lj_shelf_gate", 4_000_000, Decimal("-543.665361"), 98),
)


def order_for(index: int) -> str:
    """First arm of ABBA: A, B, B, A, and then the same four again."""
    return "AB" if index % 4 in (0, 3) else "BA"


def sequence_for(order: str, labels: tuple[str, str]) -> tuple[str, str]:
    if order == "AB":
        return labels
    if order == "BA":
        return labels[1], labels[0]
    raise RuntimeError(f"unknown order {order}")


def type7(values: list[Decimal], probability: Decimal) -> Decimal:
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("empty sample")
    if count == 1:
        return ordered[0]
    position = Decimal(1) + (Decimal(count) - 1) * probability
    lower = int(position.to_integral_value(rounding="ROUND_FLOOR"))
    frac = position - Decimal(lower)
    index = lower - 1
    if frac == 0:
        return ordered[index]
    return ordered[index] * (Decimal(1) - frac) + ordered[index + 1] * frac


def fisher_fewer(hits_a: int, n_a: int, hits_b: int, n_b: int) -> Fraction:
    """Exact one-sided Fisher p for fewer hits in the B row."""
    total = hits_a + hits_b
    ways = math.comb(n_a + n_b, n_b)
    low = max(0, total - n_a)
    high = min(total, n_b)
    if hits_b < low or hits_b > high:
        raise RuntimeError(f"hit count {hits_b} is outside the table")

    def weight(hits: int) -> int:
        return math.comb(total, hits) * math.comb(n_a + n_b - total, n_b - hits)

    extreme = sum(weight(hits) for hits in range(low, hits_b + 1))
    return Fraction(extreme, ways)


def midranks(values: list[Decimal]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    scaled = [0] * len(values)
    offset = 0
    count = len(values)
    while offset < count:
        end = offset
        while end + 1 < count and values[order[end + 1]] == values[order[offset]]:
            end += 1
        rank = offset + end + 2
        for index in range(offset, end + 1):
            scaled[order[index]] = rank
        offset = end + 1
    return scaled


def mann_whitney_greater(left: list[Decimal], right: list[Decimal]) -> Fraction | None:
    """Exact one-sided Mann-Whitney p that ``left`` is higher than ``right``."""
    if not left or not right:
        return None
    ranks = midranks(list(left) + list(right))
    n_left = len(left)
    observed = sum(ranks[:n_left])
    by_count: list[dict[int, int]] = [{0: 1}]
    for rank in ranks:
        nxt = [dict(layer) for layer in by_count]
        nxt.append({})
        for taken, layer in enumerate(by_count):
            if taken >= n_left:
                break
            dest = nxt[taken + 1]
            for size, count in layer.items():
                dest[size + rank] = dest.get(size + rank, 0) + count
        by_count = nxt[: n_left + 1]
    distribution = by_count[n_left]
    total = sum(distribution.values())
    extreme = sum(count for size, count in distribution.items() if size >= observed)
    return Fraction(extreme, total)


def adjust(raw: Fraction, tests: int) -> Fraction:
    scaled = raw * tests
    return Fraction(1) if scaled > 1 else scaled


def self_check() -> None:
    if len(SEEDS) != 12 or SEEDS[0] != 2000 or SEEDS[-1] != 2011:
        raise RuntimeError("seed set is not 2000-2011")
    if [order_for(i) for i in range(8)] != ["AB", "BA", "BA", "AB", "AB", "BA", "BA", "AB"]:
        raise RuntimeError("ABBA order drifted")
    names = [cell.name for cell in CELLS]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate cell")
    if any(name not in names for name in AA_SYSTEMS):
        raise RuntimeError("A/A cells are not on the panel")
    sc = [cell.name for cell in CELLS if cell.family == "sc"]
    if "sc126-55" not in sc or len(sc) != 9:
        raise RuntimeError("Sutton-Chen cell list is not the nine-row map")
    # A 2/2 versus B 0/2. One of the six equally likely tables has zero B hits.
    if fisher_fewer(2, 2, 0, 2) != Fraction(1, 6):
        raise RuntimeError("Fisher tail drifted")
    if fisher_fewer(0, 2, 2, 2) != Fraction(1):
        raise RuntimeError("Fisher direction drifted")
    # B values sit strictly above A. One of six rank pairs is that pair.
    if mann_whitney_greater(
        [Decimal(3), Decimal(4)], [Decimal(1), Decimal(2)]
    ) != Fraction(1, 6):
        raise RuntimeError("Mann-Whitney upper tail drifted")
    if mann_whitney_greater(
        [Decimal(1), Decimal(2)], [Decimal(3), Decimal(4)]
    ) != Fraction(1):
        raise RuntimeError("Mann-Whitney direction drifted")
    tied = mann_whitney_greater([Decimal(5), Decimal(5)], [Decimal(5), Decimal(5)])
    if tied != Fraction(1):
        raise RuntimeError("tied Mann-Whitney drifted")
    sample = [Decimal(i) for i in range(1, 5)]
    if type7(sample, Decimal("0.5")) != Decimal("2.5"):
        raise RuntimeError("type 7 median drifted")
    if type7(sample, Decimal("0.25")) != Decimal("1.75"):
        raise RuntimeError("type 7 lower quartile drifted")
    if type7(sample, Decimal("0.75")) != Decimal("3.25"):
        raise RuntimeError("type 7 upper quartile drifted")
    if adjust(Fraction(1, 6), 6) != Fraction(1):
        raise RuntimeError("Bonferroni cap drifted")
    if adjust(Fraction(1, 100), 4) != Fraction(1, 25):
        raise RuntimeError("Bonferroni product drifted")


def machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def git_sha(repo: Path, rev: str) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", rev],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def porcelain(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def refuse_stray_environment() -> None:
    stray = sorted(key for key in os.environ if key.startswith(("H2H_", "PRIOR_", "RGPOT_")))
    if stray:
        raise RuntimeError(f"refusing to run with {', '.join(stray)} set")


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(THREAD_ENV)
    for key in list(env):
        if key.startswith(("H2H_", "PRIOR_", "RGPOT_")):
            raise RuntimeError(f"refusing to run with {key} set")
    return env


def affinity_of(pid: int) -> str:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return ""
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return ""


def command(cell: Cell, exe: Path, seed: int) -> list[str]:
    if cell.driver == "lj_shelf_gate":
        return [
            str(exe),
            str(cell.n),
            str(seed),
            str(cell.budget),
            f"{cell.reference:.12f}",
        ]
    if cell.driver == "sc_shelf_gate":
        return [str(exe), "run", cell.name, str(seed)]
    if cell.driver == "h2h_morse":
        return [str(exe), "run", cell.name, str(seed)]
    raise RuntimeError(f"unknown driver {cell.driver}")


def parse_stdout(cell: Cell, text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if cell.driver in {"lj_shelf_gate", "sc_shelf_gate"}:
        if len(lines) != 2 or not lines[0].startswith("n,seed,budget,hit,"):
            raise RuntimeError(f"unexpected runner output: {text!r}")
        fields = lines[1].split(",")
        if len(fields) != 9:
            raise RuntimeError(f"unexpected runner row: {lines[1]!r}")
        return {
            "n": fields[0],
            "seed": fields[1],
            "budget": fields[2],
            "hit": fields[3],
            "calls": fields[4],
            "charged": fields[5],
            "best": fields[6],
            "wall_s": fields[7],
            "pot_s": fields[8],
        }
    header = None
    data = None
    for line in lines:
        if line.startswith("system,code,seed,budget,hit,"):
            header = line.split(",")
            continue
        if header is not None and data is None and line.startswith(f"{cell.name},"):
            data = line.split(",")
            break
    if header is None or data is None or len(header) != len(data):
        raise RuntimeError(f"unexpected morse output: {text!r}")
    row = dict(zip(header, data))
    return {
        "n": str(cell.n),
        "seed": row["seed"],
        "budget": row["budget"],
        "hit": row["hit"],
        "calls": row["evals_to_hit"],
        "charged": row["charged"],
        "best": row["best"],
        "wall_s": row["wall_s"],
        "pot_s": row["pot_s"],
    }


def hit_flag(text: str) -> int:
    if text in {"1", "true"}:
        return 1
    if text in {"0", "false"}:
        return 0
    raise RuntimeError(f"bad hit flag {text}")


def outside_seconds(wall: str, pot: str) -> str:
    value = Decimal(wall) - Decimal(pot)
    if value < 0:
        value = Decimal(0)
    return f"{value:.6f}"


def check_parsed(cell: Cell, parsed: dict[str, str], seed: int) -> None:
    if int(parsed["n"]) != cell.n:
        raise RuntimeError(f"{cell.name}: driver n {parsed['n']} is not {cell.n}")
    if int(parsed["seed"]) != seed:
        raise RuntimeError(f"{cell.name}: driver seed {parsed['seed']} is not {seed}")
    if int(parsed["budget"]) != cell.budget:
        raise RuntimeError(f"{cell.name}: driver budget {parsed['budget']} is not {cell.budget}")
    hit = hit_flag(parsed["hit"])
    calls = int(parsed["calls"])
    charged = int(parsed["charged"])
    if charged < 1 or charged > cell.budget or calls > cell.budget:
        raise RuntimeError(f"{cell.name}: calls {calls} charged {charged} are outside the budget")
    if hit == 0 and calls != cell.budget:
        raise RuntimeError(f"{cell.name}: a miss was not entered at the budget")
    if hit == 1 and calls > charged:
        raise RuntimeError(f"{cell.name}: hit calls exceed charged calls")
    Decimal(parsed["wall_s"])
    Decimal(parsed["pot_s"])
    Decimal(parsed["best"])


def jobs() -> list[dict]:
    planned = []
    for cell in CELLS:
        for index, seed in enumerate(SEEDS):
            order = order_for(index)
            for arm in sequence_for(order, ("A", "B")):
                planned.append(
                    {
                        "cell": cell,
                        "comparison": "AB",
                        "arm": arm,
                        "seed": seed,
                        "seed_index": index,
                        "order": order,
                    }
                )
    for cell in CELLS:
        if cell.name not in AA_SYSTEMS:
            continue
        for index, seed in enumerate(SEEDS):
            order = order_for(index)
            for arm in sequence_for(order, ("A1", "A2")):
                planned.append(
                    {
                        "cell": cell,
                        "comparison": "AA",
                        "arm": arm,
                        "seed": seed,
                        "seed_index": index,
                        "order": order,
                    }
                )
    return planned


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    new_file = not path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_one(
    job: dict,
    exe: Path,
    revision: str,
    tree: str,
    binary_sha: str,
    machine: str,
    log_path: Path,
) -> dict[str, str]:
    cell: Cell = job["cell"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["taskset", "-c", str(CORE), *command(cell, exe, job["seed"])],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env(),
    )
    affinity = ""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        current = affinity_of(proc.pid)
        if current:
            affinity = current
        if current == str(CORE) or proc.poll() is not None:
            break
        time.sleep(0.005)
    stdout, stderr = proc.communicate()
    log_path.write_text(stdout + ("\n" if stdout and stderr else "") + stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cell.name} {job['comparison']} {job['arm']} seed {job['seed']} "
            f"exited {proc.returncode}: {stderr[-800:]}"
        )
    if affinity != str(CORE):
        raise RuntimeError(
            f"{cell.name} seed {job['seed']} affinity {affinity!r} is not core {CORE}"
        )
    parsed = parse_stdout(cell, stdout)
    check_parsed(cell, parsed, job["seed"])
    return {
        "system": cell.name,
        "family": cell.family,
        "role": cell.role,
        "comparison": job["comparison"],
        "arm": job["arm"],
        "seed": str(job["seed"]),
        "seed_index": str(job["seed_index"]),
        "order": job["order"],
        "budget": str(cell.budget),
        "reference": f"{cell.reference:.10f}",
        "hit": str(hit_flag(parsed["hit"])),
        "calls": parsed["calls"],
        "charged": parsed["charged"],
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "outside_s": outside_seconds(parsed["wall_s"], parsed["pot_s"]),
        "core": str(CORE),
        "affinity": affinity,
        "machine_id": machine,
        "code_revision": revision,
        "tree": tree,
        "binary_sha256": binary_sha,
        "driver": cell.driver,
    }


def log_name(job: dict) -> str:
    cell: Cell = job["cell"]
    return f"{job['comparison']}_{cell.name}_{job['arm']}_{job['seed']}.log"


def verify_identity(src: Path, expected_tree: str, expected_head: str | None) -> tuple[str, str]:
    head = git_sha(src, "HEAD")
    tree = git_sha(src, "HEAD^{tree}")
    if tree != expected_tree:
        raise RuntimeError(f"{src} tree {tree} is not {expected_tree}")
    if expected_head is not None and head != expected_head:
        raise RuntimeError(f"{src} HEAD {head} is not {expected_head}")
    status = porcelain(src)
    if status.strip():
        raise RuntimeError(f"{src} is not clean:\n{status}")
    return head, tree


def stage_drivers(src: Path, harness: Path) -> None:
    for name in ("lj_shelf_gate.rs", "sc_shelf_gate.rs"):
        destination = src / "examples" / name
        if destination.exists():
            raise RuntimeError(f"{destination} already exists")
        destination.write_bytes((harness / name).read_bytes())


def unstage_drivers(src: Path) -> None:
    for name in ("lj_shelf_gate.rs", "sc_shelf_gate.rs"):
        (src / "examples" / name).unlink(missing_ok=True)


def build_arm(src: Path, bin_dir: Path, harness: Path, target: Path, jobs_n: int) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    stage_drivers(src, harness)
    try:
        subprocess.run(
            [
                "cargo",
                "+1.88.0",
                "build",
                "--release",
                "--locked",
                "--example",
                "lj_shelf_gate",
                "--example",
                "sc_shelf_gate",
                "--example",
                "h2h_morse",
                "-j",
                str(jobs_n),
                "--target-dir",
                str(target),
            ],
            cwd=src,
            check=True,
        )
    finally:
        unstage_drivers(src)
    if porcelain(src).strip():
        raise RuntimeError(f"build left {src} dirty")
    built = target / "release" / "examples"
    for name in ("lj_shelf_gate", "sc_shelf_gate", "h2h_morse"):
        source = built / name
        if not source.is_file():
            raise RuntimeError(f"{source} was not produced")
        destination = bin_dir / name
        destination.write_bytes(source.read_bytes())
        destination.chmod(0o755)


def budget_lines(exe: Path, system: str) -> list[str]:
    proc = subprocess.run(
        [str(exe), "budget", system],
        check=True,
        capture_output=True,
        text=True,
        env=child_env(),
    )
    return proc.stdout.split()


def verify_budgets(bin_a: Path, bin_b: Path) -> None:
    for cell in CELLS:
        if cell.driver == "lj_shelf_gate":
            continue
        for directory in (bin_a, bin_b):
            fields = budget_lines(directory / cell.driver, cell.name)
            if int(fields[0]) != cell.budget or abs(Decimal(fields[1]) - cell.reference) > Decimal("1e-9"):
                raise RuntimeError(
                    f"{cell.name}: {directory.name} says {fields[0]} {fields[1]}, "
                    f"the gate declares {cell.budget} and {cell.reference}"
                )


def verify_starts(bin_a: Path, bin_b: Path) -> int:
    checked = 0
    cells = [cell for cell in CELLS if cell.driver in {"sc_shelf_gate", "h2h_morse"}]
    scratch = Path("/tmp/anneal-sc-shelf-starts")
    scratch.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        for seed in SEEDS:
            left = scratch / "a"
            right = scratch / "b"
            for directory, path in ((bin_a, left), (bin_b, right)):
                proc = subprocess.run(
                    [
                        str(directory / cell.driver),
                        "coords",
                        cell.name,
                        str(seed),
                        str(path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=child_env(),
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"{cell.name} seed {seed} coords exited {proc.returncode}: {proc.stderr[-400:]}"
                    )
            if left.read_bytes() != right.read_bytes():
                raise RuntimeError(f"{cell.name} seed {seed} starts differ")
            checked += 1
    return checked


def arm_dirs(job: dict, bin_a: Path, bin_b: Path, rev_a: str, rev_b: str, tree_a: str, tree_b: str) -> tuple[Path, str, str]:
    if job["arm"] in {"A", "A1", "A2"}:
        return bin_a, rev_a, tree_a
    if job["arm"] == "B":
        return bin_b, rev_b, tree_b
    raise RuntimeError(f"unknown arm {job['arm']}")


def cmd_build(args: argparse.Namespace) -> int:
    self_check()
    harness = Path(__file__).resolve().parent
    verify_identity(Path(args.arm_a), ARM_A_TREE, ARM_A)
    verify_identity(Path(args.arm_b), ARM_B_TREE, None)
    build_arm(Path(args.arm_a), Path(args.bin_a), harness, Path(args.target_a), args.jobs)
    build_arm(Path(args.arm_b), Path(args.bin_b), harness, Path(args.target_b), args.jobs)
    verify_identity(Path(args.arm_a), ARM_A_TREE, ARM_A)
    verify_identity(Path(args.arm_b), ARM_B_TREE, None)
    for directory in (Path(args.bin_a), Path(args.bin_b)):
        for name in ("lj_shelf_gate", "sc_shelf_gate", "h2h_morse"):
            digest = file_sha256(directory / name)
            print(f"{directory.name} {name} {digest}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    self_check()
    refuse_stray_environment()
    arm_a = Path(args.arm_a)
    arm_b = Path(args.arm_b)
    rev_a, tree_a = verify_identity(arm_a, ARM_A_TREE, ARM_A)
    rev_b, tree_b = verify_identity(arm_b, ARM_B_TREE, None)
    bin_a = Path(args.bin_a)
    bin_b = Path(args.bin_b)
    verify_budgets(bin_a, bin_b)
    checked = verify_starts(bin_a, bin_b)
    print(f"starts checked {checked}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.csv"
    existing = read_rows(rows_path)
    done = {(row["comparison"], row["system"], row["arm"], row["seed"]) for row in existing}
    machine = machine_id()
    planned = jobs()
    for index, job in enumerate(planned, start=1):
        cell: Cell = job["cell"]
        key = (job["comparison"], cell.name, job["arm"], str(job["seed"]))
        log_path = logs / log_name(job)
        if key in done:
            if not log_path.is_file() or log_path.stat().st_size == 0:
                raise RuntimeError(f"row {key} has no log")
            print(f"keep {index}/{len(planned)} {key}", flush=True)
            continue
        directory, revision, tree = arm_dirs(job, bin_a, bin_b, rev_a, rev_b, tree_a, tree_b)
        exe = directory / cell.driver
        digest = file_sha256(exe)
        print(
            f"run {index}/{len(planned)} {cell.name} {job['comparison']} {job['arm']} seed {job['seed']}",
            flush=True,
        )
        row = run_one(job, exe, revision, tree, digest, machine, log_path)
        append_row(rows_path, row)
        print(
            f"done {cell.name} {job['arm']} seed {job['seed']} hit={row['hit']} calls={row['calls']} wall={row['wall_s']}",
            flush=True,
        )
    print(f"rows {len(read_rows(rows_path))}", flush=True)
    return 0


def decimal_fmt(value: Decimal | None, places: str = "0.0000") -> str:
    if value is None:
        return ""
    return f"{value:.{places.split('.')[-1].__len__()}f}" if False else format(value, "f")


def fmt_p(value: Fraction | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def fmt_dec(value: Decimal | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def rows_of(rows: list[dict[str, str]], system: str, comparison: str, arm: str) -> list[dict[str, str]]:
    found = [
        row
        for row in rows
        if row["system"] == system and row["comparison"] == comparison and row["arm"] == arm
    ]
    found.sort(key=lambda row: int(row["seed"]))
    return found


def calls_of(group: list[dict[str, str]], hits_only: bool) -> list[Decimal]:
    values = []
    for row in group:
        if hits_only and row["hit"] != "1":
            continue
        values.append(Decimal(row["calls"]))
    return values


def outside_per(group: list[dict[str, str]]) -> list[Decimal]:
    values = []
    for row in group:
        charged = Decimal(row["charged"])
        if charged <= 0:
            raise RuntimeError(f"{row['system']} charged {charged}")
        values.append(Decimal(row["outside_s"]) / charged)
    return values


def ratio(left: list[Decimal], right: list[Decimal]) -> Decimal | None:
    if not left or not right:
        return None
    base = type7(right, Decimal("0.5"))
    if base == 0:
        return None
    return type7(left, Decimal("0.5")) / base


def iqr(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    return type7(values, Decimal("0.75")) - type7(values, Decimal("0.25"))


def median(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    return type7(values, Decimal("0.5"))


def audit(rows: list[dict[str, str]], logs: Path | None) -> list[str]:
    problems = []
    planned = jobs()
    keys = [(job["comparison"], job["cell"].name, job["arm"], str(job["seed"])) for job in planned]
    seen = [(row["comparison"], row["system"], row["arm"], row["seed"]) for row in rows]
    if seen != keys and sorted(seen) != sorted(keys):
        problems.append(f"row keys {len(seen)} do not match the schedule {len(keys)}")
    if len(seen) != len(set(seen)):
        problems.append("duplicate row")
    by_key = {(row["comparison"], row["system"], row["arm"], row["seed"]): row for row in rows}
    previous = None
    for job in planned:
        cell: Cell = job["cell"]
        key = (job["comparison"], cell.name, job["arm"], str(job["seed"]))
        row = by_key.get(key)
        if row is None:
            problems.append(f"missing {key}")
            continue
        if row["order"] != job["order"] or int(row["seed_index"]) != job["seed_index"]:
            problems.append(f"{key} order {row['order']} is not {job['order']}")
        if row["tree"] != (ARM_A_TREE if job["arm"] != "B" else ARM_B_TREE):
            problems.append(f"{key} tree {row['tree']}")
        if job["arm"] != "B" and row["code_revision"] != ARM_A:
            problems.append(f"{key} revision {row['code_revision']}")
        if row["core"] != "0" or row["affinity"] != "0":
            problems.append(f"{key} core {row['core']} affinity {row['affinity']}")
        if row["driver"] != cell.driver or int(row["budget"]) != cell.budget:
            problems.append(f"{key} driver or budget drifted")
        if logs is not None:
            path = logs / f"{job['comparison']}_{cell.name}_{job['arm']}_{job['seed']}.log"
            if not path.is_file():
                problems.append(f"{key} log missing")
            else:
                stamp = path.stat().st_mtime_ns
                if previous is not None and stamp < previous:
                    problems.append(f"{key} log is older than the previous run")
                previous = stamp
    return problems


def cell_tests(rows: list[dict[str, str]]) -> tuple[list[dict], int]:
    prepared = []
    n_tests = 0
    for cell in CELLS:
        left = rows_of(rows, cell.name, "AB", "A")
        right = rows_of(rows, cell.name, "AB", "B")
        if len(left) != 12 or len(right) != 12:
            prepared.append({"cell": cell, "incomplete": True})
            continue
        hits_a = sum(row["hit"] == "1" for row in left)
        hits_b = sum(row["hit"] == "1" for row in right)
        fisher = fisher_fewer(hits_a, 12, hits_b, 12)
        calls_a = calls_of(left, True)
        calls_b = calls_of(right, True)
        mw = mann_whitney_greater(calls_b, calls_a)
        n_tests += 1
        if mw is not None:
            n_tests += 1
        prepared.append(
            {
                "cell": cell,
                "incomplete": False,
                "hits_a": hits_a,
                "hits_b": hits_b,
                "fisher": fisher,
                "calls_a": calls_a,
                "calls_b": calls_b,
                "mw": mw,
                "outside": ratio(outside_per(right), outside_per(left)),
            }
        )
    return prepared, n_tests


def markdown(rows: list[dict[str, str]], problems: list[str], compiler: str) -> str:
    prepared, n_tests = cell_tests(rows)
    lines = [
        "# Rebased shelf paired gate",
        "",
        "Pass means B loses on no cell. A cell is a loss when the exact one-sided Fisher p for fewer hits on B, after Bonferroni, is below 0.05, or when the exact one-sided Mann-Whitney p for higher calls to a hit on B, after the same adjustment, is below 0.05, or when the median outside-potential time per charged call on B divided by the median on A is above 1.1. Bonferroni uses every Fisher test and every Mann-Whitney test that has a hit on both arms. Adjusted p is the raw p times that count, capped at 1. Decisions use the exact rational p. The table shows six decimals. The A/A block is the noise band.",
        "",
    ]
    losses = []
    if problems:
        losses.append("record")
    table = []
    for item in prepared:
        cell: Cell = item["cell"]
        if item["incomplete"]:
            losses.append(cell.name)
            table.append(f"| {cell.name} | {cell.role} | {cell.budget} |  |  |  |  |  |  |  |  | incomplete |")
            continue
        fisher_adj = adjust(item["fisher"], n_tests)
        mw_adj = None if item["mw"] is None else adjust(item["mw"], n_tests)
        reasons = []
        if fisher_adj < ALPHA:
            reasons.append("fewer hits")
        if mw_adj is not None and mw_adj < ALPHA:
            reasons.append("more calls")
        outside = item["outside"]
        if outside is None or outside > OUTSIDE_LIMIT:
            reasons.append("outside time")
        if reasons:
            losses.append(cell.name)
        med_a = median(item["calls_a"])
        med_b = median(item["calls_b"])
        table.append(
            "| {name} | {role} | {budget} | {ha}/12 | {hb}/12 | {fp} | {fpa} | {ma} | {ia} | {mb} | {ib} | {mp} | {mpa} | {ratio} | {loss} |".format(
                name=cell.name,
                role=cell.role,
                budget=cell.budget,
                ha=item["hits_a"],
                hb=item["hits_b"],
                fp=fmt_p(item["fisher"]),
                fpa=fmt_p(fisher_adj),
                ma=fmt_dec(med_a),
                ia=fmt_dec(iqr(item["calls_a"])),
                mb=fmt_dec(med_b),
                ib=fmt_dec(iqr(item["calls_b"])),
                mp=fmt_p(item["mw"]),
                mpa=fmt_p(mw_adj),
                ratio="" if outside is None else f"{outside:.6f}",
                loss=", ".join(reasons),
            )
        )
    verdict = "FAIL" if losses else "PASS"
    lines.append(f"Verdict: **{verdict}**.")
    lines.append("")
    lines.append("## Record")
    lines.append("")
    lines.append(f"- Machine id: `{machine_id()}`")
    lines.append("- Core 0, affinity 0, one run at a time.")
    lines.append(f"- Compiler: {compiler}")
    lines.append(f"- Tests in the Bonferroni family: {n_tests}.")
    lines.append(f"- Arm A commit `{ARM_A}` tree `{ARM_A_TREE}`.")
    revisions = sorted({row["code_revision"] for row in rows if row["arm"] == "B"})
    lines.append(
        f"- Arm B commit `{revisions[0] if len(revisions) == 1 else revisions}` tree `{ARM_B_TREE}`."
    )
    lines.append("- Seeds 2000 through 2011. First arm ABBA. No warm-up.")
    lines.append(
        "- Sutton-Chen rows use the in-tree reduced potential. `librgpot.so` was not present, so `h2h_sc` was not the binary."
    )
    digests = []
    for arm in ("A", "B"):
        for driver in ("lj_shelf_gate", "sc_shelf_gate", "h2h_morse"):
            values = sorted(
                {
                    row["binary_sha256"]
                    for row in rows
                    if row["driver"] == driver and (row["arm"] == "B") == (arm == "B")
                }
            )
            digests.append(f"- Arm {arm} `{driver}` {values[0] if len(values) == 1 else values}")
    lines.extend(digests)
    if problems:
        lines.append("- Record problems:")
        lines.extend(f"  - {problem}" for problem in problems)
    lines.append("")
    lines.append("## A/A band")
    lines.append("")
    lines.append("| system | hits A1 | hits A2 | median calls A1 | IQR A1 | median calls A2 | IQR A2 | median difference | outside ratio A2/A1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    band = Decimal(0)
    for name in AA_SYSTEMS:
        a1 = rows_of(rows, name, "AA", "A1")
        a2 = rows_of(rows, name, "AA", "A2")
        c1 = calls_of(a1, True)
        c2 = calls_of(a2, True)
        m1 = median(c1)
        m2 = median(c2)
        diff = None if m1 is None or m2 is None else m2 - m1
        if diff is not None and abs(diff) > band:
            band = abs(diff)
        lines.append(
            f"| {name} | {sum(row['hit'] == '1' for row in a1)}/12 | {sum(row['hit'] == '1' for row in a2)}/12 | {fmt_dec(m1)} | {fmt_dec(iqr(c1))} | {fmt_dec(m2)} | {fmt_dec(iqr(c2))} | {fmt_dec(diff)} | {'' if ratio(outside_per(a2), outside_per(a1)) is None else f'{ratio(outside_per(a2), outside_per(a1)):.6f}'} |"
        )
    lines.append("")
    lines.append(f"A/A band: {fmt_dec(band)} calls, the larger absolute median-call difference on LJ38 and LJ75.")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append("| system | role | budget | hits A | hits B | Fisher p | Fisher adjusted | median calls A | IQR A | median calls B | IQR B | MW p | MW adjusted | outside B/A | loss |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    lines.extend(table)
    lines.append("")
    if losses:
        lines.append("## Losses")
        lines.append("")
        lines.extend(f"- {name}" for name in losses)
        lines.append("")
    lines.append(f"Verdict: **{verdict}**.")
    lines.append("")
    return "\n".join(lines)


def compiler_version() -> str:
    proc = subprocess.run(
        ["rustc", "+1.88.0", "-V"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def cmd_report(args: argparse.Namespace) -> int:
    self_check()
    rows = read_rows(Path(args.out) / "rows.csv")
    logs = Path(args.logs) if args.logs else None
    problems = audit(rows, logs)
    text = markdown(rows, problems, compiler_version())
    destination = Path(args.out) / "verdict.md"
    destination.write_text(text)
    print(text)
    return 0


def cmd_self_check(_args: argparse.Namespace) -> int:
    self_check()
    print("self-check ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check").set_defaults(func=cmd_self_check)
    build = sub.add_parser("build")
    build.add_argument("--arm-a", required=True)
    build.add_argument("--arm-b", required=True)
    build.add_argument("--bin-a", required=True)
    build.add_argument("--bin-b", required=True)
    build.add_argument("--target-a", required=True)
    build.add_argument("--target-b", required=True)
    build.add_argument("--jobs", type=int, default=6)
    build.set_defaults(func=cmd_build)
    run = sub.add_parser("run")
    run.add_argument("--arm-a", required=True)
    run.add_argument("--arm-b", required=True)
    run.add_argument("--bin-a", required=True)
    run.add_argument("--bin-b", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--logs", required=True)
    run.set_defaults(func=cmd_run)
    report = sub.add_parser("report")
    report.add_argument("--out", required=True)
    report.add_argument("--logs")
    report.set_defaults(func=cmd_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
