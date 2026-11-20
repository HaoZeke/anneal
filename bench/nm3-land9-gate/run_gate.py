#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial A/B panel gate declared in README.md.

The schedule, budgets, and verdict rules are read from that file. This
process runs one pinned search at a time and appends rows. It does not
rewrite a finished row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

GATE = Path(__file__).resolve().parent
REPO = GATE.parents[1]
README = GATE / "README.md"
SEEDS = tuple(range(2000, 2012))
ALPHA = 0.05
OUTSIDE_LIMIT = 1.1
ROW_FIELDS = [
    "cell",
    "core",
    "affinity",
    "machine_id",
    "binary_sha256",
    "arm",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "inside_s",
    "outside_s",
]
SUMMARY_FIELDS = [
    "cell",
    "n",
    "budget",
    "reference",
    "hits_a",
    "hits_b",
    "evals_median_a",
    "evals_median_b",
    "fisher_p",
    "mannwhitney_p",
    "outside_per_call_median_a",
    "outside_per_call_median_b",
    "outside_ratio",
    "aa_n",
    "aa_outside_per_call_median_a1",
    "aa_outside_per_call_median_a2",
    "aa_outside_ratio",
    "verdict",
    "reasons",
]
THREAD_ONES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_DOMAIN_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "TBB_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "OPENMP_NUM_THREADS",
)
BINARY_FOR_FAMILY = {
    "lj": "h2h_gmin_proof",
    "lj-seed": "h2h_seed",
    "gupta": "h2h_metal",
    "morse": "h2h_morse",
    "sc": "h2h_sc",
    "tip4p": "h2h_seed",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repo_modules():
    sys.path.insert(0, str(REPO / "scripts"))
    gmin = load_module("nm3_gmin_panel", REPO / "scripts" / "gmin_panel" / "run_panel.py")
    paper = load_module("nm3_paper_0120", REPO / "scripts" / "paper_0120" / "run_panel.py")
    sc = load_module("nm3_run_sc", REPO / "scripts" / "metal_h2h" / "run_sc.py")
    tip = load_module("nm3_tip4p_h2h", REPO / "scripts" / "tip4p_h2h" / "run_h2h.py")
    tol = load_module("nm3_target_tol", REPO / "scripts" / "target_tol.py")
    return gmin, paper, sc, tip, tol


def fenced(text: str, label: str) -> list[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != label:
            continue
        if index == 0 or lines[index - 1].strip() != "```":
            continue
        body = []
        for item in lines[index + 1 :]:
            if item.strip() == "```":
                return body
            body.append(item.strip())
        break
    raise RuntimeError(f"protocol block {label} is missing")


def parse_protocol(path: Path = README) -> tuple[list[tuple[str, list[str]]], dict[str, dict[str, str]]]:
    text = path.read_text()
    phases = []
    for line in fenced(text, "order"):
        kind, *names = line.split()
        if kind not in ("ab", "aa") or not names:
            raise RuntimeError(f"bad order line {line}")
        phases.append((kind, names))
    cells = {}
    rows = list(csv.DictReader(fenced(text, "cells")))
    if not rows:
        raise RuntimeError("cell table is empty")
    for row in rows:
        if row["cell"] in cells:
            raise RuntimeError(f"duplicate cell {row['cell']}")
        cells[row["cell"]] = row
    return phases, cells


def type7(values: list[float], p: float = 0.5) -> float:
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


def fisher_two_sided(hits_a: int, miss_a: int, hits_b: int, miss_b: int) -> float:
    """Exact two-sided Fisher p for [[hits_a, miss_a], [hits_b, miss_b]]."""
    n = hits_a + miss_a + hits_b + miss_b
    if n == 0:
        return 1.0
    row1 = hits_a + miss_a
    row2 = hits_b + miss_b
    col1 = hits_a + hits_b

    def table_p(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = table_p(hits_a)
    total = 0.0
    for x in range(max(0, col1 - row2), min(row1, col1) + 1):
        px = table_p(x)
        if px <= observed * (1.0 + 1e-12):
            total += px
    return min(1.0, total)


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mannwhitney_worse(scores_a: list[float], scores_b: list[float]) -> float:
    """One-sided tie-corrected Mann-Whitney p, alternative that B is larger.

    Continuity correction of 0.5 toward the null mean. All-tie samples
    return 1.
    """
    n1 = len(scores_a)
    n2 = len(scores_b)
    if n1 == 0 or n2 == 0:
        raise ValueError("mann-whitney needs both arms")
    pooled = sorted(scores_a + scores_b)
    ranks: dict[float, float] = {}
    tie = 0
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[end + 1] == pooled[index]:
            end += 1
        ranks[pooled[index]] = (index + 1 + end + 1) / 2.0
        width = end - index + 1
        tie += width**3 - width
        index = end + 1
    rank_b = sum(ranks[value] for value in scores_b)
    u_b = rank_b - n2 * (n2 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if sigma2 <= 0.0:
        return 1.0
    z = (u_b - mu - 0.5) / math.sqrt(sigma2)
    return min(1.0, max(0.0, normal_sf(z)))


def miss_score(hit: str, evals: float, budget: float) -> float:
    if hit == "1":
        return evals
    return budget + 1.0


def check_cells(phases, cells, gmin, paper, sc, tip) -> None:
    ordered = []
    aa = []
    for kind, names in phases:
        if kind == "ab":
            ordered.extend(names)
        else:
            aa.extend(names)
    if len(ordered) != len(set(ordered)):
        raise RuntimeError("a cell is listed twice in the A/B schedule")
    if set(ordered) != set(cells):
        raise RuntimeError("the schedule and the cell table differ")
    if any(name not in cells for name in aa):
        raise RuntimeError("an A/A cell is not in the table")
    harness = {}
    for name, kind, _n, reference, _rho, _driver, _binary in gmin.SYSTEMS:
        harness[name] = (kind, gmin.PAPER_BUDGETS[name], reference)
    if set(gmin.PAPER_BUDGETS) != set(harness):
        raise RuntimeError("gmin panel budgets and systems differ")
    for name, budget in paper.PAPER_BUDGETS.items():
        if name not in harness or harness[name][1] != budget:
            raise RuntimeError(f"headline budget for {name} disagrees")
    if abs(paper.LJ75_REFERENCE - harness["lj75"][2]) > 1e-12:
        raise RuntimeError("LJ75 reference disagrees")
    for name, budget in sc.PAPER_BUDGETS.items():
        harness[name] = ("sc", budget, sc.REFERENCES[name])
    for n in tip.SIZES:
        if n not in tip.CAMBRIDGE:
            continue
        harness[f"tip4p{n}"] = ("tip4p", tip.BUDGET, tip.CAMBRIDGE[n])
    harness["lj75-seed"] = ("lj-seed", paper.PAPER_BUDGETS["lj75"], paper.LJ75_REFERENCE)
    if set(harness) != set(cells):
        missing = sorted(set(harness) - set(cells))
        extra = sorted(set(cells) - set(harness))
        raise RuntimeError(f"cell table drift missing={missing} extra={extra}")
    for name, row in cells.items():
        family, budget, reference = harness[name]
        if row["family"] != family:
            raise RuntimeError(f"{name} family {row['family']} != {family}")
        if int(row["budget"]) != int(budget):
            raise RuntimeError(f"{name} budget {row['budget']} != {budget}")
        if abs(float(row["reference"]) - float(reference)) > 1e-12:
            raise RuntimeError(f"{name} reference {row['reference']} != {reference}")
        if row["binary"] != BINARY_FOR_FAMILY[family]:
            raise RuntimeError(f"{name} binary {row['binary']}")
        if family == "lj-seed" and name != "lj75-seed":
            raise RuntimeError(name)
        if family == "tip4p" and not name.startswith("tip4p"):
            raise RuntimeError(name)


def thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in list(env.items()):
        if key.endswith(("_NUM_THREADS", "_MAX_THREADS", "_MAXIMUM_THREADS")):
            env[key] = "1"
    for key in THREAD_ONES:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    for key in list(env):
        if key.startswith("H2H_"):
            del env[key]
    return env


def machine_id() -> str:
    raw = Path("/etc/machine-id").read_text().strip().encode()
    if not raw:
        raise RuntimeError("machine id is empty")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def affinity_of(pid: int) -> str:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return ""
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list"):
            return line.split(":", 1)[1].strip()
    return ""


def child_threads_are_one(pid: int) -> None:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except FileNotFoundError:
        return
    env = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode()] = value.decode()
    for key in THREAD_ONES:
        if env.get(key) != "1":
            raise RuntimeError(f"{key} is {env.get(key)!r}")
    if any(key.startswith("H2H_") for key in env):
        raise RuntimeError("an H2H variable reached the search")


def parse_csv_line(text: str) -> dict[str, str]:
    header = ""
    data = ""
    for line in text.splitlines():
        if line.startswith("system,code,seed,"):
            header = line
        elif header and ",anneal," in line:
            data = line
    if not header or not data:
        raise RuntimeError(f"no result row\n{text[-2000:]}")
    return next(csv.DictReader([header, data]))


def parse_seed_line(text: str) -> dict[str, str]:
    rows = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError(f"no seed row\n{text[-2000:]}")
    fields = {}
    for part in rows[0].split()[1:]:
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def run_checked(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> tuple[str, str]:
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    affinity = affinity_of(proc.pid)
    child_threads_are_one(proc.pid)
    stdout, stderr = proc.communicate()
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(stdout + stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}\n{(stdout + stderr)[-2000:]}")
    if affinity != "0":
        pins = ""
        for line in stdout.splitlines():
            if line.startswith("system,code,seed,"):
                pins = "header"
        raise RuntimeError(f"affinity {affinity!r}")
    return stdout, affinity


def budget_matches(binary: Path, cell: dict[str, str], cwd: Path) -> None:
    if cell["family"] in ("tip4p", "lj-seed"):
        return
    proc = subprocess.run(
        [str(binary), "budget", cell["cell"] if cell["family"] != "lj-seed" else "lj75"],
        cwd=cwd,
        env=thread_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"budget {cell['cell']} failed\n{proc.stderr[-1000:]}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2 or int(lines[0]) != int(cell["budget"]):
        raise RuntimeError(f"{cell['cell']} budget {lines!r}")
    if abs(float(lines[1]) - float(cell["reference"])) > 1e-8:
        raise RuntimeError(f"{cell['cell']} reference {lines[1]!r}")


def command_for(binary: Path, cell: dict[str, str], seed: int, prefix: Path) -> list[str]:
    if cell["family"] in ("tip4p", "lj-seed"):
        if cell["family"] == "lj-seed":
            kind, number, rho = "lj", "75", "0"
        else:
            kind, number, rho = "tip4p", cell["cell"].removeprefix("tip4p"), "0"
        stop = repr(float(cell["reference"]) + 1.0e-3)
        return [
            "taskset",
            "-c",
            "0",
            str(binary),
            kind,
            number,
            rho,
            cell["budget"],
            str(seed),
            "recommended",
            stop,
            str(prefix),
        ]
    return ["taskset", "-c", "0", str(binary), "run", cell["cell"], str(seed)]


def result_of(cell: dict[str, str], seed: int, text: str, tol) -> dict[str, str]:
    if cell["family"] in ("tip4p", "lj-seed"):
        parsed = parse_seed_line(text)
        expect = "lj75" if cell["family"] == "lj-seed" else cell["cell"]
        best = parsed["best"]
        charged = int(parsed["charged"])
        wall = float(parsed["wall_s"])
        inside = float(parsed["pot_s"])
        outside = float(parsed["outside_s"])
        binary_hit = parsed["hit"] in ("1", "true", "True")
        binary_evals = int(parsed["evals_to_hit"])
        if parsed["system"] != expect or int(parsed["seed"]) != seed or int(parsed["budget"]) != int(cell["budget"]):
            raise RuntimeError(f"seed row {parsed}")
    else:
        parsed = parse_csv_line(text)
        best = parsed["best"]
        charged = int(float(parsed["charged"]))
        wall = float(parsed["wall_s"])
        inside = float(parsed["pot_s"])
        outside = float(parsed["outside_s"]) if parsed.get("outside_s") else max(wall - inside, 0.0)
        binary_hit = parsed["hit"].strip().lower() in ("1", "true")
        binary_evals = int(float(parsed["evals_to_hit"]))
        if parsed["system"] != cell["cell"] or int(parsed["seed"]) != seed or int(float(parsed["budget"])) != int(cell["budget"]):
            raise RuntimeError(f"row {parsed}")
    if charged < 1:
        raise RuntimeError(f"charged {charged}")
    hit = tol.energy_hits(float(best), float(cell["reference"]), tol.DEFAULT_HIT_TOL)
    if hit != binary_hit:
        raise RuntimeError(f"hit disagreement helper={hit} binary={binary_hit} best={best}")
    evals = binary_evals if hit else int(cell["budget"])
    return {
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(charged),
        "best": best,
        "wall_s": f"{wall:.6f}",
        "inside_s": f"{inside:.6f}",
        "outside_s": f"{outside:.6f}",
    }


def arm_side(arm: str) -> str:
    if arm in ("A", "A1", "A2"):
        return "A"
    if arm == "B":
        return "B"
    raise RuntimeError(arm)


def run_schedule(args: argparse.Namespace) -> None:
    phases, cells = parse_protocol()
    gmin, paper, sc, tip, tol = repo_modules()
    check_cells(phases, cells, gmin, paper, sc, tip)
    env = thread_env()
    digest = machine_id()
    binaries = {
        "A": {name: args.a_bin / name for name in set(BINARY_FOR_FAMILY.values())},
        "B": {name: args.b_bin / name for name in set(BINARY_FOR_FAMILY.values())},
    }
    trees = {"A": args.a_tree, "B": args.b_tree}
    shas = {}
    for side, names in binaries.items():
        for name, path in names.items():
            if not path.is_file():
                raise SystemExit(f"missing binary {name}")
            shas[(side, name)] = sha256_file(path)
            if side == "B" and shas[(side, name)] == shas[("A", name)]:
                raise SystemExit(f"{name} is the same binary on both arms")
    for cell in cells.values():
        for side in ("A", "B"):
            budget_matches(binaries[side][cell["binary"]], cell, trees[side])
    done_marker = False
    for phase_index, (kind, names) in enumerate(phases):
        for name in names:
            cell = cells[name]
            rows_path = GATE / name / "rows.csv"
            have = {(row["arm"], int(row["seed"])) for row in read_rows(rows_path)}
            for seed in SEEDS:
                if kind == "ab":
                    order = ("A", "B") if seed % 2 == 0 else ("B", "A")
                else:
                    order = ("A1", "A2") if seed % 2 == 0 else ("A2", "A1")
                for arm in order:
                    if (arm, seed) in have:
                        continue
                    side = arm_side(arm)
                    binary = binaries[side][cell["binary"]]
                    log = args.logs / name / f"{arm}-{seed}.log"
                    with tempfile.TemporaryDirectory() as scratch:
                        prefix = Path(scratch) / "run"
                        command = command_for(binary, cell, seed, prefix)
                        stdout, affinity = run_checked(command, trees[side], env, log)
                    got = result_of(cell, seed, stdout, tol)
                    row = {
                        "cell": name,
                        "core": "0",
                        "affinity": affinity,
                        "machine_id": digest,
                        "binary_sha256": shas[(side, cell["binary"])],
                        "arm": arm,
                        "seed": str(seed),
                        "budget": cell["budget"],
                        **got,
                    }
                    append_row(rows_path, row)
                    print(
                        f"ROW {name} {arm} {seed} hit={row['hit']} evals={row['evals_to_hit']} wall={row['wall_s']}",
                        flush=True,
                    )
            summarise_cell(name, cells[name])
        if phase_index == 1 and not done_marker:
            print("PRIORITY_DONE", flush=True)
            done_marker = True
    write_panel_summary(cells)


def numeric(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def ratio(left: list[float], right: list[float]) -> float:
    base = type7(left)
    other = type7(right)
    if base == 0.0:
        return 1.0 if other == 0.0 else math.inf
    return other / base


def per_call(rows: list[dict[str, str]]) -> list[float]:
    values = []
    for row in rows:
        charged = float(row["charged"])
        if charged <= 0.0:
            raise RuntimeError("charged is zero")
        values.append(float(row["outside_s"]) / charged)
    return values


def check_identity(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    machines = {row["machine_id"] for row in rows}
    if len(machines) != 1 or len(next(iter(machines))) != 64:
        raise RuntimeError("machine id is not one digest")
    int(next(iter(machines)), 16)
    for row in rows:
        if row["core"] != "0" or row["affinity"] != "0":
            raise RuntimeError(f"pin {row['cell']} {row['arm']} {row['seed']}")
        if int(row["seed"]) not in SEEDS:
            raise RuntimeError(f"unexpected seed {row['seed']}")
    by_arm: dict[str, set[str]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], set()).add(row["binary_sha256"])
    for arm, digests in by_arm.items():
        if len(digests) != 1:
            raise RuntimeError(f"{arm} has more than one binary")
    a_arms = [arm for arm in by_arm if arm in ("A", "A1", "A2")]
    if a_arms:
        digests = set.union(*(by_arm[arm] for arm in a_arms))
        if len(digests) != 1:
            raise RuntimeError("arm A binaries disagree")
        if "B" in by_arm and by_arm["B"] & digests:
            raise RuntimeError("arm B matches arm A")


def pair_rows(rows: list[dict[str, str]], left: str, right: str) -> tuple[list[dict[str, str]], list[dict[str, str]]] | None:
    a = {int(row["seed"]): row for row in rows if row["arm"] == left}
    b = {int(row["seed"]): row for row in rows if row["arm"] == right}
    if set(a) != set(SEEDS) or set(b) != set(SEEDS):
        return None
    return [a[seed] for seed in SEEDS], [b[seed] for seed in SEEDS]


def summarise_cell(name: str, cell: dict[str, str]) -> dict[str, str]:
    rows = read_rows(GATE / name / "rows.csv")
    check_identity(rows)
    blank = {key: "" for key in SUMMARY_FIELDS}
    blank.update({"cell": name, "budget": cell["budget"], "reference": cell["reference"], "verdict": "INCOMPLETE", "reasons": ""})
    paired = pair_rows(rows, "A", "B")
    if paired is None:
        write_csv(GATE / name / "summary.csv", SUMMARY_FIELDS, [blank])
        return blank
    arm_a, arm_b = paired
    hits_a = sum(row["hit"] == "1" for row in arm_a)
    hits_b = sum(row["hit"] == "1" for row in arm_b)
    evals_a = numeric(arm_a, "evals_to_hit")
    evals_b = numeric(arm_b, "evals_to_hit")
    fisher_p = fisher_two_sided(hits_a, len(arm_a) - hits_a, hits_b, len(arm_b) - hits_b)
    scores_a = [miss_score(row["hit"], float(row["evals_to_hit"]), float(row["budget"])) for row in arm_a]
    scores_b = [miss_score(row["hit"], float(row["evals_to_hit"]), float(row["budget"])) for row in arm_b]
    mw_p = mannwhitney_worse(scores_a, scores_b)
    outside_a = per_call(arm_a)
    outside_b = per_call(arm_b)
    outside = ratio(outside_a, outside_b)
    reasons = []
    if hits_b < hits_a and fisher_p < ALPHA:
        reasons.append("fisher")
    if mw_p < ALPHA:
        reasons.append("mannwhitney")
    if outside > OUTSIDE_LIMIT:
        reasons.append("outside")
    aa = pair_rows(rows, "A1", "A2")
    aa_n = ""
    aa_a = ""
    aa_b = ""
    aa_ratio = ""
    if aa is not None:
        aa_n = str(len(aa[0]))
        left, right = per_call(aa[0]), per_call(aa[1])
        aa_a = f"{type7(left):.8g}"
        aa_b = f"{type7(right):.8g}"
        aa_ratio = f"{ratio(left, right):.8g}"
    summary = {
        "cell": name,
        "n": str(len(arm_a)),
        "budget": cell["budget"],
        "reference": cell["reference"],
        "hits_a": str(hits_a),
        "hits_b": str(hits_b),
        "evals_median_a": f"{type7(evals_a):.1f}",
        "evals_median_b": f"{type7(evals_b):.1f}",
        "fisher_p": f"{fisher_p:.8g}",
        "mannwhitney_p": f"{mw_p:.8g}",
        "outside_per_call_median_a": f"{type7(outside_a):.8g}",
        "outside_per_call_median_b": f"{type7(outside_b):.8g}",
        "outside_ratio": f"{outside:.8g}",
        "aa_n": aa_n,
        "aa_outside_per_call_median_a1": aa_a,
        "aa_outside_per_call_median_a2": aa_b,
        "aa_outside_ratio": aa_ratio,
        "verdict": "FAIL" if reasons else "PASS",
        "reasons": " ".join(reasons),
    }
    write_csv(GATE / name / "summary.csv", SUMMARY_FIELDS, [summary])
    print(
        f"CELL {name} {summary['verdict']} hits {hits_a}/{hits_b} evals {summary['evals_median_a']}/{summary['evals_median_b']} "
        f"fisher {summary['fisher_p']} mw {summary['mannwhitney_p']} outside {summary['outside_ratio']} aa {summary['aa_outside_ratio']}",
        flush=True,
    )
    return summary


def write_panel_summary(cells: dict[str, dict[str, str]]) -> None:
    phases, _cells = parse_protocol()
    order = []
    for kind, names in phases:
        if kind == "ab":
            order.extend(names)
    rows = []
    for name in order:
        path = GATE / name / "summary.csv"
        if path.is_file():
            found = read_rows(path)
            if found:
                rows.append(found[-1])
                continue
        rows.append(summarise_cell(name, cells[name]))
    write_csv(GATE / "summary.csv", SUMMARY_FIELDS, rows)


def self_test() -> None:
    phases, cells = parse_protocol()
    gmin, paper, sc, tip, tol = repo_modules()
    check_cells(phases, cells, gmin, paper, sc, tip)
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12 or abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7")
    twelve = [float(i) for i in range(1, 13)]
    if abs(type7(twelve) - 6.5) > 1e-12:
        raise SystemExit("type 7 n=12")
    if abs(fisher_two_sided(2, 2, 0, 4) - 12 / 28) > 1e-12:
        raise SystemExit("fisher")
    if abs(fisher_two_sided(12, 0, 7, 5) - 0.037267080745341616) > 1e-12:
        raise SystemExit(f"fisher panel {fisher_two_sided(12, 0, 7, 5)}")
    if mannwhitney_worse([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) != 1.0:
        raise SystemExit("tied mann-whitney")
    worse = mannwhitney_worse([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
    if not worse < ALPHA:
        raise SystemExit(f"worse mann-whitney {worse}")
    better = mannwhitney_worse([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0])
    if not better > 0.5:
        raise SystemExit(f"better mann-whitney {better}")
    if not tol.energy_hits(-1.0, -1.001, 0.001):
        raise SystemExit("hit edge")
    if tol.energy_hits(-1.0, -1.001 - 1e-6, 0.001):
        raise SystemExit("miss edge")
    if miss_score("0", 10, 10) <= miss_score("1", 10, 10):
        raise SystemExit("miss rank")
    if SEEDS != tuple(range(2000, 2012)):
        raise SystemExit("seeds")
    print("ok")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--a-bin", type=Path)
    parser.add_argument("--b-bin", type=Path)
    parser.add_argument("--a-tree", type=Path)
    parser.add_argument("--b-tree", type=Path)
    parser.add_argument("--logs", type=Path)
    args = parser.parse_args(argv[1:])
    if args.self_test:
        self_test()
        return 0
    if args.summarize:
        _phases, cells = parse_protocol()
        write_panel_summary(cells)
        return 0
    if args.run:
        if not args.a_bin or not args.b_bin or not args.a_tree or not args.b_tree or not args.logs:
            raise SystemExit("a run needs both binary directories, both trees, and a log directory")
        run_schedule(args)
        return 0
    raise SystemExit("choose --run, --summarize, or --self-test")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
