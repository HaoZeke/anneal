#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial A/B/S panel gate declared in README.md.

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
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GATE = Path(__file__).resolve().parent
REPO = GATE.parents[1]
README = GATE / "README.md"
PROBE_SOURCE = GATE / "probe.c"
SEEDS = tuple(range(2000, 2012))
ALPHA = 0.05
CONFIDENCE = 0.95
OUTSIDE_LIMIT = 1.1
RSS_FLOOR = 1.1
PROBE_PERIOD_US = 1000
PROBE_RSS_EVERY = 1
BAND_ARMS = ("A1", "A2")
ROW_FIELDS = [
    "cell",
    "core",
    "affinity",
    "machine_id",
    "binary_sha256",
    "probe_sha256",
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
    "rss_anon_peak_kb",
    "vm_hwm_kb",
    "rss_reads",
    "probe_ticks",
    "vector_ticks",
    "vector_ranges",
    "parity_digest",
]
SUMMARY_FIELDS = [
    "cell",
    "n",
    "budget",
    "reference",
    "hits_a",
    "hits_b",
    "hits_s",
    "evals_median_a",
    "evals_median_b",
    "fisher_p",
    "mannwhitney_p",
    "outside_per_call_median_a",
    "outside_per_call_median_b",
    "outside_ratio_unpaired",
    "outside_ratio",
    "outside_ratio_lo",
    "outside_ratio_hi",
    "rss_peak_median_a_kb",
    "rss_peak_median_b_kb",
    "rss_ratio",
    "rss_ratio_lo",
    "rss_limit",
    "aa_n",
    "aa_outside_ratio",
    "aa_outside_ratio_lo",
    "aa_outside_ratio_hi",
    "aa_rss_ratio",
    "aa_rss_ratio_hi",
    "s_outside_ratio",
    "vector_ticks_a",
    "vector_ticks_b",
    "vector_ticks_s",
    "vector_rows_b",
    "vector_rows_s",
    "vector_ranges_a",
    "vector_ranges_b",
    "vector_ranges_s",
    "parity",
    "parity_mismatches",
    "verdict",
    "reasons",
]
DIAG_FIELDS = [
    "cell",
    "arm",
    "seed",
    "move",
    "proposed",
    "accepted",
    "improved",
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
SIDES = ("A", "A2", "B", "S")
TIMING_COLUMNS = {
    "wall_s",
    "wall_seconds",
    "pot_s",
    "outside_s",
    "pot_per_eval_s",
    "outside_per_eval_s",
    "head",
    "tree",
    "cpu",
    "pins",
}
TARGET_FEATURE = re.compile(r'#\[target_feature\(enable\s*=\s*"[^"]+"\)\]')
FN_NAME = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")
SYMBOL_HASH = re.compile(r"::h[0-9a-f]{16}$")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repo_modules():
    sys.path.insert(0, str(REPO / "scripts"))
    gmin = load_module("g153_gmin_panel", REPO / "scripts" / "gmin_panel" / "run_panel.py")
    paper = load_module("g153_paper_0120", REPO / "scripts" / "paper_0120" / "run_panel.py")
    sc = load_module("g153_run_sc", REPO / "scripts" / "metal_h2h" / "run_sc.py")
    tip = load_module("g153_tip4p_h2h", REPO / "scripts" / "tip4p_h2h" / "run_h2h.py")
    tol = load_module("g153_target_tol", REPO / "scripts" / "target_tol.py")
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
        if kind not in ("abs", "aa") or not names:
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


def band_cells(phases) -> list[str]:
    return [name for kind, names in phases if kind == "aa" for name in names]


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


def binomial_cdf(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0**n


def median_rank(n: int, tail: float) -> int:
    """Largest 1-based rank r with P(Binomial(n, 1/2) <= r - 1) <= tail.

    The r-th order statistic is then a lower bound for the median with
    coverage at least 1 - tail, and the (n + 1 - r)-th is the matching
    upper bound.
    """
    rank = 0
    for r in range(1, n + 1):
        if binomial_cdf(r - 1, n) <= tail:
            rank = r
        else:
            break
    if rank < 1:
        raise ValueError(f"n={n} is too small for tail {tail}")
    return rank


def median_ci(values: list[float], confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Two-sided distribution-free interval for the median."""
    ordered = sorted(values)
    rank = median_rank(len(ordered), (1.0 - confidence) / 2.0)
    return ordered[rank - 1], ordered[len(ordered) - rank]


def median_lower(values: list[float], confidence: float = CONFIDENCE) -> float:
    ordered = sorted(values)
    return ordered[median_rank(len(ordered), 1.0 - confidence) - 1]


def median_upper(values: list[float], confidence: float = CONFIDENCE) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) - median_rank(len(ordered), 1.0 - confidence)]


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
        if kind == "abs":
            ordered.extend(names)
        else:
            aa.extend(names)
    if len(ordered) != len(set(ordered)):
        raise RuntimeError("a cell is listed twice in the A/B/S schedule")
    if set(ordered) != set(cells):
        raise RuntimeError("the schedule and the cell table differ")
    if any(name not in cells for name in aa) or len(aa) != len(set(aa)):
        raise RuntimeError("an A/A cell is not in the table or is listed twice")
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
        if key.startswith(("H2H_", "GATE_PROBE_")) or key == "LD_PRELOAD":
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


def parse_probe(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("probe ")]
    if len(lines) != 1:
        raise RuntimeError(f"probe wrote {len(lines)} lines")
    fields = dict(part.split("=", 1) for part in lines[0].split()[1:])
    if int(fields["rss_anon_peak_kb"]) <= 0 or int(fields["rss_reads"]) < 2:
        raise RuntimeError(f"probe read no RssAnon {fields}")
    return fields


def deterministic_lines(text: str) -> list[str]:
    """Output lines with the timing and checkout fields taken out.

    What is left is the trajectory record that a scalar-equivalent build
    must reproduce token for token.
    """
    kept = []
    header: list[str] = []
    for line in text.splitlines():
        if line.startswith("system,code,seed,"):
            header = line.split(",")
            kept.append(line)
            continue
        if header and ",anneal," in line:
            values = next(csv.reader([line]))
            pairs = [
                f"{key}={value}"
                for key, value in zip(header, values)
                if key not in TIMING_COLUMNS
            ]
            kept.append("row " + ",".join(pairs))
            continue
        tokens = line.split()
        if tokens and tokens[0].endswith(":") and tokens[0][:-1] in TIMING_COLUMNS:
            kept.append(tokens[0])
            continue
        tokens = [
            token
            for token in tokens
            if not ("=" in token and token.split("=", 1)[0] in TIMING_COLUMNS)
        ]
        kept.append(" ".join(tokens))
    return kept


def parity_digest(text: str, extra: list[Path]) -> str:
    digest = hashlib.sha256()
    for line in deterministic_lines(text):
        digest.update(line.encode())
        digest.update(b"\n")
    for path in extra:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<absent>")
        digest.update(b"\0")
    return digest.hexdigest()


def run_checked(
    command: list[str], cwd: Path, env: dict[str, str], log: Path, scratch: Path
) -> tuple[str, str, str]:
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    affinity = ""
    threads_ok = False
    try:
        # The mask is the parent's until taskset applies it. Keep the latest
        # reading and accept it only after it settles on CPU 0.
        for _ in range(200):
            current = affinity_of(proc.pid)
            if current:
                affinity = current
            if not threads_ok:
                try:
                    child_threads_are_one(proc.pid)
                    threads_ok = True
                except RuntimeError:
                    if proc.poll() is not None:
                        raise
            if threads_ok and affinity == "0":
                break
            if proc.poll() is not None:
                break
            time.sleep(0.005)
        if not threads_ok:
            child_threads_are_one(proc.pid)
        stdout, stderr = proc.communicate()
    except Exception:
        proc.kill()
        proc.communicate()
        raise
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text((stdout + stderr).replace(str(scratch), "<scratch>"))
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}\n{(stdout + stderr)[-2000:]}")
    if affinity != "0":
        raise RuntimeError(f"affinity {affinity!r}")
    return stdout, stderr, affinity


def move_counts(cell: str, arm: str, seed: int, text: str) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if (
            len(fields) == 8
            and fields[0] == "move"
            and fields[2] == "proposed"
            and fields[4] == "accepted"
            and fields[6] == "improved"
        ):
            rows.append(
                {
                    "cell": cell,
                    "arm": arm,
                    "seed": str(seed),
                    "move": fields[1],
                    "proposed": fields[3],
                    "accepted": fields[5],
                    "improved": fields[7],
                }
            )
    return rows


def append_move_counts(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    existing = read_rows(path)
    keys = {
        (row["cell"], row["arm"], row["seed"], row["move"])
        for row in existing
    }
    fresh = [
        row
        for row in rows
        if (row["cell"], row["arm"], row["seed"], row["move"]) not in keys
    ]
    if not fresh:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAG_FIELDS, lineterminator="\n")
        if not existing:
            writer.writeheader()
        writer.writerows(fresh)
        handle.flush()
        os.fsync(handle.fileno())


def budget_matches(binary: Path, cell: dict[str, str], cwd: Path, env: dict[str, str]) -> None:
    if cell["family"] in ("tip4p", "lj-seed"):
        return
    proc = subprocess.run(
        [str(binary), "budget", cell["cell"]],
        cwd=cwd,
        env=env,
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


def seed_outputs(cell: dict[str, str], prefix: Path) -> list[Path]:
    """Files a seed-driver run leaves that the parity digest covers."""
    if cell["family"] not in ("tip4p", "lj-seed"):
        return []
    return [Path(f"{prefix}.trace.csv"), Path(f"{prefix}.xyz")]


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
    if arm in ("A", "A1"):
        return "A"
    if arm in ("A2", "B", "S"):
        return arm
    raise RuntimeError(arm)


def arm_order(kind: str, seed: int) -> tuple[str, ...]:
    if kind == "abs":
        return ("A", "B", "S") if seed % 2 == 0 else ("S", "B", "A")
    return ("A1", "A2") if seed % 2 == 0 else ("A2", "A1")


def vector_functions(tree: Path) -> set[str]:
    """Names of the functions the tree's own sources mark #[target_feature]."""
    names = set()
    for path in sorted((tree / "src").rglob("*.rs")):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            if not TARGET_FEATURE.search(line):
                continue
            for follow in lines[index : index + 12]:
                found = FN_NAME.search(follow)
                if found:
                    names.add(found.group(1))
                    break
            else:
                raise RuntimeError(f"no function after target_feature in {path.name}:{index + 1}")
    return names


def vector_ranges(binary: Path, names: set[str]) -> list[tuple[int, int, str]]:
    """Text symbols of anneal_core whose function is in names."""
    proc = subprocess.run(
        ["nm", "--demangle", "--defined-only", "--print-size", str(binary)],
        capture_output=True,
        text=True,
        check=True,
    )
    ranges = []
    for line in proc.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) != 4 or parts[2] not in ("t", "T", "w", "W"):
            continue
        symbol = SYMBOL_HASH.sub("", parts[3])
        if "anneal_core" not in symbol:
            continue
        leaf = re.sub(r"::<.*>$", "", symbol).rsplit("::", 1)[-1]
        if leaf in names:
            ranges.append((int(parts[0], 16), int(parts[1], 16), symbol))
    return sorted(ranges)


def build_probe(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    library = out / "probe.so"
    subprocess.run(
        ["cc", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror", "-o", str(library), str(PROBE_SOURCE)],
        check=True,
    )
    return library


def run_schedule(args: argparse.Namespace) -> None:
    phases, cells = parse_protocol()
    gmin, paper, sc, tip, tol = repo_modules()
    check_cells(phases, cells, gmin, paper, sc, tip)
    env = thread_env()
    env["RGPOT_SC_LIB"] = str(args.sc_lib.resolve())
    digest = machine_id()
    bins = {"A": args.a_bin, "A2": args.a2_bin, "B": args.b_bin, "S": args.s_bin}
    trees = {"A": args.a_tree, "A2": args.a2_tree, "B": args.b_tree, "S": args.s_tree}
    binaries = {
        side: {name: (bins[side] / name).resolve() for name in set(BINARY_FOR_FAMILY.values())}
        for side in SIDES
    }
    probe_dir = args.logs / "probe"
    probe = build_probe(probe_dir)
    probe_sha = sha256_file(probe)
    shas = {}
    range_files = {}
    inventory = []
    for side, names in binaries.items():
        functions = vector_functions(trees[side])
        for name, path in names.items():
            if not path.is_file():
                raise SystemExit(f"missing binary {side} {name}")
            shas[(side, name)] = sha256_file(path)
            found = vector_ranges(path, functions)
            range_file = probe_dir / f"ranges-{side}-{name}.txt"
            range_file.write_text("".join(f"{start:x} {size:x}\n" for start, size, _ in found))
            range_files[(side, name)] = range_file
            inventory.extend(f"{side} {name} {size:x} {symbol}" for _start, size, symbol in found)
    (probe_dir / "vector-symbols.txt").write_text("\n".join(inventory) + "\n")
    for name in set(BINARY_FOR_FAMILY.values()):
        if shas[("B", name)] == shas[("A", name)]:
            raise SystemExit(f"{name} is the same binary on A and B")
        if shas[("S", name)] == shas[("B", name)]:
            raise SystemExit(f"{name} is the same binary on B and S")
    for cell in cells.values():
        for side in SIDES:
            budget_matches(binaries[side][cell["binary"]], cell, trees[side], env)
    for phase_index, (kind, names) in enumerate(phases):
        for name in names:
            cell = cells[name]
            rows_path = GATE / name / "rows.csv"
            have = {(row["arm"], int(row["seed"])) for row in read_rows(rows_path)}
            for seed in SEEDS:
                for arm in arm_order(kind, seed):
                    if (arm, seed) in have:
                        continue
                    side = arm_side(arm)
                    binary = binaries[side][cell["binary"]]
                    log = args.logs / "runs" / name / f"{arm}-{seed}.log"
                    with tempfile.TemporaryDirectory() as scratch_dir:
                        scratch = Path(scratch_dir)
                        prefix = scratch / "run"
                        probe_out = scratch / "probe.txt"
                        run_env = dict(env)
                        run_env.update(
                            {
                                "LD_PRELOAD": str(probe),
                                "GATE_PROBE_EXE": str(binary),
                                "GATE_PROBE_OUT": str(probe_out),
                                "GATE_PROBE_RANGES": str(range_files[(side, cell["binary"])]),
                                "GATE_PROBE_PERIOD_US": str(PROBE_PERIOD_US),
                                "GATE_PROBE_RSS_EVERY": str(PROBE_RSS_EVERY),
                            }
                        )
                        command = command_for(binary, cell, seed, prefix)
                        stdout, stderr, affinity = run_checked(
                            command, trees[side], run_env, log, scratch
                        )
                        probe_text = probe_out.read_text() if probe_out.is_file() else ""
                        with log.open("a") as handle:
                            handle.write(probe_text)
                        probed = parse_probe(probe_text)
                        parity = parity_digest(stdout + stderr, seed_outputs(cell, prefix))
                    got = result_of(cell, seed, stdout, tol)
                    if name == "sc96-80":
                        counts = move_counts(name, arm, seed, stdout + stderr)
                        if not counts:
                            raise RuntimeError("sc96-80 did not print move counts")
                        append_move_counts(
                            GATE / "diag" / "sc96-80-moves.csv", counts
                        )
                    row = {
                        "cell": name,
                        "core": "0",
                        "affinity": affinity,
                        "machine_id": digest,
                        "binary_sha256": shas[(side, cell["binary"])],
                        "probe_sha256": probe_sha,
                        "arm": arm,
                        "seed": str(seed),
                        "budget": cell["budget"],
                        **got,
                        "rss_anon_peak_kb": probed["rss_anon_peak_kb"],
                        "vm_hwm_kb": probed["vm_hwm_kb"],
                        "rss_reads": probed["rss_reads"],
                        "probe_ticks": probed["ticks"],
                        "vector_ticks": probed["vector_ticks"],
                        "vector_ranges": probed["vector_ranges"],
                        "parity_digest": parity,
                    }
                    append_row(rows_path, row)
                    print(
                        f"ROW {name} {arm} {seed} hit={row['hit']} evals={row['evals_to_hit']} "
                        f"wall={row['wall_s']} rss={row['rss_anon_peak_kb']} vec={row['vector_ticks']}",
                        flush=True,
                    )
            summarise_cell(name, cells[name], phases)
        print(f"PHASE_DONE {phase_index}", flush=True)
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


def paired_ratios(base: list[float], other: list[float]) -> list[float]:
    values = []
    for left, right in zip(base, other):
        if left <= 0.0:
            values.append(1.0 if right <= 0.0 else math.inf)
        else:
            values.append(right / left)
    return values


def check_identity(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    machines = {row["machine_id"] for row in rows}
    if len(machines) != 1 or len(next(iter(machines))) != 64:
        raise RuntimeError("machine id is not one digest")
    int(next(iter(machines)), 16)
    if len({row["probe_sha256"] for row in rows}) != 1:
        raise RuntimeError("more than one probe build")
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
    if "A" in by_arm and "A1" in by_arm and by_arm["A"] != by_arm["A1"]:
        raise RuntimeError("arm A1 is not the arm A binary")
    a_digests = by_arm.get("A", set()) | by_arm.get("A1", set())
    if "B" in by_arm and by_arm["B"] & a_digests:
        raise RuntimeError("arm B matches arm A")
    if "S" in by_arm and "B" in by_arm and by_arm["S"] & by_arm["B"]:
        raise RuntimeError("arm S matches arm B")


def arm_rows(rows: list[dict[str, str]], arm: str) -> list[dict[str, str]] | None:
    found = {int(row["seed"]): row for row in rows if row["arm"] == arm}
    if set(found) != set(SEEDS) or sum(row["arm"] == arm for row in rows) != len(SEEDS):
        return None
    return [found[seed] for seed in SEEDS]


def fmt(value: float) -> str:
    return f"{value:.8g}"


def band_rss_upper(phases) -> float | None:
    """Largest one-sided upper bound of the A2/A1 RssAnon ratio median."""
    uppers = []
    for name in band_cells(phases):
        rows = read_rows(GATE / name / "rows.csv")
        left, right = arm_rows(rows, "A1"), arm_rows(rows, "A2")
        if left is None or right is None:
            return None
        uppers.append(median_upper(paired_ratios(numeric(left, "rss_anon_peak_kb"), numeric(right, "rss_anon_peak_kb"))))
    return max(uppers) if uppers else None


def parity_check(arm_b: list[dict[str, str]], arm_s: list[dict[str, str]]) -> list[str]:
    mismatches = []
    for row_b, row_s in zip(arm_b, arm_s):
        fields = [
            key
            for key in ("best", "charged", "evals_to_hit", "hit", "parity_digest")
            if row_b[key] != row_s[key]
        ]
        if fields:
            mismatches.append(f"{row_b['seed']}:{'+'.join(fields)}")
    return mismatches


def summarise_cell(name: str, cell: dict[str, str], phases) -> dict[str, str]:
    rows = read_rows(GATE / name / "rows.csv")
    check_identity(rows)
    blank = {key: "" for key in SUMMARY_FIELDS}
    blank.update({"cell": name, "budget": cell["budget"], "reference": cell["reference"], "verdict": "INCOMPLETE", "reasons": ""})
    arm_a, arm_b, arm_s = arm_rows(rows, "A"), arm_rows(rows, "B"), arm_rows(rows, "S")
    rss_band = band_rss_upper(phases)
    if arm_a is None or arm_b is None or arm_s is None or rss_band is None:
        write_csv(GATE / name / "summary.csv", SUMMARY_FIELDS, [blank])
        return blank
    hits_a = sum(row["hit"] == "1" for row in arm_a)
    hits_b = sum(row["hit"] == "1" for row in arm_b)
    hits_s = sum(row["hit"] == "1" for row in arm_s)
    evals_a = numeric(arm_a, "evals_to_hit")
    evals_b = numeric(arm_b, "evals_to_hit")
    fisher_p = fisher_two_sided(hits_a, len(arm_a) - hits_a, hits_b, len(arm_b) - hits_b)
    scores_a = [miss_score(row["hit"], float(row["evals_to_hit"]), float(row["budget"])) for row in arm_a]
    scores_b = [miss_score(row["hit"], float(row["evals_to_hit"]), float(row["budget"])) for row in arm_b]
    mw_p = mannwhitney_worse(scores_a, scores_b)
    outside_a = per_call(arm_a)
    outside_b = per_call(arm_b)
    outside_pairs = paired_ratios(outside_a, outside_b)
    outside_lo, outside_hi = median_ci(outside_pairs)
    rss_a = numeric(arm_a, "rss_anon_peak_kb")
    rss_b = numeric(arm_b, "rss_anon_peak_kb")
    rss_pairs = paired_ratios(rss_a, rss_b)
    rss_lo = median_lower(rss_pairs)
    rss_limit = max(RSS_FLOOR, rss_band)
    mismatches = parity_check(arm_b, arm_s)
    reasons = []
    if hits_b < hits_a and fisher_p < ALPHA:
        reasons.append("fisher")
    if mw_p < ALPHA:
        reasons.append("mannwhitney")
    if outside_hi > OUTSIDE_LIMIT:
        reasons.append("outside")
    if rss_lo > rss_limit:
        reasons.append("rss")
    if mismatches:
        reasons.append("parity")
    aa_fields = {key: "" for key in ("aa_n", "aa_outside_ratio", "aa_outside_ratio_lo", "aa_outside_ratio_hi", "aa_rss_ratio", "aa_rss_ratio_hi")}
    band_a1, band_a2 = arm_rows(rows, "A1"), arm_rows(rows, "A2")
    if band_a1 is not None and band_a2 is not None:
        aa_outside = paired_ratios(per_call(band_a1), per_call(band_a2))
        aa_lo, aa_hi = median_ci(aa_outside)
        aa_rss = paired_ratios(numeric(band_a1, "rss_anon_peak_kb"), numeric(band_a2, "rss_anon_peak_kb"))
        aa_fields = {
            "aa_n": str(len(band_a1)),
            "aa_outside_ratio": fmt(type7(aa_outside)),
            "aa_outside_ratio_lo": fmt(aa_lo),
            "aa_outside_ratio_hi": fmt(aa_hi),
            "aa_rss_ratio": fmt(type7(aa_rss)),
            "aa_rss_ratio_hi": fmt(median_upper(aa_rss)),
        }
    vector = {arm: [int(row["vector_ticks"]) for row in group] for arm, group in (("A", arm_a), ("B", arm_b), ("S", arm_s))}
    ranges = {arm: sorted({row["vector_ranges"] for row in group}) for arm, group in (("A", arm_a), ("B", arm_b), ("S", arm_s))}
    summary = {
        "cell": name,
        "n": str(len(arm_a)),
        "budget": cell["budget"],
        "reference": cell["reference"],
        "hits_a": str(hits_a),
        "hits_b": str(hits_b),
        "hits_s": str(hits_s),
        "evals_median_a": f"{type7(evals_a):.1f}",
        "evals_median_b": f"{type7(evals_b):.1f}",
        "fisher_p": fmt(fisher_p),
        "mannwhitney_p": fmt(mw_p),
        "outside_per_call_median_a": fmt(type7(outside_a)),
        "outside_per_call_median_b": fmt(type7(outside_b)),
        "outside_ratio_unpaired": fmt(ratio(outside_a, outside_b)),
        "outside_ratio": fmt(type7(outside_pairs)),
        "outside_ratio_lo": fmt(outside_lo),
        "outside_ratio_hi": fmt(outside_hi),
        "rss_peak_median_a_kb": f"{type7(rss_a):.1f}",
        "rss_peak_median_b_kb": f"{type7(rss_b):.1f}",
        "rss_ratio": fmt(type7(rss_pairs)),
        "rss_ratio_lo": fmt(rss_lo),
        "rss_limit": fmt(rss_limit),
        **aa_fields,
        "s_outside_ratio": fmt(type7(paired_ratios(outside_b, per_call(arm_s)))),
        "vector_ticks_a": str(sum(vector["A"])),
        "vector_ticks_b": str(sum(vector["B"])),
        "vector_ticks_s": str(sum(vector["S"])),
        "vector_rows_b": str(sum(value > 0 for value in vector["B"])),
        "vector_rows_s": str(sum(value > 0 for value in vector["S"])),
        "vector_ranges_a": " ".join(ranges["A"]),
        "vector_ranges_b": " ".join(ranges["B"]),
        "vector_ranges_s": " ".join(ranges["S"]),
        "parity": "MISMATCH" if mismatches else "IDENTICAL",
        "parity_mismatches": " ".join(mismatches),
        "verdict": "FAIL" if reasons else "PASS",
        "reasons": " ".join(reasons),
    }
    write_csv(GATE / name / "summary.csv", SUMMARY_FIELDS, [summary])
    print(
        f"CELL {name} {summary['verdict']} hits {hits_a}/{hits_b}/{hits_s} fisher {summary['fisher_p']} "
        f"mw {summary['mannwhitney_p']} outside {summary['outside_ratio']} [{summary['outside_ratio_lo']}, {summary['outside_ratio_hi']}] "
        f"rss {summary['rss_ratio']} lo {summary['rss_ratio_lo']} parity {summary['parity']}",
        flush=True,
    )
    return summary


def write_panel_summary(cells: dict[str, dict[str, str]]) -> None:
    phases, _cells = parse_protocol()
    order = []
    for kind, names in phases:
        if kind == "abs":
            order.extend(names)
    rows = [summarise_cell(name, cells[name], phases) for name in order]
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
    if median_rank(12, 0.025) != 3 or median_rank(12, 0.05) != 3:
        raise SystemExit("median rank n=12")
    if median_ci(twelve) != (3.0, 10.0) or median_lower(twelve) != 3.0 or median_upper(twelve) != 10.0:
        raise SystemExit("median interval n=12")
    if abs(binomial_cdf(2, 12) - 79 / 4096) > 1e-15:
        raise SystemExit("binomial cdf")
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
    if paired_ratios([2.0, 4.0], [3.0, 2.0]) != [1.5, 0.5]:
        raise SystemExit("paired ratios")
    text = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s\n"
        "lj38,anneal,2000,400000,true,5,5,0.5,-1.0,0.25\n"
        "h2h_row system=lj75 seed=1 wall_s=2.0 best=-3.0 pot_s=1.0 outside_s=1.0\n"
        "arm single draws 3 accepts 2 best -1.0\n"
        "wall_seconds: 2.0\n"
        "outside_s: 1.0\n"
    )
    slower = (
        text.replace("0.5,", "0.9,")
        .replace("wall_s=2.0", "wall_s=7.0")
        .replace("pot_s=1.0", "pot_s=4.0")
        .replace("wall_seconds: 2.0", "wall_seconds: 7.0")
        .replace("outside_s: 1.0", "outside_s: 3.0")
    )
    if parity_digest(text, []) != parity_digest(slower, []):
        raise SystemExit("parity digest reads timing")
    if parity_digest(text, []) == parity_digest(text.replace("draws 3", "draws 4"), []):
        raise SystemExit("parity digest misses a count")
    if parity_digest(text, []) == parity_digest(text.replace(",-1.0,", ",-1.1,"), []):
        raise SystemExit("parity digest misses the best")
    probe = parse_probe("probe rss_anon_peak_kb=10 rss_anon_exit_kb=9 vm_hwm_kb=12 rss_reads=3 ticks=4 vector_ticks=1 vector_ranges=2 period_us=1000\n")
    if probe["vector_ticks"] != "1" or probe["rss_anon_peak_kb"] != "10":
        raise SystemExit("probe line")
    for arm in ("A", "A1"):
        if arm_side(arm) != "A":
            raise SystemExit("arm side")
    if arm_order("abs", 2000) != ("A", "B", "S") or arm_order("abs", 2001) != ("S", "B", "A"):
        raise SystemExit("arm order")
    parsed = move_counts(
        "sc96-80",
        "A",
        2000,
        "move cart proposed 4 accepted 3 improved 2",
    )
    if len(parsed) != 1 or parsed[0]["accepted"] != "3":
        raise SystemExit("move counts")
    if SEEDS != tuple(range(2000, 2012)):
        raise SystemExit("seeds")
    print("ok")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    for side in ("a", "a2", "b", "s"):
        parser.add_argument(f"--{side}-bin", type=Path)
        parser.add_argument(f"--{side}-tree", type=Path)
    parser.add_argument("--sc-lib", type=Path)
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
        needed = [args.a_bin, args.a2_bin, args.b_bin, args.s_bin, args.a_tree, args.a2_tree, args.b_tree, args.s_tree, args.sc_lib, args.logs]
        if any(item is None for item in needed):
            raise SystemExit("a run needs four binary directories, four trees, the Sutton-Chen library, and a log directory")
        run_schedule(args)
        return 0
    raise SystemExit("choose --run, --summarize, or --self-test")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
