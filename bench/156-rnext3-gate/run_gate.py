#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial A/B/R/A2 panel gate declared in PROTOCOL.md.

The schedule, budgets, and verdict rules are read from PROTOCOL.md. This
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
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GATE = Path(__file__).resolve().parent
REPO = GATE.parents[1]
PROTOCOL = GATE / "PROTOCOL.md"
PROBE_SOURCE = GATE / "probe.c"
ALPHA = 0.05
CONFIDENCE = 0.95
RATIO_LIMIT = 1.10
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261123
PROBE_PERIOD_US = 1000
PROBE_RSS_EVERY = 1

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
    "group",
    "n",
    "budget",
    "reference",
    "hits_a",
    "hits_b",
    "hits_r",
    "hits_a2",
    "evals_median_a",
    "evals_median_b",
    "evals_median_r",
    "fisher_p",
    "mannwhitney_p",
    "outside_per_call_median_a",
    "outside_per_call_median_b",
    "outside_ratio_unpaired",
    "outside_ratio",
    "outside_ratio_lo",
    "outside_ratio_hi",
    "outside_wall_ratio",
    "outside_wall_ratio_lo",
    "outside_wall_ratio_hi",
    "aa_outside_ratio",
    "aa_outside_ratio_lo",
    "aa_outside_ratio_hi",
    "outside_limit_aa",
    "outside_vs_aa",
    "rss_peak_median_a_kb",
    "rss_peak_median_b_kb",
    "rss_peak_median_r_kb",
    "rss_ratio",
    "rss_ratio_lo",
    "rss_ratio_hi",
    "aa_rss_ratio",
    "aa_rss_ratio_lo",
    "aa_rss_ratio_hi",
    "rss_limit_aa",
    "rss_vs_aa",
    "vector_ticks_a",
    "vector_ticks_b",
    "vector_ticks_r",
    "vector_ticks_a2",
    "vector_ranges_a",
    "vector_ranges_b",
    "vector_ranges_r",
    "vector_ranges_a2",
    "parity_br",
    "parity_br_mismatches",
    "parity_ba",
    "parity_ba_mismatches",
    "verdict",
    "verdict_vs_aa",
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

SIDES = ("A", "A2", "B", "R")

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
    gmin = load_module("g156_gmin_panel", REPO / "scripts" / "gmin_panel" / "run_panel.py")
    paper = load_module("g156_paper_0120", REPO / "scripts" / "paper_0120" / "run_panel.py")
    sc = load_module("g156_run_sc", REPO / "scripts" / "metal_h2h" / "run_sc.py")
    tip = load_module("g156_tip4p_h2h", REPO / "scripts" / "tip4p_h2h" / "run_h2h.py")
    tol = load_module("g156_target_tol", REPO / "scripts" / "target_tol.py")
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


def parse_protocol(path: Path = PROTOCOL) -> tuple[list[str], dict[str, dict[str, str]]]:
    text = path.read_text()
    order: list[str] = []
    for line in fenced(text, "order"):
        kind, *names = line.split()
        if kind != "run" or not names:
            raise RuntimeError(f"bad order line {line}")
        order.extend(names)
    cells = {}
    rows = list(csv.DictReader(fenced(text, "cells")))
    if not rows:
        raise RuntimeError("cell table is empty")
    for row in rows:
        if row["cell"] in cells:
            raise RuntimeError(f"duplicate cell {row['cell']}")
        cells[row["cell"]] = row
    return order, cells


def seeds_for(cell: dict[str, str]) -> tuple[int, ...]:
    count = int(cell["n"])
    return tuple(range(2000, 2000 + count))


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


def bootstrap_median_ci(
    values: list[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    n = len(values)
    if n < 1:
        raise ValueError("bootstrap needs a sample")
    rng = random.Random(seed)
    medians = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(type7(sample, 0.5))
    alpha = (1.0 - confidence) / 2.0
    return type7(medians, alpha), type7(medians, 1.0 - alpha)


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
    """One-sided tie-corrected Mann-Whitney p, alternative that B is larger."""
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


def check_cells(order: list[str], cells: dict[str, dict[str, str]], gmin, paper, sc, tip) -> None:
    if len(order) != len(set(order)):
        raise RuntimeError("a cell is listed twice in order")
    if set(order) != set(cells):
        raise RuntimeError("schedule and cell table differ")
    harness = {}
    for name, kind, _n, reference, _rho, _driver, _binary in gmin.SYSTEMS:
        harness[name] = (kind, gmin.PAPER_BUDGETS[name], reference)
    for name, budget in paper.PAPER_BUDGETS.items():
        if name not in harness or harness[name][1] != budget:
            raise RuntimeError(f"headline budget for {name} disagrees")
    for name, budget in sc.PAPER_BUDGETS.items():
        harness[name] = ("sc", budget, sc.REFERENCES[name])
    for n in tip.SIZES:
        if n not in tip.CAMBRIDGE:
            continue
        harness[f"tip4p{n}"] = ("tip4p", tip.BUDGET, tip.CAMBRIDGE[n])
    harness["lj75-seed"] = ("lj-seed", paper.PAPER_BUDGETS["lj75"], paper.LJ75_REFERENCE)
    for name, row in cells.items():
        if name not in harness:
            raise RuntimeError(f"unknown cell {name}")
        family, budget, reference = harness[name]
        if row["family"] != family:
            raise RuntimeError(f"{name} family {row['family']} != {family}")
        if int(row["budget"]) != int(budget):
            raise RuntimeError(f"{name} budget {row['budget']} != {budget}")
        if abs(float(row["reference"]) - float(reference)) > 1e-12:
            raise RuntimeError(f"{name} reference {row['reference']} != {reference}")
        if row["binary"] != BINARY_FOR_FAMILY[family]:
            raise RuntimeError(f"{name} binary {row['binary']}")
        if int(row["n"]) not in (12, 24):
            raise RuntimeError(f"{name} unexpected n={row['n']}")


def thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
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
    keys = {(row["cell"], row["arm"], row["seed"], row["move"]) for row in existing}
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


def arm_order(seed: int) -> tuple[str, ...]:
    return ("A", "B", "R", "A2") if seed % 2 == 0 else ("A2", "R", "B", "A")


def vector_functions(tree: Path) -> set[str]:
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
    order, cells = parse_protocol()
    gmin, paper, sc, tip, tol = repo_modules()
    check_cells(order, cells, gmin, paper, sc, tip)
    env = thread_env()
    env["RGPOT_SC_LIB"] = str(args.sc_lib.resolve())
    digest = machine_id()
    bins = {"A": args.a_bin, "A2": args.a2_bin, "B": args.b_bin, "R": args.r_bin}
    trees = {"A": args.a_tree, "A2": args.a2_tree, "B": args.b_tree, "R": args.r_tree}
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
    sha_lines = [f"{sha256_file(args.sc_lib.resolve())}  sc/librgpot_sc.so", f"{probe_sha}  probe/probe.so"]
    for side in SIDES:
        names = binaries[side]
        functions = vector_functions(trees[side])
        for name in sorted(names):
            path = names[name]
            if not path.is_file():
                raise SystemExit(f"missing binary {side} {name}")
            digest_bin = sha256_file(path)
            shas[(side, name)] = digest_bin
            sha_lines.append(f"{digest_bin}  {side}/{name}")
            found = vector_ranges(path, functions)
            range_file = probe_dir / f"ranges-{side}-{name}.txt"
            range_file.write_text("".join(f"{start:x} {size:x}\n" for start, size, _ in found))
            range_files[(side, name)] = range_file
            inventory.extend(f"{side} {name} {size:x} {symbol}" for _start, size, symbol in found)
    (probe_dir / "vector-symbols.txt").write_text("\n".join(inventory) + "\n")
    (GATE / "binary-sha256.txt").write_text("\n".join(sha_lines) + "\n")
    for name in set(BINARY_FOR_FAMILY.values()):
        if shas[("B", name)] == shas[("A", name)]:
            raise SystemExit(f"{name} is the same binary on A and B")
        if shas[("R", name)] == shas[("A", name)]:
            raise SystemExit(f"{name} is the same binary on A and R")
        if shas[("R", name)] == shas[("B", name)]:
            raise SystemExit(f"{name} is the same binary on B and R")
    for cell in cells.values():
        for side in SIDES:
            budget_matches(binaries[side][cell["binary"]], cell, trees[side], env)
    for name in order:
        cell = cells[name]
        rows_path = GATE / name / "rows.csv"
        have = {(row["arm"], int(row["seed"])) for row in read_rows(rows_path)}
        for seed in seeds_for(cell):
            for arm in arm_order(seed):
                if (arm, seed) in have:
                    continue
                binary = binaries[arm][cell["binary"]]
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
                            "GATE_PROBE_RANGES": str(range_files[(arm, cell["binary"])]),
                            "GATE_PROBE_PERIOD_US": str(PROBE_PERIOD_US),
                            "GATE_PROBE_RSS_EVERY": str(PROBE_RSS_EVERY),
                        }
                    )
                    command = command_for(binary, cell, seed, prefix)
                    stdout, stderr, affinity = run_checked(
                        command, trees[arm], run_env, log, scratch
                    )
                    probe_text = probe_out.read_text() if probe_out.is_file() else ""
                    with log.open("a") as handle:
                        handle.write(probe_text)
                    probed = parse_probe(probe_text)
                    parity = parity_digest(stdout + stderr, seed_outputs(cell, prefix))
                got = result_of(cell, seed, stdout, tol)
                if cell["family"] == "sc":
                    counts = move_counts(name, arm, seed, stdout + stderr)
                    if counts:
                        append_move_counts(GATE / "diag" / "sc-moves.csv", counts)
                row = {
                    "cell": name,
                    "core": "0",
                    "affinity": affinity,
                    "machine_id": digest,
                    "binary_sha256": shas[(arm, cell["binary"])],
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
                    f"wall={row['wall_s']} out={row['outside_s']} rss={row['rss_anon_peak_kb']} vec={row['vector_ticks']}",
                    flush=True,
                )
        summarise_cell(name, cells[name])
    write_panel_summary(order, cells)


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


def check_identity(rows: list[dict[str, str]], expected_seeds: tuple[int, ...]) -> None:
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
        if int(row["seed"]) not in expected_seeds:
            raise RuntimeError(f"unexpected seed {row['seed']}")
        if int(row["rss_anon_peak_kb"]) <= 0 or int(row["rss_reads"]) < 2:
            raise RuntimeError(f"invalid rss {row['cell']} {row['arm']} {row['seed']}")
        if int(row["vector_ranges"]) <= 0:
            raise RuntimeError(f"zero vector_ranges {row['cell']} {row['arm']} {row['seed']}")
    by_arm: dict[str, set[str]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], set()).add(row["binary_sha256"])
    for arm, digests in by_arm.items():
        if len(digests) != 1:
            raise RuntimeError(f"{arm} has more than one binary")
    if "B" in by_arm and "A" in by_arm and by_arm["B"] & by_arm["A"]:
        raise RuntimeError("arm B matches arm A")
    if "R" in by_arm and "A" in by_arm and by_arm["R"] & by_arm["A"]:
        raise RuntimeError("arm R matches arm A")
    if "R" in by_arm and "B" in by_arm and by_arm["R"] & by_arm["B"]:
        raise RuntimeError("arm R matches arm B")


def arm_rows(rows: list[dict[str, str]], arm: str, seeds: tuple[int, ...]) -> list[dict[str, str]] | None:
    found = {int(row["seed"]): row for row in rows if row["arm"] == arm}
    if set(found) != set(seeds) or sum(row["arm"] == arm for row in rows) != len(seeds):
        return None
    return [found[seed] for seed in seeds]


def fmt(value: float) -> str:
    return f"{value:.8g}"


def parity_check(left: list[dict[str, str]], right: list[dict[str, str]]) -> list[str]:
    mismatches = []
    for row_l, row_r in zip(left, right):
        fields = [
            key
            for key in ("best", "parity_digest")
            if row_l[key] != row_r[key]
        ]
        if fields:
            mismatches.append(f"{row_l['seed']}:{'+'.join(fields)}")
    return mismatches


def summarise_cell(name: str, cell: dict[str, str]) -> dict[str, str]:
    seeds = seeds_for(cell)
    rows = read_rows(GATE / name / "rows.csv")
    check_identity(rows, seeds)
    blank = {key: "" for key in SUMMARY_FIELDS}
    blank.update(
        {
            "cell": name,
            "group": cell["group"],
            "n": cell["n"],
            "budget": cell["budget"],
            "reference": cell["reference"],
            "verdict": "INCOMPLETE",
            "verdict_vs_aa": "INCOMPLETE",
            "reasons": "",
        }
    )
    arm_a = arm_rows(rows, "A", seeds)
    arm_b = arm_rows(rows, "B", seeds)
    arm_r = arm_rows(rows, "R", seeds)
    arm_a2 = arm_rows(rows, "A2", seeds)
    if arm_a is None or arm_b is None or arm_r is None or arm_a2 is None:
        write_csv(GATE / name / "summary.csv", SUMMARY_FIELDS, [blank])
        return blank
    hits_a = sum(row["hit"] == "1" for row in arm_a)
    hits_b = sum(row["hit"] == "1" for row in arm_b)
    hits_r = sum(row["hit"] == "1" for row in arm_r)
    hits_a2 = sum(row["hit"] == "1" for row in arm_a2)
    evals_a = numeric(arm_a, "evals_to_hit")
    evals_b = numeric(arm_b, "evals_to_hit")
    evals_r = numeric(arm_r, "evals_to_hit")
    fisher_p = fisher_two_sided(hits_a, len(arm_a) - hits_a, hits_b, len(arm_b) - hits_b)
    scores_a = [miss_score(row["hit"], float(row["evals_to_hit"]), float(row["budget"])) for row in arm_a]
    scores_b = [miss_score(row["hit"], float(row["evals_to_hit"]), float(row["budget"])) for row in arm_b]
    mw_p = mannwhitney_worse(scores_a, scores_b)

    outside_a = per_call(arm_a)
    outside_b = per_call(arm_b)
    outside_a2 = per_call(arm_a2)
    outside_pairs = paired_ratios(outside_a, outside_b)
    outside_lo, outside_hi = bootstrap_median_ci(outside_pairs)
    aa_outside_pairs = paired_ratios(outside_a, outside_a2)
    aa_outside_lo, aa_outside_hi = bootstrap_median_ci(aa_outside_pairs)
    outside_limit_aa = max(RATIO_LIMIT, aa_outside_hi)

    wall_outside_pairs = paired_ratios(numeric(arm_a, "outside_s"), numeric(arm_b, "outside_s"))
    wall_outside_lo, wall_outside_hi = bootstrap_median_ci(wall_outside_pairs)

    rss_a = numeric(arm_a, "rss_anon_peak_kb")
    rss_b = numeric(arm_b, "rss_anon_peak_kb")
    rss_r = numeric(arm_r, "rss_anon_peak_kb")
    rss_a2 = numeric(arm_a2, "rss_anon_peak_kb")
    rss_pairs = paired_ratios(rss_a, rss_b)
    rss_lo, rss_hi = bootstrap_median_ci(rss_pairs)
    aa_rss_pairs = paired_ratios(rss_a, rss_a2)
    aa_rss_lo, aa_rss_hi = bootstrap_median_ci(aa_rss_pairs)
    rss_limit_aa = max(RATIO_LIMIT, aa_rss_hi)

    reasons = []
    if hits_b < hits_a and fisher_p < ALPHA:
        reasons.append("fisher")
    if mw_p < ALPHA:
        reasons.append("mannwhitney")
    if outside_hi > RATIO_LIMIT:
        reasons.append("outside")
    if rss_hi > RATIO_LIMIT:
        reasons.append("rss")

    reasons_aa = []
    if hits_b < hits_a and fisher_p < ALPHA:
        reasons_aa.append("fisher")
    if mw_p < ALPHA:
        reasons_aa.append("mannwhitney")
    if outside_hi > outside_limit_aa:
        reasons_aa.append("outside")
    if rss_hi > rss_limit_aa:
        reasons_aa.append("rss")

    mismatches_br = parity_check(arm_b, arm_r)
    mismatches_ba = parity_check(arm_b, arm_a)

    vector = {
        arm: [int(row["vector_ticks"]) for row in group]
        for arm, group in (("A", arm_a), ("B", arm_b), ("R", arm_r), ("A2", arm_a2))
    }
    ranges = {
        arm: sorted({row["vector_ranges"] for row in group})
        for arm, group in (("A", arm_a), ("B", arm_b), ("R", arm_r), ("A2", arm_a2))
    }

    summary = {
        "cell": name,
        "group": cell["group"],
        "n": str(len(arm_a)),
        "budget": cell["budget"],
        "reference": cell["reference"],
        "hits_a": str(hits_a),
        "hits_b": str(hits_b),
        "hits_r": str(hits_r),
        "hits_a2": str(hits_a2),
        "evals_median_a": f"{type7(evals_a):.1f}",
        "evals_median_b": f"{type7(evals_b):.1f}",
        "evals_median_r": f"{type7(evals_r):.1f}",
        "fisher_p": fmt(fisher_p),
        "mannwhitney_p": fmt(mw_p),
        "outside_per_call_median_a": fmt(type7(outside_a)),
        "outside_per_call_median_b": fmt(type7(outside_b)),
        "outside_ratio_unpaired": fmt(ratio(outside_a, outside_b)),
        "outside_ratio": fmt(type7(outside_pairs)),
        "outside_ratio_lo": fmt(outside_lo),
        "outside_ratio_hi": fmt(outside_hi),
        "outside_wall_ratio": fmt(type7(wall_outside_pairs)),
        "outside_wall_ratio_lo": fmt(wall_outside_lo),
        "outside_wall_ratio_hi": fmt(wall_outside_hi),
        "aa_outside_ratio": fmt(type7(aa_outside_pairs)),
        "aa_outside_ratio_lo": fmt(aa_outside_lo),
        "aa_outside_ratio_hi": fmt(aa_outside_hi),
        "outside_limit_aa": fmt(outside_limit_aa),
        "outside_vs_aa": "FAIL" if outside_hi > outside_limit_aa else "PASS",
        "rss_peak_median_a_kb": f"{type7(rss_a):.1f}",
        "rss_peak_median_b_kb": f"{type7(rss_b):.1f}",
        "rss_peak_median_r_kb": f"{type7(rss_r):.1f}",
        "rss_ratio": fmt(type7(rss_pairs)),
        "rss_ratio_lo": fmt(rss_lo),
        "rss_ratio_hi": fmt(rss_hi),
        "aa_rss_ratio": fmt(type7(aa_rss_pairs)),
        "aa_rss_ratio_lo": fmt(aa_rss_lo),
        "aa_rss_ratio_hi": fmt(aa_rss_hi),
        "rss_limit_aa": fmt(rss_limit_aa),
        "rss_vs_aa": "FAIL" if rss_hi > rss_limit_aa else "PASS",
        "vector_ticks_a": str(sum(vector["A"])),
        "vector_ticks_b": str(sum(vector["B"])),
        "vector_ticks_r": str(sum(vector["R"])),
        "vector_ticks_a2": str(sum(vector["A2"])),
        "vector_ranges_a": " ".join(ranges["A"]),
        "vector_ranges_b": " ".join(ranges["B"]),
        "vector_ranges_r": " ".join(ranges["R"]),
        "vector_ranges_a2": " ".join(ranges["A2"]),
        "parity_br": "MISMATCH" if mismatches_br else "IDENTICAL",
        "parity_br_mismatches": " ".join(mismatches_br),
        "parity_ba": "MISMATCH" if mismatches_ba else "IDENTICAL",
        "parity_ba_mismatches": " ".join(mismatches_ba),
        "verdict": "FAIL" if reasons else "PASS",
        "verdict_vs_aa": "FAIL" if reasons_aa else "PASS",
        "reasons": " ".join(reasons),
    }
    write_csv(GATE / name / "summary.csv", SUMMARY_FIELDS, [summary])
    print(
        f"CELL {name} {summary['verdict']} hits={hits_a}/{hits_b}/{hits_r}/{hits_a2} "
        f"fp={summary['fisher_p']} mw={summary['mannwhitney_p']} "
        f"out={summary['outside_ratio']} [{summary['outside_ratio_lo']}, {summary['outside_ratio_hi']}] "
        f"rss={summary['rss_ratio']} [{summary['rss_ratio_lo']}, {summary['rss_ratio_hi']}] "
        f"br={summary['parity_br']}",
        flush=True,
    )
    return summary


def write_summary_md(rows: list[dict[str, str]]) -> None:
    complete = [row for row in rows if row["verdict"] != "INCOMPLETE"]
    overall = (
        "INCOMPLETE"
        if len(complete) != len(rows)
        else ("FAIL" if any(row["verdict"] != "PASS" for row in complete) else "PASS")
    )
    overall_aa = (
        "INCOMPLETE"
        if len(complete) != len(rows)
        else ("FAIL" if any(row["verdict_vs_aa"] != "PASS" for row in complete) else "PASS")
    )
    lines = [
        "# 156 r-next3 gate summary",
        "",
        f"Overall verdict (1.10 bound): **{overall}**",
        f"Overall verdict (max(1.10, A2/A upper)): **{overall_aa}**",
        f"Completed cells: {len(complete)} / {len(rows)}",
        "",
        "| Cell | Group | n | Hits A/B/R/A2 | Fisher p | MWU p | B/A Outside (95% CI) | A2/A Outside (95% CI) | B/A RSS (95% CI) | A2/A RSS (95% CI) | B vs R Parity | B vs A Parity | Verdict | vs A2/A | Reasons |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if r["verdict"] == "INCOMPLETE":
            lines.append(
                f"| {r['cell']} | {r['group']} | {r['n']} | - | - | - | - | - | - | - | - | - | INCOMPLETE | INCOMPLETE | |"
            )
            continue
        hits = f"{r['hits_a']}/{r['hits_b']}/{r['hits_r']}/{r['hits_a2']}"
        out_ci = f"{r['outside_ratio']} [{r['outside_ratio_lo']}, {r['outside_ratio_hi']}]"
        aa_out = f"{r['aa_outside_ratio']} [{r['aa_outside_ratio_lo']}, {r['aa_outside_ratio_hi']}]"
        rss_ci = f"{r['rss_ratio']} [{r['rss_ratio_lo']}, {r['rss_ratio_hi']}]"
        aa_rss = f"{r['aa_rss_ratio']} [{r['aa_rss_ratio_lo']}, {r['aa_rss_ratio_hi']}]"
        br_note = "IDENTICAL" if r["parity_br"] == "IDENTICAL" else f"MISMATCH ({len(r['parity_br_mismatches'].split())}/{r['n']})"
        ba_note = "IDENTICAL" if r["parity_ba"] == "IDENTICAL" else f"MISMATCH ({len(r['parity_ba_mismatches'].split())}/{r['n']})"
        lines.append(
            f"| {r['cell']} | {r['group']} | {r['n']} | {hits} | {r['fisher_p']} | {r['mannwhitney_p']} | {out_ci} | {aa_out} | {rss_ci} | {aa_rss} | {br_note} | {ba_note} | {r['verdict']} | {r['verdict_vs_aa']} | {r['reasons']} |"
        )
    lines.append("")
    (GATE / "SUMMARY.md").write_text("\n".join(lines))


def write_panel_summary(order: list[str], cells: dict[str, dict[str, str]]) -> None:
    rows = [summarise_cell(name, cells[name]) for name in order]
    write_csv(GATE / "summary.csv", SUMMARY_FIELDS, rows)
    write_summary_md(rows)


def self_test() -> None:
    order, cells = parse_protocol()
    gmin, paper, sc, tip, tol = repo_modules()
    check_cells(order, cells, gmin, paper, sc, tip)
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12 or abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7")
    lo1, hi1 = bootstrap_median_ci([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 0.99, 1.03, 0.97])
    lo2, hi2 = bootstrap_median_ci([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 0.99, 1.03, 0.97])
    if (lo1, hi1) != (lo2, hi2) or not (lo1 <= 1.0 <= hi1):
        raise SystemExit("bootstrap determinism")
    if abs(fisher_two_sided(2, 2, 0, 4) - 12 / 28) > 1e-12:
        raise SystemExit("fisher")
    if mannwhitney_worse([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) != 1.0:
        raise SystemExit("tied mann-whitney")
    if not (mannwhitney_worse([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]) < ALPHA):
        raise SystemExit("worse mann-whitney")
    if arm_order(2000) != ("A", "B", "R", "A2") or arm_order(2001) != ("A2", "R", "B", "A"):
        raise SystemExit("arm order")
    if len(order) != 13:
        raise SystemExit(f"expected 13 cells got {len(order)}")
    print("ok")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    for side in ("a", "a2", "b", "r"):
        parser.add_argument(f"--{side}-bin", type=Path)
        parser.add_argument(f"--{side}-tree", type=Path)
    parser.add_argument("--sc-lib", type=Path)
    parser.add_argument("--logs", type=Path)
    args = parser.parse_args(argv[1:])
    if args.self_test:
        self_test()
        return 0
    if args.summarize:
        order, cells = parse_protocol()
        write_panel_summary(order, cells)
        return 0
    if args.run:
        needed = [
            args.a_bin,
            args.a2_bin,
            args.b_bin,
            args.r_bin,
            args.a_tree,
            args.a2_tree,
            args.b_tree,
            args.r_tree,
            args.sc_lib,
            args.logs,
        ]
        if any(item is None for item in needed):
            raise SystemExit("a run needs four binary directories, four trees, the Sutton-Chen library, and a log directory")
        run_schedule(args)
        return 0
    raise SystemExit("choose --run, --summarize, or --self-test")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
