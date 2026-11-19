#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Default-on gate for the geometry-derived IRA, packing, and rigid-body change.

``run`` builds nothing. It drives three already-built ``h2h_seed`` binaries,
one fresh process per row, and writes ``rows.csv`` and ``sessions.jsonl``.
``report`` reads only the rows file and writes ``summary.csv`` and ``gate.md``.

The matrix is the paper budget list: LJ 38/75/98/150, Morse 55 at rho 14,
Morse 80 at rho 6, Morse 38 at rho 10, and TIP4P 8/15/20/21/25, plus TIP4P 6.
``h2h_seed`` accepts TIP4P 6. ``Config::for_tip4p`` requires two or more
molecules, ``Config::recommended_tip4p`` is specified at six, and
``GminTip4p::new`` accepts that count. ``WATER_HEXAMER_MOLECULES`` only
chooses the molecular SOAP and rewire kernels in the group library. The
TIP4P arm of ``h2h_seed`` is a rigid centre-and-rotation chart and does not
consult that constant, so six is a supported cell and stays in the matrix.

Hits use ``scripts/target_tol.py`` ``energy_hits`` at ``DEFAULT_HIT_TOL``
(1e-3): a finite energy hits when it is at or below the reference plus the
tolerance. References, by family:

- Lennard-Jones energies and budgets are the ``SYSTEMS`` / ``PAPER_BUDGETS``
  entries in ``scripts/gmin_panel/run_panel.py`` (Cambridge Lennard-Jones).
- Morse energies are the Cambridge Doye-Wales rows in
  ``examples/fixtures/morse_ccd.csv``. The budgets are the rule in
  ``examples/h2h_morse.rs``: 4e5 at 38, 1e6 at 55, 4e6 at 75 and above.
- TIP4P energies for n <= 21, including n = 6, are the Cambridge headers in
  ``data/tip4p_ccd.csv`` written by ``scripts/tip4p_ccd_table.py``. The paper
  harness ``scripts/tip4p_h2h/run_h2h.py`` scores those headers. (H2O)25 has
  no Cambridge header there, so that harness scores it against the lowest
  energy either arm reached. This gate does the same, over arms A and B.

``best`` is the ``h2h_row`` ``best`` field (lowest evaluated energy), which
is the energy that harness parses. ``calls`` is the ``calls:`` line. A miss
is censored at the budget before the median and the exact Mann-Whitney test.
The test uses the exact null of the midrank sum, ties included. The two-sided
p is twice the smaller tail, capped at 1. Fisher is the two-sided exact test
that sums every table whose probability is at most the observed table's.
Medians are Hyndman-Fan type 7, as in the paper harness.

B loses on a system when it has fewer hits, or its censored calls are worse
at p < 0.05, or its median ``outside_s`` exceeds A by more than the A/A
band. The band is the absolute type-7 median gap between A and the second
build A2. LJ uses the LJ75 gap, Morse the M80 rho 6 gap, TIP4P 6 its own
gap, and the other TIP4P cells the TIP4P15 gap scaled by
``median_A(system) / median_A(anchor)``. The overall verdict is default-on
OK only when no system loses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


HIT_TOL = 1.0e-3
SEEDS = tuple(range(2000, 2012))
AA_SYSTEMS = ("lj75", "m80r6", "tip4p15", "tip4p6")
P_CALLS = 0.05

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
    "JULIA_NUM_THREADS",
)

ROW_FIELDS = (
    "system",
    "seed",
    "arm",
    "commit",
    "tree",
    "calls",
    "charged",
    "best",
    "hit",
    "wall_s",
    "pot_s",
    "outside_s",
    "machine_id_hash",
    "core",
)

SUMMARY_FIELDS = (
    "system",
    "budget",
    "reference",
    "n",
    "hits_a",
    "hits_b",
    "fisher_p",
    "median_calls_a",
    "median_calls_b",
    "mw_u_calls",
    "mw_p_calls",
    "median_wall_a",
    "median_wall_b",
    "median_pot_a",
    "median_pot_b",
    "median_outside_a",
    "median_outside_b",
    "mw_u_outside",
    "mw_p_outside",
    "aa_anchor",
    "anchor_band_s",
    "outside_delta_s",
    "outside_limit_s",
    "best_identical",
    "verdict",
)


@dataclass(frozen=True)
class System:
    name: str
    kind: str
    n: int
    rho: str
    budget: int
    reference: str | None
    family: str


# rho is the text passed to h2h_seed. reference None is the pooled minimum.
SYSTEMS = (
    System("lj38", "lj", 38, "0", 400_000, "-173.928427", "lj"),
    System("lj75", "lj", 75, "0", 4_000_000, "-397.492331", "lj"),
    System("lj98", "lj", 98, "0", 4_000_000, "-543.665361", "lj"),
    System("lj150", "lj", 150, "0", 4_000_000, "-893.310258", "lj"),
    System("m55r14", "morse", 55, "14", 1_000_000, "-220.646208", "morse"),
    System("m80r6", "morse", 80, "6", 4_000_000, "-378.333471", "morse"),
    System("m38r10", "morse", 38, "10", 400_000, "-145.849817", "morse"),
    System("tip4p8", "tip4p", 8, "0", 2_000_000, "-305.5183221676", "tip4p"),
    System("tip4p15", "tip4p", 15, "0", 2_000_000, "-628.3722525077", "tip4p"),
    System("tip4p20", "tip4p", 20, "0", 2_000_000, "-872.9887544952", "tip4p"),
    System("tip4p21", "tip4p", 21, "0", 2_000_000, "-916.7074302940", "tip4p"),
    System("tip4p25", "tip4p", 25, "0", 2_000_000, None, "tip4p"),
    System("tip4p6", "tip4p", 6, "0", 2_000_000, "-197.7805260034", "tip4p"),
)

BY_NAME = {spec.name: spec for spec in SYSTEMS}
ANCHOR = {
    "lj": "lj75",
    "morse": "m80r6",
    "tip4p": "tip4p15",
}


def anchor_of(name: str) -> str:
    if name == "tip4p6":
        return "tip4p6"
    return ANCHOR[BY_NAME[name].family]


def energy_hits(energy: float, target: float, tol: float = HIT_TOL) -> bool:
    if not (math.isfinite(energy) and math.isfinite(target) and math.isfinite(tol)):
        return False
    return tol >= 0.0 and energy <= target + tol


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1 = a + b
    row2 = c + d
    col1 = a + c

    def table_p(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    p0 = table_p(a)
    total = 0.0
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    for x in range(lo, hi + 1):
        px = table_p(x)
        if px <= p0 * (1.0 + 1e-9):
            total += px
    return min(1.0, total)


def mannwhitney_exact(x: list[float], y: list[float]) -> tuple[float, float]:
    """Exact two-sided Mann-Whitney of y against x.

    Returns ``(U_y, p)``. ``U_y`` is the midrank Mann-Whitney statistic of
    ``y`` and is large when ``y`` values are large. Ties use midranks. ``p``
    is twice the smaller exact tail of the rank-sum null, capped at 1.
    """
    n1 = len(x)
    n2 = len(y)
    if n1 == 0 or n2 == 0:
        raise ValueError("empty sample")
    pooled = [float(v) for v in x + y]
    if not all(math.isfinite(v) for v in pooled):
        raise ValueError("non-finite sample")
    n = n1 + n2
    order = sorted(range(n), key=lambda i: (pooled[i], i))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[order[j + 1]] == pooled[order[i]]:
            j += 1
        average = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    scaled = []
    for rank in ranks:
        twice = int(round(rank * 2.0))
        if abs(twice / 2.0 - rank) > 1e-9:
            raise ValueError("rank is not a half-integer")
        scaled.append(twice)
    r_y = sum(ranks[n1:])
    width = sum(scaled) + 1
    ways = [[0] * width for _ in range(n2 + 1)]
    ways[0][0] = 1
    for item in scaled:
        for count in range(n2, 0, -1):
            row = ways[count]
            prev = ways[count - 1]
            for total in range(width - 1, item - 1, -1):
                add = prev[total - item]
                if add:
                    row[total] += add
    support = sum(ways[n2])
    choose = math.comb(n, n2)
    if support != choose:
        raise RuntimeError(f"rank-sum support {support} != {choose}")
    target = int(round(r_y * 2.0))
    at_most = sum(ways[n2][s] for s in range(target + 1))
    at_least = sum(ways[n2][s] for s in range(target, width))
    p = min(1.0, 2.0 * min(at_most, at_least) / support)
    u_y = r_y - n2 * (n2 + 1) / 2.0
    return u_y, p


def censored_calls(calls: int, hit: int, budget: int) -> int:
    if calls < 0 or calls > budget:
        raise ValueError(f"calls {calls} outside 0..{budget}")
    if hit:
        return calls
    return budget


def jobs():
    for seed in SEEDS:
        even = seed % 2 == 0
        for spec in SYSTEMS:
            if spec.name in AA_SYSTEMS:
                arms = ("A", "A2", "B") if even else ("A2", "B", "A")
            else:
                arms = ("A", "B") if even else ("B", "A")
            for arm in arms:
                yield spec, seed, arm


def parse_log(text: str) -> dict[str, str]:
    calls = charged = wall_seconds = pot_line = outside_line = None
    row = None
    for line in text.splitlines():
        if line.startswith("calls:"):
            calls = line.split(":", 1)[1].strip()
        elif line.startswith("charged:"):
            charged = line.split(":", 1)[1].strip()
        elif line.startswith("wall_seconds:"):
            wall_seconds = line.split(":", 1)[1].strip()
        elif line.startswith("pot_s:"):
            pot_line = line.split(":", 1)[1].strip()
        elif line.startswith("outside_s:"):
            outside_line = line.split(":", 1)[1].strip()
        elif line.startswith("h2h_row "):
            fields = {}
            for part in line.split()[1:]:
                key, value = part.split("=", 1)
                fields[key] = value
            row = fields
    if None in (calls, charged, wall_seconds, pot_line, outside_line) or row is None:
        raise ValueError("log is missing calls, charged, times, or h2h_row")
    for key in ("best", "charged", "wall_s", "pot_s", "outside_s", "seed", "budget"):
        if key not in row:
            raise ValueError(f"h2h_row missing {key}")
    if row["charged"] != charged:
        raise ValueError("charged line and h2h_row disagree")
    if row["wall_s"] != wall_seconds:
        raise ValueError("wall_seconds and h2h_row wall_s disagree")
    if row["pot_s"] != pot_line:
        raise ValueError("pot_s line and h2h_row disagree")
    if row["outside_s"] != outside_line:
        raise ValueError("outside_s line and h2h_row disagree")
    return {
        "calls": calls,
        "charged": charged,
        "best": row["best"],
        "wall_s": row["wall_s"],
        "pot_s": row["pot_s"],
        "outside_s": row["outside_s"],
        "seed": row["seed"],
        "budget": row["budget"],
    }


def machine_id_hash() -> str:
    raw = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
    if not raw:
        raise SystemExit("machine id is empty")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("cpu model line is missing")


def child_affinity(pid: int, expected: str = "h2h_seed") -> str | None:
    """Read affinity after ``taskset`` has exec'd ``expected``.

    The first status line belongs to ``taskset`` itself, before it pins
    and replaces itself. Wait until the process name is the example, then
    read ``Cpus_allowed_list``.
    """
    status = Path(f"/proc/{pid}/status")
    comm = Path(f"/proc/{pid}/comm")
    last = None
    for _ in range(2000):
        if not comm.is_file():
            return last
        name = comm.read_text(encoding="utf-8").strip()
        if status.is_file():
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("Cpus_allowed_list:"):
                    last = line.split(":", 1)[1].strip()
        if name == expected and last is not None:
            return last
        time.sleep(0.001)
    return last


def thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in THREAD_VARS:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="\n") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROW_FIELDS:
            raise SystemExit(f"rows columns are {reader.fieldnames}")
        rows = []
        for row in reader:
            if any("\r" in (value or "") for value in row.values()):
                raise SystemExit("rows contain CR")
            rows.append({key: row[key] for key in ROW_FIELDS})
    return rows


def pooled_token(tokens: list[str]) -> str:
    return min(tokens, key=lambda token: (float(token), token))


def hit_of(spec: System, best: str, pooled: str | None) -> int:
    if spec.reference is None:
        if pooled is None:
            raise ValueError("pooled reference is unset")
        target = float(pooled)
    else:
        target = float(spec.reference)
    return int(energy_hits(float(best), target, HIT_TOL))


def apply_hits(rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["system"], []).append(row)
    for spec in SYSTEMS:
        group = grouped.get(spec.name, [])
        pooled = None
        if spec.reference is None:
            tokens = [row["best"] for row in group if row["arm"] in ("A", "B")]
            if not tokens:
                continue
            pooled = pooled_token(tokens)
        for row in group:
            row["hit"] = str(hit_of(spec, row["best"], pooled))


def load_sessions(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            out.append(json.loads(line))
    return out


def append_session(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def log_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        parse_log(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return True


def require_sha(name: str, value: str) -> str:
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise SystemExit(f"{name} is not a 40-digit sha")
    return value


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"required file missing: {path.name}")
    return path


def run_campaign(args: argparse.Namespace) -> None:
    self_check()
    commit = {
        "A": require_sha("commit-a", args.commit_a),
        "B": require_sha("commit-b", args.commit_b),
        "A2": require_sha("commit-a2", args.commit_a2),
    }
    tree = {
        "A": require_sha("tree-a", args.tree_a),
        "B": require_sha("tree-b", args.tree_b),
        "A2": require_sha("tree-a2", args.tree_a2),
    }
    if commit["A"] != commit["A2"] or tree["A"] != tree["A2"]:
        raise SystemExit("A2 is not the same commit and tree as A")
    if tree["A"] == tree["B"]:
        raise SystemExit("A and B have the same tree")
    binary = {
        "A": require_file(Path(args.binary_a)),
        "B": require_file(Path(args.binary_b)),
        "A2": require_file(Path(args.binary_a2)),
    }
    rows_path = Path(args.rows)
    sessions_path = Path(args.sessions)
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    rows = read_rows(rows_path) if rows_path.is_file() else []
    done = {(row["system"], int(row["seed"]), row["arm"]) for row in rows}
    sessions = load_sessions(sessions_path)
    failed = [item for item in sessions if item.get("returncode") not in (0, None)]
    if failed:
        raise SystemExit("a previous row exited nonzero; not rerunning it")
    if sessions:
        session = sessions[0]["session"]
    else:
        session = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    machine = machine_id_hash()
    cpu = cpu_model()
    env = thread_env()
    seq = len(sessions)
    for spec, seed, arm in jobs():
        key = (spec.name, seed, arm)
        log_path = logs / f"{spec.name}-{seed}-{arm}.log"
        if key in done:
            continue
        prior = [
            item
            for item in sessions
            if item["system"] == spec.name and int(item["seed"]) == seed and item["arm"] == arm
        ]
        if prior and prior[-1].get("returncode") not in (0, None):
            raise SystemExit(f"not rerunning failed {spec.name} {seed} {arm}")
        if prior and not log_complete(log_path):
            raise SystemExit(f"{spec.name} {seed} {arm} finished without a complete log")
        if log_path.exists() and not log_complete(log_path) and not prior:
            log_path.unlink()
            for suffix in (".trace.csv", ".xyz", ".gmin", ".start.gmin"):
                extra = logs / f"{spec.name}-{seed}-{arm}{suffix}"
                if extra.exists():
                    extra.unlink()
        if log_complete(log_path):
            parsed = parse_log(log_path.read_text(encoding="utf-8"))
        else:
            prefix = logs / f"{spec.name}-{seed}-{arm}"
            command = [
                "taskset",
                "-c",
                "0",
                str(binary[arm]),
                spec.kind,
                str(spec.n),
                spec.rho,
                str(spec.budget),
                str(seed),
                "recommended",
                "-",
                str(prefix),
            ]
            with log_path.open("w", encoding="utf-8", newline="\n") as handle:
                proc = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                affinity = child_affinity(proc.pid, "h2h_seed")
                (logs / f"{spec.name}-{seed}-{arm}.affinity").write_text(
                    "" if affinity is None else affinity, encoding="utf-8"
                )
                if affinity != "0":
                    proc.kill()
                code = proc.wait()
            record = {
                "session": session,
                "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "seq": seq,
                "system": spec.name,
                "seed": seed,
                "arm": arm,
                "commit": commit[arm],
                "tree": tree[arm],
                "cpu_model": cpu,
                "machine_id_hash": machine,
                "core": "0",
                "affinity": affinity,
                "returncode": code,
                "log": log_path.name,
                "threads": "1",
            }
            trace = Path(str(prefix) + ".trace.csv")
            if trace.is_file():
                record["trace_sha256"] = sha256_file(trace)
            append_session(sessions_path, record)
            sessions.append(record)
            seq += 1
            if affinity != "0":
                raise SystemExit(f"{spec.name} {seed} {arm} affinity is {affinity}")
            if code != 0:
                raise SystemExit(f"{spec.name} {seed} {arm} exited {code}")
            parsed = parse_log(log_path.read_text(encoding="utf-8"))
        if parsed["seed"] != str(seed) or int(parsed["budget"]) != spec.budget:
            raise SystemExit(f"{spec.name} {seed} {arm} log identity mismatch")
        calls = int(parsed["calls"])
        if calls < 0 or calls > spec.budget:
            raise SystemExit(f"{spec.name} {seed} {arm} calls {calls}")
        row = {
            "system": spec.name,
            "seed": str(seed),
            "arm": arm,
            "commit": commit[arm],
            "tree": tree[arm],
            "calls": str(calls),
            "charged": parsed["charged"],
            "best": parsed["best"],
            "hit": "0",
            "wall_s": parsed["wall_s"],
            "pot_s": parsed["pot_s"],
            "outside_s": parsed["outside_s"],
            "machine_id_hash": machine,
            "core": "0",
        }
        rows.append(row)
        done.add(key)
        apply_hits(rows)
        write_csv(rows_path, ROW_FIELDS, rows)
        print(
            f"{spec.name} {seed} {arm} calls={row['calls']} charged={row['charged']} "
            f"best={row['best']} hit={row['hit']} wall_s={row['wall_s']} "
            f"outside_s={row['outside_s']}",
            flush=True,
        )
    expected = {(spec.name, seed, arm) for spec, seed, arm in jobs()}
    if done != expected:
        missing = len(expected - done)
        raise SystemExit(f"{missing} rows still missing")
    apply_hits(rows)
    write_csv(rows_path, ROW_FIELDS, rows)
    print(f"rows {len(rows)}", flush=True)


def column(rows: list[dict[str, str]], system: str, arm: str) -> list[dict[str, str]]:
    found = [row for row in rows if row["system"] == system and row["arm"] == arm]
    found.sort(key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in found]
    if seeds != list(SEEDS):
        raise SystemExit(f"{system} {arm} seeds are {seeds}")
    return found


def direction_worse(median_b: float, median_a: float, u_b: float, n: int) -> bool:
    if median_b > median_a:
        return True
    if median_b < median_a:
        return False
    return u_b > (n * n / 2.0)


def direction_better(median_b: float, median_a: float, u_b: float, n: int) -> bool:
    if median_b < median_a:
        return True
    if median_b > median_a:
        return False
    return u_b < (n * n / 2.0)


def fmt(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}"


def fmt_p(value: float) -> str:
    return f"{value:.8f}"


def reference_token(spec: System, rows: list[dict[str, str]]) -> str:
    if spec.reference is not None:
        return spec.reference
    tokens = [row["best"] for row in rows if row["arm"] in ("A", "B")]
    return "pooled:" + pooled_token(tokens)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(rows) != len(list(jobs())):
        raise SystemExit(f"expected {len(list(jobs()))} rows, found {len(rows)}")
    keys = [(row["system"], row["seed"], row["arm"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate rows")
    commits_a = {row["commit"] for row in rows if row["arm"] == "A"}
    trees_a = {row["tree"] for row in rows if row["arm"] == "A"}
    commits_b = {row["commit"] for row in rows if row["arm"] == "B"}
    trees_b = {row["tree"] for row in rows if row["arm"] == "B"}
    commits_a2 = {row["commit"] for row in rows if row["arm"] == "A2"}
    trees_a2 = {row["tree"] for row in rows if row["arm"] == "A2"}
    if len(commits_a) != 1 or len(trees_a) != 1 or len(commits_b) != 1 or len(trees_b) != 1:
        raise SystemExit("arm commit or tree is not unique")
    if commits_a != commits_a2 or trees_a != trees_a2:
        raise SystemExit("A2 commit or tree differs from A")
    if trees_a == trees_b:
        raise SystemExit("A and B trees match")
    machines = {row["machine_id_hash"] for row in rows}
    cores = {row["core"] for row in rows}
    if len(machines) != 1 or cores != {"0"}:
        raise SystemExit("machine id or core is not uniform")
    n = len(SEEDS)
    bands: dict[str, dict[str, float]] = {}
    for name in AA_SYSTEMS:
        left = [float(row["outside_s"]) for row in column(rows, name, "A")]
        right = [float(row["outside_s"]) for row in column(rows, name, "A2")]
        med_a = type7(left)
        med_a2 = type7(right)
        band_s = abs(med_a2 - med_a)
        bands[name] = {
            "med_a": med_a,
            "med_a2": med_a2,
            "band_s": band_s,
            "rel": (band_s / med_a) if med_a > 0.0 else math.inf,
        }
    summary = []
    for spec in SYSTEMS:
        arm_a = column(rows, spec.name, "A")
        arm_b = column(rows, spec.name, "B")
        hits_a = sum(int(row["hit"]) for row in arm_a)
        hits_b = sum(int(row["hit"]) for row in arm_b)
        calls_a = [
            censored_calls(int(row["calls"]), int(row["hit"]), spec.budget) for row in arm_a
        ]
        calls_b = [
            censored_calls(int(row["calls"]), int(row["hit"]), spec.budget) for row in arm_b
        ]
        u_calls, p_calls = mannwhitney_exact(
            [float(v) for v in calls_a], [float(v) for v in calls_b]
        )
        wall_a = [float(row["wall_s"]) for row in arm_a]
        wall_b = [float(row["wall_s"]) for row in arm_b]
        pot_a = [float(row["pot_s"]) for row in arm_a]
        pot_b = [float(row["pot_s"]) for row in arm_b]
        out_a = [float(row["outside_s"]) for row in arm_a]
        out_b = [float(row["outside_s"]) for row in arm_b]
        u_out, p_out = mannwhitney_exact(out_a, out_b)
        med_calls_a = type7([float(v) for v in calls_a])
        med_calls_b = type7([float(v) for v in calls_b])
        med_out_a = type7(out_a)
        med_out_b = type7(out_b)
        anchor = anchor_of(spec.name)
        band = bands[anchor]
        if spec.name == anchor:
            limit = band["band_s"]
        elif math.isfinite(band["rel"]) and med_out_a > 0.0:
            limit = band["rel"] * med_out_a
        else:
            limit = 0.0
        delta = med_out_b - med_out_a
        calls_worse = p_calls < P_CALLS and direction_worse(med_calls_b, med_calls_a, u_calls, n)
        calls_better = p_calls < P_CALLS and direction_better(
            med_calls_b, med_calls_a, u_calls, n
        )
        outside_worse = delta > limit
        outside_better = -delta > limit
        identical = sum(a["best"] == b["best"] for a, b in zip(arm_a, arm_b))
        if hits_b < hits_a or calls_worse or outside_worse:
            verdict = "LOSES"
        elif hits_b > hits_a or calls_better or outside_better:
            verdict = "WINS"
        else:
            verdict = "TIE"
        summary.append(
            {
                "system": spec.name,
                "budget": str(spec.budget),
                "reference": reference_token(spec, arm_a + arm_b),
                "n": str(n),
                "hits_a": str(hits_a),
                "hits_b": str(hits_b),
                "fisher_p": fmt_p(
                    fisher_exact_two_sided(hits_a, n - hits_a, hits_b, n - hits_b)
                ),
                "median_calls_a": fmt(med_calls_a),
                "median_calls_b": fmt(med_calls_b),
                "mw_u_calls": f"{u_calls:.4f}",
                "mw_p_calls": fmt_p(p_calls),
                "median_wall_a": fmt(type7(wall_a)),
                "median_wall_b": fmt(type7(wall_b)),
                "median_pot_a": fmt(type7(pot_a)),
                "median_pot_b": fmt(type7(pot_b)),
                "median_outside_a": fmt(med_out_a),
                "median_outside_b": fmt(med_out_b),
                "mw_u_outside": f"{u_out:.4f}",
                "mw_p_outside": fmt_p(p_out),
                "aa_anchor": anchor,
                "anchor_band_s": f"{band['band_s']:.6f}",
                "outside_delta_s": f"{delta:.6f}",
                "outside_limit_s": f"{limit:.6f}",
                "best_identical": f"{identical}/{n}",
                "verdict": verdict,
            }
        )
    return summary


def render_gate(rows: list[dict[str, str]], summary: list[dict[str, str]]) -> str:
    commit_a = next(row["commit"] for row in rows if row["arm"] == "A")
    tree_a = next(row["tree"] for row in rows if row["arm"] == "A")
    commit_b = next(row["commit"] for row in rows if row["arm"] == "B")
    tree_b = next(row["tree"] for row in rows if row["arm"] == "B")
    machine = next(iter({row["machine_id_hash"] for row in rows}))
    loses = [row["system"] for row in summary if row["verdict"] == "LOSES"]
    overall = "default-on OK" if not loses else "default-on NO"
    lines = [
        "# nmpins5 default-on gate",
        "",
        f"A commit `{commit_a}` tree `{tree_a}`.",
        f"B commit `{commit_b}` tree `{tree_b}`.",
        "A2 is a second build of A, same commit and tree, run in the same session.",
        f"Machine id sha256 `{machine}`. Core 0 on every row.",
        "",
        "Hits use `scripts/target_tol.py` `energy_hits` with tolerance 1e-3.",
        "Lennard-Jones references are the Cambridge energies in `scripts/gmin_panel/run_panel.py`.",
        "Morse references are the Cambridge rows in `examples/fixtures/morse_ccd.csv`.",
        "TIP4P references for n <= 21, including n = 6, are the Cambridge headers in `data/tip4p_ccd.csv`.",
        "(H2O)25 uses the pooled A/B minimum, as `scripts/tip4p_h2h/run_h2h.py` does when the header is absent.",
        "Call medians and the Mann-Whitney test censor a miss at that system's budget.",
        "The Mann-Whitney p is the exact two-sided midrank test (twice the smaller tail, capped at 1).",
        "Fisher p is the exact two-sided test on the hit counts.",
        "Medians are Hyndman-Fan type 7.",
        "B loses if it has fewer hits, or its calls are worse at p < 0.05, or its median outside_s exceeds A by more than the A/A band.",
        "The band is the absolute median outside_s gap of A versus A2 on the anchor: LJ75 for Lennard-Jones, M80 rho 6 for Morse, TIP4P6 for the hexamer, and TIP4P15 for the other TIP4P cells.",
        "A non-anchor limit scales that anchor gap by this system's median A outside_s over the anchor's median A outside_s.",
        "Best identity compares the recorded best token of A and B on each seed.",
        "",
        "| system | hits A/B | fisher p | median calls A/B | mw p calls | median wall A/B | median pot A/B | median outside A/B | mw p outside | delta s | limit s | best identical | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            "| {system} | {hits_a}/{hits_b} | {fisher_p} | {median_calls_a}/{median_calls_b} | {mw_p_calls} | "
            "{median_wall_a}/{median_wall_b} | {median_pot_a}/{median_pot_b} | "
            "{median_outside_a}/{median_outside_b} | {mw_p_outside} | {outside_delta_s} | "
            "{outside_limit_s} | {best_identical} | {verdict} |".format(**row)
        )
    lines.extend(["", "## A/A anchors", ""])
    lines.append("| anchor | median outside A | median outside A2 | band s |")
    lines.append("|---|---:|---:|---:|")
    seen = []
    for row in summary:
        if row["aa_anchor"] in seen:
            continue
        if row["system"] != row["aa_anchor"]:
            continue
        seen.append(row["aa_anchor"])
        arm_a = [float(item["outside_s"]) for item in column(rows, row["system"], "A")]
        arm_2 = [float(item["outside_s"]) for item in column(rows, row["system"], "A2")]
        lines.append(
            f"| {row['system']} | {fmt(type7(arm_a))} | {fmt(type7(arm_2))} | {row['anchor_band_s']} |"
        )
    lines.extend(["", f"Overall: {overall}", ""])
    if loses:
        lines.append("Losing systems: " + ", ".join(loses) + ".")
        lines.append("")
    text = "\n".join(lines)
    if "\r" in text:
        raise SystemExit("gate text contains CR")
    return text


def report(args: argparse.Namespace) -> None:
    self_check()
    rows = read_rows(Path(args.rows))
    # Recompute hits from the recorded energies so the verdict follows the
    # cited references even if a placeholder was left in the file. The stored
    # hit must already match; report does not rewrite the rows.
    check = [dict(row) for row in rows]
    apply_hits(check)
    for old, new in zip(rows, check):
        if old["hit"] != new["hit"]:
            raise SystemExit(f"stored hit disagrees for {old['system']} {old['seed']} {old['arm']}")
    summary = summarize(rows)
    write_csv(Path(args.summary), SUMMARY_FIELDS, summary)
    gate = render_gate(rows, summary)
    gate_path = Path(args.gate)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    with gate_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(gate)
        handle.flush()
        os.fsync(handle.fileno())


def self_check() -> None:
    if len(SYSTEMS) != 13 or len(AA_SYSTEMS) != 4:
        raise AssertionError("matrix size")
    if [anchor_of(name) for name in ("lj98", "m38r10", "tip4p21", "tip4p6", "lj75")] != [
        "lj75",
        "m80r6",
        "tip4p15",
        "tip4p6",
        "lj75",
    ]:
        raise AssertionError("anchors")
    planned = list(jobs())
    if len(planned) != 360:
        raise AssertionError(len(planned))
    by_seed: dict[int, list[tuple[str, str]]] = {}
    for spec, seed, arm in planned:
        by_seed.setdefault(seed, []).append((spec.name, arm))
    if by_seed[2000][:2] != [("lj38", "A"), ("lj38", "B")]:
        raise AssertionError("even atomic order")
    if ("lj75", "A") not in by_seed[2000] or ("lj75", "A2") not in by_seed[2000]:
        raise AssertionError("even aa missing")
    lj75_even = [arm for name, arm in by_seed[2000] if name == "lj75"]
    lj75_odd = [arm for name, arm in by_seed[2001] if name == "lj75"]
    lj38_odd = [arm for name, arm in by_seed[2001] if name == "lj38"]
    if lj75_even != ["A", "A2", "B"] or lj75_odd != ["A2", "B", "A"] or lj38_odd != ["B", "A"]:
        raise AssertionError((lj75_even, lj75_odd, lj38_odd))
    if abs(type7([float(v) for v in range(1, 13)]) - 6.5) > 1e-12:
        raise AssertionError("type7")
    if abs(fisher_exact_two_sided(5, 0, 0, 5) - (2.0 / 252.0)) > 1e-12:
        raise AssertionError("fisher")
    if fisher_exact_two_sided(12, 0, 12, 0) != 1.0:
        raise AssertionError("fisher tie")
    _u, p = mannwhitney_exact([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    if abs(p - (2.0 / 252.0)) > 1e-12:
        raise AssertionError(p)
    _u, p_tie = mannwhitney_exact([1, 1, 1], [1, 1, 1])
    if p_tie != 1.0:
        raise AssertionError(p_tie)
    if not energy_hits(-173.928427, -173.928427, HIT_TOL):
        raise AssertionError("hit")
    if energy_hits(-173.927426, -173.928427, HIT_TOL):
        raise AssertionError("near miss should miss")
    if energy_hits(float("nan"), -1.0, HIT_TOL):
        raise AssertionError("nan")
    sample = "\n".join(
        [
            "calls: 12",
            "charged: 14",
            "best: -1.5000000000",
            "wall_seconds: 1.250000",
            "pot_s: 1.000000",
            "outside_s: 0.250000",
            "arm kick draws 1 accepts 1 best -1.000000",
            "h2h_row system=lj38 code=anneal seed=2000 budget=400000 hit=0 "
            "evals_to_hit=400000 charged=14 best=-173.9284270000 wall_s=1.250000 "
            "pot_s=1.000000 outside_s=0.250000 basins=2",
        ]
    )
    parsed = parse_log(sample)
    if parsed["calls"] != "12" or parsed["best"] != "-173.9284270000":
        raise AssertionError(parsed)
    if parsed["wall_s"] != "1.250000" or parsed["outside_s"] != "0.250000":
        raise AssertionError(parsed)
    if censored_calls(12, 0, 400000) != 400000 or censored_calls(12, 1, 400000) != 12:
        raise AssertionError("censor")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gate.py")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--binary-a", required=True)
    run.add_argument("--binary-b", required=True)
    run.add_argument("--binary-a2", required=True)
    run.add_argument("--commit-a", required=True)
    run.add_argument("--tree-a", required=True)
    run.add_argument("--commit-b", required=True)
    run.add_argument("--tree-b", required=True)
    run.add_argument("--commit-a2", required=True)
    run.add_argument("--tree-a2", required=True)
    run.add_argument("--rows", required=True)
    run.add_argument("--sessions", required=True)
    run.add_argument("--logs", required=True)
    report_p = sub.add_parser("report")
    report_p.add_argument("--rows", required=True)
    report_p.add_argument("--summary", required=True)
    report_p.add_argument("--gate", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run_campaign(args)
    elif args.cmd == "report":
        report(args)
    else:
        raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
