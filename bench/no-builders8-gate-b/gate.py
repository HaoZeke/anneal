#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Default-on paired gate for the no-builders series, part b.

One session, one process, core 0. Arm A is the base commit and arm B is
that commit plus the series. Seeds 2000-2011 run in ABBA order: an even
offset is A then B and an odd offset is B then A. The search is the
example's default command, ``run SYSTEM SEED``, with no ``H2H_*`` variable
set. Budgets and references come from the paper panel. A hit is a finite
best energy at or below the reference plus 1e-3. A miss is stored at the
budget and is ranked strictly after every hit.

Cells are LJ38, LJ75, LJ98, every Morse cluster in the paper panel, and
every molecular system in the paper panel. The panel's molecular list is
empty, so those cells contribute no runs.

``check`` locks the schedule and the verdict rule. ``run`` appends one row
per finished search. ``report`` writes the verdict from those rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE_COMMIT = "61fa67cf380e46876b569da45e7cc091647a775b"
ARM_B_TREE = "39ac5e0642beb3e7874a340d2767aa151fa95eac"
SEED0 = 2000
SEEDS = tuple(range(SEED0, SEED0 + 12))
LJ_CELLS = ("lj38", "lj75", "lj98")
HIT_TOL = 1e-3
JOB_TIMEOUT_S = 12 * 60 * 60
CORE = "0"
OUTSIDE_RATIO_LIMIT = 1.1
CALLS_P_LIMIT = 0.05

FIELDS = (
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
    "machine_id",
    "base_commit",
    "base_tree",
    "library_commit",
    "library_tree",
    "harness_blob",
    "reference",
    "start_sha256",
    "session",
)

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DOMAIN_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "OMP_MAX_ACTIVE_LEVELS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_NESTED": "FALSE",
    "POLARS_MAX_THREADS": "1",
    "TOKIO_WORKER_THREADS": "1",
    "RUST_TEST_THREADS": "1",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_panel():
    root = repo_root()
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "scripts" / "gmin_panel"))
    import run_panel  # noqa: WPS433

    return run_panel


def paper_cells(panel) -> list[tuple]:
    """LJ38, LJ75, LJ98, then every Morse and molecular paper-panel system."""
    chosen = []
    for row in panel.SYSTEMS:
        name, kind = row[0], row[1]
        if name in LJ_CELLS or kind in ("morse", "molecular"):
            chosen.append(row)
    names = [row[0] for row in chosen]
    if names[:3] != list(LJ_CELLS):
        raise SystemExit(f"cell order {names}")
    if any(row[1] == "molecular" for row in chosen):
        molecular = [row[0] for row in chosen if row[1] == "molecular"]
        raise SystemExit(f"paper panel molecular systems need a default runner: {molecular}")
    if not any(row[1] == "morse" for row in chosen):
        raise SystemExit("paper panel has no Morse cluster")
    for row in chosen:
        name = row[0]
        if name not in panel.PAPER_BUDGETS:
            raise SystemExit(f"{name} has no paper budget")
        if row[5] not in ("h2h_gmin_proof", "h2h_morse") or row[6] != row[5]:
            raise SystemExit(f"{name} driver {row[5]} {row[6]} is not the default path")
    return chosen


def arm_sequence(seed: int) -> tuple[str, str]:
    """Even offsets run A then B. Odd offsets run B then A, so the arms read ABBA."""
    if (seed - SEED0) % 2 == 0:
        return ("A", "B")
    return ("B", "A")


def schedule(cells: list[tuple]) -> list[tuple[str, int, int, str]]:
    jobs = []
    for row in cells:
        system = row[0]
        for seed in SEEDS:
            for slot, arm in enumerate(arm_sequence(seed)):
                jobs.append((system, seed, slot, arm))
    return jobs


def assert_schedule(cells: list[tuple]) -> None:
    jobs = schedule(cells)
    if len(jobs) != len(cells) * len(SEEDS) * 2:
        raise SystemExit("schedule length")
    opening = [arm for _system, _seed, _slot, arm in jobs if _system == cells[0][0]][:4]
    if opening != ["A", "B", "B", "A"]:
        raise SystemExit(f"ABBA opening {opening}")
    for row in cells:
        system = row[0]
        pairs = [(slot, arm) for name, _seed, slot, arm in jobs if name == system]
        if [arm for _slot, arm in pairs[:4]] != ["A", "B", "B", "A"]:
            raise SystemExit(f"{system} ABBA opening {pairs[:4]}")
        for slot in (0, 1):
            for arm in ("A", "B"):
                count = sum(pair == (slot, arm) for pair in pairs)
                if count != len(SEEDS) // 2:
                    raise SystemExit(f"ABBA balance {system} slot {slot} arm {arm} count {count}")
        if len(pairs) != len(SEEDS) * 2:
            raise SystemExit(f"{system} schedule length")


def type7(values: list[float], p: float) -> float:
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


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided exact p, summing tables no more likely than the observed."""
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def probability(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = probability(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    total = sum(probability(k) for k in range(lo, hi + 1) if probability(k) <= observed * (1 + 1e-7))
    return min(1.0, total)


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """U of ``left`` and the two-sided p, with the tie and continuity corrections."""
    n1, n2 = len(left), len(right)
    if n1 < 1 or n2 < 1:
        raise ValueError("mann-whitney needs two samples")
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    index = 0
    while index < n:
        end = index
        while end + 1 < n and pooled[end + 1] == pooled[index]:
            end += 1
        rank[pooled[index]] = (index + end + 2) / 2.0
        width = end - index + 1
        ties += width**3 - width
        index = end + 1
    u = sum(rank[value] for value in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0
    deviate = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return u, min(1.0, math.erfc(deviate / math.sqrt(var) / math.sqrt(2.0)))


def is_hit(best: float, reference: float) -> bool:
    return math.isfinite(best) and math.isfinite(reference) and best <= reference + HIT_TOL


def rank_key(row: dict[str, str]) -> float:
    """Misses sort after every hit, including a hit recorded at the full budget."""
    budget = float(row["budget"])
    if row["hit"] == "1":
        return float(row["calls"])
    return budget + 1.0


def outside_per_call(row: dict[str, str]) -> float:
    charged = int(row["charged"])
    if charged <= 0:
        raise SystemExit(f"{row['system']} {row['arm']} seed {row['seed']} charged {charged}")
    return float(row["outside_s"]) / charged


def full_precision(value: float) -> str:
    text = format(value, ".17g")
    if float(text) != value:
        raise SystemExit("best does not round-trip")
    return text


def best_bits(value: float) -> str:
    import struct

    return f"{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes()
    return hashlib.sha256(raw).hexdigest()


def git_output(tree: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(tree), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def require_object(repo: Path, object_id: str) -> None:
    done = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", object_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise SystemExit(f"git cat-file -e {object_id} failed")


def harness_blob(repo: Path) -> str:
    path = Path(__file__).resolve()
    done = subprocess.run(
        ["git", "-C", str(repo), "hash-object", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    blob = done.stdout.strip()
    require_object(repo, blob)
    return blob


def clean_env() -> dict[str, str]:
    keep = (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "RUSTUP_HOME",
        "CARGO_HOME",
    )
    env = {key: os.environ[key] for key in keep if key in os.environ and os.environ[key] != ""}
    env.update(THREAD_ENV)
    env["LJOS_SEAT"] = "benchbot"
    leaked = [key for key in env if key.startswith("H2H_") or key.startswith("ANNEAL_")]
    if leaked:
        raise SystemExit(f"default path still has {leaked}")
    for key, value in THREAD_ENV.items():
        if env.get(key) != value:
            raise SystemExit(f"{key} is {env.get(key)}")
    if env.get("LJOS_SEAT") != "benchbot":
        raise SystemExit("seat")
    return env


def run_command(command: list[str], cwd: Path, env: dict[str, str], log: Path, timeout: int) -> subprocess.CompletedProcess:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(" ".join(command) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            handle.write(f"timeout {timeout}\n")
            raise SystemExit(f"timeout after {timeout}s: {' '.join(command[:4])}") from None
        handle.write(f"exit {proc.returncode}\n")
        handle.write(stdout)
        if stderr:
            handle.write("\n[stderr]\n")
            handle.write(stderr)
        handle.flush()
        os.fsync(handle.fileno())
    if proc.returncode != 0:
        tail = (stderr or stdout)[-2000:]
        raise SystemExit(f"exit {proc.returncode}: {' '.join(command[:6])}\n{tail}")
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def query_budget(binary: Path, tree: Path, system: str, env: dict[str, str], log: Path) -> tuple[int, float]:
    done = run_command([str(binary), "budget", system], tree, env, log, 120)
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"{system} budget output is short")
    return int(lines[0]), float(lines[1])


def write_start(binary: Path, tree: Path, system: str, seed: int, dest: Path, env: dict[str, str], log: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_command([str(binary), "coords", system, str(seed), str(dest)], tree, env, log, 120)
    data = dest.read_bytes()
    if not data:
        raise SystemExit(f"{system} seed {seed} wrote an empty start")
    return hashlib.sha256(data).hexdigest()


def parse_result(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line and not line.startswith("arm ")]
    for index, line in enumerate(lines):
        if line.startswith("system,code,seed,budget,hit,"):
            return next(csv.DictReader(io.StringIO(line + "\n" + lines[index + 1])))
    raise SystemExit("search stdout has no result row")


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        missing = [field for field in FIELDS if field not in row]
        if missing:
            raise SystemExit(f"row is missing {missing}")
    return rows


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def search_env_ok(env: dict[str, str]) -> None:
    if any(key.startswith("H2H_") for key in env):
        raise SystemExit("H2H_ is set")


def execute(args: argparse.Namespace) -> None:
    panel = load_panel()
    cells = paper_cells(panel)
    assert_schedule(cells)
    expected = schedule(cells)
    repo = repo_root()
    env = clean_env()
    search_env_ok(env)
    blob = harness_blob(repo)
    machine = machine_id()
    arms = {
        "A": (args.arm_a_bin.resolve(), args.arm_a_tree.resolve()),
        "B": (args.arm_b_bin.resolve(), args.arm_b_tree.resolve()),
    }
    identity = {}
    for label, (bin_dir, tree) in arms.items():
        if git_output(tree, "status", "--porcelain"):
            raise SystemExit(f"arm {label} tree is dirty")
        head = git_output(tree, "rev-parse", "HEAD")
        tree_id = git_output(tree, "rev-parse", "HEAD^{tree}")
        require_object(repo, head)
        require_object(repo, tree_id)
        identity[label] = (head, tree_id)
    if identity["A"][0] != BASE_COMMIT:
        raise SystemExit(f"arm A is {identity['A'][0]}")
    if identity["B"][1] != ARM_B_TREE:
        raise SystemExit(f"arm B tree is {identity['B'][1]}")
    base_commit, base_tree = identity["A"]
    rows = load_rows(args.rows)
    done = [(row["system"], int(row["seed"]), row["arm"]) for row in rows]
    prefix = [(system, seed, arm) for system, seed, _slot, arm in expected[: len(done)]]
    if done != prefix:
        raise SystemExit("rows are not a prefix of the ABBA schedule")
    if rows:
        session = rows[0]["session"]
        if any(row["session"] != session for row in rows):
            raise SystemExit("rows come from more than one session")
        if any(row["machine_id"] != machine for row in rows):
            raise SystemExit("rows come from more than one machine")
        if any(row["harness_blob"] != blob for row in rows):
            raise SystemExit("rows name a different harness blob")
    else:
        session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for row in cells:
        system, kind, _atoms, reference, _rho, _driver, binary_name = row
        budget = panel.PAPER_BUDGETS[system]
        for label, (bin_dir, tree) in arms.items():
            binary = bin_dir / binary_name
            got_budget, got_reference = query_budget(
                binary, tree, system, env, args.logs / "budget" / f"{system}-{label}.log"
            )
            if got_budget != budget or abs(got_reference - float(reference)) > 1e-9:
                raise SystemExit(
                    f"{system} arm {label} budget {got_budget} reference {got_reference} "
                    f"is not the paper budget {budget} {reference}"
                )
        for seed in SEEDS:
            starts = {}
            for label, (bin_dir, tree) in arms.items():
                dest = args.logs / "starts" / f"{system}-{seed}-{label}.xyz"
                starts[label] = write_start(
                    bin_dir / binary_name,
                    tree,
                    system,
                    seed,
                    dest,
                    env,
                    args.logs / "coords" / f"{system}-{seed}-{label}.log",
                )
            if starts["A"] != starts["B"]:
                raise SystemExit(f"{system} seed {seed} starts differ")
            for slot, label in enumerate(arm_sequence(seed)):
                if (system, seed, label) in set(done):
                    continue
                bin_dir, tree = arms[label]
                command = ["taskset", "-c", CORE, str(bin_dir / binary_name), "run", system, str(seed)]
                if any(part.startswith("H2H_") for part in command):
                    raise SystemExit("command sets H2H_")
                done_run = run_command(
                    command,
                    tree,
                    env,
                    args.logs / "runs" / f"{system}-{label}-{seed}.log",
                    JOB_TIMEOUT_S,
                )
                parsed = parse_result(done_run.stdout)
                if parsed["system"] != system or int(parsed["seed"]) != seed:
                    raise SystemExit(f"{system} seed {seed} returned {parsed['system']} {parsed['seed']}")
                if int(parsed["budget"]) != budget:
                    raise SystemExit(f"{system} ran budget {parsed['budget']}")
                best = float(parsed["best"])
                hit = is_hit(best, float(reference))
                calls = int(parsed["evals_to_hit"]) if hit else budget
                charged = int(parsed["charged"])
                wall = float(parsed["wall_s"])
                inside = float(parsed["pot_s"])
                outside = max(0.0, wall - inside)
                if kind == "morse" and "outside_s" in parsed:
                    printed = float(parsed["outside_s"])
                    if abs(printed - outside) > 2e-3:
                        raise SystemExit(f"{system} outside {printed} disagrees with wall-inside {outside}")
                head, tree_id = identity[label]
                if "head" in parsed and parsed["head"] and parsed["head"] != head:
                    raise SystemExit(f"{system} arm {label} reported head {parsed['head']}")
                if "tree" in parsed and parsed["tree"] and parsed["tree"] != tree_id:
                    raise SystemExit(f"{system} arm {label} reported tree {parsed['tree']}")
                if parsed.get("pins") not in (None, "", "0"):
                    raise SystemExit(f"{system} arm {label} pins {parsed.get('pins')}")
                record = {
                    "system": system,
                    "arm": label,
                    "seed": str(seed),
                    "slot": str(slot),
                    "core": CORE,
                    "hit": "1" if hit else "0",
                    "calls": str(calls),
                    "charged": str(charged),
                    "wall": f"{wall:.6f}",
                    "inside_s": f"{inside:.6f}",
                    "outside_s": f"{outside:.6f}",
                    "best": full_precision(best),
                    "best_bits": best_bits(best),
                    "budget": str(budget),
                    "machine_id": machine,
                    "base_commit": base_commit,
                    "base_tree": base_tree,
                    "library_commit": head,
                    "library_tree": tree_id,
                    "harness_blob": blob,
                    "reference": f"{float(reference):.10f}",
                    "start_sha256": starts[label],
                    "session": session,
                }
                if float(record["best"]) != best or best_bits(float(record["best"])) != record["best_bits"]:
                    raise SystemExit("stored best does not round-trip")
                if record["hit"] == "1" and not is_hit(float(record["best"]), float(record["reference"])):
                    raise SystemExit("stored hit disagrees with the reference")
                append_row(args.rows, record)
                done.append((system, seed, label))
                print(
                    f"{system} {label} seed {seed} slot {slot} hit {record['hit']} "
                    f"calls {record['calls']} charged {record['charged']} wall {record['wall']}",
                    flush=True,
                )
    final = load_rows(args.rows)
    if [(row["system"], int(row["seed"]), int(row["slot"]), row["arm"]) for row in final] != [
        (system, seed, slot, arm) for system, seed, slot, arm in expected
    ]:
        raise SystemExit("finished rows do not match the schedule")
    print(f"session {session} rows {len(final)}", flush=True)


def grouped(rows: list[dict[str, str]], system: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    arm_a = [row for row in rows if row["system"] == system and row["arm"] == "A"]
    arm_b = [row for row in rows if row["system"] == system and row["arm"] == "B"]
    arm_a.sort(key=lambda row: int(row["seed"]))
    arm_b.sort(key=lambda row: int(row["seed"]))
    return arm_a, arm_b


def ratio_of(median_b: float, median_a: float) -> float:
    if median_a == 0.0:
        return 1.0 if median_b == 0.0 else math.inf
    return median_b / median_a


def judge(arm_a: list[dict[str, str]], arm_b: list[dict[str, str]]) -> dict[str, float | int | bool | str]:
    hits_a = sum(row["hit"] == "1" for row in arm_a)
    hits_b = sum(row["hit"] == "1" for row in arm_b)
    fisher_p = fisher(hits_a, len(arm_a), hits_b, len(arm_b))
    keys_a = [rank_key(row) for row in arm_a]
    keys_b = [rank_key(row) for row in arm_b]
    u_stat, calls_p = mann_whitney(keys_a, keys_b)
    calls_loss = calls_p < CALLS_P_LIMIT and sum(keys_b) > sum(keys_a)
    outside_a = [outside_per_call(row) for row in arm_a]
    outside_b = [outside_per_call(row) for row in arm_b]
    median_outside_a = type7(outside_a, 0.5)
    median_outside_b = type7(outside_b, 0.5)
    outside_ratio = ratio_of(median_outside_b, median_outside_a)
    outside_loss = outside_ratio > OUTSIDE_RATIO_LIMIT
    hit_loss = hits_b < hits_a
    reasons = []
    if hit_loss:
        reasons.append("fewer hits")
    if calls_loss:
        reasons.append("calls")
    if outside_loss:
        reasons.append("outside time per call")
    return {
        "hits_a": hits_a,
        "hits_b": hits_b,
        "n": len(arm_a),
        "fisher_p": fisher_p,
        "u": u_stat,
        "calls_p": calls_p,
        "calls_median_a": type7([float(row["calls"]) for row in arm_a], 0.5),
        "calls_median_b": type7([float(row["calls"]) for row in arm_b], 0.5),
        "wall_median_a": type7([float(row["wall"]) for row in arm_a], 0.5),
        "wall_median_b": type7([float(row["wall"]) for row in arm_b], 0.5),
        "outside_median_a": median_outside_a,
        "outside_median_b": median_outside_b,
        "outside_ratio": outside_ratio,
        "hit_loss": hit_loss,
        "calls_loss": calls_loss,
        "outside_loss": outside_loss,
        "reasons": ", ".join(reasons),
    }


def fmt_p(value: float) -> str:
    if value == 0.0 or value == 1.0:
        return f"{value:.1f}"
    return f"{value:.6g}"


def fmt_ratio(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def verdict_table(rows: list[dict[str, str]], cells: list[tuple]) -> tuple[str, list[str], list[str]]:
    lines = []
    failing = []
    for row in cells:
        system = row[0]
        arm_a, arm_b = grouped(rows, system)
        if len(arm_a) != len(SEEDS) or len(arm_b) != len(SEEDS):
            raise SystemExit(f"{system} has {len(arm_a)} and {len(arm_b)} rows")
        if [int(item["seed"]) for item in arm_a] != list(SEEDS) or [int(item["seed"]) for item in arm_b] != list(SEEDS):
            raise SystemExit(f"{system} seeds are not 2000-2011 on both arms")
        result = judge(arm_a, arm_b)
        state = "FIX" if result["reasons"] else "PASS"
        if state == "FIX":
            failing.append(f"{system} ({result['reasons']})")
        lines.append(
            f"| {system} | {result['hits_a']}/{result['n']} | {result['hits_b']}/{result['n']} | "
            f"{fmt_p(float(result['fisher_p']))} | {result['calls_median_a']:.1f} | {result['calls_median_b']:.1f} | "
            f"{result['u']:.1f} | {fmt_p(float(result['calls_p']))} | {result['wall_median_a']:.6f} | "
            f"{result['wall_median_b']:.6f} | {result['outside_median_a']:.6e} | {result['outside_median_b']:.6e} | "
            f"{fmt_ratio(float(result['outside_ratio']))} | {state} |"
        )
    overall = "FIX" if failing else "PASS"
    return overall, failing, lines


def per_seed_lines(rows: list[dict[str, str]], system: str) -> list[str]:
    arm_a, arm_b = grouped(rows, system)
    by_a = {int(row["seed"]): row for row in arm_a}
    by_b = {int(row["seed"]): row for row in arm_b}
    lines = ["| seed | hit A | hit B | calls A | calls B |", "| --- | ---: | ---: | ---: | ---: |"]
    for seed in SEEDS:
        left, right = by_a[seed], by_b[seed]
        lines.append(
            f"| {seed} | {left['hit']} | {right['hit']} | {left['calls']} | {right['calls']} |"
        )
    return lines


def write_report(args: argparse.Namespace) -> None:
    panel = load_panel()
    cells = paper_cells(panel)
    assert_schedule(cells)
    rows = load_rows(args.rows)
    expected = schedule(cells)
    observed = [(row["system"], int(row["seed"]), int(row["slot"]), row["arm"]) for row in rows]
    if observed != expected:
        raise SystemExit(f"rows {len(rows)} do not match the schedule of {len(expected)}")
    counts: dict[tuple[str, str], int] = {}
    for _system, _seed, slot, arm in observed:
        counts[(str(slot), arm)] = counts.get((str(slot), arm), 0) + 1
    for slot in ("0", "1"):
        for arm in ("A", "B"):
            if counts.get((slot, arm), 0) != len(cells) * len(SEEDS) // 2:
                raise SystemExit(f"unbalanced slot {slot} arm {arm}")
    sessions = {row["session"] for row in rows}
    machines = {row["machine_id"] for row in rows}
    blobs = {row["harness_blob"] for row in rows}
    bases = {(row["base_commit"], row["base_tree"]) for row in rows}
    if len(sessions) != 1 or len(machines) != 1 or len(blobs) != 1 or len(bases) != 1:
        raise SystemExit("provenance is not unique")
    repo = repo_root()
    blob = next(iter(blobs))
    require_object(repo, blob)
    for row in rows:
        require_object(repo, row["library_commit"])
        require_object(repo, row["library_tree"])
        require_object(repo, row["base_commit"])
        require_object(repo, row["base_tree"])
        require_object(repo, row["harness_blob"])
        if row["base_commit"] != BASE_COMMIT:
            raise SystemExit("base commit drifted")
        if row["arm"] == "B" and row["library_tree"] != ARM_B_TREE:
            raise SystemExit("arm B tree drifted")
        if row["arm"] == "A" and row["library_commit"] != BASE_COMMIT:
            raise SystemExit("arm A commit drifted")
        if float(row["best"]) != float(full_precision(float(row["best"]))):
            raise SystemExit("best precision drifted")
        if best_bits(float(row["best"])) != row["best_bits"]:
            raise SystemExit("best bits drifted")
    libraries: dict[str, tuple[str, str]] = {}
    for row in rows:
        ident = (row["library_commit"], row["library_tree"])
        previous = libraries.setdefault(row["arm"], ident)
        if previous != ident:
            raise SystemExit(f"arm {row['arm']} library changed during the session")
    for row_cells in cells:
        system = row_cells[0]
        arm_a, arm_b = grouped(rows, system)
        for seed in SEEDS:
            left = next(item for item in arm_a if int(item["seed"]) == seed)
            right = next(item for item in arm_b if int(item["seed"]) == seed)
            if left["start_sha256"] != right["start_sha256"]:
                raise SystemExit(f"{system} seed {seed} starts differ")
    overall, failing, table = verdict_table(rows, cells)
    session = next(iter(sessions))
    machine = next(iter(machines))
    base_commit, base_tree = next(iter(bases))
    molecular = [row[0] for row in panel.SYSTEMS if row[1] == "molecular"]
    morse = [row[0] for row in cells if row[1] == "morse"]
    body = [
        "# No-builders8 default-on gate, part b",
        "",
        f"Verdict: **{overall}**",
        "",
        "Failing cells: " + (", ".join(failing) if failing else "none"),
        "",
        "## Rule",
        "",
        "FIX if B has fewer hits than A on any system, regardless of the Fisher p; "
        "or B is slower on calls to the minimum at two-sided Mann-Whitney p < 0.05; "
        "or the ratio of median outside-potential time per charged call, B over A, is above 1.1. "
        "Otherwise PASS.",
        "",
        "Hits are a finite best energy at or below the paper-panel reference plus 1e-3. "
        "Calls to the minimum are the example's count, and a miss is stored at the paper budget. "
        "The Mann-Whitney test ranks every miss strictly after every hit. "
        "Fisher's test is the two-sided exact test. Medians are Hyndman-Fan type 7. "
        "Outside time per call is `(wall - time inside the potential) / charged calls`.",
        "",
        "## Run",
        "",
        f"- Session: `{session}`",
        f"- Hashed machine id: `{machine}`",
        "- Core 0, one process at a time, thread pools at 1, no `H2H_*` variable.",
        "- Order: ABBA. Even seeds run A then B and odd seeds run B then A.",
        f"- Arm A library: `{libraries['A'][0]}` tree `{libraries['A'][1]}`",
        f"- Arm B library: `{libraries['B'][0]}` tree `{libraries['B'][1]}`",
        f"- Base commit `{base_commit}` tree `{base_tree}`",
        f"- Harness blob: `{blob}`",
        f"- Paper-panel molecular systems: {', '.join(molecular) if molecular else 'none'}",
        f"- Morse cells: {', '.join(morse)}",
        f"- Lennard-Jones cells: {', '.join(LJ_CELLS)}",
        f"- Seeds: {SEEDS[0]}-{SEEDS[-1]}, n = {len(SEEDS)} per arm",
        "",
        "## Schedule counts",
        "",
        "| slot | arm | rows |",
        "| --- | --- | ---: |",
    ]
    for slot in ("0", "1"):
        for arm in ("A", "B"):
            body.append(f"| {slot} | {arm} | {counts[(slot, arm)]} |")
    body.extend(
        [
            "",
            "## Systems",
            "",
            "| system | hits A | hits B | Fisher p | calls A | calls B | MW U | MW p | wall A | wall B | outside/call A | outside/call B | B/A | verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *table,
            "",
        ]
    )
    for row in cells:
        system = row[0]
        body.append(f"## {system} seeds")
        body.append("")
        body.extend(per_seed_lines(rows, system))
        body.append("")
    text = "\n".join(body).rstrip() + "\n"
    args.readme.parent.mkdir(parents=True, exist_ok=True)
    args.readme.write_text(text, encoding="utf-8", newline="\n")
    print(overall, flush=True)
    if failing:
        print("\n".join(failing), flush=True)


def fake_row(hit: int, calls: int, charged: int, outside: float, budget: int = 100) -> dict[str, str]:
    return {
        "hit": str(hit),
        "calls": str(calls if hit else budget),
        "charged": str(charged),
        "outside_s": f"{outside:.6f}",
        "budget": str(budget),
        "wall": "1.0",
        "best": "-1.0",
        "reference": "-1.0",
    }


def check() -> None:
    panel = load_panel()
    cells = paper_cells(panel)
    assert_schedule(cells)
    if [row[0] for row in cells] != ["lj38", "lj75", "lj98", "m55r6", "m55r14", "m80r6", "m80r14"]:
        raise SystemExit(f"unexpected cells {[row[0] for row in cells]}")
    if any(row[1] == "molecular" for row in panel.SYSTEMS):
        raise SystemExit("molecular list changed")
    observed = fisher(12, 12, 0, 12)
    if abs(observed - 7.39602301050679e-07) > 1e-18:
        raise SystemExit(f"fisher 12 vs 0 is {observed}")
    if fisher(12, 12, 12, 12) != 1.0 or fisher(11, 12, 12, 12) != 1.0:
        raise SystemExit("fisher ties")
    if not is_hit(-173.928427, -173.928427) or not is_hit(-173.927427, -173.928427):
        raise SystemExit("hit window")
    if is_hit(-173.927426, -173.928427) or is_hit(math.nan, -1.0) or is_hit(math.inf, -1.0):
        raise SystemExit("hit window rejects")
    same = [1.0, 1.0, 2.0]
    _u, tied = mann_whitney(same, list(same))
    if tied != 1.0:
        raise SystemExit("tied mann-whitney")
    equal_a = [fake_row(1, 10, 10, 1.0) for _ in range(12)]
    equal_b = [fake_row(1, 10, 10, 1.0) for _ in range(12)]
    if judge(equal_a, equal_b)["reasons"]:
        raise SystemExit("equal samples must pass")
    fewer = [fake_row(1, 10, 10, 1.0) for _ in range(11)] + [fake_row(0, 100, 100, 1.0)]
    judged = judge(equal_a, fewer)
    if not judged["hit_loss"] or judged["fisher_p"] != 1.0:
        raise SystemExit("directional hit loss")
    slower = [fake_row(1, 50, 50, 1.0) for _ in range(12)]
    judged = judge(equal_a, slower)
    if not judged["calls_loss"] or judged["calls_p"] >= CALLS_P_LIMIT:
        raise SystemExit("calls loss")
    faster = [fake_row(1, 5, 5, 1.0) for _ in range(12)]
    judged = judge(equal_a, faster)
    if judged["calls_loss"] or judged["hit_loss"]:
        raise SystemExit("a faster arm is not a calls loss")
    boundary = [fake_row(1, 10, 10, 1.1) for _ in range(12)]
    judged = judge(equal_a, boundary)
    if judged["outside_loss"] or abs(float(judged["outside_ratio"]) - 1.1) > 1e-12:
        raise SystemExit("ratio 1.1 is not above 1.1")
    over = [fake_row(1, 10, 10, 1.2) for _ in range(12)]
    if not judge(equal_a, over)["outside_loss"]:
        raise SystemExit("ratio 1.2 must fail")
    sample = -173.92842659
    if float(full_precision(sample)) != sample:
        raise SystemExit("precision")
    print("check ok", flush=True)
    print("cells " + " ".join(row[0] for row in cells), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    runner = sub.add_parser("run")
    runner.add_argument("--rows", type=Path, required=True)
    runner.add_argument("--logs", type=Path, required=True)
    runner.add_argument("--arm-a-bin", type=Path, required=True)
    runner.add_argument("--arm-a-tree", type=Path, required=True)
    runner.add_argument("--arm-b-bin", type=Path, required=True)
    runner.add_argument("--arm-b-tree", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--rows", type=Path, required=True)
    report.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "run":
        execute(args)
    else:
        write_report(args)


if __name__ == "__main__":
    main()
