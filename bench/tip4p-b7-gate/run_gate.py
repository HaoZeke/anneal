#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Same-session no-loss gate for the rigid contact-closure move.

Arm A is the stalled-archive walk. Arm B adds the default-on rigid
contact closure. Both run the paper drivers with no ``H2H_*`` variable
set: ``h2h_seed`` on the recommended rigid preset for TIP4P, and
``h2h_gmin_proof``, ``h2h_morse`` and ``h2h_sc`` for the other systems.
Budgets come from those drivers' paper definitions. TIP4P uses the
2_000_000-call ceiling named ``PAPER_BUDGETS`` and printed for every
size in the paper TIP4P table. ``H2H_BUDGET`` is never set.

Cells are rigid TIP4P for every n from 8 to 21 that has an energy in
``data/tip4p_ccd.csv``, then lj38, lj75, lj98, m38r14 and sc126-55.
Seeds 2000-2011. One serial session, the whole process on core 0, one
thread in every pool. Within a cell an even offset from 2000 runs A
then B and an odd offset runs B then A, so the arm order is ABBA
repeated. Slot 0 is the first arm of a seed and slot 1 is the second.
The gate counts ``(slot, arm)`` pairs and refuses a schedule that is
not balanced. Atomic starts are written by each arm's ``coords``
command and must match byte for byte before any search.

A hit is a finite best at or below the reference plus the default
tolerance, 0.001. Calls to a hit keep that call count. A miss is ranked
at ``budget + 1``, after every hit. Hits use the two-sided Fisher exact
test. Calls use the two-sided Mann-Whitney U test, normal approximation,
with the tie correction and the continuity correction. Outside time per
call is ``outside_s / charged`` on each row. The cell ratio is the
type-7 median on B divided by the type-7 median on A.

Verdict, fixed before the runs: FIX if B has fewer hits than A on any
system, or B is slower on calls at p < 0.05, or the outside-per-call
ratio is above 1.10 on any system. Otherwise PASS. The failing cells
are named. Non-water seeds are bit-identical when ``best_bits`` and
``charged`` match. ``best`` is the driver token and ``best_bits`` is
the IEEE-754 binary64 encoding of that token.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

SEEDS = tuple(range(2000, 2012))
TIP4P_N_MIN = 8
TIP4P_N_MAX = 21
TIP4P_BUDGET = 2_000_000
RATIO_LIMIT = 1.10
CALLS_P = 0.05
JOB_TIMEOUT_S = 4 * 60 * 60
CORE = "0"
TREE_A = "14b537fd9b73b8fe21e01721954b41a278c5d223"
TREE_B = "c75edc53ed1ec9b91b1bd76372661ddec9e33a90"
# Files the closure commit may change. Start generators are not among them.
CLOSURE_FILES = frozenset(
    {
        "bench/tip4p-h2h/b7-diagnosis.md",
        "bench/tip4p-h2h/b7-results.md",
        "docs/newsfragments/+rigid-close-gap.changed.rst",
        "src/methods/cluster_hopping/moves.rs",
    }
)
PAPER_BUDGET_LINES = (
    ("scripts/tip4p_h2h/run_campaign.py", "PAPER_BUDGETS = {15: 2_000_000, 20: 2_000_000}"),
    (
        "scripts/tip4p_h2h/run_radius.py",
        "PAPER_BUDGETS = {4: 2_000_000, 8: 2_000_000, 15: 2_000_000, 20: 2_000_000}",
    ),
)
# Budgets and references the paper drivers print. The run reads the
# binary; these values are the check that the print is the paper map.
EXPECT = {
    "lj38": (400_000, -173.928427),
    "lj75": (4_000_000, -397.492331),
    "lj98": (4_000_000, -543.665361),
    "m38r14": (400_000, -144.321054),
    "sc126-55": (1_000_000, -52884.6806),
}
NON_WATER = ("lj38", "lj75", "lj98", "m38r14", "sc126-55")
RULE = (
    "FIX if B has fewer hits than A on any system, or a calls loss at "
    "p < 0.05, or a B/A outside-time-per-call ratio above 1.10 on any "
    "system. Otherwise PASS."
)
ROW_FIELDS = [
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
    "library_tree",
    "harness_blob",
]
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "TBB_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "POLARS_MAX_THREADS",
)


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


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided exact p, summing tables no more likely than the observed one."""
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """U of ``left`` and the two-sided p, with tie and continuity corrections."""
    n1, n2 = len(left), len(right)
    if n1 < 1 or n2 < 1:
        raise ValueError("mann-whitney needs two non-empty samples")
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return u, min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))


def arm_sequence(seed: int) -> tuple[str, str]:
    """A then B on an even offset from 2000, B then A on an odd offset."""
    if (seed - SEEDS[0]) % 2 == 0:
        return ("A", "B")
    return ("B", "A")


def planned_jobs(systems: list[str]) -> list[tuple[str, int, int, str]]:
    jobs = []
    for system in systems:
        for seed in SEEDS:
            for slot, arm in enumerate(arm_sequence(seed)):
                jobs.append((system, seed, slot, arm))
    return jobs


def pair_counts(jobs: list[tuple[str, int, int, str]]) -> dict[tuple[int, str], int]:
    counts = {(0, "A"): 0, (0, "B"): 0, (1, "A"): 0, (1, "B"): 0}
    for _system, _seed, slot, arm in jobs:
        counts[(slot, arm)] += 1
    return counts


def assert_balanced(jobs: list[tuple[str, int, int, str]], systems: list[str]) -> None:
    expect = len(SEEDS) // 2 * len(systems)
    counts = pair_counts(jobs)
    for key, count in counts.items():
        if count != expect:
            raise SystemExit(f"unbalanced (slot, arm) {key}: {count} != {expect}")
    per = len(SEEDS) // 2
    for system in systems:
        cell = [job for job in jobs if job[0] == system]
        cell_counts = pair_counts(cell)
        for key, count in cell_counts.items():
            if count != per:
                raise SystemExit(f"{system} unbalanced {key}: {count} != {per}")
    if len(jobs) != len(systems) * len(SEEDS) * 2:
        raise SystemExit("job count")
    first = [arm for system, _seed, _slot, arm in jobs if system == systems[0]][:4]
    if first != ["A", "B", "B", "A"]:
        raise SystemExit(f"ABBA opening {first}")


def tip4p_cells(text: str) -> list[tuple[str, int, float]]:
    rows = list(csv.DictReader(io.StringIO(text)))
    found = []
    for row in rows:
        n = int(row["n"])
        if TIP4P_N_MIN <= n <= TIP4P_N_MAX:
            found.append((f"tip4p{n}", n, float(row["energy"])))
    found.sort(key=lambda item: item[1])
    seen = [n for _name, n, _energy in found]
    wanted = list(range(TIP4P_N_MIN, TIP4P_N_MAX + 1))
    if seen != wanted:
        raise SystemExit(f"TIP4P n {seen} does not cover {wanted}")
    return found


def miss_rank(budget: int) -> float:
    """A miss ranks one call worse than a hit on the final charged call."""
    return float(budget + 1)


def ranked_calls(hit: bool, calls: int, budget: int) -> float:
    if hit:
        if not 0 < calls <= budget:
            raise ValueError(f"hit calls {calls} budget {budget}")
        return float(calls)
    return miss_rank(budget)


def ratio_of_medians(left: list[float], right: list[float]) -> float:
    """Median of ``right`` over median of ``left``. Both zero is 1."""
    median_left = type7(left, 0.5)
    median_right = type7(right, 0.5)
    if median_left == 0.0:
        return 1.0 if median_right == 0.0 else math.inf
    return median_right / median_left


def bits_of(token: str) -> str:
    """IEEE-754 binary64 of the driver token, as 16 lowercase hex digits."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite best {token}")
    return f"{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"


def driver_for(system: str) -> str:
    if system.startswith("tip4p"):
        return "h2h_seed"
    if system.startswith("lj"):
        return "h2h_gmin_proof"
    if system.startswith("m"):
        return "h2h_morse"
    if system.startswith("sc"):
        return "h2h_sc"
    raise SystemExit(f"no driver for {system}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    if not raw:
        raise SystemExit("machine id is empty")
    return hashlib.sha256(raw).hexdigest()


def affinity_list() -> str:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list"):
            return line.split(":", 1)[1].strip()
    return ""


def git_out(tree: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", "-C", str(tree), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "git failed")
    return proc.stdout.strip()


def harness_blob() -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "hash-object", str(Path(__file__).resolve())],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "hash-object failed")
    blob = proc.stdout.strip()
    committed = git_out(ROOT, ["rev-parse", f"HEAD:bench/tip4p-b7-gate/{Path(__file__).name}"])
    if blob != committed:
        raise SystemExit("harness blob does not match the committed file")
    return blob


def single_thread_env(sc_lib: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    for key in THREAD_KEYS:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["LJOS_SEAT"] = "benchbot"
    if sc_lib is not None:
        env["RGPOT_SC_LIB"] = str(sc_lib)
    else:
        env.pop("RGPOT_SC_LIB", None)
    return env


def which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(":"):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def run_proc(command: list[str], tree: Path, env: dict[str, str], log: Path | None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        cwd=tree,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=JOB_TIMEOUT_S,
    )
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", newline="\n") as handle:
            handle.write(" ".join(command) + "\n")
            handle.write(proc.stdout)
            if proc.stdout and not proc.stdout.endswith("\n"):
                handle.write("\n")
            handle.write(proc.stderr)
            handle.flush()
            os.fsync(handle.fileno())
    return proc


def paper_spec(binary: Path, tree: Path, driver: str, system: str) -> tuple[int, float]:
    proc = run_proc(
        ["taskset", "-c", CORE, str(binary / driver), "budget", system],
        tree,
        single_thread_env(None),
        None,
    )
    if proc.returncode != 0:
        raise SystemExit(f"{system} budget exited {proc.returncode}\n{proc.stderr}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"{system} budget output {proc.stdout!r}")
    return int(lines[0]), float(lines[1])


def assert_paper_scripts(tree: Path) -> None:
    for relative, line in PAPER_BUDGET_LINES:
        text = (tree / relative).read_text()
        if line not in text:
            raise SystemExit(f"{relative} does not name the 2_000_000 paper ceiling")
    tex = (tree / "paper/jctc/manuscript.tex").read_text()
    if "2\\times 10^{6}" not in tex:
        raise SystemExit("the paper TIP4P table ceiling is not 2e6")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS:
            raise SystemExit(f"row header {reader.fieldnames}")
        return list(reader)


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def parse_hit_flag(text: str) -> bool:
    token = text.strip().lower()
    if token in ("1", "true"):
        return True
    if token in ("0", "false"):
        return False
    raise RuntimeError(f"hit flag {text}")


def resolve_hit(driver_hit: bool, best_token: str, reference: float, label: str) -> bool:
    best = float(best_token)
    if not math.isfinite(best):
        if driver_hit:
            raise RuntimeError(f"{label}: non-finite best was marked as a hit")
        return False
    measured = energy_hits(best, reference, DEFAULT_HIT_TOL)
    boundary = reference + DEFAULT_HIT_TOL
    if measured != driver_hit and abs(best - boundary) > 2e-8:
        raise RuntimeError(f"{label}: hit {int(driver_hit)} best {best_token} reference {reference}")
    if abs(best - boundary) <= 2e-8:
        return driver_hit
    return measured


def checked_result(
    label: str,
    budget: int,
    reference: float,
    driver_hit: bool,
    best_token: str,
    calls: int,
    charged: int,
    wall: float,
    inside: float,
    outside: float,
) -> dict[str, str]:
    hit = resolve_hit(driver_hit, best_token, reference, label)
    if charged < 1:
        raise RuntimeError(f"{label}: charged {charged}")
    if hit and not 0 < calls <= budget:
        raise RuntimeError(f"{label}: hit calls {calls} budget {budget}")
    if not hit and calls != budget:
        raise RuntimeError(f"{label}: miss calls {calls} budget {budget}")
    if wall < 0.0 or inside < 0.0 or outside < 0.0:
        raise RuntimeError(f"{label}: wall {wall} inside {inside} outside {outside}")
    if abs(outside - max(wall - inside, 0.0)) > 5e-6:
        raise RuntimeError(f"{label}: outside {outside} wall {wall} inside {inside}")
    return {
        "hit": "1" if hit else "0",
        "calls": str(calls),
        "charged": str(charged),
        "wall": f"{wall:.6f}",
        "inside_s": f"{inside:.6f}",
        "outside_s": f"{outside:.6f}",
        "best": best_token,
        "best_bits": bits_of(best_token),
        "budget": str(budget),
    }


def parse_h2h_row(text: str, system: str, seed: int, budget: int, reference: float) -> dict[str, str]:
    rows = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one h2h_row\n{text[-2000:]}")
    fields = dict(part.split("=", 1) for part in rows[0].split()[1:])
    if fields.get("system") != system or int(fields["seed"]) != seed or int(fields["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: row {rows[0]}")
    return checked_result(
        f"{system} seed {seed}",
        budget,
        reference,
        parse_hit_flag(fields["hit"]),
        fields["best"],
        int(fields["evals_to_hit"]),
        int(fields["charged"]),
        float(fields["wall_s"]),
        float(fields["pot_s"]),
        float(fields["outside_s"]),
    )


def parse_csv_row(text: str, system: str, seed: int, budget: int, reference: float) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("system,") or f",anneal,{seed}," in line]
    headers = [line for line in lines if line.startswith("system,")]
    data = [line for line in lines if not line.startswith("system,")]
    if len(headers) != 1 or len(data) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one result row\n{text[-2000:]}")
    row = dict(zip(headers[0].split(","), data[0].split(","), strict=True))
    if row["system"] != system or int(row["seed"]) != seed or int(row["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: row {data[0]}")
    wall = float(row["wall_s"])
    inside = float(row["pot_s"])
    if "outside_s" in row and row["outside_s"] != "":
        outside = float(row["outside_s"])
    else:
        outside = max(wall - inside, 0.0)
    return checked_result(
        f"{system} seed {seed}",
        budget,
        reference,
        parse_hit_flag(row["hit"]),
        row["best"],
        int(row["evals_to_hit"]),
        int(row["charged"]),
        wall,
        inside,
        outside,
    )


def launch(
    command: list[str],
    tree: Path,
    log: Path,
    system: str,
    seed: int,
    budget: int,
    reference: float,
    kind: str,
    sc_lib: Path | None,
) -> dict[str, str]:
    proc = run_proc(command, tree, single_thread_env(sc_lib), log)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    if kind == "h2h_row":
        return parse_h2h_row(proc.stdout, system, seed, budget, reference)
    return parse_csv_row(proc.stdout, system, seed, budget, reference)


def run_tip4p(
    binary: Path,
    tree: Path,
    n: int,
    reference: float,
    seed: int,
    scratch: Path,
    log: Path,
) -> dict[str, str]:
    stop = repr(reference + DEFAULT_HIT_TOL)
    prefix = scratch / f"tip4p{n}-{seed}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset",
        "-c",
        CORE,
        str(binary / "h2h_seed"),
        "tip4p",
        str(n),
        "0",
        str(TIP4P_BUDGET),
        str(seed),
        "recommended",
        stop,
        str(prefix),
    ]
    return launch(command, tree, log, f"tip4p{n}", seed, TIP4P_BUDGET, reference, "h2h_row", None)


def run_named(
    binary: Path,
    tree: Path,
    driver: str,
    system: str,
    seed: int,
    budget: int,
    reference: float,
    log: Path,
    sc_lib: Path | None,
) -> dict[str, str]:
    command = ["taskset", "-c", CORE, str(binary / driver), "run", system, str(seed)]
    return launch(command, tree, log, system, seed, budget, reference, "csv", sc_lib)


def write_coords(binary: Path, tree: Path, driver: str, system: str, seed: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = run_proc(
        ["taskset", "-c", CORE, str(binary / driver), "coords", system, str(seed), str(path)],
        tree,
        single_thread_env(None),
        None,
    )
    if proc.returncode != 0 or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"{system} seed {seed} coords failed\n{proc.stderr}")


def same_starts(arms: dict[str, tuple[Path, Path]], systems: list[str]) -> None:
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "tip4p-b7-starts"
    for system in systems:
        if system.startswith("tip4p"):
            continue
        driver = driver_for(system)
        for seed in SEEDS:
            paths = {}
            for arm, (binary, tree) in arms.items():
                path = scratch / arm / f"{system}-{seed}.xyz"
                write_coords(binary, tree, driver, system, seed, path)
                paths[arm] = path.read_bytes()
            if paths["A"] != paths["B"]:
                raise SystemExit(f"{system} seed {seed} starts differ")


def closure_diff(tree_a: Path, tree_b: Path) -> None:
    head_a = git_out(tree_a, ["rev-parse", "HEAD"])
    head_b = git_out(tree_b, ["rev-parse", "HEAD"])
    names = git_out(tree_b, ["diff", "--name-only", head_a, head_b])
    found = frozenset(line for line in names.splitlines() if line)
    if found != CLOSURE_FILES:
        raise SystemExit(f"arm diff is {sorted(found)}")


def parse_arm(text: str) -> tuple[str, Path, Path]:
    name, rest = text.split("=", 1)
    binary, tree = rest.rsplit("@", 1)
    if name not in ("A", "B"):
        raise SystemExit(f"arm name {name}")
    return name, Path(binary), Path(tree)


def execute(args: argparse.Namespace) -> None:
    if affinity_list() != CORE:
        raise SystemExit(f"this process is not pinned to core {CORE} (allowed {affinity_list()!r})")
    if which("taskset") is None:
        raise SystemExit("taskset is required")
    seat = os.environ.get("LJOS_SEAT")
    if seat not in (None, "", "benchbot"):
        raise SystemExit("LJOS_SEAT must be benchbot")
    arms: dict[str, tuple[Path, Path]] = {}
    for text in args.arm:
        name, binary, tree = parse_arm(text)
        arms[name] = (binary, tree)
    if set(arms) != {"A", "B"}:
        raise SystemExit("name both arms A and B")
    trees = {name: git_out(tree, ["rev-parse", "HEAD^{tree}"]) for name, (_binary, tree) in arms.items()}
    if trees["A"] != TREE_A or trees["B"] != TREE_B:
        raise SystemExit(f"trees {trees['A']} {trees['B']}")
    for name, (binary, tree) in arms.items():
        assert_paper_scripts(tree)
        for driver in ("h2h_seed", "h2h_gmin_proof", "h2h_morse", "h2h_sc"):
            path = binary / driver
            if not path.is_file() or not os.access(path, os.X_OK):
                raise SystemExit(f"{name} missing {driver}")
    csv_hash = {name: sha256_file(tree / "data" / "tip4p_ccd.csv") for name, (_binary, tree) in arms.items()}
    if csv_hash["A"] != csv_hash["B"]:
        raise SystemExit("TIP4P reference files differ")
    cells = tip4p_cells((arms["A"][1] / "data" / "tip4p_ccd.csv").read_text())
    specs: dict[str, tuple[int, float]] = {}
    for system, (want_budget, want_reference) in EXPECT.items():
        got = {}
        for name, (binary, tree) in arms.items():
            got[name] = paper_spec(binary, tree, driver_for(system), system)
        if got["A"] != got["B"]:
            raise SystemExit(f"{system} budget mismatch {got}")
        budget, reference = got["A"]
        if budget != want_budget or abs(reference - want_reference) > 5e-7:
            raise SystemExit(f"{system} paper spec {budget} {reference}")
        specs[system] = (budget, reference)
    references = {name: energy for name, _n, energy in cells}
    budgets = {name: TIP4P_BUDGET for name, _n, _energy in cells}
    sizes = {name: n for name, n, _energy in cells}
    for system, (budget, reference) in specs.items():
        references[system] = reference
        budgets[system] = budget
    systems = [name for name, _n, _energy in cells] + list(NON_WATER)
    jobs = planned_jobs(systems)
    assert_balanced(jobs, systems)
    closure_diff(arms["A"][1], arms["B"][1])
    same_starts(arms, systems)
    sc_lib = Path(args.sc_lib)
    if not sc_lib.is_file():
        raise SystemExit("Sutton-Chen library is missing")
    blob = harness_blob()
    machine = hashed_machine_id()
    print(RULE, flush=True)
    print("trees", trees["A"], trees["B"], "blob", blob, "machine", machine, flush=True)
    print(
        "abba",
        " ".join(f"{slot}:{arm}={count}" for (slot, arm), count in sorted(pair_counts(jobs).items())),
        flush=True,
    )
    rows = read_rows(args.rows)
    if len(rows) > len(jobs):
        raise SystemExit("rows are longer than the schedule")
    for index, row in enumerate(rows):
        system, seed, slot, arm = jobs[index]
        if (row["system"], int(row["seed"]), int(row["slot"]), row["arm"]) != (system, seed, slot, arm):
            raise SystemExit(f"stored row {index} is not the schedule")
        if row["library_tree"] != trees[arm] or row["harness_blob"] != blob or row["machine_id"] != machine:
            raise SystemExit(f"stored {system} {arm} seed {seed} does not match this build")
        if row["core"] != CORE or int(row["budget"]) != budgets[system]:
            raise SystemExit(f"stored {system} budget or core")
    logs = args.logs
    scratch = args.scratch
    for index, (system, seed, slot, arm) in enumerate(jobs):
        if index < len(rows):
            continue
        binary, tree = arms[arm]
        log = logs / f"{index:04d}-{system}-{arm}-s{seed}.log"
        if system.startswith("tip4p"):
            result = run_tip4p(binary, tree, sizes[system], references[system], seed, scratch / f"{system}-{arm}-s{seed}", log)
        else:
            result = run_named(
                binary,
                tree,
                driver_for(system),
                system,
                seed,
                budgets[system],
                references[system],
                log,
                sc_lib if system.startswith("sc") else None,
            )
        row = {
            "system": system,
            "arm": arm,
            "seed": str(seed),
            "slot": str(slot),
            "core": CORE,
            "machine_id": machine,
            "library_tree": trees[arm],
            "harness_blob": blob,
            **result,
        }
        append_row(args.rows, row)
        rows.append(row)
        print(
            f"{index} {system} {arm} s{seed} hit={row['hit']} calls={row['calls']} charged={row['charged']}",
            flush=True,
        )
    print(f"rows {len(rows)}", flush=True)


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["arm"]), []).append(row)
    return grouped


def require_schedule(rows: list[dict[str, str]]) -> list[str]:
    systems: list[str] = []
    for row in rows:
        if row["system"] not in systems:
            systems.append(row["system"])
    jobs = planned_jobs(systems)
    assert_balanced(jobs, systems)
    if len(rows) != len(jobs):
        raise SystemExit(f"row count {len(rows)} != jobs {len(jobs)}")
    for row, job in zip(rows, jobs, strict=True):
        system, seed, slot, arm = job
        if (row["system"], int(row["seed"]), int(row["slot"]), row["arm"]) != (system, seed, slot, arm):
            raise SystemExit(f"{system} seed {seed} is out of ABBA order")
    return systems


def cell_stats(system: str, left: list[dict[str, str]], right: list[dict[str, str]]) -> dict[str, object]:
    budget = int(left[0]["budget"])
    if any(int(row["budget"]) != budget for row in left + right):
        raise SystemExit(f"{system}: mixed budgets")
    hits_a = sum(row["hit"] == "1" for row in left)
    hits_b = sum(row["hit"] == "1" for row in right)
    calls_a = [ranked_calls(row["hit"] == "1", int(row["calls"]), budget) for row in left]
    calls_b = [ranked_calls(row["hit"] == "1", int(row["calls"]), budget) for row in right]
    u, mw_p = mann_whitney(calls_a, calls_b)
    opc_a = [float(row["outside_s"]) / int(row["charged"]) for row in left]
    opc_b = [float(row["outside_s"]) / int(row["charged"]) for row in right]
    ratio = ratio_of_medians(opc_a, opc_b)
    loses_hits = hits_b < hits_a
    calls_loss = mw_p < CALLS_P and u < (len(calls_a) * len(calls_b) / 2.0)
    ratio_over = not (ratio <= RATIO_LIMIT)
    reasons = []
    if loses_hits:
        reasons.append("hits")
    if calls_loss:
        reasons.append("calls")
    if ratio_over:
        reasons.append("outside-per-call")
    return {
        "system": system,
        "budget": budget,
        "hits_a": hits_a,
        "hits_b": hits_b,
        "n": len(left),
        "fisher_p": fisher(hits_a, len(left), hits_b, len(right)),
        "calls_a": type7(calls_a, 0.5),
        "calls_b": type7(calls_b, 0.5),
        "u": u,
        "mw_p": mw_p,
        "wall_a": type7([float(row["wall"]) for row in left], 0.5),
        "wall_b": type7([float(row["wall"]) for row in right], 0.5),
        "opc_a": type7(opc_a, 0.5),
        "opc_b": type7(opc_b, 0.5),
        "ratio": ratio,
        "reasons": reasons,
    }


def fmt_p(value: float) -> str:
    return f"{value:.6g}"


def fmt_ratio(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6g}"


def verdict_of(cells: list[dict[str, object]]) -> tuple[str, list[str]]:
    named = [f"{row['system']} ({', '.join(row['reasons'])})" for row in cells if row["reasons"]]
    if named:
        return "FIX", named
    return "PASS", []


def render_readme(rows: list[dict[str, str]], cells: list[dict[str, object]], systems: list[str]) -> str:
    verdict, named = verdict_of(cells)
    grouped = group_rows(rows)
    counts = pair_counts(
        [(row["system"], int(row["seed"]), int(row["slot"]), row["arm"]) for row in rows]
    )
    lines = [
        "# tip4p b7 no-loss gate",
        "",
        RULE,
        "",
        f"Verdict: {verdict}",
        "",
    ]
    if named:
        lines.append("Failing cells: " + "; ".join(named) + ".")
    else:
        lines.append("Failing cells: none.")
    lines.extend(
        [
            "",
            "Seeds 2000-2011. One serial session, core 0, one thread in each pool. "
            "Arms alternate ABBA. A hit is a finite best at or below the reference plus 0.001. "
            "Misses rank at budget + 1 for the calls test. "
            "Outside time per call is outside seconds divided by charged calls on each row. "
            "The ratio is the type-7 median on B over the type-7 median on A. "
            "Fisher p is the exact two-sided test. Mann-Whitney p is the two-sided normal "
            "approximation with the tie and continuity corrections. "
            "A calls loss is that p below 0.05 with B ranked slower. "
            "best is the driver token. best_bits is the IEEE-754 encoding of that token. "
            f"Per-job timeout {JOB_TIMEOUT_S} seconds.",
            "",
            "## Verdict",
            "",
            "| system | hits A | hits B | Fisher p | calls A | calls B | MW U | MW p | wall A | wall B | outside/call A | outside/call B | ratio |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in cells:
        lines.append(
            "| {system} | {hits_a}/{n} | {hits_b}/{n} | {fisher} | {calls_a:.6g} | {calls_b:.6g} | {u:.6g} | {mw:.6g} | {wall_a:.6f} | {wall_b:.6f} | {opc_a:.6g} | {opc_b:.6g} | {ratio} |".format(
                system=row["system"],
                hits_a=row["hits_a"],
                hits_b=row["hits_b"],
                n=row["n"],
                fisher=fmt_p(float(row["fisher_p"])),
                calls_a=row["calls_a"],
                calls_b=row["calls_b"],
                u=row["u"],
                mw=fmt_p(float(row["mw_p"])),
                wall_a=row["wall_a"],
                wall_b=row["wall_b"],
                opc_a=row["opc_a"],
                opc_b=row["opc_b"],
                ratio=fmt_ratio(float(row["ratio"])),
            )
        )
    lines.extend(["", "## Hits by seed", ""])
    header = "| system | " + " | ".join(str(seed) for seed in SEEDS) + " |"
    rule = "| --- | " + " | ".join("---" for _ in SEEDS) + " |"
    lines.extend([header, rule])
    for system in systems:
        left = {int(row["seed"]): row for row in grouped[(system, "A")]}
        right = {int(row["seed"]): row for row in grouped[(system, "B")]}
        marks = " | ".join(f"{left[seed]['hit']}/{right[seed]['hit']}" for seed in SEEDS)
        lines.append(f"| {system} | {marks} |")
    lines.extend(["", "Each cell is hit A / hit B.", ""])
    lines.extend(["## Non-water bit identity", ""])
    identical = True
    lines.extend(
        [
            "| system | seed | best_bits | charged |",
            "| --- | --- | --- | --- |",
        ]
    )
    for system in NON_WATER:
        if system not in systems:
            continue
        left = {int(row["seed"]): row for row in grouped[(system, "A")]}
        right = {int(row["seed"]): row for row in grouped[(system, "B")]}
        for seed in SEEDS:
            bits_same = left[seed]["best_bits"] == right[seed]["best_bits"]
            charged_same = left[seed]["charged"] == right[seed]["charged"]
            identical = identical and bits_same and charged_same
            lines.append(
                f"| {system} | {seed} | {'identical' if bits_same else 'differs'} | "
                f"{'identical' if charged_same else 'differs'} |"
            )
    lines.extend(
        [
            "",
            "Every non-water seed is bit-identical."
            if identical
            else "Not every non-water seed is bit-identical.",
            "",
            "## Schedule and provenance",
            "",
            "Counted (slot, arm) pairs: "
            + ", ".join(f"({slot}, {arm}) = {count}" for (slot, arm), count in sorted(counts.items()))
            + ".",
            "",
        ]
    )
    for arm in ("A", "B"):
        trees = sorted({row["library_tree"] for row in rows if row["arm"] == arm})
        lines.append(f"Library tree {arm}: `{trees[0]}`.")
    blobs = sorted({row["harness_blob"] for row in rows})
    machines = sorted({row["machine_id"] for row in rows})
    lines.append(f"Harness blob: `{blobs[0]}`.")
    lines.append(f"Hashed machine id: `{machines[0]}`.")
    lines.append("")
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> None:
    rows = read_rows(args.rows)
    if not rows:
        raise SystemExit("no rows")
    systems = require_schedule(rows)
    grouped = group_rows(rows)
    for system in systems:
        for arm in ("A", "B"):
            seeds = [int(row["seed"]) for row in grouped[(system, arm)]]
            if seeds != list(SEEDS):
                raise SystemExit(f"{system} {arm} seeds {seeds}")
    if len({row["machine_id"] for row in rows}) != 1:
        raise SystemExit("more than one machine id")
    if len({row["harness_blob"] for row in rows}) != 1:
        raise SystemExit("more than one harness blob")
    for arm in ("A", "B"):
        if len({row["library_tree"] for row in rows if row["arm"] == arm}) != 1:
            raise SystemExit(f"arm {arm} has more than one tree")
    cells = [cell_stats(system, grouped[(system, "A")], grouped[(system, "B")]) for system in systems]
    text = render_readme(rows, cells, systems)
    if "\r" in text:
        raise SystemExit("readme has CR")
    args.readme.parent.mkdir(parents=True, exist_ok=True)
    with args.readme.open("w", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    verdict, named = verdict_of(cells)
    print(verdict, flush=True)
    if named:
        print("; ".join(named), flush=True)


def self_test() -> None:
    systems = [f"tip4p{n}" for n in range(8, 22)] + list(NON_WATER)
    jobs = planned_jobs(systems)
    assert_balanced(jobs, systems)
    if pair_counts(jobs) != {(0, "A"): 114, (0, "B"): 114, (1, "A"): 114, (1, "B"): 114}:
        raise SystemExit(f"pair counts {pair_counts(jobs)}")
    sample = "n,energy,source,sha256\n" + "\n".join(
        f"{n},-{n}.5,a,b" for n in range(2, 22)
    )
    parsed = tip4p_cells(sample + "\n")
    if [item[1] for item in parsed] != list(range(8, 22)):
        raise SystemExit(f"tip4p filter {parsed}")
    if abs(fisher(2, 2, 0, 2) - (1.0 / 3.0)) > 1e-12:
        raise SystemExit("fisher 2x2")
    if fisher(12, 12, 12, 12) != 1.0 or fisher(0, 12, 0, 12) != 1.0:
        raise SystemExit("fisher ties")
    u, p = mann_whitney([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    if p != 1.0 or u != 4.5:
        raise SystemExit(f"mann whitney ties {u} {p}")
    slow = [miss_rank(12)] * 12
    fast = [float(i) for i in range(1, 13)]
    u_loss, p_loss = mann_whitney(fast, slow)
    if not (p_loss < CALLS_P and u_loss < 72.0):
        raise SystemExit(f"calls loss not detected {u_loss} {p_loss}")
    u_gain, p_gain = mann_whitney(slow, fast)
    if p_gain >= CALLS_P or u_gain < 72.0:
        raise SystemExit(f"a faster B was marked as a loss {u_gain} {p_gain}")
    if bits_of("-1.5") != f"{struct.unpack('<Q', struct.pack('<d', -1.5))[0]:016x}":
        raise SystemExit("bits")
    if ratio_of_medians([1.0, 1.0], [1.1, 1.1]) > RATIO_LIMIT:
        raise SystemExit("ratio at the limit was rejected")
    if ratio_of_medians([1.0, 1.0], [1.2, 1.2]) <= RATIO_LIMIT:
        raise SystemExit("ratio over the limit was accepted")
    if "\r" in Path(__file__).read_text():
        raise SystemExit("source has CR")
    if DEFAULT_HIT_TOL != 1e-3:
        raise SystemExit("tolerance")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser(description="TIP4P b7 no-loss gate")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--rows", type=Path, required=True)
    run_p.add_argument("--logs", type=Path, required=True)
    run_p.add_argument("--scratch", type=Path, required=True)
    run_p.add_argument("--sc-lib", required=True)
    run_p.add_argument("--arm", action="append", required=True)
    sum_p = sub.add_parser("summarize")
    sum_p.add_argument("--rows", type=Path, required=True)
    sum_p.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args(argv[1:])
    if args.command == "run":
        execute(args)
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
