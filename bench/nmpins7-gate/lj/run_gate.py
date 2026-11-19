#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Three-arm Lennard-Jones gate for the generalized symmetry flag.

Cells are LJ38, LJ75, LJ98 and LJ104. Seeds are 2000-2011. Budgets are
``scripts/gmin_panel/run_panel.py`` ``PAPER_BUDGETS``. A hit is the
reference minimum plus 0.001. Arms A, B and C share the coordinates the
``h2h_gmin_proof`` ``coords`` command writes for that seed. The six
permutations of A, B, C are used twice, so each arm occupies each slot
four times in a cell. LJ38 also runs the same A binary twice per seed
in ABBA order.

One process, ``taskset -c 0``, one thread. A seed still running after
7200 s is a miss: the row keeps the calls, the best energy and the wall
when the process printed them. Rows are appended and never rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gmin_panel"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_panel  # noqa: E402
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

BASE = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
KIT = "4cd811004c67f26b0e216664cb51352f04654c09"
KIT_TREE = "51c1a7e0180585d078c3457b5550b2963d1ab752"
SYSTEMS = ("lj38", "lj75", "lj98", "lj104")
SEEDS = tuple(range(2000, 2012))
PERMS = (
    ("A", "B", "C"),
    ("A", "C", "B"),
    ("B", "A", "C"),
    ("B", "C", "A"),
    ("C", "A", "B"),
    ("C", "B", "A"),
)
WALL_CAP_S = 7200.0
BONFERRONI_M = 30
ALPHA = 0.05
OUTSIDE_RATIO_LIMIT = 1.1
FLAG = "general_symmetry"
FLAG_ENV = "H2H_GENERAL_SYMMETRY"
THREAD_ONES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "TBB_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)
THREAD_FALSE = ("OMP_DYNAMIC", "MKL_DYNAMIC", "OPENBLAS_DYNAMIC")
MPI_SIZE_KEYS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "MPI_LOCALNRANKS")
PACKAGES = ("anneal-core", "rgmin", "rgpot-core", "rgpot-pair", "readcon-core")
ROW_FIELDS = [
    "seq",
    "session",
    "panel",
    "system",
    "seed",
    "arm",
    "replica",
    "slot",
    "perm",
    "budget",
    "reference",
    "hit",
    "capped",
    "calls",
    "calls_to_hit",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "outside_per_call_s",
    "core",
    "affinity",
    "machine_id",
    "code_revision",
    "code_tree",
    "binary_sha256",
    "flag",
    "utc",
]
DEFAULT_ROWS = HERE / "rows.csv"
DEFAULT_SESSIONS = HERE / "sessions.jsonl"
DEFAULT_REPORT = HERE / "REPORT.md"


def git_out(args: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def cat_file(sha: str, kind: str) -> None:
    git_out(["cat-file", "-e", f"{sha}^{{{kind}}}"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def machine_id() -> str:
    raw = Path("/etc/machine-id").read_text().strip().encode()
    return hashlib.sha256(raw).hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def thread_env(flag_on: bool) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    for key in THREAD_ONES:
        env[key] = "1"
    for key in THREAD_FALSE:
        env[key] = "FALSE"
    for key in MPI_SIZE_KEYS:
        value = env.get(key, "")
        if value not in ("", "1"):
            raise SystemExit(f"{key} is {value}")
        env.pop(key, None)
    env["LJOS_SEAT"] = os.environ.get("LJOS_SEAT", "benchbot")
    if flag_on:
        env[FLAG_ENV] = "1"
    return env


def recorded_thread_env() -> dict[str, str]:
    sample = thread_env(False)
    keys = (*THREAD_ONES, *THREAD_FALSE)
    return {key: sample[key] for key in keys}


def package_versions(tree: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    current: dict[str, str] = {}

    def finish() -> None:
        name = current.get("name", "")
        if name not in PACKAGES or name in versions:
            return
        source = current.get("source", "")
        version = current.get("version", "")
        rev = source.split("rev=", 1)[1].split("#", 1)[0] if "rev=" in source else ""
        versions[name] = f"{version}@{rev}" if rev else version

    for line in (tree / "Cargo.lock").read_text().splitlines():
        if line.startswith("[[package]]"):
            finish()
            current = {}
        elif line.startswith("name = "):
            current["name"] = line.split('"', 2)[1]
        elif line.startswith("version = ") and '"' in line:
            current["version"] = line.split('"', 2)[1]
        elif line.startswith("source = "):
            current["source"] = line.split('"', 2)[1]
    finish()
    missing = [name for name in PACKAGES if name not in versions]
    if missing:
        raise SystemExit(f"missing package versions: {missing}")
    return versions


def fisher_counts(hits_other: int, n_other: int, hits_base: int, n_base: int) -> tuple[int, int]:
    """Numerator and denominator of the one-sided exact test that the other arm has fewer hits."""
    total_n = n_base + n_other
    total_hits = hits_base + hits_other
    lo = max(0, total_hits - n_base)
    hi = min(n_other, total_hits)
    if hits_other < lo or hits_other > hi:
        raise ValueError("impossible hit table")
    den = math.comb(total_n, n_other)
    num = 0
    for k in range(lo, hits_other + 1):
        num += math.comb(total_hits, k) * math.comb(total_n - total_hits, n_other - k)
    return num, den


def fisher_p(hits_other: int, n_other: int, hits_base: int, n_base: int) -> float:
    num, den = fisher_counts(hits_other, n_other, hits_base, n_base)
    return num / den


def fisher_loss_exact(hits_other: int, n_other: int, hits_base: int, n_base: int) -> bool:
    """True when num/den < alpha/m. alpha is 1/20, so the bound is num * m * 20 < den."""
    num, den = fisher_counts(hits_other, n_other, hits_base, n_base)
    return num * BONFERRONI_M * 20 < den


def bonferroni(p: float) -> float:
    if math.isnan(p):
        return p
    return min(1.0, p * BONFERRONI_M)


def mann_whitney_parts(left: list[float], right: list[float]) -> tuple[float, float, float]:
    """U of ``left``, its tie-corrected variance, and the null mean.

    The rank sum matches ``run_panel.mann_whitney``.
    """
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
    return u, var, n1 * n2 / 2.0


def mann_whitney_two(left: list[float], right: list[float]) -> tuple[float, float]:
    u, var, mean = mann_whitney_parts(left, right)
    if var <= 0.0:
        return u, 1.0
    dev = max(abs(u - mean) - 0.5, 0.0)
    return u, min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))


def mann_whitney_greater(left: list[float], right: list[float]) -> tuple[float, float]:
    """One-sided p that ``left`` is stochastically greater than ``right``."""
    u, var, mean = mann_whitney_parts(left, right)
    if var <= 0.0:
        return u, 1.0
    z = (u - mean - 0.5) / math.sqrt(var)
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return u, min(1.0, max(0.0, p))


def mw_loss(p: float) -> bool:
    return p < ALPHA / BONFERRONI_M


def perm_for(seed: int) -> tuple[str, str, str]:
    return PERMS[(seed - SEEDS[0]) % len(PERMS)]


def noise_replicas(index: int) -> tuple[int, int]:
    """ABBA: replica 0 then 1, then 1 then 0, then 1 then 0, then 0 then 1."""
    if index % 4 in (0, 3):
        return (0, 1)
    return (1, 0)


def planned_jobs() -> list[dict[str, str | int]]:
    jobs: list[dict[str, str | int]] = []
    for system in SYSTEMS:
        for seed in SEEDS:
            perm = perm_for(seed)
            for slot, arm in enumerate(perm):
                jobs.append(
                    {
                        "panel": "gate",
                        "system": system,
                        "seed": seed,
                        "arm": arm,
                        "replica": "",
                        "slot": slot,
                        "perm": "".join(perm),
                    }
                )
        if system == "lj38":
            for index, seed in enumerate(SEEDS):
                order = noise_replicas(index)
                for slot, replica in enumerate(order):
                    jobs.append(
                        {
                            "panel": "noise",
                            "system": "lj38",
                            "seed": seed,
                            "arm": "A",
                            "replica": str(replica),
                            "slot": slot,
                            "perm": "".join(str(part) for part in order),
                        }
                    )
    return jobs


def type7(values: list[float], p: float) -> float:
    return run_panel.type7(values, p)


def iqr(values: list[float]) -> float:
    return type7(values, 0.75) - type7(values, 0.25)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def outside_repo(path: Path) -> None:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved == root or root in resolved.parents:
        raise SystemExit("logs stay outside the repository")


def parse_arm(text: str) -> dict[str, object]:
    name, rest = text.split("=", 1)
    binary_s, revision, tree_s = rest.rsplit("@", 2)
    tree = Path(tree_s)
    revision = git_out(["rev-parse", revision], tree)
    binary = Path(binary_s)
    if not binary.is_file():
        raise SystemExit(f"missing binary for {name}")
    tree_sha = git_out(["rev-parse", "HEAD^{tree}"], tree)
    status = git_out(["status", "--porcelain"], tree)
    if status:
        raise SystemExit(f"{name} tree is dirty")
    return {
        "name": name,
        "binary": binary,
        "revision": revision,
        "tree": tree,
        "tree_sha": tree_sha,
        "binary_sha256": sha256_file(binary),
    }


def require_harness_committed() -> str:
    rel = "bench/nmpins7-gate/lj/run_gate.py"
    status = git_out(["status", "--porcelain", "--", rel])
    if status:
        raise SystemExit("harness differs from HEAD")
    git_out(["rev-parse", f"HEAD:{rel}"])
    head = git_out(["rev-parse", "HEAD"])
    if git_out(["rev-parse", "--abbrev-ref", "HEAD"]) != "bench/nmpins7-gate-lj":
        raise SystemExit("record branch is not bench/nmpins7-gate-lj")
    git_out(["merge-base", "--is-ancestor", BASE, head])
    return head


def verify_builds(arms: dict[str, dict[str, object]]) -> None:
    cat_file(BASE, "commit")
    cat_file(KIT, "commit")
    cat_file(KIT_TREE, "tree")
    base_tree = git_out(["rev-parse", f"{BASE}^{{tree}}"])
    arm_a = arms["A"]
    arm_b = arms["B"]
    arm_c = arms["C"]
    if arm_a["revision"] != BASE or arm_a["tree_sha"] != base_tree:
        raise SystemExit("arm A is not the base revision")
    if arm_b["binary_sha256"] != arm_c["binary_sha256"] or arm_b["revision"] != arm_c["revision"]:
        raise SystemExit("B and C are not the same binary")
    if arm_b["revision"] == BASE:
        raise SystemExit("B is the base binary")
    parent = git_out(["rev-parse", "HEAD^"], arm_b["tree"])
    parent_tree = git_out(["rev-parse", "HEAD^^{tree}"], arm_b["tree"])
    if parent != KIT or parent_tree != KIT_TREE:
        raise SystemExit("B parent is not the kit")
    diff = git_out(["diff", "--numstat", "HEAD^", "HEAD"], arm_b["tree"])
    if diff != "4\t0\texamples/h2h_gmin_proof.rs":
        raise SystemExit(f"B differs from the kit by more than the flag read: {diff}")
    changed = git_out(["diff", "HEAD^", "HEAD"], arm_b["tree"])
    if FLAG not in changed or FLAG_ENV not in changed:
        raise SystemExit("flag read is missing from the B diff")
    if arm_a["binary_sha256"] == arm_b["binary_sha256"]:
        raise SystemExit("A and B binaries are identical")
    versions_a = package_versions(arm_a["tree"])
    versions_b = package_versions(arm_b["tree"])
    if versions_a != versions_b:
        raise SystemExit("package versions differ between A and B")
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != [0]:
        raise SystemExit(f"process affinity is {affinity}")


def run_checked(command: list[str], cwd: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{command[0]} exited {proc.returncode}\n{proc.stderr[-1000:]}")
    return proc.stdout


def verify_coords_and_budgets(arms: dict[str, dict[str, object]]) -> dict[str, str]:
    env = thread_env(False)
    hashes: dict[str, str] = {}
    for system in SYSTEMS:
        budget = run_panel.PAPER_BUDGETS[system]
        reference = float(run_panel.spec(system)[3])
        for arm in (arms["A"], arms["B"]):
            text = run_checked(
                [str(arm["binary"]), "budget", system],
                arm["tree"],
                env,
            )
            lines = [line for line in text.splitlines() if line.strip()]
            if int(lines[0]) != budget or abs(float(lines[1]) - reference) > 0.0:
                raise SystemExit(f"{arm['name']} budget {system} is {lines[:2]}")
        for seed in SEEDS:
            dest_a = Path(tempfile.gettempdir()) / f"nmpins7-coords-a-{system}-{seed}"
            dest_b = Path(tempfile.gettempdir()) / f"nmpins7-coords-b-{system}-{seed}"
            outside_repo(dest_a.parent)
            run_checked(
                [str(arms["A"]["binary"]), "coords", system, str(seed), str(dest_a)],
                arms["A"]["tree"],
                env,
            )
            run_checked(
                [str(arms["B"]["binary"]), "coords", system, str(seed), str(dest_b)],
                arms["B"]["tree"],
                env,
            )
            raw_a = dest_a.read_bytes()
            raw_b = dest_b.read_bytes()
            if raw_a != raw_b:
                raise SystemExit(f"start geometry differs for {system} seed {seed}")
            hashes[f"{system}/{seed}"] = hashlib.sha256(raw_a).hexdigest()
            dest_a.unlink()
            dest_b.unlink()
    return hashes


def parse_result(text: str, system: str, seed: int, budget: int, reference: float) -> dict[str, str]:
    header = ""
    data = ""
    for line in text.splitlines():
        if line.startswith("system,code,seed,"):
            header = line
        elif line.startswith(f"{system},"):
            data = line
    if not header or not data:
        raise RuntimeError(f"{system} seed {seed}: no result row")
    row = dict(zip(header.split(","), data.split(",")))
    if row["system"] != system or int(row["seed"]) != seed or int(row["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: unexpected row")
    best = float(row["best"])
    hit = energy_hits(best, reference, DEFAULT_HIT_TOL)
    printed = row["hit"].strip().lower() in ("1", "true")
    if printed != hit:
        raise RuntimeError(f"{system} seed {seed}: hit flag {row['hit']} best {best}")
    calls = int(row["charged"])
    calls_to_hit = int(row["evals_to_hit"])
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    if calls < 1 or wall < 0.0 or pot < 0.0 or pot > wall + 2e-6:
        raise RuntimeError(f"{system} seed {seed}: times or calls are invalid")
    if hit and not (0 < calls_to_hit <= calls <= budget):
        raise RuntimeError(f"{system} seed {seed}: hit calls {calls_to_hit} charged {calls}")
    if not hit and (calls_to_hit != budget or calls != budget):
        raise RuntimeError(f"{system} seed {seed}: miss calls {calls_to_hit} charged {calls}")
    outside = max(wall - pot, 0.0)
    return {
        "hit": "1" if hit else "0",
        "calls": str(calls),
        "calls_to_hit": str(calls_to_hit),
        "best": row["best"],
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "outside_per_call_s": f"{outside / calls:.12e}",
        "code_revision": row["head"],
        "code_tree": row["tree"],
        "affinity": row["pins"],
    }


def read_affinity(pid: int) -> str:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return ""
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return ""


def execute(binary: Path, tree: Path, system: str, seed: int, flag_on: bool, log: Path) -> tuple[str, str, int, bool, str]:
    env = thread_env(flag_on)
    command = ["taskset", "-c", "0", str(binary), "run", system, str(seed)]
    proc = subprocess.Popen(
        command,
        cwd=tree,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    affinity = read_affinity(proc.pid)
    try:
        stdout, stderr = proc.communicate(timeout=WALL_CAP_S)
        capped = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        capped = True
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        " ".join(command)
        + f"\naffinity={affinity}\nexit={proc.returncode}\ncapped={int(capped)}\n"
        + stdout
        + "\n"
        + stderr
    )
    return stdout, stderr, proc.returncode, capped, affinity


def job_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (row["panel"], row["system"], row["seed"], row["arm"], row["replica"], row["slot"])


def campaign(args: argparse.Namespace) -> None:
    outside_repo(args.logs)
    require_harness_committed()
    arms = {arm["name"]: arm for arm in (parse_arm(text) for text in args.arm)}
    if set(arms) != {"A", "B", "C"}:
        raise SystemExit("arms A, B and C are required")
    verify_builds(arms)
    rows = read_csv(args.rows)
    sessions = read_jsonl(args.sessions)
    done = {job_key(row) for row in rows}
    if sessions and sessions[0].get("event") == "meta":
        session = str(sessions[0]["session"])
        if sessions[0].get("machine_id") != machine_id():
            raise SystemExit("existing session has a different machine id")
    else:
        if rows or sessions:
            raise SystemExit("rows exist without a session meta record")
        session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        versions = package_versions(arms["A"]["tree"])
        append_jsonl(
            args.sessions,
            {
                "event": "meta",
                "session": session,
                "utc": now_utc(),
                "base": BASE,
                "base_tree": git_out(["rev-parse", f"{BASE}^{{tree}}"]),
                "kit": KIT,
                "kit_tree": KIT_TREE,
                "build_a": arms["A"]["revision"],
                "build_a_tree": arms["A"]["tree_sha"],
                "build_bc": arms["B"]["revision"],
                "build_bc_tree": arms["B"]["tree_sha"],
                "binary_a_sha256": arms["A"]["binary_sha256"],
                "binary_bc_sha256": arms["B"]["binary_sha256"],
                "harness_revision": git_out(["rev-parse", "HEAD"]),
                "rustc": subprocess.check_output(["rustup", "run", "1.88.0", "rustc", "--version"], text=True).strip(),
                "cargo": subprocess.check_output(["rustup", "run", "1.88.0", "cargo", "--version"], text=True).strip(),
                "packages": versions,
                "lock_sha256": sha256_file(arms["A"]["tree"] / "Cargo.lock"),
                "machine_id": machine_id(),
                "core": "0",
                "affinity": "0",
                "thread_env": recorded_thread_env(),
                "wall_cap_s": WALL_CAP_S,
                "seeds": list(SEEDS),
                "systems": list(SYSTEMS),
                "budgets": {system: run_panel.PAPER_BUDGETS[system] for system in SYSTEMS},
                "references": {system: run_panel.spec(system)[3] for system in SYSTEMS},
                "hit_tol": DEFAULT_HIT_TOL,
                "flag": FLAG,
                "flag_env": FLAG_ENV,
                "bonferroni_m": BONFERRONI_M,
                "alpha": ALPHA,
                "outside_ratio_limit": OUTSIDE_RATIO_LIMIT,
            },
        )
    if not any(record.get("event") == "coords" for record in read_jsonl(args.sessions)):
        hashes = verify_coords_and_budgets(arms)
        append_jsonl(args.sessions, {"event": "coords", "session": session, "sha256": hashes})
    mid = machine_id()
    seq = max((int(row["seq"]) for row in rows), default=0)
    for job in planned_jobs():
        key = (
            str(job["panel"]),
            str(job["system"]),
            str(job["seed"]),
            str(job["arm"]),
            str(job["replica"]),
            str(job["slot"]),
        )
        if key in done:
            continue
        arm_name = "A" if job["panel"] == "noise" else str(job["arm"])
        arm = arms[arm_name]
        flag_on = arm_name == "C"
        system = str(job["system"])
        seed = int(job["seed"])
        log = args.logs / f"{job['panel']}-{system}-{seed}-{job['arm']}-{job['replica']}-{job['slot']}.txt"
        stdout, stderr, code, capped, affinity = execute(arm["binary"], arm["tree"], system, seed, flag_on, log)
        budget = run_panel.PAPER_BUDGETS[system]
        reference = float(run_panel.spec(system)[3])
        if code != 0 or capped:
            raise SystemExit(f"{system} seed {seed} arm {arm_name} exited {code} capped {int(capped)}")
        parsed = parse_result(stdout, system, seed, budget, reference)
        if parsed["code_revision"] != arm["revision"] or parsed["code_tree"] != arm["tree_sha"]:
            raise SystemExit(f"{system} seed {seed} printed a different revision")
        if parsed["affinity"] != "0" or (affinity and affinity != "0"):
            raise SystemExit(f"{system} seed {seed} affinity {parsed['affinity']} proc {affinity}")
        seq += 1
        row = {
            "seq": str(seq),
            "session": session,
            "panel": str(job["panel"]),
            "system": system,
            "seed": str(seed),
            "arm": str(job["arm"]),
            "replica": str(job["replica"]),
            "slot": str(job["slot"]),
            "perm": str(job["perm"]),
            "budget": str(budget),
            "reference": f"{reference:.8f}",
            "hit": parsed["hit"],
            "capped": "0",
            "calls": parsed["calls"],
            "calls_to_hit": parsed["calls_to_hit"],
            "best": parsed["best"],
            "wall_s": parsed["wall_s"],
            "pot_s": parsed["pot_s"],
            "outside_s": parsed["outside_s"],
            "outside_per_call_s": parsed["outside_per_call_s"],
            "core": "0",
            "affinity": parsed["affinity"],
            "machine_id": mid,
            "code_revision": parsed["code_revision"],
            "code_tree": parsed["code_tree"],
            "binary_sha256": str(arm["binary_sha256"]),
            "flag": "on" if flag_on else ("off" if arm_name == "B" else "absent"),
            "utc": now_utc(),
        }
        append_csv(args.rows, row)
        append_jsonl(
            args.sessions,
            {
                "event": "run",
                "session": session,
                "panel": row["panel"],
                "system": system,
                "seed": seed,
                "arm": row["arm"],
                "replica": row["replica"],
                "slot": int(row["slot"]),
                "utc": row["utc"],
                "hit": int(row["hit"]),
                "calls": int(row["calls"]),
                "wall_s": float(row["wall_s"]),
            },
        )
        print(
            f"{row['panel']} {system} seed {seed} arm {row['arm']} slot {row['slot']} "
            f"hit {row['hit']} calls {row['calls']} to_hit {row['calls_to_hit']} wall {row['wall_s']}",
            flush=True,
        )


def calls_for_test(row: dict[str, str]) -> float:
    """Evaluations to the minimum. A miss is the budget the row already stores."""
    return float(row["calls_to_hit"])


def group_rows(rows: list[dict[str, str]], panel: str, system: str, arm: str, replica: str | None = None) -> list[dict[str, str]]:
    chosen = [
        row
        for row in rows
        if row["panel"] == panel and row["system"] == system and row["arm"] == arm and (replica is None or row["replica"] == replica)
    ]
    return sorted(chosen, key=lambda row: int(row["seed"]))


def compare(rows: list[dict[str, str]], system: str, other_name: str) -> dict[str, object]:
    base = group_rows(rows, "gate", system, "A")
    other = group_rows(rows, "gate", system, other_name)
    if [int(row["seed"]) for row in base] != list(SEEDS) or [int(row["seed"]) for row in other] != list(SEEDS):
        raise SystemExit(f"{system} {other_name} does not have seeds 2000-2011")
    hits_b = sum(row["hit"] == "1" for row in base)
    hits_o = sum(row["hit"] == "1" for row in other)
    calls_b = [float(row["calls_to_hit"]) for row in base]
    calls_o = [float(row["calls_to_hit"]) for row in other]
    ranked_b = [calls_for_test(row) for row in base]
    ranked_o = [calls_for_test(row) for row in other]
    out_b = [float(row["outside_per_call_s"]) for row in base]
    out_o = [float(row["outside_per_call_s"]) for row in other]
    u_calls, p_calls = mann_whitney_greater(ranked_o, ranked_b)
    u_out, p_out = mann_whitney_greater(out_o, out_b)
    med_calls_b = type7(calls_b, 0.5)
    med_calls_o = type7(calls_o, 0.5)
    med_out_b = type7(out_b, 0.5)
    med_out_o = type7(out_o, 0.5)
    ratio = med_out_o / med_out_b
    num, den = fisher_counts(hits_o, len(other), hits_b, len(base))
    censored = any(row["capped"] == "1" for row in base + other)
    return {
        "hits_base": hits_b,
        "hits_other": hits_o,
        "n": len(base),
        "fisher_p": num / den,
        "fisher_num": num,
        "fisher_den": den,
        "fisher_loss": (not censored) and fisher_loss_exact(hits_o, len(other), hits_b, len(base)),
        "calls_base": med_calls_b,
        "calls_other": med_calls_o,
        "calls_iqr_base": iqr(calls_b),
        "calls_iqr_other": iqr(calls_o),
        "calls_u": u_calls,
        "calls_p": p_calls,
        "calls_loss": (not censored) and mw_loss(p_calls),
        "out_base": med_out_b,
        "out_other": med_out_o,
        "out_ratio": ratio,
        "out_u": u_out,
        "out_p": p_out,
        "out_loss": (not censored) and ratio > OUTSIDE_RATIO_LIMIT,
        "censored": censored,
        "charged_base": type7([float(row["calls"]) for row in base], 0.5),
        "charged_other": type7([float(row["calls"]) for row in other], 0.5),
    }


def balance(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        if row["panel"] != "gate":
            continue
        key = (row["system"], row["slot"], row["arm"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def noise_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    left = group_rows(rows, "noise", "lj38", "A", "0")
    right = group_rows(rows, "noise", "lj38", "A", "1")
    if [int(row["seed"]) for row in left] != list(SEEDS) or [int(row["seed"]) for row in right] != list(SEEDS):
        raise SystemExit("A/A noise rows are incomplete")
    by_seed_right = {int(row["seed"]): row for row in right}
    ratios = []
    agree = 0
    for row in left:
        other = by_seed_right[int(row["seed"])]
        ratios.append(float(other["outside_per_call_s"]) / float(row["outside_per_call_s"]))
        if (
            row["hit"] == other["hit"]
            and row["calls"] == other["calls"]
            and row["calls_to_hit"] == other["calls_to_hit"]
            and f"{float(row['best']):.8f}" == f"{float(other['best']):.8f}"
        ):
            agree += 1
    out_l = [float(row["outside_per_call_s"]) for row in left]
    out_r = [float(row["outside_per_call_s"]) for row in right]
    _u, p = run_panel.mann_whitney(out_l, out_r)
    return {
        "median_0": type7(out_l, 0.5),
        "median_1": type7(out_r, 0.5),
        "ratio_of_medians": type7(out_r, 0.5) / type7(out_l, 0.5),
        "ratio_median": type7(ratios, 0.5),
        "ratio_iqr": iqr(ratios),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "ratios": ratios,
        "mannwhitney_p": p,
        "agree": agree,
        "n": len(left),
        "hits_0": sum(row["hit"] == "1" for row in left),
        "hits_1": sum(row["hit"] == "1" for row in right),
    }


def fmt_p(p: float) -> str:
    if p >= 1.0:
        return "1"
    return f"{p:.6g}"


def fmt_calls(value: float) -> str:
    return f"{value:.2f}"


def check_records(rows: list[dict[str, str]], sessions: list[dict]) -> dict:
    if not sessions or sessions[0].get("event") != "meta":
        raise SystemExit("session meta record is missing")
    meta = sessions[0]
    cat_file(str(meta["base"]), "commit")
    cat_file(str(meta["kit"]), "commit")
    cat_file(str(meta["kit_tree"]), "tree")
    cat_file(str(meta["build_a"]), "commit")
    cat_file(str(meta["build_a_tree"]), "tree")
    cat_file(str(meta["build_bc"]), "commit")
    cat_file(str(meta["build_bc_tree"]), "tree")
    cat_file(str(meta["harness_revision"]), "commit")
    data_commit = git_out(["rev-parse", "HEAD"])
    for rel in ("bench/nmpins7-gate/lj/rows.csv", "bench/nmpins7-gate/lj/sessions.jsonl"):
        git_out(["cat-file", "-e", f"{data_commit}:{rel}"])
        if git_out(["status", "--porcelain", "--", rel]):
            raise SystemExit(f"{rel} is not committed")
    meta = dict(meta)
    meta["data_commit"] = data_commit
    if meta["base"] != BASE or meta["kit"] != KIT or meta["kit_tree"] != KIT_TREE:
        raise SystemExit("session provenance does not match the kit")
    if len(rows) != len(planned_jobs()):
        raise SystemExit(f"expected {len(planned_jobs())} rows, found {len(rows)}")
    expected = planned_jobs()
    by_key = {job_key(row): row for row in rows}
    if len(by_key) != len(rows):
        raise SystemExit("duplicate measurement keys")
    for job in expected:
        key = (str(job["panel"]), str(job["system"]), str(job["seed"]), str(job["arm"]), str(job["replica"]), str(job["slot"]))
        if key not in by_key:
            raise SystemExit(f"missing row {key}")
    counts = balance(rows)
    for system in SYSTEMS:
        for slot in ("0", "1", "2"):
            for arm in ("A", "B", "C"):
                if counts.get((system, slot, arm), 0) != 4:
                    raise SystemExit(f"unbalanced {(system, slot, arm)} = {counts.get((system, slot, arm), 0)}")
    noise_slots: dict[tuple[str, str], int] = {}
    for row in rows:
        if row["panel"] != "noise":
            continue
        key = (row["slot"], row["replica"])
        noise_slots[key] = noise_slots.get(key, 0) + 1
        if row["perm"] not in ("01", "10"):
            raise SystemExit("noise perm is not ABBA")
    for slot in ("0", "1"):
        for replica in ("0", "1"):
            if noise_slots.get((slot, replica), 0) != 6:
                raise SystemExit(f"A/A slot balance {slot, replica}")
    sessions_ids = {row["session"] for row in rows}
    if sessions_ids != {meta["session"]}:
        raise SystemExit("more than one session")
    if any(row["machine_id"] != meta["machine_id"] or row["affinity"] != "0" or row["core"] != "0" for row in rows):
        raise SystemExit("machine, core or affinity is not uniform")
    if any(row["capped"] != "0" for row in rows):
        raise SystemExit("a capped row needs the censored report path")
    for row in rows:
        budget = run_panel.PAPER_BUDGETS[row["system"]]
        reference = float(run_panel.spec(row["system"])[3])
        if int(row["budget"]) != budget or abs(float(row["reference"]) - reference) > 5e-9:
            raise SystemExit(f"budget mismatch {row['system']}")
        hit = energy_hits(float(row["best"]), reference, DEFAULT_HIT_TOL)
        if (row["hit"] == "1") != hit:
            raise SystemExit(f"hit mismatch {row['system']} {row['seed']} {row['arm']}")
        outside = max(float(row["wall_s"]) - float(row["pot_s"]), 0.0)
        if abs(outside - float(row["outside_s"])) > 2e-6:
            raise SystemExit(f"outside time mismatch {row['system']} {row['seed']}")
        per_call = float(row["outside_s"]) / int(row["calls"])
        if abs(per_call - float(row["outside_per_call_s"])) / per_call > 1e-9:
            raise SystemExit(f"outside per call mismatch {row['system']} {row['seed']}")
    order = sorted(rows, key=lambda row: int(row["seq"]))
    planned_keys = [
        (str(job["panel"]), str(job["system"]), str(job["seed"]), str(job["arm"]), str(job["replica"]), str(job["slot"]))
        for job in expected
    ]
    observed = [job_key(row) for row in order]
    if observed != planned_keys:
        raise SystemExit("row order does not follow the rotation")
    return meta


def write_report(rows: list[dict[str, str]], sessions: list[dict], path: Path) -> str:
    meta = check_records(rows, sessions)
    noise = noise_summary(rows)
    lines = [
        "# nmpins7 Lennard-Jones gate",
        "",
        "Cells LJ38, LJ75, LJ98 and LJ104. Seeds 2000-2011. "
        "Arms A, B and C ran in one session on core 0.",
        "",
    ]
    verdicts = []
    compares = {}
    for system in SYSTEMS:
        pair_b = compare(rows, system, "B")
        pair_c = compare(rows, system, "C")
        compares[system] = (pair_b, pair_c)
        losses = []
        if pair_b["fisher_loss"]:
            losses.append("hits")
        if pair_b["calls_loss"]:
            losses.append("calls")
        if pair_b["out_loss"]:
            losses.append("outside time per call")
        if pair_b["censored"]:
            losses.append("censored")
        verdicts.append((system, losses, pair_b))
    if any(losses for _system, losses, _pair in verdicts):
        where = ", ".join(f"{system} ({', '.join(losses)})" for system, losses, _pair in verdicts if losses)
        lines.append(f"B loses on {where}.")
    else:
        lines.append("B loses nowhere.")
    lines.append("")
    lines.append("Calls are evaluations to the minimum, and a miss is entered at the budget. Outside time per call is `(wall - potential time) / charged calls`. The outside-time loss is a median ratio above 1.1. Hit and call losses use a one-sided test and Bonferroni m = 30 at alpha 0.05. C is the same comparison with no pass or fail.")
    lines.append("")
    lines.append("| cell | arm | hits | median calls | IQR | median outside/call (s) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for system in SYSTEMS:
        for arm in ("A", "B", "C"):
            grouped = group_rows(rows, "gate", system, arm)
            calls = [float(row["calls_to_hit"]) for row in grouped]
            outside = [float(row["outside_per_call_s"]) for row in grouped]
            hits = sum(row["hit"] == "1" for row in grouped)
            lines.append(
                f"| {system} | {arm} | {hits}/12 | {fmt_calls(type7(calls, 0.5))} | {fmt_calls(iqr(calls))} | {type7(outside, 0.5):.6e} |"
            )
    lines.append("")
    lines.append("| cell | contrast | Fisher raw | Fisher Bonferroni | calls U | calls p raw | calls p Bonferroni | outside ratio | outside U | outside p raw | outside p Bonferroni | verdict |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for system in SYSTEMS:
        for name, pair in (("B", compares[system][0]), ("C", compares[system][1])):
            if name == "B":
                if pair["fisher_loss"] or pair["calls_loss"] or pair["out_loss"] or pair["censored"]:
                    verdict = "fail"
                else:
                    verdict = "pass"
            else:
                verdict = "not gated"
            lines.append(
                f"| {system} | {name} vs A | {fmt_p(pair['fisher_p'])} | {fmt_p(bonferroni(pair['fisher_p']))} | "
                f"{pair['calls_u']:.1f} | {fmt_p(pair['calls_p'])} | {fmt_p(bonferroni(pair['calls_p']))} | "
                f"{pair['out_ratio']:.4f} | {pair['out_u']:.1f} | {fmt_p(pair['out_p'])} | "
                f"{fmt_p(bonferroni(pair['out_p']))} | {verdict} |"
            )
    lines.append("")
    lines.append(
        f"A/A on LJ38, same binary, n = 12, ABBA. Replicas agree on hit, charged calls, calls to the hit and best energy on {noise['agree']}/12 seeds. "
        f"Hits {noise['hits_0']}/12 and {noise['hits_1']}/12. "
        f"Median outside time per call {noise['median_0']:.6e} and {noise['median_1']:.6e} "
        f"(ratio of medians {noise['ratio_of_medians']:.4f}). "
        f"Per-seed replica-1/replica-0 ratios have median {noise['ratio_median']:.4f}, "
        f"IQR {noise['ratio_iqr']:.4f}, range {noise['ratio_min']:.4f} to {noise['ratio_max']:.4f}. "
        f"Two-sided Mann-Whitney p on outside time per call is {fmt_p(noise['mannwhitney_p'])}."
    )
    lines.append("")
    lines.append("| cell | B/A outside ratio | inside A/A per-seed range |")
    lines.append("| --- | ---: | --- |")
    for system in SYSTEMS:
        ratio = compares[system][0]["out_ratio"]
        inside = noise["ratio_min"] <= ratio <= noise["ratio_max"]
        lines.append(f"| {system} | {ratio:.4f} | {'yes' if inside else 'no'} |")
    lines.append("")
    lines.append("Slot counts for the gate, each (slot, arm) pair:")
    lines.append("")
    lines.append("| cell | slot | A | B | C |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    counts = balance(rows)
    for system in SYSTEMS:
        for slot in ("0", "1", "2"):
            lines.append(
                f"| {system} | {slot} | {counts[(system, slot, 'A')]} | {counts[(system, slot, 'B')]} | {counts[(system, slot, 'C')]} |"
            )
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append(
        "Release binaries, `cargo build --release --locked`. "
        "The whole process tree was pinned with `taskset -c 0`. "
        "OpenMP, OpenBLAS, MKL, Rayon, NumExpr and the other thread-count variables were 1, and the dynamic thread switches were off. "
        "No MPI launcher was used. "
        f"The wall cap was {WALL_CAP_S:.0f} s per seed. "
        "No run was discarded and none was a warm-up. "
        f"The hit tolerance is {DEFAULT_HIT_TOL}. "
        "Medians and IQRs are Hyndman-Fan type 7. "
        "Fisher's exact test is the one-sided hypergeometric tail for fewer hits. "
        "The Mann-Whitney test is the normal approximation with the tie correction and the continuity correction, upper tail, alternative that the named arm is higher. "
        f"Bonferroni uses m = {BONFERRONI_M}. A call or hit loss requires the adjusted p to be below {ALPHA}. "
        f"An outside-time loss requires the ratio of median outside time per call to exceed {OUTSIDE_RATIO_LIMIT}. "
        f"Arm C sets `{FLAG_ENV}=1`, which sets `Config::{FLAG}`. Arm B leaves that variable unset, so the field stays at its default. Arm A does not have the field."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Base commit `{meta['base']}`, tree `{meta['base_tree']}`.")
    lines.append(f"- Kit commit `{meta['kit']}`, tree `{meta['kit_tree']}`.")
    lines.append(f"- Arm A build `{meta['build_a']}`, tree `{meta['build_a_tree']}`, binary sha256 `{meta['binary_a_sha256']}`.")
    lines.append(
        f"- Arms B and C build `{meta['build_bc']}`, tree `{meta['build_bc_tree']}`, "
        f"binary sha256 `{meta['binary_bc_sha256']}`. Parent is the kit commit. "
        "The only change is the environment read of the flag."
    )
    lines.append(f"- Harness commit `{meta['harness_revision']}`.")
    lines.append(f"- Measurement commit `{meta['data_commit']}`.")
    lines.append(f"- Session `{meta['session']}` at `{meta['utc']}`.")
    lines.append(f"- Machine id sha256 `{meta['machine_id']}`. Core 0, affinity 0.")
    lines.append(f"- Compiler `{meta['rustc']}`. Cargo `{meta['cargo']}`.")
    lines.append(f"- Lock sha256 `{meta['lock_sha256']}`.")
    for name in PACKAGES:
        lines.append(f"- `{name}` {meta['packages'][name]}.")
    lines.append("- Budgets: " + ", ".join(f"{system} {meta['budgets'][system]}" for system in SYSTEMS) + ".")
    lines.append("- References: " + ", ".join(f"{system} {meta['references'][system]}" for system in SYSTEMS) + ".")
    lines.append("")
    text = "\n".join(lines)
    if "\r" in text:
        raise SystemExit("report has CR")
    path.write_text(text)
    return text


def self_check() -> None:
    jobs = planned_jobs()
    gate = [job for job in jobs if job["panel"] == "gate"]
    noise = [job for job in jobs if job["panel"] == "noise"]
    if len(gate) != 144 or len(noise) != 24:
        raise SystemExit("schedule length")
    counts: dict[tuple[str, int, str], int] = {}
    perms_used: dict[str, int] = {}
    for job in gate:
        if job["perm"] != "".join(perm_for(int(job["seed"]))):
            raise SystemExit("perm assignment")
        perms_used[str(job["perm"])] = perms_used.get(str(job["perm"]), 0) + 1
        key = (str(job["system"]), int(job["slot"]), str(job["arm"]))
        counts[key] = counts.get(key, 0) + 1
    if any(value != 4 for value in counts.values()) or len(counts) != 36:
        raise SystemExit("gate balance")
    if any(value != 24 for value in perms_used.values()) or len(perms_used) != 6:
        raise SystemExit("permutation coverage")
    first = [int(job["replica"]) for job in noise if job["slot"] == 0]
    if first != [0, 1, 1, 0] * 3:
        raise SystemExit("ABBA order")
    samples = [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ([10.0, 11.0, 12.0, 13.0], [1.0, 2.0, 3.0, 4.0]),
        ([1.0, 1.0, 2.0], [1.0, 2.0, 2.0]),
        ([1.0, math.inf], [2.0, 3.0]),
    ]
    for left, right in samples:
        u1, p1 = mann_whitney_two(left, right)
        u2, p2 = run_panel.mann_whitney(left, right)
        if u1 != u2 or abs(p1 - p2) > 1e-12:
            raise SystemExit(f"mann-whitney drift {u1} {u2} {p1} {p2}")
    num, den = fisher_counts(9, 12, 12, 12)
    expect = math.comb(21, 9) * math.comb(3, 3)
    if num != expect or den != math.comb(24, 12):
        raise SystemExit("fisher tail")
    tie_num, tie_den = fisher_counts(12, 12, 12, 12)
    if tie_num != tie_den:
        raise SystemExit("fisher tie")
    if not fisher_loss_exact(0, 12, 12, 12):
        raise SystemExit("fisher should lose when the other arm misses every seed")
    if fisher_loss_exact(12, 12, 12, 12):
        raise SystemExit("fisher should not lose on a tie")
    _u, p_greater = mann_whitney_greater([10.0, 11.0, 12.0, 13.0], [1.0, 2.0, 3.0, 4.0])
    _u, p_two = mann_whitney_two([10.0, 11.0, 12.0, 13.0], [1.0, 2.0, 3.0, 4.0])
    if abs(p_two - 2 * p_greater) > 1e-9 or p_greater >= 0.05:
        raise SystemExit(f"one-sided mann-whitney {p_greater} {p_two}")
    _u, p_less = mann_whitney_greater([1.0, 2.0], [10.0, 11.0, 12.0])
    if p_less < 0.5 or mw_loss(p_less) or not mw_loss(1e-6) or mw_loss(0.01):
        raise SystemExit("lower sample reported as greater")
    for system in SYSTEMS:
        if system not in run_panel.PAPER_BUDGETS:
            raise SystemExit(system)
    print("self-check ok", flush=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arm", action="append", required=True)
    run.add_argument("--logs", type=Path, required=True)
    run.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    run.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    report = sub.add_parser("report")
    report.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    report.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    report.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    sub.add_parser("self-check")
    args = parser.parse_args(argv)
    if args.cmd == "self-check":
        self_check()
        return 0
    if args.cmd == "run":
        campaign(args)
        return 0
    rows = read_csv(args.rows)
    sessions = read_jsonl(args.sessions)
    text = write_report(rows, sessions, args.report)
    print(text.splitlines()[4] if len(text.splitlines()) > 4 else "report written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
