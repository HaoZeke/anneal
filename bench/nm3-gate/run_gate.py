# Copyright (c) anneal developers
"""Serial A/B gate on the recommended preset and the paper budgets.

Arm A is the unmodified test-only parent. Arm B is that parent plus the
motif-removal patches. One fresh process per run, core 0, one thread.
Even seeds run A then B. Odd seeds run B then A. Rows are appended and
never rewritten.

n is 12 (seeds 2000-2011) for every supported system. That choice was
fixed from walls already stored under bench/, before any row of this
gate existed. The slowest stored full budget, Au98 at about 470 s, is
24 runs and about 3.1 h. LJ38, LJ98 and LJ104 are n=12 in any case.
"""

from __future__ import annotations

import csv
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize  # noqa: E402

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.csv"
SEEDS = tuple(range(2000, 2012))
EXPECT_A = "ef1e1d07ad214d545cfa0bcb0c2e15b8be3bcbcb"
EXPECT_B_TREE = "e54c8935d5627dc140b0ef1417aacdc44173cb07"
HIT_TOL = 0.001
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
)

# driver, budget. csv drivers print a system,code,seed line. kv is h2h_seed.
LJ = (
    ("lj38", 400_000),
    ("lj55", 1_000_000),
    ("lj75", 4_000_000),
    ("lj98", 4_000_000),
    ("lj103", 4_000_000),
    ("lj104", 4_000_000),
)
MORSE = (
    ("m80r3", 4_000_000),
    ("m80r6", 4_000_000),
    ("m80r10", 4_000_000),
    ("m80r14", 4_000_000),
)
AU = (
    ("au38", 400_000),
    ("au55", 1_000_000),
    ("au75", 4_000_000),
    ("au98", 4_000_000),
)
# Cambridge headers used by the TIP4P runners, kJ/mol.
TIP4P = (
    ("tip4p8", 8, -305.5183221676, 2_000_000),
    ("tip4p12", 12, -492.9082357122, 2_000_000),
    ("tip4p15", 15, -628.3722525077, 2_000_000),
    ("tip4p16", 16, -681.1925945898, 2_000_000),
    ("tip4p20", 20, -872.9887544952, 2_000_000),
    ("tip4p21", 21, -916.7074302940, 2_000_000),
)

ROW_FIELDS = [
    "system",
    "arm",
    "driver",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "code_revision",
    "host",
]


def systems() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, budget in LJ:
        rows.append(
            {
                "system": name,
                "driver": "h2h_gmin_proof",
                "budget": budget,
                "kind": "csv",
                "argv": ["run", name],
            }
        )
    for name, budget in MORSE:
        rows.append(
            {
                "system": name,
                "driver": "h2h_morse",
                "budget": budget,
                "kind": "csv",
                "argv": ["run", name],
            }
        )
    for name, budget in AU:
        rows.append(
            {
                "system": name,
                "driver": "h2h_metal",
                "budget": budget,
                "kind": "csv",
                "argv": ["run", name],
            }
        )
    for name, n, reference, budget in TIP4P:
        rows.append(
            {
                "system": name,
                "driver": "h2h_seed",
                "budget": budget,
                "kind": "kv",
                "n": n,
                "reference": reference,
            }
        )
    return rows


def revision(tree: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def tree_id(tree: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def require_host() -> str:
    host = socket.gethostname()
    if host != "bench":
        raise SystemExit(f"uts hostname is {host}, expected bench")
    if os.sched_getaffinity(0) != {0}:
        raise SystemExit(f"affinity is {sorted(os.sched_getaffinity(0))}, expected core 0")
    return host


def env_of() -> dict[str, str]:
    env = os.environ.copy()
    for key in THREAD_KEYS:
        env[key] = "1"
    for key in list(env):
        if key.startswith("H2H_"):
            env.pop(key)
    return env


def budget_of(binary: Path, system: str) -> int:
    proc = subprocess.run(
        [str(binary), "budget", system],
        check=True,
        capture_output=True,
        text=True,
    )
    line = proc.stdout.splitlines()[0].strip()
    return int(line)


def done(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as handle:
        return {(row["system"], row["arm"], row["seed"]) for row in csv.DictReader(handle)}


def append(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def note(text: str) -> None:
    print(text, flush=True)
    progress = os.environ.get("NM3_PROGRESS")
    if progress:
        with open(progress, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()


def parse_csv(text: str) -> dict[str, str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("system,code,seed,"):
            header = line.split(",")
            data = lines[index + 1].split(",")
            if len(data) < len(header):
                raise RuntimeError(f"short row {lines[index + 1]!r}")
            return dict(zip(header, data[: len(header)]))
    raise RuntimeError("no system,code,seed row")


def parse_kv(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("h2h_row "):
            fields = {}
            for part in line.split()[1:]:
                key, value = part.split("=", 1)
                fields[key] = value
            return fields
    raise RuntimeError("no h2h_row")


def as_hit(text: str) -> str:
    return "1" if text in {"1", "true", "True"} else "0"


def run_one(
    binary: Path,
    cwd: Path,
    spec: dict[str, object],
    seed: int,
    env: dict[str, str],
) -> dict[str, str]:
    if spec["kind"] == "csv":
        argv = list(spec["argv"]) + [str(seed)]
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), *argv],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3 * 60 * 60,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{spec['system']} seed {seed} exited {proc.returncode}\n{proc.stderr[-1500:]}"
            )
        parsed = parse_csv(proc.stdout)
        wall = float(parsed["wall_s"])
        pot = float(parsed["pot_s"])
        outside = parsed.get("outside_s")
        outside_s = float(outside) if outside else max(0.0, wall - pot)
        return {
            "budget": parsed["budget"],
            "hit": as_hit(parsed["hit"]),
            "evals_to_hit": parsed["evals_to_hit"],
            "charged": parsed["charged"],
            "best": parsed["best"],
            "wall_s": f"{wall:.6f}",
            "pot_s": f"{pot:.6f}",
            "outside_s": f"{outside_s:.6f}",
        }
    with tempfile.TemporaryDirectory(prefix="nm3-") as scratch:
        prefix = str(Path(scratch) / "run")
        stop = repr(float(spec["reference"]) + HIT_TOL)
        argv = [
            "tip4p",
            str(spec["n"]),
            "0",
            str(spec["budget"]),
            str(seed),
            "recommended",
            stop,
            prefix,
        ]
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), *argv],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3 * 60 * 60,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{spec['system']} seed {seed} exited {proc.returncode}\n{proc.stderr[-1500:]}"
        )
    parsed = parse_kv(proc.stdout)
    return {
        "budget": parsed["budget"],
        "hit": as_hit(parsed["hit"]),
        "evals_to_hit": parsed["evals_to_hit"],
        "charged": parsed["charged"],
        "best": parsed["best"],
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "outside_s": parsed["outside_s"],
    }


def main() -> None:
    summarize.self_check()
    host = require_host()
    bin_a = Path(os.environ["NM3_A_BIN"])
    bin_b = Path(os.environ["NM3_B_BIN"])
    cwd_a = Path(os.environ["NM3_A_CWD"])
    cwd_b = Path(os.environ["NM3_B_CWD"])
    rev_a = revision(cwd_a)
    rev_b = revision(cwd_b)
    if rev_a != EXPECT_A:
        raise SystemExit(f"arm A is {rev_a}")
    if tree_id(cwd_b) != EXPECT_B_TREE:
        raise SystemExit(f"arm B tree is {tree_id(cwd_b)}")
    arms = {
        "A": (bin_a, cwd_a, rev_a),
        "B": (bin_b, cwd_b, rev_b),
    }
    plan = systems()
    for spec in plan:
        if spec["kind"] != "csv":
            continue
        for label, (directory, _cwd, _rev) in arms.items():
            binary = directory / str(spec["driver"])
            reported = budget_of(binary, str(spec["system"]))
            if reported != spec["budget"]:
                raise SystemExit(f"{label} {spec['system']} budget {reported} != {spec['budget']}")
    note(
        f"plan n=12 seeds 2000-2011 systems={len(plan)} "
        f"A={rev_a} B={rev_b} host={host}"
    )
    have = done(ROWS)
    env = env_of()
    for spec in plan:
        for seed in SEEDS:
            order = ("A", "B") if seed % 2 == 0 else ("B", "A")
            for arm in order:
                key = (str(spec["system"]), arm, str(seed))
                if key in have:
                    note(f"keep {spec['system']} {arm} {seed}")
                    continue
                directory, cwd, rev = arms[arm]
                binary = directory / str(spec["driver"])
                parsed = run_one(binary, cwd, spec, seed, env)
                if int(parsed["budget"]) != int(spec["budget"]):
                    raise SystemExit(
                        f"{spec['system']} reported budget {parsed['budget']}"
                    )
                row = {
                    "system": str(spec["system"]),
                    "arm": arm,
                    "driver": str(spec["driver"]),
                    "seed": str(seed),
                    "budget": parsed["budget"],
                    "hit": parsed["hit"],
                    "evals_to_hit": parsed["evals_to_hit"],
                    "charged": parsed["charged"],
                    "best": parsed["best"],
                    "wall_s": parsed["wall_s"],
                    "pot_s": parsed["pot_s"],
                    "outside_s": parsed["outside_s"],
                    "code_revision": rev,
                    "host": host,
                }
                append(ROWS, row)
                have.add(key)
                note(
                    f"ran {spec['system']} {arm} seed {seed} "
                    f"hit {parsed['hit']} calls {parsed['evals_to_hit']} "
                    f"wall_s {parsed['wall_s']} pot_s {parsed['pot_s']} "
                    f"outside_s {parsed['outside_s']}"
                )
    note("gate rows complete")


if __name__ == "__main__":
    main()
