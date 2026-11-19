#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial A/B of Config::recommended on the Morse ASE panel.

Uses scripts/ase_nonlj/run_one.py. One core, one thread, no warm-up.
Seed order is ABBA: even offsets run the left arm then the right arm,
odd offsets run them the other way. The sequence number of every process
is recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shlex
import site
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_ONE = ROOT / "scripts" / "ase_nonlj" / "run_one.py"
PANEL_ROWS = ROOT / "bench" / "ase-nonlj" / "rows.csv"
# Revision enter_bench.sh reads when it turns for_cluster walls into the cap.
CAP_REVISION = "9a30f3190b3bb7040c675a904f8388eb68224151"
REQUESTED_A = "90b88235fe54abace3bde32789582eb6573ed40a"
SYSTEMS = ("m38r6", "m38r14", "m55r6", "m55r14")
AA_SYSTEMS = ("m38r14", "m55r6")
SEEDS = tuple(range(2000, 2012))
BUDGETS = {"m38r6": 400_000, "m38r14": 400_000, "m55r6": 1_000_000, "m55r14": 1_000_000}
ROW_FIELDS = [
    "band",
    "system",
    "arm",
    "seed",
    "arm_order",
    "run_seq",
    "budget",
    "cap_s",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "inside_s",
    "outside_s",
    "stop",
    "code_revision",
    "tree",
    "binary_sha256",
    "machine_id",
    "seat",
    "layout",
]
HOST_FIELDS = [
    "role",
    "requested_revision",
    "code_revision",
    "tree",
    "binary_sha256",
    "layout",
    "porcelain",
    "machine_id",
    "seat",
    "logical_cpus",
    "pinned_core",
    "compiler",
    "thread_env",
    "cap_revision",
    "cap_m38r6",
    "cap_m38r14",
    "cap_m55r6",
    "cap_m55r14",
]
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_PROC_BIND": "TRUE",
    "OMP_PLACES": "cores",
    "MKL_NUM_THREADS": "1",
    "MKL_DOMAIN_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "JULIA_NUM_THREADS": "1",
}


def machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    return hashlib.sha256(raw).hexdigest()[:12]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def wall_caps(path: Path, revision: str) -> dict[str, str]:
    rows = read_csv(path)
    caps = {}
    for system in SYSTEMS:
        base = [
            row
            for row in rows
            if row["arm"] == "for_cluster"
            and row["system"] == system
            and row["host"] == "bench"
            and row["code_revision"] == revision
        ]
        seeds = tuple(sorted(int(row["seed"]) for row in base))
        if seeds != SEEDS:
            raise SystemExit(f"wall cap needs for_cluster seeds 2000-2011 for {system}, found {seeds}")
        caps[system] = f"{3.0 * max(float(row['wall_s']) for row in base):.6f}"
    return caps


def compiler_line() -> str:
    proc = subprocess.run(
        ["rustc", "+1.88.0", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def logical_cpus() -> str:
    return str(os.cpu_count() or 0)


def git_porcelain(worktree: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(worktree), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return "empty" if proc.stdout.strip() == "" else "dirty"


def require_tools() -> None:
    for name in ("taskset", "sudo", "unshare", "timeout"):
        if subprocess.run(["sh", "-c", f"command -v {name}"], capture_output=True).returncode != 0:
            raise SystemExit(f"required command {name} is missing")


def run_one(
    *,
    system: str,
    seed: int,
    binary: Path,
    revision: str,
    cap: str,
    log_dir: Path,
    label: str,
) -> dict[str, str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{label}.stdout"
    stderr_path = log_dir / f"{label}.stderr"
    home = log_dir / f"{label}.home"
    home.mkdir(parents=True, exist_ok=True)
    exports = " ".join(f"export {key}={shlex.quote(value)};" for key, value in THREAD_ENV.items())
    user_site = site.getusersitepackages()
    pythonpath = os.pathsep.join(item for item in (user_site, os.environ.get("PYTHONPATH", "")) if item)
    inner = " ".join(
        [
            "allowed=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status);",
            'if [ "$allowed" != 0 ]; then echo "affinity $allowed" >&2; exit 3; fi;',
            'if [ "$(nproc)" != 1 ]; then echo "nproc $(nproc)" >&2; exit 3; fi;',
            "hostname bench;",
            f"export HOME={shlex.quote(str(home))};",
            f"export PYTHONPATH={shlex.quote(pythonpath)};",
            f"export NUMBA_CACHE_DIR={shlex.quote(str(home))};",
            exports,
            "export LJOS_SEAT=benchbot PYTHONUNBUFFERED=1;",
            f"export ASE_NONLJ={shlex.quote(str(binary))};",
            f"export ASE_NONLJ_WALL_CAP={shlex.quote(cap)};",
            f"export ASE_NONLJ_REVISION={shlex.quote(revision)};",
            "exec taskset -c 0",
            shlex.quote(sys.executable),
            shlex.quote(str(RUN_ONE)),
            system,
            "recommended",
            str(seed),
        ]
    )
    timeout_s = str(int(float(cap) + 90))
    command = [
        "taskset",
        "-c",
        "0",
        "timeout",
        "-k",
        "20",
        timeout_s,
        "sudo",
        "-n",
        "unshare",
        "--uts",
        "sh",
        "-c",
        inner,
    ]
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env["LJOS_SEAT"] = "benchbot"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"{system} recommended seed {seed} exited {proc.returncode}")
    row_line = next((line for line in proc.stdout.splitlines() if line.startswith(system + ",")), "")
    fields = row_line.split(",")
    if len(fields) != 15 or fields[1] != "recommended" or fields[2] != str(seed):
        raise SystemExit(f"{system} seed {seed} did not emit a harness row")
    return {
        "budget": fields[3],
        "hit": fields[4],
        "evals_to_hit": fields[5],
        "charged": fields[6],
        "best": fields[7],
        "wall_s": fields[8],
        "inside_s": fields[9],
        "outside_s": fields[10],
        "stop": fields[13],
        "cap_s": fields[14],
    }


def abba(left: str, right: str, offset: int) -> tuple[str, str]:
    order = (left, right) if offset % 2 == 0 else (right, left)
    return order, ",".join(order)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--hosts", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--a-bin", type=Path, required=True)
    parser.add_argument("--a-rev", required=True)
    parser.add_argument("--a-tree", required=True)
    parser.add_argument("--a-worktree", type=Path, required=True)
    parser.add_argument("--b-bin", type=Path, required=True)
    parser.add_argument("--b-rev", required=True)
    parser.add_argument("--b-tree", required=True)
    parser.add_argument("--b-worktree", type=Path, required=True)
    parser.add_argument("--b2-bin", type=Path, required=True)
    parser.add_argument("--b2-worktree", type=Path, required=True)
    parser.add_argument("--layout-bin", action="append", default=[], metavar="KIND=PATH")
    parser.add_argument("--layout-worktree", action="append", default=[], metavar="KIND=PATH")
    parser.add_argument("--bands", default="AB,AA,PL")
    args = parser.parse_args(argv)

    if os.environ.get("LJOS_SEAT") != "benchbot":
        raise SystemExit("LJOS_SEAT must be benchbot")
    require_tools()
    source = (ROOT / "scripts" / "ase_nonlj" / "run_one.py").read_text()
    needles = (
        '"m38r6": {"n": 38, "rho": 6.0, "budget": 400_000, "reference": -157.477108}',
        '"m38r14": {"n": 38, "rho": 14.0, "budget": 400_000, "reference": -144.321054}',
        '"m55r6": {"n": 55, "rho": 6.0, "budget": 1_000_000, "reference": -250.286609}',
        '"m55r14": {"n": 55, "rho": 14.0, "budget": 1_000_000, "reference": -220.646208}',
    )
    for needle in needles:
        if needle not in source:
            raise SystemExit("scripts/ase_nonlj/run_one.py system table changed")
    caps = wall_caps(PANEL_ROWS, CAP_REVISION)
    expected = {
        "m38r6": "10.465089",
        "m38r14": "11.142690",
        "m55r6": "15.803787",
        "m55r14": "54.130152",
    }
    if caps != expected:
        raise SystemExit(f"wall caps {caps} differ from the panel rows")

    layouts: dict[str, Path] = {}
    for item in args.layout_bin:
        kind, path = item.split("=", 1)
        layouts[kind] = Path(path)
    layout_trees: dict[str, Path] = {}
    for item in args.layout_worktree:
        kind, path = item.split("=", 1)
        layout_trees[kind] = Path(path)
    if set(layouts) != {"end64", "start256", "align1024"}:
        raise SystemExit("placement needs end64, start256, and align1024 binaries")

    arms = {
        "A": {"bin": args.a_bin, "rev": args.a_rev, "tree": args.a_tree, "layout": "", "work": args.a_worktree, "requested": REQUESTED_A},
        "B": {"bin": args.b_bin, "rev": args.b_rev, "tree": args.b_tree, "layout": "", "work": args.b_worktree, "requested": args.b_rev},
        "B2": {"bin": args.b2_bin, "rev": args.b_rev, "tree": args.b_tree, "layout": "", "work": args.b2_worktree, "requested": args.b_rev},
    }
    for kind, binary in layouts.items():
        arms[kind] = {
            "bin": binary,
            "rev": args.b_rev,
            "tree": args.b_tree,
            "layout": kind,
            "work": layout_trees[kind],
            "requested": args.b_rev,
        }
    for spec in arms.values():
        if not Path(spec["bin"]).is_file():
            raise SystemExit("a release binary is missing")
        spec["sha"] = sha256_file(Path(spec["bin"]))
        spec["porcelain"] = git_porcelain(Path(spec["work"]))

    mid = machine_id()
    seat = "benchbot"
    compiler = compiler_line()
    threads = ",".join(f"{key}={value}" for key, value in THREAD_ENV.items())
    host_rows = []
    for role, spec in arms.items():
        host_rows.append(
            {
                "role": role,
                "requested_revision": spec["requested"],
                "code_revision": spec["rev"],
                "tree": spec["tree"],
                "binary_sha256": spec["sha"],
                "layout": spec["layout"],
                "porcelain": spec["porcelain"],
                "machine_id": mid,
                "seat": seat,
                "logical_cpus": logical_cpus(),
                "pinned_core": "0",
                "compiler": compiler,
                "thread_env": threads,
                "cap_revision": CAP_REVISION,
                "cap_m38r6": caps["m38r6"],
                "cap_m38r14": caps["m38r14"],
                "cap_m55r6": caps["m55r6"],
                "cap_m55r14": caps["m55r14"],
            }
        )
    write_csv(args.hosts, HOST_FIELDS, host_rows)

    done = {
        (row["band"], row["system"], row["arm"], row["seed"], row["binary_sha256"])
        for row in read_csv(args.rows)
    }
    existing = read_csv(args.rows)
    seq = 1 + max((int(row["run_seq"]) for row in existing), default=0)
    bands = [band.strip() for band in args.bands.split(",") if band.strip()]

    def execute(band: str, system: str, arm: str, seed: int, order: str) -> None:
        nonlocal seq
        spec = arms[arm]
        key = (band, system, arm, str(seed), spec["sha"])
        if key in done:
            return
        parsed = run_one(
            system=system,
            seed=seed,
            binary=Path(spec["bin"]),
            revision=spec["rev"],
            cap=caps[system],
            log_dir=args.logs / band / system,
            label=f"{seed}-{arm}",
        )
        if parsed["budget"] != str(BUDGETS[system]) or parsed["cap_s"] != caps[system]:
            raise SystemExit(f"{system} seed {seed} budget or cap does not match the panel")
        row = {
            "band": band,
            "system": system,
            "arm": arm,
            "seed": str(seed),
            "arm_order": order,
            "run_seq": str(seq),
            "budget": parsed["budget"],
            "cap_s": parsed["cap_s"],
            "hit": parsed["hit"],
            "evals_to_hit": parsed["evals_to_hit"],
            "charged": parsed["charged"],
            "best": parsed["best"],
            "wall_s": parsed["wall_s"],
            "inside_s": parsed["inside_s"],
            "outside_s": parsed["outside_s"],
            "stop": parsed["stop"],
            "code_revision": spec["rev"],
            "tree": spec["tree"],
            "binary_sha256": spec["sha"],
            "machine_id": mid,
            "seat": seat,
            "layout": spec["layout"],
        }
        append_csv(args.rows, ROW_FIELDS, row)
        done.add(key)
        print(
            f"{band} {system} seed {seed} {order} arm {arm} hit {row['hit']} "
            f"evals {row['evals_to_hit']} best {row['best']} wall {row['wall_s']}",
            flush=True,
        )
        seq += 1

    if "AB" in bands:
        for system in SYSTEMS:
            for offset, seed in enumerate(SEEDS):
                order, tag = abba("A", "B", offset)
                for arm in order:
                    execute("AB", system, arm, seed, tag)
    if "AA" in bands:
        for system in AA_SYSTEMS:
            for offset, seed in enumerate(SEEDS):
                order, tag = abba("B", "B2", offset)
                for arm in order:
                    execute("AA", system, arm, seed, tag)
    if "PL" in bands:
        kinds = ("end64", "start256", "align1024")
        for system in AA_SYSTEMS:
            for offset, seed in enumerate(SEEDS):
                rotated = kinds[offset % 3 :] + kinds[: offset % 3]
                tag = ",".join(rotated)
                for arm in rotated:
                    execute("PL", system, arm, seed, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
