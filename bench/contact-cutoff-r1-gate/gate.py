#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Paired gate of contact-cutoff-r1 against develop 61fa67cf.

Three arms. DEVELOP is develop 61fa67cf. CC is develop with the
contact-cutoff-r1 patch applied. DEVELOP2 is a second, separate release
build of develop, the A/A band. Each arm is a source checkout and a
directory of release binaries; the checkout's tree must be the one named
in ``TREES``, and it is read again after the last run.

Part 1 is bit identity at length one: LJ38, LJ55 and LJ75, seeds
2000-2002, arms DEVELOP and CC, through ``h2h_gmin_proof`` (the panel
driver) and ``h2h_seed`` (the seed driver, which writes the trace).

Part 2 is every non-Lennard-Jones campaign cell listed in ``PART2``,
seeds 2000-2011, all three arms.

Every budget is the paper budget: a driver's built-in table, or the
runner's ``PAPER_BUDGETS`` passed on the command line where the driver
takes it as an argument. No ``H2H_*`` or ``ANNEAL_*`` variable reaches a
run. Every thread variable is 1 and every run is ``taskset -c 0``. Runs are
serial. Within a seed the arms run in order on even seed positions and in
reverse on odd ones (ABBA). There is no warm-up and no shortened run.

A hit is a finite best at or below the reference plus 1e-3. ``calls`` is
calls to the hit, or the budget on a miss. ``best`` is the driver's printed
value as printed; ``best_bits`` is the IEEE-754 bits of that value.

Rows are appended to ``rows.csv`` after every run. A row already present
for the same part, cell, arm, seed, tree and harness blob is kept and its
run skipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.csv"
SESSION = HERE / "session.json"

TREES = {
    "DEVELOP": "cff796c507a8fa1d0bfb28888c3570b61988055e",
    "CC": "c5b961b7e51e83ae54d3f748f166ed74f0d5a432",
    "DEVELOP2": "cff796c507a8fa1d0bfb28888c3570b61988055e",
}
PART1_ARMS = ("DEVELOP", "CC")
PART2_ARMS = ("DEVELOP", "CC", "DEVELOP2")
PART1_SEEDS = (2000, 2001, 2002)
PART2_SEEDS = tuple(range(2000, 2012))
HIT_TOL = 1.0e-3
CORE = "0"
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
    "KMP_NUM_THREADS",
)
BINARIES = ("h2h_gmin_proof", "h2h_seed", "h2h_sc", "h2h_metal", "h2h_morse", "ase_nonlj", "ase_panel")

# Cambridge Cluster Database Lennard-Jones minima and the panel's paper budgets.
PART1 = (
    {"cell": "lj38/h2h_gmin_proof", "system": "lj38", "driver": "h2h_gmin_proof", "n": 38, "budget": 400_000, "reference": -173.928427},
    {"cell": "lj55/h2h_gmin_proof", "system": "lj55", "driver": "h2h_gmin_proof", "n": 55, "budget": 1_000_000, "reference": -279.248470},
    {"cell": "lj75/h2h_gmin_proof", "system": "lj75", "driver": "h2h_gmin_proof", "n": 75, "budget": 4_000_000, "reference": -397.492331},
    {"cell": "lj38/h2h_seed", "system": "lj38", "driver": "h2h_seed", "n": 38, "budget": 400_000, "reference": -173.928427},
    {"cell": "lj55/h2h_seed", "system": "lj55", "driver": "h2h_seed", "n": 55, "budget": 1_000_000, "reference": -279.248470},
    {"cell": "lj75/h2h_seed", "system": "lj75", "driver": "h2h_seed", "n": 75, "budget": 4_000_000, "reference": -397.492331},
)

# ``length`` is the Config::length_scale the driver's search runs at.
# Sutton-Chen: examples/h2h_sc.rs and scripts/metal_h2h/run_sc.py.
# Au55 and Morse on the GMIN panel: scripts/gmin_panel/run_panel.py.
# Morse and Au55 on the ASE panels: scripts/ase_nonlj/run_one.py and
# scripts/ase_panel/run_one.py, which pass the budget on the command line.
PART2 = (
    {"cell": "sc126-38", "system": "sc126-38", "driver": "h2h_sc", "budget": 400_000, "reference": -35419.9804, "length": "1"},
    {"cell": "sc126-55", "system": "sc126-55", "driver": "h2h_sc", "budget": 1_000_000, "reference": -52884.6806, "length": "1"},
    {"cell": "sc126-80", "system": "sc126-80", "driver": "h2h_sc", "budget": 1_000_000, "reference": -78414.6271, "length": "1"},
    {"cell": "sc108-38", "system": "sc108-38", "driver": "h2h_sc", "budget": 400_000, "reference": -10117.2454, "length": "1"},
    {"cell": "sc108-55", "system": "sc108-55", "driver": "h2h_sc", "budget": 1_000_000, "reference": -14814.5225, "length": "1"},
    {"cell": "sc108-80", "system": "sc108-80", "driver": "h2h_sc", "budget": 1_000_000, "reference": -21819.9952, "length": "1"},
    {"cell": "sc96-38", "system": "sc96-38", "driver": "h2h_sc", "budget": 400_000, "reference": -8917.7056, "length": "1"},
    {"cell": "sc96-55", "system": "sc96-55", "driver": "h2h_sc", "budget": 1_000_000, "reference": -13217.8963, "length": "1"},
    {"cell": "sc96-80", "system": "sc96-80", "driver": "h2h_sc", "budget": 1_000_000, "reference": -19560.8966, "length": "1"},
    {"cell": "au55/h2h_metal", "system": "au55", "driver": "h2h_metal", "budget": 1_000_000, "reference": -191.03085280, "length": "1"},
    {"cell": "m55r6/h2h_morse", "system": "m55r6", "driver": "h2h_morse", "budget": 1_000_000, "reference": -250.286609, "length": "1"},
    {"cell": "m55r14/h2h_morse", "system": "m55r14", "driver": "h2h_morse", "budget": 1_000_000, "reference": -220.646208, "length": "1"},
    {"cell": "m80r6/h2h_morse", "system": "m80r6", "driver": "h2h_morse", "budget": 4_000_000, "reference": -378.333471, "length": "1"},
    {"cell": "m80r14/h2h_morse", "system": "m80r14", "driver": "h2h_morse", "budget": 4_000_000, "reference": -340.811371, "length": "1"},
    {"cell": "m38r6/ase_nonlj/recommended", "system": "m38r6", "driver": "ase_nonlj", "preset": "recommended", "budget": 400_000, "reference": -157.477108, "length": "2^(-1/6)"},
    {"cell": "m38r14/ase_nonlj/recommended", "system": "m38r14", "driver": "ase_nonlj", "preset": "recommended", "budget": 400_000, "reference": -144.321054, "length": "2^(-1/6)"},
    {"cell": "m55r6/ase_nonlj/recommended", "system": "m55r6", "driver": "ase_nonlj", "preset": "recommended", "budget": 1_000_000, "reference": -250.286609, "length": "2^(-1/6)"},
    {"cell": "m55r14/ase_nonlj/recommended", "system": "m55r14", "driver": "ase_nonlj", "preset": "recommended", "budget": 1_000_000, "reference": -220.646208, "length": "2^(-1/6)"},
    {"cell": "m38r6/ase_nonlj/for_cluster", "system": "m38r6", "driver": "ase_nonlj", "preset": "for_cluster", "budget": 400_000, "reference": -157.477108, "length": "2^(-1/6)"},
    {"cell": "m38r14/ase_nonlj/for_cluster", "system": "m38r14", "driver": "ase_nonlj", "preset": "for_cluster", "budget": 400_000, "reference": -144.321054, "length": "2^(-1/6)"},
    {"cell": "m55r6/ase_nonlj/for_cluster", "system": "m55r6", "driver": "ase_nonlj", "preset": "for_cluster", "budget": 1_000_000, "reference": -250.286609, "length": "2^(-1/6)"},
    {"cell": "m55r14/ase_nonlj/for_cluster", "system": "m55r14", "driver": "ase_nonlj", "preset": "for_cluster", "budget": 1_000_000, "reference": -220.646208, "length": "2^(-1/6)"},
    {"cell": "au55/ase_panel/recommended", "system": "au55", "driver": "ase_panel", "preset": "recommended", "budget": 1_000_000, "reference": -191.03085280, "length": "Gupta dimer"},
    {"cell": "au55/ase_panel/for_cluster", "system": "au55", "driver": "ase_panel", "preset": "for_cluster", "budget": 1_000_000, "reference": -191.03085280, "length": "Gupta dimer"},
)

ROW_FIELDS = [
    "part",
    "cell",
    "system",
    "driver",
    "preset",
    "arm",
    "seed",
    "slot",
    "core",
    "hit",
    "driver_hit",
    "calls",
    "charged",
    "wall",
    "inside_s",
    "outside_s",
    "best",
    "best_bits",
    "budget",
    "reference",
    "length",
    "trace_sha",
    "machine_id",
    "library_tree",
    "harness_blob",
    "session",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def machine_id() -> str:
    return hashlib.sha256(Path("/etc/machine-id").read_text().strip().encode()).hexdigest()


def tree_of(source: Path) -> str:
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise SystemExit(f"{source} has local changes")
    return subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def is_hit(best: float, reference: float) -> bool:
    return math.isfinite(best) and best <= reference + HIT_TOL


def child_env(rgpot: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith(("H2H_", "ANNEAL_"))}
    for key in THREAD_VARS:
        env[key] = "1"
    env["RGPOT_SC_LIB"] = str(rgpot)
    return env


def arm_order(arms: tuple[str, ...], position: int) -> tuple[str, ...]:
    return arms if position % 2 == 0 else tuple(reversed(arms))


def csv_row(text: str, system: str, seed: int) -> list[str]:
    data = [line for line in text.splitlines() if line.startswith(f"{system},anneal,{seed},")]
    if len(data) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one result row\n{text[-2000:]}")
    return data[0].split(",")


def parse_panel_csv(text: str, cell: dict, seed: int) -> dict[str, str]:
    # system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,... with
    # pot_s and outside_s named in the header.
    fields = csv_row(text, cell["system"], seed)
    headers = [line for line in text.splitlines() if line.startswith("system,code,")]
    if len(headers) != 1:
        raise RuntimeError(f"{cell['cell']} seed {seed}: expected one header")
    names = headers[0].split(",")
    if cell["driver"] == "h2h_gmin_proof":
        # head,tree,cpu,pins sit between best and pot_s; cpu may hold commas.
        row = dict(zip(names[:9], fields[:9]))
        row["pot_s"] = fields[-1]
    else:
        if len(names) != len(fields):
            raise RuntimeError(f"{cell['cell']} seed {seed}: header and row differ")
        row = dict(zip(names, fields))
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    outside = row["outside_s"] if "outside_s" in row else f"{max(wall - pot, 0.0):.6f}"
    return {
        "budget": row["budget"],
        "driver_hit": "1" if row["hit"] in {"1", "true", "True"} else "0",
        "calls": row["evals_to_hit"],
        "charged": row["charged"],
        "wall": row["wall_s"],
        "inside_s": row["pot_s"],
        "outside_s": outside,
        "best": row["best"],
    }


def parse_seed(text: str, cell: dict) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(": ")
        if sep and key not in fields:
            fields[key] = value.strip()
    return {
        "budget": fields["budget"],
        "driver_hit": "",
        "calls": fields["calls"],
        "charged": fields["charged"],
        "wall": fields["wall_seconds"],
        "inside_s": fields["pot_s"],
        "outside_s": fields["outside_s"],
        "best": fields["best"],
    }


def parse_kv(text: str, cell: dict, seed: int) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith(f"system={cell['system']} ")]
    if len(lines) != 1:
        raise RuntimeError(f"{cell['cell']} seed {seed}: expected one result line\n{text[-2000:]}")
    fields = dict(part.split("=", 1) for part in lines[0].split() if "=" in part)
    if fields["arm"] != cell["preset"] or fields["seed"] != str(seed):
        raise RuntimeError(f"{cell['cell']} seed {seed}: line {lines[0]}")
    hit = fields["hit"] == "1"
    return {
        "budget": fields["budget"],
        "driver_hit": "1" if hit else "0",
        "calls": fields["first_calls"] if hit else fields["budget"],
        "charged": fields["charged"],
        "wall": fields["wall_s"],
        "inside_s": fields["inside_s"],
        "outside_s": fields["outside_s"],
        "best": fields["best"],
    }


def command_of(cell: dict, seed: int, binary: Path, prefix: Path) -> list[str]:
    driver = cell["driver"]
    if driver in ("h2h_gmin_proof", "h2h_sc", "h2h_metal", "h2h_morse"):
        return [str(binary), "run", cell["system"], str(seed)]
    if driver == "h2h_seed":
        stop = repr(cell["reference"] + HIT_TOL)
        return [str(binary), "lj", str(cell["n"]), "0", str(cell["budget"]), str(seed), "recommended", stop, str(prefix)]
    if driver == "ase_nonlj":
        return [str(binary), "run", cell["system"], str(cell["budget"]), str(seed), cell["preset"]]
    if driver == "ase_panel":
        return [str(binary), cell["system"], str(cell["budget"]), str(seed), cell["preset"]]
    raise SystemExit(f"unknown driver {driver}")


def one_run(part: str, cell: dict, arm: str, seed: int, slot: int, ctx: dict) -> dict[str, str]:
    source, bin_dir = ctx["arms"][arm]
    binary = bin_dir / cell["driver"]
    stem = f"{part}-{cell['cell'].replace('/', '_')}-{arm}-{seed}"
    log = ctx["logs"] / f"{stem}.log"
    prefix = ctx["logs"] / "traces" / stem
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = ["taskset", "-c", CORE] + command_of(cell, seed, binary, prefix)
    started = time.time()
    proc = subprocess.run(command, cwd=source, env=ctx["env"], capture_output=True, text=True, check=False)
    log.write_text(
        f"# {' '.join(command)}\n# start {started:.3f} end {time.time():.3f} exit {proc.returncode}\n"
        + proc.stdout
        + "\n# stderr\n"
        + proc.stderr
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{stem} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    if cell["driver"] == "h2h_seed":
        parsed = parse_seed(proc.stdout, cell)
        trace_sha = sha256_file(Path(str(prefix) + ".trace.csv"))
    elif cell["driver"] in ("ase_nonlj", "ase_panel"):
        parsed = parse_kv(proc.stdout, cell, seed)
        trace_sha = ""
    else:
        parsed = parse_panel_csv(proc.stdout, cell, seed)
        trace_sha = ""
    if int(parsed["budget"]) != cell["budget"]:
        raise RuntimeError(f"{stem}: budget {parsed['budget']} is not the paper budget {cell['budget']}")
    best = float(parsed["best"])
    return {
        "part": part,
        "cell": cell["cell"],
        "system": cell["system"],
        "driver": cell["driver"],
        "preset": cell.get("preset", "recommended"),
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "core": CORE,
        "hit": "1" if is_hit(best, cell["reference"]) else "0",
        "driver_hit": parsed["driver_hit"],
        "calls": parsed["calls"],
        "charged": parsed["charged"],
        "wall": parsed["wall"],
        "inside_s": parsed["inside_s"],
        "outside_s": parsed["outside_s"],
        "best": parsed["best"],
        "best_bits": bits(best),
        "budget": str(cell["budget"]),
        "reference": repr(cell["reference"]),
        "length": cell.get("length", "1"),
        "trace_sha": trace_sha,
        "machine_id": ctx["machine"],
        "library_tree": ctx["trees"][arm],
        "harness_blob": ctx["blob"],
        "session": ctx["session"],
    }


def load_rows() -> list[dict[str, str]]:
    if not ROWS.is_file():
        return []
    with ROWS.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(row: dict[str, str]) -> None:
    fresh = not ROWS.is_file()
    with ROWS.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def schedule(parts: list[str]) -> list[tuple[str, dict, str, int, int]]:
    jobs = []
    for part in parts:
        cells, seeds, arms = (PART1, PART1_SEEDS, PART1_ARMS) if part == "1" else (PART2, PART2_SEEDS, PART2_ARMS)
        for cell in cells:
            for position, seed in enumerate(seeds):
                for slot, arm in enumerate(arm_order(arms, position), start=1):
                    jobs.append((part, cell, arm, seed, slot))
    return jobs


def check_trees(arms: dict[str, tuple[Path, Path]]) -> dict[str, str]:
    trees = {}
    for arm, (source, _bin_dir) in arms.items():
        tree = tree_of(source)
        if tree != TREES[arm]:
            raise SystemExit(f"{arm} tree {tree} is not {TREES[arm]}")
        trees[arm] = tree
    return trees


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=SOURCE:BINDIR")
    parser.add_argument("--rgpot", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--part", action="append", choices=("1", "2"), required=True)
    args = parser.parse_args(argv)
    if os.sched_getaffinity(0) != {int(CORE)}:
        raise SystemExit(f"the harness is not pinned to core {CORE}")
    arms: dict[str, tuple[Path, Path]] = {}
    for spec in args.arm:
        name, _, paths = spec.partition("=")
        source, _, bin_dir = paths.partition(":")
        arms[name] = (Path(source), Path(bin_dir))
    if set(arms) != set(TREES):
        raise SystemExit(f"arms must be {sorted(TREES)}")
    trees = check_trees(arms)
    blob = git_blob(Path(__file__))
    machine = machine_id()
    binaries = {
        arm: {name: sha256_file(bin_dir / name) for name in BINARIES} for arm, (_source, bin_dir) in arms.items()
    }
    session = hashlib.sha256(json.dumps([trees, binaries, blob, machine, args.part], sort_keys=True).encode()).hexdigest()[:16]
    record = {
        "session": session,
        "parts": args.part,
        "trees": trees,
        "binary_sha256": binaries,
        "rgpot_sha256": sha256_file(args.rgpot),
        "harness_blob": blob,
        "machine_id": machine,
        "core": CORE,
        "thread_vars": list(THREAD_VARS),
        "hit_tol": HIT_TOL,
    }
    sessions = json.loads(SESSION.read_text()) if SESSION.is_file() else []
    if all(item.get("session") != session for item in sessions):
        sessions.append(record)
        SESSION.write_text(json.dumps(sessions, indent=1, sort_keys=True) + "\n")
    args.logs.mkdir(parents=True, exist_ok=True)
    ctx = {
        "arms": arms,
        "trees": trees,
        "blob": blob,
        "machine": machine,
        "session": session,
        "logs": args.logs,
        "env": child_env(args.rgpot),
    }
    done = {
        (row["part"], row["cell"], row["arm"], row["seed"], row["library_tree"], row["harness_blob"])
        for row in load_rows()
    }
    jobs = schedule(args.part)
    for index, (part, cell, arm, seed, slot) in enumerate(jobs, start=1):
        key = (part, cell["cell"], arm, str(seed), trees[arm], blob)
        if key in done:
            continue
        row = one_run(part, cell, arm, seed, slot, ctx)
        append_row(row)
        print(
            f"[{index}/{len(jobs)}] part {part} {cell['cell']} {arm} seed {seed} "
            f"hit={row['hit']} calls={row['calls']} best={row['best']} wall={row['wall']}",
            flush=True,
        )
    if check_trees(arms) != trees:
        raise SystemExit("a tree changed during the session")
    print("finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
