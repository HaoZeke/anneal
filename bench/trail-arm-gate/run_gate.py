#!/usr/bin/env python3
"""Run the serial three-arm trail-arm gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = tuple(range(2000, 2012))
ARMS = ("A", "A2", "B")
ORDERS = (
    ("A", "B", "A2"),
    ("A2", "B", "A"),
    ("B", "A2", "A"),
    ("A", "A2", "B"),
    ("A2", "A", "B"),
    ("B", "A", "A2"),
)
HEADS = {
    "A": "364679f060386949f805a72dea9d7d63b7e9ea4f",
    "A2": "364679f060386949f805a72dea9d7d63b7e9ea4f",
    "B": "6d37bb671d2072636aa612145e7066235281e748",
}
CANONICAL_TREES = {
    "A": "79610c980d39f19a2d383ffe3c53e4505ddb9b6c",
    "A2": "79610c980d39f19a2d383ffe3c53e4505ddb9b6c",
    "B": "34c4de6ad5d38a06b30f0c59cd2baae36c1fa613",
}
MEASURED_TREES = {
    "A": "79610c980d39f19a2d383ffe3c53e4505ddb9b6c",
    "A2": "79610c980d39f19a2d383ffe3c53e4505ddb9b6c",
    "B": "baaa2f0e996c3839d47b7833ea14ef3481eaaa86",
}
TIP4P_SIZES = (*range(9, 22), 25)
TIP4P_BUDGET = 2_000_000
HIT_TOL = 1.0e-3
PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj102": 4_000_000,
    "lj103": 4_000_000,
    "lj104": 4_000_000,
    "lj150": 4_000_000,
    "lj75-seed": 4_000_000,
    "m55r6": 1_000_000,
    "m55r14": 1_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
    "au55-seed": 1_000_000,
}
REFERENCES = {
    "lj38": -173.928427,
    "lj55": -279.248470,
    "lj75": -397.492331,
    "lj98": -543.665361,
    "lj102": -569.363652,
    "lj103": -575.766131,
    "lj104": -582.086642,
    "lj150": -893.310258,
    "lj75-seed": -397.492331,
    "m55r6": -250.286609,
    "m55r14": -220.646208,
    "m80r6": -378.333471,
    "m80r14": -340.811371,
    "au55-seed": -191.03085280,
}
DRIVERS = ("h2h_seed", "h2h_gmin_proof", "h2h_morse", "h2h_metal")
THREAD_VARS = (
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
ROW_FIELDS = (
    "cell",
    "arm",
    "seed",
    "position",
    "slot",
    "canonical_tree",
    "measured_tree",
    "exit_code",
    "outside_s",
    "calls",
    "hit",
    "energy_bits",
    "rss_anon_kb",
    "stall_triggers",
    "trail_redraws",
    "budget",
    "reference",
    "charged",
    "wall_s",
    "inside_s",
    "energy",
    "core",
    "affinity",
    "exec_image",
    "binary_sha256",
)


def tip4p_references() -> dict[int, float]:
    with (ROOT / "data" / "tip4p_ccd.csv").open(newline="") as handle:
        refs = {
            int(row["n"]): float(row["energy"])
            for row in csv.DictReader(handle)
        }
    refs[25] = -1113.0346568571
    return refs


def cells() -> list[tuple[str, int, float]]:
    refs = tip4p_references()
    values = [
        (f"tip4p{size}", TIP4P_BUDGET, refs[size])
        for size in TIP4P_SIZES
    ]
    values += [
        (name, PAPER_BUDGETS[name], REFERENCES[name])
        for name in PAPER_BUDGETS
    ]
    assert len(values) == 28
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def energy_bits(text: str) -> str:
    return struct.pack(">d", float(text)).hex()


def driver(cell: str) -> str:
    if cell.startswith("tip4p") or cell == "lj75-seed":
        return "h2h_seed"
    if cell.startswith("lj"):
        return "h2h_gmin_proof"
    if cell == "au55-seed":
        return "h2h_metal"
    return "h2h_morse"


class Arm:
    def __init__(self, name: str, spec: str) -> None:
        binary, source = spec.split(":", 1)
        self.name = name
        self.binary = Path(binary).resolve()
        self.source = Path(source).resolve()
        self.binary_sha = {
            item: sha256_file(self.binary / item) for item in DRIVERS
        }

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.source), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def validate(self) -> None:
        if self.git("rev-parse", "HEAD") != HEADS[self.name]:
            raise SystemExit(f"{self.name} head mismatch")
        if self.git("rev-parse", "HEAD^{tree}") != MEASURED_TREES[self.name]:
            raise SystemExit(f"{self.name} tree mismatch")
        if self.git("status", "--porcelain"):
            raise SystemExit(f"{self.name} worktree is dirty")
        for item, digest in self.binary_sha.items():
            if sha256_file(self.binary / item) != digest:
                raise SystemExit(f"{self.name} binary changed: {item}")


def parse_arms(specs: list[str]) -> dict[str, Arm]:
    arms = {}
    for spec in specs:
        name, value = spec.split("=", 1)
        if name not in ARMS:
            raise SystemExit(f"unknown arm {name}")
        arms[name] = Arm(name, value)
    if set(arms) != set(ARMS):
        raise SystemExit("all three arms are required")
    return arms


def child_env(probe: Path, arm: str, budget: int) -> dict[str, str]:
    env = {
        "HOME": os.environ["HOME"],
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LD_PRELOAD": str(probe),
        "H2H_BUDGET": str(budget),
    }
    for name in THREAD_VARS:
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    if arm == "B":
        env["ANNEAL_TRAIL_ARM_GATE"] = "1"
    return env


def command(
    arm: Arm,
    cell: str,
    budget: int,
    reference: float,
    seed: int,
    prefix: Path,
    core: int,
) -> list[str]:
    pin = ["taskset", "-c", str(core)]
    if cell.startswith("tip4p"):
        size = cell.removeprefix("tip4p")
        return pin + [
            str(arm.binary / "h2h_seed"),
            "tip4p",
            size,
            "0",
            str(budget),
            str(seed),
            "recommended",
            repr(reference + HIT_TOL),
            str(prefix),
        ]
    if cell == "lj75-seed":
        return pin + [
            str(arm.binary / "h2h_seed"),
            "lj",
            "75",
            "0",
            str(budget),
            str(seed),
            "recommended",
            repr(reference + HIT_TOL),
            str(prefix),
        ]
    if cell.startswith("lj"):
        return pin + [
            str(arm.binary / "h2h_gmin_proof"),
            "run",
            cell,
            str(seed),
        ]
    if cell == "au55-seed":
        return pin + [
            str(arm.binary / "h2h_metal"),
            "run",
            "au55",
            str(seed),
        ]
    return pin + [
        str(arm.binary / "h2h_morse"),
        "run",
        cell,
        str(seed),
    ]


def parse_output(cell: str, text: str) -> dict[str, str]:
    if cell.startswith("tip4p") or cell == "lj75-seed":
        line = next(
            item for item in text.splitlines() if item.startswith("h2h_row ")
        )
        row = dict(token.split("=", 1) for token in line.split()[1:])
        calls = re.search(r"^calls: (\d+)$", text, re.M)
        if calls is None:
            raise ValueError("calls line missing")
        return {
            "calls": calls.group(1),
            "charged": row["charged"],
            "wall": row["wall_s"],
            "inside": row["pot_s"],
            "outside": row["outside_s"],
            "best": row["best"],
            "budget": row["budget"],
        }
    lines = text.splitlines()
    index = next(
        i for i, line in enumerate(lines) if line.startswith("system,code,seed,")
    )
    row = dict(
        zip(lines[index].split(","), lines[index + 1].split(","), strict=True)
    )
    wall = float(row["wall_s"])
    inside = float(row["pot_s"])
    return {
        "calls": row["charged"],
        "charged": row["charged"],
        "wall": row["wall_s"],
        "inside": row["pot_s"],
        "outside": f"{max(wall - inside, 0.0):.6f}",
        "best": row["best"],
        "budget": row["budget"],
    }


def append_row(path: Path, row: dict[str, object]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ROW_FIELDS,
            lineterminator="\n",
        )
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_process(
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, str, str, float]:
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    elapsed = time.monotonic() - started
    return (
        process.returncode,
        stdout_path.read_text(errors="replace"),
        stderr_path.read_text(errors="replace"),
        elapsed,
    )


def parse_probe(stderr: str) -> tuple[int, str, str]:
    matches = re.findall(
        r"^rss_anon_self_kb=(\d+) exec_image=(\S+) affinity=(\S+)$",
        stderr,
        re.M,
    )
    if len(matches) != 1:
        raise ValueError(f"expected one RSS line, got {len(matches)}")
    rss, image, affinity = matches[0]
    return int(rss), image, affinity


def parse_counter(arm: str, stderr: str) -> tuple[str, str]:
    matches = re.findall(
        r"^trail_arm_gate triggers=(\d+) redraws=(\d+)$",
        stderr,
        re.M,
    )
    expected = 1 if arm == "B" else 0
    if len(matches) != expected:
        raise ValueError(f"expected {expected} counter lines")
    return matches[0] if matches else ("", "")


def expected_image(argv: list[str]) -> str:
    return next(Path(item).name for item in argv if Path(item).name in DRIVERS)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args(argv[1:])
    arms = parse_arms(args.arm)
    for arm in arms.values():
        arm.validate()
    core = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {core})
    probe = args.probe.resolve()
    work = args.work.resolve()
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    rows = out / "rows.csv"
    if rows.exists():
        raise SystemExit("rows.csv already exists")
    session = out / "session.csv"
    with session.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["event", "value"])
        writer.writerow(["core", core])
        writer.writerow(["threads", 1])
        writer.writerow(["seeds", "2000-2011"])
        writer.writerow(["cells", 28])
    slot = 0
    for cell, budget, reference in cells():
        for seed_index, seed in enumerate(SEEDS):
            for position, name in enumerate(
                ORDERS[seed_index % len(ORDERS)],
                1,
            ):
                slot += 1
                arm = arms[name]
                tag = f"{cell}-s{seed}-{name.lower()}"
                prefix = logs / tag
                command_line = command(
                    arm,
                    cell,
                    budget,
                    reference,
                    seed,
                    prefix,
                    core,
                )
                stdout_path = Path(f"{prefix}.out")
                stderr_path = Path(f"{prefix}.err")
                rc, stdout, stderr, elapsed = run_process(
                    command_line,
                    logs,
                    child_env(probe, name, budget),
                    stdout_path,
                    stderr_path,
                )
                row: dict[str, object] = {
                    "cell": cell,
                    "arm": name,
                    "seed": seed,
                    "position": position,
                    "slot": slot,
                    "canonical_tree": CANONICAL_TREES[name],
                    "measured_tree": MEASURED_TREES[name],
                    "exit_code": rc,
                    "budget": budget,
                    "reference": repr(reference),
                    "core": core,
                    "binary_sha256": arm.binary_sha[driver(cell)],
                }
                try:
                    if rc != 0:
                        raise ValueError(f"exit {rc}")
                    rss, image, affinity = parse_probe(stderr)
                    if image != expected_image(command_line):
                        raise ValueError(f"unexpected image {image}")
                    if affinity != str(core):
                        raise ValueError(f"unexpected affinity {affinity}")
                    triggers, redraws = parse_counter(name, stderr)
                    parsed = parse_output(cell, stdout)
                    if int(parsed["budget"]) != budget:
                        raise ValueError("budget mismatch")
                    hit = float(parsed["best"]) <= reference + HIT_TOL
                    row.update(
                        outside_s=parsed["outside"],
                        calls=parsed["calls"],
                        hit=int(hit),
                        energy_bits=energy_bits(parsed["best"]),
                        rss_anon_kb=rss,
                        stall_triggers=triggers,
                        trail_redraws=redraws,
                        charged=parsed["charged"],
                        wall_s=parsed["wall"],
                        inside_s=parsed["inside"],
                        energy=parsed["best"],
                        affinity=affinity,
                        exec_image=image,
                    )
                except Exception as error:
                    row.update(
                        outside_s="",
                        calls="",
                        hit=0,
                        energy_bits="",
                        rss_anon_kb="",
                        stall_triggers="",
                        trail_redraws="",
                        charged="",
                        wall_s=f"{elapsed:.6f}",
                        inside_s="",
                        energy="",
                        affinity="",
                        exec_image="",
                    )
                    with session.open("a", newline="") as handle:
                        csv.writer(handle, lineterminator="\n").writerow(
                            ["failure", f"{cell}:{seed}:{name}:{type(error).__name__}"]
                        )
                append_row(rows, row)
    for arm in arms.values():
        arm.validate()
    with session.open("a", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(["rows", slot])


if __name__ == "__main__":
    main(sys.argv)
