#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Outside time per charged call.

One seed at a time, pinned to core 0, with one BLAS thread and one
Rayon thread. The process refuses to run unless the UTS hostname is
bench and the working directory is /work.

Seeds 2000–2011. Au98 and Au55 use the Gupta budgets in the metal
example. sc96-80 uses 1_000_000. LJ75 uses the 4_000_000 paper budget
and stops at the database minimum plus the 0.001 hit tolerance. Morse
80 at rho 3 uses the budget in the Morse example.

A row is appended and never rewritten. The key is the system, the seed
and the code revision. `outside_us` is the time outside the potential
divided by the charged calls, in microseconds.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

for _candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (_candidate / "target_tol.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
from target_tol import DEFAULT_HIT_TOL  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROWS = ROOT / "bench" / "outside-time" / "rows.csv"
FIELDS = [
    "system",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
    "outside_us",
    "code_revision",
    "host",
]
SEEDS = list(range(2000, 2012))
LJ75_STOP = -397.492331 + DEFAULT_HIT_TOL
LJ150_STOP = -893.310258 + DEFAULT_HIT_TOL
TIP4P20_STOP = -872.9887544952 + DEFAULT_HIT_TOL


def revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def have(rows: list[dict[str, str]], system: str, seed: int, rev: str) -> bool:
    return any(
        row["system"] == system
        and row["seed"] == str(seed)
        and row["code_revision"] == rev
        for row in rows
    )


def run(cmd: list[str], env: dict[str, str]) -> str:
    completed = subprocess.run(
        ["taskset", "-c", "0", *cmd],
        cwd="/work",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        sys.stderr.write(completed.stdout)
        raise SystemExit(f"{cmd[0]} exited {completed.returncode}")
    return completed.stdout


def child_env(sc_lib: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    lib_dir = str(Path(sc_lib).resolve().parent)
    env["RGPOT_SC_LIB"] = sc_lib
    env["LD_LIBRARY_PATH"] = lib_dir + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    return env


def parse_csv_line(text: str, system: str) -> dict[str, str]:
    for line in reversed(text.splitlines()):
        if line.startswith(system + ","):
            header = [
                "system",
                "code",
                "seed",
                "budget",
                "hit",
                "evals_to_hit",
                "charged",
                "wall_s",
                "best",
                "pot_s",
                "outside_s",
            ]
            values = line.split(",")
            return dict(zip(header, values, strict=False))
    raise SystemExit(f"no row for {system}")


def parse_seed(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("h2h_row "):
            for token in line.split()[1:]:
                key, value = token.split("=", 1)
                found[key] = value
    if "outside_s" not in found:
        raise SystemExit("h2h_seed printed no row")
    found["hit"] = "true" if found.get("hit") == "1" else "false"
    return found


def one_row(
    system: str,
    seed: int,
    parsed: dict[str, str],
    rev: str,
    host: str,
) -> dict[str, str]:
    charged = int(parsed["charged"])
    outside_s = float(parsed["outside_s"])
    return {
        "system": system,
        "seed": str(seed),
        "budget": parsed["budget"],
        "hit": parsed["hit"],
        "evals_to_hit": parsed["evals_to_hit"],
        "charged": str(charged),
        "wall_s": parsed["wall_s"],
        "best": parsed["best"],
        "pot_s": parsed["pot_s"],
        "outside_s": f"{outside_s:.6f}",
        "outside_us": f"{outside_s / charged * 1.0e6:.3f}",
        "code_revision": rev,
        "host": host,
    }


def jobs(seed: int, bindir: Path, scratch: Path) -> list[tuple[str, list[str]]]:
    prefix = scratch / f"lj75-{seed}"
    return [
        ("au98", [str(bindir / "h2h_metal"), "run", "au98", str(seed)]),
        ("au55", [str(bindir / "h2h_metal"), "run", "au55", str(seed)]),
        ("sc96-80", [str(bindir / "h2h_sc"), "run", "sc96-80", str(seed)]),
        (
            "lj75",
            [
                str(bindir / "h2h_seed"),
                "lj",
                "75",
                "0",
                "4000000",
                str(seed),
                "recommended",
                f"{LJ75_STOP:.6f}",
                str(prefix),
            ],
        ),
        ("m80r3", [str(bindir / "h2h_morse"), "run", "m80r3", str(seed)]),
        ("m80r6", [str(bindir / "h2h_morse"), "run", "m80r6", str(seed)]),
        (
            "lj150",
            [
                str(bindir / "h2h_seed"),
                "lj",
                "150",
                "0",
                "4000000",
                str(seed),
                "recommended",
                f"{LJ150_STOP:.6f}",
                str(scratch / f"lj150-{seed}"),
            ],
        ),
        (
            "tip4p20",
            [
                str(bindir / "h2h_seed"),
                "tip4p",
                "20",
                "0",
                "2000000",
                str(seed),
                "recommended",
                f"{TIP4P20_STOP:.10f}",
                str(scratch / f"tip4p20-{seed}"),
            ],
        ),
    ]


def main() -> None:
    if os.uname().nodename != "bench":
        raise SystemExit("uts hostname is not bench")
    if Path.cwd() != Path("/work"):
        raise SystemExit("working directory is not /work")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument(
        "--sc-lib",
        default="/work/rgpot-build/CppCore/librgpot.so",
    )
    parser.add_argument("--systems", default="au98,au55,sc96-80,lj75,m80r3")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--seeds", default="")
    args = parser.parse_args()
    wanted = set(args.systems.split(","))
    seeds = [int(item) for item in args.seeds.split(",")] if args.seeds else SEEDS
    rev = revision()
    host = os.uname().nodename
    env = child_env(args.sc_lib)
    scratch = Path("/work/outside-time-traces")
    scratch.mkdir(parents=True, exist_ok=True)
    rows_path = args.rows
    rows = load_rows(rows_path)
    for seed in seeds:
        for system, cmd in jobs(seed, args.bin_dir, scratch):
            if system not in wanted:
                continue
            if have(rows, system, seed, rev):
                print(f"skip {system} {seed} {rev[:12]}", flush=True)
                continue
            text = run(cmd, env)
            parsed = (
                parse_seed(text)
                if system.startswith("lj") or system.startswith("tip4p")
                else parse_csv_line(text, system)
            )
            row = one_row(system, seed, parsed, rev, host)
            append_row(rows_path, row)
            rows.append(row)
            print(
                f"{system} seed {seed} outside {row['outside_us']} us/call "
                f"charged {row['charged']} best {row['best']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
