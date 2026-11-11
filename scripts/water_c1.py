# Copyright (c) anneal developers
"""Regenerate the water hexamer c1 sample.

Logs, trajectories, and tables are written under WATER_C1_OUT.
That directory is not part of this repository. Run the script again
to reproduce them.

    python3 scripts/water_c1.py campaign --first 48 --count 48
    python3 scripts/water_c1.py equivalence --draws 6

The anneal arm is `water_hexamer gfn2 good_turing` with one seed.
The CREST arm is `crest water.xyz -T 1`. Each timed command runs
inside `sudo unshare --uts sh -c 'hostname bench; exec <cmd>'`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

BUDGET = 4000
RESERVE = 618
CAP = BUDGET - RESERVE
HIGH_CAP = "5000000"


def out_root() -> Path:
    raw = os.environ.get("WATER_C1_OUT", "/tmp/water-c1")
    path = Path(raw).resolve()
    repo = Path(__file__).resolve().parents[1]
    if path == repo or repo in path.parents:
        raise SystemExit(f"WATER_C1_OUT {path} is inside the git tree")
    path.mkdir(parents=True, exist_ok=True)
    return path


def require(name: str) -> Path:
    text = os.environ.get(name, "")
    if not text:
        raise SystemExit(f"{name} is required")
    path = Path(text)
    if not path.is_file():
        raise SystemExit(f"{name} is not a file: {path}")
    return path


def wrapper(argv: list[str], cwd: Path, extra: dict[str, str]) -> tuple[int, float]:
    cwd.mkdir(parents=True, exist_ok=True)
    exports = [
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "export RAYON_NUM_THREADS=1",
    ]
    lib = os.environ.get("LD_LIBRARY_PATH", "")
    if lib:
        exports.append(f"export LD_LIBRARY_PATH={lib}")
    for key, value in extra.items():
        exports.append(f"export {key}={value}")
    script = "hostname bench; " + "; ".join(exports) + '; exec "$@"'
    (cwd / "wrapper.txt").write_text(
        "wrapper: sudo unshare --uts sh -c 'hostname bench; exec <cmd>'\n"
        f"cmd: {' '.join(argv)}\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    completed = subprocess.run(
        ["sudo", "-n", "unshare", "--uts", "sh", "-c", script, "_", *argv],
        check=False,
        capture_output=True,
        cwd=cwd,
    )
    wall = time.perf_counter() - started
    (cwd / "stdout.txt").write_bytes(completed.stdout)
    (cwd / "stderr.txt").write_bytes(completed.stderr)
    (cwd / "wall.txt").write_text(f"{wall:.6f}\n", encoding="utf-8")
    text = completed.stdout.decode("utf-8", errors="replace")
    if "hostname=" in text and "hostname=bench" not in text:
        raise SystemExit(f"{cwd} did not record hostname bench")
    return completed.returncode, wall


def anneal_one(binary: Path, seed: int, cwd: Path, start: Path, isomers: Path) -> None:
    wrapper(
        [str(binary), "gfn2", "good_turing"],
        cwd,
        {
            "WATER_SEED_COUNT": "1",
            "WATER_SEED_FIRST": str(seed),
            "WATER_ISOMER_DIR": str(isomers),
            "WATER_START_XYZ": str(start),
            "TBLITE_BUILD": os.environ.get(
                "TBLITE_BUILD", "fa8a4416e8fe093d0075bc10ac875494c2a449a9"
            ),
        },
    )


def crest_one(binary: Path, cwd: Path, start: Path, cap: str | None) -> None:
    shutil.copyfile(start, cwd / "water.xyz")
    extra: dict[str, str] = {}
    if cap is not None:
        extra["CREST_ENGRAD_CAP"] = cap
    wrapper([str(binary), "water.xyz", "-T", "1"], cwd, extra)


def campaign(args: argparse.Namespace) -> int:
    root = out_root()
    binary = require("WATER_HEXAMER")
    start = require("WATER_START_XYZ")
    isomers = Path(os.environ["WATER_ISOMER_DIR"])
    crest = Path(os.environ["CREST_BIN"]) if os.environ.get("CREST_BIN") else None
    if args.crest and crest is None:
        raise SystemExit("CREST_BIN is required for the CREST arm")
    for seed in range(args.first, args.first + args.count):
        anneal_one(binary, seed, root / "anneal" / str(seed), start, isomers)
        if args.crest and crest is not None:
            crest_one(crest, root / "crest" / str(seed), start, str(CAP))
        print(f"seed {seed}", flush=True)
    print(f"wrote {root}", flush=True)
    return 0


def equivalence(args: argparse.Namespace) -> int:
    root = out_root() / "equivalence"
    local = require("CREST_BIN")
    forge = require("CREST_FORGE")
    start = require("WATER_START_XYZ")
    for draw in range(1, args.draws + 1):
        crest_one(local, root / f"d{draw:02d}" / "local", start, HIGH_CAP)
        crest_one(forge, root / f"d{draw:02d}" / "forge", start, None)
        print(f"draw {draw}", flush=True)
    print(f"wrote {root}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    camp = sub.add_parser("campaign")
    camp.add_argument("--first", type=int, default=48)
    camp.add_argument("--count", type=int, default=48)
    camp.add_argument("--crest", action="store_true")
    camp.set_defaults(func=campaign)
    eq = sub.add_parser("equivalence")
    eq.add_argument("--draws", type=int, default=6)
    eq.set_defaults(func=equivalence)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
