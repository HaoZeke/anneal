# Copyright (c) anneal developers
"""ASV regression benchmark for GMIN-panel bookkeeping hot paths."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT.parent / "tmp" / "anneal-target" / "release" / "examples"
sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    "lj150": 4_000_000,
    "lj104": 4_000_000,
    "m80r6": 4_000_000,
    "au55": 1_000_000,
}
COMMANDS = {
    "lj150": (BIN / "h2h_gmin_proof", 13),
    "lj104": (BIN / "h2h_gmin_proof", 13),
    "m80r6": (BIN / "h2h_morse", 9),
    "au55": (BIN / "h2h_metal", 9),
}


class OutsideHotPath:
    params = (tuple(COMMANDS),)
    param_names = ("system",)
    timeout = 900

    def setup(self, system: str) -> None:
        binary, _pot_column = COMMANDS[system]
        if not binary.is_file():
            raise RuntimeError(f"missing {binary}")

    def track_outside_seconds(self, system: str) -> float:
        binary, pot_column = COMMANDS[system]
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
            env[name] = "1"
        env["H2H_BUDGET"] = "50000"
        proc = subprocess.run(
            [str(binary), "run", system, "2000"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        row = next(line for line in proc.stdout.splitlines() if line.startswith(f"{system},anneal,"))
        fields = row.split(",")
        return float(fields[7]) - float(fields[pot_column])


class Smoke:
    timeout = 1800

    def time_smoke(self) -> None:
        examples = {
            "lj150": "h2h_gmin_proof",
            "lj104": "h2h_gmin_proof",
            "m80r6": "h2h_morse",
            "au55": "h2h_metal",
        }
        for system, budget in PAPER_BUDGETS.items():
            check_budget(examples[system], system, budget)
        smoke("h2h_gmin_proof", ["run", "lj104", "2000"])
        smoke("h2h_morse", ["run", "m80r6", "2000"])
        smoke("h2h_metal", ["run", "au55", "2000"])
