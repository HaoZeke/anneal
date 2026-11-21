# Copyright (c) anneal developers
"""ASV smoke for the outside-potential panel."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    "lj150": 4_000_000,
    "lj104": 4_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
    "au55": 1_000_000,
    "blj55s130": 1_000_000,
    "blj38s105": 400_000,
}
EXAMPLES = {
    "lj150": "h2h_gmin_proof",
    "lj104": "h2h_gmin_proof",
    "m80r6": "h2h_morse",
    "m80r14": "h2h_morse",
    "au55": "h2h_metal",
    "blj55s130": "h2h_blj",
    "blj38s105": "h2h_blj",
}


class Smoke:
    timeout = 1800

    def time_smoke(self) -> None:
        for system, budget in PAPER_BUDGETS.items():
            check_budget(EXAMPLES[system], system, budget)
        smoke("h2h_gmin_proof", ["run", "lj104", "2000"])
        smoke("h2h_morse", ["run", "m80r6", "2000"])
        smoke("h2h_metal", ["run", "au55", "2000"])
        smoke("h2h_blj", ["run", "blj38s105", "2000"])
