# Copyright (c) anneal developers
"""Cells for the morse m38 rho-14 round-five gate.

Budgets and references are copied from the committed panel
tables. The hexamer budget is the ledger in the hexamer
search test. This list is part of the rule commit.
"""

from __future__ import annotations

from dataclasses import dataclass

HIT_TOL = 1.0e-3
SEEDS = tuple(range(2000, 2024))
# Worse than every declared budget. Error and timeout rows use it.
SENTINEL = 10**12
BOOT_SEED = 1712026
BOOT_DRAWS = 10000
OUTSIDE_CAP = 1.10
HANG_S = 14400
MAX_REPLACEMENTS = 2
TARGET_CELL = "m38r14"
TARGET_DRIVER = "h2h_morse"

BASE = "61fa67cf380e46876b569da45e7cc091647a775b"
BASE_TREE = "cff796c507a8fa1d0bfb28888c3570b61988055e"
PR_HEAD = "633d5ce5e8176ebd99cfe13f779a10752a059910"
PR_TREE = "a5a1423fb2c00475962770dd792f37944cf61174"


@dataclass(frozen=True)
class Cell:
    name: str
    driver: str
    # "panel" runs `<driver> run <name> <seed>`.
    # "seed" runs h2h_seed with model, n, rho, budget.
    invoke: str
    model: str
    n: int
    rho: float
    budget: int
    reference: float
    role: str


def _panel(name, driver, n, rho, budget, reference, role="reaches"):
    return Cell(name, driver, "panel", "", n, rho, budget, reference, role)


def _seed(name, model, n, rho, budget, reference, role="reaches"):
    return Cell(name, "h2h_seed", "seed", model, n, rho, budget, reference, role)


# Round three's five remaining clause-two panel cells, both m38 rho-14
# hit-loss drivers, and the two LJ75 drivers used for the scoring check.
CELLS = (
    _panel("lj75", "h2h_gmin_proof", 75, 0.0, 4_000_000, -397.492331),
    _panel("lj98", "h2h_gmin_proof", 98, 0.0, 4_000_000, -543.665361),
    _panel("m55r14", "h2h_morse", 55, 14.0, 1_000_000, -220.646208),
    _panel("m80r14", "h2h_morse", 80, 14.0, 4_000_000, -340.811371),
    _panel("m30r14", "h2h_morse", 30, 14.0, 400_000, -106.835790),
    _panel("m38r6", "h2h_morse", 38, 6.0, 400_000, -157.477108),
    _panel("m38r14", "h2h_morse", 38, 14.0, 400_000, -144.321054),
    _seed("m38r14", "morse", 38, 14.0, 400_000, -144.321054),
    _seed("lj75", "lj", 75, 0.0, 4_000_000, -397.492331),
)


def arm_order(seed: int) -> tuple[str, str, str, str]:
    """Each seed uses the pre-registered A-B-B-A order."""
    del seed
    return ("A", "B", "B2", "A2")


def stop_token(reference: float) -> str:
    """Reference plus the 0.001 hit window, for the h2h_seed stop."""
    return format(reference + HIT_TOL, ".10f")


def self_test() -> None:
    keys = [(cell.name, cell.driver) for cell in CELLS]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate cell")
    if len(CELLS) != 9:
        raise SystemExit(f"cell count {len(CELLS)}")
    if arm_order(2000) != ("A", "B", "B2", "A2"):
        raise SystemExit("arm order")
    if SEEDS[0] != 2000 or SEEDS[-1] != 2023 or len(SEEDS) != 24:
        raise SystemExit("seeds")
    names = [cell.name for cell in CELLS]
    for required in ("lj75", "lj98", "m80r14", "m38r14"):
        if required not in names:
            raise SystemExit(f"missing {required}")
    print("cells ok")


if __name__ == "__main__":
    self_test()
