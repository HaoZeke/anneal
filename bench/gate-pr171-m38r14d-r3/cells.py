# Copyright (c) anneal developers
"""Cells for the morse m38 rho-14 gate.

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

BASE = "61fa67cf380e46876b569da45e7cc091647a775b"
BASE_TREE = "cff796c507a8fa1d0bfb28888c3570b61988055e"
PR_HEAD = "03476abc24dc56e0620e2ce5508b5f1f21e8a3f1"
PR_TREE = "5d4ce286694a171f2507edc442d15fcfcb908113"


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


# gmin panel, scripts/gmin_panel/run_panel.py SYSTEMS and PAPER_BUDGETS.
# lj103 resolves to h2h_gmin_proof when that binary lists lj103.
# morse panel extras follow, then the two h2h_seed cells, then the hexamer.
CELLS = (
    _panel("lj38", "h2h_gmin_proof", 38, 0.0, 400_000, -173.928427),
    _panel("lj55", "h2h_gmin_proof", 55, 0.0, 1_000_000, -279.248470),
    _panel("lj75", "h2h_gmin_proof", 75, 0.0, 4_000_000, -397.492331),
    _panel("lj98", "h2h_gmin_proof", 98, 0.0, 4_000_000, -543.665361),
    _panel("lj102", "h2h_gmin_proof", 102, 0.0, 4_000_000, -569.363652),
    _panel("lj103", "h2h_gmin_proof", 103, 0.0, 4_000_000, -575.766131),
    _panel("lj104", "h2h_gmin_proof", 104, 0.0, 4_000_000, -582.086642),
    _panel("lj150", "h2h_gmin_proof", 150, 0.0, 4_000_000, -893.310258),
    _panel("au55", "h2h_metal", 55, 0.0, 1_000_000, -191.03085280),
    _panel("m55r6", "h2h_morse", 55, 6.0, 1_000_000, -250.286609),
    _panel("m55r14", "h2h_morse", 55, 14.0, 1_000_000, -220.646208),
    _panel("m80r6", "h2h_morse", 80, 6.0, 4_000_000, -378.333471),
    _panel("m80r14", "h2h_morse", 80, 14.0, 4_000_000, -340.811371),
    _panel("m30r6", "h2h_morse", 30, 6.0, 400_000, -118.432844),
    _panel("m30r14", "h2h_morse", 30, 14.0, 400_000, -106.835790),
    _panel("m38r6", "h2h_morse", 38, 6.0, 400_000, -157.477108),
    _panel("m38r14", "h2h_morse", 38, 14.0, 400_000, -144.321054),
    _panel("m80r3", "h2h_morse", 80, 3.0, 4_000_000, -690.577890),
    _seed("m38r14", "morse", 38, 14.0, 400_000, -144.321054),
    _seed("lj75", "lj", 75, 0.0, 4_000_000, -397.492331),
    _seed(
        "hexamer",
        "tip4p",
        6,
        0.0,
        40_000,
        -197.7805260034,
        "no-regression guard",
    ),
)


def arm_order(seed: int) -> tuple[str, str, str]:
    """Even seeds run A, A', B. Odd seeds run the reverse."""
    if seed % 2 == 0:
        return ("A", "A'", "B")
    return ("B", "A'", "A")


def stop_token(reference: float) -> str:
    """Reference plus the 0.001 hit window, for the h2h_seed stop."""
    return format(reference + HIT_TOL, ".10f")


def self_test() -> None:
    keys = [(cell.name, cell.driver) for cell in CELLS]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate cell")
    if len(CELLS) != 21:
        raise SystemExit(f"cell count {len(CELLS)}")
    if arm_order(2000) != ("A", "A'", "B"):
        raise SystemExit("even order")
    if arm_order(2001) != ("B", "A'", "A"):
        raise SystemExit("odd order")
    if SEEDS[0] != 2000 or SEEDS[-1] != 2023 or len(SEEDS) != 24:
        raise SystemExit("seeds")
    names = [cell.name for cell in CELLS]
    for required in ("lj38", "lj75", "m80r6", "au55", "m38r14", "hexamer"):
        if required not in names:
            raise SystemExit(f"missing {required}")
    if sum(cell.role == "no-regression guard" for cell in CELLS) != 1:
        raise SystemExit("guard count")
    print("cells ok")


if __name__ == "__main__":
    self_test()
