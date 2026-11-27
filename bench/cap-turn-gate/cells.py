# Copyright (c) anneal developers
"""Pre-registered panel, paper budgets, references, and ABBA schedules."""

from __future__ import annotations

from dataclasses import dataclass

HIT_TOL: float = 0.001
SEEDS: tuple[int, ...] = tuple(range(2000, 2012))
REPLACEMENT_CAP: int = 2
OUTSIDE_RATIO_CAP: float = 1.10
RSS_ANON_ABS_TOL_KB: float = 8.0
BOOTSTRAP_DRAWS: int = 10_000
BOOTSTRAP_SEED: int = 20261121

CANDIDATE_COMMIT: str = "16ef5cb869f68c46e71b1eae2cbf7cee9321d8d4"
CANDIDATE_TREE: str = "0e08e2e2086180ca3b186f977191f6c1a2d12cd1"
DEVELOP_COMMIT: str = "9837358ae1027067ff654fac8b48969020853452"
DEVELOP_TREE: str = "15032ef401fb681e230d27c7b315d89f7ac53e4e"


@dataclass(frozen=True)
class CellSpec:
    name: str
    family: str
    n: int
    budget: int
    reference: float
    rigid_body: bool = False


PANEL_CELLS: tuple[CellSpec, ...] = (
    # Lennard-Jones paper panel (8)
    CellSpec("lj38", "lj", 38, 400_000, -173.928427),
    CellSpec("lj55", "lj", 55, 1_000_000, -279.248470),
    CellSpec("lj75", "lj", 75, 4_000_000, -397.492331),
    CellSpec("lj98", "lj", 98, 4_000_000, -543.665361),
    CellSpec("lj102", "lj", 102, 4_000_000, -569.363652),
    CellSpec("lj103", "lj", 103, 4_000_000, -575.766131),
    CellSpec("lj104", "lj", 104, 4_000_000, -582.086642),
    CellSpec("lj150", "lj", 150, 4_000_000, -893.310258),
    # Binary Lennard-Jones Doye-Meyer narrow-ratio panel (4)
    CellSpec("blj105n38", "blj", 38, 400_000, -177.2090846748),
    CellSpec("blj105n55", "blj", 55, 1_000_000, -283.1545313810),
    CellSpec("blj115n38", "blj", 38, 400_000, -187.3676867453),
    CellSpec("blj115n55", "blj", 55, 1_000_000, -289.2931853058),
    # Cambridge Gupta pure gold panel (2)
    CellSpec("au38", "au", 38, 400_000, -130.54001708),
    CellSpec("au55", "au", 55, 1_000_000, -191.03085280),
    # Doye-Wales Morse panel, rho=6 and rho=14 (6)
    CellSpec("m38r6", "morse", 38, 400_000, -157.477108),
    CellSpec("m38r14", "morse", 38, 400_000, -144.321054),
    CellSpec("m55r6", "morse", 55, 1_000_000, -250.286609),
    CellSpec("m55r14", "morse", 55, 1_000_000, -220.646208),
    CellSpec("m80r6", "morse", 80, 4_000_000, -378.333471),
    CellSpec("m80r14", "morse", 80, 4_000_000, -340.811371),
    # Rigid-body water / GMIN TIP4P panel (14)
    CellSpec("tip4p8", "water", 8, 2_000_000, -305.5183221676, True),
    CellSpec("tip4p9", "water", 9, 2_000_000, -344.4352092012, True),
    CellSpec("tip4p10", "water", 10, 2_000_000, -391.0226842125, True),
    CellSpec("tip4p11", "water", 11, 2_000_000, -431.4890723318, True),
    CellSpec("tip4p12", "water", 12, 2_000_000, -492.9082357122, True),
    CellSpec("tip4p13", "water", 13, 2_000_000, -532.9705363445, True),
    CellSpec("tip4p14", "water", 14, 2_000_000, -582.9911036211, True),
    CellSpec("tip4p15", "water", 15, 2_000_000, -628.3722525077, True),
    CellSpec("tip4p16", "water", 16, 2_000_000, -681.1925945898, True),
    CellSpec("tip4p17", "water", 17, 2_000_000, -723.8077531390, True),
    CellSpec("tip4p18", "water", 18, 2_000_000, -773.2318693143, True),
    CellSpec("tip4p19", "water", 19, 2_000_000, -821.0369191145, True),
    CellSpec("tip4p20", "water", 20, 2_000_000, -872.9887544952, True),
    CellSpec("tip4p21", "water", 21, 2_000_000, -916.7074302940, True),
)

CELL_BY_NAME: dict[str, CellSpec] = {c.name: c for c in PANEL_CELLS}

# Execution order puts fast/medium cells across all 5 families first, then
# heavy 4M / large-water cells, without altering per-cell ABBA structure.
MAIN_EXECUTION_ORDER: tuple[str, ...] = (
    "lj38",
    "lj55",
    "blj105n38",
    "blj105n55",
    "blj115n38",
    "au38",
    "m38r6",
    "m38r14",
    "m55r6",
    "m55r14",
    "m80r6",
    "tip4p8",
    "tip4p9",
    "tip4p10",
    "tip4p11",
    "tip4p12",
    "tip4p13",
    "tip4p14",
    "tip4p15",
    "tip4p16",
    "blj115n55",
    "au55",
    "lj75",
    "lj98",
    "lj102",
    "lj103",
    "lj104",
    "m80r14",
    "tip4p17",
    "tip4p18",
    "tip4p19",
    "tip4p20",
    "tip4p21",
    "lj150",
)

# Representative 10-cell calibration subset spanning every family (including
# blj105n38 where allocator rounding shifted in #202).
PLACEMENT_CELLS: tuple[str, ...] = (
    "lj38",
    "lj55",
    "blj105n38",
    "blj115n38",
    "au38",
    "m38r6",
    "m38r14",
    "m55r6",
    "tip4p8",
    "tip4p12",
)

MAIN_ARMS: tuple[str, ...] = ("A", "B", "D")
PLACE_ARMS: tuple[str, ...] = ("PA", "A2", "P1", "P2", "P3", "P4", "P5")
PADDING_KB_BY_ARM: dict[str, int] = {
    "A": 0,
    "B": 0,
    "D": 0,
    "PA": 0,
    "A2": 0,
    "P1": 4,
    "P2": 16,
    "P3": 64,
    "P4": 256,
    "P5": 1024,
}

# Exact 12-seed balanced schedule:
# - Every one of the 6 permutations of (A, B, D) appears twice.
# - Each arm occupies slot 0, slot 1, and slot 2 four times each.
# - Projecting onto (A, B) yields AB, BA, BA, AB repeated 3 times (ABBA).
MAIN_ROTATIONS: tuple[tuple[str, str, str], ...] = (
    ("A", "B", "D"),
    ("B", "D", "A"),
    ("D", "B", "A"),
    ("A", "D", "B"),
    ("D", "A", "B"),
    ("B", "A", "D"),
    ("B", "D", "A"),
    ("A", "B", "D"),
    ("A", "D", "B"),
    ("D", "B", "A"),
    ("B", "A", "D"),
    ("D", "A", "B"),
)


def main_arm_order(cell_index: int, seed_index: int) -> tuple[str, str, str]:
    # Offset by multiples of 4 on cell_index so ABBA blocks stay intact while
    # rotating which permutation leads each cell.
    idx = (seed_index + 4 * (cell_index % 3)) % len(MAIN_ROTATIONS)
    return MAIN_ROTATIONS[idx]


def place_arm_order(cell_index: int, seed_index: int) -> tuple[str, ...]:
    shift = (seed_index + cell_index) % len(PLACE_ARMS)
    rotated = PLACE_ARMS[shift:] + PLACE_ARMS[:shift]
    if seed_index % 2 == 1:
        return tuple(reversed(rotated))
    return rotated
